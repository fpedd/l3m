// AVX-512 VNNI + BF16 kernels: each 64-byte line as one zmm, up to NT tiles at once.
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include "kernels.h"

#define T  L3M_TILE
#define QB L3M_QBLOCK
#define NT 8

#define ALWAYS static inline __attribute__((always_inline))
#define UNROLL _Pragma("GCC unroll 8")

// body(nt) with a constant nt, so the accumulators stay in registers.
#define FOR_TILES(body) for (int t = 0; t < ntiles; t += NT) switch (ntiles - t < NT ? ntiles - t : NT) { \
    case 8: body(8); break; case 7: body(7); break; case 6: body(6); break; case 5: body(5); break; \
    case 4: body(4); break; case 3: body(3); break; case 2: body(2); break; default: body(1); break; }

ALWAYS void f32_tiles(float *y, const float *w, int nt, int k, const float *x) {
    __m512 acc[NT];
    UNROLL for (int j = 0; j < nt; j++) acc[j] = _mm512_setzero_ps();
    for (int i = 0; i < k; i++) {
        __m512 a = _mm512_set1_ps(x[i]);
        UNROLL for (int j = 0; j < nt; j++) acc[j] = _mm512_fmadd_ps(a, _mm512_loadu_ps(w + ((size_t)j * k + i) * T), acc[j]);
    }
    UNROLL for (int j = 0; j < nt; j++) _mm512_storeu_ps(y + j * T, acc[j]);
}

static void gemv_f32(float *y, const void *tiles, int ntiles, int k, const l3m_vec *x) {
#define BODY(n) f32_tiles(y + t * T, (const float *)tiles + (size_t)t * k * T, n, k, x->f32)
    FOR_TILES(BODY);
#undef BODY
}

ALWAYS void bf16_tiles(float *y, const uint16_t *w, int nt, int k, const uint16_t *x) {
    __m512 acc[NT];
    UNROLL for (int j = 0; j < nt; j++) acc[j] = _mm512_setzero_ps();
    for (int i = 0; i < k / 2; i++) {
        int32_t pair;
        memcpy(&pair, x + 2 * i, 4);
        __m512bh a = (__m512bh)_mm512_set1_epi32(pair);
        UNROLL for (int j = 0; j < nt; j++)
            acc[j] = _mm512_dpbf16_ps(acc[j], a, (__m512bh)_mm512_loadu_si512(w + ((size_t)j * k / 2 + i) * T * 2));
    }
    UNROLL for (int j = 0; j < nt; j++) _mm512_storeu_ps(y + j * T, acc[j]);
}

static void gemv_bf16(float *y, const void *tiles, int ntiles, int k, const l3m_vec *x) {
#define BODY(n) bf16_tiles(y + t * T, (const uint16_t *)tiles + (size_t)t * k * T, n, k, x->bf16)
    FOR_TILES(BODY);
#undef BODY
}

ALWAYS __m512i xquad(const int8_t *q) { int32_t v; memcpy(&v, q, 4); return _mm512_set1_epi32(v); }

ALWAYS __m512 f16x16(const void *p) { return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p)); }
// e8m0 -> 2^(e-128)
ALWAYS __m512 e8m0_half16(const void *p) {
    __m512i e = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)p));
    return _mm512_scalef_ps(_mm512_set1_ps(1.0f), _mm512_cvtepi32_ps(_mm512_sub_epi32(e, _mm512_set1_epi32(128))));
}

ALWAYS __m512 block_epilogue(__m512 acc, __m512i acc_i, __m512 d, const l3m_vec *x, int b, int offset) {
    __m512 corr = _mm512_sub_ps(_mm512_cvtepi32_ps(acc_i), _mm512_set1_ps((float)(offset * x->sum[b])));
    return _mm512_fmadd_ps(corr, _mm512_mul_ps(d, _mm512_set1_ps(x->scale[b])), acc);
}

ALWAYS void q8_tiles(float *y, const uint8_t *tiles, size_t tb, int nt, int k, const l3m_vec *x) {
    __m512 acc[NT];
    UNROLL for (int j = 0; j < nt; j++) acc[j] = _mm512_setzero_ps();
    for (int b = 0; b < k / QB; b++) {
        const uint8_t *blk = tiles + b * (2 * T + QB * T);
        __m512i acc_i[NT];
        UNROLL for (int j = 0; j < nt; j++) acc_i[j] = _mm512_setzero_si512();
        for (int g = 0; g < QB / 4; g++) {
            __m512i a = xquad(x->q8 + b * QB + 4 * g);
            UNROLL for (int j = 0; j < nt; j++)
                acc_i[j] = _mm512_dpbusd_epi32(acc_i[j], _mm512_loadu_si512(blk + j * tb + 2 * T + g * 64), a);
        }
        UNROLL for (int j = 0; j < nt; j++) acc[j] = block_epilogue(acc[j], acc_i[j], f16x16(blk + j * tb), x, b, L3M_OFFSET_Q8);
    }
    UNROLL for (int j = 0; j < nt; j++) _mm512_storeu_ps(y + j * T, acc[j]);
}

static void gemv_q8(float *y, const void *tiles, int ntiles, int k, const l3m_vec *x) {
    size_t tb = l3m_tile_bytes(L3M_Q8_0, k);
#define BODY(n) q8_tiles(y + t * T, (const uint8_t *)tiles + t * tb, tb, n, k, x)
    FOR_TILES(BODY);
#undef BODY
}

// mx: nibbles through lut16 and e8m0 scales (mxfp4), else raw nibbles and f16 scales (q4_0).
ALWAYS void q4_tiles(float *y, const uint8_t *tiles, size_t tb, int nt, int k, const l3m_vec *x, const int8_t *lut16, int offset, int mx) {
    __m512 acc[NT];
    __m512i mask = _mm512_set1_epi8(15);
    __m512i lut = mx ? _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)lut16)) : mask;
    size_t scale_bytes = mx ? T : 2 * T;
    UNROLL for (int j = 0; j < nt; j++) acc[j] = _mm512_setzero_ps();
    for (int b = 0; b < k / QB; b++) {
        const uint8_t *blk = tiles + b * (scale_bytes + QB * T / 2);
        __m512i acc_i[NT];
        UNROLL for (int j = 0; j < nt; j++) acc_i[j] = _mm512_setzero_si512();
        for (int g = 0; g < 4; g++) {
            __m512i alo = xquad(x->q8 + b * QB + 8 * g), ahi = xquad(x->q8 + b * QB + 8 * g + 4);
            UNROLL for (int j = 0; j < nt; j++) {
                __m512i line = _mm512_loadu_si512(blk + j * tb + scale_bytes + g * 64);
                __m512i wlo = _mm512_and_si512(line, mask), whi = _mm512_and_si512(_mm512_srli_epi16(line, 4), mask);
                if (mx) { wlo = _mm512_shuffle_epi8(lut, wlo); whi = _mm512_shuffle_epi8(lut, whi); }
                acc_i[j] = _mm512_dpbusd_epi32(acc_i[j], wlo, alo);
                acc_i[j] = _mm512_dpbusd_epi32(acc_i[j], whi, ahi);
            }
        }
        UNROLL for (int j = 0; j < nt; j++) {
            __m512 d = mx ? e8m0_half16(blk + j * tb) : f16x16(blk + j * tb);
            acc[j] = block_epilogue(acc[j], acc_i[j], d, x, b, offset);
        }
    }
    UNROLL for (int j = 0; j < nt; j++) _mm512_storeu_ps(y + j * T, acc[j]);
}

ALWAYS void gemv_q4_lut(float *y, const void *tiles, int ntiles, int k, const l3m_vec *x, const int8_t *lut16, int offset, int mx) {
    size_t tb = l3m_tile_bytes(mx ? L3M_MXFP4 : L3M_Q4_0, k);
#define BODY(n) q4_tiles(y + t * T, (const uint8_t *)tiles + t * tb, tb, n, k, x, lut16, offset, mx)
    FOR_TILES(BODY);
#undef BODY
}

static void gemv_q4(float *y, const void *t, int n, int k, const l3m_vec *x)    { gemv_q4_lut(y, t, n, k, x, NULL, L3M_OFFSET_Q4, 0); }
static void gemv_mxfp4(float *y, const void *t, int n, int k, const l3m_vec *x) { gemv_q4_lut(y, t, n, k, x, L3M_MXFP4_LUT16, L3M_OFFSET_MXFP4, 1); }

// 2^n * 2^f, f in [-0.5, 0.5]
ALWAYS __m512 exp16(__m512 x) {
    __m512 t = _mm512_mul_ps(x, _mm512_set1_ps(1.44269504f));
    __m512 n = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), f = _mm512_sub_ps(t, n);
    __m512 p = _mm512_set1_ps(1.5403530e-4f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.3333558e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(9.6181291e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.5504109e-2f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.4022651e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.9314718e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}

ALWAYS __m512 bf16x16(const uint16_t *p, __mmask16 m) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(m, p)), 16));
}
ALWAYS __mmask16 lanes(int n) { return n >= 16 ? 0xFFFF : (__mmask16)((1u << n) - 1); }

static void attend(float *out, const float *q, const uint16_t *k, const uint16_t *v, int pos, int head_dim, int stride) {
    int n = pos + 1;
    float s[(n + 15) & ~15];
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int t = 0; t < n; t++) {
        const uint16_t *kt = k + (size_t)t * stride;
        __m512 acc = _mm512_setzero_ps();
        for (int i = 0; i < head_dim; i += 16)
            acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(lanes(head_dim - i), q + i), bf16x16(kt + i, lanes(head_dim - i)), acc);
        s[t] = _mm512_reduce_add_ps(acc) * scale;
    }
    float max = -INFINITY;
    for (int t = 0; t < n; t++) if (s[t] > max) max = s[t];
    __m512 sum = _mm512_setzero_ps(), vmax = _mm512_set1_ps(max);
    for (int t = 0; t < n; t += 16) {
        __m512 e = exp16(_mm512_sub_ps(_mm512_maskz_loadu_ps(lanes(n - t), s + t), vmax));
        e = _mm512_maskz_mov_ps(lanes(n - t), e);
        _mm512_storeu_ps(s + t, e);
        sum = _mm512_add_ps(sum, e);
    }
    float inv = 1.0f / _mm512_reduce_add_ps(sum);
    for (int i = 0; i < head_dim; i += 16) {
        __m512 acc = _mm512_setzero_ps();
        for (int t = 0; t < n; t++)
            acc = _mm512_fmadd_ps(_mm512_set1_ps(s[t]), bf16x16(v + (size_t)t * stride + i, lanes(head_dim - i)), acc);
        _mm512_mask_storeu_ps(out + i, lanes(head_dim - i), _mm512_mul_ps(acc, _mm512_set1_ps(inv)));
    }
}

const l3m_kernels l3m_kernels_avx512 = {
    .name = "avx512",
    .gemv = { [L3M_F32] = gemv_f32, [L3M_BF16] = gemv_bf16, [L3M_Q8_0] = gemv_q8, [L3M_Q4_0] = gemv_q4, [L3M_MXFP4] = gemv_mxfp4 },
    .attend = attend,
};
