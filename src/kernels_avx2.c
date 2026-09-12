// AVX2 + FMA + F16C kernels: each 64-byte line as two ymm halves, up to NT tiles at once.
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include "kernels.h"
#include "ops.h"

#define T  L3M_TILE
#define QB L3M_QBLOCK
#define NT 4

#define ALWAYS static inline __attribute__((always_inline))
#define UNROLL _Pragma("GCC unroll 8")

// body(nt) with a constant nt, so the accumulators stay in registers.
#define FOR_TILES(body) for (int t = 0; t < ntiles; t += NT) switch (ntiles - t < NT ? ntiles - t : NT) { \
    case 4: body(4); break; case 3: body(3); break; case 2: body(2); break; default: body(1); break; }

ALWAYS void f32_tiles(float *y, const float *w, int nt, int k, const float *x) {
    __m256 acc[2 * NT];
    UNROLL for (int j = 0; j < 2 * nt; j++) acc[j] = _mm256_setzero_ps();
    for (int i = 0; i < k; i++) {
        __m256 a = _mm256_set1_ps(x[i]);
        UNROLL for (int j = 0; j < nt; j++) {
            const float *line = w + ((size_t)j * k + i) * T;
            acc[2 * j]     = _mm256_fmadd_ps(a, _mm256_loadu_ps(line), acc[2 * j]);
            acc[2 * j + 1] = _mm256_fmadd_ps(a, _mm256_loadu_ps(line + 8), acc[2 * j + 1]);
        }
    }
    UNROLL for (int j = 0; j < 2 * nt; j++) _mm256_storeu_ps(y + j * 8, acc[j]);
}

static void gemv_f32(float *y, const void *tiles, int ntiles, int k, const l3m_vec *x) {
#define BODY(n) f32_tiles(y + t * T, (const float *)tiles + (size_t)t * k * T, n, k, x->f32)
    FOR_TILES(BODY);
#undef BODY
}

// A 32-bit lane holds a bf16 pair: even k low, odd k high.
ALWAYS __m256 bf16_lo(__m256i v) { return _mm256_castsi256_ps(_mm256_slli_epi32(v, 16)); }
ALWAYS __m256 bf16_hi(__m256i v) { return _mm256_castsi256_ps(_mm256_and_si256(v, _mm256_set1_epi32((int)0xFFFF0000))); }

ALWAYS void bf16_tiles(float *y, const uint16_t *w, int nt, int k, const uint16_t *x) {
    __m256 acc[2 * NT];
    UNROLL for (int j = 0; j < 2 * nt; j++) acc[j] = _mm256_setzero_ps();
    for (int i = 0; i < k / 2; i++) {
        __m256 a0 = _mm256_set1_ps(l3m_bf16_to_f32(x[2 * i])), a1 = _mm256_set1_ps(l3m_bf16_to_f32(x[2 * i + 1]));
        UNROLL for (int j = 0; j < nt; j++) {
            const __m256i *line = (const __m256i *)(w + ((size_t)j * k / 2 + i) * T * 2);
            __m256i l0 = _mm256_loadu_si256(line), l1 = _mm256_loadu_si256(line + 1);
            acc[2 * j]     = _mm256_fmadd_ps(a0, bf16_lo(l0), acc[2 * j]);
            acc[2 * j]     = _mm256_fmadd_ps(a1, bf16_hi(l0), acc[2 * j]);
            acc[2 * j + 1] = _mm256_fmadd_ps(a0, bf16_lo(l1), acc[2 * j + 1]);
            acc[2 * j + 1] = _mm256_fmadd_ps(a1, bf16_hi(l1), acc[2 * j + 1]);
        }
    }
    UNROLL for (int j = 0; j < 2 * nt; j++) _mm256_storeu_ps(y + j * 8, acc[j]);
}

static void gemv_bf16(float *y, const void *tiles, int ntiles, int k, const l3m_vec *x) {
#define BODY(n) bf16_tiles(y + t * T, (const uint16_t *)tiles + (size_t)t * k * T, n, k, x->bf16)
    FOR_TILES(BODY);
#undef BODY
}

// maddubs saturates at int16: madd_u8 is exact for w <= 127 (the 4-bit paths). madd_s8 takes signed w
// (q8 xor 0x80) as maddubs(|w|, x * sign(w)), where |-128| reads as 128; it needs |x| <= 127.
ALWAYS __m256i madd_u8(__m256i acc, __m256i w, __m256i xb) {
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(w, xb), _mm256_set1_epi16(1)));
}
ALWAYS __m256i madd_s8(__m256i acc, __m256i w, __m256i xb) {
    __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(xb, w));
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}
ALWAYS __m256i xquad(const int8_t *q) { int32_t v; memcpy(&v, q, 4); return _mm256_set1_epi32(v); }

ALWAYS __m256 f16x8(const void *p) { return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p)); }
// e8m0 -> 2^(e-128); e < 2 is subnormal.
ALWAYS __m256 e8m0_half8(const void *p) {
    __m256i e = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)p));
    __m256i normal = _mm256_slli_epi32(_mm256_sub_epi32(e, _mm256_set1_epi32(1)), 23);
    __m256i tiny = _mm256_sllv_epi32(_mm256_set1_epi32(0x00200000), e);
    return _mm256_castsi256_ps(_mm256_blendv_epi8(normal, tiny, _mm256_cmpgt_epi32(_mm256_set1_epi32(2), e)));
}

ALWAYS __m256 block_epilogue(__m256 acc, __m256i acc_i, __m256 d, const l3m_vec *x, int b, int offset) {
    __m256 corr = _mm256_sub_ps(_mm256_cvtepi32_ps(acc_i), _mm256_set1_ps((float)(offset * x->sum[b])));
    return _mm256_fmadd_ps(corr, _mm256_mul_ps(d, _mm256_set1_ps(x->scale[b])), acc);
}

ALWAYS void q8_tiles(float *y, const uint8_t *tiles, size_t tb, int nt, int k, const l3m_vec *x) {
    __m256 acc[2 * NT];
    __m256i x80 = _mm256_set1_epi8((char)0x80);
    UNROLL for (int j = 0; j < 2 * nt; j++) acc[j] = _mm256_setzero_ps();
    for (int b = 0; b < k / QB; b++) {
        const uint8_t *blk = tiles + b * (2 * T + QB * T);
        __m256i acc_i[2 * NT];
        UNROLL for (int j = 0; j < 2 * nt; j++) acc_i[j] = _mm256_setzero_si256();
        for (int g = 0; g < QB / 4; g++) {
            __m256i a = xquad(x->q8 + b * QB + 4 * g);
            UNROLL for (int j = 0; j < 2 * nt; j++) {
                __m256i half = _mm256_loadu_si256((const __m256i *)(blk + j / 2 * tb + 2 * T + g * 64 + j % 2 * 32));
                acc_i[j] = madd_s8(acc_i[j], _mm256_xor_si256(half, x80), a);
            }
        }
        UNROLL for (int j = 0; j < 2 * nt; j++) acc[j] = block_epilogue(acc[j], acc_i[j], f16x8(blk + j / 2 * tb + j % 2 * 16), x, b, 0);
    }
    UNROLL for (int j = 0; j < 2 * nt; j++) _mm256_storeu_ps(y + j * 8, acc[j]);
}

static void gemv_q8(float *y, const void *tiles, int ntiles, int k, const l3m_vec *x) {
    size_t tb = l3m_tile_bytes(L3M_Q8_0, k);
#define BODY(n) q8_tiles(y + t * T, (const uint8_t *)tiles + t * tb, tb, n, k, x)
    FOR_TILES(BODY);
#undef BODY
}

// mx: nibbles through lut16 and e8m0 scales (mxfp4), else raw nibbles and f16 scales (q4_0).
ALWAYS void q4_tiles(float *y, const uint8_t *tiles, size_t tb, int nt, int k, const l3m_vec *x, const int8_t *lut16, int offset, int mx) {
    __m256 acc[2 * NT];
    __m256i mask = _mm256_set1_epi8(15);
    __m256i lut = mx ? _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)lut16)) : mask;
    size_t scale_bytes = mx ? T : 2 * T;
    UNROLL for (int j = 0; j < 2 * nt; j++) acc[j] = _mm256_setzero_ps();
    for (int b = 0; b < k / QB; b++) {
        const uint8_t *blk = tiles + b * (scale_bytes + QB * T / 2);
        __m256i acc_i[2 * NT];
        UNROLL for (int j = 0; j < 2 * nt; j++) acc_i[j] = _mm256_setzero_si256();
        for (int g = 0; g < 4; g++) {
            __m256i alo = xquad(x->q8 + b * QB + 8 * g), ahi = xquad(x->q8 + b * QB + 8 * g + 4);
            UNROLL for (int j = 0; j < 2 * nt; j++) {
                __m256i half = _mm256_loadu_si256((const __m256i *)(blk + j / 2 * tb + scale_bytes + g * 64 + j % 2 * 32));
                __m256i wlo = _mm256_and_si256(half, mask), whi = _mm256_and_si256(_mm256_srli_epi16(half, 4), mask);
                if (mx) { wlo = _mm256_shuffle_epi8(lut, wlo); whi = _mm256_shuffle_epi8(lut, whi); }
                acc_i[j] = madd_u8(madd_u8(acc_i[j], wlo, alo), whi, ahi);
            }
        }
        UNROLL for (int j = 0; j < 2 * nt; j++) {
            const uint8_t *sc = blk + j / 2 * tb + j % 2 * scale_bytes / 2;
            __m256 d = mx ? e8m0_half8(sc) : f16x8(sc);
            acc[j] = block_epilogue(acc[j], acc_i[j], d, x, b, offset);
        }
    }
    UNROLL for (int j = 0; j < 2 * nt; j++) _mm256_storeu_ps(y + j * 8, acc[j]);
}

ALWAYS void gemv_q4_lut(float *y, const void *tiles, int ntiles, int k, const l3m_vec *x, const int8_t *lut16, int offset, int mx) {
    size_t tb = l3m_tile_bytes(mx ? L3M_MXFP4 : L3M_Q4_0, k);
#define BODY(n) q4_tiles(y + t * T, (const uint8_t *)tiles + t * tb, tb, n, k, x, lut16, offset, mx)
    FOR_TILES(BODY);
#undef BODY
}

static void gemv_q4(float *y, const void *t, int n, int k, const l3m_vec *x)    { gemv_q4_lut(y, t, n, k, x, NULL, L3M_OFFSET_Q4, 0); }
static void gemv_mxfp4(float *y, const void *t, int n, int k, const l3m_vec *x) { gemv_q4_lut(y, t, n, k, x, L3M_MXFP4_LUT16, L3M_OFFSET_MXFP4, 1); }

// 2^n * 2^f, f in [-0.5, 0.5]; x >= -87 keeps the exponent add from underflowing.
ALWAYS __m256 exp8(__m256 x) {
    __m256 t = _mm256_mul_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0f)), _mm256_set1_ps(1.44269504f));
    __m256 n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), f = _mm256_sub_ps(t, n);
    __m256 p = _mm256_set1_ps(1.5403530e-4f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.3333558e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(9.6181291e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.5504109e-2f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.4022651e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.9314718e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    __m256i e = _mm256_slli_epi32(_mm256_cvtps_epi32(n), 23);
    return _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(p), e));
}

ALWAYS __m256 bf16x8(const uint16_t *p) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16));
}
ALWAYS float hsum(__m256 x) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    return _mm_cvtss_f32(h) + _mm_cvtss_f32(_mm_shuffle_ps(h, h, 1));
}

// head_dim % 8 == 0
static void attend(float *out, const float *q, const uint16_t *k, const uint16_t *v, int pos, int head_dim, int stride) {
    int n = pos + 1;
    float s[(n + 7) & ~7];
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int t = 0; t < n; t++) {
        const uint16_t *kt = k + (size_t)t * stride;
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < head_dim; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), bf16x8(kt + i), acc);
        s[t] = hsum(acc) * scale;
    }
    float max = -INFINITY;
    for (int t = 0; t < n; t++) if (s[t] > max) max = s[t];
    for (int t = n; t < ((n + 7) & ~7); t++) s[t] = -INFINITY;     // exp8 gives ~2^-126, negligible
    __m256 sum = _mm256_setzero_ps(), vmax = _mm256_set1_ps(max);
    for (int t = 0; t < n; t += 8) {
        __m256 e = exp8(_mm256_sub_ps(_mm256_loadu_ps(s + t), vmax));
        _mm256_storeu_ps(s + t, e);
        sum = _mm256_add_ps(sum, e);
    }
    float inv = 1.0f / hsum(sum);
    for (int i = 0; i < head_dim; i += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int t = 0; t < n; t++) acc = _mm256_fmadd_ps(_mm256_set1_ps(s[t]), bf16x8(v + (size_t)t * stride + i), acc);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(acc, _mm256_set1_ps(inv)));
    }
}

const l3m_kernels l3m_kernels_avx2 = {
    .name = "avx2",
    .gemv = { [L3M_F32] = gemv_f32, [L3M_BF16] = gemv_bf16, [L3M_Q8_0] = gemv_q8, [L3M_Q4_0] = gemv_q4, [L3M_MXFP4] = gemv_mxfp4 },
    .attend = attend,
};
