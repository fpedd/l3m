// File layout <-> tile layout, dequantization, activation preparation. See kernels.h for the layouts.
#include <math.h>
#include <string.h>
#include "kernels.h"
#include "ops.h"

#define QB L3M_QBLOCK
#define T  L3M_TILE

size_t l3m_row_bytes(int dtype, int k) {
    switch (dtype) {
    case L3M_F32:   return (size_t)k * 4;
    case L3M_BF16:  return (size_t)k * 2;
    case L3M_Q8_0:  return (size_t)k / QB * sizeof(l3m_block_q8_0);
    case L3M_Q4_0:  return (size_t)k / QB * sizeof(l3m_block_q4_0);
    case L3M_MXFP4: return (size_t)k / QB * sizeof(l3m_block_mxfp4);
    }
    return 0;
}

size_t l3m_tile_bytes(int dtype, int k) {
    switch (dtype) {
    case L3M_F32:   return (size_t)k * T * 4;
    case L3M_BF16:  return (size_t)k * T * 2;
    case L3M_Q8_0:  return (size_t)k / QB * (2 * T + QB * T);       // per block: f16 d, then bytes
    case L3M_Q4_0:  return (size_t)k / QB * (2 * T + QB * T / 2);   // per block: f16 d, then nibbles
    case L3M_MXFP4: return (size_t)k / QB * (T + QB * T / 2);       // per block: e8m0 e, then nibbles
    }
    return 0;
}

// Element k of a 4-bit block, as its nibble.
static inline int nib(const uint8_t *qs, int k) { return k < 16 ? qs[k] & 15 : qs[k - 16] >> 4; }

// Scale of block b of a row, as f32.
static float block_scale(int dtype, const void *row, int b) {
    switch (dtype) {
    case L3M_Q8_0:  return l3m_f16_to_f32(((const l3m_block_q8_0 *)row)[b].d);
    case L3M_Q4_0:  return l3m_f16_to_f32(((const l3m_block_q4_0 *)row)[b].d);
    case L3M_MXFP4: return ldexpf(1.0f, (int)((const l3m_block_mxfp4 *)row)[b].e - 128);   // 2^(e-127) / 2
    }
    return 0;
}

// Integer value of element k in block b of a row (before scaling).
static int block_q(int dtype, const void *row, int b, int k) {
    switch (dtype) {
    case L3M_Q8_0:  return ((const l3m_block_q8_0 *)row)[b].qs[k];
    case L3M_Q4_0:  return nib(((const l3m_block_q4_0 *)row)[b].qs, k) - 8;
    case L3M_MXFP4: return L3M_MXFP4_LUT[nib(((const l3m_block_mxfp4 *)row)[b].qs, k)];
    }
    return 0;
}

void l3m_dequant_row(float *out, int dtype, const void *row, int k) {
    if (dtype == L3M_F32) { memcpy(out, row, (size_t)k * 4); return; }
    if (dtype == L3M_BF16) { for (int i = 0; i < k; i++) out[i] = l3m_bf16_to_f32(((const uint16_t *)row)[i]); return; }
    for (int b = 0; b < k / QB; b++) {
        float d = block_scale(dtype, row, b);
        for (int i = 0; i < QB; i++) out[b * QB + i] = d * (float)block_q(dtype, row, b, i);
    }
}

void l3m_pack_tile(void *dst, int dtype, const void *const rows[T], int k) {
    memset(dst, 0, l3m_tile_bytes(dtype, k));
    if (dtype == L3M_F32) {
        float *w = dst;
        for (int c = 0; c < T; c++) if (rows[c])
            for (int i = 0; i < k; i++) w[i * T + c] = ((const float *)rows[c])[i];
        return;
    }
    if (dtype == L3M_BF16) {
        uint16_t *w = dst;
        for (int c = 0; c < T; c++) if (rows[c])
            for (int i = 0; i < k; i++) w[(i / 2 * T + c) * 2 + i % 2] = ((const uint16_t *)rows[c])[i];
        return;
    }
    // Quantized: per block { scales in file precision, bytes }. q8 bytes get +128; nibbles stay as in the file.
    size_t block_bytes = l3m_tile_bytes(dtype, QB), scale_bytes = dtype == L3M_MXFP4 ? T : 2 * T;
    for (int b = 0; b < k / QB; b++) {
        uint8_t *blk = (uint8_t *)dst + b * block_bytes;
        uint8_t *q = blk + scale_bytes;
        for (int c = 0; c < T; c++) {
            if (!rows[c]) continue;
            if (dtype == L3M_Q8_0) {
                const l3m_block_q8_0 *src = &((const l3m_block_q8_0 *)rows[c])[b];
                ((uint16_t *)blk)[c] = src->d;
                for (int i = 0; i < QB; i++) q[(i / 4 * T + c) * 4 + i % 4] = (uint8_t)(src->qs[i] + 128);
            } else {
                const uint8_t *qs;
                if (dtype == L3M_Q4_0) { const l3m_block_q4_0 *src = &((const l3m_block_q4_0 *)rows[c])[b]; ((uint16_t *)blk)[c] = src->d; qs = src->qs; }
                else                   { const l3m_block_mxfp4 *src = &((const l3m_block_mxfp4 *)rows[c])[b]; blk[c] = src->e; qs = src->qs; }
                for (int g = 0; g < 4; g++)
                    for (int j = 0; j < 4; j++)
                        q[(g * T + c) * 4 + j] = (uint8_t)(nib(qs, 8 * g + j) | nib(qs, 8 * g + 4 + j) << 4);
            }
        }
    }
}

void l3m_vec_prepare(l3m_vec *v, int dtype, const float *x, int n, void *scratch) {
    v->f32 = x; v->bf16 = NULL; v->q8 = NULL; v->scale = NULL; v->sum = NULL;
    if (dtype == L3M_BF16) {
        uint16_t *h = scratch;
        for (int i = 0; i < n; i++) h[i] = l3m_f32_to_bf16(x[i]);
        v->bf16 = h;
    } else if (dtype != L3M_F32) {          // scratch: q8[n], scale[n/32], sum[n/32]
        int8_t *q = scratch;
        float *scale = (float *)(q + n);
        int32_t *sum = (int32_t *)(scale + n / QB);
        for (int b = 0; b < n / QB; b++) {
            const float *xb = x + b * QB;
            float amax = 0;
            for (int i = 0; i < QB; i++) amax = fabsf(xb[i]) > amax ? fabsf(xb[i]) : amax;
            float inv = amax > 0 ? 127.0f / amax : 0;
            int32_t s = 0;
            for (int i = 0; i < QB; i++) s += q[b * QB + i] = (int8_t)(int)rintf(xb[i] * inv);
            scale[b] = amax / 127.0f;
            sum[b] = s;
        }
        v->q8 = q; v->scale = scale; v->sum = sum;
    }
}
