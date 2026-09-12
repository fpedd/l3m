// Reference kernels: one plain loop per dtype over the tile layout.
#include <math.h>
#include "kernels.h"
#include "ops.h"

#define T  L3M_TILE
#define QB L3M_QBLOCK

static void gemv_f32(float *y, const void *tiles, int ntiles, int k, const l3m_vec *x) {
    for (int t = 0; t < ntiles; t++) {
        const float *w = (const float *)tiles + (size_t)t * k * T;
        float acc[T] = {0};
        for (int i = 0; i < k; i++)
            for (int c = 0; c < T; c++) acc[c] += w[i * T + c] * x->f32[i];
        for (int c = 0; c < T; c++) y[t * T + c] = acc[c];
    }
}

static void gemv_bf16(float *y, const void *tiles, int ntiles, int k, const l3m_vec *x) {
    for (int t = 0; t < ntiles; t++) {
        const uint16_t *w = (const uint16_t *)tiles + (size_t)t * k * T;
        float acc[T] = {0};
        for (int i = 0; i < k; i++)
            for (int c = 0; c < T; c++)
                acc[c] += l3m_bf16_to_f32(w[(i / 2 * T + c) * 2 + i % 2]) * l3m_bf16_to_f32(x->bf16[i]);
        for (int c = 0; c < T; c++) y[t * T + c] = acc[c];
    }
}

static void gemv_q8(float *y, const void *tiles, int ntiles, int k, const l3m_vec *x) {
    size_t tb = l3m_tile_bytes(L3M_Q8_0, k);
    for (int t = 0; t < ntiles; t++) {
        const uint8_t *blk = (const uint8_t *)tiles + t * tb;
        float acc[T] = {0};
        for (int b = 0; b < k / QB; b++, blk += 2 * T + QB * T) {
            const uint16_t *d = (const uint16_t *)blk;
            const uint8_t *q = blk + 2 * T;
            int32_t acc_i[T] = {0};
            for (int i = 0; i < QB; i++)
                for (int c = 0; c < T; c++) acc_i[c] += q[(i / 4 * T + c) * 4 + i % 4] * x->q8[b * QB + i];
            for (int c = 0; c < T; c++) acc[c] += x->scale[b] * l3m_f16_to_f32(d[c]) * (float)(acc_i[c] - L3M_OFFSET_Q8 * x->sum[b]);
        }
        for (int c = 0; c < T; c++) y[t * T + c] = acc[c];
    }
}

// mx: e8m0 scales (mxfp4), else f16 (q4_0).
static void gemv_q4_lut(float *y, const void *tiles, int ntiles, int k, const l3m_vec *x, const int8_t *lut, int offset, int mx) {
    size_t tb = l3m_tile_bytes(mx ? L3M_MXFP4 : L3M_Q4_0, k), scale_bytes = mx ? T : 2 * T;
    for (int t = 0; t < ntiles; t++) {
        const uint8_t *blk = (const uint8_t *)tiles + t * tb;
        float acc[T] = {0};
        for (int b = 0; b < k / QB; b++, blk += scale_bytes + QB * T / 2) {
            const uint8_t *q = blk + scale_bytes;
            const int8_t *xq = x->q8 + b * QB;
            int32_t acc_i[T] = {0};
            for (int g = 0; g < 4; g++)
                for (int c = 0; c < T; c++)
                    for (int j = 0; j < 4; j++) {
                        uint8_t byte = q[(g * T + c) * 4 + j];
                        acc_i[c] += lut[byte & 15] * xq[8 * g + j] + lut[byte >> 4] * xq[8 * g + 4 + j];
                    }
            for (int c = 0; c < T; c++) {
                float d = mx ? ldexpf(1.0f, blk[c] - 128) : l3m_f16_to_f32(((const uint16_t *)blk)[c]);
                acc[c] += x->scale[b] * d * (float)(acc_i[c] - offset * x->sum[b]);
            }
        }
        for (int c = 0; c < T; c++) y[t * T + c] = acc[c];
    }
}

static const int8_t IDENTITY[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

static void gemv_q4(float *y, const void *t, int n, int k, const l3m_vec *x)    { gemv_q4_lut(y, t, n, k, x, IDENTITY, L3M_OFFSET_Q4, 0); }
static void gemv_mxfp4(float *y, const void *t, int n, int k, const l3m_vec *x) { gemv_q4_lut(y, t, n, k, x, L3M_MXFP4_LUT16, L3M_OFFSET_MXFP4, 1); }

const l3m_kernels l3m_kernels_scalar = {
    .name = "scalar",
    .gemv = { [L3M_F32] = gemv_f32, [L3M_BF16] = gemv_bf16, [L3M_Q8_0] = gemv_q8, [L3M_Q4_0] = gemv_q4, [L3M_MXFP4] = gemv_mxfp4 },
    .attend = l3m_attend,
};
