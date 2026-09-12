// Tile layout: L3M_TILE outputs x K inputs as one byte stream, each 64-byte line one lane per output.
//
//   f32   [K][16] float
//   bf16  [K/2][16][2] bf16
//   q8_0  per 32 inputs: d[16] f16, q[8][16][4] u8 = value + 128                 544 bytes
//   q4_0  per 32 inputs: d[16] f16, q[4][16][4] nibbles, value = nibble - 8     288 bytes
//         byte j of lane c in group i: k = 8i+j (low nibble), 8i+4+j (high)
//   mxfp4 per 32 inputs: e[16] e8m0, same nibbles, value = LUT16[nibble] - 16   272 bytes
//         scale = 2^(e-127) / 2
//
//   y += x.scale[b] * d[b] * (acc_i32 - offset * x.sum[b]),  offset = 128, 8, 16
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "model.h"

#define L3M_TILE 16

// mxfp4 nibble -> doubled e2m1 value, and the same + 16.
static const int8_t L3M_MXFP4_LUT[16]  = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static const int8_t L3M_MXFP4_LUT16[16] = {16, 17, 18, 19, 20, 22, 24, 28, 16, 15, 14, 13, 12, 10, 8, 4};
enum { L3M_OFFSET_Q8 = 128, L3M_OFFSET_Q4 = 8, L3M_OFFSET_MXFP4 = 16 };

typedef struct {
    const float    *f32;
    const uint16_t *bf16;
    const int8_t   *q8;    // x[i] ~= q8[i] * scale[i/32]
    const float    *scale; // n/32 entries
    const int32_t  *sum;   // n/32 entries: sum of q8 over the block
} l3m_vec;

// y[0 .. ntiles*16) = tiles . x
typedef void (*l3m_gemv_fn)(float *y, const void *tiles, int ntiles, int k, const l3m_vec *x);

typedef struct {
    const char *name;
    l3m_gemv_fn gemv[L3M_NDTYPES];
    // as l3m_attend in ops.h
    void (*attend)(float *out, const float *q, const uint16_t *k, const uint16_t *v, int pos, int head_dim, int stride);
} l3m_kernels;

extern const l3m_kernels l3m_kernels_scalar, l3m_kernels_avx2, l3m_kernels_avx512;

// L3M_ISA=scalar|avx2|avx512 forces a lower table.
const l3m_kernels *l3m_kernels_pick(void);
int l3m_kernels_usable(const l3m_kernels *k);

size_t l3m_row_bytes(int dtype, int k);
size_t l3m_tile_bytes(int dtype, int k);
void   l3m_pack_tile(void *dst, int dtype, const void *const rows[L3M_TILE], int k);  // NULL row = zeros
void   l3m_dequant_row(float *out, int dtype, const void *row, int k);
void   l3m_vec_prepare(l3m_vec *v, int dtype, const float *x, int n, void *scratch);  // scratch >= 2n bytes, n % 32 == 0 for quantized
