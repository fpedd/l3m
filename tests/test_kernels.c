// Every kernel table against double-precision references and the scalar table, then throughput.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernels.h"
#include "ops.h"
#include "perf.h"

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float frand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (float)(int64_t)(rng >> 11) / (float)(1ll << 52) - 1.0f; }

static uint16_t f32_to_f16(float f) {             // nearest even, normal range only
    union { float f; uint32_t u; } v = { f };
    uint32_t sign = (v.u >> 16) & 0x8000; int exp = (int)((v.u >> 23) & 255) - 127 + 15; uint32_t mant = v.u & 0x7FFFFF;
    if (exp <= 0) return (uint16_t)sign;
    uint32_t h = (uint32_t)exp << 10 | mant >> 13, rem = mant & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return (uint16_t)(sign | h);
}

// ggml's quantizers.
static void quantize_row(int dtype, void *dst, const float *x, int k) {
    if (dtype == L3M_F32) { memcpy(dst, x, k * 4); return; }
    if (dtype == L3M_BF16) { for (int i = 0; i < k; i++) ((uint16_t *)dst)[i] = l3m_f32_to_bf16(x[i]); return; }
    for (int b = 0; b < k / 32; b++) {
        const float *xb = x + b * 32;
        float amax = 0, max = 0;
        for (int i = 0; i < 32; i++) if (fabsf(xb[i]) > amax) { amax = fabsf(xb[i]); max = xb[i]; }
        if (dtype == L3M_Q8_0) {
            l3m_block_q8_0 *q = (l3m_block_q8_0 *)dst + b;
            float d = amax / 127, id = d ? 1 / d : 0;
            q->d = f32_to_f16(d);
            for (int i = 0; i < 32; i++) q->qs[i] = (int8_t)roundf(xb[i] * id);
        } else if (dtype == L3M_Q4_0) {
            l3m_block_q4_0 *q = (l3m_block_q4_0 *)dst + b;
            float d = max / -8, id = d ? 1 / d : 0;
            q->d = f32_to_f16(d);
            for (int i = 0; i < 16; i++) {
                int lo = (int)(xb[i] * id + 8.5f), hi = (int)(xb[i + 16] * id + 8.5f);
                q->qs[i] = (uint8_t)((lo > 15 ? 15 : lo) | (hi > 15 ? 15 : hi) << 4);
            }
        } else {
            l3m_block_mxfp4 *q = (l3m_block_mxfp4 *)dst + b;
            q->e = amax > 0 ? (uint8_t)(floorf(log2f(amax)) - 2 + 127) : 0;
            float id = amax > 0 ? ldexpf(1.0f, 128 - q->e) : 0;
            uint8_t nib[32];
            for (int i = 0; i < 32; i++) {          // nearest doubled-e2m1 magnitude, sign in bit 3
                float v = fabsf(xb[i]) * id; int best = 0;
                for (int m = 1; m < 8; m++) if (fabsf(v - L3M_MXFP4_LUT[m]) < fabsf(v - L3M_MXFP4_LUT[best])) best = m;
                nib[i] = (uint8_t)(best | (xb[i] < 0 && best ? 8 : 0));
            }
            for (int i = 0; i < 16; i++) q->qs[i] = (uint8_t)(nib[i] | nib[i + 16] << 4);
        }
    }
}

static const l3m_kernels *const TABLES[3] = { &l3m_kernels_scalar, &l3m_kernels_avx2, &l3m_kernels_avx512 };

// Legal values the quantizers never produce: q8_0 weight -128, subnormal mxfp4 scales, and (in check) a zero activation block.
static void extremes(int dtype, uint8_t *row, int k, int r) {
    for (int b = 0; b < k / 32; b++) {
        if (dtype == L3M_Q8_0) ((l3m_block_q8_0 *)row)[b].qs[b % 32] = -128;
        if (dtype == L3M_MXFP4) ((l3m_block_mxfp4 *)row)[b].e = (uint8_t)((r + b) % 3);
    }
}

// `missing` NULL rows at the end; `edge` applies extremes().
static int check(int dtype, int ntiles, int k, int missing, int edge) {
    int n = ntiles * L3M_TILE, ok = 1;
    size_t rb = l3m_row_bytes(dtype, k), tb = l3m_tile_bytes(dtype, k);
    uint8_t *rows = malloc(n * rb), *tiles = malloc(ntiles * tb);
    float *src = malloc(k * 4), *x = malloc(k * 4), *deq = malloc(k * 4);
    double *ref = malloc(n * sizeof(double)), *ref_x = malloc(n * sizeof(double)), mag = 0;   // vs prepared x, exact x
    float *y[3]; for (int i = 0; i < 3; i++) y[i] = malloc(n * 4);
    const void *rp[L3M_TILE];
    for (int r = 0; r < n; r++) {
        for (int i = 0; i < k; i++) src[i] = frand();
        quantize_row(dtype, rows + r * rb, src, k);
        if (edge) extremes(dtype, rows + r * rb, k, r);
    }
    for (int i = 0; i < k; i++) x[i] = edge && i < 32 ? 0 : frand() * 4;
    for (int t = 0; t < ntiles; t++) {
        for (int c = 0; c < L3M_TILE; c++) rp[c] = (t * L3M_TILE + c < n - missing) ? rows + (t * L3M_TILE + c) * rb : NULL;
        l3m_pack_tile(tiles + t * tb, dtype, rp, k);
    }
    l3m_vec v; void *scratch = malloc(2 * k);
    l3m_vec_prepare(&v, dtype, x, k, scratch);
    for (int r = 0; r < n; r++) {
        ref[r] = ref_x[r] = 0;
        if (r >= n - missing) continue;
        l3m_dequant_row(deq, dtype, rows + r * rb, k);
        double abs_sum = 0;
        for (int i = 0; i < k; i++) {
            double xi = v.q8 ? v.q8[i] * (double)v.scale[i / 32] : v.bf16 ? l3m_bf16_to_f32(v.bf16[i]) : x[i];
            ref[r] += (double)deq[i] * xi;
            ref_x[r] += (double)deq[i] * x[i];
            abs_sum += fabs((double)deq[i] * x[i]);
        }
        mag = fmax(mag, abs_sum);                    // rounding x errs relative to this
    }
    double scale = 0;
    for (int r = 0; r < n; r++) scale = fmax(scale, fabs(ref[r]));
    for (int ti = 0; ti < 3; ti++) {
        if (!l3m_kernels_usable(TABLES[ti])) continue;
        TABLES[ti]->gemv[dtype](y[ti], tiles, ntiles, k, &v);
        double err = 0, err_x = 0, err_s = 0;
        for (int r = 0; r < n; r++) {
            err = fmax(err, fabs(y[ti][r] - ref[r])); err_x = fmax(err_x, fabs(y[ti][r] - ref_x[r]));
            err_s = fmax(err_s, fabs(y[ti][r] - y[0][r]));
        }
        int pass = err <= 1e-4 * scale && err_s <= 1e-4 * scale && err_x <= (dtype == L3M_F32 ? 1e-5 : dtype == L3M_BF16 ? 1e-3 : 2e-3) * mag;
        if (!pass) printf("  FAIL %-6s %-5s ntiles=%d k=%d missing=%d edge=%d: err=%.3g vs-exact-x=%.3g vs-scalar=%.3g scale=%.3g\n",
                          TABLES[ti]->name, l3m_dtype_name(dtype), ntiles, k, missing, edge, err, err_x, err_s, scale);
        ok &= pass;
    }
    free(rows); free(tiles); free(src); free(x); free(deq); free(ref); free(ref_x); free(scratch);
    for (int i = 0; i < 3; i++) free(y[i]);
    return ok;
}

static int check_attend(int head_dim, int pos, float qscale) {   // large qscale: scores past exp's clamp
    int stride = 3 * head_dim, ok = 1, n = (pos + 1) * stride;
    float *q = malloc(head_dim * 4), *kf = malloc(n * 4), *vf = malloc(n * 4), *out = malloc(head_dim * 4);
    uint16_t *k = malloc(n * 2), *v = malloc(n * 2);
    double *s = malloc((pos + 1) * sizeof(double)), *ref = calloc(head_dim, sizeof(double));
    for (int i = 0; i < head_dim; i++) q[i] = frand() * qscale;
    for (int i = 0; i < n; i++) { k[i] = l3m_f32_to_bf16(frand()); v[i] = l3m_f32_to_bf16(frand()); kf[i] = l3m_bf16_to_f32(k[i]); vf[i] = l3m_bf16_to_f32(v[i]); }
    double max = -1e300, sum = 0;
    for (int t = 0; t <= pos; t++) {
        s[t] = 0;
        for (int i = 0; i < head_dim; i++) s[t] += (double)q[i] * kf[t * stride + i];
        s[t] /= sqrt(head_dim);
        if (s[t] > max) max = s[t];
    }
    for (int t = 0; t <= pos; t++) sum += s[t] = exp(s[t] - max);
    for (int t = 0; t <= pos; t++) for (int i = 0; i < head_dim; i++) ref[i] += s[t] / sum * vf[t * stride + i];
    for (int ti = 0; ti < 3; ti++) {
        if (!l3m_kernels_usable(TABLES[ti])) continue;
        TABLES[ti]->attend(out, q, k, v, pos, head_dim, stride);
        double err = 0;
        for (int i = 0; i < head_dim; i++) err = fmax(err, fabs(out[i] - ref[i]));
        if (err > 1e-5) { printf("  FAIL %-6s attend head_dim=%d pos=%d: err=%.3g\n", TABLES[ti]->name, head_dim, pos, err); ok = 0; }
    }
    free(q); free(kf); free(vf); free(out); free(k); free(v); free(s); free(ref);
    return ok;
}

static double bench_ns(void (*fn)(void)) {
    double t0 = l3m_now_ns(), ns; int iters = 0;
    do { for (int i = 0; i < 100; i++) fn(); iters += 100; ns = l3m_now_ns() - t0; } while (ns < 1e8);
    return ns / iters;
}

static struct { const l3m_kernels *kt; int dtype, ntiles, k; void *tiles; float *y; l3m_vec v; } G;
static void run_gemv(void) { G.kt->gemv[G.dtype](G.y, G.tiles, G.ntiles, G.k, &G.v); }
static struct { const l3m_kernels *kt; float *q, *out; uint16_t *k, *v; } A;
static void run_attend(void) { A.kt->attend(A.out, A.q, A.k, A.v, 255, 64, 192); }
static struct { int dtype; float *x; void *scratch; l3m_vec v; } P;
static void run_prepare(void) { l3m_vec_prepare(&P.v, P.dtype, P.x, 768, P.scratch); }

static void throughput(const l3m_kernels *kt, int dtype) {
    int k = 3072;
    size_t tb = l3m_tile_bytes(dtype, k);
    G.kt = kt; G.dtype = dtype; G.k = k; G.ntiles = (int)((3u << 20) / tb);
    size_t bytes = G.ntiles * tb;
    G.tiles = aligned_alloc(64, bytes); memset(G.tiles, 0x3F, bytes);   // normal floats in every dtype
    G.y = malloc(G.ntiles * L3M_TILE * 4);
    float *x = malloc(k * 4); void *scratch = malloc(2 * k);
    for (int i = 0; i < k; i++) x[i] = frand();
    l3m_vec_prepare(&G.v, dtype, x, k, scratch);
    printf("  %-6s %-5s %5.1f GB/s  (%d tiles of %zu bytes, one core, resident)\n", kt->name, l3m_dtype_name(dtype), bytes / bench_ns(run_gemv), G.ntiles, tb);
    free(G.tiles); free(G.y); free(x); free(scratch);
}

int main(void) {
    for (int ti = 0; ti < 3; ti++) if (!l3m_kernels_usable(TABLES[ti])) printf("skip %s: not supported by this CPU\n", TABLES[ti]->name);
    const int KS[] = {288, 1376, 3072}, KS_FLOAT[] = {2, 18, 34};   // f32 takes any k, bf16 any even k
    int fails = 0;
    for (int dtype = 0; dtype < L3M_NDTYPES; dtype++) {
        int ok = 1;
        for (int ki = 0; ki < 3; ki++)
            for (int ntiles = 1; ntiles <= 9; ntiles++)
                for (int missing = 0; missing < 16; missing += 5) ok &= check(dtype, ntiles, KS[ki], missing, 0);
        for (int ki = 0; ki < 3; ki++) ok &= dtype <= L3M_BF16 ? check(dtype, 3, KS_FLOAT[ki], 1, 0) : check(dtype, 3, KS[ki], 1, 1);
        printf("%s %-5s  all tables vs double reference and scalar\n", ok ? "PASS" : "FAIL", l3m_dtype_name(dtype));
        fails += !ok;
    }
    const l3m_kernels *kt = l3m_kernels_pick();
    printf("throughput with %s:\n", kt->name);
    for (int dtype = 0; dtype < L3M_NDTYPES; dtype++) throughput(kt, dtype);
    int ok = 1;
    for (int h = 0; h < 5; h++)                      // 40: a partial zmm
        for (int p = 0; p < 4; p++) ok &= check_attend((int[]){16, 32, 40, 48, 64}[h], (int[]){0, 1, 37, 255}[p], 2);
    ok &= check_attend(64, 255, 40);
    printf("%s attend all tables vs double reference\n", ok ? "PASS" : "FAIL");
    fails += !ok;
    A.q = calloc(64, 4); A.out = calloc(64, 4); A.k = calloc(256 * 192, 2); A.v = calloc(256 * 192, 2);
    for (int ti = 0; ti < 3; ti++) { if (!l3m_kernels_usable(TABLES[ti])) continue; A.kt = TABLES[ti]; printf("  %-6s attend  %6.0f ns  (pos 255, head_dim 64)\n", A.kt->name, bench_ns(run_attend)); }
    P.x = calloc(768, 4); P.scratch = calloc(768, 2);
    for (int i = 0; i < 768; i++) P.x[i] = frand();
    for (P.dtype = L3M_BF16; P.dtype <= L3M_Q8_0; P.dtype++) printf("  vec_prepare %-5s %6.0f ns  (n = 768)\n", l3m_dtype_name(P.dtype), bench_ns(run_prepare));
    return fails != 0;
}
