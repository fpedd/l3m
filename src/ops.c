// Norms, RoPE, attention, activations: plain C shared by the reference and the engine.
#include <math.h>
#include <stddef.h>
#include "ops.h"

float l3m_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 31, mant = h & 1023;
    union { uint32_t u; float f; } v;
    if (exp == 0) return ldexpf((float)mant, -24) * (sign ? -1.0f : 1.0f);      // zero or subnormal
    v.u = exp == 31 ? sign | 0x7F800000 | mant << 13 : sign | (exp + 112) << 23 | mant << 13;
    return v.f;
}

void l3m_rmsnorm(float *y, const float *x, const float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float s = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * s * w[i];
}

void l3m_layernorm(float *y, const float *x, const float *w, const float *b, int n, float eps) {
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    float s = 1.0f / sqrtf(var / n + eps);
    for (int i = 0; i < n; i++) y[i] = (x[i] - mean) * s * w[i] + b[i];
}

void l3m_rope(float *v, int n_heads, int head_dim, int pos, float theta) {
    for (int i = 0; i < head_dim / 2; i++) {
        float angle = pos * powf(theta, -2.0f * i / head_dim);
        float c = cosf(angle), s = sinf(angle);
        for (int h = 0; h < n_heads; h++) {
            float *p = v + h * head_dim + 2 * i;
            float a = p[0], b = p[1];
            p[0] = a * c - b * s;
            p[1] = a * s + b * c;
        }
    }
}

static void softmax(float *x, int n) {
    float max = x[0], sum = 0;
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    for (int i = 0; i < n; i++) sum += x[i] = expf(x[i] - max);
    for (int i = 0; i < n; i++) x[i] /= sum;
}

void l3m_silu_mul(float *h, const float *gate, int n) {
    for (int i = 0; i < n; i++) h[i] *= gate[i] / (1.0f + expf(-gate[i]));
}

void l3m_gelu(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = 0.5f * x[i] * (1.0f + tanhf(0.7978845608f * (x[i] + 0.044715f * x[i] * x[i] * x[i])));
}

void l3m_attend(float *out, const float *q, const uint16_t *k, const uint16_t *v, int pos, int head_dim, int stride) {
    float s[pos + 1];
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int t = 0; t <= pos; t++) {
        const uint16_t *kt = k + (size_t)t * stride;
        float dot = 0;
        for (int i = 0; i < head_dim; i++) dot += q[i] * l3m_bf16_to_f32(kt[i]);
        s[t] = dot * scale;
    }
    softmax(s, pos + 1);
    for (int i = 0; i < head_dim; i++) out[i] = 0;
    for (int t = 0; t <= pos; t++) {
        const uint16_t *vt = v + (size_t)t * stride;
        for (int i = 0; i < head_dim; i++) out[i] += s[t] * l3m_bf16_to_f32(vt[i]);
    }
}
