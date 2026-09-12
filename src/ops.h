// Per-token math shared by every path: norms, RoPE, attention, activations.
#pragma once
#include <stdint.h>

static inline float    l3m_bf16_to_f32(uint16_t h) { union { uint32_t u; float f; } v = { (uint32_t)h << 16 }; return v.f; }
static inline uint16_t l3m_f32_to_bf16(float f)    { union { float f; uint32_t u; } v = { f }; return (uint16_t)((v.u + 0x7FFF + ((v.u >> 16) & 1)) >> 16); }
float l3m_f16_to_f32(uint16_t h);

void l3m_rmsnorm(float *y, const float *x, const float *w, int n, float eps);
void l3m_layernorm(float *y, const float *x, const float *w, const float *b, int n, float eps);
void l3m_rope(float *v, int n_heads, int head_dim, int pos, float theta);   // in place, pairs (2i, 2i+1)
void l3m_silu_mul(float *h, const float *gate, int n);                       // h = silu(gate) * h
void l3m_gelu(float *x, int n);                                              // tanh approximation

// One head of causal attention at position `pos` over a bf16 KV cache.
// k and v point at row 0 of this head; rows are `stride` elements apart.
void l3m_attend(float *out, const float *q, const uint16_t *k, const uint16_t *v, int pos, int head_dim, int stride);
