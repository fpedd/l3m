// l3m on-disk format and model configuration.
//
// A .l3m file is: [header 1024 B][tensor data, each 64-byte aligned][toc][tokenizer].
// Matrices are [rows = outputs][cols = inputs], row-major, quantized in blocks of 32 along the inputs as in ggml.
#pragma once
#include <stdint.h>

#define L3M_VERSION 1

enum l3m_dtype { L3M_F32, L3M_BF16, L3M_Q8_0, L3M_Q4_0, L3M_MXFP4, L3M_NDTYPES };
#define L3M_QBLOCK 32   // inputs per quantization block

static inline const char *l3m_dtype_name(int dtype) {
    static const char *names[L3M_NDTYPES] = { "f32", "bf16", "q8_0", "q4_0", "mxfp4" };
    return dtype >= 0 && dtype < L3M_NDTYPES ? names[dtype] : "?";
}

// Architecture flags. Everything else (biases, gate matrix, lm_head) is "tensor present or not".
enum l3m_flags {
    L3M_F_LAYERNORM   = 1 << 0,  // LayerNorm with bias tensors, else RMSNorm
    L3M_F_LEARNED_POS = 1 << 1,  // pos_embed tensor, else RoPE (adjacent-pair, llama2.c convention)
    L3M_F_GELU        = 1 << 2,  // MLP is gelu_tanh(up), else silu(gate) * up
    L3M_F_TIED        = 1 << 3,  // lm_head is the embed tensor
    L3M_F_SP_PREFIX   = 1 << 4,  // SentencePiece tokenizer, prepends a space; else byte-level BPE
    L3M_F_DIGITS      = 1 << 5,  // byte-level BPE splits every digit off
};

typedef struct {
    char     magic[4];
    uint32_t version;
    char     arch[16];            // "llama", "gpt2", ... for display only
    uint32_t flags;
    uint32_t dim, n_layers, n_heads, n_kv_heads, ffn_dim, vocab, ctx;
    float    rope_theta, norm_eps;
    uint32_t n_tensors, reserved;         // reserved keeps the u64s 8-byte aligned
    uint64_t toc_offset, tokenizer_offset;
    uint8_t  pad[1024 - 88];
} l3m_header;
_Static_assert(sizeof(l3m_header) == 1024, "header is 1024 bytes");

// Table of contents entry, with a canonical name (below).
typedef struct {
    char     name[64];
    uint32_t dtype, ndim;
    uint64_t shape[4];            // [rows, cols] for matrices, [n] for vectors
    uint64_t offset, nbytes;      // offset from file start
} l3m_tensor;

// Tokenizer: header then n_tokens x { float score; uint16_t len; uint8_t bytes[len]; }.
// Scores order merges (higher first), as in llama2.c; -rank for byte-level BPE.
typedef struct { uint32_t n_tokens; int32_t bos, eos; uint32_t max_len; } l3m_tok_header;

// Quantization blocks, bit-for-bit ggml's.
typedef struct { uint16_t d; int8_t  qs[32]; } l3m_block_q8_0;   // f16 scale, x = q * d
typedef struct { uint16_t d; uint8_t qs[16]; } l3m_block_q4_0;   // f16 scale, nibble i: low = x[i], high = x[i+16], x = (q - 8) * d
typedef struct { uint8_t  e; uint8_t qs[16]; } l3m_block_mxfp4;  // e8m0 scale, same nibble order, x = lut[q] * 2^(e-127) / 2

// Canonical tensor names:
//   embed [vocab, dim]     pos_embed [ctx, dim]        norm, norm_b [dim]      lm_head [vocab, dim]
//   blk.N.attn_norm, attn_norm_b [dim]      blk.N.ffn_norm, ffn_norm_b [dim]
//   blk.N.wq [dim, dim]  wk, wv [kv_dim, dim]  wo [dim, dim]  bq, bo [dim]  bk, bv [kv_dim]
//   blk.N.w_gate, w_up [ffn, dim]  w_down [dim, ffn]  b_up [ffn]  b_down [dim]

static inline int l3m_header_ok(const l3m_header *h) {
    return h->magic[0] == 'l' && h->magic[1] == '3' && h->magic[2] == 'm' && h->version == L3M_VERSION;
}
