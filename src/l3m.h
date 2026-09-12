// l3m public C API: load a model, generate, tokenize, read the counters.
#pragma once
#include <stdint.h>
#include <stdio.h>

#define L3M_API __attribute__((visibility("default")))

typedef struct l3m_model l3m_model;

typedef struct {
    const int *cpus;       // cores to run on, in rank order; NULL = every physical core of the largest L3,
                           // or of the machine if the model does not fit there
    int        n_cpus;
    int        ctx;        // KV cache length, 0 = min(model context, 512)
    int        reference;  // 1 = single-thread fp32 reference path
    int        force;      // 1 = load even if the shards will not stay cache resident
    int        verbose;    // print each rank's shard sizes against its cache budget, on stderr
    int        cold;       // 1 = flush the weights from every cache before each token
} l3m_opts;

typedef struct {
    double tokens;         // forward passes since the previous read, prompt included
    double ns_token;       // wall time per token
    double ns_compute;     // per token outside l3m_sync, averaged over ranks (mostly gemv)
    double ns_wait;        // per token in l3m_sync, averaged over ranks
    double bytes_token;    // weight bytes streamed per token
    double l2_bytes;       // private L2 summed over the ranks
    double l3_miss, dram_bytes, energy_j, llc_occupancy;   // counters per token, occupancy in bytes; NAN if unavailable
} l3m_perf;

// opts may be NULL. Returns NULL on failure, with the reason on stderr.
L3M_API l3m_model   *l3m_load(const char *path, const l3m_opts *opts);
L3M_API void         l3m_free(l3m_model *m);

// Feed the prompt, then sample up to max_new tokens, stopping early at EOS, BOS or the end of the
// context. on_token (may be NULL) is called with each new token, never the prompt, EOS or BOS,
// and stops generation by returning 0. Returns the number of tokens generated, 0 if the prompt
// is empty or does not fit in ctx, -1 if a prompt id is outside [0, vocab). temperature 0 = greedy.
// A model runs one generate at a time: calls on the same model must not overlap, nor come from on_token.
L3M_API int          l3m_generate(l3m_model *m, const int32_t *prompt, int n_prompt, int max_new,
                                  float temperature, uint64_t seed, int (*on_token)(int32_t tok, void *user), void *user);
L3M_API const float *l3m_logits(const l3m_model *m);   // vocab floats: the prediction after the last token fed
L3M_API int          l3m_vocab(const l3m_model *m);
L3M_API int          l3m_ctx(const l3m_model *m);
L3M_API const char  *l3m_describe(const l3m_model *m);  // "<file>: <dtype>, <n> layers, <size> MiB on <n> cores (<isa>)"

// l3m_encode returns the token count, or max when the ids do not fit in max.
L3M_API int          l3m_encode(const l3m_model *m, const char *text, int add_bos, int32_t *out, int max);
L3M_API const char  *l3m_decode(const l3m_model *m, int32_t tok, int *len);   // raw bytes, not NUL-terminated
L3M_API int          l3m_bos(const l3m_model *m);   // -1 if the model has none
L3M_API int          l3m_eos(const l3m_model *m);

L3M_API int          l3m_perf_read(l3m_model *m, l3m_perf *out);   // since the previous read; -1 if nothing ran

L3M_API int          l3m_parse_cpus(const char *list, int *cpus, int max);   // "0-7,12" -> count, -1 on error
L3M_API void         l3m_print_hw(FILE *out);                        // cores, L3 groups, per-core cache budgets
