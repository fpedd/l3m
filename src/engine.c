// The sharded engine: every matrix is output-sharded across the ranks; activations are all-gathered.
// Each shared buffer is written and read once per layer, and a rank cannot start layer l+1 before every
// rank has passed the last gather of layer l, so one copy of each is enough.
#include <cpuid.h>
#include <math.h>
#include <sched.h>
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include "l3m.h"
#include "file.h"
#include "hw.h"
#include "pool.h"
#include "kernels.h"
#include "ops.h"
#include "perf.h"
#include "ref.h"

#define T L3M_TILE
#define L3M_DEFAULT_CTX 512
#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef struct { const uint8_t *tiles; int ntiles, k, dtype; } shard;   // one rank's slice, packed

typedef struct {
    shard        qkv, wo, gate, up, down;
    const float *bqkv;                    // gathered like the qkv rows, padded; NULL without biases
    const float *bo, *b_up, *b_down;      // full vectors in the file, or NULL
    const float *attn_norm, *attn_norm_b, *ffn_norm, *ffn_norm_b;
    uint16_t    *kc, *vc;                 // KV cache of the owned kv heads: [ctx][kv_hi - kv_lo][hd] bf16
} layer_w;

typedef struct {
    int      rank, cpu;
    int      head_lo, head_hi;            // owned query heads
    int      kv_lo, kv_hi;                // the kv heads they read, shared with a neighbour when a group is split
    int      d_lo, d_hi, f_lo, f_hi, v_lo, v_hi;   // owned tiles of dim-, ffn- and vocab-sized outputs
    layer_w *layer;
    shard    lm_head;
    uint8_t *arena;                       // one huge-page region: this rank's shards, qkv bias and KV cache
    size_t   arena_size, arena_used, weight_bytes;
    float   *x, *xb, *qkv, *gate;         // private activations
    void    *scratch;                     // for l3m_vec_prepare
    float   *probs;                       // sampling scratch over the owned vocab slice
    double   cold_wait_ns;                // time in the post-flush sync, taken out of wait
} worker;

typedef struct { alignas(128) float max; int idx; double sum; } reduce_slot;

typedef struct {                          // each element written by one rank, read by all after a sync
    float       *attn, *proj, *h, *down, *logits, *probs;
    reduce_slot *red;
    alignas(128) int32_t tok;
    int          stop[2];
    double       u;
} shared;

struct l3m_model {
    l3m_file           f;
    l3m_hw             hw;
    const l3m_kernels *kk;
    l3m_pool          *pool;
    l3m_ref           *ref;                // set instead of the pool in reference mode
    int                n, dim, hd, ffn, vocab, ctx, n_heads, group, n_layers;   // group: query heads per kv head
    unsigned           flags;
    int                cold;
    float              eps, theta;
    worker            *w;
    shared             sh;
    size_t             weight_bytes;
    const l3m_tensor  *embed, *pos_embed, *lm_head;
    const float       *norm, *norm_b;
    // state of the running l3m_generate
    const int32_t     *prompt;
    int                n_prompt, max_new, n_gen;
    float              temp;
    uint64_t           rng;
    int              (*on_token)(int32_t, void *);
    void              *user;
    // accounting for l3m_perf_read
    l3m_hwcounters    *hc;
    double             tokens, ns, wait_ns, cold_ns;
    char               desc[160];
};

static void split(int n, int parts, int r, int *lo, int *hi) {
    *lo = (int)((long)n * r / parts);
    *hi = (int)((long)n * (r + 1) / parts);
}
static int tiles_of(int n) { return (n + T - 1) / T; }

static const void *row(const l3m_file *f, const l3m_tensor *t, int i) {
    return (const uint8_t *)l3m_file_data(f, t) + (size_t)i * l3m_row_bytes((int)t->dtype, (int)t->shape[1]);
}

static double rand01(uint64_t *s) {         // xorshift64*
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    return (double)((*s * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
}

static void *arena_alloc(worker *w, size_t bytes) {   // bump allocator; sizing pass when arena is NULL
    void *p = w->arena ? w->arena + w->arena_used : NULL;
    w->arena_used += (bytes + 63) & ~(size_t)63;
    return p;
}

// ---- building shards ----------------------------------------------------------------------

// Pack rows[0..n) (file layout, NULL = zero row) into whole tiles in the rank's arena.
static shard build(worker *w, int dtype, int k, const void **rows, int n) {
    shard s = { .ntiles = tiles_of(n), .k = k, .dtype = dtype };
    size_t tb = l3m_tile_bytes(dtype, k);
    uint8_t *dst = arena_alloc(w, (size_t)s.ntiles * tb);
    s.tiles = dst;
    if (!dst) return s;
    for (int t = 0; t < s.ntiles; t++) {
        const void *tr[T];
        for (int j = 0; j < T; j++) tr[j] = t * T + j < n ? rows[t * T + j] : NULL;
        l3m_pack_tile(dst + (size_t)t * tb, dtype, tr, k);
    }
    return s;
}

// The owned output tiles [lo, hi) of a matrix.
static shard build_range(worker *w, const l3m_file *f, const l3m_tensor *t, int lo, int hi) {
    int n = (int)t->shape[0], first = lo * T, count = MIN(hi * T, n) - first;
    const void **rows = malloc(sizeof *rows * (count > 0 ? count : 1));
    for (int i = 0; i < count; i++) rows[i] = row(f, t, first + i);
    shard s = build(w, (int)t->dtype, (int)t->shape[1], rows, count);
    free(rows);
    return s;
}

// The q rows of the owned heads, then the k and v rows of their kv heads; biases in the same order.
static void build_qkv(l3m_model *m, worker *w, int l, layer_w *L) {
    static const char *const mat[3] = { "wq", "wk", "wv" }, *const vec[3] = { "bq", "bk", "bv" };
    int lo[3] = { w->head_lo, w->kv_lo, w->kv_lo }, hi[3] = { w->head_hi, w->kv_hi, w->kv_hi };
    int n = (w->head_hi - w->head_lo + 2 * (w->kv_hi - w->kv_lo)) * m->hd;
    const void **rows = malloc(sizeof *rows * (n > 0 ? n : 1));
    float *bias = l3m_file_blk(&m->f, l, "bq") ? arena_alloc(w, (size_t)tiles_of(n) * T * sizeof(float)) : NULL;
    for (int p = 0, i = 0; p < 3; p++) {
        const l3m_tensor *t = l3m_file_blk(&m->f, l, mat[p]);
        const float *b = l3m_file_f32(&m->f, l3m_file_blk(&m->f, l, vec[p]));
        for (int r = lo[p] * m->hd; r < hi[p] * m->hd; r++, i++) {
            rows[i] = row(&m->f, t, r);
            if (bias) bias[i] = b[r];
        }
    }
    L->qkv  = build(w, (int)l3m_file_blk(&m->f, l, "wq")->dtype, m->dim, rows, n);
    L->bqkv = bias;
    free(rows);
}

// Everything a rank holds. With a NULL arena it only measures.
static void build_worker(l3m_model *m, worker *w) {
    l3m_file *f = &m->f;
    w->arena_used = 0;
    for (int l = 0; l < m->n_layers; l++) {
        layer_w *L = &w->layer[l];
        build_qkv(m, w, l, L);
        L->wo   = build_range(w, f, l3m_file_blk(f, l, "wo"), w->d_lo, w->d_hi);
        L->up   = build_range(w, f, l3m_file_blk(f, l, "w_up"), w->f_lo, w->f_hi);
        if (!(m->flags & L3M_F_GELU)) L->gate = build_range(w, f, l3m_file_blk(f, l, "w_gate"), w->f_lo, w->f_hi);
        L->down = build_range(w, f, l3m_file_blk(f, l, "w_down"), w->d_lo, w->d_hi);
        L->bo = l3m_file_f32(f, l3m_file_blk(f, l, "bo"));
        L->b_up = l3m_file_f32(f, l3m_file_blk(f, l, "b_up"));
        L->b_down = l3m_file_f32(f, l3m_file_blk(f, l, "b_down"));
        L->attn_norm = l3m_file_f32(f, l3m_file_blk(f, l, "attn_norm"));
        L->attn_norm_b = l3m_file_f32(f, l3m_file_blk(f, l, "attn_norm_b"));
        L->ffn_norm = l3m_file_f32(f, l3m_file_blk(f, l, "ffn_norm"));
        L->ffn_norm_b = l3m_file_f32(f, l3m_file_blk(f, l, "ffn_norm_b"));
    }
    w->lm_head = build_range(w, f, m->lm_head, w->v_lo, w->v_hi);
    w->weight_bytes = w->arena_used;
    size_t kv = (size_t)m->ctx * (w->kv_hi - w->kv_lo) * m->hd * sizeof(uint16_t);
    for (int l = 0; l < m->n_layers; l++) {
        w->layer[l].kc = arena_alloc(w, kv);
        w->layer[l].vc = arena_alloc(w, kv);
    }
}

static void job_build(l3m_pool *p, int rank, void *arg) {   // on the owning core, so pages are first-touched there
    (void)p;
    l3m_model *m = arg;
    worker *w = &m->w[rank];
    build_worker(m, w);
    w->arena_size = w->arena_used;
    w->arena = l3m_alloc_huge(w->arena_size);
    if (w->arena) build_worker(m, w);
}

// Vocab tiles go greedily to the rank with the most budget left.
static void deal_vocab(l3m_model *m) {
    double tb = (double)l3m_tile_bytes((int)m->lm_head->dtype, m->dim), room[L3M_MAX_CORES];
    int count[L3M_MAX_CORES] = {0};
    for (int r = 0; r < m->n; r++) {
        m->w[r].v_lo = m->w[r].v_hi = 0;
        build_worker(m, &m->w[r]);
        room[r] = (double)l3m_hw_nominal(&m->hw, m->w[r].cpu) * L3M_RESIDENT_FRACTION - (double)m->w[r].arena_used;
    }
    for (int t = 0; t < tiles_of(m->vocab); t++) {
        int r = 0;
        for (int q = 1; q < m->n; q++) if (room[q] > room[r]) r = q;
        room[r] -= tb; count[r]++;
    }
    int lo = 0;
    for (int r = 0; r < m->n; r++) {
        m->w[r].v_lo = lo;
        lo += count[r];
        m->w[r].v_hi = lo;
    }
}

// ---- the forward pass, one rank -----------------------------------------------------------

static void norm(const l3m_model *m, float *y, const float *x, const float *w, const float *b) {
    if (m->flags & L3M_F_LAYERNORM) l3m_layernorm(y, x, w, b, m->dim, m->eps);
    else l3m_rmsnorm(y, x, w, m->dim, m->eps);
}

static void gemv(const l3m_model *m, worker *w, float *y, const shard *s, const float *x) {
    l3m_vec v;
    l3m_vec_prepare(&v, s->dtype, x, s->k, w->scratch);
    m->kk->gemv[s->dtype](y, s->tiles, s->ntiles, s->k, &v);
}

static void add_bias(float *y, const float *b, int lo, int hi) {   // y is indexed from lo
    if (b) for (int i = lo; i < hi; i++) y[i - lo] += b[i];
}
static void add(float *x, const float *y, int n) { for (int i = 0; i < n; i++) x[i] += y[i]; }

static void forward(l3m_model *m, worker *w, int32_t tok, int pos, int want_logits) {
    shared *s = &m->sh;
    l3m_file *f = &m->f;
    int dim = m->dim, hd = m->hd, nloc = w->head_hi - w->head_lo, nkv = w->kv_hi - w->kv_lo, r = w->rank;
    int d_lo = w->d_lo * T, d_hi = MIN(w->d_hi * T, dim), f_lo = w->f_lo * T, f_hi = MIN(w->f_hi * T, m->ffn);

    l3m_dequant_row(w->x, (int)m->embed->dtype, row(f, m->embed, tok), dim);
    if (m->pos_embed) add(w->x, row(f, m->pos_embed, pos), dim);

    for (int l = 0; l < m->n_layers; l++) {
        layer_w *L = &w->layer[l];
        norm(m, w->xb, w->x, L->attn_norm, L->attn_norm_b);
        if (nloc) {
            float *q = w->qkv, *k = q + nloc * hd, *v = k + nkv * hd;
            gemv(m, w, q, &L->qkv, w->xb);
            if (L->bqkv) add(q, L->bqkv, (nloc + 2 * nkv) * hd);
            if (!(m->flags & L3M_F_LEARNED_POS)) { l3m_rope(q, nloc, hd, pos, m->theta); l3m_rope(k, nkv, hd, pos, m->theta); }
            uint16_t *kc = L->kc + (size_t)pos * nkv * hd, *vc = L->vc + (size_t)pos * nkv * hd;
            for (int i = 0; i < nkv * hd; i++) { kc[i] = l3m_f32_to_bf16(k[i]); vc[i] = l3m_f32_to_bf16(v[i]); }
            for (int j = 0; j < nloc; j++) {
                int kvj = (w->head_lo + j) / m->group - w->kv_lo;
                m->kk->attend(s->attn + (w->head_lo + j) * hd, q + j * hd, L->kc + kvj * hd, L->vc + kvj * hd, pos, hd, nkv * hd);
            }
        }
        l3m_sync(m->pool, r);

        gemv(m, w, s->proj + d_lo, &L->wo, s->attn);
        add_bias(s->proj + d_lo, L->bo, d_lo, d_hi);
        l3m_sync(m->pool, r);
        add(w->x, s->proj, dim);

        norm(m, w->xb, w->x, L->ffn_norm, L->ffn_norm_b);
        gemv(m, w, s->h + f_lo, &L->up, w->xb);
        add_bias(s->h + f_lo, L->b_up, f_lo, f_hi);
        if (m->flags & L3M_F_GELU) {
            l3m_gelu(s->h + f_lo, f_hi - f_lo);
        } else {
            gemv(m, w, w->gate, &L->gate, w->xb);
            l3m_silu_mul(s->h + f_lo, w->gate, f_hi - f_lo);
        }
        l3m_sync(m->pool, r);

        gemv(m, w, s->down + d_lo, &L->down, s->h);
        add_bias(s->down + d_lo, L->b_down, d_lo, d_hi);
        l3m_sync(m->pool, r);
        add(w->x, s->down, dim);
    }
    if (!want_logits) return;
    norm(m, w->xb, w->x, m->norm, m->norm_b);
    gemv(m, w, s->logits + w->v_lo * T, &w->lm_head, w->xb);
    l3m_sync(m->pool, r);
}

// Unnormalized; below e^-16 of the maximum counts as 0.
static float prob(float logit, float gmax, float temp) {
    float d = (logit - gmax) / temp;
    return d < -16 ? 0 : expf(d);
}

// Each rank reduces its vocab slice; the rank whose slice holds the sample picks it.
static int32_t sample(l3m_model *m, worker *w) {
    shared *s = &m->sh;
    int r = w->rank, lo = w->v_lo * T, hi = MIN(w->v_hi * T, m->vocab);
    float mx = -INFINITY; int idx = lo;
    for (int i = lo; i < hi; i++) if (s->logits[i] > mx) { mx = s->logits[i]; idx = i; }
    s->red[r].max = mx; s->red[r].idx = idx;
    if (r == 0) s->u = rand01(&m->rng);
    l3m_sync(m->pool, r);

    float gmax = -INFINITY; int best = 0;
    for (int q = 0; q < m->n; q++) if (s->red[q].max > gmax) { gmax = s->red[q].max; best = s->red[q].idx; }
    if (!(m->temp > 0)) return best;                   // NaN too

    double sum = 0;
    for (int i = lo; i < hi; i++) sum += w->probs[i - lo] = prob(s->logits[i], gmax, m->temp);
    s->red[r].sum = sum;
    l3m_sync(m->pool, r);

    double z = 0;
    for (int q = 0; q < m->n; q++) z += s->red[q].sum;
    double target = s->u * z;
    int owner = 0;
    while (owner < m->n - 1 && target >= s->red[owner].sum) target -= s->red[owner++].sum;
    if (r == owner) {
        int t = hi - 1; double c = 0;
        for (int i = lo; i < hi; i++) if ((c += w->probs[i - lo]) > target) { t = i; break; }
        s->tok = t;
    }
    l3m_sync(m->pool, r);
    return s->tok;
}

static int exhausted(const l3m_model *m, int pos) { return m->n_gen >= m->max_new || pos + 1 >= m->ctx; }
static int accepted(l3m_model *m, int32_t tok) {   // 0 at EOS, BOS, or when on_token says stop
    if (tok == m->f.tok.eos || tok == m->f.tok.bos) return 0;
    m->n_gen++;
    return !m->on_token || m->on_token(tok, m->user);
}

// --cold: flush the rank's weights from every cache level; the flush and its sync are not token time.
__attribute__((target("clflushopt")))
static void evict(l3m_model *m, worker *w) {
    double t0 = l3m_now_ns(), w0 = l3m_sync_wait_ns(m->pool, w->rank);
    for (size_t i = 0; i < w->weight_bytes; i += 64) _mm_clflushopt(w->arena + i);
    _mm_sfence();
    l3m_sync(m->pool, w->rank);
    w->cold_wait_ns += l3m_sync_wait_ns(m->pool, w->rank) - w0;
    if (w->rank == 0) m->cold_ns += l3m_now_ns() - t0;
}

// Rank 0 decides when to stop; the others read it after a sync. Two slots by position parity,
// so rank 0's next decision never races a slow rank's read.
static void job_generate(l3m_pool *p, int rank, void *arg) {
    l3m_model *m = arg;
    worker *w = &m->w[rank];
    int32_t tok = 0;
    int pos = 0;
    for (;; pos++) {
        int last = pos >= m->n_prompt - 1;
        int *stop = &m->sh.stop[pos & 1];
        if (pos < m->n_prompt) tok = m->prompt[pos];
        if (m->cold) evict(m, w);
        if (rank == 0) *stop = last && exhausted(m, pos);   // published by the logits sync inside forward
        forward(m, w, tok, pos, last);
        if (!last) continue;
        if (*stop) break;
        tok = sample(m, w);
        if (rank == 0) *stop = !accepted(m, tok);
        l3m_sync(p, rank);
        if (*stop) break;
    }
    if (rank == 0) m->tokens += pos + 1;
}

static void generate_ref(l3m_model *m) {   // the same loop on one thread
    int32_t tok = 0;
    int pos = 0;
    for (;; pos++) {
        if (pos < m->n_prompt) tok = m->prompt[pos];
        const float *logits = l3m_ref_forward(m->ref, tok, pos);
        if (pos < m->n_prompt - 1) continue;
        memcpy(m->sh.logits, logits, m->vocab * sizeof(float));
        if (exhausted(m, pos)) break;
        float gmax = -INFINITY; int best = 0;
        for (int i = 0; i < m->vocab; i++) if (logits[i] > gmax) { gmax = logits[i]; best = i; }
        tok = best;
        if (m->temp > 0) {
            double z = 0;
            for (int i = 0; i < m->vocab; i++) z += m->sh.probs[i] = prob(logits[i], gmax, m->temp);
            double target = rand01(&m->rng) * z, c = 0;
            tok = m->vocab - 1;
            for (int i = 0; i < m->vocab; i++) if ((c += m->sh.probs[i]) > target) { tok = i; break; }
        }
        if (!accepted(m, tok)) break;
    }
    m->tokens += pos + 1;
}

// ---- public API ---------------------------------------------------------------------------

// One allowed CPU per physical core of the largest L3, or of every L3 when wide.
static int default_cpus(const l3m_hw *hw, int wide, int *cpus) {
    cpu_set_t ok;
    sched_getaffinity(0, sizeof ok, &ok);
    int pick[L3M_MAX_CORES], g = -1, n = 0;
    for (int c = 0; c < hw->n_cores; c++) pick[c] = -1;
    for (int cpu = 0; cpu < CPU_SETSIZE && cpu < L3M_MAX_CPUS; cpu++) {
        int c = l3m_hw_core(hw, cpu);
        if (c >= 0 && pick[c] < 0 && CPU_ISSET(cpu, &ok)) pick[c] = cpu;
    }
    for (int c = 0; c < hw->n_cores; c++)
        if (pick[c] >= 0 && (g < 0 || hw->l3_bytes[hw->core[c].group] > hw->l3_bytes[g])) g = hw->core[c].group;
    for (int c = 0; c < hw->n_cores; c++)
        if (pick[c] >= 0 && (wide || hw->core[c].group == g)) cpus[n++] = pick[c];
    return n;
}

// 2 if a cpu repeats, 1 if two cpus are SMT siblings of one core.
static int shares_core(const l3m_hw *hw, const int *cpus, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        int c = l3m_hw_core(hw, cpus[i]);
        for (int j = 0; j < i; j++) {
            if (cpus[i] == cpus[j]) return 2;
            if (c >= 0 && c == l3m_hw_core(hw, cpus[j])) s = 1;
        }
    }
    return s;
}

static int has_clflushopt(void) {
    unsigned a, b, c, d;
    return __get_cpuid_count(7, 0, &a, &b, &c, &d) && (b & bit_CLFLUSHOPT);
}

static l3m_model *fail(l3m_model *m, const char *msg) { fprintf(stderr, "l3m: %s\n", msg); l3m_free(m); return NULL; }

l3m_model *l3m_load(const char *path, const l3m_opts *opts) {
    l3m_opts o = opts ? *opts : (l3m_opts){0};
    l3m_model *m = aligned_alloc(alignof(l3m_model), sizeof *m);
    if (!m) return NULL;
    memset(m, 0, sizeof *m);
    if (l3m_file_open(&m->f, path)) { free(m); return NULL; }
    const l3m_header *h = m->f.h;
    m->dim = h->dim; m->n_heads = h->n_heads; m->group = h->n_heads / h->n_kv_heads; m->hd = h->dim / h->n_heads;
    m->ffn = h->ffn_dim; m->vocab = h->vocab; m->n_layers = h->n_layers; m->flags = h->flags;
    m->eps = h->norm_eps; m->theta = h->rope_theta; m->cold = o.cold;
    m->ctx = MIN((int)h->ctx, o.ctx > 0 ? o.ctx : L3M_DEFAULT_CTX);
    m->embed = l3m_file_tensor(&m->f, "embed");
    m->pos_embed = (m->flags & L3M_F_LEARNED_POS) ? l3m_file_tensor(&m->f, "pos_embed") : NULL;
    m->lm_head = l3m_file_tensor(&m->f, (m->flags & L3M_F_TIED) ? "embed" : "lm_head");
    m->norm = l3m_file_f32(&m->f, l3m_file_tensor(&m->f, "norm"));
    m->norm_b = l3m_file_f32(&m->f, l3m_file_tensor(&m->f, "norm_b"));
    l3m_hw_probe(&m->hw);
    m->kk = l3m_kernels_pick();
    const char *base = strrchr(path, '/'); base = base ? base + 1 : path;
    int dtype = (int)l3m_file_blk(&m->f, 0, "w_up")->dtype;

    if (o.reference) {
        m->ref = l3m_ref_load(&m->f, m->ctx);
        m->sh.logits = calloc(m->vocab, sizeof(float));
        m->sh.probs = calloc(m->vocab, sizeof(float));
        snprintf(m->desc, sizeof m->desc, "%s: %s, %d layers, reference fp32 on one core", base, l3m_dtype_name(dtype), m->n_layers);
        return m;
    }

    int cpus[L3M_MAX_CORES], n = o.cpus ? o.n_cpus : default_cpus(&m->hw, 0, cpus);
    if (n < 1 || n > L3M_MAX_CORES) return fail(m, "bad core list");
    if (o.cpus) memcpy(cpus, o.cpus, n * sizeof(int));
    int shared_core = shares_core(&m->hw, cpus, n);
    if (shared_core == 2) return fail(m, "a cpu appears twice in --cores");
    if (shared_core && !o.force) return fail(m, "two --cores on one physical core: use one SMT sibling per core, or --force");
    if (m->cold && !has_clflushopt()) return fail(m, "--cold needs clflushopt, which this CPU lacks");
    m->n = n;
    m->pool = l3m_pool_start(cpus, n);
    if (!m->pool) return fail(m, "could not pin a thread to each core (--cores: every cpu must exist, be allowed, and appear once)");

    int scratch_n = m->dim > m->ffn ? m->dim : m->ffn;
    m->w = calloc(n, sizeof *m->w);
    for (int r = 0; r < n; r++) {
        worker *w = &m->w[r];
        w->rank = r; w->cpu = cpus[r];
        w->layer = calloc(m->n_layers, sizeof *w->layer);
        split(m->n_heads, n, r, &w->head_lo, &w->head_hi);
        w->kv_lo = w->head_lo / m->group;
        w->kv_hi = w->head_hi > w->head_lo ? (w->head_hi - 1) / m->group + 1 : w->kv_lo;
        split(tiles_of(m->dim), n, r, &w->d_lo, &w->d_hi);
        split(tiles_of(m->ffn), n, r, &w->f_lo, &w->f_hi);
        w->x = calloc(m->dim, sizeof(float));
        w->xb = calloc(m->dim, sizeof(float));
        w->qkv = calloc((size_t)tiles_of((w->head_hi - w->head_lo + 2 * (w->kv_hi - w->kv_lo)) * m->hd) * T, sizeof(float));
        w->gate = calloc((size_t)(w->f_hi - w->f_lo) * T, sizeof(float));
        w->scratch = calloc(2 * scratch_n + 64, 1);
    }
    deal_vocab(m);
    for (int r = 0; r < n; r++) m->w[r].probs = calloc((size_t)(m->w[r].v_hi - m->w[r].v_lo) * T + 1, sizeof(float));
    shared *s = &m->sh;
    s->attn = calloc((size_t)tiles_of(m->dim) * T, sizeof(float));
    s->proj = calloc((size_t)tiles_of(m->dim) * T, sizeof(float));
    s->down = calloc((size_t)tiles_of(m->dim) * T, sizeof(float));
    s->h = calloc((size_t)tiles_of(m->ffn) * T, sizeof(float));
    s->logits = calloc((size_t)tiles_of(m->vocab) * T, sizeof(float));
    s->red = aligned_alloc(128, n * sizeof(reduce_slot));
    memset(s->red, 0, n * sizeof(reduce_slot));

    l3m_pool_run(m->pool, job_build, m);
    for (int r = 0; r < n; r++) if (!m->w[r].arena) return fail(m, "shard allocation failed");

    // Over budget: the default placement first widens to every L3, then the load fails unless forced.
    int over = 0;
    for (int r = 0; r < n; r++) over |= m->w[r].arena_size > l3m_hw_nominal(&m->hw, m->w[r].cpu) * L3M_RESIDENT_FRACTION;
    int all[L3M_MAX_CORES], n_all = over && !o.cpus ? default_cpus(&m->hw, 1, all) : 0;
    if (n_all > n) {
        o.cpus = all; o.n_cpus = n_all;
        l3m_free(m);
        return l3m_load(path, &o);
    }
    double budget = 0;
    for (int r = 0; r < n; r++) {
        worker *w = &m->w[r];
        double nominal = (double)l3m_hw_nominal(&m->hw, w->cpu);
        int bad = w->arena_size > nominal * L3M_RESIDENT_FRACTION;
        budget += nominal * L3M_RESIDENT_FRACTION;
        m->weight_bytes += w->weight_bytes;
        if (o.verbose || (bad && !o.force))
            fprintf(stderr, "l3m: rank %2d cpu %3d  heads [%d, %d)  vocab tiles [%d, %d)  weights %.2f + kv %.2f MiB, "
                            "resident budget %.2f of %.2f nominal%s\n",
                    r, w->cpu, w->head_lo, w->head_hi, w->v_lo, w->v_hi,
                    w->weight_bytes / MiB, (w->arena_size - w->weight_bytes) / MiB,
                    nominal * L3M_RESIDENT_FRACTION / MiB, nominal / MiB, bad ? "  OVER" : "");
    }
    if (over && !o.force) {
        char msg[256], ctx[48] = "";
        if (m->weight_bytes < budget) snprintf(ctx, sizeof ctx, "a shorter --ctx (now %d), ", m->ctx);
        snprintf(msg, sizeof msg, "the shards will not stay cache resident on %d core%s: use %s%sa smaller --dtype at export, "
                 "or --force", n, n > 1 ? "s" : "", n < m->hw.n_cores ? "more --cores, " : "", ctx);
        return fail(m, msg);
    }
    m->hc = l3m_hwcounters_open(o.verbose);
    snprintf(m->desc, sizeof m->desc, "%s: %s, %d layers, %.1f MiB on %d core%s (%s)",
             base, l3m_dtype_name(dtype), m->n_layers, m->weight_bytes / MiB, n, n > 1 ? "s" : "", m->kk->name);
    if (o.verbose) fprintf(stderr, "l3m: %s, ctx %d\n", m->desc, m->ctx);
    return m;
}

void l3m_free(l3m_model *m) {
    if (!m) return;
    if (m->pool) l3m_pool_stop(m->pool);
    if (m->ref) l3m_ref_free(m->ref);
    if (m->hc) l3m_hwcounters_close(m->hc);
    for (int r = 0; m->w && r < m->n; r++) {
        worker *w = &m->w[r];
        if (w->arena) l3m_free_huge(w->arena, w->arena_size);
        free(w->layer); free(w->x); free(w->xb); free(w->qkv); free(w->gate); free(w->probs); free(w->scratch);
    }
    free(m->w);
    free(m->sh.attn); free(m->sh.proj); free(m->sh.down); free(m->sh.h); free(m->sh.logits); free(m->sh.probs); free(m->sh.red);
    l3m_file_close(&m->f);
    free(m);
}

int l3m_generate(l3m_model *m, const int32_t *prompt, int n_prompt, int max_new, float temperature, uint64_t seed,
                 int (*on_token)(int32_t, void *), void *user) {
    if (n_prompt < 1 || n_prompt >= m->ctx) return 0;
    for (int i = 0; i < n_prompt; i++)
        if (prompt[i] < 0 || prompt[i] >= m->vocab) { fprintf(stderr, "l3m: token %d out of range\n", prompt[i]); return -1; }
    m->prompt = prompt; m->n_prompt = n_prompt; m->max_new = max_new; m->n_gen = 0;
    m->temp = temperature; m->rng = seed * 0x9E3779B97F4A7C15ULL + 1; m->on_token = on_token; m->user = user;
    m->sh.stop[0] = m->sh.stop[1] = 0;
    double t0 = l3m_now_ns();
    if (m->ref) generate_ref(m);
    else l3m_pool_run(m->pool, job_generate, m);
    m->ns += l3m_now_ns() - t0;
    return m->n_gen;
}

const float *l3m_logits(const l3m_model *m) { return m->sh.logits; }
int l3m_vocab(const l3m_model *m) { return m->vocab; }
int l3m_ctx(const l3m_model *m) { return m->ctx; }
const char *l3m_describe(const l3m_model *m) { return m->desc; }
int l3m_bos(const l3m_model *m) { return m->f.tok.bos; }
int l3m_eos(const l3m_model *m) { return m->f.tok.eos; }

int l3m_encode(const l3m_model *m, const char *text, int add_bos, int32_t *out, int max) {
    return l3m_tokenizer_encode(&m->f.tok, text, add_bos, out, max);
}
const char *l3m_decode(const l3m_model *m, int32_t tok, int *len) {
    return (const char *)l3m_tokenizer_decode(&m->f.tok, tok, len);
}

int l3m_perf_read(l3m_model *m, l3m_perf *out) {
    memset(out, 0, sizeof *out);
    out->l3_miss = out->dram_bytes = out->energy_j = out->llc_occupancy = NAN;
    double wait = 0;
    for (int r = 0; r < m->n; r++) wait += l3m_sync_wait_ns(m->pool, r) - m->w[r].cold_wait_ns;
    wait = m->n ? wait / m->n : 0;
    double tokens = m->tokens, ns = m->ns - m->cold_ns, dwait = wait - m->wait_ns;
    m->tokens = 0; m->ns = 0; m->cold_ns = 0; m->wait_ns = wait;
    if (tokens <= 0) return -1;
    out->tokens = tokens;
    out->ns_token = ns / tokens;
    out->ns_wait = dwait / tokens;
    out->ns_compute = out->ns_token - out->ns_wait;
    out->bytes_token = (double)m->weight_bytes;
    out->l2_bytes = 0;
    for (int r = 0; r < m->n; r++) out->l2_bytes += (double)l3m_hw_l2(&m->hw, m->w[r].cpu);
    if (m->hc) {
        l3m_counters c;
        l3m_hwcounters_read(m->hc, &c);
        out->l3_miss = c.l3_miss / tokens;
        out->dram_bytes = c.dram_bytes / tokens;
        out->energy_j = c.energy_j / tokens;
        out->llc_occupancy = c.llc_occupancy;
    }
    return 0;
}

int l3m_parse_cpus(const char *list, int *cpus, int max) {
    for (int n = 0;; list++) {
        char *end;
        if (*list < '0' || *list > '9') return -1;
        long a = strtol(list, &end, 10), b = a;
        if (*end == '-') {
            if (end[1] < '0' || end[1] > '9') return -1;
            b = strtol(end + 1, &end, 10);
        }
        if (b < a || b >= L3M_MAX_CPUS) return -1;
        for (long c = a; c <= b; c++) { if (n == max) return -1; cpus[n++] = (int)c; }
        if (!*end) return n;
        if (*end != ',') return -1;
        list = end;
    }
}

void l3m_print_hw(FILE *out) {
    l3m_hw hw;
    if (l3m_hw_probe(&hw)) { fprintf(out, "topology unavailable\n"); return; }
    l3m_hw_print(&hw, out);
}
