// The reference forward pass: one thread, fp32, every matrix dequantized at load. The oracle for engine.c.
#include <stdlib.h>
#include <string.h>
#include "kernels.h"
#include "ops.h"
#include "ref.h"

typedef struct {
    const float *attn_norm, *attn_norm_b, *ffn_norm, *ffn_norm_b;   // f32 vectors in the file
    const float *bq, *bk, *bv, *bo, *b_up, *b_down;                // optional biases
    float *wq, *wk, *wv, *wo, *w_gate, *w_up, *w_down;             // dequantized [rows][cols]
} ref_layer;

struct l3m_ref {
    l3m_header h;
    int ctx, head_dim, kv_dim;
    ref_layer *layer;
    const float *norm, *norm_b, *pos_embed;
    float *embed, *lm_head;              // lm_head == embed when tied
    uint16_t *kcache, *vcache;           // [layer][pos][kv_dim] bf16
    float *x, *xb, *q, *k, *v, *att, *hb, *g, *logits;   // per-token scratch
};

// Dequantize a 2-D tensor to f32 [rows][cols]. NULL in, NULL out.
static float *mat(const l3m_file *f, const l3m_tensor *t) {
    if (!t) return NULL;
    int rows = (int)t->shape[0], cols = (int)t->shape[1];
    float *out = malloc((size_t)rows * cols * sizeof(float));
    const char *src = l3m_file_data(f, t);
    size_t stride = l3m_row_bytes((int)t->dtype, cols);
    for (int r = 0; r < rows; r++) l3m_dequant_row(out + (size_t)r * cols, (int)t->dtype, src + r * stride, cols);
    return out;
}

// out[n] = W[n][k] . x[k]  (+ bias)
static void matvec(float *out, const float *w, const float *x, const float *bias, int n, int k) {
    for (int i = 0; i < n; i++) {
        const float *row = w + (size_t)i * k;
        float acc = 0.f;
        for (int j = 0; j < k; j++) acc += row[j] * x[j];
        out[i] = acc + (bias ? bias[i] : 0.f);
    }
}

static void add(float *x, const float *y, int n) { for (int i = 0; i < n; i++) x[i] += y[i]; }

static void norm(const l3m_ref *r, float *y, const float *x, const float *w, const float *b) {
    if (r->h.flags & L3M_F_LAYERNORM) l3m_layernorm(y, x, w, b, (int)r->h.dim, r->h.norm_eps);
    else l3m_rmsnorm(y, x, w, (int)r->h.dim, r->h.norm_eps);
}

l3m_ref *l3m_ref_load(const l3m_file *f, int ctx) {
    l3m_ref *r = calloc(1, sizeof *r);
    r->h = *f->h;
    r->ctx = ctx;
    r->head_dim = (int)(r->h.dim / r->h.n_heads);
    r->kv_dim = r->head_dim * (int)r->h.n_kv_heads;
    int dim = (int)r->h.dim, ffn = (int)r->h.ffn_dim, vocab = (int)r->h.vocab;

    r->embed = mat(f, l3m_file_tensor(f, "embed"));
    r->lm_head = (r->h.flags & L3M_F_TIED) ? r->embed : mat(f, l3m_file_tensor(f, "lm_head"));
    r->norm = l3m_file_f32(f, l3m_file_tensor(f, "norm"));
    r->norm_b = l3m_file_f32(f, l3m_file_tensor(f, "norm_b"));
    r->pos_embed = l3m_file_f32(f, l3m_file_tensor(f, "pos_embed"));

    r->layer = calloc(r->h.n_layers, sizeof *r->layer);
    for (int l = 0; l < (int)r->h.n_layers; l++) {
        ref_layer *L = &r->layer[l];
        L->attn_norm = l3m_file_f32(f, l3m_file_blk(f, l, "attn_norm"));
        L->attn_norm_b = l3m_file_f32(f, l3m_file_blk(f, l, "attn_norm_b"));
        L->ffn_norm = l3m_file_f32(f, l3m_file_blk(f, l, "ffn_norm"));
        L->ffn_norm_b = l3m_file_f32(f, l3m_file_blk(f, l, "ffn_norm_b"));
        L->wq = mat(f, l3m_file_blk(f, l, "wq"));
        L->wk = mat(f, l3m_file_blk(f, l, "wk"));
        L->wv = mat(f, l3m_file_blk(f, l, "wv"));
        L->wo = mat(f, l3m_file_blk(f, l, "wo"));
        L->w_gate = mat(f, l3m_file_blk(f, l, "w_gate"));
        L->w_up = mat(f, l3m_file_blk(f, l, "w_up"));
        L->w_down = mat(f, l3m_file_blk(f, l, "w_down"));
        L->bq = l3m_file_f32(f, l3m_file_blk(f, l, "bq"));
        L->bk = l3m_file_f32(f, l3m_file_blk(f, l, "bk"));
        L->bv = l3m_file_f32(f, l3m_file_blk(f, l, "bv"));
        L->bo = l3m_file_f32(f, l3m_file_blk(f, l, "bo"));
        L->b_up = l3m_file_f32(f, l3m_file_blk(f, l, "b_up"));
        L->b_down = l3m_file_f32(f, l3m_file_blk(f, l, "b_down"));
    }

    size_t kv = (size_t)r->h.n_layers * r->ctx * r->kv_dim;
    r->kcache = calloc(kv, sizeof(uint16_t));
    r->vcache = calloc(kv, sizeof(uint16_t));
    r->x = malloc(dim * sizeof(float));   r->xb = malloc(dim * sizeof(float));
    r->q = malloc(dim * sizeof(float));   r->att = malloc(dim * sizeof(float));
    r->k = malloc(r->kv_dim * sizeof(float)); r->v = malloc(r->kv_dim * sizeof(float));
    r->hb = malloc(ffn * sizeof(float));  r->g = malloc(ffn * sizeof(float));
    r->logits = malloc(vocab * sizeof(float));
    return r;
}

const float *l3m_ref_forward(l3m_ref *r, int32_t tok, int pos) {
    const l3m_header *h = &r->h;
    int dim = (int)h->dim, ffn = (int)h->ffn_dim, hd = r->head_dim;
    int heads_per_kv = (int)(h->n_heads / h->n_kv_heads);
    float *x = r->x, *xb = r->xb;

    memcpy(x, r->embed + (size_t)tok * dim, dim * sizeof(float));
    if (h->flags & L3M_F_LEARNED_POS) add(x, r->pos_embed + (size_t)pos * dim, dim);

    for (int l = 0; l < (int)h->n_layers; l++) {
        ref_layer *L = &r->layer[l];

        norm(r, xb, x, L->attn_norm, L->attn_norm_b);
        matvec(r->q, L->wq, xb, L->bq, dim, dim);
        matvec(r->k, L->wk, xb, L->bk, r->kv_dim, dim);
        matvec(r->v, L->wv, xb, L->bv, r->kv_dim, dim);
        if (!(h->flags & L3M_F_LEARNED_POS)) {
            l3m_rope(r->q, (int)h->n_heads, hd, pos, h->rope_theta);
            l3m_rope(r->k, (int)h->n_kv_heads, hd, pos, h->rope_theta);
        }
        uint16_t *kc = r->kcache + ((size_t)l * r->ctx + pos) * r->kv_dim;
        uint16_t *vc = r->vcache + ((size_t)l * r->ctx + pos) * r->kv_dim;
        for (int i = 0; i < r->kv_dim; i++) { kc[i] = l3m_f32_to_bf16(r->k[i]); vc[i] = l3m_f32_to_bf16(r->v[i]); }
        for (int hh = 0; hh < (int)h->n_heads; hh++) {
            size_t off = (size_t)l * r->ctx * r->kv_dim + (hh / heads_per_kv) * hd;
            l3m_attend(r->att + hh * hd, r->q + hh * hd, r->kcache + off, r->vcache + off, pos, hd, r->kv_dim);
        }
        matvec(xb, L->wo, r->att, L->bo, dim, dim);
        add(x, xb, dim);

        norm(r, xb, x, L->ffn_norm, L->ffn_norm_b);
        matvec(r->hb, L->w_up, xb, L->b_up, ffn, dim);
        if (h->flags & L3M_F_GELU) {
            l3m_gelu(r->hb, ffn);
        } else {
            matvec(r->g, L->w_gate, xb, NULL, ffn, dim);
            l3m_silu_mul(r->hb, r->g, ffn);
        }
        matvec(xb, L->w_down, r->hb, L->b_down, dim, ffn);
        add(x, xb, dim);
    }

    norm(r, xb, x, r->norm, r->norm_b);
    matvec(r->logits, r->lm_head, xb, NULL, (int)h->vocab, dim);
    return r->logits;
}

void l3m_ref_free(l3m_ref *r) {
    if (!r) return;
    for (int l = 0; l < (int)r->h.n_layers; l++) {
        ref_layer *L = &r->layer[l];
        free(L->wq); free(L->wk); free(L->wv); free(L->wo); free(L->w_gate); free(L->w_up); free(L->w_down);
    }
    if (r->lm_head != r->embed) free(r->lm_head);
    free(r->embed); free(r->layer); free(r->kcache); free(r->vcache);
    free(r->x); free(r->xb); free(r->q); free(r->k); free(r->v); free(r->att); free(r->hb); free(r->g); free(r->logits);
    free(r);
}
