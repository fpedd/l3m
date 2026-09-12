// A mapped .l3m file, validated once at open so a malformed file is an error, never a bad read later.
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "file.h"
#include "kernels.h"

#define MAX_DIM (1u << 24)   // bound on every header size and tensor dimension, so products stay in range

static int bad(l3m_file *f, const char *path, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "%s: ", path);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    l3m_file_close(f);
    return -1;
}

static const char *tensor_error(const l3m_file *f, const l3m_tensor *t) {
    if (!memchr(t->name, 0, sizeof t->name)) return "unterminated name";
    if (t->dtype >= L3M_NDTYPES || t->ndim < 1 || t->ndim > 2) return "bad dtype or ndim";
    uint64_t rows = t->ndim == 2 ? t->shape[0] : 1, cols = t->shape[t->ndim - 1];
    if (!rows || !cols || rows > MAX_DIM || cols > MAX_DIM) return "bad shape";
    if ((t->dtype >= L3M_Q8_0 && cols % L3M_QBLOCK) || (t->dtype == L3M_BF16 && cols % 2)) return "rows are not whole blocks";
    if (t->nbytes != rows * l3m_row_bytes((int)t->dtype, (int)cols)) return "nbytes does not match the shape";
    if (t->offset % 64 || t->offset > f->size || t->nbytes > f->size - t->offset) return "misaligned or past the end of the file";
    return NULL;
}

// 1 if present and well-shaped, 0 if absent and optional, else -1 with a message. cols 0 means a vector.
static int want(const l3m_file *f, const char *path, const char *name, int required, int f32, uint64_t rows, uint64_t cols) {
    const l3m_tensor *t = l3m_file_tensor(f, name);
    if (!t) {
        if (required) fprintf(stderr, "%s: tensor %s missing\n", path, name);
        return required ? -1 : 0;
    }
    if (t->ndim != (cols ? 2u : 1u) || t->shape[0] != rows || (cols && t->shape[1] != cols) || (f32 && t->dtype != L3M_F32)) {
        char shape[48];
        snprintf(shape, sizeof shape, cols ? "[%llu, %llu]" : "[%llu]", (unsigned long long)rows, (unsigned long long)cols);
        fprintf(stderr, "%s: tensor %s: expected %s%s\n", path, name, f32 ? "f32 " : "", shape);
        return -1;
    }
    return 1;
}

static int check_model(const l3m_file *f, const char *path) {
    const l3m_header *h = f->h;
    uint32_t dim = h->dim, ffn = h->ffn_dim, vocab = h->vocab;
    if (!dim || !ffn || !vocab || !h->ctx || !h->n_layers || !h->n_heads || !h->n_kv_heads || dim > MAX_DIM || ffn > MAX_DIM ||
        vocab > MAX_DIM || h->ctx > MAX_DIM || h->n_layers > MAX_DIM || dim % h->n_heads || h->n_heads % h->n_kv_heads ||
        dim / h->n_heads % 8) {
        fprintf(stderr, "%s: bad config: dim %u, heads %u, kv_heads %u (head_dim a multiple of 8), ffn %u, layers %u, vocab %u, ctx %u\n",
                path, dim, h->n_heads, h->n_kv_heads, ffn, h->n_layers, vocab, h->ctx);
        return -1;
    }
    uint32_t kv = dim / h->n_heads * h->n_kv_heads;
    int ln = h->flags & L3M_F_LAYERNORM, gelu = h->flags & L3M_F_GELU, e = 0;
    e |= want(f, path, "embed", 1, 0, vocab, dim) < 0;
    if (!(h->flags & L3M_F_TIED)) e |= want(f, path, "lm_head", 1, 0, vocab, dim) < 0;
    e |= want(f, path, "pos_embed", h->flags & L3M_F_LEARNED_POS, 1, h->ctx, dim) < 0;
    e |= want(f, path, "norm", 1, 1, dim, 0) < 0;
    e |= want(f, path, "norm_b", ln, 1, dim, 0) < 0;
    if (e) return -1;
    char b[80];
#define BLK(s) (snprintf(b, sizeof b, "blk.%u.%s", l, s), b)
    for (uint32_t l = 0; l < h->n_layers; l++) {
        e |= want(f, path, BLK("attn_norm"), 1, 1, dim, 0) < 0;
        e |= want(f, path, BLK("attn_norm_b"), ln, 1, dim, 0) < 0;
        e |= want(f, path, BLK("ffn_norm"), 1, 1, dim, 0) < 0;
        e |= want(f, path, BLK("ffn_norm_b"), ln, 1, dim, 0) < 0;
        e |= want(f, path, BLK("wq"), 1, 0, dim, dim) < 0;
        e |= want(f, path, BLK("wk"), 1, 0, kv, dim) < 0;
        e |= want(f, path, BLK("wv"), 1, 0, kv, dim) < 0;
        e |= want(f, path, BLK("wo"), 1, 0, dim, dim) < 0;
        e |= want(f, path, BLK("w_up"), 1, 0, ffn, dim) < 0;
        e |= want(f, path, BLK("w_gate"), !gelu, 0, ffn, dim) < 0;
        e |= want(f, path, BLK("w_down"), 1, 0, dim, ffn) < 0;
        e |= want(f, path, BLK("bo"), 0, 1, dim, 0) < 0;
        e |= want(f, path, BLK("b_up"), 0, 1, ffn, 0) < 0;
        e |= want(f, path, BLK("b_down"), 0, 1, dim, 0) < 0;
        int bq = want(f, path, BLK("bq"), 0, 1, dim, 0), bk = want(f, path, BLK("bk"), 0, 1, kv, 0), bv = want(f, path, BLK("bv"), 0, 1, kv, 0);
        if (e || bq < 0 || bk < 0 || bv < 0) return -1;
        const l3m_tensor *q = l3m_file_blk(f, (int)l, "wq"), *k = l3m_file_blk(f, (int)l, "wk"), *v = l3m_file_blk(f, (int)l, "wv");
        if (bq != bk || bq != bv || k->dtype != q->dtype || v->dtype != q->dtype) {   // the engine packs q, k, v into one shard
            fprintf(stderr, "%s: blk.%u: wq, wk, wv must share a dtype, and bq, bk, bv come together\n", path, l);
            return -1;
        }
    }
#undef BLK
    return 0;
}

int l3m_file_open(l3m_file *f, const char *path) {
    memset(f, 0, sizeof *f);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return -1; }
    struct stat st;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || (size_t)st.st_size < sizeof(l3m_header)) {
        close(fd);
        return bad(f, path, "not an l3m v%d file", L3M_VERSION);
    }
    f->size = (size_t)st.st_size;
    f->map = mmap(NULL, f->size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (f->map == MAP_FAILED) { perror("mmap"); f->map = NULL; return -1; }

    const l3m_header *h = f->h = (const l3m_header *)f->map;
    if (!l3m_header_ok(h)) return bad(f, path, "not an l3m v%d file", L3M_VERSION);
    if (h->toc_offset % 8 || h->toc_offset > f->size || h->n_tensors > (f->size - h->toc_offset) / sizeof(l3m_tensor) ||
        h->tokenizer_offset > f->size)
        return bad(f, path, "truncated file");
    f->toc = (const l3m_tensor *)((const char *)f->map + h->toc_offset);
    for (uint32_t i = 0; i < h->n_tensors; i++) {
        const char *e = tensor_error(f, &f->toc[i]);
        if (e) return bad(f, path, "tensor %u (%.64s): %s", i, f->toc[i].name, e);
    }
    if (check_model(f, path)) { l3m_file_close(f); return -1; }
    if (l3m_tokenizer_init(&f->tok, (const char *)f->map + h->tokenizer_offset, f->size - h->tokenizer_offset, h->flags))
        return bad(f, path, "bad tokenizer section");
    if (f->tok.n > (int)h->vocab) return bad(f, path, "tokenizer has %d tokens, vocab only %u", f->tok.n, h->vocab);
    return 0;
}

void l3m_file_close(l3m_file *f) {
    l3m_tokenizer_free(&f->tok);
    if (f->map) munmap(f->map, f->size);
    memset(f, 0, sizeof *f);
}

const l3m_tensor *l3m_file_tensor(const l3m_file *f, const char *name) {
    for (uint32_t i = 0; i < f->h->n_tensors; i++)
        if (strcmp(f->toc[i].name, name) == 0) return &f->toc[i];
    return NULL;
}

const l3m_tensor *l3m_file_blk(const l3m_file *f, int layer, const char *name) {
    char buf[64];
    snprintf(buf, sizeof buf, "blk.%d.%s", layer, name);
    return l3m_file_tensor(f, buf);
}

const void *l3m_file_data(const l3m_file *f, const l3m_tensor *t) {
    return (const char *)f->map + t->offset;
}

const float *l3m_file_f32(const l3m_file *f, const l3m_tensor *t) {
    return t ? l3m_file_data(f, t) : NULL;
}
