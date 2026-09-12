// Greedy highest-score pair merging: llama2.c for SentencePiece, score = -rank for byte-level BPE.
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"

// Bytes, then length, then higher score first so lookup finds the best duplicate.
static int cmp_tok(const void *a, const void *b, void *arg) {
    const l3m_tokenizer *t = arg;
    int x = *(const int *)a, y = *(const int *)b;
    int n = t->len[x] < t->len[y] ? t->len[x] : t->len[y];
    int c = memcmp(t->bytes[x], t->bytes[y], n);
    if (c) return c;
    if (t->len[x] != t->len[y]) return t->len[x] - t->len[y];
    return t->score[x] < t->score[y] ? 1 : t->score[x] > t->score[y] ? -1 : 0;
}

int l3m_tokenizer_init(l3m_tokenizer *t, const void *section, size_t size, unsigned flags) {
    l3m_tok_header h;
    memset(t, 0, sizeof *t);
    if (size < sizeof h) return -1;
    memcpy(&h, section, sizeof h);                      // unaligned
    if (h.n_tokens > (size - sizeof h) / 6 || h.bos >= (int64_t)h.n_tokens) return -1;
    t->n = (int)h.n_tokens; t->bos = h.bos < 0 ? -1 : h.bos; t->eos = h.eos; t->max_len = (int)h.max_len;
    t->sp_prefix = (flags & L3M_F_SP_PREFIX) != 0;
    t->digits = (flags & L3M_F_DIGITS) != 0;
    const uint8_t **bytes = malloc((t->n + 1) * sizeof *bytes);
    uint16_t *len = malloc((t->n + 1) * sizeof *len);
    float *score = malloc((t->n + 1) * sizeof *score);
    t->bytes = bytes; t->len = len; t->score = score;
    t->sorted = malloc((t->n + 1) * sizeof *t->sorted);
    if (!bytes || !len || !score || !t->sorted) { l3m_tokenizer_free(t); return -1; }
    const uint8_t *p = (const uint8_t *)section + sizeof h, *end = (const uint8_t *)section + size;
    for (int i = 0; i < t->n; i++) {
        if (end - p < 6) { l3m_tokenizer_free(t); return -1; }
        memcpy(&score[i], p, 4); memcpy(&len[i], p + 4, 2);
        if (end - p - 6 < len[i]) { l3m_tokenizer_free(t); return -1; }
        bytes[i] = p + 6; p += 6 + len[i];
        t->sorted[i] = i;
    }
    t->sp_words = t->sp_prefix;                         // unless a token has a space after a non-space
    for (int i = 0; i < t->n && t->sp_words; i++)
        for (int j = 1; j < len[i]; j++) if (bytes[i][j] == ' ' && bytes[i][j - 1] != ' ') t->sp_words = 0;
    qsort_r(t->sorted, t->n, sizeof *t->sorted, cmp_tok, t);
    return 0;
}

void l3m_tokenizer_free(l3m_tokenizer *t) {
    free((void *)t->bytes); free((void *)t->len); free((void *)t->score); free(t->sorted);
    memset(t, 0, sizeof *t);
}

// Token id for exactly these bytes, or -1.
static int lookup(const l3m_tokenizer *t, const uint8_t *s, int n) {
    int lo = 0, hi = t->n;
    while (lo < hi) {
        int mid = (lo + hi) / 2, id = t->sorted[mid];
        int m = t->len[id] < n ? t->len[id] : n;
        int c = memcmp(t->bytes[id], s, m);
        if (c == 0) c = t->len[id] - n;
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    if (lo == t->n) return -1;
    int id = t->sorted[lo];
    return t->len[id] == n && memcmp(t->bytes[id], s, n) == 0 ? id : -1;
}

static int utf8_len(const uint8_t *p) {
    int n = p[0] < 0x80 ? 1 : p[0] < 0xE0 ? 2 : p[0] < 0xF0 ? 3 : 4;
    for (int i = 1; i < n; i++) if ((p[i] & 0xC0) != 0x80) return 1;
    return n;
}

// Candidate merges in a max-heap (leftmost on ties), validated when popped.
typedef struct { float score; int left, right, a, b, id; } pair;

static int before(const pair *x, const pair *y) { return x->score > y->score || (x->score == y->score && x->left < y->left); }

static void push(const l3m_tokenizer *t, pair *h, int *nh, const int32_t *s, int left, int right) {
    uint8_t buf[512];
    int a = s[left], b = s[right], l = t->len[a] + t->len[b];
    if (l > t->max_len || l > (int)sizeof buf) return;
    memcpy(buf, t->bytes[a], t->len[a]); memcpy(buf + t->len[a], t->bytes[b], t->len[b]);
    int id = lookup(t, buf, l);
    if (id < 0) return;
    pair c = { t->score[id], left, right, a, b, id };
    int i = (*nh)++;
    for (; i && before(&c, &h[(i - 1) / 2]); i = (i - 1) / 2) h[i] = h[(i - 1) / 2];
    h[i] = c;
}

static pair pop(pair *h, int *nh) {
    pair top = h[0], last = h[--*nh];
    int i = 0;
    for (int c; (c = 2 * i + 1) < *nh; i = c) {
        if (c + 1 < *nh && before(&h[c + 1], &h[c])) c++;
        if (!before(&h[c], &last)) break;
        h[i] = h[c];
    }
    h[i] = last;
    return top;
}

// Merge within out[lo, n); returns the new n. A merged-away symbol is -1.
static int merge(const l3m_tokenizer *t, int32_t *out, int lo, int n) {
    int m = n - lo, nh = 0, k = 0;
    if (m < 2) return n;
    int32_t *s = out + lo;
    int *next = malloc(m * sizeof *next), *prev = malloc(m * sizeof *prev);
    pair *h = malloc(3 * (size_t)m * sizeof *h);                  // m - 1 pairs, at most two per merge
    if (!next || !prev || !h) { free(next); free(prev); free(h); return n; }
    for (int i = 0; i < m; i++) { next[i] = i + 1; prev[i] = i - 1; }
    for (int i = 0; i + 1 < m; i++) push(t, h, &nh, s, i, i + 1);
    while (nh) {
        pair c = pop(h, &nh);
        if (next[c.left] != c.right || s[c.left] != c.a || s[c.right] != c.b) continue;   // stale
        s[c.left] = c.id; s[c.right] = -1;
        next[c.left] = next[c.right];
        if (next[c.left] < m) prev[next[c.left]] = c.left;
        if (prev[c.left] >= 0) push(t, h, &nh, s, prev[c.left], c.left);
        if (next[c.left] < m) push(t, h, &nh, s, c.left, next[c.left]);
    }
    for (int i = 0; i < m; i++) if (s[i] >= 0) s[k++] = s[i];
    free(next); free(prev); free(h);
    return lo + k;
}

enum { SPACE, LETTER, DIGIT, PUNCT };
static int cls(uint8_t c) {   // non-ASCII counts as a letter
    if (c == ' ' || (c >= '\t' && c <= '\r')) return SPACE;
    if (c >= '0' && c <= '9') return DIGIT;
    return c >= 0x80 || ((c | 32) >= 'a' && (c | 32) <= 'z') ? LETTER : PUNCT;
}

// End of the pre-tokenizer piece at p. Byte-level BPE: GPT-2's regex
// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+, digits split if t->digits.
// SentencePiece: a word with its leading spaces if sp_words, else the whole text.
static const uint8_t *piece_end(const l3m_tokenizer *t, const uint8_t *p) {
    static const char *contraction[] = { "s", "t", "re", "ve", "m", "ll", "d" };
    if (t->sp_prefix) {
        while (*p == ' ') p++;
        while (*p && (*p != ' ' || !t->sp_words)) p++;
        return p;
    }
    if (*p == '\'')
        for (int i = 0; i < 7; i++)
            if (!strncmp((const char *)p + 1, contraction[i], strlen(contraction[i]))) return p + 1 + strlen(contraction[i]);
    if (t->digits && cls(*p) == DIGIT) return p + 1;
    const uint8_t *q = p + (*p == ' ' && p[1] && cls(p[1]) != SPACE && !(t->digits && cls(p[1]) == DIGIT));
    int k = cls(*q);
    if (k != SPACE) { while (*q && cls(*q) == k) q++; return q; }
    while (*q && cls(*q) == SPACE) q++;
    int end = !*q || (t->digits && cls(*q) == DIGIT);
    return q - p > 1 && !end ? q - 1 : q;   // the last space goes to the next word
}

int l3m_tokenizer_encode(const l3m_tokenizer *t, const char *text, int add_bos, int32_t *out, int max) {
    int n = 0;
    if (add_bos && t->bos >= 0 && n < max) out[n++] = t->bos;
    int lo = n, space = lookup(t, (const uint8_t *)" ", 1);
    if (t->sp_prefix && *text && space >= 0 && n < max) out[n++] = space;
    for (const uint8_t *p = (const uint8_t *)text; *p; lo = n) {
        for (const uint8_t *end = piece_end(t, p); p < end; ) {
            int len = t->sp_prefix ? utf8_len(p) : 1, id = lookup(t, p, len);
            if (id >= 0) { if (n == max) return max; out[n++] = id; }
            else for (int i = 0; i < len; i++) if ((id = lookup(t, p + i, 1)) >= 0) { if (n == max) return max; out[n++] = id; }
            p += len;
        }
        n = merge(t, out, lo, n);
    }
    return n;
}

const uint8_t *l3m_tokenizer_decode(const l3m_tokenizer *t, int32_t tok, int *len) {
    if (tok < 0 || tok >= t->n) { *len = 0; return (const uint8_t *)""; }
    *len = t->len[tok];
    return t->bytes[tok];
}
