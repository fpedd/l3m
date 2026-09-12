// The l3m command line: generate, chat, bench, check, info, hw.
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "l3m.h"
#include "model.h"

#define MiB (1024.0 * 1024.0)

typedef struct {
    const char *model, *prompt, *cores;
    const char *models[64];
    int n_models, md;
    int max_new, ctx, ref, force, verbose, cold;
    float temperature;
    uint64_t seed;
} args;

_Noreturn static void usage(int code) {
    fprintf(code ? stderr : stdout,
        "usage: l3m generate|chat|bench|check|info model.l3m [options]\n"
        "       l3m bench a.l3m b.l3m ... [--md]   (several files, or --md: one markdown row each)\n"
        "       l3m hw\n"
        "  -p \"prompt\"    -n max_new (256)    -t temperature (0.8, 0 = greedy)    -s seed (the time)\n"
        "  --cores 0-7    --ctx N (KV cache length, default min(model, 512))    -v (load details)\n"
        "  --force (ignore the cache budget)    --cold (evict the weights before each token)\n"
        "  --ref (single-thread fp32 reference)\n");
    exit(code);
}

_Noreturn static void bad_arg(const char *what, const char *v) {
    fprintf(stderr, "l3m: bad %s %s\n", what, v);
    exit(2);
}

static long whole(const char *o, const char *v, long lo) {
    char *end;
    long x = strtol(v, &end, 10);
    if (end == v || *end || x < lo || x > INT_MAX) bad_arg(o, v);
    return x;
}

static args parse(int argc, char **argv) {
    args a = { .max_new = 256, .temperature = 0.8f, .seed = (uint64_t)time(NULL) };
    for (int i = 2; i < argc; i++) {
        const char *o = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        char *end;
        if (o[0] != '-') {
            if (a.n_models == 64) bad_arg("model count, over 64:", o);
            a.models[a.n_models++] = o;
        }
        else if (!strcmp(o, "-p") && v) a.prompt = argv[++i];
        else if (!strcmp(o, "-n") && v) a.max_new = (int)whole(o, argv[++i], 0);
        else if (!strcmp(o, "--ctx") && v) a.ctx = (int)whole(o, argv[++i], 2);
        else if (!strcmp(o, "--cores") && v) a.cores = argv[++i];
        else if (!strcmp(o, "-t") && v) {
            a.temperature = strtof(argv[++i], &end);
            if (end == argv[i] || *end || !(a.temperature >= 0) || isinf(a.temperature)) bad_arg(o, argv[i]);
        }
        else if (!strcmp(o, "-s") && v) {
            a.seed = strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end || argv[i][0] == '-') bad_arg(o, argv[i]);
        }
        else if (!strcmp(o, "--ref")) a.ref = 1;
        else if (!strcmp(o, "--force")) a.force = 1;
        else if (!strcmp(o, "--cold")) a.cold = 1;
        else if (!strcmp(o, "--md")) a.md = 1;
        else if (!strcmp(o, "-v")) a.verbose = 1;
        else { fprintf(stderr, "l3m: unknown option %s%s\n", o, v ? "" : " (or missing value)"); usage(2); }
    }
    if (!a.n_models) usage(2);
    a.model = a.models[0];
    if (a.ref && (a.cores || a.cold)) { fprintf(stderr, "l3m: --ref runs on one thread, without --cores or --cold\n"); exit(2); }
    if (a.n_models > 1 && strcmp(argv[1], "bench")) { fprintf(stderr, "l3m: only bench takes several files\n"); exit(2); }
    return a;
}

static l3m_model *load(const args *a, int ref) {
    static int cpus[256];
    l3m_opts o = { .reference = ref, .force = a->force, .verbose = a->verbose, .ctx = a->ctx, .cold = a->cold };
    if (a->cores) {
        o.n_cpus = l3m_parse_cpus(a->cores, cpus, 256);
        if (o.n_cpus <= 0) { fprintf(stderr, "l3m: bad --cores %s\n", a->cores); exit(2); }
        o.cpus = cpus;
    }
    l3m_model *m = l3m_load(a->model, &o);
    if (!m) exit(1);
    return m;
}

typedef struct { l3m_model *m; int32_t *toks; int n; } sink;
static int print_token(int32_t tok, void *user) {
    sink *s = user;
    int len;
    const char *bytes = l3m_decode(s->m, tok, &len);
    fwrite(bytes, 1, len, stdout);
    fflush(stdout);
    return 1;
}
static int collect_token(int32_t tok, void *user) {
    sink *s = user;
    s->toks[s->n++] = tok;
    return 1;
}

static void perf_line(l3m_model *m, FILE *out) {
    l3m_perf p;
    if (l3m_perf_read(m, &p)) return;
    double us = p.ns_token / 1e3;
    fprintf(out, "%s\n%.0f tok/s  %.1f us/tok  (compute %.1f, wait %.1f)",
            l3m_describe(m), 1e9 / p.ns_token, us, p.ns_compute / 1e3, p.ns_wait / 1e3);
    if (p.bytes_token == 0) { fprintf(out, "\n"); return; }   // reference
    fprintf(out, "  %.0f GB/s effective", p.bytes_token / p.ns_token);
    if (!isnan(p.l3_miss)) fprintf(out, "  L3 miss %.0f/tok", p.l3_miss);
    if (!isnan(p.dram_bytes)) fprintf(out, "  DRAM %.2f MB/tok", p.dram_bytes / 1e6);
    if (!isnan(p.energy_j)) fprintf(out, "  %.2f mJ/tok", p.energy_j * 1e3);
    if (!isnan(p.llc_occupancy)) fprintf(out, "  LLC %.1f MiB", p.llc_occupancy / MiB);
    if (isnan(p.dram_bytes)) fprintf(out, "  (DRAM, L3 and energy counters: run as root)");
    fprintf(out, "\n");
    fprintf(out, "weights: %.1f MiB per token at %.0f GB/s", p.bytes_token / MiB, p.bytes_token / p.ns_token);
    if (!isnan(p.dram_bytes)) fprintf(out, ", DRAM reads %.1f%% of that", 100 * p.dram_bytes / p.bytes_token);
    if (!isnan(p.llc_occupancy))
        fprintf(out, ", LLC occupancy %.1f MiB plus up to %.0f MiB in L2", p.llc_occupancy / MiB, p.l2_bytes / MiB);
    fprintf(out, "\n");
}

// Token count, or -1 with a message if the prompt does not fit ids and the KV cache.
static int encode(l3m_model *m, const char *text, int32_t *ids, int max) {
    int n = l3m_encode(m, text ? text : "", 1, ids, max), fit = l3m_ctx(m) < max ? l3m_ctx(m) : max;
    if (n >= 1 && n < fit) return n;
    if (n == max) fprintf(stderr, "l3m: the prompt is over %d tokens\n", max - 1);
    else fprintf(stderr, "l3m: the prompt is %d tokens, the limit is %d (--ctx %d)\n", n, fit - 1, l3m_ctx(m));
    return -1;
}

static int cmd_generate(const args *a) {
    l3m_model *m = load(a, a->ref);
    int32_t ids[4096];
    int n = encode(m, a->prompt, ids, 4096);
    if (n < 0) { l3m_free(m); return 1; }
    if (a->prompt) fputs(a->prompt, stdout);
    sink s = { m, NULL, 0 };
    int made = l3m_generate(m, ids, n, a->max_new, a->temperature, a->seed, print_token, &s);
    printf("\n");
    fprintf(stderr, "\n[%d tokens] ", made);
    perf_line(m, stderr);
    l3m_free(m);
    return 0;
}

// The whole conversation is re-fed each turn as plain text, no chat template.
static int cmd_chat(const args *a) {
    l3m_model *m = load(a, a->ref);
    char history[1 << 16] = "", line[4096];
    int turn = 0;
    int32_t ids[4096];
    if (a->prompt) snprintf(history, sizeof history, "%s\n", a->prompt);
    for (;;) {
        fputs("\n> ", stdout); fflush(stdout);
        if (!fgets(line, sizeof line, stdin)) break;
        if (strlen(history) + strlen(line) + 1 >= sizeof history) { fprintf(stderr, "l3m: history full\n"); break; }
        strcat(history, line);
        int n = encode(m, history, ids, 4096);
        if (n < 0) { fprintf(stderr, "l3m: the conversation fills the context, ending\n"); break; }
        int32_t out[4096];
        sink s = { m, out, 0 };
        int made = l3m_generate(m, ids, n, a->max_new < 4096 ? a->max_new : 4096, a->temperature, a->seed + (uint64_t)turn++, collect_token, &s);
        for (int i = 0; i < made; i++) {
            int len;
            const char *b = l3m_decode(m, out[i], &len);
            fwrite(b, 1, len, stdout);
            if (strlen(history) + len + 1 < sizeof history) strncat(history, b, len);
        }
        fflush(stdout);
    }
    l3m_free(m);
    return 0;
}

static int cmd_bench(const args *a) {
    if (a->max_new < 1) { fprintf(stderr, "l3m: bench needs -n 1 or more\n"); return 2; }
    int md = a->md || a->n_models > 1;
    if (md) {
        printf("| model | tok/s | us/tok | compute | wait | GB/s | DRAM %% | LLC MiB | mJ/tok |\n");
        printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    }
    for (int i = 0; i < a->n_models; i++) {
        args one = *a; one.model = a->models[i];
        l3m_model *m = load(&one, a->ref);
        int32_t bos = l3m_bos(m) < 0 ? 0 : l3m_bos(m), out[8192];
        sink s = { m, out, 0 };
        l3m_perf p;
        l3m_generate(m, &bos, 1, 16, 0.f, 0, collect_token, &s);   // warm up, then reset counters
        l3m_perf_read(m, &p);
        s.n = 0;
        int n = a->max_new < 8192 ? a->max_new : 8192;
        while (s.n < n && l3m_generate(m, &bos, 1, n - s.n, 0.f, 0, collect_token, &s)) {}   // again after an end of text
        if (!md) perf_line(m, stdout);
        else if (l3m_perf_read(m, &p)) printf("| %s | no tokens |\n", l3m_describe(m));
        else {
            printf("| %s | %.0f | %.1f | %.1f | %.1f | %.0f | %.1f | %.1f | %.2f |\n", l3m_describe(m),
                   1e9 / p.ns_token, p.ns_token / 1e3, p.ns_compute / 1e3, p.ns_wait / 1e3, p.bytes_token / p.ns_token,
                   100 * p.dram_bytes / p.bytes_token, p.llc_occupancy / MiB, p.energy_j * 1e3);
        }
        l3m_free(m);
    }
    return 0;
}

// KL(reference || fast) of the next-token distributions, in nats.
static double kl(const float *lr, const float *lf, int n) {
    double mr = lr[0], mf = lf[0], zr = 0, zf = 0, sum = 0;
    for (int i = 1; i < n; i++) { if (lr[i] > mr) mr = lr[i]; if (lf[i] > mf) mf = lf[i]; }
    for (int i = 0; i < n; i++) { zr += exp(lr[i] - mr); zf += exp(lf[i] - mf); }
    for (int i = 0; i < n; i++) sum += exp(lr[i] - mr) / zr * ((lr[i] - mr - log(zr)) - (lf[i] - mf - log(zf)));
    return sum;
}

// Fast path vs reference: judged by KL < 0.01 nats and argmax; greedy tokens are informational.
static int cmd_check(const args *a) {
    l3m_model *ref = load(a, 1), *fast = load(a, 0);
    int32_t ids[4096], tr[64], tf[64];
    const char *prompt = a->prompt ? a->prompt : "Once upon a time";
    int n = encode(ref, prompt, ids, 4096), vocab = l3m_vocab(ref);
    if (n < 0) { l3m_free(ref); l3m_free(fast); return 1; }

    l3m_generate(ref, ids, n, 0, 0.f, 0, NULL, NULL);
    l3m_generate(fast, ids, n, 0, 0.f, 0, NULL, NULL);
    const float *lr = l3m_logits(ref), *lf = l3m_logits(fast);
    double maxerr = 0, d = kl(lr, lf, vocab); int ar = 0, af = 0;
    for (int i = 0; i < vocab; i++) {
        if (fabs(lr[i] - lf[i]) > maxerr) maxerr = fabs(lr[i] - lf[i]);
        if (lr[i] > lr[ar]) ar = i;
        if (lf[i] > lf[af]) af = i;
    }
    int ok = ar == af && d < 0.01;
    printf("logits after prompt: max-abs error %.3g, KL %.2g nats, argmax %s (%d vs %d)\n",
           maxerr, d, ar == af ? "agree" : "DIFFER", ar, af);

    sink sr = { ref, tr, 0 }, sf = { fast, tf, 0 };
    int nr = l3m_generate(ref, ids, n, 32, 0.f, 0, collect_token, &sr);
    int nf = l3m_generate(fast, ids, n, 32, 0.f, 0, collect_token, &sf);
    int same = 0;
    while (same < nr && same < nf && tr[same] == tf[same]) same++;
    printf("greedy tokens: %d of %d agree\n", same, nr);
    printf("%s\n", ok ? "PASS" : "FAIL");
    l3m_free(ref); l3m_free(fast);
    return !ok;
}

static int cmd_info(const args *a) {
    FILE *f = fopen(a->model, "rb");
    if (!f) { perror(a->model); return 1; }
    l3m_header h;
    if (fread(&h, sizeof h, 1, f) != 1 || !l3m_header_ok(&h)) { fprintf(stderr, "l3m: %s: not an l3m file\n", a->model); fclose(f); return 1; }
    printf("%s  arch %.16s  flags 0x%x\n", a->model, h.arch, h.flags);
    printf("dim %u  layers %u  heads %u  kv_heads %u  ffn %u  vocab %u  ctx %u  rope_theta %g  eps %g\n",
           h.dim, h.n_layers, h.n_heads, h.n_kv_heads, h.ffn_dim, h.vocab, h.ctx, h.rope_theta, h.norm_eps);
    int ok = h.toc_offset <= (uint64_t)LONG_MAX && !fseek(f, (long)h.toc_offset, SEEK_SET);
    double total = 0;
    for (uint32_t i = 0; ok && i < h.n_tensors; i++) {
        l3m_tensor t;
        if (!(ok = fread(&t, sizeof t, 1, f) == 1)) break;
        char shape[48];
        snprintf(shape, sizeof shape, t.ndim > 1 ? "%llu x %llu" : "%llu",
                 (unsigned long long)t.shape[0], (unsigned long long)t.shape[1]);
        printf("  %-24.64s %-6s %-14s %10llu bytes\n", t.name, l3m_dtype_name((int)t.dtype), shape, (unsigned long long)t.nbytes);
        total += (double)t.nbytes;
    }
    fclose(f);
    if (!ok) { fprintf(stderr, "l3m: %s: truncated toc\n", a->model); return 1; }
    printf("%u tensors, %.1f MiB\n", h.n_tensors, total / MiB);
    return 0;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) usage(0);
    if (argc >= 2 && !strcmp(argv[1], "hw")) { l3m_print_hw(stdout); return 0; }
    args a = parse(argc, argv);
    const char *cmd = argv[1];
    if (!strcmp(cmd, "generate")) return cmd_generate(&a);
    if (!strcmp(cmd, "chat")) return cmd_chat(&a);
    if (!strcmp(cmd, "bench")) return cmd_bench(&a);
    if (!strcmp(cmd, "check")) return cmd_check(&a);
    if (!strcmp(cmd, "info")) return cmd_info(&a);
    fprintf(stderr, "l3m: unknown command %s\n", cmd);
    usage(2);
}
