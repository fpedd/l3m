// ns per barrier and per all-gather over a set of cores.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hw.h"
#include "l3m.h"
#include "perf.h"
#include "pool.h"

enum { CHUNK = 1000 };                // iterations per timed sample

typedef struct {
    int     n, iters, d;              // d = 0: barrier only
    float  *buf;                      // two shared vectors, by iteration parity
    double  ns[L3M_MAX_CORES];        // per iteration
    float   sink[L3M_MAX_CORES * 32];
} job_t;

static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static void xchg_job(l3m_pool *p, int rank, void *v) {
    job_t *j = v;
    int n = j->n, lo = j->d * rank / n, hi = j->d * (rank + 1) / n;
    int chunks = j->iters / CHUNK;
    double *t = malloc(chunks * sizeof *t);
    float *mine = calloc(j->d + 1, sizeof *mine);
    float acc = 0;
    for (int c = 0; c < chunks; c++) {
        double t0 = l3m_now_ns();
        for (int it = 0; it < CHUNK; it++) {
            float *buf = j->buf + (it & 1) * (j->d + 16);   // never rewrite what a slow rank may still read
            for (int i = lo; i < hi; i++) buf[i] = (float)(it + rank);
            l3m_sync(p, rank);
            memcpy(mine, buf, j->d * sizeof *mine);
            acc += mine[it % (j->d + 1)];
        }
        t[c] = (l3m_now_ns() - t0) / CHUNK;
    }
    qsort(t, chunks, sizeof *t, cmp);
    j->ns[rank] = t[chunks / 2];
    j->sink[rank * 32] = acc;
    free(t); free(mine);
}

int main(int argc, char **argv) {
    const char *cpulist = NULL;                     // default: the first 8 physical cores
    int iters = 100000;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cpus") && i + 1 < argc) cpulist = argv[++i];
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--cpus 0-7] [--iters 100000]\n", argv[0]); return 1; }
    }
    int cpus[L3M_MAX_CORES], n = cpulist ? l3m_parse_cpus(cpulist, cpus, L3M_MAX_CORES) : 0;
    l3m_hw hw;
    if (!cpulist && !l3m_hw_probe(&hw)) for (; n < hw.n_cores && n < 8; n++) cpus[n] = hw.core[n].cpu;
    if (n < 1) { fprintf(stderr, "bad cpu list\n"); return 1; }
    if (iters < CHUNK) iters = CHUNK;
    l3m_pool *pool = l3m_pool_start(cpus, n);
    if (!pool) { fprintf(stderr, "bad cpu list\n"); return 1; }
    printf("%d cores (cpus", n); for (int i = 0; i < n; i++) printf(" %d", cpus[i]);
    printf("), median over %d iterations\n%-22s %11s %11s\n", iters, "exchange", "min rank", "max rank");
    int dims[] = {0, 768, 3072};
    for (int k = 0; k < 3; k++) {
        job_t *j = calloc(1, sizeof *j);
        j->n = n; j->iters = iters; j->d = dims[k];
        j->buf = aligned_alloc(64, 2 * (dims[k] + 16) * sizeof(float));
        l3m_pool_run(pool, xchg_job, j);
        double lo = 1e30, hi = 0;
        for (int r = 0; r < n; r++) { lo = fmin(lo, j->ns[r]); hi = fmax(hi, j->ns[r]); }
        char name[32];
        if (dims[k]) snprintf(name, sizeof name, "all-gather d=%d", dims[k]); else snprintf(name, sizeof name, "barrier");
        printf("%-22s %8.0f ns %8.0f ns\n", name, lo, hi);
        free(j->buf); free(j);
    }
    l3m_pool_stop(pool);
    return 0;
}
