// Residency sweep: every core streams its own shard at each size, with cache and DRAM counters.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "hw.h"
#include "l3m.h"
#include "perf.h"
#include "pool.h"

typedef struct {
    size_t   bytes;
    double   seconds;
    uint8_t *shard[L3M_MAX_CORES];
    double   gbs[L3M_MAX_CORES];
    uint64_t sink[L3M_MAX_CORES * 16];    // padded
} job_t;

static void alloc_job(l3m_pool *p, int rank, void *v) {
    job_t *j = v;
    (void)p;
    j->shard[rank] = l3m_alloc_huge(j->bytes);           // first touch on the owning core
    for (size_t i = 0; i < j->bytes; i++) j->shard[rank][i] = (uint8_t)(i * 2654435761u >> 24);
}

typedef uint64_t v4u64 __attribute__((vector_size(32)));

__attribute__((target("avx2")))
static void stream_job(l3m_pool *p, int rank, void *v) {
    job_t *j = v;
    (void)p;
    const v4u64 *w = (const v4u64 *)j->shard[rank];
    size_t n = j->bytes / sizeof(v4u64);
    v4u64 a0 = {0}, a1 = {0}, a2 = {0}, a3 = {0};
    double t0 = l3m_now_ns(), t;
    long passes = 0;
    do {
        for (size_t i = 0; i + 4 <= n; i += 4) { a0 += w[i]; a1 += w[i + 1]; a2 += w[i + 2]; a3 += w[i + 3]; }
        passes++;
        t = l3m_now_ns();
    } while (t - t0 < j->seconds * 1e9);
    v4u64 s = a0 + a1 + a2 + a3;
    j->sink[rank * 16] = s[0] + s[1] + s[2] + s[3];
    j->gbs[rank] = (double)passes * (double)j->bytes / (t - t0);
}

int main(int argc, char **argv) {
    const char *cpulist = NULL, *mibs = "2,3,4,4.25,5";
    double seconds = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cpus") && i + 1 < argc) cpulist = argv[++i];
        else if (!strcmp(argv[i], "--mib") && i + 1 < argc) mibs = argv[++i];
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--pages") && i + 1 < argc && !strcmp(argv[i + 1], "4k")) { setenv("L3M_NO_HUGEPAGES", "1", 1); i++; }
        else { fprintf(stderr, "usage: %s [--cpus 0-15] [--mib 2,3,4,4.25,5] [--pages 4k] [--seconds 1]\n", argv[0]); return 1; }
    }
    l3m_hw hw;
    if (l3m_hw_probe(&hw)) { fprintf(stderr, "cannot read topology\n"); return 1; }
    int cpus[L3M_MAX_CORES], n;
    if (cpulist) n = l3m_parse_cpus(cpulist, cpus, L3M_MAX_CORES);
    else { n = hw.n_cores; for (int i = 0; i < n; i++) cpus[i] = hw.core[i].cpu; }
    if (n < 1) { fprintf(stderr, "bad cpu list\n"); return 1; }
    l3m_hw_print(&hw, stdout);
    l3m_hwcounters *hc = l3m_hwcounters_open(0);
    int missing = !hc;
    l3m_pool *pool = l3m_pool_start(cpus, n);
    if (!pool) { fprintf(stderr, "bad cpu list\n"); return 1; }

    printf("\n%8s %8s %10s %10s %10s %10s %10s %10s\n", "MiB/core", "total", "min GB/s", "max GB/s", "aggregate", "L3miss/s", "DRAM GB/s", "LLC MiB");
    char *list = strdup(mibs);
    for (char *tok = strtok(list, ","); tok; tok = strtok(NULL, ",")) {
        job_t *j = calloc(1, sizeof *j);
        j->bytes = (size_t)(atof(tok) * 1048576.0) & ~(size_t)127;   // whole 128-byte steps of stream_job
        j->seconds = seconds;
        l3m_pool_run(pool, alloc_job, j);
        l3m_pool_run(pool, stream_job, j);               // warm-up
        l3m_counters c = {0};
        if (hc) l3m_hwcounters_read(hc, &c);
        l3m_pool_run(pool, stream_job, j);
        if (hc) l3m_hwcounters_read(hc, &c);
        missing |= !hc || isnan(c.l3_miss) || isnan(c.llc_occupancy);
        double lo = 1e30, hi = 0, agg = 0;
        for (int r = 0; r < n; r++) { lo = fmin(lo, j->gbs[r]); hi = fmax(hi, j->gbs[r]); agg += j->gbs[r]; }
        printf("%8s %8.1f %10.1f %10.1f %10.1f %10.3g %10.1f %10.1f\n", tok, (double)j->bytes * n / 1048576.0, lo, hi, agg,
               hc ? c.l3_miss / (c.ns * 1e-9) : NAN, hc ? c.dram_bytes / c.ns : NAN, hc ? c.llc_occupancy / 1048576.0 : NAN);
        for (int r = 0; r < n; r++) l3m_free_huge(j->shard[r], j->bytes);
        free(j);
    }
    free(list);
    if (missing) printf(geteuid() ? "(nan: counters need root for perf events, and a mounted resctrl for occupancy)\n"
                                  : "(nan: no amd_l3/amd_umc PMU, or no mounted resctrl for occupancy)\n");
    l3m_pool_stop(pool);
    l3m_hwcounters_close(hc);
    return 0;
}
