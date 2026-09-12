// Pinned SPMD pool and the flag exchange. Rank 0 is the caller; ranks 1..n-1 are persistent threads.
// Between jobs workers spin for about a millisecond, then poll with a short sleep; inside a job nothing sleeps.
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>
#include "pool.h"
#include "hw.h"
#include "perf.h"

// A cache line pair per rank: the adjacent-line prefetcher defeats 64-byte padding.
typedef struct { _Alignas(128) _Atomic unsigned v; char pad[128 - sizeof(_Atomic unsigned)]; } line;

typedef struct { l3m_pool *pool; int rank, cpu; } worker_arg;

struct l3m_pool {
    int        n;
    pthread_t  thread[L3M_MAX_CORES];
    worker_arg arg_of[L3M_MAX_CORES];
    line      *flag;                 // per rank: last sync step published
    line      *done;                 // per rank: last job generation finished
    line      *release;              // one line: the last step rank 0 released
    double    *wait_ticks;           // per rank at [rank * 16], TSC ticks
    _Atomic unsigned job_gen;        // bumped by rank 0 to start a job
    l3m_job    job;
    void      *arg;
};

static void run_job(l3m_pool *p, int rank, unsigned gen) {
    p->job(p, rank, p->arg);
    atomic_store_explicit(&p->done[rank].v, gen, memory_order_release);
}

static void *worker(void *v) {
    worker_arg *a = v;
    l3m_pool *p = a->pool;
    l3m_pin(a->cpu);
    unsigned seen = 0;
    for (;;) {
        double t0 = l3m_now_ns();
        while (atomic_load_explicit(&p->job_gen, memory_order_acquire) == seen) {
            if (l3m_now_ns() - t0 < 1e6) _mm_pause();
            else nanosleep(&(struct timespec){0, 50000}, NULL);
        }
        seen = atomic_load_explicit(&p->job_gen, memory_order_acquire);
        if (!p->job) return NULL;                       // stop request
        run_job(p, a->rank, seen);
    }
}

// Each cpu must exist, be ours to run on, and appear once: pin to each in turn, then restore the caller.
l3m_pool *l3m_pool_start(const int *cpus, int n) {
    cpu_set_t saved;
    pthread_getaffinity_np(pthread_self(), sizeof saved, &saved);
    int ok = 1;
    for (int i = 0; i < n && ok; i++) {
        ok = !l3m_pin(cpus[i]);
        for (int j = 0; j < i; j++) ok &= cpus[j] != cpus[i];
    }
    pthread_setaffinity_np(pthread_self(), sizeof saved, &saved);
    if (!ok) return NULL;
    l3m_pool *p = calloc(1, sizeof *p);
    p->n = n;
    p->flag = aligned_alloc(128, n * sizeof(line));
    p->done = aligned_alloc(128, n * sizeof(line));
    p->release = aligned_alloc(128, sizeof(line));
    p->wait_ticks = aligned_alloc(128, n * 16 * sizeof(double));
    atomic_init(&p->release->v, 0);
    for (int i = 0; i < n; i++) { atomic_init(&p->flag[i].v, 0); atomic_init(&p->done[i].v, 0); p->wait_ticks[i * 16] = 0; }
    l3m_now_ns();                                       // calibrate the clock before anyone spins
    for (int i = 0; i < n; i++) p->arg_of[i] = (worker_arg){p, i, cpus[i]};
    for (int i = 1; i < n; i++)
        if (pthread_create(&p->thread[i], NULL, worker, &p->arg_of[i])) { p->n = i; l3m_pool_stop(p); return NULL; }
    return p;
}

// The caller is pinned to cpus[0] for the job, then given its own affinity back.
void l3m_pool_run(l3m_pool *p, l3m_job job, void *arg) {
    cpu_set_t saved;
    pthread_getaffinity_np(pthread_self(), sizeof saved, &saved);
    int move = CPU_COUNT(&saved) != 1 || !CPU_ISSET(p->arg_of[0].cpu, &saved);
    if (move) l3m_pin(p->arg_of[0].cpu);
    p->job = job; p->arg = arg;
    unsigned gen = atomic_load_explicit(&p->job_gen, memory_order_relaxed) + 1;
    atomic_store_explicit(&p->job_gen, gen, memory_order_release);
    run_job(p, 0, gen);
    for (int i = 1; i < p->n; i++)
        while (atomic_load_explicit(&p->done[i].v, memory_order_acquire) != gen) _mm_pause();
    if (move) pthread_setaffinity_np(pthread_self(), sizeof saved, &saved);
}

void l3m_pool_stop(l3m_pool *p) {
    p->job = NULL;
    atomic_fetch_add_explicit(&p->job_gen, 1, memory_order_release);
    for (int i = 1; i < p->n; i++) pthread_join(p->thread[i], NULL);
    free(p->flag); free(p->done); free(p->release); free(p->wait_ticks); free(p);
}

static inline int before(unsigned a, unsigned b) { return (int)(a - b) < 0; }

// Publish my step; rank 0 polls every flag, then releases the others through one line.
void l3m_sync(l3m_pool *p, int rank) {
    unsigned step = atomic_load_explicit(&p->flag[rank].v, memory_order_relaxed) + 1;
    atomic_store_explicit(&p->flag[rank].v, step, memory_order_release);
    uint64_t t0 = __rdtsc();
    if (rank == 0) {
        for (int i = 1; i < p->n; i++)
            while (before(atomic_load_explicit(&p->flag[i].v, memory_order_acquire), step)) _mm_pause();
        atomic_store_explicit(&p->release->v, step, memory_order_release);
    } else {
        while (before(atomic_load_explicit(&p->release->v, memory_order_acquire), step)) _mm_pause();
    }
    p->wait_ticks[rank * 16] += (double)(__rdtsc() - t0);
}

double l3m_sync_wait_ns(const l3m_pool *p, int rank) { return p->wait_ticks[rank * 16] * l3m_ns_per_tick(); }
