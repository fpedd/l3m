// Pinned SPMD thread pool; rank 0 is the calling thread and every rank runs the same job.
// Shared-memory writes before an l3m_sync are visible to every rank after it.
#pragma once

typedef struct l3m_pool l3m_pool;
typedef void (*l3m_job)(l3m_pool *p, int rank, void *arg);

l3m_pool *l3m_pool_start(const int *cpus, int n);       // thread i pinned to cpus[i], n <= L3M_MAX_CORES; NULL if a cpu is unusable or repeated
void      l3m_pool_run(l3m_pool *p, l3m_job job, void *arg);   // caller is rank 0, on cpus[0] until every rank finished the job
void      l3m_pool_stop(l3m_pool *p);

void      l3m_sync(l3m_pool *p, int rank);                      // barrier over every rank
double    l3m_sync_wait_ns(const l3m_pool *p, int rank);        // time this rank spent waiting, cumulative
