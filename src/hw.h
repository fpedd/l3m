// Machine topology for the engine: cores, L3 groups, per-core cache budgets. Read from sysfs and /proc/cpuinfo.
#pragma once
#include <stddef.h>
#include <stdio.h>

#define L3M_MAX_CORES  256
#define L3M_MAX_GROUPS 32
#define L3M_MAX_CPUS   4096                // logical CPU ids, as l3m_parse_cpus accepts them
#define MiB            (1024.0 * 1024.0)

typedef struct {
    int    cpu;                        // logical CPU of this physical core (first SMT sibling)
    int    group;                      // index into l3 groups
    size_t l2_bytes;                   // this core's share of its L2, which E-core clusters share
} l3m_core;

typedef struct {
    int      n_cores;
    l3m_core core[L3M_MAX_CORES];
    short    core_of[L3M_MAX_CPUS];    // logical CPU -> index into core, -1 if unknown
    int      n_groups;                 // one per distinct L3
    int      group_cores[L3M_MAX_GROUPS];
    size_t   l3_bytes[L3M_MAX_GROUPS]; // per group
    char     name[64];                 // model name from /proc/cpuinfo
} l3m_hw;

// The resident budget as a fraction of the nominal L2 + L3 / cores share.
#define L3M_RESIDENT_FRACTION 0.80

int    l3m_hw_probe(l3m_hw *hw);                                  // 0 on success
int    l3m_hw_core(const l3m_hw *hw, int cpu);                    // index into core, -1 if unknown
size_t l3m_hw_l2(const l3m_hw *hw, int cpu);                      // this core's L2 share, 0 if unknown
size_t l3m_hw_nominal(const l3m_hw *hw, int cpu);                 // L2 share + L3 / cores in that L3
void   l3m_hw_print(const l3m_hw *hw, FILE *out);

int    l3m_pin(int cpu);                                          // pin calling thread to one CPU
void  *l3m_alloc_huge(size_t bytes);                              // 2 MiB aligned, MADV_HUGEPAGE, zeroed, mlocked
void   l3m_free_huge(void *p, size_t bytes);
