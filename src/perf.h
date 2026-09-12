// Timers from rdtsc; hardware counters from perf events on the uncore PMUs and RAPL, and from resctrl.
#pragma once
#include <stdint.h>

double   l3m_ns_per_tick(void);         // the TSC period, calibrated once
double   l3m_now_ns(void);              // rdtsc converted with it

// Counter deltas, NAN where unavailable. l3_miss, dram_bytes and energy_j are machine-wide and need
// root for perf_event_open; llc_occupancy is this process's own as root, the whole machine's otherwise.
typedef struct {
    double l3_miss;          // amd_l3 lookups that missed, summed over L3s
    double dram_bytes;       // amd_umc CAS reads * 64, summed over channels
    double energy_j;         // package energy via RAPL
    double llc_occupancy;    // bytes, from resctrl
    double ns;               // wall time of the sample window
} l3m_counters;

typedef struct l3m_hwcounters l3m_hwcounters;
l3m_hwcounters *l3m_hwcounters_open(int verbose);       // NULL if nothing is available; verbose reports why
void            l3m_hwcounters_read(l3m_hwcounters *c, l3m_counters *out);  // deltas since last read
void            l3m_hwcounters_close(l3m_hwcounters *c);
