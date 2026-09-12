// Machine topology from sysfs, per-core cache budgets, thread pinning, huge-page allocation.
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "hw.h"

// Read a small sysfs file into buf; 0 on success.
static int read_str(char *buf, size_t n, const char *fmt, int cpu, int idx) {
    char path[256];
    snprintf(path, sizeof path, fmt, cpu, idx);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (!fgets(buf, (int)n, f)) buf[0] = 0;
    fclose(f);
    buf[strcspn(buf, "\n")] = 0;
    return 0;
}

static long read_long(const char *fmt, int cpu, int idx) {
    char buf[64];
    return read_str(buf, sizeof buf, fmt, cpu, idx) ? -1 : atol(buf);
}

// Sizes are "32K" or "1024K" or "32768K" in sysfs.
static size_t read_size(const char *fmt, int cpu, int idx) {
    char buf[64];
    if (read_str(buf, sizeof buf, fmt, cpu, idx)) return 0;
    size_t v = (size_t)atol(buf);
    char *end = buf; while (*end >= '0' && *end <= '9') end++;
    return *end == 'K' ? v << 10 : *end == 'M' ? v << 20 : v;
}

int l3m_hw_probe(l3m_hw *hw) {
    memset(hw, 0, sizeof *hw);
    memset(hw->core_of, -1, sizeof hw->core_of);
    int l3_id[L3M_MAX_GROUPS];
    long n_cpus = sysconf(_SC_NPROCESSORS_CONF);
    for (int cpu = 0; cpu < n_cpus && cpu < L3M_MAX_CPUS; cpu++) {
        char sib[256];                                   // no topology/ directory: the CPU is offline
        if (read_str(sib, sizeof sib, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu, 0)) continue;
        int first = atoi(sib);
        if (first != cpu) {                              // an SMT sibling of a core we already have
            if (first >= 0 && first < cpu) hw->core_of[cpu] = hw->core_of[first];
            continue;
        }
        if (hw->n_cores == L3M_MAX_CORES) break;
        hw->core_of[cpu] = (short)hw->n_cores;
        l3m_core *c = &hw->core[hw->n_cores];
        c->cpu = cpu; c->group = -1;
        for (int i = 0; i < 8; i++) {                    // cache/indexN: find L2 size and L3 id
            long level = read_long("/sys/devices/system/cpu/cpu%d/cache/index%d/level", cpu, i);
            if (level < 0) break;
            char type[32] = "";
            read_str(type, sizeof type, "/sys/devices/system/cpu/cpu%d/cache/index%d/type", cpu, i);
            if (strcmp(type, "Instruction") == 0) continue;
            if (level == 2) hw->l2_bytes = read_size("/sys/devices/system/cpu/cpu%d/cache/index%d/size", cpu, i);
            if (level == 3) {
                int id = (int)read_long("/sys/devices/system/cpu/cpu%d/cache/index%d/id", cpu, i), g;
                for (g = 0; g < hw->n_groups && l3_id[g] != id; g++) {}
                if (g == hw->n_groups) {
                    if (g == L3M_MAX_GROUPS) return -1;
                    l3_id[g] = id;
                    hw->l3_bytes[g] = read_size("/sys/devices/system/cpu/cpu%d/cache/index%d/size", cpu, i);
                    hw->n_groups++;
                }
                c->group = g;
                hw->group_cores[g]++;
            }
        }
        hw->n_cores++;
    }
    FILE *f = fopen("/proc/cpuinfo", "r");
    char line[256];
    while (f && fgets(line, sizeof line, f))
        if (strncmp(line, "model name", 10) == 0) {
            char *v = strchr(line, ':');
            if (v) { snprintf(hw->name, sizeof hw->name, "%s", v + 2); hw->name[strcspn(hw->name, "\n")] = 0; }
            break;
        }
    if (f) fclose(f);
    if (hw->n_cores == 0) {                              // no sysfs topology at all: one core per online CPU
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        for (int i = 0; i < n && i < L3M_MAX_CORES; i++) { hw->core[i] = (l3m_core){ i, -1 }; hw->core_of[i] = (short)i; hw->n_cores++; }
    }
    if (hw->l2_bytes == 0 || hw->n_groups == 0) {        // no cache info, typically a VM: ask CPUID, one shared L3
        long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE), l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
        hw->l2_bytes = l2 > 0 ? (size_t)l2 : 1u << 20;
        hw->l3_bytes[0] = l3 > 0 ? (size_t)l3 : 32u << 20;
        hw->n_groups = 1;
        hw->group_cores[0] = hw->n_cores;
        for (int i = 0; i < hw->n_cores; i++) hw->core[i].group = 0;
        fprintf(stderr, "l3m: no cache topology in sysfs, assuming %.1f MiB L2 per core and one %.1f MiB L3\n",
                hw->l2_bytes / MiB, hw->l3_bytes[0] / MiB);
    }
    return hw->n_cores > 0 ? 0 : -1;
}

int l3m_hw_core(const l3m_hw *hw, int cpu) {
    return cpu >= 0 && cpu < L3M_MAX_CPUS ? hw->core_of[cpu] : -1;
}

size_t l3m_hw_nominal(const l3m_hw *hw, int cpu) {
    int c = l3m_hw_core(hw, cpu), g = c < 0 ? -1 : hw->core[c].group;
    return hw->l2_bytes + (g < 0 ? 0 : hw->l3_bytes[g] / hw->group_cores[g]);
}

void l3m_hw_print(const l3m_hw *hw, FILE *out) {
    fprintf(out, "%s: %d cores, %d L3 groups, L2 %.1f MiB per core\n", hw->name, hw->n_cores, hw->n_groups, hw->l2_bytes / MiB);
    for (int g = 0; g < hw->n_groups; g++) {
        fprintf(out, "  L3 %d: %5.1f MiB, %2d cores, cpus", g, hw->l3_bytes[g] / MiB, hw->group_cores[g]);
        for (int i = 0; i < hw->n_cores; i++) if (hw->core[i].group == g) fprintf(out, " %d", hw->core[i].cpu);
        double nominal = (double)(hw->l2_bytes + hw->l3_bytes[g] / hw->group_cores[g]);
        fprintf(out, "\n        per core: %.2f MiB nominal, %.2f budget (%.1f MiB per group)\n",
                nominal / MiB, nominal * L3M_RESIDENT_FRACTION / MiB, nominal * L3M_RESIDENT_FRACTION * hw->group_cores[g] / MiB);
    }
}

int l3m_pin(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

#define HUGE_PAGE ((size_t)2 << 20)

static size_t huge_span(size_t bytes) { return (bytes + HUGE_PAGE - 1) / HUGE_PAGE * HUGE_PAGE; }

// The memset is the first touch, so pin before allocating. mlock is best effort.
void *l3m_alloc_huge(size_t bytes) {
    size_t span = huge_span(bytes ? bytes : 1), len = span + HUGE_PAGE;
    char *raw = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return NULL;
    char *p = (char *)(((uintptr_t)raw + HUGE_PAGE - 1) & ~(uintptr_t)(HUGE_PAGE - 1));
    if (p > raw) munmap(raw, (size_t)(p - raw));
    munmap(p + span, (size_t)(raw + len - (p + span)));
    if (!getenv("L3M_NO_HUGEPAGES")) madvise(p, span, MADV_HUGEPAGE);
    memset(p, 0, bytes);
    mlock(p, bytes);
    return p;
}

void l3m_free_huge(void *p, size_t bytes) {
    if (p) munmap(p, huge_span(bytes));
}
