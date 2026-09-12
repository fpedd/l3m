// Timers from rdtsc; counters from perf_event_open (amd_l3, amd_umc, RAPL) and resctrl.
#include <dirent.h>
#include <errno.h>
#include <linux/perf_event.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>
#include "perf.h"

static double monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

// Calibrated once over 20 ms; assumes a constant-rate TSC.
double l3m_ns_per_tick(void) {
    static double npt;
    if (npt == 0) {
        double t0 = monotonic_ns(); uint64_t c0 = __rdtsc();
        while (monotonic_ns() - t0 < 20e6) {}
        npt = (monotonic_ns() - t0) / (double)(__rdtsc() - c0);
    }
    return npt;
}

double l3m_now_ns(void) { return (double)__rdtsc() * l3m_ns_per_tick(); }

// ---- hardware counters --------------------------------------------------------------------

#define MAX_FDS 16

struct l3m_hwcounters {
    int   l3_fd[MAX_FDS], n_l3;        // one per cpu in the amd_l3 cpumask
    int   umc_fd[MAX_FDS], n_umc;      // one per memory channel
    int   rapl_fd;
    double rapl_scale;                 // joules per count
    double t0;
    uint64_t l3_last, umc_last, rapl_last;
    char  mon_data[300];               // resctrl occupancy directory
};

static int verbose;

static long read_sysfs_long(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char buf[64] = "";
    if (!fgets(buf, sizeof buf, f)) { fclose(f); return -1; }
    fclose(f);
    return strtol(buf, NULL, 0);
}

static int pmu_type(const char *pmu) {
    char path[256];
    snprintf(path, sizeof path, "/sys/bus/event_source/devices/%s/type", pmu);
    return (int)read_sysfs_long(path);
}

// Open one counting event on a PMU, system-wide on `cpu`. Returns fd or -1.
static int open_event(int type, uint64_t config, int cpu) {
    struct perf_event_attr a = {0};
    a.type = (uint32_t)type; a.size = sizeof a; a.config = config;
    int fd = (int)syscall(SYS_perf_event_open, &a, -1, cpu, -1, 0);
    if (fd < 0 && verbose) fprintf(stderr, "perf: type %d config %#lx on cpu %d: %s\n", type, (unsigned long)config, cpu, strerror(errno));
    return fd;
}

// Open `config` once per cpu listed in the PMU's cpumask ("0,8"), up to MAX_FDS fds.
static int open_per_cpumask(const char *pmu, uint64_t config, int *fds) {
    int type = pmu_type(pmu);
    if (type < 0) return 0;
    char path[256], mask[256] = "";
    snprintf(path, sizeof path, "/sys/bus/event_source/devices/%s/cpumask", pmu);
    FILE *f = fopen(path, "r");
    if (!f || !fgets(mask, sizeof mask, f)) { if (f) fclose(f); return 0; }
    fclose(f);
    int n = 0;
    for (char *tok = strtok(mask, ",\n"); tok && n < MAX_FDS; tok = strtok(NULL, ",\n")) {
        int fd = open_event(type, config, atoi(tok));
        if (fd >= 0) fds[n++] = fd;
    }
    return n;
}

static uint64_t read_sum(const int *fds, int n) {
    uint64_t sum = 0, v;
    for (int i = 0; i < n; i++) if (read(fds[i], &v, sizeof v) == (ssize_t)sizeof v) sum += v;
    return sum;
}

// LLC occupancy: as root, from a monitoring group of this process's threads; otherwise machine-wide.
#define RESCTRL "/sys/fs/resctrl"

static int write_str(const char *path, const char *s) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    int ok = fputs(s, f) >= 0;
    return fclose(f) == 0 && ok ? 0 : -1;
}

static void mon_group_join(char *group, size_t n) {   // group: the mon_data directory to read, root if unprivileged
    snprintf(group, n, RESCTRL "/mon_data");
    char dir[256], path[512], tid[300];
    snprintf(dir, sizeof dir, RESCTRL "/mon_groups/l3m.%d", getpid());
    if (mkdir(dir, 0755) && errno != EEXIST) return;
    DIR *tasks = opendir("/proc/self/task");
    if (!tasks) return;
    snprintf(path, sizeof path, "%s/tasks", dir);
    for (struct dirent *e; (e = readdir(tasks));) {
        if (e->d_name[0] == '.') continue;
        snprintf(tid, sizeof tid, "%s\n", e->d_name);
        if (write_str(path, tid)) { closedir(tasks); rmdir(dir); return; }
    }
    closedir(tasks);
    snprintf(group, n, "%s/mon_data", dir);
}

static void mon_group_leave(void) {
    char dir[256];
    snprintf(dir, sizeof dir, RESCTRL "/mon_groups/l3m.%d", getpid());
    rmdir(dir);
}

static double llc_occupancy(const char *mon_data) {
    DIR *d = opendir(mon_data);
    if (!d) return NAN;
    double sum = 0; int found = 0;
    for (struct dirent *e; (e = readdir(d));) {
        if (strncmp(e->d_name, "mon_L3_", 7)) continue;
        char path[512];
        snprintf(path, sizeof path, "%s/%s/llc_occupancy", mon_data, e->d_name);
        long v = read_sysfs_long(path);
        if (v >= 0) { sum += (double)v; found = 1; }
    }
    closedir(d);
    return found ? sum : NAN;
}

l3m_hwcounters *l3m_hwcounters_open(int v) {
    verbose = v;
    l3m_hwcounters *c = calloc(1, sizeof *c);
    c->rapl_fd = -1;
    // amd_l3/l3_lookup_state.l3_miss/ and amd_umc_N/umc_cas_cmd.rd/
    c->n_l3 = open_per_cpumask("amd_l3", 0x104, c->l3_fd);
    char pmu[32];
    for (int i = 0; c->n_umc < MAX_FDS; i++) {
        snprintf(pmu, sizeof pmu, "amd_umc_%d", i);
        if (pmu_type(pmu) < 0) break;
        int fd[MAX_FDS], n = open_per_cpumask(pmu, 0x10a, fd);   // one channel: count it once
        if (n) c->umc_fd[c->n_umc++] = fd[0];
        for (int j = 1; j < n; j++) close(fd[j]);
    }
    int power = pmu_type("power");
    FILE *f = fopen("/sys/bus/event_source/devices/power/events/energy-pkg", "r");   // "event=0x02"
    char buf[64] = "";
    if (power >= 0 && f && fgets(buf, sizeof buf, f) && strchr(buf, '=')) {
        c->rapl_fd = open_event(power, strtoull(strchr(buf, '=') + 1, NULL, 0), 0);
        FILE *s = fopen("/sys/bus/event_source/devices/power/events/energy-pkg.scale", "r");
        if (s) { if (fscanf(s, "%lf", &c->rapl_scale) != 1) c->rapl_scale = 0; fclose(s); }
    }
    if (f) fclose(f);
    mon_group_join(c->mon_data, sizeof c->mon_data);
    int any = c->n_l3 || c->n_umc || c->rapl_fd >= 0 || !isnan(llc_occupancy(c->mon_data));
    if (!any) { free(c); return NULL; }
    c->t0 = l3m_now_ns();
    c->l3_last = read_sum(c->l3_fd, c->n_l3);
    c->umc_last = read_sum(c->umc_fd, c->n_umc);
    c->rapl_last = c->rapl_fd >= 0 ? read_sum(&c->rapl_fd, 1) : 0;
    return c;
}

void l3m_hwcounters_read(l3m_hwcounters *c, l3m_counters *out) {
    double t = l3m_now_ns();
    uint64_t l3 = read_sum(c->l3_fd, c->n_l3), umc = read_sum(c->umc_fd, c->n_umc);
    uint64_t rapl = c->rapl_fd >= 0 ? read_sum(&c->rapl_fd, 1) : 0;
    out->ns = t - c->t0;
    out->l3_miss = c->n_l3 ? (double)(l3 - c->l3_last) : NAN;
    out->dram_bytes = c->n_umc ? (double)(umc - c->umc_last) * 64 : NAN;
    out->energy_j = c->rapl_fd >= 0 ? (double)(rapl - c->rapl_last) * c->rapl_scale : NAN;
    out->llc_occupancy = llc_occupancy(c->mon_data);
    c->t0 = t; c->l3_last = l3; c->umc_last = umc; c->rapl_last = rapl;
}

void l3m_hwcounters_close(l3m_hwcounters *c) {
    if (!c) return;
    mon_group_leave();
    for (int i = 0; i < c->n_l3; i++) close(c->l3_fd[i]);
    for (int i = 0; i < c->n_umc; i++) close(c->umc_fd[i]);
    if (c->rapl_fd >= 0) close(c->rapl_fd);
    free(c);
}
