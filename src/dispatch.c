// Pick the kernel table for this CPU. L3M_ISA=scalar|avx2|avx512 forces a lower one for testing.
#include <cpuid.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernels.h"

static int os_saves(unsigned bits) {          // does the OS preserve these XCR0 state components?
    unsigned eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (eax & bits) == bits;
}

// AVX2 is the build baseline; only AVX-512 is optional.
static const l3m_kernels *best(void) {
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return &l3m_kernels_avx2;
    int avx512 = (b & bit_AVX512F) && (b & bit_AVX512BW) && (b & bit_AVX512VL) && (c & bit_AVX512VNNI) && os_saves(0xE6);
    if (avx512 && __get_cpuid_count(7, 1, &a, &b, &c, &d) && (a & bit_AVX512BF16)) return &l3m_kernels_avx512;
    return &l3m_kernels_avx2;
}

int l3m_kernels_usable(const l3m_kernels *k) {   // every CPU runs scalar; the rest is ordered by capability
    const l3m_kernels *b = best();
    return k == &l3m_kernels_scalar || b == k || (k == &l3m_kernels_avx2 && b == &l3m_kernels_avx512);
}

const l3m_kernels *l3m_kernels_pick(void) {
    const l3m_kernels *k = best();
    const char *want = getenv("L3M_ISA");
    if (!want || !strcmp(want, k->name)) return k;
    if (!strcmp(want, "scalar")) return &l3m_kernels_scalar;
    if (!strcmp(want, "avx2") && k == &l3m_kernels_avx512) return &l3m_kernels_avx2;
    fprintf(stderr, "l3m: L3M_ISA=%s not available here, using %s\n", want, k->name);
    return k;
}
