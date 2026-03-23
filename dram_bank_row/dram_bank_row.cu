/*
 * dram_bank_row.cu — DRAM Bank & Row Size Discovery for NVIDIA GPUs
 *
 * Automatically identifies memory technology (HBM / GDDR) and adapts
 * the probing strategy.  Supports testing every GPU in a single launch.
 *
 * Phase 1 – Stride pointer-chase  (all memory types)
 *   Sequential P-chase with ld.global.cg at varying strides.
 *   Working-set > L2 ⇒ raw DRAM latency.
 *     GDDR (open-page): row-buffer conflicts elevate latency at specific
 *       strides, revealing bank stride and row size.
 *     HBM  (closed-page): latency is ~flat in the DRAM region; shows the
 *       L2→DRAM transition but NOT bank conflicts.
 *
 * Phase 2 – Memory-type-specific probing
 *   GDDR: pair-wise conflict test with L2 flush  (opt-in: --pair-test).
 *   HBM:  "hammer" – 2-element alternating P-chase using ld.global.lu
 *         (last-use cache eviction).  Same-bank pairs are serialised by
 *         the DRAM row-cycle time (tRC), producing higher per-step
 *         latency than different-bank pairs.  The scan covers fine
 *         (channel interleave), medium (bank), and coarse (stack)
 *         granularity to expose the 3-D HBM structure.
 *
 * Build:
 *   make ARCH=ampere   # sm_80  (A100 / A6000)
 *   make ARCH=hopper   # sm_90  (H100 / H800)
 *   make ARCH=both     # fat binary – needed for multi-GPU with mixed arch
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdarg>
#include <ctime>
#include <vector>
#include <algorithm>

/* ================================================================== */
/*  Macros                                                             */
/* ================================================================== */

#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t _e = (call);                                       \
        if (_e != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d – %s\n", __FILE__,      \
                    __LINE__, cudaGetErrorString(_e));                  \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

/* ================================================================== */
/*  Types                                                              */
/* ================================================================== */

enum MemType {
    MEM_HBM2   = 0,
    MEM_HBM2E  = 1,
    MEM_HBM3   = 2,
    MEM_GDDR6  = 3,
    MEM_GDDR6X = 4,
    MEM_UNKNOWN = 5,
};

static const char *memtype_name(MemType t) {
    static const char *names[] =
        {"HBM2", "HBM2e", "HBM3", "GDDR6", "GDDR6X", "Unknown"};
    return names[t <= MEM_UNKNOWN ? (int)t : (int)MEM_UNKNOWN];
}
static bool memtype_is_hbm(MemType t) { return t <= MEM_HBM3; }

struct HBMSpec {
    int  stacks;
    int  ch_per_stack;
    int  banks_per_ch;
    int  row_bytes;
};

struct Config {
    int    target_dev;       /* -1 = all */
    size_t alloc_mb;
    size_t stride_min, stride_max;
    int    steps, warmup, iters;
    bool   pair_test;        /* GDDR explicit opt-in */
    size_t pair_range_mb, pair_stride;
    int    pair_iters;
    int    hbm_steps;        /* HBM hammer steps per probe */
    size_t hbm_range_mb;
    const char *csv_path;
    const char *log_path;
    bool   verbose;
};

/* ================================================================== */
/*  Logging                                                            */
/* ================================================================== */

static FILE *g_log = nullptr;

static void emit(const char *fmt, ...) {
    va_list a;
    va_start(a, fmt); vprintf(fmt, a); va_end(a);
    if (g_log) {
        va_start(a, fmt); vfprintf(g_log, fmt, a); va_end(a);
        fflush(g_log);
    }
}

static void emit_log(const char *fmt, ...) {
    if (!g_log) return;
    va_list a;
    va_start(a, fmt); vfprintf(g_log, fmt, a); va_end(a);
    fflush(g_log);
}

/* ================================================================== */
/*  Device helpers  (inline PTX, sm_80+)                               */
/* ================================================================== */

__device__ __forceinline__ uint64_t ld_cg_u64(const void *p) {
    uint64_t v;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ uint32_t ld_ca_u32(const void *p) {
    uint32_t v;
    asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ uint32_t ld_cg_u32(const void *p) {
    uint32_t v;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ uint64_t ld_lu_u64(const void *p) {
    uint64_t v;
    asm volatile("ld.global.lu.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
}

/* ================================================================== */
/*  Kernels                                                            */
/* ================================================================== */

__global__ void init_pchase(char *buf, uint64_t stride, uint64_t n) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)blockDim.x * gridDim.x;
    for (uint64_t i = tid; i < n; i += total)
        *(uint64_t *)(buf + i * stride) = ((i + 1) % n) * stride;
}

/* Phase 1: stride pointer chase (ld.global.cg, single thread). */
__global__ void stride_pchase(const char *buf, int steps, int warmup,
                              uint64_t *out_cycles) {
    if (threadIdx.x | blockIdx.x) return;
    uint64_t off = 0;
    for (int i = 0; i < warmup; i++) off = ld_cg_u64(buf + off);
    uint64_t t0 = clock64();
    for (int i = 0; i < steps; i++) off = ld_cg_u64(buf + off);
    asm volatile("" ::"l"(off));
    *out_cycles = clock64() - t0;
    if (off == 0xDEADDEADDEADDEADULL) *out_cycles = off;
}

/*
 * HBM diagnostic: 2-element chase comparing ld.cg (L2-cached) vs
 * ld.lu (last-use, should invalidate L2 line after read).
 * If ld.lu latency >> ld.cg latency  → ld.lu bypasses L2 effectively.
 */
__global__ void chase2_diag(char *buf, uint64_t oa, uint64_t ob,
                            int steps, int warmup,
                            uint64_t *out_cg, uint64_t *out_lu) {
    if (threadIdx.x | blockIdx.x) return;

    *(volatile uint64_t *)(buf + oa) = ob;
    *(volatile uint64_t *)(buf + ob) = oa;
    __threadfence();

    uint64_t off, t0;

    off = oa;
    for (int i = 0; i < warmup; i++) off = ld_cg_u64(buf + off);
    t0 = clock64();
    for (int i = 0; i < steps; i++) off = ld_cg_u64(buf + off);
    asm volatile("" ::"l"(off));
    *out_cg = clock64() - t0;

    off = oa;
    for (int i = 0; i < warmup; i++) off = ld_lu_u64(buf + off);
    t0 = clock64();
    for (int i = 0; i < steps; i++) off = ld_lu_u64(buf + off);
    asm volatile("" ::"l"(off));
    *out_lu = clock64() - t0;

    if (off == 0xDEADDEADDEADDEADULL) { *out_cg = off; *out_lu = off; }
}

/*
 * HBM hammer: batch of 2-element alternating P-chases with ld.lu.
 *
 * For each test offset the kernel writes a tiny circular chase
 *   ref → test → ref → test → …
 * then times `steps` iterations.  ld.lu invalidates each cache line
 * after reading so every step goes to DRAM.
 *
 * Same-bank pairs are serialised by tRC (row-cycle time) ⇒ higher
 * per-step latency.  Different-bank (or different-channel / different-
 * stack) pairs allow bank-level pipelining ⇒ lower latency.
 * This captures the full 3-D HBM hierarchy: intra-bank > intra-channel
 * > intra-stack > inter-stack.
 */
__global__ void hbm_hammer(char *buf, uint64_t ref_off,
                           const uint64_t *test_offs, uint32_t *out_lat,
                           int n_tests, int steps, int warmup) {
    if (threadIdx.x | blockIdx.x) return;

    for (int t = 0; t < n_tests; t++) {
        uint64_t to = test_offs[t];

        *(volatile uint64_t *)(buf + ref_off) = to;
        *(volatile uint64_t *)(buf + to) = ref_off;
        __threadfence();

        uint64_t off = ref_off;
        for (int i = 0; i < warmup; i++) off = ld_lu_u64(buf + off);

        uint64_t t0 = clock64();
        for (int i = 0; i < steps; i++) off = ld_lu_u64(buf + off);
        asm volatile("" ::"l"(off));
        out_lat[t] = (uint32_t)((clock64() - t0) / steps);
    }
}

/*
 * GDDR pair-wise conflict test with L2 flush.
 * (For GDDR open-page DRAM where row-buffer conflicts produce a clear
 * latency signal.)
 */
__global__ void gddr_pair_test(
    const char *buf, const char *flush_buf,
    uint32_t flush_lines, uint64_t ref_off,
    const uint64_t *test_offs, uint32_t *out_lat,
    int n_tests, int iters)
{
    if (threadIdx.x | blockIdx.x) return;
    uint32_t sink = 0;
    for (int t = 0; t < n_tests; t++) {
        uint64_t toff = test_offs[t];
        uint64_t total = 0;
        for (int it = 0; it < iters; it++) {
            for (uint32_t f = 0; f < flush_lines; f++)
                sink += ld_ca_u32(flush_buf + (uint64_t)f * 128);
            asm volatile("membar.gl;");
            uint32_t rv = ld_cg_u32(buf + ref_off);
            uint32_t dep;
            asm volatile("and.b32 %0, %1, 0;" : "=r"(dep) : "r"(rv));
            uint64_t dep64 = dep;
            uint64_t s, e;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(s));
            uint32_t tv = ld_cg_u32(buf + toff + dep64);
            asm volatile("" ::"r"(tv));
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(e));
            sink += rv + tv;
            total += e - s;
        }
        out_lat[t] = (uint32_t)(total / iters);
    }
    if (sink == 0xDEADBEEFu) out_lat[0] = sink;
}

/* ================================================================== */
/*  Host helpers                                                       */
/* ================================================================== */

static const char *fmt_bytes(uint64_t b, char *s, size_t n) {
    if (b >= (1ULL << 20))
        snprintf(s, n, "%7.1f MB", b / (1024.0 * 1024.0));
    else if (b >= 1024)
        snprintf(s, n, "%7.1f KB", b / 1024.0);
    else
        snprintf(s, n, "%7llu  B", (unsigned long long)b);
    return s;
}

static MemType detect_mem_type(const cudaDeviceProp &p) {
    /*
     * HBM GPUs expose a memory bus ≥ 1024 bits (multiple stacked channels).
     * GDDR GPUs use 128 / 192 / 256 / 384-bit buses.
     */
    if (p.memoryBusWidth >= 1024) {
        if (p.major >= 9)  return MEM_HBM3;   /* Hopper  */
        if (p.major >= 8)  return MEM_HBM2E;  /* Ampere A100 */
        return MEM_HBM2;                       /* Volta V100  */
    }
    return MEM_GDDR6;
}

static HBMSpec estimate_hbm_spec(const cudaDeviceProp &p, MemType mt) {
    HBMSpec h{};
    size_t gb = p.totalGlobalMem >> 30;
    switch (mt) {
    case MEM_HBM3:
        h.ch_per_stack = 16;
        h.banks_per_ch = 32;
        h.row_bytes    = 2048;
        h.stacks       = (gb > 60) ? 6 : 5;
        break;
    case MEM_HBM2E:
        h.ch_per_stack = 8;
        h.banks_per_ch = 16;
        h.row_bytes    = 2048;
        h.stacks       = (gb > 60) ? 6 : 5;
        break;
    case MEM_HBM2:
        h.ch_per_stack = 8;
        h.banks_per_ch = 16;
        h.row_bytes    = 1024;
        h.stacks       = (int)(gb / 4);
        if (h.stacks < 1) h.stacks = 4;
        break;
    default: break;
    }
    return h;
}

/* Build a mixed fine + coarse scan sequence for HBM hammer. */
static std::vector<uint64_t> hbm_scan_offsets(size_t max_range) {
    std::vector<uint64_t> v;
    for (uint64_t o = 64; o <= 4096 && o <= max_range; o += 64)
        v.push_back(o);
    for (uint64_t s = 4096; s <= max_range; s *= 2) {
        if (s > 4096) v.push_back(s);
        uint64_t m = s + s / 2;
        if (m > s && m <= max_range) v.push_back(m);
    }
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

/* Build stride list for Phase 1 (powers-of-2 with 1.5× midpoints). */
static std::vector<size_t> build_strides(size_t lo, size_t hi) {
    std::vector<size_t> v;
    for (size_t s = lo; s <= hi; s *= 2) {
        if (s >= 8) v.push_back(s);
        size_t m = (s + s / 2 + 127) & ~127ULL;
        if (m > s && m < s * 2 && m <= hi && m >= 8) v.push_back(m);
    }
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

/* ================================================================== */
/*  Phase 1 – Stride P-chase                                          */
/* ================================================================== */

struct StrideResult { size_t stride; double cycles; bool dram; };

static std::vector<StrideResult>
run_phase1(char *d_buf, size_t alloc, size_t l2,
           double clk_ghz, const Config &cfg, FILE *csv, int dev) {

    emit("\n=== Phase 1: Stride P-chase ===\n\n");
    emit("  Working-set > L2 → DRAM latency;  marked [L2] otherwise.\n\n");
    emit("%-12s  %-10s  %-13s  %-12s  %-12s  %s\n",
         "Stride", "#Elem", "Cycles/step", "ns/step", "WorkingSet", "");
    emit("%-12s  %-10s  %-13s  %-12s  %-12s  %s\n",
         "------", "-----", "-----------", "-------", "----------", "");

    auto strides = build_strides(cfg.stride_min, cfg.stride_max);

    uint64_t *d_cyc;
    CUDA_CHECK(cudaMalloc(&d_cyc, sizeof(uint64_t)));

    std::vector<StrideResult> results;

    if (csv) fprintf(csv, "phase,device,stride_bytes,n_elements,"
                          "cycles_per_step,ns_per_step,working_set_bytes,"
                          "is_dram\n");

    for (size_t stride : strides) {
        if (stride > alloc / 2) break;
        uint64_t n_elem = alloc / stride;
        if (n_elem < 2) continue;
        size_t ws = n_elem * 128;
        bool is_dram = ws > l2;

        int thr = 256;
        int blk = (int)std::min((uint64_t)512, (n_elem + thr - 1) / thr);
        init_pchase<<<blk, thr>>>(d_buf, stride, n_elem);
        CUDA_CHECK(cudaDeviceSynchronize());

        double avg = 0;
        for (int it = 0; it < cfg.iters; it++) {
            CUDA_CHECK(cudaMemset(d_cyc, 0, sizeof(uint64_t)));
            stride_pchase<<<1, 1>>>(d_buf, cfg.steps, cfg.warmup, d_cyc);
            CUDA_CHECK(cudaDeviceSynchronize());
            uint64_t c;
            CUDA_CHECK(cudaMemcpy(&c, d_cyc, sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost));
            double cps = (double)c / cfg.steps;
            avg += cps;
            emit_log("    [P1 stride=%zu iter=%d/%d: %.1f cyc/step]\n",
                     stride, it + 1, cfg.iters, cps);
        }
        avg /= cfg.iters;
        double ns = avg / clk_ghz;

        char s1[32], s2[32];
        emit("%-12s  %-10llu  %11.1f    %10.1f  %-12s  %s\n",
             fmt_bytes(stride, s1, sizeof(s1)),
             (unsigned long long)n_elem, avg, ns,
             fmt_bytes(ws, s2, sizeof(s2)),
             is_dram ? "" : "[L2]");

        results.push_back({stride, avg, is_dram});

        if (csv)
            fprintf(csv, "stride,%d,%zu,%llu,%.2f,%.2f,%zu,%d\n",
                    dev, stride, (unsigned long long)n_elem, avg, ns,
                    ws, is_dram ? 1 : 0);
    }
    CUDA_CHECK(cudaFree(d_cyc));
    return results;
}

static void analyse_phase1(const std::vector<StrideResult> &all,
                           double clk_ghz, bool is_hbm) {
    emit("\n=== Phase 1 Analysis ===\n\n");

    std::vector<StrideResult> dr;
    for (auto &r : all) if (r.dram) dr.push_back(r);

    if (dr.size() < 3) {
        emit("  Too few DRAM-level measurements.  Increase --alloc.\n");
        return;
    }

    double lo = 1e18, hi = 0;
    size_t lo_s = 0, hi_s = 0;
    for (auto &r : dr) {
        if (r.cycles < lo) { lo = r.cycles; lo_s = r.stride; }
        if (r.cycles > hi) { hi = r.cycles; hi_s = r.stride; }
    }
    char b1[32], b2[32];
    emit("  DRAM latency range : %.1f – %.1f cycles  (%.1f – %.1f ns)\n",
         lo, hi, lo / clk_ghz, hi / clk_ghz);
    emit("  Min @ stride %-10s  Max @ stride %s\n",
         fmt_bytes(lo_s, b1, sizeof(b1)),
         fmt_bytes(hi_s, b2, sizeof(b2)));
    emit("  Conflict ratio     : %.2fx\n\n", hi / lo);

    if (is_hbm && hi / lo < 1.15) {
        emit("  HBM closed-page policy detected (flat DRAM latency).\n"
             "  Stride P-chase cannot resolve bank conflicts on HBM;\n"
             "  the HBM hammer (Phase 2) will probe bank structure.\n");
        return;
    }

    double thr = lo + 0.3 * (hi - lo);
    emit("  Conflict threshold (30%% of range): %.1f cycles\n", thr);
    emit("  Strides with row-buffer conflicts:\n");
    for (auto &r : dr) {
        if (r.cycles > thr) {
            char bb[32];
            emit("    %-12s  %.1f cyc  (%.1f ns)\n",
                 fmt_bytes(r.stride, bb, sizeof(bb)),
                 r.cycles, r.cycles / clk_ghz);
        }
    }

    size_t first_c = 0, last_c = 0;
    for (auto &r : dr) {
        if (r.cycles > thr) {
            if (!first_c) first_c = r.stride;
            last_c = r.stride;
        }
    }
    if (first_c) {
        char c1[32], c2[32];
        emit("\n  Estimated DRAM parameters (heuristic):\n");
        emit("    First conflict stride : %s\n",
             fmt_bytes(first_c, c1, sizeof(c1)));
        emit("    Last  conflict stride : %s\n",
             fmt_bytes(last_c, c2, sizeof(c2)));
        if (last_c > first_c) {
            emit("    → Approx bank count  : ~%zu\n", last_c / first_c);
            emit("    → Approx row size    : %s\n",
                 fmt_bytes(last_c, c2, sizeof(c2)));
        }
    }
}

/* ================================================================== */
/*  Phase 2 – GDDR pair-wise conflict test                             */
/* ================================================================== */

static void run_phase2_gddr(char *d_buf, size_t alloc, size_t l2,
                            double clk_ghz, size_t free_mem,
                            const Config &cfg, FILE *csv, int dev) {

    emit("\n=== Phase 2 (GDDR): Pair-wise Conflict Test ===\n\n");

    size_t flush_bytes = l2 * 3 / 2;
    if (flush_bytes > free_mem / 4) flush_bytes = l2;

    char *d_flush;
    CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));
    CUDA_CHECK(cudaMemset(d_flush, 0xAA, flush_bytes));
    uint32_t flush_lines = (uint32_t)(flush_bytes / 128);

    size_t pr_bytes = cfg.pair_range_mb << 20;
    if (pr_bytes > alloc - cfg.pair_stride) pr_bytes = alloc - cfg.pair_stride;
    size_t n_tests = pr_bytes / cfg.pair_stride;
    if (n_tests > 100000) n_tests = 100000;

    emit("  Reference : offset 0\n");
    emit("  Range     : %zu MB   stride : %zu B   tests : %zu\n",
         pr_bytes >> 20, cfg.pair_stride, n_tests);
    emit("  Flush buf : %zu MB   iters  : %d\n\n",
         flush_bytes >> 20, cfg.pair_iters);

    std::vector<uint64_t> h_to(n_tests);
    for (size_t i = 0; i < n_tests; i++) h_to[i] = (i + 1) * cfg.pair_stride;

    uint64_t *d_to; uint32_t *d_lat;
    CUDA_CHECK(cudaMalloc(&d_to,  n_tests * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_lat, n_tests * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_to, h_to.data(),
                          n_tests * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));

    const int B = 128;
    int nb = (int)((n_tests + B - 1) / B);
    emit("  Processing %d batches ", nb); fflush(stdout);
    for (size_t o = 0; o < n_tests; o += B) {
        int cnt = (int)std::min((size_t)B, n_tests - o);
        gddr_pair_test<<<1, 1>>>(d_buf, d_flush, flush_lines, 0ULL,
                                 d_to + o, d_lat + o, cnt, cfg.pair_iters);
        CUDA_CHECK(cudaDeviceSynchronize());
        putchar('.'); fflush(stdout);
        emit_log("  [GDDR pair batch %zu/%d done]\n", o / B + 1, nb);
    }
    emit(" done.\n\n");

    std::vector<uint32_t> h_lat(n_tests);
    CUDA_CHECK(cudaMemcpy(h_lat.data(), d_lat,
                          n_tests * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    uint32_t lo = *std::min_element(h_lat.begin(), h_lat.end());
    uint32_t hi = *std::max_element(h_lat.begin(), h_lat.end());
    uint32_t ct = lo + (hi - lo) / 2;

    emit("  %-14s  %-10s  %-10s  %s\n", "Offset","Cycles","ns","Conflict?");
    emit("  %-14s  %-10s  %-10s  %s\n", "------","------","--","---------");

    size_t cf_cnt = 0;
    if (csv) fprintf(csv, "\nphase,device,offset_bytes,cycles,ns,"
                          "is_conflict\n");

    for (size_t i = 0; i < n_tests; i++) {
        uint64_t ob = h_to[i];
        bool cf = h_lat[i] > ct;
        if (cf) cf_cnt++;
        bool show = (n_tests <= 200) || (i < 80) ||
                    (i >= n_tests - 20) || cf;
        if (show) {
            char bb[32];
            emit("  %-14s  %-10u  %8.1f  %s\n",
                 fmt_bytes(ob, bb, sizeof(bb)),
                 h_lat[i], h_lat[i] / clk_ghz, cf ? "***" : "");
        } else if (i == 80) {
            emit("  … (%zu middle non-conflict entries hidden) …\n",
                 n_tests - 100);
        }
        if (csv) fprintf(csv, "pair,%d,%llu,%u,%.2f,%d\n",
                         dev, (unsigned long long)ob,
                         h_lat[i], h_lat[i] / clk_ghz, cf ? 1 : 0);
        emit_log("  [GDDR pair off=%llu lat=%u cf=%d]\n",
                 (unsigned long long)ob, h_lat[i], cf ? 1 : 0);
    }

    emit("\n  Pair-test summary:\n");
    emit("    Latency range : %u – %u cycles (%.1f – %.1f ns)\n",
         lo, hi, lo / clk_ghz, hi / clk_ghz);
    emit("    Threshold     : %u cycles\n", ct);
    emit("    Conflicts     : %zu / %zu (%.1f%%)\n",
         cf_cnt, n_tests, 100.0 * cf_cnt / n_tests);
    if (cf_cnt > 0 && cf_cnt < n_tests) {
        double frac = (double)cf_cnt / n_tests;
        emit("    Same-bank frac: %.4f → ~%d banks\n",
             frac, (int)(1.0 / frac + 0.5));
    }

    CUDA_CHECK(cudaFree(d_flush));
    CUDA_CHECK(cudaFree(d_to));
    CUDA_CHECK(cudaFree(d_lat));
}

/* ================================================================== */
/*  Phase 2 – HBM hammer                                               */
/* ================================================================== */

static void run_phase2_hbm(char *d_buf, size_t alloc,
                           double clk_ghz, const HBMSpec &spec,
                           const Config &cfg, FILE *csv, int dev) {

    emit("\n=== Phase 2 (HBM): Bank / Channel Hammer ===\n\n");

    int total_ch    = spec.stacks * spec.ch_per_stack;
    int total_banks = total_ch * spec.banks_per_ch;

    emit("  HBM 3-D structure (estimated):\n");
    emit("    Stacks             : %d\n", spec.stacks);
    emit("    Channels / stack   : %d  (total %d)\n",
         spec.ch_per_stack, total_ch);
    emit("    Banks / channel    : %d  (total %d)\n",
         spec.banks_per_ch, total_banks);
    emit("    Row size / bank    : %d B\n\n", spec.row_bytes);

    /* ---- ld.lu diagnostic ---- */
    emit("  ld.lu diagnostic (does ld.lu bypass L2?):\n");

    uint64_t *d_cg, *d_lu;
    CUDA_CHECK(cudaMalloc(&d_cg, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_lu, sizeof(uint64_t)));

    uint64_t diag_off_a = 0, diag_off_b = 4096;
    chase2_diag<<<1, 1>>>(d_buf, diag_off_a, diag_off_b,
                          20000, 500, d_cg, d_lu);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint64_t cg_cyc, lu_cyc;
    CUDA_CHECK(cudaMemcpy(&cg_cyc, d_cg, 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&lu_cyc, d_lu, 8, cudaMemcpyDeviceToHost));
    double cg_cps = cg_cyc / 20000.0, lu_cps = lu_cyc / 20000.0;
    double lu_ratio = lu_cps / cg_cps;

    emit("    ld.cg 2-elem chase : %7.1f cyc/step  (%.1f ns)  [L2 cached]\n",
         cg_cps, cg_cps / clk_ghz);
    emit("    ld.lu 2-elem chase : %7.1f cyc/step  (%.1f ns)\n",
         lu_cps, lu_cps / clk_ghz);
    emit("    Ratio ld.lu / ld.cg: %.2fx", lu_ratio);

    bool lu_ok = lu_ratio > 1.5;
    if (lu_ok)
        emit("  → ld.lu effective ✓\n\n");
    else
        emit("  → ld.lu NOT effective (falling back to L2 flush)\n\n");

    emit_log("  [diag cg=%.1f lu=%.1f ratio=%.2f ok=%d]\n",
             cg_cps, lu_cps, lu_ratio, lu_ok);

    CUDA_CHECK(cudaFree(d_cg));
    CUDA_CHECK(cudaFree(d_lu));

    /* ---- build scan offsets ---- */
    size_t range = (cfg.hbm_range_mb << 20);
    if (range > alloc - 64) range = alloc - 64;

    auto offsets = hbm_scan_offsets(range);
    size_t n_tests = offsets.size();

    emit("  Hammer test (%s):\n",
         lu_ok ? "ld.lu 2-element P-chase"
               : "L2-flush pair-test (fallback)");
    emit("    Reference  : offset 0\n");
    emit("    Scan points: %zu  (up to %s)\n\n",
         n_tests, ({char b[32]; fmt_bytes(range, b, sizeof(b)); b;}));

    /* allocate device arrays */
    uint64_t *d_to; uint32_t *d_lat;
    CUDA_CHECK(cudaMalloc(&d_to,  n_tests * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_lat, n_tests * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_to, offsets.data(),
                          n_tests * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));

    if (lu_ok) {
        /* ---- hammer with ld.lu ---- */
        const int B = 64;
        for (size_t o = 0; o < n_tests; o += B) {
            int cnt = (int)std::min((size_t)B, n_tests - o);
            hbm_hammer<<<1, 1>>>(d_buf, 0ULL, d_to + o, d_lat + o,
                                 cnt, cfg.hbm_steps, 200);
            CUDA_CHECK(cudaDeviceSynchronize());
            emit_log("  [hammer batch %zu/%zu done]\n",
                     o / B + 1, (n_tests + B - 1) / B);
        }
    } else {
        /* ---- fallback: reuse GDDR L2-flush pair test ---- */
        size_t l2 = 50ULL << 20; /* conservative */
        {
            cudaDeviceProp p;
            CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
            l2 = p.l2CacheSize;
        }
        size_t fb = l2 * 3 / 2;
        char *d_flush;
        CUDA_CHECK(cudaMalloc(&d_flush, fb));
        CUDA_CHECK(cudaMemset(d_flush, 0xAA, fb));
        uint32_t fl = (uint32_t)(fb / 128);
        const int B = 32;
        for (size_t o = 0; o < n_tests; o += B) {
            int cnt = (int)std::min((size_t)B, n_tests - o);
            gddr_pair_test<<<1, 1>>>(d_buf, d_flush, fl, 0ULL,
                                     d_to + o, d_lat + o, cnt, 5);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaFree(d_flush));
    }

    /* ---- read back ---- */
    std::vector<uint32_t> h_lat(n_tests);
    CUDA_CHECK(cudaMemcpy(h_lat.data(), d_lat,
                          n_tests * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    uint32_t lo = *std::min_element(h_lat.begin(), h_lat.end());
    uint32_t hi = *std::max_element(h_lat.begin(), h_lat.end());
    uint32_t ct = lo + (hi - lo) / 2;

    emit("  %-14s  %-12s  %-10s  %s\n",
         "Offset", "Cyc/step", "ns/step", "Level");
    emit("  %-14s  %-12s  %-10s  %s\n",
         "------", "--------", "-------", "-----");

    if (csv) fprintf(csv, "\nphase,device,offset_bytes,cycles,ns,"
                          "is_same_bank\n");

    size_t same_cnt = 0;
    std::vector<uint64_t> same_bank_offs;

    for (size_t i = 0; i < n_tests; i++) {
        uint64_t ob = offsets[i];
        bool same = h_lat[i] > ct;
        if (same) { same_cnt++; same_bank_offs.push_back(ob); }

        char bb[32];
        emit("  %-14s  %-12u  %8.1f  %s\n",
             fmt_bytes(ob, bb, sizeof(bb)),
             h_lat[i], h_lat[i] / clk_ghz,
             same ? "same-bank" : "diff-bank");

        if (csv) fprintf(csv, "hammer,%d,%llu,%u,%.2f,%d\n",
                         dev, (unsigned long long)ob,
                         h_lat[i], h_lat[i] / clk_ghz, same ? 1 : 0);
        emit_log("  [hammer off=%llu lat=%u same=%d]\n",
                 (unsigned long long)ob, h_lat[i], same ? 1 : 0);
    }

    /* ---- analysis ---- */
    emit("\n  Hammer summary:\n");
    emit("    Same-bank latency  : ~%u cycles (%.1f ns)\n",
         hi, hi / clk_ghz);
    emit("    Diff-bank latency  : ~%u cycles (%.1f ns)\n",
         lo, lo / clk_ghz);
    emit("    Ratio              : %.2fx\n", (double)hi / lo);
    emit("    Same-bank fraction : %zu / %zu (%.1f%%)\n",
         same_cnt, n_tests, 100.0 * same_cnt / n_tests);

    if (same_bank_offs.size() >= 2) {
        /* look for the smallest repeating gap among same-bank offsets
           in the fine region (≤ 4 KB) to estimate channel interleave */
        std::vector<uint64_t> fine_sb;
        for (auto o : same_bank_offs) if (o <= 4096) fine_sb.push_back(o);

        if (fine_sb.size() >= 2) {
            uint64_t min_gap = UINT64_MAX;
            for (size_t i = 1; i < fine_sb.size(); i++) {
                uint64_t g = fine_sb[i] - fine_sb[i - 1];
                if (g > 0 && g < min_gap) min_gap = g;
            }
            char gb[32];
            emit("    Fine-region same-bank gap : %s\n",
                 fmt_bytes(min_gap, gb, sizeof(gb)));
            if (min_gap > 64) {
                uint64_t ch_stride = min_gap;
                emit("    → Est. channel interleave: %s\n",
                     fmt_bytes(ch_stride, gb, sizeof(gb)));
            }
        }

        /* coarse periodicity: look for the GCD of all same-bank offsets */
        uint64_t g = same_bank_offs[0];
        for (size_t i = 1; i < same_bank_offs.size(); i++) {
            uint64_t a = g, b = same_bank_offs[i];
            while (b) { uint64_t t = b; b = a % b; a = t; }
            g = a;
        }
        if (g >= 64) {
            char gb[32];
            emit("    GCD of same-bank offsets : %s\n",
                 fmt_bytes(g, gb, sizeof(gb)));
        }
    }

    CUDA_CHECK(cudaFree(d_to));
    CUDA_CHECK(cudaFree(d_lat));
}

/* ================================================================== */
/*  Per-device test orchestrator                                       */
/* ================================================================== */

static void run_device(int dev, const Config &cfg, FILE *csv) {
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    MemType mt = detect_mem_type(prop);
    bool    hbm = memtype_is_hbm(mt);
    size_t  l2  = (size_t)prop.l2CacheSize;
    double  ghz = prop.clockRate / 1.0e6;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    emit("╔══════════════════════════════════════════════════╗\n");
    emit("║  Device %d : %-38s║\n", dev, prop.name);
    emit("╚══════════════════════════════════════════════════╝\n\n");

    emit("  Compute      : %d.%d\n", prop.major, prop.minor);
    emit("  Memory type  : %s\n", memtype_name(mt));
    emit("  Memory       : %zu MB total, %zu MB free\n",
         total_mem >> 20, free_mem >> 20);
    emit("  L2 Cache     : %zu KB\n", l2 >> 10);
    emit("  SM Clock     : %d MHz (%.3f GHz)\n",
         prop.clockRate / 1000, ghz);
    emit("  Mem bus      : %d bit\n", prop.memoryBusWidth);

    HBMSpec hspec{};
    if (hbm) {
        hspec = estimate_hbm_spec(prop, mt);
        emit("\n  HBM 3-D structure (estimated):\n");
        emit("    Stacks           : %d\n", hspec.stacks);
        emit("    Channels / stack : %d  (total %d)\n",
             hspec.ch_per_stack,
             hspec.stacks * hspec.ch_per_stack);
        emit("    Banks / channel  : %d  (total %d)\n",
             hspec.banks_per_ch,
             hspec.stacks * hspec.ch_per_stack * hspec.banks_per_ch);
        emit("    Row size / bank  : %d B\n", hspec.row_bytes);
    }

    /* allocation */
    size_t alloc = cfg.alloc_mb << 20;
    if (alloc > free_mem * 9 / 10)
        alloc = (free_mem * 9 / 10) & ~0xFFFULL;
    if (alloc < l2 * 4) alloc = l2 * 4;
    emit("\n  Buffer       : %zu MB\n", alloc >> 20);

    emit_log("  [alloc=%zu free=%zu l2=%zu clk=%.3f]\n",
             alloc, free_mem, l2, ghz);

    char *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, alloc));
    CUDA_CHECK(cudaMemset(d_buf, 0, alloc));

    /* Phase 1 */
    auto p1 = run_phase1(d_buf, alloc, l2, ghz, cfg, csv, dev);
    analyse_phase1(p1, ghz, hbm);

    /* Phase 2 */
    if (hbm) {
        run_phase2_hbm(d_buf, alloc, ghz, hspec, cfg, csv, dev);
    }
    if (cfg.pair_test) {
        run_phase2_gddr(d_buf, alloc, l2, ghz, free_mem, cfg, csv, dev);
    }

    CUDA_CHECK(cudaFree(d_buf));
    emit("\n");
}

/* ================================================================== */
/*  CLI                                                                */
/* ================================================================== */

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n\n"
        "General:\n"
        "  --device <N|all>       GPU to test (default: all)\n"
        "  --log <file>           Log file    (default: dram_bench.log)\n"
        "  --csv <file>           CSV output\n"
        "  -v, --verbose          Verbose\n"
        "  -h, --help             Help\n\n"
        "Phase 1 (stride P-chase):\n"
        "  --alloc <MB>           Buffer size       (default: 4096)\n"
        "  --stride-min <B>       Min stride        (default: 128)\n"
        "  --stride-max <B>       Max stride        (default: 4194304)\n"
        "  --steps <N>            Timed steps       (default: 200000)\n"
        "  --warmup <N>           Warmup steps      (default: 10000)\n"
        "  --iters <N>            Averaging iters   (default: 3)\n\n"
        "Phase 2 – GDDR pair test (opt-in):\n"
        "  --pair-test            Enable GDDR pair test\n"
        "  --pair-range <MB>      Scan range        (default: 32)\n"
        "  --pair-stride <B>      Granularity       (default: 256)\n"
        "  --pair-iters <N>       Repeats           (default: 10)\n\n"
        "Phase 2 – HBM hammer (automatic for HBM GPUs):\n"
        "  --hbm-steps <N>        Steps per probe   (default: 20000)\n"
        "  --hbm-range <MB>       Scan range        (default: 64)\n",
        prog);
}

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */

int main(int argc, char **argv) {

    Config cfg{};
    cfg.target_dev   = -1;
    cfg.alloc_mb     = 4096;
    cfg.stride_min   = 128;
    cfg.stride_max   = 4ULL << 20;
    cfg.steps        = 200000;
    cfg.warmup       = 10000;
    cfg.iters        = 3;
    cfg.pair_test    = false;
    cfg.pair_range_mb= 32;
    cfg.pair_stride  = 256;
    cfg.pair_iters   = 10;
    cfg.hbm_steps    = 20000;
    cfg.hbm_range_mb = 64;
    cfg.csv_path     = nullptr;
    cfg.log_path     = "dram_bench.log";
    cfg.verbose      = false;

    for (int i = 1; i < argc; i++) {
#define ARG(f) (!strcmp(argv[i], f))
#define NEXT (i + 1 < argc ? argv[++i] : \
              (fprintf(stderr, "missing arg for %s\n", argv[i]), \
               exit(1), (char*)0))
        if (ARG("--device")) {
            const char *v = NEXT;
            cfg.target_dev = strcmp(v, "all") ? atoi(v) : -1;
        }
        else if (ARG("--log"))         cfg.log_path     = NEXT;
        else if (ARG("--alloc"))       cfg.alloc_mb     = strtoull(NEXT,0,0);
        else if (ARG("--stride-min"))  cfg.stride_min   = strtoull(NEXT,0,0);
        else if (ARG("--stride-max"))  cfg.stride_max   = strtoull(NEXT,0,0);
        else if (ARG("--steps"))       cfg.steps        = atoi(NEXT);
        else if (ARG("--warmup"))      cfg.warmup       = atoi(NEXT);
        else if (ARG("--iters"))       cfg.iters        = atoi(NEXT);
        else if (ARG("--pair-test"))   cfg.pair_test    = true;
        else if (ARG("--pair-range"))  cfg.pair_range_mb= strtoull(NEXT,0,0);
        else if (ARG("--pair-stride")) cfg.pair_stride  = strtoull(NEXT,0,0);
        else if (ARG("--pair-iters"))  cfg.pair_iters   = atoi(NEXT);
        else if (ARG("--hbm-steps"))   cfg.hbm_steps    = atoi(NEXT);
        else if (ARG("--hbm-range"))   cfg.hbm_range_mb = strtoull(NEXT,0,0);
        else if (ARG("--csv"))         cfg.csv_path     = NEXT;
        else if (ARG("-v")||ARG("--verbose")) cfg.verbose = true;
        else if (ARG("-h")||ARG("--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr,"Unknown: %s\n",argv[i]);
               usage(argv[0]); return 1; }
#undef ARG
#undef NEXT
    }

    /* ---- log file ---- */
    g_log = fopen(cfg.log_path, "w");
    if (!g_log) { perror(cfg.log_path); return 1; }

    time_t now = time(nullptr);
    emit_log("=== DRAM Bank/Row Discovery – %s", ctime(&now));
    emit_log("Command:");
    for (int i = 0; i < argc; i++) emit_log(" %s", argv[i]);
    emit_log("\n\n");

    /* ---- CSV header ---- */
    FILE *csv = cfg.csv_path ? fopen(cfg.csv_path, "w") : nullptr;
    if (cfg.csv_path && !csv) { perror(cfg.csv_path); return 1; }

    /* ---- enumerate GPUs ---- */
    int n_dev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_dev));
    if (n_dev == 0) { emit("No CUDA devices found.\n"); return 1; }

    emit("=== DRAM Bank/Row Discovery ===\n");
    emit("  CUDA devices : %d\n\n", n_dev);

    emit_log("Total devices: %d  target: %s\n\n",
             n_dev, cfg.target_dev < 0 ? "all"
                    : std::to_string(cfg.target_dev).c_str());

    /* quick inventory */
    for (int d = 0; d < n_dev; d++) {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, d));
        MemType mt = detect_mem_type(p);
        emit("  [%d] %-30s  %s  %zu MB  sm_%d%d\n",
             d, p.name, memtype_name(mt),
             (size_t)(p.totalGlobalMem >> 20),
             p.major, p.minor);
    }
    emit("\n");

    /* ---- run per-device tests ---- */
    for (int d = 0; d < n_dev; d++) {
        if (cfg.target_dev >= 0 && d != cfg.target_dev) continue;
        run_device(d, cfg, csv);
    }

    if (csv) fclose(csv);
    emit("All done.  Log written to %s\n", cfg.log_path);
    if (g_log) fclose(g_log);
    return 0;
}
