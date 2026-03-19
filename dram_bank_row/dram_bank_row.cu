/*
 * dram_bank_row.cu — DRAM Bank & Row Size Discovery for NVIDIA GPUs
 *
 * Supports Ampere (sm_80) and Hopper (sm_90) architectures.
 *
 * Phase 1 – Stride P-chase:
 *   Pointer-chasing at power-of-two strides through a large buffer (> L2).
 *   Every chase step bypasses L1 (ld.global.cg) and, because the working
 *   set exceeds L2, hits DRAM.  Row-buffer conflicts at certain strides
 *   produce measurably higher per-step latency.
 *
 * Phase 2 – Pair-wise conflict test (optional, --pair-test):
 *   For each candidate address, flush L2, load a reference address (opens
 *   the row in its bank), then time a load from the candidate.  A row-
 *   buffer conflict (same bank, different row) increases the measured
 *   latency.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>

/* ------------------------------------------------------------------ */
/*  CUDA error checking                                                */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t _e = (call);                                       \
        if (_e != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d – %s\n", __FILE__,      \
                    __LINE__, cudaGetErrorString(_e));                  \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Device helpers                                                     */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/*  Kernels                                                            */
/* ------------------------------------------------------------------ */

/*
 * Initialise a sequential pointer chase in-place on the GPU.
 * Element at byte offset (i * stride) stores the byte offset of the
 * next element: ((i+1) % n) * stride.  Values are uint64_t so that
 * offsets > 4 GB are supported.
 */
__global__ void init_pchase(char *buf, uint64_t stride, uint64_t n) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t step = (uint64_t)blockDim.x * gridDim.x;
    for (uint64_t i = tid; i < n; i += step)
        *(uint64_t *)(buf + i * stride) = ((i + 1) % n) * stride;
}

/*
 * Phase 1: stride pointer chase.
 * Single-thread kernel.  Each step follows a data-dependent chain of
 * ld.global.cg.u64 loads, serialising DRAM accesses.
 */
__global__ void stride_pchase(const char *buf, int steps, int warmup,
                              uint64_t *out_cycles) {
    if (threadIdx.x | blockIdx.x) return;

    uint64_t off = 0;
    for (int i = 0; i < warmup; i++)
        off = ld_cg_u64(buf + off);

    uint64_t t0 = clock64();
    for (int i = 0; i < steps; i++)
        off = ld_cg_u64(buf + off);
    asm volatile("" ::"l"(off));
    uint64_t t1 = clock64();

    *out_cycles = t1 - t0;
    if (off == 0xDEADDEADDEADDEADULL) *out_cycles = off;
}

/*
 * Phase 2: pair-wise conflict test with L2 flush.
 * Processes a batch of test offsets against a single reference.
 *
 *   For each test offset:
 *     1. Read through flush_buf to fill L2 with unrelated data.
 *     2. Load from ref_off  → L2 miss → DRAM → opens row in ref's bank.
 *     3. Time a load from test_off → L2 miss → DRAM.
 *        If same bank / different row → row-buffer conflict → higher cycles.
 *
 * Reports the average cycle count over `iters` repetitions.
 */
__global__ void pair_conflict_kernel(
    const char *buf,
    const char *flush_buf,
    uint32_t   flush_lines,   /* number of 128-byte lines in flush buf */
    uint64_t   ref_off,       /* byte offset of reference address      */
    const uint64_t *test_offs,
    uint32_t  *out_lat,       /* per-test average cycles               */
    int        n_tests,
    int        iters)
{
    if (threadIdx.x | blockIdx.x) return;

    uint32_t sink = 0;
    for (int t = 0; t < n_tests; t++) {
        uint64_t toff = test_offs[t];
        uint64_t total = 0;

        for (int it = 0; it < iters; it++) {
            /* 1. flush L2 */
            for (uint32_t f = 0; f < flush_lines; f++)
                sink += ld_ca_u32(flush_buf + (uint64_t)f * 128);
            asm volatile("membar.gl;");

            /* 2. access reference → DRAM, opens row */
            uint32_t rv = ld_cg_u32(buf + ref_off);
            /*
             * Create a data dependency so the ref load must retire
             * before the test load issues – keeps the row buffer in
             * the expected state.
             */
            uint32_t dep;
            asm volatile("and.b32 %0, %1, 0;" : "=r"(dep) : "r"(rv));
            uint64_t dep64 = dep;

            /* 3. time test access (address depends on ref via dep64) */
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

/* ------------------------------------------------------------------ */
/*  Host utilities                                                     */
/* ------------------------------------------------------------------ */

static const char *fmt_bytes(uint64_t b, char *s, size_t n) {
    if (b >= (1ULL << 20))
        snprintf(s, n, "%7.1f MB", b / (1024.0 * 1024.0));
    else if (b >= 1024)
        snprintf(s, n, "%7.1f KB", b / 1024.0);
    else
        snprintf(s, n, "%7llu  B", (unsigned long long)b);
    return s;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n\n"
        "  --arch ampere|hopper   GPU architecture (default: auto-detect)\n"
        "  --alloc <MB>           GPU buffer in MB         (default: 4096)\n"
        "  --stride-min <bytes>   Min P-chase stride       (default: 128)\n"
        "  --stride-max <bytes>   Max P-chase stride       (default: 4194304)\n"
        "  --steps <N>            P-chase timed steps      (default: 200000)\n"
        "  --warmup <N>           P-chase warmup steps     (default: 10000)\n"
        "  --iters <N>            Averaging iterations     (default: 3)\n"
        "  --pair-test            Enable Phase 2\n"
        "  --pair-range <MB>      Pair scan range           (default: 32)\n"
        "  --pair-stride <bytes>  Pair scan granularity     (default: 256)\n"
        "  --pair-iters <N>       Pair measurement repeats  (default: 10)\n"
        "  --csv <file>           Write results to CSV\n"
        "  -v, --verbose          Verbose output\n"
        "  -h, --help             Show help\n", prog);
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {

    /* --- defaults --- */
    int    arch_ov     = 0;
    size_t alloc_mb    = 4096;
    size_t stride_min  = 128;
    size_t stride_max  = 4ULL << 20;
    int    steps       = 200000;
    int    warmup      = 10000;
    int    iters       = 3;
    bool   pair_test   = false;
    size_t pair_range  = 32;          /* MB */
    size_t pair_stride = 256;
    int    pair_iters  = 10;
    const char *csv_path = nullptr;
    bool   verbose     = false;

    /* --- parse args --- */
    for (int i = 1; i < argc; i++) {
#define ARG(flag) (!strcmp(argv[i], flag))
#define NEXT (i + 1 < argc ? argv[++i] : (fprintf(stderr, "missing arg for %s\n", argv[i]), exit(1), (char *)0))
        if (ARG("--arch")) {
            const char *a = NEXT;
            if (!strcmp(a, "ampere"))      arch_ov = 8;
            else if (!strcmp(a, "hopper")) arch_ov = 9;
            else { fprintf(stderr, "Unknown arch: %s\n", a); return 1; }
        }
        else if (ARG("--alloc"))        alloc_mb    = strtoull(NEXT, nullptr, 0);
        else if (ARG("--stride-min"))   stride_min  = strtoull(NEXT, nullptr, 0);
        else if (ARG("--stride-max"))   stride_max  = strtoull(NEXT, nullptr, 0);
        else if (ARG("--steps"))        steps       = atoi(NEXT);
        else if (ARG("--warmup"))       warmup      = atoi(NEXT);
        else if (ARG("--iters"))        iters       = atoi(NEXT);
        else if (ARG("--pair-test"))    pair_test   = true;
        else if (ARG("--pair-range"))   pair_range  = strtoull(NEXT, nullptr, 0);
        else if (ARG("--pair-stride"))  pair_stride = strtoull(NEXT, nullptr, 0);
        else if (ARG("--pair-iters"))   pair_iters  = atoi(NEXT);
        else if (ARG("--csv"))          csv_path    = NEXT;
        else if (ARG("-v") || ARG("--verbose")) verbose = true;
        else if (ARG("-h") || ARG("--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown flag: %s\n", argv[i]); usage(argv[0]); return 1; }
#undef ARG
#undef NEXT
    }

    /* --- GPU info --- */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int    compute   = arch_ov ? arch_ov : prop.major;
    size_t l2_bytes  = (size_t)prop.l2CacheSize;
    double clk_ghz   = prop.clockRate / 1.0e6;   /* SM boost clock */

    const char *arch_name = "Unknown";
    if (compute == 8)      arch_name = "Ampere";
    else if (compute == 9) arch_name = "Hopper";

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    printf("=== GPU Info ===\n");
    printf("  Device     : %s\n",   prop.name);
    printf("  Compute    : %d.%d (%s)\n", prop.major, prop.minor, arch_name);
    printf("  Memory     : %zu MB total, %zu MB free\n",
           total_mem >> 20, free_mem >> 20);
    printf("  L2 Cache   : %zu KB\n", l2_bytes >> 10);
    printf("  SM Clock   : %d MHz (%.3f GHz)\n",
           prop.clockRate / 1000, clk_ghz);

    /* --- allocation --- */
    size_t alloc_bytes = alloc_mb << 20;
    if (alloc_bytes > free_mem * 9 / 10) {
        alloc_bytes = (free_mem * 9 / 10) & ~0xFFFULL;
        printf("  [capped allocation to %zu MB (90%% of free)]\n",
               alloc_bytes >> 20);
    }
    if (alloc_bytes < l2_bytes * 4) {
        alloc_bytes = l2_bytes * 4;
        printf("  [raised allocation to %zu MB (4x L2)]\n",
               alloc_bytes >> 20);
    }
    printf("  Buffer     : %zu MB\n\n", alloc_bytes >> 20);

    char *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, alloc_bytes));
    CUDA_CHECK(cudaMemset(d_buf, 0, alloc_bytes));

    uint64_t *d_cycles;
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(uint64_t)));

    /* optional CSV */
    FILE *csv = csv_path ? fopen(csv_path, "w") : nullptr;
    if (csv_path && !csv) { perror(csv_path); return 1; }

    /* ================================================================ */
    /*  Phase 1 – Stride P-chase                                        */
    /* ================================================================ */

    printf("=== Phase 1: Stride P-chase ===\n\n");
    printf("  Pointer-chase latency vs stride.  Strides whose working\n"
           "  set exceeds L2 measure raw DRAM latency; those marked\n"
           "  [L2] fit in cache and measure L2 latency instead.\n\n");

    printf("%-12s  %-10s  %-13s  %-12s  %-12s  %s\n",
           "Stride", "#Elem", "Cycles/step", "ns/step", "WorkingSet", "");
    printf("%-12s  %-10s  %-13s  %-12s  %-12s  %s\n",
           "------", "-----", "-----------", "-------", "----------", "");

    struct StrideResult { size_t stride; double cycles; bool dram; };
    std::vector<StrideResult> results;

    /* stride sequence: powers of two with 1.5x midpoints */
    std::vector<size_t> strides;
    for (size_t s = stride_min; s <= stride_max; s *= 2) {
        if (s >= 8) strides.push_back(s);
        size_t mid = s + s / 2;
        mid = (mid + 127) & ~127ULL;
        if (mid > s && mid < s * 2 && mid <= stride_max && mid >= 8)
            strides.push_back(mid);
    }
    std::sort(strides.begin(), strides.end());
    strides.erase(std::unique(strides.begin(), strides.end()), strides.end());

    if (csv) fprintf(csv, "phase,stride_bytes,n_elements,cycles_per_step,"
                          "ns_per_step,working_set_bytes,is_dram\n");

    for (size_t stride : strides) {
        if (stride > alloc_bytes / 2) break;
        uint64_t n_elem = alloc_bytes / stride;
        if (n_elem < 2) continue;

        size_t ws = n_elem * 128; /* approximate working set */
        bool is_dram = ws > l2_bytes;

        /* init chase on device */
        int thr = 256;
        int blk = (int)std::min((uint64_t)512, (n_elem + thr - 1) / thr);
        init_pchase<<<blk, thr>>>(d_buf, stride, n_elem);
        CUDA_CHECK(cudaDeviceSynchronize());

        double avg_cyc = 0;
        for (int it = 0; it < iters; it++) {
            CUDA_CHECK(cudaMemset(d_cycles, 0, sizeof(uint64_t)));
            stride_pchase<<<1, 1>>>(d_buf, steps, warmup, d_cycles);
            CUDA_CHECK(cudaDeviceSynchronize());

            uint64_t cyc;
            CUDA_CHECK(cudaMemcpy(&cyc, d_cycles, sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost));
            avg_cyc += (double)cyc / steps;
            if (verbose)
                printf("    [iter %d/%d: %.1f cyc/step]\n",
                       it + 1, iters, (double)cyc / steps);
        }
        avg_cyc /= iters;
        double ns = avg_cyc / clk_ghz;

        char s1[32], s2[32];
        printf("%-12s  %-10llu  %11.1f    %10.1f  %-12s  %s\n",
               fmt_bytes(stride, s1, sizeof(s1)),
               (unsigned long long)n_elem,
               avg_cyc, ns,
               fmt_bytes(ws, s2, sizeof(s2)),
               is_dram ? "" : "[L2]");

        results.push_back({stride, avg_cyc, is_dram});

        if (csv)
            fprintf(csv, "stride,%zu,%llu,%.2f,%.2f,%zu,%d\n",
                    stride, (unsigned long long)n_elem, avg_cyc, ns,
                    ws, is_dram ? 1 : 0);
    }

    /* ---------- Phase 1 analysis ---------- */

    printf("\n=== Phase 1 Analysis ===\n\n");

    std::vector<StrideResult> dram_res;
    for (auto &r : results)
        if (r.dram) dram_res.push_back(r);

    if (dram_res.size() < 3) {
        printf("  Too few DRAM-level measurements.  Increase --alloc.\n");
    } else {
        double lo = 1e18, hi = 0;
        size_t lo_s = 0, hi_s = 0;
        for (auto &r : dram_res) {
            if (r.cycles < lo) { lo = r.cycles; lo_s = r.stride; }
            if (r.cycles > hi) { hi = r.cycles; hi_s = r.stride; }
        }
        char b1[32], b2[32];
        printf("  DRAM latency range : %.1f – %.1f cycles  (%.1f – %.1f ns)\n",
               lo, hi, lo / clk_ghz, hi / clk_ghz);
        printf("  Min @ stride %-10s  Max @ stride %s\n",
               fmt_bytes(lo_s, b1, sizeof(b1)),
               fmt_bytes(hi_s, b2, sizeof(b2)));
        printf("  Conflict ratio     : %.2fx\n\n", hi / lo);

        double thr = lo + 0.3 * (hi - lo);
        printf("  Conflict threshold (30%% of range): %.1f cycles\n", thr);
        printf("  Strides showing row-buffer conflicts:\n");
        for (auto &r : dram_res) {
            if (r.cycles > thr) {
                char bb[32];
                printf("    %-12s  %.1f cyc  (%.1f ns)\n",
                       fmt_bytes(r.stride, bb, sizeof(bb)),
                       r.cycles, r.cycles / clk_ghz);
            }
        }

        size_t first_c = 0, last_c = 0;
        for (auto &r : dram_res) {
            if (r.cycles > thr) {
                if (!first_c) first_c = r.stride;
                last_c = r.stride;
            }
        }
        if (first_c) {
            printf("\n  Estimated DRAM parameters (heuristic):\n");
            char c1[32], c2[32];
            printf("    First conflict stride : %s\n",
                   fmt_bytes(first_c, c1, sizeof(c1)));
            printf("    Last  conflict stride : %s\n",
                   fmt_bytes(last_c, c2, sizeof(c2)));
            if (last_c > first_c) {
                printf("    → Approx bank count  : ~%zu\n",
                       last_c / first_c);
                printf("    → Approx row size    : %s\n",
                       fmt_bytes(last_c, c2, sizeof(c2)));
            }
            printf("\n  NOTE: exact values depend on the address-mapping\n"
                   "  scheme (XOR hashing, channel interleaving, etc.).\n"
                   "  Use --pair-test for detailed per-address probing.\n");
        }
    }

    /* ================================================================ */
    /*  Phase 2 – Pair-wise conflict test                               */
    /* ================================================================ */

    if (pair_test) {
        printf("\n=== Phase 2: Pair-wise Conflict Test ===\n\n");

        /* flush buffer: 1.5× L2 */
        size_t flush_bytes = l2_bytes * 3 / 2;
        if (flush_bytes > free_mem / 4) flush_bytes = l2_bytes;

        char *d_flush;
        CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));
        CUDA_CHECK(cudaMemset(d_flush, 0xAA, flush_bytes));
        uint32_t flush_lines = (uint32_t)(flush_bytes / 128);

        size_t pr_bytes  = pair_range << 20;
        if (pr_bytes > alloc_bytes - pair_stride)
            pr_bytes = alloc_bytes - pair_stride;

        size_t n_tests = pr_bytes / pair_stride;
        if (n_tests > 100000) n_tests = 100000;

        printf("  Reference  : offset 0\n");
        printf("  Scan range : %zu MB   stride : %zu B   tests : %zu\n",
               pr_bytes >> 20, pair_stride, n_tests);
        printf("  Flush buf  : %zu MB   iters  : %d\n\n",
               flush_bytes >> 20, pair_iters);

        std::vector<uint64_t> h_toffs(n_tests);
        for (size_t i = 0; i < n_tests; i++)
            h_toffs[i] = (i + 1) * pair_stride;

        uint64_t *d_toffs;
        uint32_t *d_lat;
        CUDA_CHECK(cudaMalloc(&d_toffs, n_tests * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_lat,   n_tests * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_toffs, h_toffs.data(),
                              n_tests * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));

        const int batch = 128;
        int n_batch = (int)((n_tests + batch - 1) / batch);
        printf("  Processing %d batches ", n_batch);
        fflush(stdout);

        for (size_t off = 0; off < n_tests; off += batch) {
            int cnt = (int)std::min((size_t)batch, n_tests - off);
            pair_conflict_kernel<<<1, 1>>>(
                d_buf, d_flush, flush_lines,
                0ULL,              /* ref at start of buffer */
                d_toffs + off,
                d_lat   + off,
                cnt, pair_iters);
            CUDA_CHECK(cudaDeviceSynchronize());
            if (verbose)
                printf("  batch %zu/%d\n", off / batch + 1, n_batch);
            else {
                putchar('.');
                fflush(stdout);
            }
        }
        printf(" done.\n\n");

        std::vector<uint32_t> h_lat(n_tests);
        CUDA_CHECK(cudaMemcpy(h_lat.data(), d_lat,
                              n_tests * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        uint32_t lat_lo = *std::min_element(h_lat.begin(), h_lat.end());
        uint32_t lat_hi = *std::max_element(h_lat.begin(), h_lat.end());
        uint32_t cthresh = lat_lo + (lat_hi - lat_lo) / 2;

        printf("  %-14s  %-10s  %-10s  %s\n",
               "Offset", "Cycles", "ns", "Conflict?");
        printf("  %-14s  %-10s  %-10s  %s\n",
               "------", "------", "--", "---------");

        size_t conf_cnt = 0;
        size_t first_conf = 0;

        if (csv) fprintf(csv, "\nphase,offset_bytes,cycles,ns,is_conflict\n");

        for (size_t i = 0; i < n_tests; i++) {
            uint64_t ob = h_toffs[i];
            bool cf = h_lat[i] > cthresh;
            if (cf) {
                conf_cnt++;
                if (!first_conf) first_conf = ob;
            }

            bool show = (n_tests <= 200) || (i < 80) ||
                        (i >= n_tests - 20) || cf;
            if (show) {
                char bb[32];
                printf("  %-14s  %-10u  %8.1f  %s\n",
                       fmt_bytes(ob, bb, sizeof(bb)),
                       h_lat[i], h_lat[i] / clk_ghz,
                       cf ? "***" : "");
            } else if (i == 80) {
                printf("  … (%zu middle non-conflict entries hidden) …\n",
                       n_tests - 100);
            }

            if (csv)
                fprintf(csv, "pair,%llu,%u,%.2f,%d\n",
                        (unsigned long long)ob, h_lat[i],
                        h_lat[i] / clk_ghz, cf ? 1 : 0);
        }

        printf("\n  Pair-test summary:\n");
        printf("    Latency range : %u – %u cycles  (%.1f – %.1f ns)\n",
               lat_lo, lat_hi, lat_lo / clk_ghz, lat_hi / clk_ghz);
        printf("    Threshold     : %u cycles\n", cthresh);
        printf("    Conflicts     : %zu / %zu (%.1f%%)\n",
               conf_cnt, n_tests, 100.0 * conf_cnt / n_tests);

        if (conf_cnt > 0 && conf_cnt < n_tests) {
            double frac = (double)conf_cnt / n_tests;
            printf("    Same-bank frac: %.4f  → ~%d banks (1 / fraction)\n",
                   frac, (int)(1.0 / frac + 0.5));
        }
        if (first_conf) {
            char bb[32];
            printf("    First conflict: offset %s\n",
                   fmt_bytes(first_conf, bb, sizeof(bb)));
        }

        /* ---- build conflict set & row set ---- */
        if (conf_cnt > 1) {
            printf("\n  Conflict-set size: %zu addresses\n", conf_cnt);

            /* row-set: keep only one address per row.
             * Two conflict-set addresses that do NOT conflict with each
             * other are in the same row.  Here we approximate by looking
             * at consecutive conflict addresses: if the gap between two
             * conflict addresses is small (< 2× pair_stride), they may
             * be in the same row; large gaps indicate different rows.
             */
            std::vector<uint64_t> conflict_addrs;
            for (size_t i = 0; i < n_tests; i++)
                if (h_lat[i] > cthresh) conflict_addrs.push_back(h_toffs[i]);

            size_t row_count = 1;
            for (size_t i = 1; i < conflict_addrs.size(); i++) {
                uint64_t gap = conflict_addrs[i] - conflict_addrs[i - 1];
                if (gap > pair_stride * 2)
                    row_count++;
            }
            printf("  Approx distinct rows in scan range: %zu\n", row_count);
            if (row_count > 1) {
                size_t row_est = (conflict_addrs.back() - conflict_addrs[0])
                                 / row_count;
                char bb[32];
                printf("  Approx row size (rough): %s\n",
                       fmt_bytes(row_est, bb, sizeof(bb)));
            }
        }

        CUDA_CHECK(cudaFree(d_flush));
        CUDA_CHECK(cudaFree(d_toffs));
        CUDA_CHECK(cudaFree(d_lat));
    }

    /* --- cleanup --- */
    if (csv) fclose(csv);
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_cycles));
    printf("\nDone.\n");
    return 0;
}
