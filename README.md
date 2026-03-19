# GPU-Microarch-Bench

CUDA micro-benchmarks for probing GPU DRAM micro-architecture.

## Benchmarks

### `dram_bank_row` — DRAM Bank & Row Size Discovery

Discovers DRAM bank count and row size by detecting row-buffer conflicts
through latency measurements.  Supports **Ampere** (sm_80) and **Hopper**
(sm_90) GPUs.

**How it works:**

| Phase | Method | What it measures |
|-------|--------|------------------|
| 1 — Stride P-chase | Pointer-chase at different strides through a buffer larger than L2. | Per-step DRAM latency.  Strides that place consecutive accesses in the same bank / different row show elevated latency due to row-buffer conflicts. |
| 2 — Pair conflict test | L2 flush, then time individual `(ref, test)` pairs. | Whether a specific address conflicts with the reference (same bank, different row). |

## Building

Requires CUDA Toolkit (≥ 11.0) and `nvcc`.

```bash
# Ampere (default)
make

# Hopper
make ARCH=hopper

# Fat binary (both architectures)
make ARCH=both
```

## Running

```bash
cd dram_bank_row

# Phase 1 only (fast, ~10 s)
./dram_bank_row

# Phase 1 + Phase 2
./dram_bank_row --pair-test

# Larger allocation for better large-stride coverage
./dram_bank_row --alloc 8192

# Save results to CSV for plotting
./dram_bank_row --pair-test --csv results.csv
```

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--arch ampere\|hopper` | auto-detect | Override GPU architecture |
| `--alloc <MB>` | 4096 | GPU buffer size |
| `--stride-min <B>` | 128 | Minimum P-chase stride |
| `--stride-max <B>` | 4 MB | Maximum P-chase stride |
| `--steps <N>` | 200000 | Timed P-chase steps |
| `--warmup <N>` | 10000 | Warmup steps |
| `--iters <N>` | 3 | Averaging iterations |
| `--pair-test` | off | Enable Phase 2 |
| `--pair-range <MB>` | 32 | Scan range for pair test |
| `--pair-stride <B>` | 256 | Granularity of pair scan |
| `--pair-iters <N>` | 10 | Repetitions per pair measurement |
| `--csv <file>` | — | Write results to CSV |
| `-v` | off | Verbose output |

## Interpreting results

In Phase 1, look for a band of strides with noticeably higher latency.
This indicates row-buffer conflicts (same bank, different row):

- **First conflict stride** ≈ the inter-bank stride (distance between
  addresses mapping to the same bank).
- **Last conflict stride** ≈ the row size.
- **Bank count** ≈ last / first.

Phase 2 gives per-address conflict data.  Addresses with latency above the
automatically chosen threshold are in the same bank as the reference
(offset 0).

> **Note on NUMA effects:** On GPUs with distributed DRAM (large HBM
> configurations), memory access latency varies with physical location.
> This can widen the latency distribution and cause overlap between
> conflict / non-conflict populations.  Phase 1 (stride P-chase) averages
> over many accesses and is more robust to this effect than Phase 2
> (individual pair measurements).
