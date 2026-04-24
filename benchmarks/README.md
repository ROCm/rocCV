# rocCV Benchmarking Suite

The rocCV benchmark suite measures operator performance on CPU and GPU. It produces a CSV (or JSON) of raw per-run samples that downstream tooling can aggregate, filter, or compare. A companion Python script (`analyze_results.py`) inspects the raw output for noise and exports a cleaned, per-configuration summary CSV.

This document is split into two parts:

- **[How to Run](#how-to-run)** — practical commands for building, executing, and analyzing benchmarks.
- **[Methodology](#methodology)** — what the suite measures, how it measures it, and the conventions you should follow when collecting comparable results.

---

## How to Run

### Dependencies

OpenCV development libraries are required for the OpenCV reference benchmarks. They are not needed for rocCV-only benchmarks.

```shell
apt install libopencv-dev
```

The analysis script requires a recent Python with `pandas`.

### Building

From the rocCV project root:

```bash
mkdir build && cd build
cmake -DBENCHMARKS=ON ..
make -j$(nproc)
```

This produces `build/bin/roccv_bench`.

### Running Benchmarks

From the build directory, point at the default configuration:

```bash
./bin/roccv_bench --config ../benchmarks/config.json
```

By default this writes `roccv_bench_results.json`. Use `--output` to change the path; `.csv` or `.json` extensions are both supported (CSV is recommended — `analyze_results.py` operates on CSV directly).

```bash
./bin/roccv_bench --config ../benchmarks/config.json --output results.csv
```

Useful options (see `--help` for the full list):

| flag | purpose |
|---|---|
| `--list` / `-l` | List all benchmark categories and exit. |
| `--select Cat1,Cat2` / `-s` | Run only the listed categories. |
| `--exclude Cat1,Cat2` / `-e` | Run everything except the listed categories. |
| `--types CPU,GPU` / `-t` | Restrict to specific benchmark types within each category. |

**Output shape.** Each row in the CSV is one timed iteration. A configuration with `runs: 25` yields 25 rows (one per timed run, ordered via the `run_index` column), so a downstream tool can compute mean, median, percentiles, or apply outlier filtering without losing information.

### Analyzing Results

`benchmarks/analyze_results.py` reads the raw CSV and serves two purposes:

**1. Inspection** — print per-group statistics so you can see how noisy each configuration was.

```bash
python3 benchmarks/analyze_results.py build/results.csv
python3 benchmarks/analyze_results.py build/results.csv --category Resize
python3 benchmarks/analyze_results.py build/results.csv --category Resize --show-runs
```

For each group of repeated runs, the table reports `n`, `mean`, `median`, `std`, `cv` (std/mean), `min`, `max`, `range/med`, and the count of samples each outlier rule would flag (`>2σ` and `>1.5·IQR`). `--show-runs` additionally dumps every per-run timing in execution order — useful for spotting drift.

**2. Export** — collapse to one row per configuration with outliers removed.

```bash
python3 benchmarks/analyze_results.py build/results.csv --export build/results_clean.csv
python3 benchmarks/analyze_results.py build/results.csv --export build/results_clean.csv --outlier-rule tukey
```

The export schema adds `n_total`, `n_kept`, `n_dropped`, `mean`, `median`, `std`, `q1`, `q3`, `min`, `max` over the kept samples. Outlier rules (`--outlier-rule`):

| rule | description |
|---|---|
| `tukey-upper` (default) | Drop samples above `Q3 + 1.5·IQR`. One-sided; suspiciously fast samples are kept. |
| `tukey` | Two-sided Tukey fences (drop both tails). |
| `sigma` | Drop samples beyond `mean ± 2·std`. Assumes roughly normal distribution. |
| `none` | No filtering; useful as a baseline or when filtering downstream. |

See the [Methodology](#methodology) section below for guidance on which rule fits what data.

### Configuring Benchmarks

Configurations live in a JSON file. The default is `benchmarks/config.json`:

```json
{
    "params": [
        {
            "samples": 16,
            "height": 1080,
            "width": 1920,
            "runs": 25,
            "warmup_runs": 10
        }
    ]
}
```

Each entry specifies one shape × batch configuration; every selected benchmark runs against every entry. Field meanings:

- `samples`, `height`, `width` — input batch / image shape.
- `runs` — number of timed iterations recorded per configuration.
- `warmup_runs` — number of additional iterations executed first and discarded (defaults to 5). Each iteration touches caches, the GPU command queue, and the driver state, so the first few are typically slower than steady-state.

Recommended starting point: `runs: 25, warmup_runs: 10`. For sub-100µs operators or when fine-grained statistical analysis matters, bump `runs` to 50 or higher to give the outlier rules enough samples to work with.

---

## Methodology

This section describes how the suite times operations and why the surrounding conventions exist. Read this before interpreting results, especially when comparing collected data across days, machines, or driver versions.

### Per-Run Sample Export

The benchmark binary does **not** aggregate timings at collection time. Each timed iteration is emitted as its own row in the output CSV, identified by `run_index` (0-based, in execution order) within an otherwise-identical group of parameters.

Rationale: aggregation throws away information. With raw samples in hand, downstream tools can compute any statistic they want (median, p95, trimmed mean), apply outlier-rejection rules, or compare run-to-run drift — none of which are possible from a pre-collapsed mean. The price is a larger output file, which the analysis script collapses into a per-configuration summary on demand.

### Timing Strategy

Two timer specializations live in `benchmarks/roccvbench/include/roccvbench/utils.hpp`:

- **CPU benchmarks** (rocCV CPU implementations and OpenCV reference benchmarks) use `std::chrono::steady_clock` around the operator call. `steady_clock` is monotonic and unaffected by wall-clock adjustments such as NTP.
- **GPU benchmarks** use HIP events (`hipEventRecord` / `hipEventSynchronize` / `hipEventElapsedTime`). The recorded interval is the kernel's **device-side execution time**, exclusive of host scheduling, command-queue wait, and stream synchronization overhead.

The two strategies measure different things and the numbers should not be compared as if they were on equal footing — see [Comparing CPU and GPU Numbers](#comparing-cpu-and-gpu-numbers).

### Warmup Runs

The first `warmup_runs` iterations are discarded. This protects against:

- Cold instruction caches and texture caches.
- Lazy driver initialization of the command queue.
- One-off kernel JIT or constant-buffer setup.
- GPU clocks ramping from idle to active state.

Warmup samples are real benchmark calls with full timing — they are simply not recorded in the output. Five iterations is the default; ten is safer for short kernels where startup costs are a larger fraction of the measured interval.

### The Analysis Pipeline

The intended workflow for a serious comparison run is three steps:

1. **Collect** raw samples by running `roccv_bench`. Output is one row per timed iteration.
2. **Inspect** the raw CSV with `analyze_results.py` (no flags). Look at `cv` and `range/med` per group to see how noisy each configuration was. Drill into specific groups with `--category X --show-runs` to look at the per-iteration sequence (drift, single-spike outliers, bimodal patterns).
3. **Export** a collapsed CSV with `--export`, choosing an outlier rule based on what step 2 revealed.

Skipping the inspection step risks applying a filter that doesn't fit the data. The two outlier rules in the inspection table (`>2σ` and `>1.5·IQR`) are intentionally both shown — when they disagree, the data isn't normally distributed and `tukey-upper` is the safer choice.

### Outlier Rejection Rules

Brief guidance on choosing `--outlier-rule` for the export step:

- **`tukey-upper` (default)** — drops samples above `Q3 + 1.5·IQR`. One-sided because in benchmark contexts a suspiciously fast sample is typically real (the kernel did execute that fast at least once), while a suspiciously slow sample is almost always contamination (preemption, page fault, transient throttle). Robust to skewed distributions; doesn't assume normality.
- **`tukey`** — adds the lower fence. Use only if you suspect timing artifacts that produce too-small numbers (e.g., a clock-source bug). Not the default because it can discard legitimate best-case timings.
- **`sigma`** — classic mean ± 2σ. Convenient but has a known weakness: a single outlier inflates the std, which widens the fence, which lets the outlier escape. Acceptable when the distribution is genuinely normal; risky as a default for benchmark data.
- **`none`** — no filtering. Useful as a baseline against which to diff the filtered export.

### Reducing Run-to-Run Variance

For comparable results across collection sessions, fix as many sources of variation as possible before running:

- **Lock GPU clocks.** With dynamic power management (DPM) free to roam, the GPU may spend different fractions of each run in boost clocks vs nominal clocks. Pin to a stable performance state before benchmarking:
  ```bash
  sudo rocm-smi --setperflevel high
  ```
  Restore with `sudo rocm-smi --setperflevel auto` afterwards. Note that `high` typically pins to a stable maximum that may be **slower** than opportunistic boost — locked numbers are better for ranking; unlocked numbers are closer to deployment behavior.
- **Quiet the system.** Background work — display compositors with GPU acceleration, browser tabs, file indexers, container runtimes — can perturb both CPU and GPU benchmarks. Run on an idle machine when possible.
- **Use enough samples.** The Tukey rule needs a stable IQR; with `runs: 10`, dropping one outlier leaves a thin remainder. `runs: 25` is a reasonable lower bound for analysis; `runs: 50` makes filtering robust and shrinks summary statistics' confidence intervals.

### Comparing CPU and GPU Numbers

Direct comparison of a CPU `mean` against a GPU `mean` measures different things:

- The CPU measurement is wall-clock time including everything between the start and end of the operator call.
- The GPU measurement is device-side kernel time only, with host scheduling and synchronization excluded.

For rough order-of-magnitude comparisons this is fine. For "how much faster is the GPU than the CPU in deployed software?" you must include the host-side cost of dispatching and synchronizing the GPU work, which the GPU benchmark deliberately excludes. The right baseline for such a comparison is end-to-end pipeline measurement, not these microbenchmarks.

### Known Limitations

- **Sub-100µs operators have a noise floor.** `hipEventElapsedTime` returns float milliseconds with roughly 0.5µs resolution, which is a meaningful fraction of a 10-50µs kernel. Even with locked clocks and 50 runs, expect coefficient-of-variation around 1-5% for these. The most reliable mitigation is to repeat the kernel call inside the timed window so the measured interval is well above the timer floor; this is not currently exposed but is on the roadmap.
- **GPU command-queue scheduling** introduces additional run-to-run variance for short kernels — not just within a session but between collection sessions — that no host-side change can suppress.
- **Random tensor data** is generated with a fixed seed (`roccvbench::kBenchSeed`), so input bytes are identical across collection sessions. This eliminates data-dependent operators (e.g., conditional thresholding) as a source of run-to-run variance. Change the seed only if you intentionally want to vary inputs.
