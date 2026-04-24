# ##############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################

"""
Inspect a raw rocCV benchmark CSV for noise and outliers, and optionally
collapse it to a per-configuration summary CSV with outliers removed.

Each row in the input CSV is a single timed run. This script groups rows that
share the same parameters and prints per-group statistics intended to help
decide on an outlier-rejection rule. With --export, it also writes a CSV with
one row per group, holding aggregate statistics over the kept samples.
"""

import argparse
import sys
import pandas as pd

OUTLIER_RULES = ("tukey-upper", "tukey", "sigma", "none")

# Columns describing the host/device — constant across every row, lifted to a
# header block instead of being repeated per group.
METADATA_COLS = ["gpu", "cpu", "cpu_threads"]

# Columns that vary within a single configuration's repeated runs, or are
# derived facts about a configuration rather than identifying it. Excluded
# from the group key.
SAMPLE_COLS = ["run_index", "execution_time", "read_memory_bytes", "written_memory_bytes"]


def load(path: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    metadata = {c: df[c].iloc[0] for c in METADATA_COLS if c in df.columns}
    df = df.drop(columns=[c for c in METADATA_COLS if c in df.columns])
    return df, metadata


def category_view(df: pd.DataFrame, category: str) -> pd.DataFrame:
    # Drop columns that are entirely empty for this category so each operator's
    # parameter schema is dense and readable.
    sub = df[df["category"] == category].copy()
    return sub.dropna(axis=1, how="all")


def group_key_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in SAMPLE_COLS]


def group_stats(times: pd.Series) -> dict:
    n = len(times)
    mean = times.mean()
    median = times.median()
    std = times.std(ddof=1) if n > 1 else 0.0
    mn = times.min()
    mx = times.max()
    cv = std / mean if mean > 0 else float("nan")
    range_over_median = (mx - mn) / median if median > 0 else float("nan")

    sigma_outliers = int(((times - mean).abs() > 2 * std).sum()) if std > 0 else 0

    q1 = times.quantile(0.25)
    q3 = times.quantile(0.75)
    iqr = q3 - q1
    iqr_outliers = int(((times < q1 - 1.5 * iqr) | (times > q3 + 1.5 * iqr)).sum())

    return {
        "n": n,
        "mean": mean,
        "median": median,
        "std": std,
        "cv": cv,
        "min": mn,
        "max": mx,
        "range/med": range_over_median,
        ">2σ": sigma_outliers,
        ">1.5·IQR": iqr_outliers,
    }


def summarize_category(sub: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    keys = group_key_columns(sub)
    grouped = sub.groupby(keys, dropna=False, sort=False)
    rows = []
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key_tuple))
        row.update(group_stats(group["execution_time"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["name", "samples"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out, keys


def format_summary(summary: pd.DataFrame) -> str:
    # Convert seconds → milliseconds for readability; CV / range_over_median
    # are dimensionless ratios printed as percentages.
    ms = lambda x: f"{x * 1e3:10.4f}"
    pct = lambda x: f"{x * 100:7.2f}%"
    formatters = {
        "mean": ms,
        "median": ms,
        "std": ms,
        "min": ms,
        "max": ms,
        "cv": pct,
        "range/med": pct,
    }
    return summary.to_string(index=False, formatters=formatters, na_rep="")


def filter_outliers(times: pd.Series, rule: str) -> pd.Series:
    """Return the subset of `times` retained under the chosen outlier rule.

    Rules:
      tukey-upper — drop samples above Q3 + 1.5·IQR (one-sided; fast samples
                    aren't suspicious in a benchmark context).
      tukey       — drop samples outside [Q1 - 1.5·IQR, Q3 + 1.5·IQR].
      sigma       — drop samples beyond mean ± 2·std.
      none        — keep everything.
    """
    if rule == "none" or len(times) < 3:
        return times
    if rule == "sigma":
        std = times.std(ddof=1)
        if std == 0:
            return times
        mean = times.mean()
        return times[(times - mean).abs() <= 2 * std]
    q1 = times.quantile(0.25)
    q3 = times.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    if rule == "tukey-upper":
        return times[times <= upper]
    # tukey (two-sided)
    lower = q1 - 1.5 * iqr
    return times[(times >= lower) & (times <= upper)]


def collapsed_row(group: pd.DataFrame, key_columns: list[str], rule: str) -> dict:
    """Collapse a group of repeated runs into one summary row."""
    times = group["execution_time"]
    kept = filter_outliers(times, rule)
    n_total = len(times)
    n_kept = len(kept)

    row: dict = {col: group[col].iloc[0] for col in key_columns}
    row["n_total"] = n_total
    row["n_kept"] = n_kept
    row["n_dropped"] = n_total - n_kept

    if n_kept == 0:
        for k in ("mean", "median", "std", "q1", "q3", "min", "max"):
            row[k] = float("nan")
    else:
        row["mean"] = kept.mean()
        row["median"] = kept.median()
        row["std"] = kept.std(ddof=1) if n_kept > 1 else 0.0
        row["q1"] = kept.quantile(0.25)
        row["q3"] = kept.quantile(0.75)
        row["min"] = kept.min()
        row["max"] = kept.max()

    # Memory bytes are constant per group; carry them through unchanged.
    for col in ("read_memory_bytes", "written_memory_bytes"):
        if col in group.columns:
            row[col] = group[col].iloc[0]

    return row


def export_cleaned(df: pd.DataFrame, output_path: str, rule: str) -> tuple[int, int]:
    """Write a collapsed CSV. Returns (n_groups, n_total_dropped)."""
    rows: list[dict] = []
    total_dropped = 0
    for cat in sorted(df["category"].dropna().unique()):
        sub = category_view(df, cat)
        keys = group_key_columns(sub)
        for _, group in sub.groupby(keys, dropna=False, sort=False):
            row = collapsed_row(group, keys, rule)
            total_dropped += row["n_dropped"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return len(rows), total_dropped


def print_raw_runs(sub: pd.DataFrame, keys: list[str]) -> None:
    for key, group in sub.groupby(keys, dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        header = ", ".join(f"{k}={v}" for k, v in zip(keys, key_tuple))
        print(f"\n  Group: {header}")
        for _, row in group.sort_values("run_index").iterrows():
            print(f"    run {int(row['run_index']):3d}: {row['execution_time'] * 1e3:.4f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Path to a roccv_bench CSV results file.")
    parser.add_argument("--category", help="Restrict output to a single category.")
    parser.add_argument(
        "--show-runs",
        action="store_true",
        help="After each summary table, dump every per-run sample for the same groups.",
    )
    parser.add_argument(
        "--export",
        metavar="PATH",
        help="Write a collapsed CSV (one row per group) with outliers removed.",
    )
    parser.add_argument(
        "--outlier-rule",
        choices=OUTLIER_RULES,
        default="tukey-upper",
        help="Outlier rejection rule for --export (default: tukey-upper).",
    )
    args = parser.parse_args()

    df, metadata = load(args.csv)

    print("=== Metadata ===")
    for k, v in metadata.items():
        print(f"  {k}: {v}")
    print()

    categories = sorted(df["category"].dropna().unique())
    if args.category:
        if args.category not in categories:
            print(
                f"Category '{args.category}' not found. Available: {categories}",
                file=sys.stderr,
            )
            return 1
        categories = [args.category]

    for cat in categories:
        sub = category_view(df, cat)
        summary, keys = summarize_category(sub)
        print(f"=== {cat} ===")
        print(format_summary(summary))
        if args.show_runs:
            print_raw_runs(sub, keys)
        print()

    if args.export:
        n_groups, n_dropped = export_cleaned(df, args.export, args.outlier_rule)
        print(
            f"Exported {n_groups} groups to {args.export} "
            f"(rule={args.outlier_rule}, samples dropped={n_dropped})."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
