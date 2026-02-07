#!/usr/bin/env python3
"""
Aggregate model-level telemetry metrics from multiple generation HPX CSV exports.it is AVERAGE AMONG PROMTPS (NOT NECESSARILYK NORMALIZED BY NUMBER OF OUTPUTS) But since metrics have 100 outputs per propmt, i would say it evens out

Produces two CSV reports:

1. model_metrics_by_category.csv
      Columns: category, model, avg_generation_time, avg_gpu_utilization, avg_memory_utilization
      Each row is the mean of all rows for (category, model).

2. model_metrics_overall.csv
      Columns: model, avg_generation_time, avg_gpu_utilization, avg_memory_utilization
      Each row is the mean across every category for that model.

Both files are written to OUTPUT_ROOT.
"""

import os
import pandas as pd

CSV_PATHS = [
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/fft/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/graph/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/transform/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/scan/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/search/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/sort/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/stencil/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/la/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/reduce/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/histogram/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/geometry/driver/tcmalloc/json_runtime_summary.csv",
]

OUTPUT_ROOT = "/work/pi_mrobson_smith_edu/ParEval_amt/analysis/visuals_specific"
METRIC_COLS = ["avg_generation_time",  "avg_gpu_memory_utilization", "avg_gpu_utilization", "avg_virtual_memory_used"]


def get_category_from_path(path: str) -> str:
    parts = path.split(os.sep)
    try:
        idx = parts.index("generation_hpx")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return "unknown"


def main() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    per_category_records = []

    for csv_path in CSV_PATHS:
        if not os.path.isfile(csv_path):
            print(f"[skip] missing file: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        if "model" not in df.columns:
            print(f"[warn] 'model' column missing in {csv_path}; skipping.")
            continue

        category = get_category_from_path(csv_path)

        # Ensure all metric columns exist (fill missing with NaN)
        for metric in METRIC_COLS:
            if metric not in df.columns:
                df[metric] = pd.NA

        grouped = (
            df.groupby("model", dropna=False)[METRIC_COLS]
            .mean(numeric_only=True)
            .reset_index()
        )
        grouped.insert(0, "category", category)

        per_category_records.append(grouped)

    if not per_category_records:
        print("[abort] no valid data collected from provided CSV paths.")
        return

    per_category_df = pd.concat(per_category_records, ignore_index=True)

    per_category_out = os.path.join(OUTPUT_ROOT, "model_metrics_by_category.csv")
    per_category_df.to_csv(per_category_out, index=False)
    print(f"[ok] per-category summary written to {per_category_out}")

    overall_df = (
        per_category_df.groupby("model", dropna=False)[METRIC_COLS]
        .mean(numeric_only=True)
        .reset_index()
    )
    overall_out = os.path.join(OUTPUT_ROOT, "model_metrics_overall.csv")
    overall_df.to_csv(overall_out, index=False)
    print(f"[ok] overall summary written to {overall_out}")


if __name__ == "__main__":
    main()