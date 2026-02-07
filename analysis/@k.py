#!/usr/bin/env python3
"""
Aggregate @k metrics from a fixed set of `data.csv` files and write
overall + per-prompt summaries as CSVs.

The input files are enumerated in `CSV_PATHS` below.  Adjust the list
if new benchmarks are added.

Outputs
-------
summaries/
    overall_summary.csv
    per_prompt_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


# --------------------------------------------------------------------------- #
# Input locations
# --------------------------------------------------------------------------- #
CSV_PATHS = [
     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/fft/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/futures_promises/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/geometry/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/graph/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/histogram/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/la/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/reduce/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/scan/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/search/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/sort/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/stencil/driver/tcmalloc/data.csv",
     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/transform/driver/tcmalloc/data.csv"
]


OUTDIR = Path("summaries")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def ensure_paths(paths: Iterable[str]) -> List[Path]:
    """Validate that all CSV paths exist and return them as Path objects."""
    csv_paths: List[Path] = []
    missing = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            csv_paths.append(path)
        else:
            missing.append(path)
    if missing:
        missing_str = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(f"The following CSV files were not found:\n  {missing_str}")
    return csv_paths


def load_and_concat(csv_paths: Iterable[Path]) -> pd.DataFrame:
    """Load all CSVs and concatenate them, keeping track of the source file."""
    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df["__source__"] = csv_path.as_posix()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def metric_columns(df: pd.DataFrame) -> List[str]:
    """Return ordered list of @k metric columns (excluding thread-scale cols)."""
    ks = [1, 5, 10, 20]
    prefixes = [
        "build",
        "pass",
        "strict_pass",
        "speedup",
        "speedup_max",
        "efficiency",
        "efficiency_max",
    ]
    cols = []
    for prefix in prefixes:
        for k in ks:
            col = f"{prefix}@{k}"
            if col in df.columns:
                cols.append(col)
    return cols


# --------------------------------------------------------------------------- #
# Main reporting logic
# --------------------------------------------------------------------------- #
def main() -> None:
    csv_paths = ensure_paths(CSV_PATHS)
    data = load_and_concat(csv_paths)

    required_cols = {"model", "problem type"}
    missing_required = required_cols.difference(data.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_required))}")

    metric_cols = metric_columns(data)
    if not metric_cols:
        raise ValueError("No @k metric columns found in the data.")

    OUTDIR = "/work/pi_mrobson_smith_edu/ParEval_amt/analysis/@k"

    overall = (
        data.groupby("model", sort=True)[metric_cols]
        .mean()
        .reset_index()
        .sort_values("model")
    )

    per_prompt = (
        data.groupby(["problem type", "model"], sort=True)[metric_cols]
        .mean()
        .reset_index()
        .sort_values(["problem type", "model"])
    )

    overall_path = OUTDIR + "/overall_summary.csv"
    per_prompt_path = OUTDIR + "/per_prompt_summary.csv"

    overall.to_csv(overall_path, index=False)
    per_prompt.to_csv(per_prompt_path, index=False)

    print(f"Wrote overall summary to {overall_path}")
    print(f"Wrote per-prompt summary to {per_prompt_path}")


if __name__ == "__main__":
    main()