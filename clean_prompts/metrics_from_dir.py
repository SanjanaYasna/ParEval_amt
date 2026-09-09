#!/usr/bin/env python3
"""
Compute aggregate metrics for every CSV produced by run-all.py in a directory.

For each file named '{model-name}_combined.csv' the script:
  • parses the model name automatically,
  • computes the same build/pass/speedup/efficiency metrics that run-all.py reports,
  • augments the HPX rows with per-thread speedup/efficiency averages at thread
    counts 1, 2, 4, 8, 16, 32, and 64 (columns named
    'speedup_num_threads_<N>@1' and 'efficiency_num_threads_<N>@1'),
  • and writes the union of all models’ results to one CSV (data.csv by default).
"""

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# -----------------------------------------------------------------------------
# argument parsing
# -----------------------------------------------------------------------------
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing '{model-name}_combined.csv' files from run-all.py.",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="K values for pass@k, build@k, speedup@k, efficiency@k (default: 1 5 10 20).",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=1,
        help="N value for speedup@k / efficiency@k (default: 1).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data.csv",
        help="Name of the combined CSV to write (default: data.csv in input_dir).",
    )
    parser.add_argument(
        "--problem-sizes",
        type=str,
        default="../drivers/problem-sizes.json",
        help="JSON with problem sizes used for GPU efficiency calculations.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# helpers shared with the original script
# -----------------------------------------------------------------------------
def has_outputs(prompt: Dict[str, Any]) -> bool:
    outputs = prompt.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return False
    if all(isinstance(o, str) for o in outputs):
        return False
    return all(isinstance(o, dict) for o in outputs)


def check(df: pd.DataFrame) -> None:
    if df.empty:
        return
    agg = df.groupby(["name", "parallelism_model"]).agg({"did_build": "sum"})
    zero = agg[agg["did_build"] == 0]
    if not zero.empty:
        print("Warning: (name, parallelism_model) pairs with zero successful builds:")
        print(zero)


def metric_value(prompt: Dict[str, Any], key: str, idx: int) -> Optional[Any]:
    values = prompt.get(key)
    if values is None:
        return None
    if isinstance(values, list):
        if idx < len(values):
            return values[idx]
        return None
    return values


def get_correctness_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["strict_valid_run"] = df["is_valid"] & df["is_source_valid"]

    agg = (
        df.groupby(["name", "parallelism_model", "problem_type", "output_idx"], as_index=False)
        .agg(
            total_runs=("is_valid", "count"),
            valid_sum=("is_valid", "sum"),
            strict_sum=("strict_valid_run", "sum"),
        )
    )

    agg["is_valid"] = agg["valid_sum"] == agg["total_runs"]
    agg["is_strict_valid"] = agg["strict_sum"] == agg["total_runs"]
    return agg.drop(columns=["total_runs", "valid_sum", "strict_sum"])


def nCr(n: int, r: int) -> int:
    if n < r:
        return 1
    return comb(n, r)


def _passk(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))

def buildk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    agg = df.groupby(["name", "parallelism_model", "problem_type"]).agg(
        {"did_build": ["count", "sum"]}
    )
    agg.columns = ["total_build_attempts", "successful_builds"]
    agg = agg.reset_index()
    agg[f"build@{k}"] = agg.apply(
        lambda x: _passk(x["total_build_attempts"], x["successful_builds"], k), axis=1
    )
    if k == 1:
        print(agg.groupby(["parallelism_model", "problem_type"])[
            ["total_build_attempts", "successful_builds"]
        ].sum())
    return agg.groupby(["parallelism_model", "problem_type"]).agg({f"build@{k}": "mean"})

# # BUILD@K BY THE PAREVAL AUTHORS OVERCOUNTS THE SUCCESSES FOR EACH OF THE RUNS ACROSS THREAD COUNTS. USE BELOW BUILD@K IF YOU WANT RESULTS THAT COUNT SUCCESSFUL RUNS AMONG THREADS AS JUST 1 INSTEAD OF 7  
# def buildk(df: pd.DataFrame, k: int) -> pd.DataFrame:
#     # # Collapse to one row per (name, parallelism_model, problem_type, output_idx) to avoid recounting successful builds from ultiple thraed runs
#     build_df = (
#         df.groupby(["name", "parallelism_model", "problem_type", "output_idx"])
#         .agg(
#             total_runs=("did_build", "count"),
#             build_sum=("did_build", "sum"),
#         )
#     )
#     build_df["did_build"] = build_df["build_sum"] == build_df["total_runs"]

#     # Now aggregate across output_idx to get per-(name, parallelism_model, problem_type) stats
#     agg = build_df.groupby(["name", "parallelism_model", "problem_type"]).agg(
#         {"did_build": ["count", "sum"]}
#     )
#     agg.columns = ["total_build_attempts", "successful_builds"]
#     agg = agg.reset_index()
    
#     # Apply pass@k formula
#     agg[f"build@{k}"] = agg.apply(
#         lambda x: _passk(x["total_build_attempts"], x["successful_builds"], k), axis=1
#     )
    
#     # Optional: print build stats when k=1 for debugging
#     if k == 1:
#         print(agg.groupby(["parallelism_model", "problem_type"])[
#             ["total_build_attempts", "successful_builds"]
#         ].sum())
    
#     return agg.groupby(["parallelism_model", "problem_type"]).agg({f"build@{k}": "mean"})


def passk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    agg = df.groupby(["name", "parallelism_model", "problem_type"]).agg(
        {"is_valid": ["count", "sum"]}
    )
    agg.columns = ["total_runs", "valid_count"]
    agg = agg.reset_index()
    agg[f"pass@{k}"] = agg.apply(
        lambda x: _passk(x["total_runs"], x["valid_count"], k), axis=1
    )
    return agg.groupby(["parallelism_model", "problem_type"]).agg({f"pass@{k}": "mean"})


def passk_strict(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df[df["parallelism_model"] == "hpx"].copy()
    if df.empty:
        empty_index = pd.MultiIndex(levels=[[], []], codes=[[], []],
                                    names=["parallelism_model", "problem_type"])
        return pd.DataFrame(index=empty_index, columns=[f"strict_pass@{k}"])
    agg = df.groupby(["name", "parallelism_model", "problem_type"]).agg(
        {"is_strict_valid": ["count", "sum"]}
    )
    agg.columns = ["total_runs", "strict_valid_count"]
    agg = agg.reset_index()
    agg[f"strict_pass@{k}"] = agg.apply(
        lambda x: _passk(x["total_runs"], x["strict_valid_count"], k), axis=1
    )
    return agg.groupby(["parallelism_model", "problem_type"]).agg({f"strict_pass@{k}": "mean"})


def _speedupk(
    runtimes: Union[pd.Series, np.ndarray],
    baseline_runtime: float,
    k: int,
    col_name: str = "speedup@{}",
) -> pd.Series:
    if isinstance(runtimes, pd.Series):
        runtimes = runtimes.values.copy()
    else:
        runtimes = np.array(runtimes, copy=True)

    runtimes.sort()
    total = 0.0
    num_samples = runtimes.shape[0]
    for j in range(1, num_samples + 1):
        num = nCr(j - 1, k - 1) * baseline_runtime
        den = nCr(num_samples, k) * max(runtimes[j - 1], 1e-8)
        total += num / den
    return pd.Series({col_name.format(k): total})


def speedupk(df: pd.DataFrame, k: int, n: int) -> pd.DataFrame:
    df = df.copy()
    df = df[df["is_valid"] == True]

    df = df[
        (df["parallelism_model"].isin(["serial", "cuda", "hip"]))
        | ((df["parallelism_model"] == "hpx") & (df["num_threads"] == 64))
        | ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32))
        | ((df["parallelism_model"] == "omp") & (df["num_threads"] == 32))
    ]

    df["best_sequential_runtime"] = df.groupby(
        ["name", "parallelism_model", "output_idx"]
    )["best_sequential_runtime"].transform("min")

    grouped = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
        lambda row: _speedupk(
            row["runtime"], np.min(row["best_sequential_runtime"]), k
        )[f"speedup@{k}"]
    )
    result = grouped.groupby(level=["parallelism_model", "problem_type"]).mean()
    if isinstance(result, pd.Series):
        result = result.to_frame(name=f"speedup@{k}")
    else:
        col = result.columns[0]
        result = result[[col]].rename(columns={col: f"speedup@{k}"})
    return result


def speedupk_max(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df.copy()
    df = df[df["is_valid"] == True]

    df["runtime"] = df.groupby(
        ["name", "parallelism_model", "output_idx"]
    )["runtime"].transform("min")
    df["best_sequential_runtime"] = df.groupby(
        ["name", "parallelism_model", "output_idx"]
    )["best_sequential_runtime"].transform("min")

    df["run_idx"] = df["run_idx"].astype(int)
    df = df[df["run_idx"] == 0]

    grouped = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
        lambda row: _speedupk(
            row["runtime"],
            np.min(row["best_sequential_runtime"]),
            k,
            col_name="speedup_max@{}",
        )[f"speedup_max@{k}"]
    )
    result = grouped.groupby(level=["parallelism_model", "problem_type"]).mean()
    if isinstance(result, pd.Series):
        result = result.to_frame(name=f"speedup_max@{k}")
    else:
        col = result.columns[0]
        result = result[[col]].rename(columns={col: f"speedup_max@{k}"})
    return result


def _efficiencyk(
    runtimes: Union[pd.Series, np.ndarray],
    baseline_runtime: float,
    k: int,
    n_resources: Union[pd.Series, np.ndarray],
    col_name: str = "efficiency@{}",
) -> pd.Series:
    if isinstance(runtimes, pd.Series):
        runtimes = runtimes.values.copy()
    else:
        runtimes = np.array(runtimes, copy=True)

    if isinstance(n_resources, pd.Series):
        n_resources = n_resources.values.copy()
    else:
        n_resources = np.array(n_resources, copy=True)

    runtimes.sort()
    total = 0.0
    num_samples = runtimes.shape[0]
    for j in range(1, num_samples + 1):
        num = nCr(j - 1, k - 1) * baseline_runtime
        den = nCr(num_samples, k) * max(runtimes[j - 1], 1e-8) * n_resources[j - 1]
        total += num / den
    return pd.Series({col_name.format(k): total})


def efficiencyk(df: pd.DataFrame, k: int, n: int) -> pd.DataFrame:
    df = df.copy()
    df = df[df["is_valid"] == True]

    df = df[
        (df["parallelism_model"].isin(["serial", "cuda", "hip"]))
        | ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32))
        | ((df["parallelism_model"] == "omp") & (df["num_threads"] == 32))
        | ((df["parallelism_model"] == "hpx") & (df["num_threads"] == 64))
    ]

    df["n_resources"] = 1
    df.loc[df["parallelism_model"] == "cuda", "n_resources"] = df["problem_size"]
    df.loc[df["parallelism_model"] == "hip", "n_resources"] = df["problem_size"]
    df.loc[df["parallelism_model"] == "hpx", "n_resources"] = 64
    df.loc[df["parallelism_model"] == "kokkos", "n_resources"] = 32
    df.loc[df["parallelism_model"] == "omp", "n_resources"] = 8
    df.loc[df["parallelism_model"] == "mpi", "n_resources"] = 512
    df.loc[df["parallelism_model"] == "mpi+omp", "n_resources"] = 4 * 64

    # df = df.copy()

    # # use min best_sequential_runtime
    # df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")


    # # # group by name, parallelism_model, and output_idx and call _efficiencyk
    # df = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
    #         lambda row: _efficiencyk(row["runtime"], np.min(row["best_sequential_runtime"]), k, row["n_resources"])
    #     ).reset_index()
    
    # # to avoid duplicate problem_type
    # #df = df.groupby(["name", "parallelism_model", "problem_type"]).apply(lambda g: _efficiencyk(g["runtime"], g["best_sequential_runtime"].min(), k, g["n_resources"])).rename(f"efficiency@{k}").reset_index()
    # # compute the mean efficiency@k
    # df = df.groupby(["parallelism_model", "problem_type"]).agg({f"efficiency@{k}": "mean"})

    # return df

    df["best_sequential_runtime"] = df.groupby(
        ["name", "parallelism_model", "output_idx"]
    )["best_sequential_runtime"].transform("min")

    grouped = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
        lambda row: _efficiencyk(
            row["runtime"],
            np.min(row["best_sequential_runtime"]),
            k,
            row["n_resources"],
        )[f"efficiency@{k}"]
    )
    # to avoid duplicate problem_type
    result = grouped.groupby(level=["parallelism_model", "problem_type"]).mean()
    if isinstance(result, pd.Series):
        result = result.to_frame(name=f"efficiency@{k}")
    else:
        col = result.columns[0]
        result = result[[col]].rename(columns={col: f"efficiency@{k}"})
    return result


def efficiencyk_max(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df.copy()
    df = df[df["is_valid"] == True]

    df["n_resources"] = 1
    df.loc[df["parallelism_model"] == "cuda", "n_resources"] = df["problem_size"]
    df.loc[df["parallelism_model"] == "hip", "n_resources"] = df["problem_size"]
    df.loc[df["parallelism_model"] == "kokkos", "n_resources"] = df["num_threads"]
    df.loc[df["parallelism_model"] == "omp", "n_resources"] = df["num_threads"]
    df.loc[df["parallelism_model"] == "hpx", "n_resources"] = df["num_threads"]

    df = (
        df.groupby(["name", "parallelism_model", "output_idx"])
        .apply(lambda row: row.iloc[np.argmin(row["runtime"] * row["n_resources"])])
        .reset_index(drop=True)
    )

    df["best_sequential_runtime"] = df.groupby(
        ["name", "parallelism_model", "output_idx"]
    )["best_sequential_runtime"].transform("min")

    grouped = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
        lambda row: _efficiencyk(
            row["runtime"],
            np.min(row["best_sequential_runtime"]),
            k,
            row["n_resources"],
            col_name="efficiency_max@{}",
        )[f"efficiency_max@{k}"]
    )
    result = grouped.groupby(level=["parallelism_model", "problem_type"]).mean()
    if isinstance(result, pd.Series):
        result = result.to_frame(name=f"efficiency_max@{k}")
    else:
        col = result.columns[0]
        result = result[[col]].rename(columns={col: f"efficiency_max@{k}"})
    return result


def parse_problem_size(problem_size: str) -> int:
    num = problem_size.split("<<")[1][:-1]
    return 2 ** int(num)


# -----------------------------------------------------------------------------
# per-model processing
# -----------------------------------------------------------------------------
def derive_model_name(csv_path: Path) -> str:
    stem = csv_path.stem
    if stem.endswith("_combined"):
        stem = stem[:-len("_combined")]
    return stem


def assign_problem_sizes(df: pd.DataFrame, problem_sizes: Dict[str, Any]) -> None:
    if not problem_sizes:
        return
    for problem in problem_sizes:
        for parallelism_model, problem_size in problem_sizes[problem].items():
            mask = (df["name"] == problem) & (df["parallelism_model"] == parallelism_model)
            df.loc[mask, "problem_size"] = parse_problem_size(problem_size)


def add_hpx_thread_metrics(merged_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    desired_threads = [1, 2, 4, 8, 16, 32, 64]
    for threads in desired_threads:
        speed_col = f"speedup_num_threads_{threads}@1"
        eff_col = f"efficiency_num_threads_{threads}@1"
        if speed_col not in merged_df.columns:
            merged_df[speed_col] = np.nan
        if eff_col not in merged_df.columns:
            merged_df[eff_col] = np.nan

    hp = df[(df["parallelism_model"] == "hpx") & df["is_valid"]].copy()
    if hp.empty:
        return merged_df

    hp["best_seq"] = hp.groupby(["name", "output_idx"])["best_sequential_runtime"].transform("min")
    hp = hp[(hp["best_seq"] > 0) & (hp["runtime"] > 0)]
    hp["speedup"] = hp["best_seq"] / hp["runtime"]
    hp.replace([np.inf, -np.inf], np.nan, inplace=True)
    hp = hp.dropna(subset=["speedup"])

    hp["efficiency"] = hp["speedup"] / hp["num_threads"]

    metrics = (
        hp.groupby(["problem_type", "num_threads"])
        .agg(speedup_mean=("speedup", "mean"), efficiency_mean=("efficiency", "mean"))
        .reset_index()
    )

    for _, row in metrics.iterrows():
        threads = int(row["num_threads"])
        if threads not in desired_threads:
            continue
        mask = (merged_df["execution model"] == "hpx") & (
            merged_df["problem type"] == row["problem_type"]
        )
        merged_df.loc[mask, f"speedup_num_threads_{threads}@1"] = row["speedup_mean"]
        merged_df.loc[mask, f"efficiency_num_threads_{threads}@1"] = row["efficiency_mean"]

    return merged_df


def dataframe_from_json(csv_df: pd.DataFrame) -> pd.DataFrame:
    # Here csv_df already originates from the CSV, so we simply return it.
    return csv_df


def process_model_csv(
    df: pd.DataFrame,
    model_name: str,
    args: argparse.Namespace,
    problem_sizes: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    df = df.copy()
    assign_problem_sizes(df, problem_sizes or {})
    # print("DF COLUMNS", model_name, df.columns)
    if "num_threads" not in df.columns:
        df["num_threads"] = np.nan
    if "runtime" not in df.columns:
        df["runtime"] = np.nan
    if "run_idx" not in df.columns:
        df["run_idx"] = 0
    df = df[~((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 64))]

    if "did_run" not in df.columns:
        df["did_run"] = False
    if "is_valid" not in df.columns:
        df["is_valid"] = False
    if "is_source_valid" not in df.columns: 
        df["is_source_valid"] = False

    df["did_run"] = df["did_run"].fillna(False)
    df["is_valid"] = df["is_valid"].fillna(False)
    df["is_source_valid"] = df["is_source_valid"].fillna(False)
    df["num_threads"] = pd.to_numeric(df.get("num_threads", np.nan), errors="coerce")

    check(df)
    valid_runs = get_correctness_df(df)

    all_results: List[pd.DataFrame] = []
    for k in args.k:
        build_values = buildk(df, k)
        pass_values = passk(valid_runs, k)
        pass_strict_values = passk_strict(valid_runs, k)
        speedup_values = speedupk(df, k, args.n)
        speedup_max_values = speedupk_max(df, k)
        efficiency_values = efficiencyk(df, k, args.n)
        efficiency_max_values = efficiencyk_max(df, k)

        all_results.extend(
            [
                build_values,
                pass_values,
                pass_strict_values,
                speedup_values,
                speedup_max_values,
                efficiency_values,
                efficiency_max_values,
            ]
        )

    merged_df = pd.concat(all_results, axis=1).reset_index()

    for k in args.k:
        for col in [f"speedup@{k}", f"speedup_max@{k}", f"efficiency@{k}", f"efficiency_max@{k}"]:
            if col not in merged_df.columns:
                merged_df[col] = 0.0
            else:
                merged_df[col] = merged_df[col].fillna(0.0)

    merged_df.insert(0, "model", model_name)

    column_name_map = {
        "parallelism_model": "execution model",
        "problem_type": "problem type",
    }
    merged_df = merged_df.rename(columns=column_name_map)

    merged_df = add_hpx_thread_metrics(merged_df, df)
    return merged_df


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main() -> None:
    args = get_args()
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Error: {input_dir} is not a directory.")

    problem_sizes = {}
    problem_sizes_path = Path(args.problem_sizes)
    if problem_sizes_path.is_file():
        with problem_sizes_path.open("r") as fh:
            problem_sizes = json.load(fh)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {input_dir}")

    combined_frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        model_name = derive_model_name(csv_path)
        df = pd.read_csv(csv_path)
        processed = process_model_csv(df, model_name, args, problem_sizes)
        combined_frames.append(processed)

    final_df = pd.concat(combined_frames, ignore_index=True)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = input_dir / output_path
    final_df.to_csv(output_path, index=False)
    print(f"Wrote combined metrics to {output_path}")


if __name__ == "__main__":
    main()