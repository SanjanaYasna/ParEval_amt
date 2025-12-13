"""Compute HPX metrics per prompt (name) with speedup@k restricted to 16 threads."""

import argparse
import json
from math import comb
from typing import Union, List

import numpy as np
import pandas as pd


HPX_THREAD_TARGET = 16


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=str, help="Input CSV containing test cases.")
    parser.add_argument(
        "-k", "--k",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="K values for pass@k, build@k, speedup@k, efficiency@k.",
    )
    parser.add_argument("-n", "--n", type=int, default=1, help="(Unused for HPX-only run; API parity).")
    parser.add_argument("-o", "--output", type=str, help="Output CSV file for metrics.")
    parser.add_argument(
        "--problem-sizes",
        type=str,
        default="../drivers/problem-sizes.json",
        help="Json with problem sizes (kept for compatibility).",
    )
    parser.add_argument("--model-name", type=str, help="Add model name column with this value.")
    parser.add_argument("--gpu_metrics", action="store_true", default=False)
    parser.add_argument("--cpu_metrics", action="store_true", default=False)
    return parser.parse_args()


def get_correctness_df(df: pd.DataFrame) -> pd.DataFrame:
    """Mark an output (name/output_idx) valid only if every run is valid."""
    df = df.copy()
    #strict pass@k considers is_source_valid too
    df['strict_valid_run'] = df['is_valid'] & df['is_source_valid']
    agg = df.groupby(['name', 'parallelism_model', 'output_idx']).agg(
                                                            total_runs=('is_valid', 'count')
                                                            , valid_sum=('is_valid', 'sum')
                                                            , strict_sum = ('strict_valid_run', 'sum')).reset_index()
    
    # agg = df.groupby(["name", "parallelism_model", "output_idx"]).agg({"is_valid": ["count", "sum"]})
    # agg.columns = ["count", "sum"]
    # agg["is_valid"] = agg["count"] == agg["sum"]
    # agg = agg.reset_index().drop(columns=["count", "sum"])
    # return agg
    agg["is_valid"] = agg["valid_sum"] == agg["total_runs"]
    agg["is_strict_valid"] = agg["strict_sum"] == agg["total_runs"]
    return agg.drop(columns=["total_runs", "valid_sum", "strict_sum"])

def nCr(n: int, r: int) -> int:
    if n < r:
        return 1
    return comb(n, r)


def _empty_metric_df(column: str) -> pd.DataFrame:
    mi = pd.MultiIndex.from_arrays([[], []], names=["parallelism_model", "name"])
    return pd.DataFrame(columns=[column], index=mi)


def _passk(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))


def buildk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df[df["parallelism_model"] == "hpx"]
    if df.empty:
        return _empty_metric_df(f"build@{k}")

    agg = df.groupby(["name", "parallelism_model"]).agg({"did_build": ["count", "sum"]})
    agg.columns = ["total_build_attempts", "successful_builds"]
    agg[f"build@{k}"] = _passk_vectorized(agg["total_build_attempts"], agg["successful_builds"], k)
    return agg[[f"build@{k}"]].reorder_levels(["parallelism_model", "name"]).sort_index()


def _passk_vectorized(total_runs: pd.Series, total_success: pd.Series, k: int) -> pd.Series:
    return total_runs.combine(total_success, lambda n, c: _passk(int(n), int(c), k))


def passk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df[df["parallelism_model"] == "hpx"]
    if df.empty:
        return _empty_metric_df(f"pass@{k}")

    agg = df.groupby(["name", "parallelism_model"]).agg({"is_valid": ["count", "sum"]})
    agg.columns = ["total_runs", "valid_count"]
    agg[f"pass@{k}"] = _passk_vectorized(agg["total_runs"], agg["valid_count"], k)
    return agg[[f"pass@{k}"]].reorder_levels(["parallelism_model", "name"]).sort_index()

def passk_strict(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df[df["parallelism_model"] == "hpx"]
    if df.empty:
        return _empty_metric_df(f"strict_pass@{k}")

    agg = df.groupby(["name", "parallelism_model"]).agg({"is_strict_valid": ["count", "sum"]})
    agg.columns = ["total_runs", "strict_valid_count"]
    agg[f"strict_pass@{k}"] = _passk_vectorized(agg["total_runs"], agg["strict_valid_count"], k)
    return agg[[f"strict_pass@{k}"]].reorder_levels(["parallelism_model", "name"]).sort_index()

def _speedupk(runtimes: Union[pd.Series, np.ndarray], baseline_runtime: float, k: int) -> float:
    runtimes = runtimes.values.copy() if isinstance(runtimes, pd.Series) else runtimes.copy()
    runtimes.sort()

    total = 0.0
    num_samples = runtimes.shape[0]
    for j in range(1, num_samples + 1):
        num = nCr(j - 1, k - 1) * baseline_runtime
        den = nCr(num_samples, k) * max(runtimes[j - 1], 1e-8)
        total += num / den
    return total


def speedupk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df[
        (df["parallelism_model"] == "hpx")
        & (df["is_valid"])
        & (df["num_threads"] == HPX_THREAD_TARGET)
    ]
    if df.empty:
        return _empty_metric_df(f"speedup@{k}")

    df = df.copy()
    df["best_sequential_runtime"] = df.groupby(["name", "output_idx"])["best_sequential_runtime"].transform("min")

    series = df.groupby(["parallelism_model", "name"]).apply(
        lambda g: _speedupk(g["runtime"], g["best_sequential_runtime"].min(), k)
    )
    return series.to_frame(name=f"speedup@{k}")


def speedupk_max(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df[df["parallelism_model"] == "hpx"].copy()
    df.drop(columns=["prompt"], inplace=True, errors="ignore")
    df = df[df["is_valid"]]

    if df.empty:
        return _empty_metric_df(f"speedup_max@{k}")

    df["runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["runtime"].transform("min")
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")
    df["run_idx"] = pd.to_numeric(df["run_idx"], errors="coerce").fillna(-1).astype(int)
    df = df[df["run_idx"] == 0]

    if df.empty:
        return _empty_metric_df(f"speedup_max@{k}")

    series = df.groupby(["parallelism_model", "name"]).apply(
        lambda g: _speedupk(g["runtime"], g["best_sequential_runtime"].min(), k)
    )
    return series.to_frame(name=f"speedup_max@{k}")


def _efficiencyk(
    runtimes: Union[pd.Series, np.ndarray],
    baseline_runtime: float,
    k: int,
    n_resources: Union[pd.Series, np.ndarray],
) -> float:
    runtimes = runtimes.values.copy() if isinstance(runtimes, pd.Series) else runtimes.copy()
    n_resources = n_resources.values.copy() if isinstance(n_resources, pd.Series) else n_resources.copy()

    runtimes.sort()
    total = 0.0
    num_samples = runtimes.shape[0]
    for j in range(1, num_samples + 1):
        num = nCr(j - 1, k - 1) * baseline_runtime
        den = nCr(num_samples, k) * max(runtimes[j - 1], 1e-8) * n_resources[j - 1]
        total += num / den
    return total


def efficiencyk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df[(df["parallelism_model"] == "hpx") & (df["is_valid"])]
    if df.empty:
        return _empty_metric_df(f"efficiency@{k}")

    df = df.copy()
    df["n_resources"] = df.get("num_threads", 1).fillna(1).astype(float)
    df["best_sequential_runtime"] = df.groupby(["name", "output_idx"])["best_sequential_runtime"].transform("min")

    series = df.groupby(["parallelism_model", "name"]).apply(
        lambda g: _efficiencyk(g["runtime"], g["best_sequential_runtime"].min(), k, g["n_resources"])
    )
    return series.to_frame(name=f"efficiency@{k}")


def efficiencyk_max(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df[(df["parallelism_model"] == "hpx") & (df["is_valid"])]
    if df.empty:
        return _empty_metric_df(f"efficiency_max@{k}")

    df = df.copy()
    df["n_resources"] = df.get("num_threads", 1).fillna(1).astype(float)

    df = df.groupby(["name", "parallelism_model", "output_idx"]).apply(
        lambda g: g.iloc[np.argmin(g["runtime"] * g["n_resources"])]
    ).reset_index(drop=True)

    if df.empty:
        return _empty_metric_df(f"efficiency_max@{k}")

    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    series = df.groupby(["parallelism_model", "name"]).apply(
        lambda g: _efficiencyk(g["runtime"], g["best_sequential_runtime"].min(), k, g["n_resources"])
    )
    return series.to_frame(name=f"efficiency_max@{k}")


def parse_problem_size(problem_size: str) -> int:
    num = problem_size.split("<<")[1][:-1]
    return 2 ** int(num)


def concat_metrics(metrics: List[pd.DataFrame]) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["parallelism_model", "name"]))
    result = pd.concat(metrics, axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    return result


def main():
    args = get_args()

    df = pd.read_csv(args.input_csv)

    with open(args.problem_sizes, "r") as f:
        problem_sizes = json.load(f)
    for problem in problem_sizes:
        for parallelism_model, problem_size in problem_sizes[problem].items():
            df.loc[
                (df["name"] == problem) & (df["parallelism_model"] == parallelism_model),
                "problem_size",
            ] = parse_problem_size(problem_size)

    df = df[df["parallelism_model"] == "hpx"].copy()

    #there is no did_run nor is_valid if nothing passes
    if "did_run" not in df.columns:
        df["did_run"] = False
    if "is_valid" not in df.columns:
        df["is_valid"] = False
    if "is_source_valid" not in df.columns:
        df["is_source_valid"] = False

    df["did_run"] = df["did_run"].fillna(False)
    df["is_valid"] = df["is_valid"].fillna(False)
    df["is_source_valid"] = df['is_source_valid'].fillna(False)
    df["num_threads"] = pd.to_numeric(df.get("num_threads", np.nan), errors="coerce")

    valid_runs = get_correctness_df(df)

    all_results: List[pd.DataFrame] = []
    for k in args.k:
        build_values = buildk(df, k)
        pass_values = passk(valid_runs, k)
        pass_strict_values = passk_strict(valid_runs, k)
        speedup_values = speedupk(df, k)
        speedup_max_values = speedupk_max(df, k)
        efficiency_values = efficiencyk(df, k)
        efficiency_max_values = efficiencyk_max(df, k)
        all_results.extend([
            build_values,
            pass_values,
            pass_strict_values,
            speedup_values,
            speedup_max_values,
            efficiency_values,
            efficiency_max_values,
        ])

    merged_df = concat_metrics(all_results)

    metric_cols = [
        "max_new_tokens",
        "virtual_memory_used",
        "max_gpu_memory_usage",
        "max_gpu_utilization",
        "average_gpu_memory_usage",
        "average_gpu_utilization",
        "generation_times",
    ]
    existing_metric_cols = [c for c in metric_cols if c in df.columns]
    if existing_metric_cols:
        df_metrics = df[["parallelism_model", "name"] + existing_metric_cols].copy()
        for col in existing_metric_cols:
            df_metrics[col] = pd.to_numeric(df_metrics[col], errors="coerce")
        metric_means = df_metrics.groupby(["parallelism_model", "name"])[existing_metric_cols].mean()
        merged_df = merged_df.join(metric_means, how="outer")

    merged_df = merged_df.reset_index()

    for k in args.k:
        for metric in [f"speedup@{k}", f"speedup_max@{k}", f"efficiency@{k}", f"efficiency_max@{k}"]:
            if metric in merged_df.columns:
                merged_df[metric] = merged_df[metric].fillna(0.0)

    if args.model_name:
        merged_df.insert(0, "model_name", args.model_name)

    column_name_map = {
        "model_name": "model",
        "parallelism_model": "execution model",
        "name": "prompt",
    }
    merged_df = merged_df.rename(columns=column_name_map)

    if args.output:
        merged_df.to_csv(args.output, index=False)
    else:
        pd.set_option("display.max_columns", merged_df.shape[1] + 1)
        pd.set_option("display.max_rows", merged_df.shape[0] + 1)
        print(merged_df)


if __name__ == "__main__":
    main()