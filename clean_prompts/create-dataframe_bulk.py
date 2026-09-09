#!/usr/bin/env python3
"""
Create a CSV for each JSON file produced by run-all.py within a directory.

Usage:
    python make_csvs.py <directory-with-jsons>
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing JSON files from run-all.py",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def has_outputs(prompt: Dict[str, Any]) -> bool:
    """Check if a prompt dictionary contains non-empty dict outputs."""
    outputs = prompt.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return False
    if all(isinstance(o, str) for o in outputs):
        return False
    return all(isinstance(o, dict) for o in outputs)


def metric_value(prompt: Dict[str, Any], key: str, idx: int) -> Optional[Any]:
    """Safely extract prompt[key][idx] if it exists; otherwise return None."""
    values = prompt.get(key)
    if values is None:
        return None
    if isinstance(values, list):
        if idx < len(values):
            return values[idx]
        return None
    return values  # fall back to scalar if not a list


def check(df: pd.DataFrame) -> None:
    """Warn about (name, parallelism_model) combos that never built successfully."""
    if df.empty:
        return
    agg = df.groupby(["name", "parallelism_model"], dropna=False).agg({"did_build": "sum"})
    zero_builds = agg[agg["did_build"] == 0]
    if not zero_builds.empty:
        print("The following (name, parallelism_model) pairs have zero successful builds:")
        print(zero_builds)


def json_to_dataframe(prompts: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert the list of prompts/outputs into a DataFrame."""
    rows: List[Dict[str, Any]] = []

    for prompt in prompts:
        outputs = prompt["outputs"]

        for output_idx, output in enumerate(outputs):
            if output.get("runs") is None:
                rows.append({
                    "prompt": prompt["prompt"],
                    "name": prompt["name"],
                    "problem_type": prompt["problem_type"],
                    "language": prompt["language"],
                    "parallelism_model": prompt["parallelism_model"],
                    "temperature": prompt["temperature"],
                    "top_p": prompt["top_p"],
                    "do_sample": prompt["do_sample"],
                    "max_new_tokens": prompt["max_new_tokens"],
                    "prompted": prompt.get("prompted", False),
                    "generated_output": output["generated_output"],
                    "did_build": output["did_build"],
                    "is_source_valid": output["is_source_valid"],
                    "best_sequential_runtime": output["best_sequential_runtime"],
                    "output_idx": output_idx,
                    "virtual_memory_used": metric_value(prompt, "virtual_memory_used", output_idx),
                    "max_gpu_memory_usage": metric_value(prompt, "max_gpu_memory_usage", output_idx),
                    "max_gpu_utilization": metric_value(prompt, "max_gpu_utilization", output_idx),
                    "average_gpu_memory_usage": metric_value(prompt, "average_gpu_memory_usage", output_idx),
                    "average_gpu_utilization": metric_value(prompt, "average_gpu_utilization", output_idx),
                    "generation_times": metric_value(prompt, "generation_times", output_idx),
                })
                continue

            for run_idx, run in enumerate(output["runs"]):
                row = {
                    "prompt": prompt["prompt"],
                    "name": prompt["name"],
                    "problem_type": prompt["problem_type"],
                    "language": prompt["language"],
                    "parallelism_model": prompt["parallelism_model"],
                    "temperature": prompt["temperature"],
                    "top_p": prompt["top_p"],
                    "do_sample": prompt["do_sample"],
                    "max_new_tokens": prompt["max_new_tokens"],
                    "prompted": prompt.get("prompted", False),
                    "generated_output": output["generated_output"],
                    "did_build": output["did_build"],
                    "is_source_valid": output["is_source_valid"],
                    "best_sequential_runtime": output["best_sequential_runtime"],
                    "output_idx": output_idx,
                    "run_idx": run_idx,
                    "virtual_memory_used": metric_value(prompt, "virtual_memory_used", output_idx),
                    "max_gpu_memory_usage": metric_value(prompt, "max_gpu_memory_usage", output_idx),
                    "max_gpu_utilization": metric_value(prompt, "max_gpu_utilization", output_idx),
                    "average_gpu_memory_usage": metric_value(prompt, "average_gpu_memory_usage", output_idx),
                    "average_gpu_utilization": metric_value(prompt, "average_gpu_utilization", output_idx),
                    "generation_times": metric_value(prompt, "generation_times", output_idx),
                    **run,
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.prompt = df.prompt.str.replace("\n", "\\n", regex=False)
        df.generated_output = df.generated_output.str.replace("\n", "\\n", regex=False)
    return df


def process_file(json_path: Path) -> None:
    """Convert a single JSON file to a CSV with matching stem."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = [entry for entry in data if has_outputs(entry)]
    if not prompts:
        print(f"Skipping {json_path.name}: no valid outputs.")
        return

    df = json_to_dataframe(prompts)
    check(df)

    csv_path = json_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path.name}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = get_args()
    input_dir = Path(args.input_dir).resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Error: {input_dir} is not a directory.")

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}.")
        return

    for json_path in json_files:
        try:
            process_file(json_path)
        except Exception as exc:
            print(f"Error processing {json_path.name}: {exc}")


if __name__ == "__main__":
    main()
