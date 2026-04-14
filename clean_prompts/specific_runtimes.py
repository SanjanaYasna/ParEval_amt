#!/usr/bin/env python3
"""
NOTE: CODE GENERATED WITH HELP OF GPT5-CODEX, ADJUSTED AFTERWARDS
Summarize per-prompt (by `name`), per-thread runtimes plus generation/GPU/memory
averages from JSON.

For every required *_combined.json input the script:
  • finds entries containing a prompt identifier (default: the `name` field),
    a thread count (num_threads by default) and a runtime (overall_runtime or
    runtime by default) and averages runtimes by (name, thread count),
  • gathers per-prompt statistics such as average generation time, GPU utilization,
    or memory utilization—even when stored in nested lists/dicts—and averages them
    per name,
  • writes one CSV row per (model, name), with columns runtime_<threads> plus
    avg_generation_time, avg_gpu_utilization, avg_memory_utilization.

Missing metrics remain NaN in the CSV.
If all nan, the metrics raw just doesn't appear since it's using defaultdict
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_PROMPT_LABEL = "(unknown name)"
PROMPT_COLUMN = "name"
MODEL_COLUMN = "model"


# --------------------------------------------------------------------------- #
# argument parsing                                                            #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing JSON result files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="json_runtime_summary.csv",
        help="Name (or path) of the CSV to write (default: json_runtime_summary.csv).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern for JSON files (default: *.json).",
    )
    parser.add_argument(
        "--runtime-keys",
        type=str,
        nargs="+",
        default="runtime",
        help="Keys (searched in order) for runtime values.",
    )
    parser.add_argument(
        "--threads-key",
        type=str,
        default="num_threads",
        help="Key that stores the thread count.",
    )
    parser.add_argument(
        "--prompt-key",
        action="append",
        default="name",
        help=(
            "Keys to treat as prompt/name identifiers. "
            "Repeat to add multiple keys (default: name, prompt, prompt_id, task, task_id)."
        ),
    )
    parser.add_argument(
        "--metric-key", #file key : renamed key that is avg
        action="append",
        default=[
            "generation_times:avg_generation_time",
            "virtual_memory_used:avg_virtual_memory_used",
            "average_gpu_memory_usage:avg_gpu_memory_utilization", 
            "average_gpu_utilization:avg_gpu_utilization",
        ],
        help="Extra averages to gather. Use KEY:OUTPUT_COL (repeat for multiple keys).",
    )
    return parser.parse_args()


def parse_metric_spec(specs: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(f"Invalid --metric-key '{spec}' (expected KEY:OUTPUT_COL)")
        key, column = spec.split(":", 1)
        key = key.strip()
        column = column.strip()
        if not key or not column:
            raise SystemExit(f"Invalid --metric-key '{spec}' (empty key/column)")
        mapping[key] = column
    return mapping


# --------------------------------------------------------------------------- #
# JSON traversal helpers                                                      #
# --------------------------------------------------------------------------- #
def collect_numbers(value: Any) -> List[float]:
    """
    Extract all numeric values (as floats) from value, recursing into lists/dicts.
    Strings can contain numbers (e.g., '12.3s').
    In reality it's all floats, but just put there anyway (AI generated)
    """
    numbers: List[float] = []
    if value is None:
        return numbers
    if isinstance(value, (int, float)):
        numbers.append(float(value))
        return numbers
    if isinstance(value, list):
        for item in value:
            numbers.extend(collect_numbers(item))
        return numbers
    if isinstance(value, dict):
        for item in value.values():
            numbers.extend(collect_numbers(item))
        return numbers

    text = str(value).strip()
    if not text:
        return numbers
    for match in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text):
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    return numbers


def extract_prompt_data(
    obj: Any,
    runtime_keys: Iterable[str],
    threads_key: str,
    prompt_keys: Sequence[str],
    metric_keys: Dict[str, str],
) -> Tuple[
    Dict[str, Dict[int, List[float]]],
    Dict[str, Dict[str, List[float]]],
    List[str],
]:
    """
    Traverse the JSON payload to collect:
      • per-name runtimes keyed by thread count,
      • per-name metric values (e.g., avg_generation_time),
      • the order in which names were first observed.
    """
    per_prompt_runtimes: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_prompt_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    prompt_order: List[str] = []
    seen_prompts: set[str] = set()

    column_aliases: Dict[str, str] = {}
    for key, column in metric_keys.items():
        column_aliases[key] = column
        column_aliases[column] = column

    def register_prompt(name: Optional[str]) -> str:
        prompt_name = name if name else DEFAULT_PROMPT_LABEL
        if prompt_name not in seen_prompts:
            seen_prompts.add(prompt_name)
            prompt_order.append(prompt_name)
        return prompt_name

    def parse_threads(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            ivalue = int(value)
            return ivalue if ivalue >= 0 else None

    #alaways null or float, but ai generated robust parsing here
    def parse_runtime(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        
    def parse_prompt(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        stripped = str(value).strip()
        return stripped if stripped else None

#expect dict output for a prompt
    def recurse(node: Any, current_prompt: Optional[str]) -> None:
        if isinstance(node, dict):
            prompt_here = current_prompt
            if "name" in node.keys():
                prompt_here = parse_prompt(node["name"])
            prompt_name = prompt_here if prompt_here else DEFAULT_PROMPT_LABEL
            
            #get runtime and threads corresponding to each entry
            runtime_val: Optional[float] = None
            if "runtime" in node.keys():
                runtime_val = parse_runtime(node["runtime"])

            threads_val = parse_threads(node.get(threads_key))
            if runtime_val is not None and threads_val is not None:
                prompt_name = register_prompt(prompt_here)
                per_prompt_runtimes[prompt_name][threads_val].append(runtime_val)

            for json_key, column in metric_keys.items():
                if json_key in node:
                    numbers = collect_numbers(node[json_key])
                    if numbers:
                        prompt_name = register_prompt(prompt_here)
                        per_prompt_metrics[prompt_name][column].extend(numbers)

            for child in node.values():
                recurse(child, prompt_here)
        elif isinstance(node, list):
            for item in node:
                recurse(item, current_prompt)

    recurse(obj, None)
    return per_prompt_runtimes, per_prompt_metrics, prompt_order


# --------------------------------------------------------------------------- #
# per-file processing                                                         #
# --------------------------------------------------------------------------- #
def summarize_file_by_prompt(
    json_path: Path,
    runtime_keys: Iterable[str],
    threads_key: str,
    prompt_keys: Sequence[str],
    metric_keys: Dict[str, str],
) -> List[Dict[str, Any]]:
    with json_path.open("r") as fh:
        payload = json.load(fh)

    model_name = json_path.stem
    per_prompt_runtimes, per_prompt_metrics, prompt_order = extract_prompt_data(
        payload,
        runtime_keys,
        threads_key,
        prompt_keys,
        metric_keys,
    )
    # print(json_path)
    # print(per_prompt_runtimes)
    # print("__________________")
    prompts = list(prompt_order)
    extra_prompts = (set(per_prompt_runtimes) | set(per_prompt_metrics)) - set(prompts)
    prompts.extend(extra_prompts)
    if not prompts:
        prompts = [DEFAULT_PROMPT_LABEL]

    rows: List[Dict[str, Any]] = []
    for prompt in prompts:
        row: Dict[str, Any] = {MODEL_COLUMN: model_name, PROMPT_COLUMN: prompt}
        #take avg of runtimes and name as runtime_#
        for threads, runtimes in per_prompt_runtimes.get(prompt, {}).items():
            if runtimes:
                row[f"runtime_{threads}"] = mean(runtimes)
        #average out metrics (memory, gpu util, etc)
        for column, values in per_prompt_metrics.get(prompt, {}).items():
            if values:
                row[column] = mean(values)

        rows.append(row)

    return rows


def main() -> None:
    args = parse_args()
    metric_keys = parse_metric_spec(args.metric_key)

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Error: {input_dir} is not a directory.")

    prompt_keys = args.prompt_key or []
    #5 model files to look out for
    required_files = [
        "hpc-coder_combined.json",
        "magicoder_combined.json",
        "sonnet_combined.json",
        "starcoder_combined.json",
        "gpt5_combined.json", 
        "minimax_combined.json", 
        "glm_combined.json", 
        "oss_combined.json"
    ]
    json_files: List[Path] = []
    for fname in required_files:
        fpath = input_dir / fname
        if not fpath.is_file(): #gpt-5 or gpt5 are ok names (to fix naming inconsistencies...)
            if fname == "gpt5_combined.json":
                new_fpath = input_dir / "gpt-5_combined.json"
                if not new_fpath.is_file():
                    raise SystemExit(f"Missing required file: {fpath}")
                else: 
                    fpath = new_fpath 
            else:
                raise SystemExit(f"Missing required file: {fpath}")
        json_files.append(fpath)

    rows: List[Dict[str, Any]] = []
    thread_counts: set[int] = set()
    aux_columns: set[str] = set(metric_keys.values())

    for json_path in json_files:
        try:
            #get per prompt summary for this specific model
            prompt_rows = summarize_file_by_prompt(
                json_path,
                args.runtime_keys,
                args.threads_key,
                prompt_keys,
                metric_keys,
            )
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: skipping {json_path} ({exc})")
            continue

        rows.extend(prompt_rows)
        for row in prompt_rows:
            for key in row:
                if key.startswith("runtime_"):
                    try:
                        thread_counts.add(int(key.split("_", 1)[1]))
                    except ValueError:
                        pass

    if not rows:
        raise SystemExit("No valid entries were found in the JSON files.")

    runtime_cols = [f"runtime_{threads}" for threads in sorted(thread_counts)]
    columns = [MODEL_COLUMN, PROMPT_COLUMN] + runtime_cols + sorted(aux_columns)

    df = pd.DataFrame(rows)
    df = df.reindex(columns=columns)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = input_dir / output_path
    df.to_csv(output_path, index=False)
    print(f"Wrote runtime summary to {output_path}")


if __name__ == "__main__":
    main()


