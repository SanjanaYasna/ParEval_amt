#!/usr/bin/env python3
"""
NOTE: CODE GENERATED WITH SONNET HELP 
Combine model run logs from tcmalloc 1.5.1 and 1.10.0 (or copy if only one exists).

Usage:
    python combine_runs.py <prompt-category-directory>

Given a category directory (e.g. /.../geometry), the script expects to find:
    category/driver/tcmalloc/1.5.1/*.json
    category/driver/tcmalloc/1.10.0/*.json

For each JSON filename:
  * If it exists in both versions, the script merges them:
      - For each thread, keeps the run with the smallest runtime (or the one that
        actually succeeded if only one did).
      - Unions `did_run`/`is_valid` flags.
      - Unions per-output validity and recomputes `num_valid_outputs`.
      - Writes the result to driver/tcmalloc/<model>_combined.json.
  * If it exists in just one version, the script copies it unchanged (aside from the
    `_combined.json` name) to the same output directory.
"""

import json
import sys
from copy import deepcopy
from itertools import zip_longest
from pathlib import Path

from typing import Any, Dict, Iterable, List, Optional

JSONValue = Any


# ---------------------------------------------------------------------------#
#  Helpers for run merging
# ---------------------------------------------------------------------------#

def is_run_list(value: Any) -> bool:
    """Return True if `value` looks like a list of threaded run dicts."""
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(item, dict) and "num_threads" in item for item in value)


def combine_bool(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    """Union of two optional booleans."""
    bools = [v for v in (a, b) if isinstance(v, bool)]
    if not bools:
        return True if (a is True or b is True) else None
    return True if True in bools else False


def extract_runtime(run: Dict[str, Any]) -> Optional[float]:
    """Return the runtime as a float (if numeric), otherwise None."""
    value = run.get("runtime")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def merge_single_run(run_a: Optional[Dict[str, Any]],
                     run_b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge two run dictionaries for the same thread."""
    if run_a is None and run_b is None:
        return {}

    if run_a is None:
        base = deepcopy(run_b)
    elif run_b is None:
        base = deepcopy(run_a)
    else:
        runtime_a = extract_runtime(run_a)
        runtime_b = extract_runtime(run_b)
        if runtime_a is None and runtime_b is not None:
            base = deepcopy(run_b)
        elif runtime_b is None and runtime_a is not None:
            base = deepcopy(run_a)
        elif runtime_a is None and runtime_b is None:
            base = deepcopy(run_a)
        else:
            base = deepcopy(run_a if runtime_a <= runtime_b else run_b)

        if runtime_a is not None and runtime_b is not None:
            base["runtime"] = min(runtime_a, runtime_b)
        elif runtime_a is not None:
            base["runtime"] = runtime_a
        else:
            base["runtime"] = runtime_b

        base["did_run"] = bool(run_a.get("did_run")) or bool(run_b.get("did_run"))
        if "is_valid" in run_a or "is_valid" in run_b:
            base["is_valid"] = combine_bool(run_a.get("is_valid"), run_b.get("is_valid"))
        return base

    # Only one run existed
    if run_a is None:
        base["did_run"] = bool(base.get("did_run"))
    else:
        base["did_run"] = bool(base.get("did_run"))
    if "is_valid" in base:
        base["is_valid"] = combine_bool(base.get("is_valid"), base.get("is_valid"))
    return base


def merge_run_lists(list_a: Optional[List[Dict[str, Any]]],
                    list_b: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge two lists of run dicts keyed by num_threads."""
    list_a = list_a or []
    list_b = list_b or []

    mapping: Dict[Any, Dict[str, Dict[str, Any]]] = {}
    order: List[Any] = []

    def register(run: Dict[str, Any], label: str) -> None:
        key = run.get("num_threads")
        if key is None:
            return
        mapping.setdefault(key, {})[label] = run
        if key not in order:
            order.append(key)

    for item in list_a:
        register(item, "a")
    for item in list_b:
        register(item, "b")

    merged = [
        merge_single_run(mapping[thread].get("a"), mapping[thread].get("b"))
        for thread in order
    ]

    def without_thread(runs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [deepcopy(run) for run in runs if run.get("num_threads") is None]

    merged.extend(without_thread(list_a))
    merged.extend(without_thread(list_b))

    return merged


# ---------------------------------------------------------------------------#
#  Generic structure merging
# ---------------------------------------------------------------------------#

def merge_generic(a: Any, b: Any) -> Any:
    """Recursively merge two JSON fragments, preferring `a` unless missing."""
    if a is None:
        return deepcopy(b)
    if b is None:
        return deepcopy(a)

    if isinstance(a, dict) and isinstance(b, dict):
        return merge_generic_dict(a, b)

    if isinstance(a, list) and isinstance(b, list):
        if is_run_list(a) or is_run_list(b):
            return merge_run_lists(a, b)
        merged: List[Any] = []
        for item_a, item_b in zip_longest(a, b, fillvalue=None):
            merged.append(merge_generic(item_a, item_b))
        return merged

    return deepcopy(a)


def merge_generic_dict(dict_a: Optional[Dict[str, Any]],
                       dict_b: Optional[Dict[str, Any]]):
    dict_a = dict_a or {}
    dict_b = dict_b or {}
    result = deepcopy(dict_a)
    bsr_a = dict_a.get("best_sequential_runtime")
    bsr_b = dict_b.get("best_sequential_runtime")
    if isinstance(bsr_a, str): bsr_a = None
    if isinstance(bsr_b, str): bsr_b = None
    
    if bsr_a is not None and bsr_b is not None:
        result["best_sequential_runtime"] = min(bsr_a, bsr_b)
       # print("RESULT UPDATED")
    elif bsr_a is not None:
        result["best_sequential_runtime"] = bsr_a
    elif bsr_b is not None:
        result["best_sequential_runtime"] = bsr_b

    for key, value_b in dict_b.items():
        if key == "runs" and (is_run_list(result.get(key)) or is_run_list(value_b)):
            result[key] = merge_run_lists(result.get(key), value_b)
        elif key == "is_valid":
            result[key] = combine_bool(result.get(key), value_b)
        elif key in result:
            result[key] = merge_generic(result[key], value_b)
        else:
            result[key] = deepcopy(value_b)
    return result


# ---------------------------------------------------------------------------#
#  Output-level logic
# ---------------------------------------------------------------------------#

def iter_run_lists(node: Any) -> Iterable[List[Dict[str, Any]]]:
    """Yield every run-list encountered in a nested structure."""
    if isinstance(node, list):
        if is_run_list(node):
            yield node
        else:
            for item in node:
                yield from iter_run_lists(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from iter_run_lists(value)


def output_valid_flag(output: Optional[Any]) -> bool:
    """Determine whether an output is considered valid."""
    if not isinstance(output, dict):
        return False
    value = output.get("is_valid")
    if value is True:
        return True
    for run_list in iter_run_lists(output):
        for run in run_list:
            if run.get("is_valid") is True:
                return True
    return False


def merge_single_output(out_a: Optional[Any], out_b: Optional[Any]) -> Any:
    if out_a is None:
        return deepcopy(out_b)
    if out_b is None:
        return deepcopy(out_a)

    if not isinstance(out_a, dict) or not isinstance(out_b, dict):
        return deepcopy(out_a)

    result = deepcopy(out_a)

    runs_a = out_a.get("runs")
    runs_b = out_b.get("runs")
    if runs_a is not None or runs_b is not None:
        result["runs"] = merge_run_lists(runs_a or [], runs_b or [])

   
        
    for key, value_b in out_b.items():
        if key in ("runs", "is_valid", "best_sequential_runtime"):
            continue
        if key in result:
            result[key] = merge_generic(result[key], value_b)
        else:
            result[key] = deepcopy(value_b)

    valid_union = output_valid_flag(out_a) or output_valid_flag(out_b)
    if not valid_union:
        valid_union = output_valid_flag(result)

    if valid_union or "is_valid" in result:
        result["is_valid"] = valid_union

    return result


def merge_outputs(list_a: Optional[List[Any]],
                  list_b: Optional[List[Any]]) -> (List[Any], int):
    list_a = list_a or []
    list_b = list_b or []

    merged_outputs: List[Any] = []
    valid_count = 0

    for out_a, out_b in zip_longest(list_a, list_b, fillvalue=None):
        merged = merge_single_output(out_a, out_b)
        merged_outputs.append(merged)

        valid_union = output_valid_flag(out_a) or output_valid_flag(out_b)
        if not valid_union:
            valid_union = output_valid_flag(merged)
        if valid_union:
            valid_count += 1

    return merged_outputs, valid_count


def combine_data(data_a: JSONValue, data_b: JSONValue) -> JSONValue:
    if isinstance(data_a, dict) and isinstance(data_b, dict):
        combined = deepcopy(data_a)

        outputs_a = data_a.get("outputs")
        outputs_b = data_b.get("outputs")
        if isinstance(outputs_a, list) or isinstance(outputs_b, list):
            merged_outputs, valid_count = merge_outputs(outputs_a, outputs_b)
            combined["outputs"] = merged_outputs
            combined["num_valid_outputs"] = valid_count

        for key, value_b in data_b.items():
            if key in ("outputs", "num_valid_outputs"):
                continue
            if key in combined:
                combined[key] = merge_generic(combined[key], value_b)
            else:
                combined[key] = deepcopy(value_b)

        if "outputs" in combined and not isinstance(combined.get("num_valid_outputs"), int):
            combined["num_valid_outputs"] = sum(
                1 for output in combined["outputs"] if output_valid_flag(output)
            )

        return combined

    if isinstance(data_a, list) and isinstance(data_b, list):
        merged_outputs, _ = merge_outputs(data_a, data_b)
        return merged_outputs

    return deepcopy(data_a)


# ---------------------------------------------------------------------------#
#  File system orchestration
# ---------------------------------------------------------------------------#

def derive_model_name(filename: str) -> str:
    stem = Path(filename).stem
    if stem.endswith("_exclusive"):
        stem = stem[: -len("_exclusive")]
    if "_run" in stem:
        stem = stem.split("_run")[0]
    return stem


def load_json(path: Path) -> JSONValue:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: JSONValue) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------#
#  Main entry point
# ---------------------------------------------------------------------------#

def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <prompt-category-directory>", file=sys.stderr)
        sys.exit(1)

    base = Path(sys.argv[1]).resolve()
    dir_151 = base / "driver" / "tcmalloc" / "1.5.1"
    dir_110 = base / "driver" / "tcmalloc" / "1.10.0"

    if not dir_151.is_dir() or not dir_110.is_dir():
        print("Expected directories 'driver/tcmalloc/1.5.1' and 'driver/tcmalloc/1.10.0'.",
              file=sys.stderr)
        sys.exit(1)

    output_dir = base / "driver" / "tcmalloc"
    output_dir.mkdir(parents=True, exist_ok=True)

    files_151 = {p.name: p for p in dir_151.glob("*.json")}
    files_110 = {p.name: p for p in dir_110.glob("*.json")}
    all_names = sorted(set(files_151) | set(files_110))

    if not all_names:
        print("No JSON files to process.", file=sys.stderr)
        return

    for name in all_names:
        path_151 = files_151.get(name)
        path_110 = files_110.get(name)

        if path_151 and path_110:
            data_151 = load_json(path_151)
            data_110 = load_json(path_110)
            combined = combine_data(data_151, data_110)
            mode = "merged"
        else:
            single_path = path_151 or path_110
            combined = load_json(single_path)
            mode = "copied"

        model_name = derive_model_name(name)
        output_path = output_dir / f"{model_name}_combined.json"
        write_json(output_path, combined)
        rel_path = output_path.relative_to(base)
        print(f"{mode.capitalize():>7}: {rel_path}")

    missing_151 = sorted(set(files_110) - set(files_151))
    missing_110 = sorted(set(files_151) - set(files_110))
    if missing_151:
        print("Files only in 1.10.0:", ", ".join(missing_151), file=sys.stderr)
    if missing_110:
        print("Files only in 1.5.1:", ", ".join(missing_110), file=sys.stderr)


if __name__ == "__main__":
    main()
