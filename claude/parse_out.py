#!/usr/bin/env python3
"""
Split a combined prompt JSON file into separate files per problem_type.

Usage:
    python split_by_problem_type.py INPUT.json OUTPUT_DIR
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


def sanitize(name: str) -> str:
    """Make a safe filename from the problem_type string."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()) or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path, help="Path to the combined prompts JSON file.")
    parser.add_argument("output_dir", type=Path, help="Directory to write the per-category JSON files.")
    args = parser.parse_args()

    if not args.input_json.is_file():
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.input_json.open("r", encoding="utf-8") as f:
        prompts = json.load(f)

    grouped = defaultdict(list)
    for entry in prompts:
        problem_type = entry.get("problem_type", "unknown")
        grouped[problem_type].append(entry)

    for problem_type, entries in grouped.items():
        filename = sanitize(problem_type) + ".json"
        output_path = args.output_dir / filename
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        print(f"Wrote {len(entries):>4} entries → {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()