#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path

DROP_TYPES = {"fft", "geometry", "graph", "histogram", "reduce"}

def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input.json>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1]).resolve()
    if not input_path.is_file():
        print(f"Error: {input_path} is not a file.", file=sys.stderr)
        sys.exit(1)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    groups = defaultdict(list)
    for entry in data:
        problem_type = entry.get("problem_type")
        if problem_type in DROP_TYPES:
            continue
        groups[problem_type].append(entry)

    if not groups:
        print("No remaining problem types after filtering.", file=sys.stderr)
        sys.exit(1)

    for problem_type, entries in groups.items():
        if len(entries) < 5:
            raise RuntimeError(
                f"Problem type '{problem_type}' only has {len(entries)} entries; need at least 5."
            )

        out_path = input_path.with_name(f"sonnet_{problem_type}.json")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(entries[:5], f, indent=2)
        print(f"Wrote {problem_type}.json with 5 entries")

if __name__ == "__main__":
    main()