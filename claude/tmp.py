#!/usr/bin/env python3
import json
from pathlib import Path

INPUT_PATH = Path("/work/pi_mrobson_smith_edu/ParEval_amt/generate/claude/sonnet_run_archive.json")
OUTPUT_PATH = INPUT_PATH.with_name("sonnet_la.json")
KEEP_TYPES = {"dense_la", "sparse_la"}

def main() -> None:
    if not INPUT_PATH.is_file():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = [entry for entry in data if entry.get("problem_type") in KEEP_TYPES]

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2)

    print(f"Wrote {len(filtered)} entries to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
