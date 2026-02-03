#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, List, Optional


def get_first_implementation(src: str, signature: str) -> Optional[str]:
    """Return the first implementation (signature + braces) matching signature."""
    pattern = re.escape(signature) + r"\s*\{"
    match = re.search(pattern, src)
    if not match:
        return None

    func_start = match.start()
    brace_start = match.end() - 1
    depth = 0

    for idx in range(brace_start, len(src)):
        ch = src[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return src[func_start : idx + 1]

    return None  # Unbalanced braces


def find_signature(snippet: str) -> Optional[str]:
    """Extract the first C/C++-style function signature preceding a brace."""
    match = re.search(r"^\s*([^\n{]+\([^\n]*\))\s*\{", snippet, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def iter_json_records(text: str) -> Iterable[dict]:
    """
    Yield JSON objects from either a JSON array or newline-delimited JSON.
    """
    stripped = text.strip()
    if not stripped:
        return []

    # First try as a single JSON value (list or dict).
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        obj = None

    if isinstance(obj, list):
        for item in obj:
            yield item
        return
    if isinstance(obj, dict):
        yield obj
        return

    # Otherwise assume JSON Lines.
    for line in stripped.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def clean_outputs(record: dict) -> dict:
    outputs: List[str] = record.get("outputs") or []
    signature = None
    for snippet in outputs:
        signature = find_signature(snippet)
        if signature:
            break

    cleaned = []
    if signature:
        for snippet in outputs:
            impl = get_first_implementation(snippet, signature)
            cleaned.append(impl if impl is not None else "None")
    else:
        cleaned = ["None"] * len(outputs)

    record["outputs"] = cleaned
    return record


def main() -> None:
    parser = ArgumentParser(description="Clean model outputs")
    parser.add_argument("--input", required=True, help="Path to the input JSON/JSONL file")
    parser.add_argument(
        "--output",
        required=False,
        help=(
            "Optional output file. If omitted, writes to <input stem>_cleaned.json "
            "next to the input file. If a directory is given, the cleaned file "
            "is written inside that directory."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")

    raw_text = input_path.read_text(encoding="utf-8")
    records = [clean_outputs(dict(rec)) for rec in iter_json_records(raw_text)]

    if not records:
        raise SystemExit("No records found in input file.")

    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir():
            output_path = output_path / f"{input_path.stem}_cleaned.json"
    else:
        output_path = input_path.with_name(f"{input_path.stem}_cleaned.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()