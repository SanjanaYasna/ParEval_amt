#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_batch_text(path: Path) -> dict:
    """
    Load a batch request from `path`. If it's not valid JSON on its own,
    try to extract the first {...} block out of it (e.g. from a .sh file).
    """
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()
    if not stripped:
        raise ValueError("Input file is empty.")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Could not locate JSON object inside input file.")
        snippet = raw[start:end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse embedded JSON block: {e}")


def expand_requests(batch_json: dict, copies: int = 100, top_p: float = 0.95) -> dict:
    """Duplicate each request `copies` times and set params['top_p'] to `top_p`."""
    new_requests = []

    for req in batch_json.get("requests", []):
        base_id = req.get("custom_id", "request")

        # Ensure params exists so we can set top_p
        params = req.get("params")
        if params is None:
            params = {}
            req["params"] = params
        params["top_p"] = top_p

        for i in range(copies):
            cloned = json.loads(json.dumps(req))  # deep copy
            cloned["custom_id"] = f"{base_id}_copy_{i+1:03d}"
            new_requests.append(cloned)

    return {"requests": new_requests}


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand Claude batch to 100 copies per prompt (adds top_p).")
    parser.add_argument("--input", required=True, help="Path to the original batch JSON")
    parser.add_argument("--output", help="Optional output path (default <input_stem>_x100.json)")
    parser.add_argument("--copies", type=int, default=100, help="Copies per original request (default 100)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Value to set for params['top_p'] (default 0.95)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    try:
        batch_data = load_batch_text(input_path)
    except ValueError as exc:
        raise SystemExit(f"Error loading batch: {exc}")

    expanded = expand_requests(batch_data, copies=args.copies, top_p=args.top_p)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_x100.json")

    output_path.write_text(json.dumps(expanded, indent=2), encoding="utf-8")
    print(f"Wrote expanded batch ({len(expanded['requests'])} requests) to {output_path}")


if __name__ == "__main__":
    main()