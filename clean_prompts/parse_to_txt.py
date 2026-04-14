#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def iter_json_objects(text):
    """Yield each JSON object found in raw text."""
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    while idx < length:
        idx = text.find('{', idx)
        if idx == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        yield obj
        idx = end

def candidate_to_text(candidate):
    """Return a readable string from a candidate output."""
    if isinstance(candidate, str):
        return candidate

    if isinstance(candidate, dict):
        for key in ("text", "output", "content", "message"):
            if key in candidate:
                return candidate_to_text(candidate[key])
        return json.dumps(candidate, indent=2, ensure_ascii=False)

    if isinstance(candidate, (list, tuple)):
        parts = [candidate_to_text(part) for part in candidate]
        parts = [p for p in parts if p]
        return "\n".join(parts) if parts else ""

    return str(candidate)

def format_prompt_outputs(entries):
    """Return a string with prompt outputs nicely formatted."""
    lines = []
    for entry in entries:
        outputs = entry.get("outputs")
        if not outputs:
            continue

        name = entry.get("name", "(unknown name)")
        prompt_body = entry.get("prompt")

        lines.append(f"Prompt: {name}")
        if isinstance(prompt_body, str):
            body_text = prompt_body.strip()
            if body_text:
                lines.append("  Prompt body:")
                for line in body_text.splitlines():
                    lines.append(f"    {line}")
        elif prompt_body is not None:
            lines.append("  Prompt body:")
            body_text = json.dumps(prompt_body, indent=2, ensure_ascii=False)
            for line in body_text.splitlines():
                lines.append(f"    {line}")

        for idx, candidate in enumerate(outputs, start=1):
            candidate_text = candidate_to_text(candidate)
            lines.append(f"  Output {idx}:")
            for line in candidate_text.splitlines():
                lines.append(f"    {line}")
        lines.append("")
    return "\n".join(lines)

def resolve_output_path(cache_path: Path, output_arg: str | None) -> Path:
    """Decide where to write the formatted file."""
    stem = cache_path.stem
    if output_arg:
        output_path = Path(output_arg)
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / f"raw_{stem}.txt"
        elif output_path.suffix:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / f"raw_{stem}.txt"
    else:
        base_dir = cache_path.resolve().parent
        output_dir = base_dir / "raw_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"raw_{stem}.txt"
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description="Format prompt outputs from raw JSON cache files."
    )
    parser.add_argument(
        "cache_file",
        help="Path to the cache JSON file (e.g., gpt5_cache_low_reasoning.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output file or directory. If a directory is given (or a path without a suffix), "
            "the formatted file is written as raw_<cache_stem>.txt inside that directory."
        ),
    )
    args = parser.parse_args()

    cache_path = Path(args.cache_file)
    with cache_path.open("r", encoding="utf-8") as fh:
        content = fh.read()

    entries = list(iter_json_objects(content))
    if not entries:
        print("No JSON objects found in the file.")
        return

    formatted = format_prompt_outputs(entries)
    output_path = resolve_output_path(cache_path, args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outf:
        outf.write(formatted)

    print(f"Formatted prompt outputs written to: {output_path}")

if __name__ == "__main__":
    main()
