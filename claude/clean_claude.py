#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, List, Optional

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

CODE_FENCE_OPEN = re.compile(r'^```[^\n]*\n?', flags=re.MULTILINE)
CODE_FENCE_CLOSE = re.compile(r'```', flags=re.MULTILINE)

# Match a C++ function (optionally templated) whose parameter list may span lines.
FUNC_SIG_PATTERN = re.compile(
    r"""
    ^\s*                                     # leading whitespace
    (?:template\s*<[^>]+>\s*)*               # optional template line(s)
    (?:                                     # optional qualifiers (each may be multi-word)
        (?:inline|constexpr|static|friend|typename)\s+
    )*
    (?:[\w:\<\>\*\&\s]+?)                    # return type, qualifiers, etc.
    [A-Za-z_~][\w:\<\>]*                     # function name (allow dtor/qualified)
    \s*\(                                    # opening paren for params
        (?:[^()]|\([^()]*\))*               # parameters; allow balanced (...) inside
    \)
    \s*
    (?:const\s*)?                            # optional const
    (?:noexcept(?:\s*\([^)]*\))?\s*)?        # optional noexcept or noexcept(expr)
    (?:->\s*[\w:\<\>\s\*&]+)?                # optional trailing return type
    \s*\{                                    # opening brace
    """,
    re.MULTILINE | re.VERBOSE
)


def strip_code_fences(text: str) -> str:
    """Remove Markdown code fences but leave the code body intact."""
    text = CODE_FENCE_OPEN.sub('', text)
    text = CODE_FENCE_CLOSE.sub('', text)
    return text


def strip_includes(text: str) -> str:
    """Remove lines that start with #include (ignores leading whitespace)."""
    return "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("#include")
    )


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
                return src[func_start: idx + 1]

    return None  # Unbalanced braces


def find_signature(snippet: str) -> Optional[str]:
    """Locate the first function signature (before the opening brace)."""
    match = FUNC_SIG_PATTERN.search(snippet)
    if not match:
        return None

    sig_with_brace = match.group(0)
    brace_idx = sig_with_brace.rfind("{")
    signature = sig_with_brace[:brace_idx].rstrip()
    return signature


def iter_json_records(text: str) -> Iterable[dict]:
    """
    Yield JSON objects from either a JSON array or newline-delimited JSON.
    """
    stripped = text.strip()
    if not stripped:
        return []

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

    for line in stripped.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def clean_outputs(record: dict) -> dict:
    outputs: List[str] = record.get("outputs") or []
    if not outputs:
        record["outputs"] = []
        return record

    sanitized = [strip_code_fences(out) for out in outputs]

    signature = None
    for snip in sanitized:
        signature = find_signature(snip)
        if signature:
            break

    cleaned: List[str]
    if signature:
        cleaned = []
        for original in outputs:
            working = strip_code_fences(original)
            impl = get_first_implementation(working, signature)
            if impl is not None:
                impl = strip_includes(strip_code_fences(impl)).strip()
                cleaned.append(impl if impl else "None")
            else:
                cleaned.append("None")
    else:
        cleaned = ["None"] * len(outputs)

    record["outputs"] = cleaned
    return record


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

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