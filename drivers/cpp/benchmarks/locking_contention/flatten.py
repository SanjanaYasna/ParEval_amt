#!/usr/bin/env python3
"""
Flatten a commented code listing into a single-line, newline-escaped string.

Usage:
    python make_single_line.py path/to/input.txt

The script writes a file alongside the input called
`<stem>_single_line.txt` containing the flattened output.
"""

import argparse
import re
from pathlib import Path


def strip_comment_prefix(line: str, in_block: bool) -> tuple[str, bool]:
    """
    Remove leading comment markers from a line.

    Returns the cleaned line and updated in_block state.
    Supports // comments and simple /* ... */ blocks.
    """
    text = line.rstrip("\n")

    if in_block:
        end = text.find("*/")
        if end != -1:
            snippet = text[:end]
            in_block = False
        else:
            snippet = text
        return snippet.lstrip(" *"), in_block

    match_slash = re.match(r"\s*//\s?(.*)$", text)
    if match_slash:
        return match_slash.group(1), in_block

    start = text.find("/*")
    if start != -1:
        after = text[start + 2 :]
        end = after.find("*/")
        if end != -1:
            snippet = after[:end]
            in_block = False
        else:
            snippet = after
            in_block = True
        return snippet.lstrip(" *"), in_block

    return text, in_block


def flatten_file(input_path: Path) -> Path:
    lines = input_path.read_text().splitlines()
    in_block = False
    stripped_lines = []

    for line in lines:
        cleaned, in_block = strip_comment_prefix(line, in_block)
        stripped_lines.append(cleaned.rstrip())

    escaped_lines = [
        line.replace("\\", "\\\\").replace('"', '\\"') for line in stripped_lines
    ]
    flattened = "\\n".join(escaped_lines)

    output_path = input_path.with_name(f"{input_path.stem}_single_line.txt")
    output_path.write_text(flattened)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten commented code into a single-line string with newline escapes."
    )
    parser.add_argument("input", type=Path, help="Path to the input .txt file")
    args = parser.parse_args()

    input_path = args.input
    if not input_path.is_file():
        parser.error(f"No such file: {input_path}")

    output_path = flatten_file(input_path)
    print(f"Wrote flattened content to {output_path}")


if __name__ == "__main__":
    main()