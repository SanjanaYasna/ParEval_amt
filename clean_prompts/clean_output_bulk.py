#!/usr/bin/env python3
import argparse
import json
import os
import re
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Iterable, Tuple


def find_function_name(prompt: str):
    """Find the function name from the prompt text (token immediately before the first '(' after the comment end)."""
    if not isinstance(prompt, str):
        return None
    start = 0
    comment_end = prompt.find("*/")
    if comment_end != -1:
        start = comment_end + 2
    paren = prompt.find("(", start)
    if paren == -1:
        return None
    i = paren - 1
    while i >= start and prompt[i].isspace():
        i -= 1
    if i < start:
        return None
    j = i
    while j >= start and (prompt[j].isalnum() or prompt[j] in "_:"):
        j -= 1
    fname = prompt[j + 1 : i + 1]
    return fname or None


def clean(out: str, func_name: str):
    """
    Remove lines starting with @@ (like @@ Response) and extract the function
    definition for func_name (including body). If func_name not found, just
    return the cleaned text.
    """
    if out is None:
        return out
    if not isinstance(out, str):
        out = str(out)

    out_cleaned = re.sub(r"(?m)^\s*@@.*\n?", "", out)

    if not func_name:
        return out_cleaned.strip()

    match = re.search(r"\b" + re.escape(func_name) + r"\s*\(", out_cleaned)
    if not match:
        return out_cleaned.strip()

    start_idx = out_cleaned.rfind("\n", 0, match.start()) + 1
    if start_idx < 0:
        start_idx = 0

    brace_idx = out_cleaned.find("{", match.end())
    if brace_idx == -1:
        return out_cleaned[start_idx:].strip()

    depth = 0
    i = brace_idx
    L = len(out_cleaned)
    while i < L:
        ch = out_cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return out_cleaned[start_idx : i + 1].strip()
        i += 1

    return out_cleaned[start_idx:].strip()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except JSONDecodeError:
        decoder = json.JSONDecoder()
        objs = []
        idx = 0
        L = len(raw)
        while True:
            while idx < L and raw[idx].isspace():
                idx += 1
            if idx >= L:
                break
            obj, end = decoder.raw_decode(raw, idx)
            objs.append(obj)
            idx = end
        return objs


def find_prompts_container(parsed):
    """
    Return the list of prompts (each should be a dict) to process.
    If parsed is already a list, return it.
    If parsed is a dict, try common keys that contain lists of prompts.
    Otherwise, if parsed looks like a single prompt dict, return [parsed].
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in (
            "outputs",
            "results",
            "items",
            "candidates",
            "generation_results",
            "prompts",
        ):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        if (
            "prompt" in parsed
            or "outputs" in parsed
            or "generated_output" in parsed
            or "generated_outputs" in parsed
        ):
            return [parsed]
    return None

def _is_ifft_problem(prompt: dict) -> bool:
    """Check whether this prompt is the FFT 05 special case, in which case ignore forward declared fft function"""
    return (
        prompt.get("problem_type") == "fft"
        and prompt.get("name") == "05_fft_inverse_fft"
    )
    

def process_file(input_path: Path, output_path: Path):
    parsed = load_json(str(input_path))
    prompts = find_prompts_container(parsed)
    if prompts is None:
        raise RuntimeError(f"Could not locate list of prompts in {input_path}")

    if isinstance(parsed, dict):
        for key in (
            "outputs",
            "results",
            "items",
            "candidates",
            "generation_results",
            "prompts",
        ):
            if key in parsed and isinstance(parsed[key], list):
                prompts = parsed[key]
                break

    for prompt in prompts:
        if not isinstance(prompt, dict):
            continue
        prompt_text = prompt.get("prompt", "") or prompt.get("instruction", "")
        func_name = find_function_name(prompt_text)
        #05 override
        if _is_ifft_problem(prompt):
            func_name =  "ifft"

        if "outputs" in prompt and isinstance(prompt["outputs"], list):
            for i, out in enumerate(prompt["outputs"]):
                try:
                    prompt["outputs"][i] = clean(out, func_name)
                except Exception:
                    prompt["outputs"][i] = re.sub(
                        r"(?m)^\s*@@.*\n?", "", str(out)
                    ).strip()
        elif "generated_outputs" in prompt and isinstance(
            prompt["generated_outputs"], list
        ):
            for i, out in enumerate(prompt["generated_outputs"]):
                try:
                    prompt["generated_outputs"][i] = clean(out, func_name)
                except Exception:
                    prompt["generated_outputs"][i] = re.sub(
                        r"(?m)^\s*@@.*\n?", "", str(out)
                    ).strip()
        elif "generated_output" in prompt and isinstance(
            prompt["generated_output"], str
        ):
            try:
                prompt["generated_output"] = clean(
                    prompt["generated_output"], func_name
                )
            except Exception:
                prompt["generated_output"] = re.sub(
                    r"(?m)^\s*@@.*\n?", "", prompt["generated_output"]
                ).strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)


def iter_input_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_dir():
        yield from sorted(p for p in input_path.glob("*.json") if p.is_file())
    else:
        yield input_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clean generated outputs by removing '@@' lines and extracting function "
            "bodies. If INPUT is a directory, every *.json in that directory is processed."
        )
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input JSON file or directory of JSON files"
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    processed_any = False
    for json_file in iter_input_files(input_path):
        processed_any = True
        output_dir = json_file.parent / "cleaned"
        output_path = output_dir / json_file.name
        try:
            process_file(json_file, output_path)
            print(f"Cleaned file written to: {output_path}")
        except Exception as exc:
            print(f"Error processing {json_file}: {exc}")

    if not processed_any:
        print(f"No JSON files found in {input_path}")


if __name__ == "__main__":
    main()
