#!/usr/bin/env python3
import argparse
import json
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Iterable


def find_function_name_python(prompt: str):
    """Find the function name from Python prompt text (after 'def ' and before '(')."""
    if not isinstance(prompt, str):
        return None
    # Look for 'def function_name(' pattern
    match = re.search(r'\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', prompt)
    if match:
        return match.group(1)
    return None


def clean_python(out: str, func_name: str):
    """
    Remove lines starting with @@ (like @@ Response) and extract the Python
    function definition for func_name (including body). If func_name not found,
    just return the cleaned text.
    """
    if out is None:
        return out
    if not isinstance(out, str):
        out = str(out)

    # Remove @@ lines
    out_cleaned = re.sub(r"(?m)^\s*@@.*\n?", "", out)

    if not func_name:
        return out_cleaned.strip()

    # Find the function definition line
    # Pattern: def func_name(...) : or def func_name(...):
    func_pattern = rf"\bdef\s+{re.escape(func_name)}\s*\("
    match = re.search(func_pattern, out_cleaned)
    if not match:
        return out_cleaned.strip()

    # Find the start of the function definition line
    start_idx = out_cleaned.rfind("\n", 0, match.start()) + 1
    if start_idx < 0:
        start_idx = 0

    # Find the matching colon for the function definition
    # Need to handle nested parentheses in type hints
    paren_depth = 0
    colon_idx = -1
    for i in range(match.start(), len(out_cleaned)):
        ch = out_cleaned[i]
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch == ":" and paren_depth == 0:
            colon_idx = i
            break
    
    if colon_idx == -1:
        return out_cleaned[start_idx:].strip()

    # Check if the body is on the same line as the def (inline function)
    # Look at what comes after the colon
    after_colon = out_cleaned[colon_idx + 1:].lstrip()
    
    # If there's content after the colon on the same line, it's an inline function
    def_line_end = colon_idx + 1
    while def_line_end < len(out_cleaned) and out_cleaned[def_line_end] != '\n':
        def_line_end += 1
    
    inline_content = out_cleaned[colon_idx + 1:def_line_end].strip()
    
    if inline_content:
        # Inline function like: def func(): return x
        # Return just the def line
        def_line = out_cleaned[start_idx:def_line_end].strip()
        return def_line

    # Body is on subsequent lines - extract indented block
    after_def = out_cleaned[def_line_end:]
    lines = after_def.splitlines()
    if not lines:
        return out_cleaned[start_idx:def_line_end].strip()

    # Determine the indentation of the first body line
    # Skip empty lines
    body_lines = []
    base_indent = None
    
    for line in lines:
        stripped = line.lstrip()
        if not stripped:  # Empty line
            continue
        # Get the indentation
        indent = len(line) - len(stripped)
        if base_indent is None:
            base_indent = indent
            body_lines.append(line)
        elif indent > base_indent:
            # Deeper indentation - still in body
            body_lines.append(line)
        elif indent == base_indent:
            # Same indentation - still in body
            body_lines.append(line)
        elif indent < base_indent and base_indent > 0:
            # We've dedented - end of function
            break
        else:
            # No indentation at all - end of function
            break

    if not body_lines:
        return out_cleaned[start_idx:def_line_end].strip()

    # Reconstruct: def line + body lines
    def_line = out_cleaned[start_idx:def_line_end].rstrip()
    body = "\n".join(body_lines)
    
    # Check if body starts with proper indentation
    if body_lines:
        first_body_indent = len(body_lines[0]) - len(body_lines[0].lstrip())
        if first_body_indent <= 0:
            # Body isn't indented - malformed
            return out_cleaned[start_idx:].strip()

    result = def_line + "\n" + body if body.strip() else def_line
    return result.strip()


def find_function_name(prompt: str):
    """Find the function name from the prompt text."""
    if not isinstance(prompt, str):
        return None
    
    # Try C-style first (for backwards compatibility)
    start = 0
    comment_end = prompt.find("*/")
    if comment_end != -1:
        start = comment_end + 2
    paren = prompt.find("(", start)
    if paren != -1:
        i = paren - 1
        while i >= start and prompt[i].isspace():
            i -= 1
        if i >= start:
            j = i
            while j >= start and (prompt[j].isalnum() or prompt[j] in "_:"):
                j -= 1
            fname = prompt[j + 1 : i + 1]
            if fname:
                return fname
    
    # Try Python style
    return find_function_name_python(prompt)


def clean(out: str, func_name: str):
    """
    Remove lines starting with @@ and extract the function definition.
    Uses Python-aware parsing for .py prompts, falls back to C-style for others.
    """
    if out is None:
        return out
    if not isinstance(out, str):
        out = str(out)

    out_cleaned = re.sub(r"(?m)^\s*@@.*\n?", "", out)

    if not func_name:
        return out_cleaned.strip()

    # Try Python-style cleaning first if func_name looks Python-like
    # Check if the output contains 'def ' which suggests Python
    if "def " in out_cleaned:
        result = clean_python(out_cleaned, func_name)
        # If we got a non-trivial result with def in it, use it
        if result and "def " in result:
            return result
    
    # Fall back to C-style brace matching
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
    """Check whether this prompt is the FFT 05 special case."""
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
        # 05 override
        if _is_ifft_problem(prompt):
            func_name = "ifft"

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
            "bodies. Supports both Python (indentation-based) and C-style (brace-based) "
            "languages. If INPUT is a directory, every *.json in that directory is processed."
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
        output_dir = json_file.parent / "cleaned_python"
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
