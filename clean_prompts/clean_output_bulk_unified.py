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


def extract_markdown_code(text: str) -> str:
    """Extract code from markdown code blocks (```python ... ```) and remove common indentation."""
    if not text:
        return text
    
    # Find all ```python ... ``` blocks
    # Pattern: ```python\ncode\n``` or ```\ncode\n```
    pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if not matches:
        return text
    
    # Process each match to remove common indentation
    processed_blocks = []
    for block in matches:
        lines = block.splitlines()
        if not lines:
            processed_blocks.append(block)
            continue
        
        # Find minimum non-empty line indentation
        indents = []
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                indents.append(indent)
        
        if indents:
            min_indent = min(indents)
            # Remove common indentation
            dedented = []
            for line in lines:
                if line.strip():  # Non-empty line
                    dedented.append(line[min_indent:] if len(line) >= min_indent else line)
                else:
                    dedented.append('')  # Keep empty lines
            processed_blocks.append('\n'.join(dedented))
        else:
            processed_blocks.append(block)
    
    return '\n'.join(processed_blocks)


def extract_chapel_code(text: str) -> str:
    """Extract code from chapel code blocks (```chapel ... ```)."""
    if not text:
        return text
    
    # Find chapel code blocks
    code_block_match = re.search(r"```(?:chapel)?\s*([\s\S]*?)```", text)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    return text


def clean_python_from_text(out: str, func_name: str):
    """
    Extract function from text using indentation tracking.
    """
    if not func_name:
        return out.strip()

    # Find the function definition line
    func_pattern = rf"\bdef\s+{re.escape(func_name)}\s*\("
    match = re.search(func_pattern, out)
    if not match:
        return out.strip()

    # Find the matching colon for the function definition
    # Need to handle nested parentheses in type hints
    paren_depth = 0
    colon_idx = -1
    for i in range(match.start(), len(out)):
        ch = out[i]
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch == ":" and paren_depth == 0:
            colon_idx = i
            break
    
    if colon_idx == -1:
        return out.strip()

    # Find the start of the function definition line (beginning of the line)
    def_start = out.rfind('\n', 0, match.start()) + 1
    
    # Check if the body is on the same line as the def (inline function)
    def_line_end = colon_idx + 1
    while def_line_end < len(out) and out[def_line_end] != '\n':
        def_line_end += 1
    
    inline_content = out[colon_idx + 1:def_line_end].strip()
    
    if inline_content:
        # Inline function like: def func(): return x
        def_line = out[def_start:def_line_end].strip()
        # Check if this is the only def line or if there's more
        # Look for other def lines in the text
        all_defs = list(re.finditer(rf"\bdef\s+{re.escape(func_name)}\s*\(", out))
        if len(all_defs) > 1:
            # There are multiple definitions - find the one with actual implementation
            for m in reversed(all_defs):
                # Look ahead for indented content
                search_start = m.end()
                remaining = out[search_start:]
                if '\n    ' in remaining or '\n\t' in remaining:
                    # This one has indented content - use it
                    return clean_python_from_text(out[search_start:], func_name)
        return def_line

    # Body is on subsequent lines - extract indented block
    after_def = out[def_line_end:]
    lines = after_def.splitlines()
    if not lines:
        return out[def_start:def_line_end].strip()

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
        return out[def_start:def_line_end].strip()

    # Reconstruct: def line + body lines
    def_line = out[def_start:def_line_end].rstrip()
    body = "\n".join(body_lines)
    
    # Check if body starts with proper indentation
    if body_lines:
        first_body_indent = len(body_lines[0]) - len(body_lines[0].lstrip())
        if first_body_indent <= 0:
            # Body isn't indented - malformed
            return out.strip()

    result = def_line + "\n" + body if body.strip() else def_line
    return result.strip()


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
        # Try to extract from markdown if present
        code = extract_markdown_code(out_cleaned)
        if code and code != out_cleaned:
            return code.strip()
        return out_cleaned.strip()

    # Try to extract from markdown code blocks first
    code = extract_markdown_code(out_cleaned)
    
    if code and code != out_cleaned:
        # We found markdown code blocks - work with the extracted code
        result = clean_python_from_text(code, func_name)
        if result:
            return result
        # If no function found in markdown, try the full text
        return clean_python_from_text(out_cleaned, func_name)
    
    # No markdown blocks, use the cleaned text directly
    return clean_python_from_text(out_cleaned, func_name)


def clean(out: str, func_name: str):
    """
    Remove lines starting with @@ and extract the function definition.
    Handles both Python (indentation-based) and C-style (brace-based) languages.
    For gemma format: extracts from ```python blocks
    For magicoder format: handles @@ Instruction/Response and extracts code
    """
    if out is None:
        return out
    if not isinstance(out, str):
        out = str(out)

    out_cleaned = re.sub(r"(?m)^\s*@@.*\n?", "", out)

    if not func_name:
        # Try to extract from markdown (both python and chapel)
        code = extract_markdown_code(out_cleaned)
        if code and code != out_cleaned:
            return code.strip()
        # Try chapel format
        code = extract_chapel_code(out_cleaned)
        if code and code != out_cleaned:
            return code.strip()
        return out_cleaned.strip()

    # Try Python-style cleaning first if output contains 'def ' or markdown
    if "def " in out_cleaned or "```" in out_cleaned:
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
                    # Handle both string outputs and dict outputs with 'content' field
                    if isinstance(out, dict):
                        output_text = out.get("content", "")
                    else:
                        output_text = out
                    prompt["outputs"][i] = clean(output_text, func_name)
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
            "Clean generated outputs by removing '@@' lines, extracting from markdown "
            "code blocks, and extracting function bodies. Handles both gemma "
            "(direct code in markdown) and magicoder (instruction/response with markdown) formats. "
            "If INPUT is a directory, every *.json in that directory is processed."
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
        output_dir = json_file.parent / "cleaned_unified"
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