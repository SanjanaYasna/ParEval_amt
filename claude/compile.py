#!/usr/bin/env python3
"""
Aggregate Claude batch request/response logs into a single JSON summary.

The script expects:
  * a batch *request script* (JSON/JSONL) describing every prompt (`--requests`)
  * a directory containing the batch *response* JSON/JSONL files (`--responses`)

For every logical prompt we:
  * strip the trailing `_copy_###` suffix from `custom_id`
  * collect every generation belonging to that prompt into a single `outputs` list
  * merge the request metadata (prompt text, sampling params, etc.)
  * emit one JSON array entry per prompt

Example usage
-------------
    python aggregate_batches.py --requests batch.json --responses outputs_dir > combined.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


COPY_ID_RE = re.compile(r"^(?P<base>.+?)_copy_(?P<idx>\d+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine Claude batch request/response files into one JSON summary."
    )
    parser.add_argument(
        "--requests",
        required=True,
        type=Path,
        help="Path to the batch request script (JSON or JSONL).",
    )
    parser.add_argument(
        "--responses",
        required=True,
        type=Path,
        help="Directory (or single file) containing response JSON/JSONL output.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Pretty-print JSON with the given indentation (default: 2).",
    )
    return parser.parse_args()


def iter_json_objects(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Yield every JSON object stored in `path`.

    Supports:
      * a single JSON object
      * a JSON array of objects
      * JSON Lines (one object per line)
    """
    text = path.read_text().strip()
    if not text:
        return

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
        return

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        yield data
    else:
        raise ValueError(f"Unsupported JSON top-level type in {path}: {type(data)}")


def split_custom_id(custom_id: str) -> Tuple[str, Optional[int]]:
    match = COPY_ID_RE.match(custom_id)
    if not match:
        return custom_id, None
    return match.group("base"), int(match.group("idx"))


def build_prompt(params: Dict[str, Any]) -> str:
    system_parts = []
    for item in params.get("system", []):
        if isinstance(item, dict) and item.get("type") == "text":
            system_parts.append(item.get("text", ""))

    user_parts = []
    for message in params.get("messages", []):
        if message.get("role") != "user":
            continue
        for chunk in message.get("content", []):
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                user_parts.append(chunk.get("text", ""))

    system_text = "\n".join(part.strip() for part in system_parts if part.strip())
    user_text = "\n".join(part.strip() for part in user_parts if part.strip())

    if system_text and user_text:
        return f"{system_text}\n{user_text}"
    return system_text or user_text


def infer_problem_type(name: str) -> str:
    parts = name.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    return "_".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "unknown")


def infer_language(prompt: str) -> str:
    lower = prompt.lower()
    if "#include" in prompt or "std::" in prompt or "c++" in lower:
        return "cpp"
    if "def " in lower or ("import " in lower and "python" in lower):
        return "python"
    if "function" in lower and "javascript" in lower:
        return "javascript"
    return "unknown"


def infer_parallelism(prompt: str) -> str:
    lower = prompt.lower()
    if "hpx" in lower:
        return "hpx"
    if "cuda" in lower:
        return "cuda"
    if "openmp" in lower:
        return "openmp"
    if "mpi" in lower:
        return "mpi"
    return "unknown"


def extract_text_blocks(content: Any) -> str:
    """
    Flatten the assistant message content into a single string, ignoring thinking traces.
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(filter(None, (t.strip() for t in texts)))

    if isinstance(content, dict) and content.get("type") == "text":
        return content.get("text", "").strip()

    return ""


def process_request(
    request: Dict[str, Any],
    request_data: Dict[str, Dict[str, Any]],
) -> None:
    custom_id = request.get("custom_id")
    if not custom_id:
        return
    base_name, _ = split_custom_id(custom_id)

    params = request.get("params", {})
    prompt = build_prompt(params)

    entry = request_data.setdefault(base_name, {})
    entry.setdefault("prompt", prompt)
    entry.setdefault("temperature", params.get("temperature", 0.2))
    entry.setdefault("top_p", params.get("top_p", 0.95))
    entry.setdefault("do_sample", params.get("do_sample", True))
    entry.setdefault("max_new_tokens", params.get("max_tokens"))
    entry.setdefault("prompted", params.get("prompted", False))

    entry.setdefault("problem_type", infer_problem_type(base_name))
    entry.setdefault("language", infer_language(prompt))
    entry.setdefault("parallelism_model", infer_parallelism(prompt))


def process_response(
    response: Dict[str, Any],
    response_data: Dict[str, List[Tuple[Optional[int], str]]],
) -> None:
    custom_id = response.get("custom_id")
    if not custom_id:
        return
    base_name, copy_idx = split_custom_id(custom_id)

    result = response.get("result", {})
    message = result.get("message", {})
    content = message.get("content", [])
    text = extract_text_blocks(content)

    response_data.setdefault(base_name, []).append((copy_idx, text))


def load_requests(request_path: Path) -> Dict[str, Dict[str, Any]]:
    if not request_path.exists():
        raise SystemExit(f"Request script not found: {request_path}")

    request_meta: Dict[str, Dict[str, Any]] = {}

    try:
        for obj in iter_json_objects(request_path):
            if "requests" in obj:
                for req in obj["requests"]:
                    process_request(req, request_meta)
            elif "custom_id" in obj and "params" in obj:
                process_request(obj, request_meta)
            else:
                continue
    except ValueError as exc:
        raise SystemExit(f"Failed to parse {request_path}: {exc}") from exc

    return request_meta


def load_responses(responses_path: Path) -> Dict[str, List[Tuple[Optional[int], str]]]:
    if not responses_path.exists():
        raise SystemExit(f"Responses path does not exist: {responses_path}")

    response_meta: Dict[str, List[Tuple[Optional[int], str]]] = {}

    paths: List[Path]
    if responses_path.is_file():
        paths = [responses_path]
    else:
        paths = sorted(p for p in responses_path.glob("**/*") if p.is_file())

    for path in paths:
        try:
            for obj in iter_json_objects(path):
                if "custom_id" in obj and "result" in obj:
                    process_response(obj, response_meta)
        except ValueError as exc:
            raise SystemExit(f"Failed to parse {path}: {exc}") from exc

    return response_meta


def main() -> None:
    args = parse_args()

    request_meta = load_requests(args.requests)
    response_meta = load_responses(args.responses)

    all_names = sorted(set(request_meta) | set(response_meta))

    summary: List[Dict[str, Any]] = []
    for name in all_names:
        req_info = request_meta.get(name, {})
        outputs_raw = response_meta.get(name, [])
        outputs_sorted = sorted(
            outputs_raw,
            key=lambda pair: (pair[0] is None, pair[0]),
        )
        outputs = [text for _, text in outputs_sorted if text]

        length = len(outputs)
        generation_times = ["N/A"] * length
        virtual_memory_used = ["N/A"] * length
        cpu_percent = ["N/A"] * length

        entry = {
            "problem_type": req_info.get("problem_type", infer_problem_type(name)),
            "language": "cpp",
            "name": name,
            "parallelism_model": req_info.get(
                "parallelism_model", infer_parallelism(req_info.get("prompt", ""))
            ),
            "prompt": req_info.get("prompt", ""),
            "temperature": "N/A",
            "top_p": 0.95,
            "do_sample": req_info.get("do_sample", True),
            "max_new_tokens": req_info.get("max_new_tokens"),
            "prompted": req_info.get("prompted", False),
            "outputs": outputs,
            "generation_times": generation_times,
            "virtual_memory_used": virtual_memory_used,
            "cpu_percent": cpu_percent,
        }

        summary.append(entry)

    print(json.dumps(summary, indent=args.indent, ensure_ascii=False))


if __name__ == "__main__":
    main()