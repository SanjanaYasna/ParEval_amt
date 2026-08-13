#!/usr/bin/env python3
"""
Generate ready-to-run curl batch commands (with Anthropic “thinking” enabled)
for multiple models from one or more JSON prompt definition files.

Each input JSON file must contain a list of objects with at least:
  - "name":   unique identifier per prompt (used as custom_id)
  - "prompt": user prompt text

Example usage:
    python make_batches.py prompts1.json prompts2.json \
        --out-dir batches/ --system "custom system text"

Outputs two shell snippets (claude_sonnet_4_5_batch.sh and
claude_opus_4_5_batch.sh by default). Each snippet is a full curl command
using a quoted here-document so apostrophes or other characters inside prompts
do not break the shell. The requests include:

    "thinking": {
        "type": "enabled"
        "budget_tokens": 1400
    }
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Any

DEFAULT_SYSTEM = (
    "You are an expert in high-performance computing and parallel programming. Generate efficient code for the requested function, without helper functions. Focus on performance optimization and correctness."
)

CURL_HEADER = """curl https://api.anthropic.com/v1/messages/batches \\
  --header "x-api-key: $ANTHROPIC_API_KEY" \\
  --header "anthropic-version: 2023-06-01" \\
  --header "content-type: application/json" \\
  --data @- <<'JSON'"""


def build_batch_payload(
    prompts: Iterable[Mapping[str, Any]],
    model: str,
    system_text: str,
    max_tokens: int,
    thinking_budget: int,
) -> dict[str, Any]:
    """Return the batch request payload for a given model."""
    requests = []
    for item in prompts:
        requests.append(
            {
                "custom_id": item["name"],
                "params": {
                    "model": model,
                    "max_tokens": max_tokens,
                    "top_p": 0.95,
                    "system": [
                        {
                            "type": "text",
                            "text": system_text,
                        }
                    ],
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": thinking_budget,
                    },
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": item["prompt"],
                                }
                            ],
                        }
                    ],
                },
            }
        )
    return {"requests": requests}


def load_prompt_specs(paths: Iterable[Path]) -> list[Mapping[str, Any]]:
    """Load and concatenate prompt specs from multiple JSON files."""
    prompts: list[Mapping[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path}: top-level JSON must be a list.")
        prompts.extend(data)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert prompt definitions into ready-to-run Anthropic curl batch commands."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more JSON files containing lists of prompt definitions.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory to write the shell snippets (default: current directory).",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help="System prompt to use for all requests.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="max_tokens value to use in each request (default: 2048).",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=1400,
        help="Budget (in tokens) for the Anthropic thinking block (default: 1400).",
    )
    args = parser.parse_args()

    prompt_specs = load_prompt_specs(args.inputs)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model in ("claude-sonnet-4-5", "claude-opus-4-5"):
        payload = build_batch_payload(
            prompt_specs,
            model=model,
            system_text=args.system,
            max_tokens=args.max_tokens,
            thinking_budget=args.thinking_budget,
        )
        json_blob = json.dumps(payload, indent=2)
        command = f"{CURL_HEADER}\n{json_blob}\nJSON\n"

        out_path = args.out_dir / f"{model.replace('-', '_')}_batch.sh"
        with open(out_path, "w", encoding="utf-8") as out_file:
            out_file.write(command)

        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()