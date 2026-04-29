#!/usr/bin/env python3
"""
send_batches.py -- submit an expanded Claude batch in chunks, starting at an optional offset.

Example:
    python send_batches.py \
        --batch batch_x100.json \
        --offset 1000 \
        --chunk-size 1000 \
        --sleep 300
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import requests

# Default Anthropic batches endpoint
ANTHROPIC_BATCH_ENDPOINT = "https://api.anthropic.com/v1/messages/batches"


def load_requests(batch_path: Path) -> List[dict]:
    """
    Load the list of requests from the expanded batch file.
    The file must be a JSON object with a "requests" array.
    """
    try:
        data = json.loads(batch_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse batch JSON: {exc}") from exc

    requests_list = data.get("requests", [])
    if not requests_list:
        raise RuntimeError("No requests found in batch file (missing 'requests' array).")

    return requests_list


def send_chunk(payload: dict, api_key: str, timeout: int = 60) -> None:
    """
    POST the payload to Anthropic's batch endpoint.
    Raises RuntimeError on non-success responses.
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    resp = requests.post(ANTHROPIC_BATCH_ENDPOINT, headers=headers, json=payload, timeout=timeout)

    if not resp.ok:
        raise RuntimeError(f"Batch submission failed: {resp.status_code} {resp.text}")

    try:
        resp_json = resp.json()
    except ValueError:
        print("Submitted chunk; non-JSON response returned.")
        return

    print(
        "Submitted chunk with",
        len(payload["requests"]),
        "requests; response id =",
        resp_json.get("id", "<unknown>"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Claude batch requests in rate-limited chunks.")
    parser.add_argument(
        "--batch",
        required=True,
        help="Path to expanded batch JSON (e.g., output of expand_batch.py).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of requests per chunk (default: 1000).",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=60,
        help="Delay between chunks in seconds (default: 300 = 5 minutes).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of requests already sent (skip this many).",
    )
    parser.add_argument(
        "--endpoint",
        default=ANTHROPIC_BATCH_ENDPOINT,
        help="Override the Anthropic batch endpoint (for testing).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP request timeout in seconds (default: 60).",
    )
    args = parser.parse_args()

    batch_path = Path(args.batch)
    if not batch_path.exists():
        raise SystemExit(f"Batch file not found: {batch_path}")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Please set ANTHROPIC_API_KEY in your environment before running this script.")

    requests_list = load_requests(batch_path)
    total = len(requests_list)
    print(f"Loaded {total} requests from {batch_path}")

    start_index = min(max(args.offset, 0), total)
    if args.offset != start_index:
        print(f"Adjusted offset to {start_index} (within valid range).")
    if start_index >= total:
        print("Nothing to send: offset >= total number of requests.")
        return

    print(f"Beginning submission at offset {start_index} (chunk size = {args.chunk_size})")

    chunk_number = start_index // args.chunk_size + 1
    for i in range(start_index, total, args.chunk_size):
        chunk = requests_list[i : i + args.chunk_size]
        payload = {"requests": chunk}

        print(f"Submitting chunk #{chunk_number} ({len(chunk)} requests, indices {i}–{i + len(chunk) - 1})")

        try:
            send_chunk(payload, api_key=api_key, timeout=args.timeout)
        except Exception as exc:
            print(f"Error while submitting chunk {chunk_number}: {exc}")
            print("Stopping; adjust --offset accordingly and rerun once ready.")
            return

        next_index = i + args.chunk_size
        if next_index < total:
            print(f"Sleeping {args.sleep} seconds before next chunk...")
            time.sleep(args.sleep)

        chunk_number += 1

    print("All chunks submitted successfully.")


if __name__ == "__main__":
    main()