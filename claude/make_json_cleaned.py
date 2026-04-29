import json
import sys
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------

def parse_records(stream: str) -> List[dict]:
    """Parse one or more consecutive JSON objects from a string."""
    records = []
    decoder = json.JSONDecoder()
    idx = 0
    length = len(stream)

    while idx < length:
        while idx < length and stream[idx].isspace():
            idx += 1
        if idx >= length:
            break

        obj, offset = decoder.raw_decode(stream, idx)
        records.append(obj)
        idx = offset

    return records


def extract_outputs(content: List[dict]) -> List[str]:
    """Pull the final assistant text snippets out of a message content array."""
    return [
        item["text"].strip()
        for item in content
        if item.get("type") == "text"
    ]


def load_prompts_from_shell(batch_path: str) -> Dict[str, str]:
    """Return {custom_id: prompt_text} scraped from the batch .sh file."""
    if not batch_path:
        return {}

    try:
        with open(batch_path, "r", encoding="utf-8") as f:
            shell_text = f.read()
    except OSError:
        return {}

    start = shell_text.find("{")
    if start == -1:
        return {}

    json_candidate = shell_text[start:]
    end = json_candidate.rfind("}")
    if end == -1:
        return {}

    json_text = json_candidate[: end + 1]

    try:
        batch_objects = parse_records(json_text)
    except json.JSONDecodeError:
        try:
            batch_objects = [json.loads(json_text)]
        except json.JSONDecodeError:
            return {}

    prompts: Dict[str, str] = {}

    for batch in batch_objects:
        for request in batch.get("requests", []):
            custom_id = request.get("custom_id")
            if not custom_id:
                continue

            params = request.get("params", {})
            text_pieces: List[str] = []

            for item in params.get("system", []):
                if item.get("type") == "text":
                    text_pieces.append(item.get("text", ""))

            for message in params.get("messages", []):
                for item in message.get("content", []):
                    if item.get("type") == "text":
                        text_pieces.append(item.get("text", ""))

            prompts[custom_id] = "\n".join(text_pieces)

    return prompts


def load_output_records(outputs_path: str) -> List[dict]:
    """Read the JSON/JSONL outputs file into a list of objects."""
    if not outputs_path:
        return []

    with open(outputs_path, "r", encoding="utf-8") as f:
        raw = f.read()

    if not raw.strip():
        return []

    try:
        return parse_records(raw)
    except json.JSONDecodeError:
        try:
            return [json.loads(raw)]
        except json.JSONDecodeError:
            raise SystemExit("Unable to parse the outputs JSON file.")


# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python convert.py path/to/batch.sh path/to/outputs.jsonl"
        )

    batch_path = sys.argv[1]
    outputs_path = sys.argv[2]

    prompt_map = load_prompts_from_shell(batch_path)
    records = load_output_records(outputs_path)

    converted = []

    for record in records:
        custom_id = record.get("custom_id", "")
        result = record.get("result", {})
        message = result.get("message", {})
        content = message.get("content", [])

        outputs = extract_outputs(content)
        if not outputs:
            continue

        parts = custom_id.split('_')
        problem_type = parts[1] if len(parts) > 1 else custom_id

        entry = {
            "problem_type": problem_type,
            "language": "cpp",
            "name": custom_id,
            "parallelism_model": "hpx",
            "prompt": prompt_map.get(custom_id, ""),
            "temperature": 0.2,
            "top_p": 0.95,
            "do_sample": True,
            "max_new_tokens": 2048,
            "prompted": False,
            "outputs": outputs,
            "generation_times": ["N/A"] * len(outputs),
            "virtual_memory_used": ["N/A"] * len(outputs),
            "cpu_percent": ["N/A"] * len(outputs),
        }

        converted.append(entry)

    output_file = Path(outputs_path)
    stem = output_file.stem
    cleaned_path = output_file.with_name(f"{stem}_cleaned.json")
    
    with open(cleaned_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2)

    print(f"Wrote {cleaned_path}")


if __name__ == "__main__":
    main()