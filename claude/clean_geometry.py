#!/usr/bin/env python3
"""
Clean JSON outputs by removing headers and code fences, keeping only function implementations.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def clean_code_output(code: str) -> str:
    """
    Remove #include statements, code fences, and other headers from code output.
    Keep only function implementations.
    """
    if not isinstance(code, str):
        return code
    
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip code fence markers
        if stripped.startswith('```'):
            continue
        
        # Skip #include statements
        if stripped.startswith('#include'):
            continue
        
        # Skip namespace declarations at file scope (optional)
        # if stripped.startswith('namespace') and '{' in stripped:
        #     continue
        
        cleaned_lines.append(line)
    
    # Join and strip leading/trailing whitespace
    result = '\n'.join(cleaned_lines).strip()
    
    # Remove leading namespace/struct declarations if present
    # (keep only if they're part of the function body)
    
    return result


def clean_json_outputs(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean all outputs in the JSON data structure.
    """
    for entry in data:
        if 'outputs' in entry and isinstance(entry['outputs'], list):
            entry['outputs'] = [clean_code_output(output) for output in entry['outputs']]
    
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Input JSON file')
    parser.add_argument('-o', '--output', type=Path, help='Output JSON file (default: input_cleaned.json)')
    args = parser.parse_args()
    
    # Load input JSON
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Clean outputs
    cleaned_data = clean_json_outputs(data)
    
    # Determine output path
    output_path = args.output or args.input.with_name(f"{args.input.stem}_cleaned.json")
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    print(f"Cleaned JSON written to {output_path}")


if __name__ == '__main__':
    main()