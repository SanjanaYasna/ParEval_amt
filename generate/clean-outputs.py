from argparse import ArgumentParser
import json
import sys
parser = ArgumentParser(description='clean model outputs')
parser.add_argument('--input', required=True, help='Path to the input JSON file')
parser.add_argument('--output', required=False, help='Path to the output JSON file')
args = parser.parse_args()


import json
import os
import re
from typing import Optional

import re
from typing import Optional

def get_first_implementation(src: str, signature: str) -> Optional[str]:
    """
    Return the first implementation (including the signature and braces) of the
    function whose signature matches `signature`. If the signature is not found,
    return None.
    """
    pattern = re.escape(signature) + r'\s*\{'
    match = re.search(pattern, src)
    if not match:
        return None

    func_start = match.start()
    brace_start = match.end() - 1  # index of the '{'
    depth = 0

    for i in range(brace_start, len(src)):
        ch = src[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return src[func_start:i + 1]

    # Unbalanced braces
    return None

def find_signature(prompt: str) -> str | None:
    """
    Extract the C/C++ function signature that precedes an opening brace.
    Returns the signature without the trailing brace or leading/trailing spaces.
    """
    match = re.search(r'^\s*([^\n{]+\([^\n]*\))\s*\{', prompt, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def main():
    # Add correct input and output directories
    input_path = args.input
    if not args.output:
        args.output = input_path
    output_file = os.path.join(args.output, f'{args.output.split(".")[0]}_cleaned.json')
    print(output_file)
    # Make output directory if doesn't exist
    # Copy original JSON into output file
    os.system(f'cp "{input_path}" "{output_file}"')
    outputs = []
    with open(output_file) as file:
        for line in file:
            cleaned_out = []
            line = json.loads(line)
            generated = line["outputs"]
            signature = find_signature(generated[0])
            #print(signature)
            for i in range(0, len(generated)): 
                cleaned = get_first_implementation(generated[i], signature)
                if cleaned is None:
                    cleaned = "None"
                cleaned_out.append(cleaned)
            line["outputs"] = cleaned_out
            outputs.append(line)
    with open(output_file, 'w') as file:
        json.dump(outputs, file)
main()
