import json
import os
import sys
import re

def find_func_declaration():
    pass
def main(file_path, output_json_path):
    try:
        with open(file_path) as file:
            generated = json.load(file)

        i= 2
        for output in generated:
            amt = output["parallelism_model"]
            prompt = output["name"]
            for output in output['raw_outputs']:
                print(output)
                i += 1
                if i == 2:
                    sys.exit()
                
            # clean_output = ((output["raw_outputs"][0]))

            # # Create new .c file for each prompt's generated output
            # with open(f"{amt}_{prompt}.c", "w") as outfile:
            #     outfile.writelines(clean_output)

    except FileNotFoundError:
        print(f"{file_path} not found")

main("/work/pi_mrobson_smith_edu/scratch/generation_hpx/la_test/cache/2048_tokens/hpc-coder_cache.json", "blah")