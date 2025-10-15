# std imports
import argparse
import json
import os
import sys
import time
from tqdm import tqdm
import psutil
# tpl imports
import torch
from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, AutoTokenizer, AutoModelForCausalLM
# local imports
from utils import BalancedBracketsCriteria, PromptDataset, clean_output, get_inference_config


""" Parse command line arguments """
parser = argparse.ArgumentParser(description='Generate code')
parser.add_argument('--prompts', required=True, help='Path to the prompt JSON file')
parser.add_argument('--model_names', required=True, help='Model Name', nargs='+')
parser.add_argument('--output', required=True, help='Path to the output JSON file')
parser.add_argument('--restart', action='store_true', help='Restart generation from scratch (default: False)', default=False)
parser.add_argument('--cache', help='JSONL file to cache intermediate results in. Will be restored from if it ' +
    'already exists and --restart is not specified')
parser.add_argument('--num_cores_used', type=int, help='Number of CPU cores to use for generation', default=None)
parser.add_argument('--restore_from', help='JSON file to restore old results from. Will be restored from ' +
    'if it already exists and --restart is not specified. Is different from --cache in that it is a JSON file, not a ' +
    'JSONL file, and it is only used to restore old results where the prompt is equivalent. Cached results are ' +
    'prioritized over restored results.')
parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum number of new tokens to generate (default: 1024)')
parser.add_argument('--num_samples_per_prompt', type=int, default=50, help='Number of code samples to generate (default: 50)')
parser.add_argument('--temperature', type=float, default=0.2, help='Temperature for controlling randomness (default: 0.2)')
parser.add_argument('--top_p', type=float, default=0.95, help='Top p value for nucleus sampling (default: 0.95)')
parser.add_argument('--do_sample', action='store_true', help='Enable sampling (default: False)')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generation (default: 8)')
parser.add_argument('--prompted', action='store_true', help='Use prompted generation. See StarCoder paper (default: False)')
parser.add_argument('--hf_token', type=str, help='HuggingFace API token for loading models')
args = parser.parse_args()

""" Load prompts """
with open(args.prompts, 'r') as json_file:
    prompts = json.load(json_file)
    

def load_model(model_name):
    try:
        if model_name == 'gpt2':
            model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
            tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
        elif model_name == 'gpt-neo':
            model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')
            tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
        elif model_name == 'poly-coder':
            model = AutoModelForCausalLM.from_pretrained('NinedayWang/PolyCoder-2.7B')
            tokenizer = AutoTokenizer.from_pretrained('NinedayWang/PolyCoder-2.7B')
        elif model_name == 'hpc-coder':
            model = AutoModelForCausalLM.from_pretrained('hpcgroup/hpc-coder-v2-6.7b')
            tokenizer = AutoTokenizer.from_pretrained('hpcgroup/hpc-coder-v2-6.7b')
        elif model_name == 'meta-llama':
            model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B', token=hf_token)
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', token=hf_token)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        #if gpu available, put model to gpu
        if torch.cuda.is_available():
            model = model.to('cuda')
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None
    
def generate_code(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors='pt')

    start_time = time.time()

    # Get initial metrics
    start_memory = psutil.virtual_memory().used
    start_cpu = psutil.cpu_percent(interval=None)

    outputs = model.generate(inputs['input_ids']
                            , max_length=600
                            , temperature=args.temperature
                            , top_p = args.top_p
                            , do_sample = args.do_sample) #increased from 200  to avoid incompletion due to restriction
    end_time = time.time()
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generation_time = end_time - start_time

    # Get final metrics
    end_memory = psutil.virtual_memory().used
    end_cpu = psutil.cpu_percent(interval=None)

    memory_used = end_memory - start_memory
    cpu_used = end_cpu - start_cpu

    return generated_code, generation_time, memory_used, cpu_used


models = ['gpt2', 'gpt-neo', 'poly-coder', 'hpc-coder', 'meta-llama']
prompts_repeated = [p for p in prompts for _ in range(args.num_samples_per_prompt)]


cur_prompt = None
for model_name in args.model_names:
    results = []
    #if gpu, put model to gpu
    model, tokenizer = load_model(model_name)
    for i, prompt in enumerate(prompts):
        generated_code, generation_time, memory_used, cpu_used = generate_code(model, tokenizer, prompt['prompt'])
        if i % args.num_samples_per_prompt == 0:
            cur_prompt = prompt.copy()
            cur_prompt.update({"temperature": args.temperature, "top_p": args.top_p, "do_sample": args.do_sample, "max_new_tokens": args.max_new_tokens, "prompted": args.prompted})
            cur_prompt["raw_outputs"] = []
            cur_prompt["generation_times"] = []
            cur_prompt["memory_used"] = []
            cur_prompt["cpu_used"] = []
            cur_prompt["cpu_cores"] = args.num_cores_used
            cur_prompt["model_name"] = model_name
        cur_prompt["raw_outputs"].append(generated_code)
        #TODO: IMPLEMENT A CODE CLEANING FUNCTION
        cur_prompt["generation_times"].append(generation_time)
        cur_prompt["memory_used"].append(memory_used)
        cur_prompt["cpu_used"].append(cpu_used)
        if i % args.num_samples_per_prompt == args.num_samples_per_prompt - 1:
            results.append(cur_prompt)
        if not args.restart and args.cache is not None:
            with open(args.cache, 'a') as jsonl_file:
                jsonl_file.write(json.dumps(cur_prompt) + "\n")

with open(args.output, 'w') as output_file:
    json.dump(results, output_file, indent=4)
    
    