# std imports
import argparse
import json
import time
from tqdm import tqdm
# tpl imports
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, pipeline, BitsAndBytesConfig
# local imports
from utils import BalancedBracketsCriteria, PromptDataset, clean_output, get_inference_config
from utils import GPUCPUMonitor
from google import genai
from collections import defaultdict
from google.genai.errors import ClientError
import os
from openai import OpenAI
from google.genai import types
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
parser.add_argument('--do_sample', action='store_true',  help='Enable sampling (default: True)', default =True)
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generation (default: 8)')
parser.add_argument('--prompted', action='store_true', help='Use prompted generation. See StarCoder paper (default: False)')
parser.add_argument('--hf_token', type=str, help='HuggingFace API token for loading models')
parser.add_argument('--gpt_reasoning_level', type=str, default='low', help='GPT* model reasoning level, low, medium, or high')
parser.add_argument('--quantize_starcoder', action="store_true")
parser.add_argument('--gpt_verbosity_level', type=str, default = "medium")
args = parser.parse_args()

client = OpenAI(timeout=800.0)
gemini_client = genai.Client()
""" Load prompts """
with open(args.prompts, 'r') as json_file:
    prompts = json.load(json_file)


cached_names = set()

if not args.restart and args.cache is not None and os.path.exists(args.cache):
    #get prompt "name" already in
    data = [json.loads(line) for line in open(args.cache, 'r')]
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                name = entry.get("name")
                if name:
                    cached_names.add(name)
#if pipeline argument is used for generation
#use_pipeline = False
def load_model(model_name):
    """
    Loads the model and tokenizer, if applicable  
    
    Input: Model name (str)
    """
    #device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    try:
        #best in pareval expected:
        if model_name == 'phind-v2': #large
            model = LlamaForCausalLM.from_pretrained("Phind/Phind-CodeLlama-34B-v2"
                                                    , torch_dtype = torch.bloat16
                                                    , device_map="auto") 
            model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
            tokenizer = AutoTokenizer.from_pretrained("Phind/Phind-CodeLlama-34B-v2")
        elif model_name == "starcoder2-15b": #large
            #If too big, quantize 
            #https://huggingface.co/bigcode/starcoder2-15b 
            if args.quantize_starcoder: #TO DEBUG
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained('bigcode/starcoder2-15b', device_map="auto", quantization_config=quantization_config)
            else:
                model = AutoModelForCausalLM.from_pretrained('bigcode/starcoder2-15b', device_map="auto", torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained('bigcode/starcoder2-15b')
        #slightly higher than phind-v2 in parallel pass@1
        elif model_name == 'hpc-coder':
            model = AutoModelForCausalLM.from_pretrained('hpcgroup/hpc-coder-v2-6.7b', device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained('hpcgroup/hpc-coder-v2-6.7b')
        #comparable to phind-v2 in paralell pass@1
        elif model_name == 'magicoder': #can run in <32MB 
            generator = pipeline(
                model ="ise-uiuc/Magicoder-S-DS-6.7B",
                task="text-generation",
                torch_dtype=torch.bfloat16,
                device = 0
            )
            return generator, True
        
        #PROPRIETARY AND API BASED: GPT5 AND GEMINI PRO 2.5
        elif model_name == "gemini-3-pro":
            model = "gemini-3-pro"
            tokenizer = -1
            model, tokenizer = model_name, -1
        elif model_name == "gpt-5": 
            model, tokenizer = model_name, -1
        elif model_name =="gpt-5.1-codex":
            model, tokenizer = model_name, -1
        elif model_name == "gpt-5-codex":
            model, tokenizer = model_name, -1
        elif model_name == "gemini-2.5_pro":
            gemini_client = genai.GenerativeModel("gemini-2.5-pro")
            model = {"name": "gemini-2.5-pro", "client": gemini_client}
            tokenizer = -1
        #----lower performance expected
        elif model_name == 'gpt-neo':
            model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')
            tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
        elif model_name == 'poly-coder':
            model = AutoModelForCausalLM.from_pretrained('NinedayWang/PolyCoder-2.7B')
            tokenizer = AutoTokenizer.from_pretrained('NinedayWang/PolyCoder-2.7B')
        
        elif model_name == 'meta-llama':
            model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B', token=args.hf_token)
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', token=args.hf_token)
        elif model_name == 'gpt2':
            model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
            tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        # #if gpu available, put model to gpu
        # if torch.cuda.is_available():
        #     print(f"Model {model_name} ported to cuda")
        #     model = model.to('cuda')
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None
    
#FREE TIER (doable with gemini 2.5 pro)
API_RATE_LIMITS = {
    "gemini-2.5-pro": {"requests": 2, "sleep": 60},  # wait 60s after every 2 calls
}
api_request_counts = defaultdict(int)


def enforce_rate_limit(model_name: str):
    """Sleep when the free-tier rate limit for an API model requires it."""
    if model_name in API_RATE_LIMITS:
        cfg = API_RATE_LIMITS[model_name]
        api_request_counts[model_name] += 1
        if api_request_counts[model_name] % cfg["requests"] == 0:
            time.sleep(cfg["sleep"])
     
def profile_generation(model, tokenizer, device, prompt):
    #determine profiler from device and start
    if device == 'cuda' and tokenizer != -1:
        gpu_monitor = GPUCPUMonitor(monitor_interval=2, gpu=True)
        gpu_monitor.start()
    if device == 'cpu' and tokenizer != -1:
        cpu_monitor = GPUCPUMonitor(monitor_interval=2, gpu=False)
        cpu_monitor.start()
        
    #if pipeline is needed
    if type(tokenizer) == type(True):
        generated_code= generate_code_with_generator(model, prompt['prompt'])
    elif tokenizer != -1: #regular generation
        generated_code = generate_code(model, tokenizer, prompt['prompt'])
        
    else: #api-based, indicated by -1 value of tokenizer 
        api_time_start = time.time()
        if model.startswith("gpt"): #no temperature nor sampling support 
            if model == "gpt-5":
                response = client.responses.create( 
                model =model  
                , input = f"{prompt['prompt']}"
                , max_output_tokens= args.max_new_tokens 
                ,reasoning={ "effort": args.gpt_reasoning_level }
                ,text={ "verbosity": args.gpt_verbosity_level }
                , service_tier="flex"
            ) 
            else: #codex doesn't support high/low verb., nor flex
                response = client.responses.create( 
                model =model   
                , input = f"{prompt['prompt']}"
                , max_output_tokens= args.max_new_tokens
                ,reasoning={ "effort": args.gpt_reasoning_level }
                ,text={ "verbosity": "medium" }
            ) 
            generated_code = response.output_text
        elif model.startswith("gemini"):
            while True:
                try:
                    response = gemini_client.models.generate_content(
                    model="gemini-3-pro-preview"
                        ,contents=f"{prompt['prompt']}"
                        ,config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_level="low")
                            , max_output_tokens=args.max_new_tokens
                            , temperature = args.temperature
                            , topP = args.top_p
                        ),
                    )
                    generated_code = response.text 
                    break
                except ClientError as e:
                    if e.code == 429:
                        time.sleep(60)
                        continue
                    raise
            # response = gemini_client.models.generate_content(
            #     model="gemini-3-pro-preview"
            #     ,contents=f"{prompt['prompt']}"
            #     ,config=types.GenerateContentConfig(
            #         thinking_config=types.ThinkingConfig(thinking_level="low")
            #         , max_output_tokens=args.max_new_tokens
            #         , temperature = args.temperature
            #         , topP = args.top_p
            #     ),
            # )
            # generated_code = response.text 

    #end profilers  and collect metrics
    if  device == 'cuda':
        if tokenizer == -1:
            api_time_end = time.time()
            time_total = api_time_end - api_time_start
            return generated_code, 'N/A', 'N/A', 'N/A', 'N/A', time_total, 'N/A'
        gpu_monitor.stop()
        max_gpu_memory_usage = gpu_monitor.get_max_gpu_memory_usage()
        max_gpu_utilization = gpu_monitor.get_max_gpu_utilization() 
        average_gpu_memory_usage = gpu_monitor.get_average_gpu_memory_usage()
        average_gpu_utilization = gpu_monitor.get_average_gpu_utilization()
        gen_time = gpu_monitor._time
        vram = gpu_monitor._memory 
        return generated_code, max_gpu_memory_usage, max_gpu_utilization, average_gpu_memory_usage, average_gpu_utilization, gen_time, vram
    else:
        if tokenizer == -1:
            api_time_end = time.time()
            time_total = api_time_end - api_time_start
            return generated_code, 'N/A', time_total, 'N/A' 
        cpu_monitor.stop()
        cpu_percent = cpu_monitor._cpu_percent
        gen_time = cpu_monitor._time
        vram = cpu_monitor._memory  
        return generated_code, cpu_percent, gen_time, vram

        
def generate_code_with_generator(generator, prompt):
    MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that generates high-performance computing code for the problem below.
        
        @@ Instruction
        {instruction}

        @@ Response
        """
    prompt = MAGICODER_PROMPT.format(instruction = prompt)
    result = generator(prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p = args.top_p,
                        do_sample = args.do_sample
                        )
    generated_code = result[0]['generated_text']

    return generated_code


def generate_code(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    if torch.cuda.is_available():
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
    outputs = model.generate(inputs['input_ids']
                            , attention_mask = inputs['attention_mask']
                            , max_new_tokens=args.max_new_tokens
                            , temperature=args.temperature
                            , top_p = args.top_p
                            , do_sample = args.do_sample) #increased from 200  to avoid incompletion due to restriction
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code

cur_prompt = None
for model_name in args.model_names:
    results = []
    #if gpu, put model to gpu
    #for pipeline, model is actually the encased generator, and tokenizer is True
    model, tokenizer = load_model(model_name)
    print("Loaded model", model_name)
    
    #get whether model device is cpu or gpu
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'  
    for idx, prompt in enumerate(prompts):
        prompt_name = prompt.get("name")
        if (not args.restart) and args.cache is not None and prompt_name in cached_names:
            print(f"Skipping prompt idx={idx} name={prompt_name} for model {model_name} — already in cache.")
            continue
        for i in range(args.num_samples_per_prompt):
            #get return metrics by device type
            if device == 'cuda':
                generated_code, max_gpu_memory_usage, max_gpu_utilization, average_gpu_memory_usage, average_gpu_utilization, gen_time, vram = profile_generation(model, tokenizer, device, prompt)
                if i % args.num_samples_per_prompt == 0:
                    cur_prompt = prompt.copy()
                    cur_prompt.update({"temperature": args.temperature, "top_p": args.top_p, "do_sample": args.do_sample, "max_new_tokens": args.max_new_tokens, "prompted": args.prompted})
                    cur_prompt["outputs"] = []
                    cur_prompt["generation_times"] = []
                    cur_prompt["virtual_memory_used"] = []
                    cur_prompt["max_gpu_memory_usage"] = []
                    cur_prompt["max_gpu_utilization"] = []
                    cur_prompt["average_gpu_memory_usage"] = []
                    cur_prompt["average_gpu_utilization"] = []
                    cur_prompt["model_name"] = model_name   
                cur_prompt["outputs"].append(generated_code)
                cur_prompt["generation_times"].append(gen_time) 
                cur_prompt["virtual_memory_used"].append(vram)
                cur_prompt["max_gpu_memory_usage"].append(max_gpu_memory_usage)
                cur_prompt["max_gpu_utilization"].append(max_gpu_utilization)
                cur_prompt["average_gpu_memory_usage"].append(average_gpu_memory_usage)
                cur_prompt["average_gpu_utilization"].append(average_gpu_utilization)
            elif device == 'cpu':
                generated_code, cpu_percent, gen_time, vram = profile_generation(model, tokenizer, device, prompt)
                if i % args.num_samples_per_prompt == 0:
                    cur_prompt = prompt.copy()
                    cur_prompt.update({"temperature": args.temperature, "top_p": args.top_p, "do_sample": args.do_sample, "max_new_tokens": args.max_new_tokens, "prompted": args.prompted})
                    cur_prompt["outputs"] = []
                    cur_prompt["generation_times"] = []
                    cur_prompt["virtual_memory_used"] = []
                    cur_prompt["cpu_percent"] = []
                    cur_prompt["cpu_cores"] = args.num_cores_used
                    cur_prompt["model_name"] = model_name
                cur_prompt["outputs"].append(generated_code)
                cur_prompt["generation_times"].append(gen_time) 
                cur_prompt["virtual_memory_used"].append(vram)
                cur_prompt["cpu_percent"].append(cpu_percent)
            if i % args.num_samples_per_prompt == args.num_samples_per_prompt - 1:
                results.append(cur_prompt)
        #write to cache once reaching num_samples results 
        if not args.restart and args.cache is not None:
            with open(args.cache, 'a+') as jsonl_file:
                jsonl_file.write(json.dumps(cur_prompt) + "\n")

with open(args.output, 'a+') as output_file:
    json.dump(results, output_file, indent=4)
