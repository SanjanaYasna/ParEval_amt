#!/bin/bash
#SBATCH -J gpt_la
#SBATCH --output=reduce.txt
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G #comfortable min for phind
#SBATCH --constraint=a100
#SBATCH -t 72:00:00 # Job time limit

#Needed for phindv2 to run at all
#module load conda/latest
#conda activate cuda_12_6
#module load cuda/12.6

export HF_HOME="/work/pi_mrobson_smith_edu/scratch/hf"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

#works for other models
module load conda/latest
conda activate hpc_llm 

cd /work/pi_mrobson_smith_edu/ParEval_amt/generate
source ../../.hpc_src

python generate_with_metrics.py --prompts "/work/pi_mrobson_smith_edu/ParEval_amt/prompts/reduce_prompts_hpx.json" \
        --model_names "starcoder2-15b" \
        --output "/work/pi_mrobson_smith_edu/scratch/generation_hpx/reduce/reduce_out_hpx_2048_tokens.json" \
        --num_samples_per_prompt 100 \
        --do_sample \
        --hf_token $HF_TOKEN \
        --cache "/work/pi_mrobson_smith_edu/scratch/generation_hpx/reduce/cache/starcoder2-15b_cache.json" \
        --max_new_tokens 2048
    
