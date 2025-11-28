#!/bin/bash
#SBATCH -J gpt5
#SBATCH --output=gpt5.txt
#SBATCH -N 1      # num. nodes
#SBATCH -c 8 # Number of Cores per Task
#SBATCH --mem=16G # Requested Memory
#SBATCH -t 24:00:00 # Job time limit
#SBATCH -o slurm-%j.out # %j = gpt2

module load conda/latest
conda activate hpc_llm

export HF_HOME="/work/pi_mrobson_smith_edu/scratch/hf"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"


cd /work/pi_mrobson_smith_edu/ParEval_amt/generate
source ../../.hpc_src

python generate_with_metrics.py --prompts "/work/pi_mrobson_smith_edu/ParEval_amt/prompts/la_prompts_hpx.json" \
    --model_names "gpt5" \
    --output "/work/pi_mrobson_smith_edu/scratch/generation_hpx/la_test/la_out_hpx_2048_tokens.json" \
    --num_samples_per_prompt 100 \
    --do_sample \
    --hf_token $HF_TOKEN \
    --cache "/work/pi_mrobson_smith_edu/scratch/generation_hpx/la_test/cache/2048_tokens/gpt5_cache.json" \
    --max_new_tokens 2048


    