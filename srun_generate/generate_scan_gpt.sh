#!/bin/bash
#SBATCH -J gpt_scan
#SBATCH --output=gpt_scan
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=12G
#SBATCH -t 48:00:00 

module load conda/latest
conda activate hpc_llm

export HF_HOME="/work/pi_mrobson_smith_edu/scratch/hf"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

cd /work/pi_mrobson_smith_edu/ParEval_amt/generate
source ../../.hpc_src

BASE_OUT="/work/pi_mrobson_smith_edu/scratch/generation_hpx/scan"
CACHE_FILE="${BASE_OUT}/cache/gpt-5.json"
OUT_FILE="${BASE_OUT}/cumulative_out.json"

python generate_with_metrics.py --prompts "/work/pi_mrobson_smith_edu/ParEval_amt/prompts/scan.json" \
    --model_names "gpt-5" \
    --output "${OUT_FILE}" \
    --num_samples_per_prompt 100 \
    --do_sample \
    --hf_token "${HF_TOKEN}" \
    --gpt_reasoning_level "low" \
    --cache "${CACHE_FILE}" \
    --max_new_tokens 2048 \
    --gpt_verbosity_level "low"