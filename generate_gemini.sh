#!/bin/bash
#SBATCH -J 2_gemini
#SBATCH --output=2_gemini.txt
#SBATCH -N 1      # num. nodes
#SBATCH -c 12 # Number of Cores per Task
#SBATCH -p cpu
#SBATCH --mem=5G # Requested Memory
#SBATCH -t 24:00:00 # Job time limit
#SBATCH -o gemini_%A_%a.out  
#SBATCH -a 0-10%2



export HF_HOME="/work/pi_mrobson_smith_edu/scratch/hf"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

module load conda/latest
conda activate hpc_llm
cd /work/pi_mrobson_smith_edu/ParEval_amt/generate
source ../../.hpc_src

declare -a prompt=("la" "graph" "reduce" "scan" "search" "sort" "transform" "stencil" "geometry" "fft" "histogram")
category=${prompt[$SLURM_ARRAY_TASK_ID]}
PROMPT_FILE="/work/pi_mrobson_smith_edu/ParEval_amt/prompts/${category}.json"
BASE_OUT="/work/pi_mrobson_smith_edu/scratch/generation_hpx"
CACHE_FILE="${BASE_OUT}/${category}/cache/gemini.json"
OUT_FILE="${BASE_OUT}/${category}/cumulative_out.json"

python generate_with_metrics.py --prompts "${PROMPT_FILE}"\
        --model_names "gemini-3-pro" \
        --output "${OUT_FILE}" \
        --num_samples_per_prompt 100 \
        --do_sample \
        --hf_token $HF_TOKEN \
        --cache "${CACHE_FILE}" \
        --max_new_tokens 2048
