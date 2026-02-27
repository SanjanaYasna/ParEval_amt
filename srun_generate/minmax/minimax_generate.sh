#!/bin/bash -l
#SBATCH --job-name=minimax
#SBATCH --output=minimax_%A_%a.txt
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=10GB
#SBATCH --time=10:00:00
#SBATCH -a 0-12

export HF_HOME="/work/pi_mrobson_smith_edu/scratch/hf"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# module load conda/latest
# conda activate hpc_llm
source /work/pi_mrobson_smith_edu/pareval/.venv/bin/activate
cd /work/pi_mrobson_smith_edu/ParEval_amt/generate
source ../../.hpc_src


declare -a tasks=("la" "fft" "futures_promises" "geometry" "graph" "histogram" "locking_contention" "reduce" "scan" "search" "sort" "stencil" "transform")
TASK="${tasks[$SLURM_ARRAY_TASK_ID]}"
PROMPT_DIR="/work/pi_mrobson_smith_edu/ParEval_amt/prompts"

OUT_DIR="/work/pi_mrobson_smith_edu/scratch/generation_hpx/"

MODEL_NAME="minimax"

# Construct paths
PROMPT_FILE="${PROMPT_DIR}/${TASK}.json"
TASK_OUT_DIR="${OUT_DIR}/${TASK}/cache"
CACHE_FILE="${TASK_OUT_DIR}/minimax.json"
OUT_FILE="${OUT_DIR}/${TASK}/cumulative_out.json"

python generate_with_metrics.py \
    --prompts "${PROMPT_FILE}" \
    --model_names "${MODEL_NAME}" \
    --output "${OUT_FILE}" \
    --cache "${CACHE_FILE}" \
    --num_samples_per_prompt 100 \
    --max_new_tokens 2048 \
    --do_sample \
    --hf_token "${HF_TOKEN}"

echo "Task ${TASK} completed."