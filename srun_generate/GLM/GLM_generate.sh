#!/bin/bash -l
#SBATCH --job-name=GLM
#SBATCH -o %x_%A_%a-%j.txt
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G #comfortable min for phind
#SBATCH --constraint=a100
#SBATCH --time=24:00:00
#SBATCH -a 0-8

export HF_HOME="/work/pi_mrobson_smith_edu/scratch/hf"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# module load conda/latest
# conda activate hpc_llm
source /work/pi_mrobson_smith_edu/pareval/.venv/bin/activate
cd ../../generate
source ~/work/.hpc_src


declare -a tasks=("la" "fft" "futures_promises" "geometry" "graph" "histogram" "locking_contention" "reduce" "scan" "search" "sort" "stencil" "transform")
TASK="${tasks[$SLURM_ARRAY_TASK_ID]}"
PROMPT_DIR="../prompts"

OUT_DIR="/work/pi_mrobson_smith_edu/scratch/generation_hpx/"

MODEL_NAME="glm-4.7-flash"

# Construct paths
PROMPT_FILE="${PROMPT_DIR}/${TASK}.json"
TASK_OUT_DIR="${OUT_DIR}/${TASK}/cache"
CACHE_FILE="${TASK_OUT_DIR}/glm.json"
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