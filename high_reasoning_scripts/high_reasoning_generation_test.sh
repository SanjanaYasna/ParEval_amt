#!/bin/bash
#SBATCH -J gpt_high_array
#SBATCH --output=gpt_high
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=8G
#SBATCH -t 24:00:00
#SBATCH -a 0-17

module load conda/latest
conda activate hpc_llm

export HF_HOME="/work/pi_mrobson_smith_edu/scratch/hf"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

cd /work/pi_mrobson_smith_edu/ParEval_amt/generate
source ../../.hpc_src

PROMPT_FILES=(
  "/work/pi_mrobson_smith_edu/ParEval_amt/prompts/la_prompts_hpx.json"
  "/work/pi_mrobson_smith_edu/ParEval_amt/prompts/graph_prompts_hpx.json"
  "/work/pi_mrobson_smith_edu/ParEval_amt/prompts/reduce_prompts_hpx.json"
)
PROMPT_NAMES=(la graph reduce)
TOKENS_LIST=(4096 8192 16384)
COMBOS_PER_MODEL=$(( ${#PROMPT_FILES[@]} * ${#TOKENS_LIST[@]} ))

#first 12 jobs are GPT-5 
#last 12 are gpt-5.1-codex

TASK_ID=${SLURM_ARRAY_TASK_ID}

if [[ ${TASK_ID} -lt ${COMBOS_PER_MODEL} ]]; then
    MODEL="gpt-5"
    COMBO_ID=${TASK_ID}
else
    MODEL="gpt-5.1-codex"
    COMBO_ID=$(( TASK_ID - COMBOS_PER_MODEL ))
fi


PROMPT_IDX=$(( COMBO_ID / ${#TOKENS_LIST[@]} ))
TOKEN_IDX=$(( COMBO_ID % ${#TOKENS_LIST[@]} ))

PROMPT_FILE=${PROMPT_FILES[$PROMPT_IDX]}
PROMPT_NAME=${PROMPT_NAMES[$PROMPT_IDX]}
MAX_NEW_TOKENS=${TOKENS_LIST[$TOKEN_IDX]}

MODEL_SAFE=$(echo "${MODEL}" | tr -cs '[:alnum:]' '_')

BASE_OUT="/work/pi_mrobson_smith_edu/scratch/generation_hpx/high_reasoning"
OUTPUT_DIR="${BASE_OUT}/${PROMPT_NAME}/${MODEL_SAFE}"
CACHE_DIR="${BASE_OUT}/cache/${PROMPT_NAME}/${MODEL_SAFE}"
mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

OUT_FILE="${OUTPUT_DIR}/${PROMPT_NAME}_${MODEL_SAFE}_${MAX_NEW_TOKENS}.json"
CACHE_FILE="${CACHE_DIR}/${MODEL_SAFE}_cache_${MAX_NEW_TOKENS}.json"

if [[ "${MODEL}" == "gpt-5" ]]; then
    EXTRA_FLAGS+=' --gpt_verbosity_level low'
fi

echo "[$(date)] SLURM_ARRAY_TASK_ID=${TASK_ID}"
echo "Model           : ${MODEL}"
echo "Prompt file     : ${PROMPT_FILE}"
echo "Max new tokens  : ${MAX_NEW_TOKENS}"
echo "Output file     : ${OUT_FILE}"
echo "Cache file      : ${CACHE_FILE}"
echo


python generate_with_metrics.py \
    --prompts "${PROMPT_FILE}" \
    --model_names "${MODEL}" \
    --output "${OUT_FILE}" \
    --num_samples_per_prompt 10 \
    --do_sample \
    --hf_token "${HF_TOKEN}" \
    --gpt_reasoning_level "high" \
    --cache "${CACHE_FILE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    ${EXTRA_FLAGS}
