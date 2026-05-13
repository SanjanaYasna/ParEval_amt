#!/bin/bash -l
#SBATCH --job-name=futures_promises
#SBATCH --output=futures_promises.txt
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G #comfortable min for phind
#SBATCH --constraint=a100
#SBATCH --time=11:00:00
#SBATCH -a 0-1

export HF_HOME="/work/pi_mrobson_smith_edu/scratch/hf"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

module load conda/latest
conda activate hpc_llm
cd ../generate
source ~/work/.hpc_src

declare -a models=("starcoder2-15b" "magicoder" )
curr_model=${models[$SLURM_ARRAY_TASK_ID]}
echo "Running model: $curr_model"

BASE_OUT="/work/pi_mrobson_smith_edu/scratch/generation_hpx/futures_promises"
CACHE_FILE="${BASE_OUT}/cache/${curr_model}.json"
OUT_FILE="${BASE_OUT}/cumulative_out.json"

#GEOMETRY 
python generate_with_metrics.py --prompts "../prompts/futures_promises.json" \
        --model_names $curr_model \
        --output "${OUT_FILE}" \
        --num_samples_per_prompt 100 \
        --do_sample \
        --hf_token $HF_TOKEN \
        --cache "${CACHE_FILE}" \
        --max_new_tokens 2048