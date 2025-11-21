#!/bin/bash -l
#SBATCH --job-name=hpc_mag
#SBATCH --output=hpx_hpc_mag.txt
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G #comfortable min for phind
#SBATCH --constraint=a100
#SBATCH --time=12:00:00
#SBATCH -a 0-1

export HF_HOME="/work/pi_mrobson_smith_edu/scratch/hf"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

module load conda/latest
conda activate hpc_llm
cd /work/pi_mrobson_smith_edu/ParEval_amt/generate
export OPENAI_API_KEY=TODO

#TODO: DEBUG phind-v2 
declare -a models=("magicoder" "hpc-coder")
curr_model=${models[$SLURM_ARRAY_TASK_ID]}
echo "Running model: $curr_model"
#output cache is in format modelname_cache.json
output_cache="/work/pi_mrobson_smith_edu/scratch/generation_hpx/la_test/cache/2048_tokens/${curr_model}_cache.json"
python generate_with_metrics.py --prompts "/work/pi_mrobson_smith_edu/ParEval_amt/prompts/la_prompts_hpx.json" \
    --model_names $curr_model \
    --output "/work/pi_mrobson_smith_edu/scratch/generation_hpx/la_test/la_out_hpx_2048_tokens.json" \
    --num_samples_per_prompt 100 \
    --do_sample \
    --hf_token TODO \
    --cache $output_cache \
    --max_new_tokens 2048

<<com
General testing
python generate_with_metrics.py --prompts "/work/pi_mrobson_smith_edu/ParEval_amt/prompts/la_prompts_hpx.json" \
    --model_names "gpt5" \
    --output "/work/pi_mrobson_smith_edu/scratch/generation_hpx/test/la_out_magi.json" \
    --num_samples_per_prompt 2 \
    --do_sample \
    --hf_token TODO \
    --cache "/work/pi_mrobson_smith_edu/scratch/generation_hpx/test/magi_cache.json" \
    --max_new_tokens 50
com 
