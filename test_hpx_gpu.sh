#!/bin/bash -l
#SBATCH --job-name=red_graph
#SBATCH --output=graph_reduce.txt
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G #comfortable min for phind
#SBATCH --constraint=a100
#SBATCH --time=24:00:00
#SBATCH -a 0-1

export HF_HOME="/work/pi_mrobson_smith_edu/scratch/hf"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

module load conda/latest
conda activate hpc_llm
cd /work/pi_mrobson_smith_edu/ParEval_amt/generate
source ../../.hpc_src

#TODO: DEBUG phind-v2 
declare -a models=("gpt5")
curr_model=${models[$SLURM_ARRAY_TASK_ID]}
echo "Running model: $curr_model"

<< com
DENSE LA AND SPARSE LA
output_cache="/work/pi_mrobson_smith_edu/scratch/generation_hpx/la_test/cache/2048_tokens/${curr_model}_cache.json"
python generate_with_metrics.py --prompts "/work/pi_mrobson_smith_edu/ParEval_amt/prompts/la_prompts_hpx.json" \
    --model_names $curr_model \
    --output "/work/pi_mrobson_smith_edu/scratch/generation_hpx/la_test/la_out_hpx_2048_tokens.json" \
    --num_samples_per_prompt 100 \
    --do_sample \
    --hf_token $HF_TOKEN \
    --cache $output_cache \
    --max_new_tokens 2048
com

#if SLURM_TASK_ID == 0, hten graph, else reduce
# GRAPH
if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    python generate_with_metrics.py --prompts "/work/pi_mrobson_smith_edu/ParEval_amt/prompts/graph_prompts_hpx.json" \
        --model_names $curr_model \
        --output "/work/pi_mrobson_smith_edu/scratch/generation_hpx/graph/graph_out_hpx_2048_tokens.json" \
        --num_samples_per_prompt 100 \
        --do_sample \
        --hf_token $HF_TOKEN \
        --cache "/work/pi_mrobson_smith_edu/scratch/generation_hpx/graph/cache/${curr_model}_cache.json" \
        --max_new_tokens 2048
else
# REDUCE
    python generate_with_metrics.py --prompts "/work/pi_mrobson_smith_edu/ParEval_amt/prompts/reduce_prompts_hpx.json" \
        --model_names $curr_model \
        --output "/work/pi_mrobson_smith_edu/scratch/generation_hpx/reduce/reduce_out_hpx_2048_tokens.json" \
        --num_samples_per_prompt 100 \
        --do_sample \
        --hf_token $HF_TOKEN \
        --cache "/work/pi_mrobson_smith_edu/scratch/generation_hpx/reduce/cache/${curr_model}_cache.json" \
        --max_new_tokens 2048
fi

<<com
General testing
python generate_with_metrics.py --prompts "/work/pi_mrobson_smith_edu/ParEval_amt/prompts/la_prompts_hpx.json" \
    --model_names "gpt5" \
    --output "/work/pi_mrobson_smith_edu/scratch/generation_hpx/test/la_out_magi.json" \
    --num_samples_per_prompt 2 \
    --do_sample \
    --hf_token $HF_TOKEN \
    --cache "/work/pi_mrobson_smith_edu/scratch/generation_hpx/test/magi_cache.json" \
    --max_new_tokens 50
com 

