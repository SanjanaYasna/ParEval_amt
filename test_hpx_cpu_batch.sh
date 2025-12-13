#!/bin/bash -l
#SBATCH --job-name=red_graph
#SBATCH --output=graph_reduce.txt
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH -c 16 # Number of Cores per Task
#SBATCH --mem=16G # Requested Memory
#SBATCH --time=36:00:00
#SBATCH -a 0-1

export HF_HOME="/work/pi_mrobson_smith_edu/scratch/hf"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

module load conda/latest
conda activate hpc_llm
cd /work/pi_mrobson_smith_edu/ParEval_amt/generate
source ../../.hpc_src


#if SLURM_TASK_ID == 0, hten graph, else reduce
# GRAPH
if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    python generate_with_metrics.py --prompts "/work/pi_mrobson_smith_edu/ParEval_amt/prompts/graph_prompts_hpx.json" \
        --model_names "gpt5" \
        --output "/work/pi_mrobson_smith_edu/scratch/generation_hpx/graph/graph_out_hpx_2048_tokens.json" \
        --num_samples_per_prompt 100 \
        --do_sample \
        --hf_token $HF_TOKEN \
        --cache "/work/pi_mrobson_smith_edu/scratch/generation_hpx/graph/cache/gpt5_cache_medium_reasoning.json" \
        --max_new_tokens 2048
else
# REDUCE
    python generate_with_metrics.py --prompts "/work/pi_mrobson_smith_edu/ParEval_amt/prompts/reduce_prompts_hpx.json" \
        --model_names "gpt5" \
        --output "/work/pi_mrobson_smith_edu/scratch/generation_hpx/reduce/reduce_out_hpx_2048_tokens.json" \
        --num_samples_per_prompt 100 \
        --do_sample \
        --hf_token $HF_TOKEN \
        --cache "/work/pi_mrobson_smith_edu/scratch/generation_hpx/reduce/cache/gpt5_cache_medium_reasoning.json" \
        --max_new_tokens 2048
fi