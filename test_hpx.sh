#!/bin/bash
#SBATCH -J gpt2_pareval
#SBATCH --output=gpt2_pareval.txt
#SBATCH -N 1      # num. nodes
#SBATCH -c 16 # Number of Cores per Task
#SBATCH --mem=16G # Requested Memory
#SBATCH -t 02:00:00 # Job time limit
#SBATCH -o slurm-%j.out # %j = gpt2

module load conda/latest
conda activate hpc_llm

cd /work/pi_mrobson_smith_edu/ParEval_amt/generate
python generate_with_metrics.py --prompts "/work/pi_mrobson_smith_edu/ParEval_amt/prompts/la_prompts_hpx.json" \
    --model_name "gpt2" \
    --output "/work/pi_mrobson_smith_edu/ParEval_amt/prompts/la_out_hpx.json" \
    --num_samples_per_prompt 2 \
    --hf_token TODO \
    --cache std_cache_gpt2.json
