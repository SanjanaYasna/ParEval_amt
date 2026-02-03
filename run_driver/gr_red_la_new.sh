#!/bin/bash
#SBATCH -J adj_og
#SBATCH --output=adj_og.txt

#SBATCH -N 1      # num. nodes
#SBATCH -c 64 # Number of Cores per Task
#SBATCH -p cpu
#SBATCH --constraint=amd7763
#SBATCH --mem=500G # Requested Memory
#SBATCH --distribution=block
#SBATCH --sockets-per-node=1

#SBATCH -t 8:00:00 # Job time limit
#SBATCH -o slurm-%j.out # 
#SBATCH -a 0-11



source /work/pi_mrobson_smith_edu/.hpx_1_10_0


cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm 

declare -a model_files=("magicoder_cache_cleaned.json" "hpc-coder_cache_cleaned.json" "starcoder_cache_cleaned.json" "gpt5_cache_cleaned_low_reasoning.json")
declare -a model=("magicoder_run_exclusive.json" "hpc-coder_run_exclusive.json" "starcoder_run_exclusive.json" "gpt5_run_exclusive.json")
declare -a problem_types=("graph" "reduce" "la")


num_models=${#model_files[@]}
num_problem_types=${#problem_types[@]}
total_jobs=$((num_models * num_problem_types))

idx=${SLURM_ARRAY_TASK_ID}

problem_idx=$(( idx / num_models ))
model_idx=$(( idx % num_models ))

problem_type=${problem_types[$problem_idx]}
model_in=${model_files[$model_idx]}
model_out=${model[$model_idx]}


input_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${problem_type}/cache/${model_in}"
output_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${problem_type}/driver/tcmalloc/1.10.0/${model_out}"

echo "Model in:      ${input_file}"
echo "Problem Type:    ${problem_type}"
echo "Model out: ${output_file}"

python run-all.py \
    "${input_file}" \
    -o "${output_file}" \
    --yes-to-all \
    --include-models "hpx" \
    --build-timeout 30 \
    --run-timeout 45 \
    --log-build-errors \
    --log-runs