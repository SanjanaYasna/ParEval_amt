#!/bin/bash
#SBATCH -J adj_og
#SBATCH --output=adj_og.txt
#SBATCH -c 64 # Number of Cores per Task
#SBATCH -p cpu
#SBATCH --constraint=amd7763
#SBATCH --distribution=block
#SBATCH --mem=500G # Requested Memory
#SBATCH -t 12:00:00 # Job time limit
#SBATCH -o slurm-%j.out # 
#SBATCH -a 0-11


#HPX 1.5.1 WITH TCMALLOC
source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1

# 1.10.0 WITH TCMALLOC: /work/pi_mrobson_smith_edu/.hpx_1_10_0

cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm 

declare -a model_files=("magicoder_cache_cleaned.json" "hpc-coder_cache_cleaned.json" "starcoder_cache_cleaned.json" "gpt5_cache_cleaned_low_reasoning.json")
declare -a model=("magicoder_run_first.json" "hpc-coder_run_first.json" "starcoder_run_first.json" "gpt5_run_first.json")
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
output_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${problem_type}/driver/tcmalloc/1.5.1/${model_out}"

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