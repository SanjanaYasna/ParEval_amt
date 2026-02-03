#!/bin/bash
#SBATCH -J gpt5_high_reasoning
#SBATCH -N 1
#SBATCH -c 20
#SBATCH -p cpu
#SBATCH --mem=20G
#SBATCH -t 18:00:00
#SBATCH -o slurm-%A_%a.out        # separate log per array task
#SBATCH -a 0-23                   # 24 jobs (2 models × 3 problem types × 4 token limits)

cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

# HPX 1.5.1 environment
export BOOST_ROOT=/work/pi_mrobson_smith_edu/hpx_practice/boost_1_64_0/install
export LD_LIBRARY_PATH=/work/pi_mrobson_smith_edu/hpx_practice/boost_1_64_0/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/pi_mrobson_smith_edu/hpx_practice/hpx/1_5_1_install/lib:$LD_LIBRARY_PATH
export HPX_LOCATION=/work/pi_mrobson_smith_edu/hpx_practice/hpx/1_5_1_install/
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$HPX_LOCATION/lib/pkgconfig

module load conda/latest
conda activate hpc_llm

# --- parameter spaces -------------------------------------------------------
models=("gpt_5" "gpt_5_1_codex")
problem_types=("graph" "reduce" "la")
token_limits=(2048 4096 8192 16384)

num_models=${#models[@]}             
num_problems=${#problem_types[@]}    
num_tokens=${#token_limits[@]}       
tasks_per_model=$((num_problems * num_tokens))   

array_id=${SLURM_ARRAY_TASK_ID}

# guard against invalid index
total_tasks=$((num_models * num_problems * num_tokens))
if (( array_id < 0 || array_id >= total_tasks )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${array_id}; must be in [0, $((total_tasks-1))]"
    exit 1
fi

model_idx=$(( array_id / tasks_per_model ))
remain=$(( array_id % tasks_per_model ))
problem_idx=$(( remain / num_tokens ))
token_idx=$(( remain % num_tokens ))

model="${models[$model_idx]}"
problem="${problem_types[$problem_idx]}"
token="${token_limits[$token_idx]}"

# Directory suffix for caches (models have trailing underscore in directory names)
model_dir="${model}_"

# --- paths -------------------------------------------------------------------
cache_root="/work/pi_mrobson_smith_edu/scratch/generation_hpx/high_reasoning/cache"
driver_root="/work/pi_mrobson_smith_edu/scratch/generation_hpx/high_reasoning/driver"

input_file="${cache_root}/${problem}/${model_dir}/cleaned/${model}__cache_${token}.json"
output_file="${driver_root}/${problem}/${model}_${token}_run.json"

echo "SLURM task: ${array_id}"
echo "Model:      ${model}"
echo "Problem:    ${problem}"
echo "Token limit:${token}"
echo "Cache:      ${input_file}"
echo "Output:     ${output_file}"

if [[ ! -f "${input_file}" ]]; then
    echo "ERROR: Cache file not found: ${input_file}" >&2
    exit 2
fi

python run-all.py "${input_file}" \
    -o "${output_file}" \
    --yes-to-all \
    --include-models "hpx" \
    --build-timeout 60 \
    --run-timeout 240 \
    --log-build-errors \
    --log-runs