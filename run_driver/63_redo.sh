#!/bin/bash
#SBATCH -J full_misc
#SBATCH --output=full_misc.txt
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p cpu
#SBATCH --mem=10G
#SBATCH -t 4:00:00
#SBATCH -a 0
cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm
source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1

declare -a MODEL_FILES=(
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/cache/cleaned/63_starcoder.json"
)

model_idx=$SLURM_ARRAY_TASK_ID
model_in=${MODEL_FILES[$model_idx]}
base_name=$(basename "$model_in")
model_out_file="${base_name%.json}_run.json"


output_dir="/work/pi_mrobson_smith_edu/scratch/generation_hpx/to_merge/pass1_combined"

mkdir -p "${output_dir}"

input_file="${model_in}"
output_file="${output_dir}/${model_out_file}"

echo "Running job ${SLURM_ARRAY_TASK_ID}: ${model_in} -> ${output_file}"

python run-all.py \
    "${input_file}" \
    -o "${output_file}" \
    --yes-to-all \
    --include-models "hpx" \
    --build-timeout 30 \
    --run-timeout 60 \
    --log-build-errors \
    --log-runs \
    --launch-configs "/work/pi_mrobson_smith_edu/ParEval_amt/generate/claude/launch_quick.json"