#!/bin/bash
#SBATCH -J claude
#SBATCH --output=claude_backup.txt
#SBATCH -N 1      # num. nodes
#SBATCH -c 6 # Number of Cores per Task
#SBATCH -p cpu
#SBATCH --mem=20G # Requested Memory
#SBATCH -t 24:00:00 # Job time limit
#SBATCH -o slurm-%j.out
#SBATCH -a 0-6

#PURPOSE: GET STRICTPASS@1 FOR 1.10.0 HPX
source /work/pi_mrobson_smith_edu/.hpx_1_10_0 

cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm

BASE_DIR="/work/pi_mrobson_smith_edu/ParEval_amt/generate/claude"

# Input files array
declare -a INPUT_FILES=(
    "${BASE_DIR}/sonnet_fut_prom_try_cleaned.json"
    "${BASE_DIR}/sonnet_locking_contention.json"
    "${BASE_DIR}/sonnet_scan.json"
    "${BASE_DIR}/sonnet_search.json"
    "${BASE_DIR}/sonnet_sort.json"
    "${BASE_DIR}/sonnet_stencil.json"
    "${BASE_DIR}/sonnet_transform.json"
)

# Output files array
declare -a OUTPUT_FILES=(
    "${BASE_DIR}/1.10.0/sonnet_futures_promises_run.json"
    "${BASE_DIR}/1.10.0/sonnet_locking_contention_run.json"
    "${BASE_DIR}/1.10.0/sonnet_scan_run.json"
    "${BASE_DIR}/1.10.0/sonnet_search_run.json"
    "${BASE_DIR}/1.10.0/sonnet_sort_run.json"
    "${BASE_DIR}/1.10.0/sonnet_stencil_run.json"
    "${BASE_DIR}/1.10.0/sonnet_transform_run.json"
)

idx=${SLURM_ARRAY_TASK_ID}
input_file="${INPUT_FILES[$idx]}"
output_file="${OUTPUT_FILES[$idx]}"

# Create output directory if needed (for job 0)
if [ ${SLURM_ARRAY_TASK_ID} -eq 0 ]; then
    mkdir -p "${BASE_DIR}/1.10.0"
fi

echo "Job ID:     ${SLURM_ARRAY_TASK_ID}"
echo "Input:      ${input_file}"
echo "Output:     ${output_file}"

python run-all.py \
    "${input_file}" \
    -o "${output_file}" \
    --yes-to-all \
    --include-models "hpx" \
    --build-timeout 30 \
    --run-timeout 60 \
    --log-build-errors \
    --log-runs \
    --launch-configs "${BASE_DIR}/launch_quick.json"