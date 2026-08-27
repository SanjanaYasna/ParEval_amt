#!/bin/bash
#SBATCH -J geometry
#SBATCH --output=geometry_%a.txt
#SBATCH -N 1      # num. nodes
#SBATCH -c 64 # Number of Cores per Task
#SBATCH -p cpu
#SBATCH --constraint=amd7763
#SBATCH --mem=500G # Requested Memory
#SBATCH -t 9:00:00 # Job time limit 
#SBATCH -a 1-2

# We only need 2 tasks now: Task 1 for HPX 1.5.1, Task 2 for HPX 1.10.0
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    HPX_VERSION="1.5.1"
    source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1
else
    HPX_VERSION="1.10.0"
    source /work/pi_mrobson_smith_edu/.hpx_1_10_0
fi

cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm 

prompt_type="geometry"

# Hardcode directly to Sonnet
model_in="sonnet_lenient_cleaned_parse.json"
model_out="sonnet_rerun_proper.json"

input_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${prompt_type}/cache/cleaned/${model_in}"
output_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${prompt_type}/driver/tcmalloc/${HPX_VERSION}/${model_out}"

python run-all.py \
    "${input_file}" \
    -o "${output_file}" \
    --yes-to-all \
    --include-models "hpx" \
    --build-timeout 30 \
    --run-timeout 60 \
    --log-build-errors \
    --log-runs 