#!/bin/bash
#SBATCH -J geometry
#SBATCH --output=geometry.txt
#SBATCH -N 1      # num. nodes
#SBATCH -c 64 # Number of Cores per Task
#SBATCH -p cpu
#SBATCH --constraint=amd7763
#SBATCH --mem=500G # Requested Memory
#SBATCH -t 9:00:00 # Job time limit 
#SBATCH -a 1-9
#MUST RUN ALL 4 MODELS


if [ "$SLURM_ARRAY_TASK_ID" -le 4 ]; then
    HPX_VERSION="1.5.1"
    source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1
else
    HPX_VERSION="1.10.0"
    source /work/pi_mrobson_smith_edu/.hpx_1_10_0
fi


cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm 

#SONNET LENIENT NEEDS TO BE REPARSED 
prompt_type="geometry"

models_in=("glm.json" "oss.json" "minimax.json" "sonnet_lenient_cleaned_parse.json" "gpt-5.json")
models_out=("glm_rerun.json" "oss_rerun.json" "minimax_rerun.json" "sonnet_rerun.json" "gpt5_rerun.json")
model_idx=$((SLURM_ARRAY_TASK_ID % 5))

model_in="${models_in[$model_idx]}"
model_out="${models_out[$model_idx]}"

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
