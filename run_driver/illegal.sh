#!/bin/bash
#SBATCH -J futures_promises
#SBATCH --output=futures_promises_pass_only.txt
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -p cpu-preempt
#SBATCH --mem=10G
#SBATCH -t 4:00:00
#SBATCH -o slurm-%A_%a.out
#SBATCH -a 0-9

cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm

prompt_type="futures_promises"

declare -a MODEL_FILES=( "sonnet.json" "gpt-5.json" "hpc-coder.json" "magicoder.json" "starcoder.json" )
declare -a MODEL_OUTPUTS=( "sonnet_run.json" "gpt-5_run.json" "hpc-coder_run.json" "magicoder_run.json" "starcoder_run.json" )

model_idx=$((SLURM_ARRAY_TASK_ID % ${#MODEL_FILES[@]}))
model_in=${MODEL_FILES[$model_idx]}
model_out_file=${MODEL_OUTPUTS[$model_idx]}

if (( SLURM_ARRAY_TASK_ID < 5 )); then
    source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1
    output_dir="/work/pi_mrobson_smith_edu/scratch/generation_hpx/futures_promises/cache/lenient_clean/1.5.1"
else
    source /work/pi_mrobson_smith_edu/.hpx_1_10_0
    output_dir="/work/pi_mrobson_smith_edu/scratch/generation_hpx/futures_promises/cache/lenient_clean/1.10.0"
fi

mkdir -p "${output_dir}"

input_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/futures_promises/cache/lenient_clean/${model_in}"
output_file="${output_dir}/${model_out_file}"

echo "Running job ${SLURM_ARRAY_TASK_ID}: ${model_in} -> ${output_file}"

python run-all.py \
    "${input_file}" \
    -o "${output_file}" \
    --yes-to-all \
    --include-models "hpx" \
    --build-timeout 30 \
    --run-timeout 35 \
    --log-build-errors \
    --log-runs \
    --launch-configs "/work/pi_mrobson_smith_edu/ParEval_amt/run_driver/launch-1.json"


#NOW RUN 1.10.0 
#source /work/pi_mrobson_smith_edu/.hpx_1_10_0 