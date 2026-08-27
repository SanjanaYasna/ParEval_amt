#!/bin/bash
#SBATCH -J reduce
#SBATCH --output=reduce_%A_%a.txt
#SBATCH --array=0-9
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -p cpu
#SBATCH --constraint=amd7763
#SBATCH --mem=10G
#SBATCH -t 24:00:00

# AS DISCUSSED BY ROBSON, MODELS FOR IJPP PAPER:
# HPCCODER, MAGICODER, GPT5, SONNET, OSS, GEMMA (TODO)
#
# 5 models x 2 HPX versions = 10 tasks
# ONLY 2 THREADS TO CAPTURE BUILD ERRORS + RUN ERRORS, THAT'S IT


prompt_type="reduce"

launch_configs="/work/pi_mrobson_smith_edu/ParEval_amt/run_driver/get_errors_build_reruns/launch-configs_2_threads.json"

input_dir="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${prompt_type}/cache/cleaned"
# NEW OUTPUT DIR
# STRUCTURE:
# FFT/ 
#   /1.10.0
#   /1.5.1
base_output_dir="/work/pi_mrobson_smith_edu/scratch/generation_hpx/build_errors/${prompt_type}"


# ============================================================================
# MODELS
# ============================================================================

models_in=(
    "hpc-coder_cache_cleaned.json"
    "magicoder_cache_cleaned.json"
    "gpt5_cache_cleaned_low_reasoning.json"
    "sonnet.json"
    "oss.json"
)

models_out=(
    "hpc-coder_run.json"
    "magicoder_run.json"
    "gpt5_run.json"
    "sonnet_run.json"
    "oss_run.json"
)

NUM_MODELS=${#models_in[@]}

# set up hpx proper env
if [ "${SLURM_ARRAY_TASK_ID}" -lt "${NUM_MODELS}" ]; then

    # Tasks 0-4: HPX 1.5.1
    HPX_VERSION="1.5.1"
    HPX_ENV="/work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1"

    model_index="${SLURM_ARRAY_TASK_ID}"

else

    # Tasks 5-9: HPX 1.10.0
    HPX_VERSION="1.10.0"
    HPX_ENV="/work/pi_mrobson_smith_edu/.hpx_1_10_0"

    model_index=$((SLURM_ARRAY_TASK_ID - NUM_MODELS))

fi


model_in="${models_in[$model_index]}"
model_out="${models_out[$model_index]}"

input_file="${input_dir}/${model_in}"

output_dir="${base_output_dir}/${HPX_VERSION}"
output_file="${output_dir}/${model_out}"

mkdir -p "${output_dir}"

echo "============================================================"
echo "HPX BUILD/RUNTIME ERROR RERUN"
echo "============================================================"
echo "SLURM job ID:     ${SLURM_JOB_ID}"
echo "SLURM array task: ${SLURM_ARRAY_TASK_ID}"
echo "CPUs per task:    ${SLURM_CPUS_PER_TASK}"
echo
echo "Prompt type:      ${prompt_type}"
echo "Model:            ${model_in}"
echo "HPX version:      ${HPX_VERSION}"
echo "HPX environment:  ${HPX_ENV}"
echo
echo "Input:            ${input_file}"
echo "Output:           ${output_file}"
echo "============================================================"

source "${HPX_ENV}"
cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers || exit 1

module load conda/latest
conda activate hpc_llm

#ensure PROPER LAUCNH CONFIGS PASSED
python run-all.py \
    "${input_file}" \
    -o "${output_file}" \
    --yes-to-all \
    --include-models "hpx" \
    --build-timeout 30 \
    --run-timeout 60 \
    --launch-configs "${launch_configs}" \
    --log-build-errors \
    --log-runs

exit_code=$?

echo

if [ "${exit_code}" -ne 0 ]; then
    echo "============================================================"
    echo "FAILED"
    echo "============================================================"
    echo "Model:       ${model_in}"
    echo "HPX version: ${HPX_VERSION}"
    echo "Exit code:   ${exit_code}"
    echo "============================================================"
else
    echo "============================================================"
    echo "COMPLETED"
    echo "============================================================"
    echo "Model:       ${model_in}"
    echo "HPX version: ${HPX_VERSION}"
    echo "Output:      ${output_file}"
    echo "============================================================"
fi

exit "${exit_code}"