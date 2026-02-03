#!/bin/bash
#SBATCH -J fft
#SBATCH --output=fft.txt
#SBATCH -N 1      # num. nodes
#SBATCH -c 128 # Number of Cores per Task
#SBATCH -p cpu
#SBATCH --constraint=amd7763
#SBATCH --mem=1000G # Requested Memory
#SBATCH -t 24:00:00 # Job time limit 

#MUST RUN ALL 4 MODELS


#HPX 1.5.1 WITH TCMALLOC
source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1

# 1.10.0 WITH TCMALLOC: /work/pi_mrobson_smith_edu/.hpx_1_10_0

cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm 

prompt_type="fft"

models_in=( "hpc-coder.json" "magicoder.json" "gpt-5.json" "starcoder2-15b.json" )
models_out=( "hpc-coder_run_exclusive.json" "magicoder_run_exclusive.json" "gpt5_run_exclusive.json" "starcoder_run_exclusive.json" )

for i in "${!models_in[@]}"; do
    model_in="${models_in[$i]}"
    model_out="${models_out[$i]}"

    input_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${prompt_type}/cache/cleaned/${model_in}"
    output_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${prompt_type}/driver/tcmalloc/1.5.1/${model_out}"

    echo "Running model: ${model_in} -> ${model_out}"
    echo "  input:  ${input_file}"
    echo "  output: ${output_file}"

    python run-all.py \
        "${input_file}" \
        -o "${output_file}" \
        --yes-to-all \
        --include-models "hpx" \
        --build-timeout 30 \
        --run-timeout 45 \
        --log-build-errors \
        --log-runs 
    echo "Completed ${model_in}"
done

echo "All models completed."


# python run-all.py \
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/${problem_type}/cache/hpc-coder.json" \
#     -o "/work/pi_mrobson_smith_edu/scratch/generation_hpx/${problem_type}/driver/tcmalloc/1.5.1/hpc-coder_run.json" \
#     --yes-to-all \
#     --include-models "hpx" \
#     --build-timeout 60 \
#     --run-timeout 240 \
#     --log-build-errors \
#     --log-runs


# python run-all.py \
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/${problem_type}/cache/magicoder.json" \
#     -o "/work/pi_mrobson_smith_edu/scratch/generation_hpx/${problem_type}/driver/tcmalloc/1.5.1/magicoder_run.json" \
#     --yes-to-all \
#     --include-models "hpx" \
#     --build-timeout 60 \
#     --run-timeout 240 \
#     --log-build-errors \
#     --log-runs

# python run-all.py \
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/${problem_type}/cache/gpt-5.json" \
#     -o "/work/pi_mrobson_smith_edu/scratch/generation_hpx/${problem_type}/driver/tcmalloc/1.5.1/gpt5_run.json" \
#     --yes-to-all \
#     --include-models "hpx" \
#     --build-timeout 60 \
#     --run-timeout 240 \
#     --log-build-errors \
#     --log-runs

# python run-all.py \
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/${problem_type}/cache/starcoder2-15b.json" \
#     -o "/work/pi_mrobson_smith_edu/scratch/generation_hpx/${problem_type}/driver/tcmalloc/1.5.1/starcoder_run.json" \
#     --yes-to-all \
#     --include-models "hpx" \
#     --build-timeout 60 \
#     --run-timeout 240 \
#     --log-build-errors \
#     --log-runs

