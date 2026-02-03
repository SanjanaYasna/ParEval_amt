#!/bin/bash
#SBATCH -J histogram
#SBATCH --output=histogram.txt

#SBATCH -N 1      # num. nodes
#SBATCH -c 64 # Number of Cores per Task
#SBATCH -p cpu
#SBATCH --constraint=amd7763
#SBATCH --mem=500G # Requested Memory
#SBATCH --distribution=block
#SBATCH --sockets-per-node=1

#SBATCH -t 24:00:00 # Job time limit 

#MUST RUN ALL 4 MODELS

# 1.10.0 WITH TCMALLOC: 
source /work/pi_mrobson_smith_edu/.hpx_1_10_0

cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm 

prompt_type="histogram"

models_in=( "hpc-coder.json" "magicoder.json" "gpt-5.json" "starcoder2-15b.json" )
models_out=( "hpc-coder_run_exclusive.json" "magicoder_run_exclusive.json" "gpt5_run_exclusive.json" "starcoder_run_exclusive.json" )

for i in "${!models_in[@]}"; do
    model_in="${models_in[$i]}"
    model_out="${models_out[$i]}"

    input_file="l${prompt_type}/cache/cleaned/${model_in}"
    output_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${prompt_type}/driver/tcmalloc/1.10.0/${model_out}"

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