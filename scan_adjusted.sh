#!/bin/bash
#SBATCH -J scan_adjusted
#SBATCH --output=scan_adjusted.txt
#SBATCH -N 1      # num. nodes
#SBATCH -c 64 # Number of Cores per Task
#SBATCH -p cpu
#SBATCH --constraint=amd7763
#SBATCH --mem=500G # Requested Memory
#SBATCH -t 20:00:00 # Job time limit 

#MUST RUN ALL 4 MODELS


#HPX 1.5.1 WITH TCMALLOC
source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1

# 1.10.0 WITH TCMALLOC: /work/pi_mrobson_smith_edu/.hpx_1_10_0

cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm 

prompt_type="scan"

models_in=( "gpt-5.json" "starcoder2-15b.json" )
models_out=( "gpt5_run.json" "starcoder_run.json" )

for i in "${!models_in[@]}"; do
    model_in="${models_in[$i]}"
    model_out="${models_out[$i]}"

    input_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${prompt_type}/cache/cleaned/${model_in}"
    output_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${prompt_type}/driver/tcmalloc/1.5.1/${model_out}_secondary"

    echo "Running model: ${model_in} -> ${model_out}"
    echo "  input:  ${input_file}"
    echo "  output: ${output_file}"

    python run-all.py \
        "${input_file}" \
        -o "${output_file}" \
        --yes-to-all \
        --include-models "hpx" \
        --build-timeout 60 \
        --run-timeout 90 \
        --log-build-errors \
        --log-runs 
    echo "Completed ${model_in}"
done

echo "All models completed."