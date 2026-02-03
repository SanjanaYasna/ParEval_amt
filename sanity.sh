#!/bin/bash
#SBATCH -J sanity
#SBATCH --output=sanity.txt
#SBATCH -N 1      # num. nodes
#SBATCH -c 64 # Number of Cores per Task
#SBATCH -p cpu
#SBATCH --mem=20G # Requested Memory
#SBATCH -t 05:00:00 # Job time limit 

#MUST RUN ALL 4 MODELS


#HPX 1.5.1 WITH TCMALLOC
source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1

# 1.10.0 WITH TCMALLOC: /work/pi_mrobson_smith_edu/.hpx_1_10_0

cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm 

prompt_type="scan"



input_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/scan/driver/tcmalloc/1.5.1/gpt5_run.json"
output_file="/work/pi_mrobson_smith_edu/ParEval_amt/sanity.json"

python run-all.py \
        "${input_file}" \
        -o "${output_file}" \
        --yes-to-all \
        --include-models "hpx" \
        --build-timeout 30 \
        --run-timeout 45 \
        --log-build-errors \
        --log-runs 
