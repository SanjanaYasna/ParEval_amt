#!/bin/bash
#SBATCH --dependency=afterok:51249052_2
#SBATCH -N 1      # num. nodes
#SBATCH -c 64 # Number of Cores per Task
#SBATCH -p cpu
#SBATCH --constraint=amd7763
#SBATCH --mem=500G # Requested Memory
#SBATCH --distribution=block
#SBATCH --sockets-per-node=1
#SBATCH -t 6:00:00 # Job time limit 

source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1 

cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm 

prompt_type="locking_contention"

python /work/pi_mrobson_smith_edu/scratch/generation_hpx/clean_output_final.py -i /work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/cache/starcoder2-15b.json -o /work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/cache/cleaned/starcoder2-15b.json

input_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${prompt_type}/cache/cleaned/starcoder2-15b.json"
output_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${prompt_type}/driver/tcmalloc/1.5.1/starcoder_run.json"
python run-all.py \
    "${input_file}" \
    -o "${output_file}" \
    --yes-to-all \
    --include-models "hpx" \
    --build-timeout 30 \
    --run-timeout 30 \
    --log-build-errors \
    --log-runs 