#!/bin/bash
#SBATCH -J 05_magi
#SBATCH --output=05_magi.txt

#SBATCH -N 1      # num. nodes
#SBATCH -c 64 # Number of Cores per Task
#SBATCH -p cpu
#SBATCH --constraint=amd7763
#SBATCH --mem=500G # Requested Memory
#SBATCH --distribution=block
#SBATCH --sockets-per-node=1

#SBATCH -t 8:00:00 # Job time limit 

#rerun 05 magicoder after parsing specifically for ifft (as it forward declares fft as prompt reiteration ) 

source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1

cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm 

prompt_type="fft"

python run-all.py \
        "/work/pi_mrobson_smith_edu/scratch/generation_hpx/fft/cache/05_magicoder/cleaned/magicoder.json" \
        -o "/work/pi_mrobson_smith_edu/scratch/generation_hpx/fft/driver/tcmalloc/05_magi/1.5.1/magicoder_05.json" \
        --yes-to-all \
        --include-models "hpx" \
        --build-timeout 30 \
        --run-timeout 60 \
        --log-build-errors \
        --log-runs 

source /work/pi_mrobson_smith_edu/.hpx_1_10_0

python run-all.py \
        "/work/pi_mrobson_smith_edu/scratch/generation_hpx/fft/cache/05_magicoder/cleaned/magicoder.json" \
        -o "/work/pi_mrobson_smith_edu/scratch/generation_hpx/fft/driver/tcmalloc/05_magi/1.10.0/magicoder_05.json" \
        --yes-to-all \
        --include-models "hpx" \
        --build-timeout 30 \
        --run-timeout 60 \
        --log-build-errors \
        --log-runs 