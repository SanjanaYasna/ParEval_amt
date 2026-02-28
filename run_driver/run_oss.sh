#!/bin/bash
#SBATCH -J oss_dr
#SBATCH --output=oss.txt

#SBATCH -N 1      # num. nodes
#SBATCH -c 64 # Number of Cores per Task
#SBATCH -p cpu
#SBATCH --constraint=amd7763
#SBATCH --mem=500G # Requested Memory
#SBATCH --distribution=block
#SBATCH --sockets-per-node=1

#SBATCH -t 8:00:00 # Job time limit
#SBATCH -o slurm-%j.out # 
#SBATCH -a 0-25


cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm 

model_name="oss"
declare -a tasks=("la" "fft" "futures_promises" "geometry" "graph" "histogram" "locking_contention" "reduce" "scan" "search" "sort" "stencil" "transform")
TASK_IDX=$(( SLURM_ARRAY_TASK_ID % 13 ))
TASK="${tasks[$TASK_IDX]}"

if [ "$SLURM_ARRAY_TASK_ID" -le 12 ]; then
    HPX_VERSION="1.5.1"
    source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1
else
    HPX_VERSION="1.10.0"
    source /work/pi_mrobson_smith_edu/.hpx_1_10_0
fi

input_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${TASK}/cache/cleaned/${model_name}.json"
output_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${TASK}/driver/tcmalloc/${HPX_VERSION}/${model_name}_run.json"

python run-all.py 
    "${input_file}" \
    -o "${output_file}" \
    --yes-to-all \
    --include-models "hpx" \
    --build-timeout 30 \
    --run-timeout 60 \
    --log-build-errors \
    --log-runs 

# #DOING FOR 1.5.1
# source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1
# input_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${TASK}/cache/cleaned/${model_name}.json" 
# output_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${TASK}/driver/tcmalloc/1.5.1/${model_name}_run.json" 

# python run-all.py \
#     "${input_file}" \
#     -o "${output_file}" \
#     --yes-to-all \
#     --include-models "hpx" \
#     --build-timeout 30 \
#     --run-timeout 60 \
#     --log-build-errors \
#     --log-runs 

# #NOW 1.10.0 
# exec bash 
# cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers
# module load conda/latest
# conda activate hpc_llm 
# source /work/pi_mrobson_smith_edu/.hpx_1_10_0  
# input_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${TASK}/cache/cleaned/${model_name}.json" 
# output_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${TASK}/driver/tcmalloc/1.10.0/${model_name}_run.json" 

# python run-all.py \
#     "${input_file}" \
#     -o "${output_file}" \
#     --yes-to-all \
#     --include-models "hpx" \
#     --build-timeout 30 \
#     --run-timeout 60 \
#     --log-build-errors \
#     --log-runs 
