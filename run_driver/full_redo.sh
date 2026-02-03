#!/bin/bash
#SBATCH -J full_misc
#SBATCH --output=full_misc.txt
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -p cpu
#SBATCH --constraint=amd7763
#SBATCH --mem=500G
#SBATCH --distribution=block
#SBATCH --sockets-per-node=1
#SBATCH -t 3:30:00
#SBATCH -o slurm-%j.out
#SBATCH -a 0-9

#TODO RUN 1.10.0 #14 FOR SONNET

cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm

declare -a MODEL_FILES=(
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/cache/cleaned/63_gpt5.json"
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/cache/cleaned/63_sonnet.json"
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/cache/cleaned/63_starcoder.json"
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/cache/cleaned/63_magicoder.json"
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/cache/cleaned/63_hpc-coder.json"
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/cache/cleaned/63_gpt5.json"
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/cache/cleaned/63_sonnet.json"
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/cache/cleaned/63_starcoder.json"
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/cache/cleaned/63_magicoder.json"
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/cache/cleaned/63_hpc-coder.json"
)

# declare -a MODEL_FILES=(
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/geometry/cache/cleaned/14_sonnet.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/reduce/cache/cleaned/28_sonnet.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/graph/cache/cleaned/17_sonnet.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/fft/cache/cleaned/09_sonnet.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/geometry/cache/cleaned/14_gpt-5.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/geometry/cache/cleaned/14_hpc-coder.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/geometry/cache/cleaned/14_magicoder.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/geometry/cache/cleaned/14_starcoder2-15b.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/reduce/cache/cleaned/28_hpc-coder.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/reduce/cache/cleaned/28_magicoder.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/reduce/cache/cleaned/28_starcoder.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/reduce/cache/cleaned/28_gpt5.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/graph/cache/cleaned/17_gpt5.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/graph/cache/cleaned/17_hpc-coder.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/graph/cache/cleaned/17_magicoder.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/graph/cache/cleaned/17_starcoder.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/fft/cache/cleaned/09_magicoder.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/fft/cache/cleaned/09_gpt5.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/fft/cache/cleaned/09_starcoder.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/fft/cache/cleaned/09_hpc-coder.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/geometry/cache/cleaned/14_sonnet.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/reduce/cache/cleaned/28_sonnet.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/graph/cache/cleaned/17_sonnet.json"
#     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/fft/cache/cleaned/09_sonnet.json"
# )

model_idx=$SLURM_ARRAY_TASK_ID
model_in=${MODEL_FILES[$model_idx]}
base_name=$(basename "$model_in")
model_out_file="${base_name%.json}_run.json"


# For the first 4 model files (indices 0..3) use the 1.5.1 setup,
# otherwise use the 1.10.0 setup.
if [ "$model_idx" -ge 0 ] && [ "$model_idx" -le 4 ]; then
    source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1
    output_dir="/work/pi_mrobson_smith_edu/scratch/generation_hpx/to_merge/1.5.1"
else
    source /work/pi_mrobson_smith_edu/.hpx_1_10_0
    output_dir="/work/pi_mrobson_smith_edu/scratch/generation_hpx/to_merge/1.10.0"
fi

mkdir -p "${output_dir}"

input_file="${model_in}"
output_file="${output_dir}/${model_out_file}"

echo "Running job ${SLURM_ARRAY_TASK_ID}: ${model_in} -> ${output_file}"

python run-all.py \
    "${input_file}" \
    -o "${output_file}" \
    --yes-to-all \
    --include-models "hpx" \
    --build-timeout 30 \
    --run-timeout 60 \
    --log-build-errors \
    --log-runs