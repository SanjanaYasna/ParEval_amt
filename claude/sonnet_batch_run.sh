#!/bin/bash
#SBATCH -J graph
#SBATCH --output=sonnet_graph_rerun.txt

#SBATCH -N 1      # num. nodes
#SBATCH -c 64 # Number of Cores per Task
#SBATCH -p cpu
#SBATCH --constraint=amd7763
#SBATCH --mem=500G # Requested Memory
#SBATCH --distribution=block
#SBATCH --sockets-per-node=1

#SBATCH -t 11:00:00 # Job time limit
#SBATCH -o slurm-%A_%a.out # %A = job ID, %a = array task ID
#SBATCH -a 0

cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm 

# # RERUN GRAPH 1.5.1 TO SEE IF 16 AND 18 REALLY SHOUDL HAVE SUCH HIGH PASS
source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1
hpx_version="1.5.1"

input_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/graph/cache/cleaned/sonnet.json"
output_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/graph/driver/tcmalloc/1.5.1/sonnet_REDO_run.json"

python run-all.py \
    "${input_file}" \
    -o "${output_file}" \
    --yes-to-all \
    --include-models "hpx" \
    --build-timeout 30 \
    --run-timeout 60 \
    --log-build-errors \
    --log-runs

# if [ ${SLURM_ARRAY_TASK_ID} -eq 0 ]; then
#     source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1
#     hpx_version="1.5.1"
# else
#     source /work/pi_mrobson_smith_edu/.hpx_1_10_0
#     hpx_version="1.10.0"
# fi

# #TODO: WHY SO FEW CORRECT FOR GEOMETRY? (REPLACE 1.5.1 RESULTS WITH ORIGINAL COMBINED FILES) (despite lenient clean, is it true first 4 don't run at all for sonnet?)
# # + DID FFT FINISH, OR DOES IT KEEP ON HITTING TIMEOUT?
# #TODO: FFT last 2, sonnet_last_tlwo.json
# declare -a problem_types=( "geometry" "geometry" "graph" "histogram" "reduce" "fft")

# idx=${SLURM_ARRAY_TASK_ID}
# problem_type=${problem_types[$idx]}

# if [ "${problem_type}" == "fft" ]; then
#     input_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/fft/cache/cleaned/sonnet_last_two.json"
#     output_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/fft/driver/tcmalloc/1.10.0/sonnet_continued.json"
# else
#     input_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${problem_type}/cache/cleaned/sonnet.json"
#     output_file="/work/pi_mrobson_smith_edu/scratch/generation_hpx/${problem_type}/driver/tcmalloc/${hpx_version}/sonnet_run.json"
# fi


# echo "Job ID:        ${SLURM_ARRAY_TASK_ID}"
# echo "HPX Version:   ${hpx_version}"
# echo "Problem Type:  ${problem_type}"
# echo "Model in:      ${input_file}"
# echo "Model out:     ${output_file}"

# python run-all.py \
#     "${input_file}" \
#     -o "${output_file}" \
#     --yes-to-all \
#     --include-models "hpx" \
#     --build-timeout 30 \
#     --run-timeout 60 \
#     --log-build-errors \
#     --log-runs