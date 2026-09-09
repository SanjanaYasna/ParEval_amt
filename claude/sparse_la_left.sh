#!/bin/bash
#SBATCH -J sparse
#SBATCH --output=sparse.txt
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p cpu
#SBATCH -t 3:30:00

#TODO RUN 1.10.0 #14 FOR SONNET

cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

module load conda/latest
conda activate hpc_llm

source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1

python run-all.py \
    "/work/pi_mrobson_smith_edu/ParEval_amt/generate/claude/1.10.0_IN/scan_continued.json" \
    -o "/work/pi_mrobson_smith_edu/ParEval_amt/generate/claude/temp_1.10/sparse_scan_FINISHED.json" \
    --yes-to-all \
    --include-models "hpx" \
    --build-timeout 30 \
    --run-timeout 60 \
    --log-build-errors \
    --log-runs \
    --launch-configs "/work/pi_mrobson_smith_edu/ParEval_amt/generate/claude/launch_quick.json"