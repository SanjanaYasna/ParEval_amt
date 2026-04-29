#!/bin/bash
#SBATCH -J sonnet
#SBATCH --output=sonnet_2.txt

#SBATCH -N 1      # num. nodes
#SBATCH -c 64 # Number of Cores per Task
#SBATCH -p cpu
#SBATCH --constraint=amd7763
#SBATCH --mem=500G # Requested Memory
#SBATCH --distribution=block
#SBATCH --sockets-per-node=1
#SBATCH -t 30:00:00 # Job time limit 

#MUST RUN ALL 4 MODELS

# 1.5.1 WITH TCMALLOC: 


python run-all.py "/work/pi_mrobson_smith_edu/ParEval_amt/generate/claude/sonnet_revised.json" -o "/work/pi_mrobson_smith_edu/ParEval_amt/generate/claude/sonnet_run.json" \
    --yes-to-all \
    --include-models "hpx" \
    --build-timeout 30 \
    --run-timeout 45 \
    --log-build-errors \
    --log-runs 

echo "Sonnet Complete"