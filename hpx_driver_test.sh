#!/bin/bash
#SBATCH -J gpt5_la
#SBATCH --output=la_out.txt
#SBATCH -N 1      # num. nodes
#SBATCH -c 20 # Number of Cores per Task
#SBATCH -p cpu
#SBATCH --mem=20G # Requested Memory
#SBATCH -t 36:00:00 # Job time limit
#SBATCH -o slurm-%j.out # 
#SBATCH -a 0-4

cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

#HPX 1.5.1
export BOOST_ROOT=/work/pi_mrobson_smith_edu/hpx_practice/boost_1_64_0/install
export LD_LIBRARY_PATH=/work/pi_mrobson_smith_edu/hpx_practice/boost_1_64_0/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/pi_mrobson_smith_edu/hpx_practice/hpx/1_5_1_install/lib:$LD_LIBRARY_PATH
export HPX_LOCATION=/work/pi_mrobson_smith_edu/hpx_practice/hpx/1_5_1_install/
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$HPX_LOCATION/lib/pkgconfig

module load conda/latest
conda activate hpc_llm 

declare -a model_files=("magicoder_cache_cleaned.json" "hpc-coder_cache_cleaned.json" "starcoder_cache_cleaned.json" "gpt5_cache_cleaned_low_reasoning.json" "gpt5_cache_cleaned_medium_reasoning.json")
declare -a model=("magicoder_run.json" "hpc-coder_run.json" "starcoder_run.json" "gpt5_low_reasoning_run.json" "gpt5_medium_reasoning_run.json")


curr_file=${model_files[$SLURM_ARRAY_TASK_ID]}
echo "Running file: $curr_file"
model=${model[$SLURM_ARRAY_TASK_ID]}
python run-all.py "/work/pi_mrobson_smith_edu/scratch/generation_hpx/la_test/cache/2048_tokens/${curr_file}" \
    -o "/work/pi_mrobson_smith_edu/scratch/generation_hpx/la_test/driver/${model}" \
    --yes-to-all --include-models "hpx" --build-timeout 60 --run-timeout 240 \
    --log-build-errors --log-runs