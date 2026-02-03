#!/bin/bash
#SBATCH -J where_is_gen
#SBATCH -N 1
#SBATCH -c 20
#SBATCH -p cpu
#SBATCH --mem=10G
#SBATCH -t 00:30:00
#SBATCH -o slurm=try.out
cd /work/pi_mrobson_smith_edu/ParEval_amt/drivers

# HPX 1.5.1 environment
export BOOST_ROOT=/work/pi_mrobson_smith_edu/hpx_practice/boost_1_64_0/install
export LD_LIBRARY_PATH=/work/pi_mrobson_smith_edu/hpx_practice/boost_1_64_0/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/pi_mrobson_smith_edu/hpx_practice/hpx/1_5_1_install/lib:$LD_LIBRARY_PATH
export HPX_LOCATION=/work/pi_mrobson_smith_edu/hpx_practice/hpx/1_5_1_install/
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$HPX_LOCATION/lib/pkgconfig

module load conda/latest
conda activate hpc_llm

python run-all.py test.json -o bleh.json --yes-to-all  --include-models "hpx"     --build-timeout 60     --run-timeout 240     --log-build-errors     --log-runs --scratch-dir "/work/pi_mrobson_smith_edu/scratch/try"
