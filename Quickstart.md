
```sh
git clone --recurse-submodules git@github.com:mpr-lab/ParEval_AMT.git`

# (optional) On Unity:
module load cuda/12.1
module load conda/latest

conda env create --file environment.yml
conda activate hpc_llm

# Method 1: Direct wheel install (bypasses the build) (via Claude Sonnet 4.5)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

---

to generate: `srun_generate/`

to clean: `work/scratch/generation_hpx/usage.txt`

to run: `run_driver/`
