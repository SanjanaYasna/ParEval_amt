# ParEval

[![HPDC 2024](https://img.shields.io/badge/Paper-HPDC'24-e87053.svg?style=flat)](https://pssg.cs.umd.edu/assets/papers/2024-06-pareval-hpdc.pdf)&nbsp;[![arXiv](https://img.shields.io/badge/arXiv-2401.12554-b31b1b.svg)](https://arxiv.org/abs/2401.12554)&nbsp;[![GitHub license](https://badgen.net/github/license/parallelcodefoundry/ParEval)](https://github.com/parallelcodefoundry/ParEval/blob/develop/LICENSE)


This repo contains the Parallel Code Evaluation (ParEval) Benchmark for
evaluating the ability of Large Language Models to write parallel code. See the
[ParEval Leaderboard](https://pssg.cs.umd.edu/blog/2024/pareval/) for
up-to-date results on different LLMs. We have extended this to include testing for HPX and Legion generation, and in addition translation from HPX -> Legion 


## Overview

The organization of the repo is as follows.

- `prompts/` -- the prompts in ParEval alongside some utility scripts
- `generate/` -- scripts for generating LLM outputs
- `drivers/` -- scripts to evaluate LLM outputs
- `analysis/` -- scripts to analyze driver results and compute metrics
    - `@k/` -- summary csvs of HPX generation performance 
    - `visuals/` -- per-prompt category net runtime graphs 
    - `visuals_specific/` -- individual prompt runtime graphs
- `tpl/` -- git submodule dependencies
- `prompts/` -- all prompt file jsons that are to be used as inputs for the generation phase
- `run_driver` -- miscellaneous driver running scripts 
- `srun_generate/` -- miscellaneous code generation scripts

Each subdirectory has further documentation on its contents. The general
workflow is to use `generate/generate.py` to generate LLM outputs, run
`drivers/run-all.py` to evaluate outputs, and `analysis/metrics.py` to
post-process the results summaries.

## Setup and Installation

A couple core systems software are assumed to be installed: Python >=3.7, a C++
compiler that supports C++17 and OpenMP, Make, CMake, and an MPI implementation.
If you are testing the CUDA and HIP prompts, then you will need access to NVIDIA
and AMD GPUs alongside their respective software stacks.

First, clone the repo.

```sh
git clone https://github.com/SanjanaYasna/ParEval_amt.git 
```

Next, you need to build Kokkos (if you want to include it in testing).
Kokkos source code was installed under directory ParEval_amt/tpl/kokkos/kokkos, set to version 4.5.01
```sh
cd amt/tpl/kokkos/kokkos
git checkout {stable version of choice}
module load gcc/9.4.0
module load mpich/4.2.1
#configure again at your own preferences, with build set to builddir below 
#my config...
cmake -B builddir \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_THREADS=ON \
    -DKokkos_ARCH_NATIVE=ON \
    -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF
cmake --build builddir
#set install prefix for kokkos to build under tpl/kokkos, as that's where pareval make files check
cmake --install builddir /work/pi_mrobson_smith_edu/ParEval_amt/tpl/kokkos/build
```

You will need to be able to use HPX for this project. There are two versions of HPX being tested: 1.5.1, and 1.10.0
There are setup scripts on the Unity cluster to get the respective version of HPX running:
```sh 
#HPX 1.5.1 
source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1
#OR 
#HPX 1.10.0 
source /work/pi_mrobson_smith_edu/.hpx_1_10_0
```

Finally, you need to install the Python dependencies. `requirements_AMT.txt` has
the set of dependencies. Use UV for the easiest time installing these.

```sh
#get uv in whatever environment you have
pip install uv
#ensure you have uv
which uv 
#if you're on unity cluster, there is a uv environment you can activate
source /work/pi_mrobson_smith_edu/pareval/.venv/bin/activate

#otherwise, take from the .txt environment file and make a uv environment from these packages 
uv add -r requirements_AMT.txt
```

## Citing ParEval original repo contents

```
@misc{nichols2024large,
      title={Can Large Language Models Write Parallel Code?}, 
      author={Daniel Nichols and Joshua H. Davis and Zhaojun Xie and 
              Arjun Rajaram and Abhinav Bhatele},
      year={2024},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      booktitle = {Proceedings of the 33rd International Symposium on High-Performance Parallel and Distributed Computing},
      series = {HPDC '24}
}
```

## License

ParEval is distributed under the terms of the [MIT license](/LICENSE).
