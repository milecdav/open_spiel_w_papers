#!/bin/bash
#SBATCH --partition=amdfast
#SBATCH --time=3:00:00
#SBATCH --mem=8G
#SBATCH --output=./slurm_output/dlr/slurm-%j.out

# RCI cluster modules
ml Clang/12.0.1-GCCcore-10.3.0
ml CMake/3.20.1-GCCcore-10.3.0
ml Python/3.9.5-GCCcore-10.3.0
ml magma/2.6.1-foss-2021a-CUDA-11.3.1
ml SciPy-bundle/2021.05-foss-2021a
ml typing-extensions/3.10.0.0-GCCcore-10.3.0
ml protobuf-python/3.17.3-GCCcore-10.3.0
ml matplotlib/3.4.2-foss-2021a

./open_spiel/build/papers_with_code/depth_limited_responses/dlr_main --secfr --iterations=1000

