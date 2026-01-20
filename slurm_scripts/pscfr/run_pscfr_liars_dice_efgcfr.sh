#!/bin/bash
#SBATCH --partition=amdlong
#SBATCH --time=48:00:00
#SBATCH --mem=64G

# RCI cluster modules
ml Clang/12.0.1-GCCcore-10.3.0
ml CMake/3.20.1-GCCcore-10.3.0
ml Python/3.9.5-GCCcore-10.3.0
ml magma/2.6.1-foss-2021a-CUDA-11.3.1
ml SciPy-bundle/2021.05-foss-2021a
ml typing-extensions/3.10.0.0-GCCcore-10.3.0
ml protobuf-python/3.17.3-GCCcore-10.3.0
ml matplotlib/3.4.2-foss-2021a

./open_spiel/build/papers_with_code/public_state_cfr/my_main -efgcfr -game_name='liars_dice(dice_sides=4,numdice=2)' -exploitability --iterations=2

