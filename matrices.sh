#!/bin/bash

#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -o gen_fem_%j.log

module load apptainer/1.5.2
julia --project=@. bench/gen_fem_matrices.jl --apptainer --jobs=4
