#!/bin/bash

#SBATCH -p mit_preemptable
#SBATCH -c 64
#SBATCH --mem=32G

module load apptainer/1.5.2
cd 18.335-FinalProject
julia --project=@. bench/gen_fem_matrices.jl --apptainer
