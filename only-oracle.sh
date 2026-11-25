#!/bin/bash

#SBATCH -p mit_preemptable
#SBATCH -c 32
#SBATCH --mem=32G

julia --project=@. bench/run_benchmark.jl --matrices matrices/fem --oracle-eltype Complex{Float64x2} --oracle-only --oracle arnoldi --skip-sanity -o results/fem-c64x2-oracle-preempt.json
