#!/bin/bash

#SBATCH -p mit_preemptable
#SBATCH -c 32
#SBATCH --mem=32G

cd 18.335-FinalProject
julia --project=@. bench/run_benchmark.jl --matrices matrices/fem --eltype ComplexF64 --start-strategy largest --search-shift 0.1 --maxiter 100 --taus 0.01,0.1,1 --skip-sanity -o results/fem-c64-largest-0.02.json
