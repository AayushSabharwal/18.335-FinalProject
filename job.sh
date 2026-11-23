#!/bin/bash

#SBATCH -p mit_preemptable
#SBATCH -c 32
#SBATCH --mem=32G

julia --project=@. bench/run_benchmark.jl --matrices matrices/fem --eltype ComplexF64 --start-strategy largest --search-shift 0.1,0.1+0.1i,0.1+0.2i,0.1+0.4i,0.1+0.8i,0.1+1.6i --maxiter 100 --taus 0.01,0.1,1 --skip-sanity -o results/fem-c64-largest-0.02.json
