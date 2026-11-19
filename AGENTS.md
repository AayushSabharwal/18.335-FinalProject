# Agent notes

## Environment

This repo uses the project environment at the **repo root**
(`Project.toml` + committed `Manifest.toml`), NOT `TestEnv/`. Reproducibility
requires the pinned manifest: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`.

## Layout

- `src/RQIBench.jl` — RQI variant implementations (module `RQIBench`)
- `bench/run_benchmark.jl` — CLI driver (ArgParse + BenchmarkTools + JSON out)
- `bench/make_matrices.jl` — regenerates `matrices/*.txt`
- `results/` — benchmark JSON output

## Commands

```sh
julia --project=. bench/make_matrices.jl          # regenerate test matrices

# find-all-eigenvalues experiment (shift sweep over Gershgorin range):
julia --project=. bench/run_benchmark.jl \
    --matrices matrices/ --out results/bench.json \
    --start-strategy grid --nstarts 10 --samples 5

# largest-real-part experiment, in Float32:
julia --project=. bench/run_benchmark.jl \
    --matrices matrices/ --out results/bench_f32.json \
    --start-strategy largest --eltype Float32 --tol 1e-5 --inner-reltol 1e-6
```

Quick sanity check (all methods converge on a small symmetric matrix):

```sh
jld eval --scratch 'include("src/RQIBench.jl"); using .RQIBench; RQIBench.sanity_check()'
```

Note: `jld run` injects daemon args into `ARGS`; scripts that parse `ARGS`
must filter `--*` entries (see `make_matrices.jl`) or run under plain `julia`.
