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

## FreeFEM → FEM matrices (linearized NS stability)

`bench/stability_{cavity,cylinder,channel}.edp` assemble the pencil
`J x = λ M x` for the eigenproblem in `TestEnv/doc.tex` (Taylor–Hood
P2/P1). Geometry parameters are injected by generated wrappers (no
`getARGV()` in FreeFEM 4.12); `bench/gen_fem_matrices.jl` drives the
whole pipeline: wrapper → `docker run freefem/freefem` (`ff-mpirun -np 1`)
→ `savemtx` dumps (`J.mtx`/`M.mtx` + `bnd.txt`) in
`TestEnv/fem/<key>_Re<Re>/` → boundary-DOF elimination → stacked 2n×n
`[J; M]` Matrix Market file in `matrices/fem/fem_<key>_Re<Re>.mtx`
(created automatically).

Base flow (cavity/cylinder): PETSc `SNESSolve` (trust-region Newton +
line search, `-pc_type lu`) with geometric Re continuation — halve the
step and retry from the last converged state when `|R|inf > 1e-4`,
`assert`-fail if the step collapses. Newton uses `epsN=1e-6` (softer
than the pencil's `epsP=1e-8`, which is too stiff for SNES). Cavity
`on()` order matters: lid BC must come LAST — corner DOFs move with the
lid; wall-last stalls SNES at ~1e-4. `res_base` in `info.txt` records
the base-flow residual — check it's ≲1e-4 before trusting a pencil.

```sh
julia --project=. bench/gen_fem_matrices.jl            # all jobs
julia --project=. bench/gen_fem_matrices.jl cavity:4000 # one case
julia --project=. bench/gen_fem_matrices.jl --skip-ff channel # reconvert only
julia --project=. bench/gen_fem_matrices.jl --apptainer # apptainer instead of docker
julia --project=. bench/gen_fem_matrices.jl --apptainer --jobs=8  # parallel containers
```

`--apptainer` uses `apptainer exec --cleanenv --bind <repo>:/work` with
`docker://$FREEFEM_IMAGE` (or a local `.sif` path in `FREEFEM_IMAGE`) —
the HPC path; identical `.edp` scripts and outputs either way.

`bench/make_fem_matrices.jl` does the conversion standalone
(`--mode pencil|shiftinvert`) and is `include`d by the generator.

Geometries: cavity (lid on top, PETSc Newton base flow; discrete
Re_c≈4000 on N=16 — physical Re_c≈8000), cylinder wake (Hopf, Re_c≈46
resolved — shifts UP on coarse meshes), Poiseuille channel (x-periodic
fespace, analytic U=1-y², Re_c=5772).

Benchmark a pencil (`largest` needs `--search-shift` for singular B;
target slightly right of the leading eigenvalue — see the `%` comment
block inside each .mtx for ARPACK reference eigenvalues):

```sh
julia --project=. bench/run_benchmark.jl \
    --matrices matrices/fem/fem_channel_L2piN16_Re5000.mtx \
    --out results/fem.json --start-strategy largest --search-shift 0.05
#    Complex pairs need --eltype ComplexF64.
```

Quick sanity check (all methods converge on a small symmetric matrix):

```sh
jld eval --scratch 'include("src/RQIBench.jl"); using .RQIBench; RQIBench.sanity_check()'
```

Note: `jld run` injects daemon args into `ARGS`; scripts that parse `ARGS`
must filter `--*` entries (see `make_matrices.jl`) or run under plain `julia`.
