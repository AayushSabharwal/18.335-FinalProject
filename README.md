# RQI Benchmarking

Benchmarks comparing Rayleigh-quotient-iteration (RQI) variants against a
full-diagonalization oracle, for studying the tradeoff between factorization
reuse and re-shifting. Solves the standard problem `Ax = λx` or the
generalized pencil `Ax = λBx` (`B = I` recovers the standard case).

## Layout

- `src/RQIBench.jl` — solver methods, `SolverState`, counters, the RQI
  iteration, oracle, matrix I/O, and `build_starts` (all benchmarked code
  lives here; the driver is orchestration only)
- `bench/run_benchmark.jl` — CLI driver: timing, JSON output, optional FLOP
  instrumentation
- `bench/profile.jl` — standalone sampling profiler for the iterate phase
- `bench/output.schema.json` — JSON Schema for result files
- `bench/make_matrices.jl` — regenerates `matrices/*.txt`; `fem` mode
  converts a FreeFEM COO pencil dump into a stacked `.mtx`
- `matrices/` — input problems, one per file
- `results/` — JSON output
- `PLAN.md` — design notes and algorithm details

## Input format

Dense delimited text (`.txt`) or sparse Matrix Market (`.mtx`); `.mtx`
inputs run all solves on `SparseMatrixCSC` via UMFPACK:

- `n×n` — standard problem, `B = I`
- `2n×n` — generalized pencil, stacked blocks: `A` = first `n` rows,
  `B` = last `n` rows

## Methods

- `fixed_shift_lu` — one LU of `A − σ₀B`, reused for all iterations
- `rqi_lu` — refactorize `A − σB` whenever the shift moves
- `rqi_gmres_input_prec` — GMRES preconditioned by the initial `lu(A − σ₀B)`
- `rqi_gmres_prev_prec` — GMRES with the most recent factorization as
  preconditioner; `τ` controls when it refactorizes
- `oracle_eig` — `eigen(A)` / `eigen(A, B)` baseline (`--oracle dense`), or
  `eigs` (`--oracle arpack`); `auto` picks arpack for sparse inputs

## Running

```sh
julia --project=. bench/run_benchmark.jl \
    --matrices matrices/ --out results/bench.json --start-strategy largest
```

All options (`--eltype`, `--taus`, `--start-strategy`, `--search-shift`,
tolerances, seeds, etc.) are documented in:

```sh
julia --project=. bench/run_benchmark.jl --help
```

`--matrices` accepts a directory or a comma-separated file list. Output is
full-precision JSON validated against `bench/output.schema.json`; the
recorded `meta.args` plus per-entry `search` block make every run
reproducible. To replay a previous result file verbatim (CLI overrides
still allowed):

```sh
julia --project=. bench/run_benchmark.jl --from-results old.json --out new.json
```

## Notes

- Search strategies: `rq` (single start), `grid` (Gershgorin-shifted starts),
  `largest` (shift-invert power iteration targeting the eigenpair nearest
  the search shift; `--search-shift` required for pencils with singular `B`).
- Complex eigenpairs need `--eltype ComplexF64`.
- FLOP counts via LIKWID are collected on Linux only (`null` elsewhere).
- Develop/test with `jld` (persistent Julia daemon); the root
  `Project.toml`/`Manifest.toml` is the pinned environment — run
  `Pkg.instantiate()` once before first use.
