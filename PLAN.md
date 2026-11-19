# RQI Benchmarking — Implementation Plan

Benchmark five methods (4 RQI variants + oracle) for computing a single
eigenpair of a dense real matrix `A`, tracking the factorization-reuse vs.
re-shifting tradeoff.

## Repository layout

```
ProjectReal/
  PLAN.md                    <- this file
  Project.toml               <- environment at repo root
  Manifest.toml              <- pinned, committed for reproducibility
  src/RQIBench.jl            <- generic infrastructure (module RQIBench)
  bench/run_benchmark.jl     <- driver script
  matrices/                  <- input matrices, one per text file
  results/                   <- JSON output
```

`src/RQIBench.jl` is a plain module included by the driver (no package
structure needed). The driver is run as
`julia --project=. bench/run_benchmark.jl --matrices matrices/ --out results/bench.json`.
`Manifest.toml` is pinned and committed so the exact dependency versions
are reproducible via `Pkg.instantiate()`.

## Algorithms

All variants run the same outer iteration; they differ only in how the
inner shifted solve `(A - σ I) w = v` is performed:

```
v = v0 / ||v0||;  σ = σ0
repeat:
    w = solve (A - σ I) \ v          # method-specific
    v = w / ||w||
    σ = v' A v                        # Rayleigh quotient
    res = ||A v - σ v||
until res < tol or maxiter reached
```

Variants (`method` tag in JSON):

| tag | description |
|---|---|
| `oracle_eig` | `eigen(A)` via LAPACK; baseline. Full spectrum + eigenvectors stored once per matrix. |
| `fixed_shift_lu` | Inverse iteration: factorize `LU(A - σ0 I)` once, `set_b` + reuse for all iterations. |
| `rqi_lu` | Classic RQI: σ updated every iteration; `set_A` triggers refactorization via `LUFactorization()`. |
| `rqi_gmres_input_prec` | σ updated every iteration; inner solve by `KrylovJL_GMRES()` preconditioned on the left by `Pl = lu(A - σ0 I)`, fixed forever. |
| `rqi_gmres_prev_prec` | Same, but `Pl = lu(A - σ_fac I)` where `σ_fac` is the *last factorized* shift. Refactorization policy below. |

### Refactorization policy (the "goldilocks" knob)

`rqi_gmres_prev_prec` carries a threshold `τ`: refactorize
`Pl = lu(A - σ I)` only when `|σ - σ_fac| > τ * scale` where
`scale = max(|σ_fac|, norm(A, Inf))` (relative shift), OR when the previous
GMRES solve exceeded `inner_maxiter` (preconditioner clearly stale).
`τ = 0` degenerates to "refactorize every shift"; `τ = Inf` degenerates to
`rqi_gmres_input_prec`. The driver sweeps a list of `τ` values per matrix,
which is exactly the goldilocks curve you want.

### Inner-solve machinery

Everything goes through `LinearSolve.jl` (battle-tested; GMRES is
Krylov.jl under the hood):

```julia
prob = LinearProblem(A - σ*I, v)
cache = init(prob, LUFactorization())          # LU path
cache = init(prob, KrylovJL_GMRES(); Pl=Pl)    # GMRES path
# per iteration: LinearSolve.set_A / set_b on the cache, then solve!(cache)
# inner iteration count from sol.iters
```

Near-singular `(A - σ I)` is the norm in RQI. If `lu` reports a zero pivot
(`!issuccess(F)`), perturb `σ` by `eps(normA)` and count it in
`n_perturbations`.

## Code structure (`src/RQIBench.jl`)

- `mutable struct RQICounters`: `outer_iters, n_shifts, n_factorizations,
  n_perturbations, gmres_iters_total, gmres_iters_per_step::Vector{Int},
  shift_history, residual_history`
- `abstract type RQIMethod`; one concrete struct per variant with its
  parameters (`τ`, tolerances).
- Generic driver:
  `run_rqi(method, A; σ0, v0, tol, maxiter, inner_reltol, inner_maxiter) -> RQIResult`
  holding the single outer loop, convergence test, and counter updates;
  each method implements three hooks:
  `init_state(method, A, σ0)` / `solve_step!(state, A, σ, v)` /
  `on_shift!(state, σ_new)`. So the loop/timing/counting code is written
  once.
- `run_oracle(A)` -> `(eigenvalues, eigenvectors)`.
- `BenchmarkResult` struct + `to_dict` -> JSON.
- `load_matrix(path)`: delimited dense text via `readdlm`; delimiter
  sniffed from the first line (`,` if present, else whitespace).

## Driver script (`bench/run_benchmark.jl`)

- CLI via `ArgParse.jl`: `--matrices DIR` (or file
  list), `--out FILE`, `--tol`, `--maxiter`, `--inner-reltol`,
  `--inner-maxiter`, `--taus 0,0.01,0.1,1`, `--samples 5`, `--seed 1234`.
- Initial guess: `v0 = normalize(rand(MersenneTwister(seed), n))`
  (deterministic, recorded); `σ0 = v0' A v0` unless `--shift` given.
- Timing via `BenchmarkTools.jl` (`evals=1`, `samples=N` per trial) in
  three phases, so that method costs are decomposed and the search cost
  is included in the honest end-to-end total:
  - `build_starts` — once per matrix (shared by all methods); for
    `largest` this is the shifted power iteration.
  - `init_state` — state setup + first factorization (per method).
  - Iterate — ALL starts on ONE `SolverState`, `reset_state!` between
    them, measured as a single trial; `setup=` rebuilds the state
    untimed per sample.
  `end_to_end_runtime_ns_median = starts + init + iterate` is the honest
  cost of the search strategy. The `Trial` gives per-sample times,
  `allocs`, and `memory` estimates; eigenpairs/counters come from a
  dedicated deterministic pass that mirrors the timed loop.
- Also records `Sys.maxrss()` delta per run as a peak-memory proxy.
- Writes one JSON file at the end (also flushed incrementally per matrix
  so a crash mid-run loses nothing).

## Statistics recorded (JSON schema)

Per (matrix × method) run entry — `init`/`totals` aggregate counters, and
`starts` holds the per-start results (eigenpair, histories, oracle errors):

```json
{
  "method": "rqi_gmres_prev_prec",
  "params": {"tau": 0.1, "tol": 1e-10, "inner_reltol": 1e-10, "seed": 1234},
  "n_starts": 10,
  "init": {"n_factorizations": 1},
  "totals": {"outer_iterations": 25, "n_shifts": 25,
             "n_factorizations": 4, "gmres_iterations_total": 210},
  "starts_runtime_ns_median": 30000,
  "init_runtime_ns_median": 3000,
  "iterate_runtime_ns_median": 500000,
  "end_to_end_runtime_ns_median": 533000,
  "init_bytes_allocated": 4096,
  "iterate_bytes_allocated": 45678,
  "maxrss_delta_bytes": 0,
  "starts": [{
    "sigma0": 2.5, "converged": true, "outer_iterations": 5,
    "eigenvalue": 3.141592653589793, "eigenvector": [0.123..., ...],
    "final_residual": 1.2e-14, "residual_history": [3.4e-1, ...],
    "shift_history": [2.9, 3.13, ...],
    "oracle_nearest_eigenvalue": 3.141592653589793,
    "eigenvalue_abs_error": 1e-15,
    "eigenvector_error": 2.3e-14
  }]
}
```

Top level: julia/OS version, CPU, BLAS lib + thread count, timestamp,
per-matrix section (`n`, `symmetric`, `norm`, `file`, `oracle` results,
`runs` array).

Full precision: JSON.jl emits shortest-round-trip floats (bit-exact).
Eigenvectors serialized as plain arrays; complex pairs stored as
`{"re":..., "im":...}` components if a nonsymmetric input produces them.

## Update: multi-start + generic eltype

Each `run_rqi` still computes one eigenpair. To target the analysis goal
(find all eigenvalues / the one with largest real part), the driver runs
multiple starts per (matrix, method):

- `--start-strategy rq`: single start, σ0 = RQ(v0) (original behavior).
- `--start-strategy grid`: `--nstarts` σ0's spread evenly over the
  Gershgorin real-axis range (no oracle information used).
- `--start-strategy largest`: v0 built from shifted power iterations on
  `A - lo·I` (concentrates on the rightmost eigenvector — a raw far-shift
  start was shown to land mid-spectrum), iterating until the Rayleigh
  quotient stagnates or the `--power-iters` cap (default 200) is hit;
  σ0 = RQ(v0). For a nonsymmetric `A` whose rightmost eigenvalues form a
  complex pair, use a complex `--eltype` so RQI can converge to them.

Every run records `start_index` and `sigma0`. A per-matrix `coverage`
section reports, per method label, how many oracle eigenvalues were hit by
at least one converged run (within `match_tol`), `found_largest`, and the
summed median runtimes ("cost to find the eigenvalues").

Everything is generic over `T = --eltype` (Float16/32/64, BigFloat,
ComplexF32/64): `RQICounters{T,R}`, `RQIResult{T,R}`,
`SolverState{M,C,T,R}`. The oracle always runs in ≥ Float64 (`BigFloat`
stays `BigFloat`) so low-precision runs are measured against an accurate
reference. `eigenvector_error` is the phase-invariant `sin θ` subspace
distance.

## Environment

`Project.toml` at the repo root: `LinearSolve`, `Krylov`, `JSON`,
`ArgParse`, `BenchmarkTools` + stdlibs (`Printf`, `DelimitedFiles`,
`LinearAlgebra`, `Random`, `Dates`). `Manifest.toml` generated and
committed. FLOPS are not tracked (no reliable option on macOS).

## Testing

`sanity_check` run before writing JSON: on a random symmetric matrix,
assert all four methods converge (`res < tol`) and the eigenpair matches
the oracle's nearest eigenpair to `~tol`. Development/testing via `jld`
(the root `Project.toml` is picked up automatically).
