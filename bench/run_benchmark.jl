# Driver for the RQI benchmark. See PLAN.md for the design.
#
#   julia --project=. bench/run_benchmark.jl --matrices matrices/ --out results/bench.json

using ArgParse
using BenchmarkTools
using JSON
using LinearAlgebra
using Random
using Dates
using Printf

include(joinpath(@__DIR__, "..", "src", "RQIBench.jl"))
using .RQIBench

const ELTYPE_MAP = Dict(
    "Float16" => Float16,
    "Float32" => Float32,
    "Float64" => Float64,
    "BigFloat" => BigFloat,
    "ComplexF32" => ComplexF32,
    "ComplexF64" => ComplexF64,
    "Complex{BigFloat}" => Complex{BigFloat},
)

function parse_cli()
    s = ArgParseSettings(description = "Benchmark RQI variants on dense matrices.")
    @add_arg_table! s begin
        "--matrices", "-m"
            help = "Directory of matrix text files, or comma-separated list of files"
            arg_type = String
            required = true
        "--out", "-o"
            help = "Output JSON path"
            arg_type = String
            default = joinpath("results", "bench.json")
        "--eltype"
            help = "Element type for the computation: $(join(sort(collect(keys(ELTYPE_MAP))), ", "))"
            arg_type = String
            default = "Float64"
        "--start-strategy"
            help = "rq: single start at σ0=RQ(v0); grid: nstarts shifts spread over the Gershgorin range; largest: shifted power iterations build v0 near the rightmost eigenvector, σ0=RQ(v0)"
            arg_type = String
            default = "rq"
            range_tester = x -> x in ("rq", "grid", "largest")
        "--nstarts"
            help = "Number of starts for --start-strategy=grid"
            arg_type = Int
            default = 10
        "--power-iters"
            help = "Cap on shifted power iterations used to build v0 for --start-strategy=largest (stops early once the Rayleigh quotient stagnates)"
            arg_type = Int
            default = 200
        "--shift"
            help = "Explicit initial shift σ0 (forces a single start)"
            arg_type = Float64
            default = NaN
        "--tol"
            help = "Outer residual tolerance, relative to ||A||_Inf"
            arg_type = Float64
            default = 1e-10
        "--maxiter"
            help = "Maximum outer RQI iterations"
            arg_type = Int
            default = 50
        "--inner-reltol"
            help = "GMRES relative residual tolerance"
            arg_type = Float64
            default = 1e-10
        "--inner-abstol"
            help = "GMRES absolute residual tolerance"
            arg_type = Float64
            default = 0.0
        "--inner-maxiter"
            help = "GMRES max iterations (0 → matrix size)"
            arg_type = Int
            default = 0
        "--gmres-restart"
            help = "GMRES restart length (0 → no restart)"
            arg_type = Int
            default = 0
        "--taus"
            help = "Comma-separated refactorization thresholds for rqi_gmres_prev_prec (use 'inf' to disable refactoring)"
            arg_type = String
            default = "0,0.01,0.1,1"
        "--methods"
            help = "Comma-separated subset of: fixed_shift_lu,rqi_lu,rqi_gmres_input_prec,rqi_gmres_prev_prec (default: all)"
            arg_type = String
            default = "all"
        "--seed"
            help = "Base seed for the initial vectors v0 (start j uses seed+1000j)"
            arg_type = Int
            default = 1234
        "--samples"
            help = "BenchmarkTools samples per (matrix, method, start)"
            arg_type = Int
            default = 5
        "--match-tol"
            help = "Tolerance for matching a found eigenvalue to an oracle eigenvalue in the coverage summary (NaN → auto: max(10*tol, 10*sqrt(eps(T)))*||A||)"
            arg_type = Float64
            default = NaN
        "--skip-oracle-vectors"
            help = "Do not serialize the oracle eigenvector matrix"
            action = :store_true
        "--skip-sanity"
            help = "Skip the pre-flight sanity check"
            action = :store_true
    end
    return parse_args(s)
end

function matrix_files(spec::String)
    files = String[]
    for part in split(spec, ',')
        p = strip(part)
        if isdir(p)
            append!(files, sort!([joinpath(p, f) for f in readdir(p)
                                 if endswith(f, ".txt") && isfile(joinpath(p, f))]))
        elseif isfile(p)
            push!(files, p)
        else
            error("matrix path not found: $p")
        end
    end
    return files
end

function build_methods(methods_spec::String, taus_spec::String)
    wanted = methods_spec == "all" ?
        ["fixed_shift_lu", "rqi_lu", "rqi_gmres_input_prec", "rqi_gmres_prev_prec"] :
        split(methods_spec, ',')
    taus = [strip(t) == "inf" ? Inf : parse(Float64, t) for t in split(taus_spec, ',')]
    methods = RQIMethod[]
    "fixed_shift_lu" in wanted && push!(methods, FixedShiftLU())
    "rqi_lu" in wanted && push!(methods, RefactorLU())
    "rqi_gmres_input_prec" in wanted && push!(methods, GMRESInputPrec())
    if "rqi_gmres_prev_prec" in wanted
        for τ in taus
            push!(methods, GMRESPrevPrec(τ))
        end
    end
    return methods
end

"""
    build_starts(strategy, A; seed, nstarts, power_iters, shift) -> Vector{Tuple{Vector{T}, T}}

Initial (v0, σ0) pairs. `grid` spreads `nstarts` shifts evenly across the
Gershgorin real-axis range; `largest` builds `v0` from `power_iters`
shifted power iterations on `A - lo*I` (whose dominant eigenpair is the one
with the largest real part of `A`) and starts at `σ0 = RQ(v0)`. An explicit
`shift` always forces a single start. Takes only concrete values — the CLI
Dict is unpacked once at the call site so the (timed) search phase is
type-stable.
"""
function build_starts(strategy::String, A::AbstractMatrix{T};
                      seed::Integer, nstarts::Integer, power_iters::Integer,
                      shift::Union{Nothing,T} = nothing) where {T}
    n = size(A, 1)
    if shift !== nothing || strategy == "rq"
        v0 = normalize(rand(MersenneTwister(seed), T, n))
        σ0 = shift === nothing ? dot(v0, A * v0) : T(shift)
        return [(v0, T(σ0))]
    end
    lo, hi = gershgorin_bounds(A)
    if strategy == "largest"
        # Power iteration on A - lo·I: its eigenvalue of largest modulus is
        # λmax(A) - lo, so v0 concentrates on the rightmost eigenvector.
        # Iterate until the Rayleigh quotient stagnates (a converged
        # dominant eigenvector) or the cap is hit — a fixed small count
        # under-converges when the spectral gap is small. For a nonsymmetric
        # matrix whose rightmost eigenvalues are a complex pair, v0 can only
        # reach the invariant 2-plane and the RQ oscillates; a complex
        # --eltype is needed for RQI to converge to a complex λ.
        v0 = normalize(rand(MersenneTwister(seed), T, n))
        B = A - T(lo) * I
        scale = max(real(hi - lo), eps(real(T)))
        rq = dot(v0, A * v0)
        for _ in 1:power_iters
            w = B * v0
            nw = norm(w)
            nw < eps(real(T)) && break   # λmax == lo, degenerate
            v0 = w ./ nw
            rq_new = dot(v0, A * v0)
            abs(rq_new - rq) <= 1e-8 * scale && (rq = rq_new; break)
            rq = rq_new
        end
        return [(v0, T(rq))]
    end
    strategy != "grid" && error("unknown start strategy: $strategy")
    K = nstarts
    K < 1 && error("nstarts must be ≥ 1")
    σs = K == 1 ? [(lo + hi) / 2] : collect(range(lo, hi, K))
    return [(normalize(rand(MersenneTwister(seed + 1000j), T, n)), T(σ))
            for (j, σ) in enumerate(σs)]
end

"""Element type for the oracle: at least Float64 precision."""
oracle_eltype(::Type{T}) where {T} =
    real(T) == BigFloat ? T : (T <: Complex ? ComplexF64 : Float64)

function method_label(m::RQIMethod)
    tag = method_tag(m)
    return m isa GMRESPrevPrec ? "$tag (tau=$(m.tau))" : tag
end

# JSON.jl chokes on NaN/Inf; serialize them as null (full finite precision is
# preserved — Float64 prints shortest-round-trip).
_sanitize(x) = x
_sanitize(x::AbstractFloat) = isfinite(x) ? x : nothing
_sanitize(x::Complex) = Dict("re" => _sanitize(real(x)), "im" => _sanitize(imag(x)))
_sanitize(x::AbstractVector) = [_sanitize(v) for v in x]
_sanitize(x::AbstractMatrix) = [_sanitize(collect(r)) for r in eachrow(x)]
_sanitize(d::AbstractDict) = Dict(k => _sanitize(v) for (k, v) in pairs(d))

"""Serialization for one start's deterministic RQIResult (timing lives at
the method level: the iterate trial measures ALL starts together)."""
function start_entry_dict(r::RQIResult, start_index, σ0, oracle_F)
    i, λo, u = nearest_oracle(oracle_F, r.eigenvalue)
    c = r.counters
    _sanitize(Dict{String,Any}(
        "start_index" => start_index,
        "sigma0" => σ0,
        "converged" => r.converged,
        "fail_reason" => r.fail_reason,
        "outer_iterations" => c.outer_iters,
        "n_shifts" => c.n_shifts,
        "n_factorizations" => c.n_factorizations,
        "n_perturbations" => c.n_perturbations,
        "gmres_iterations_total" => c.gmres_iters_total,
        "gmres_stalled_solves" => c.gmres_stalled,
        "gmres_iterations_per_step" => c.gmres_iters_per_step,
        "eigenvalue" => r.eigenvalue,
        "eigenvector" => r.eigenvector,
        "final_residual" => r.final_residual,
        "residual_history" => c.residual_history,
        "shift_history" => c.shift_history,
        "oracle_nearest_eigenvalue" => λo,
        "eigenvalue_abs_error" => abs(r.eigenvalue - λo),
        "eigenvector_error" => eigenvector_error(r.eigenvector, u),
    ))
end

"""Method-level entry: one init trial, one iterate trial covering ALL starts
(same SolverState reset between them), plus the per-start results."""
function run_entry_dict(label, params, results::Vector{RQIResult}, starts,
                        init_cnt, trial, init_trial, starts_ns_median,
                        maxrss_delta, oracle_F)
    start_entries = [start_entry_dict(r, j - 1, starts[j][2], oracle_F)
                     for (j, r) in enumerate(results)]
    totals = Dict{String,Any}(
        "outer_iterations" => sum(r -> r.counters.outer_iters, results),
        "n_shifts" => sum(r -> r.counters.n_shifts, results),
        # factorizations = init's first LU + every iteration-phase fac
        "n_factorizations" =>
            init_cnt.n_factorizations +
            sum(r -> r.counters.n_factorizations, results),
        "n_perturbations" =>
            init_cnt.n_perturbations +
            sum(r -> r.counters.n_perturbations, results),
        "gmres_iterations_total" =>
            sum(r -> r.counters.gmres_iters_total, results),
        "gmres_stalled_solves" => sum(r -> r.counters.gmres_stalled, results),
    )
    _sanitize(Dict{String,Any}(
        "method" => results[1].method,
        "label" => label,
        "params" => params,
        "n_starts" => length(results),
        "init" => Dict{String,Any}(
            "n_factorizations" => init_cnt.n_factorizations,
            "n_perturbations" => init_cnt.n_perturbations,
        ),
        "totals" => totals,
        # search phase (shared across methods, timed once per matrix)
        "starts_runtime_ns_median" => starts_ns_median,
        # init phase: state setup + first factorization
        "init_runtime_ns_median" => median(init_trial.times),
        "init_runtime_ns_min" => minimum(init_trial.times),
        "init_runtime_ns_samples" => init_trial.times,
        "init_bytes_allocated" => init_trial.memory,
        "init_alloc_count" => init_trial.allocs,
        # iterate phase: ALL starts on one reused SolverState, incl. resets
        "iterate_runtime_ns_median" => median(trial.times),
        "iterate_runtime_ns_min" => minimum(trial.times),
        "iterate_runtime_ns_samples" => trial.times,
        "gc_time_ns_max" => maximum(trial.gctimes),
        "iterate_bytes_allocated" => trial.memory,
        "iterate_alloc_count" => trial.allocs,
        # honest end-to-end cost to run the search strategy
        "end_to_end_runtime_ns_median" =>
            starts_ns_median + median(init_trial.times) + median(trial.times),
        "maxrss_delta_bytes" => maxrss_delta,
        "starts" => start_entries,
    ))
end

"""
Per-method coverage over the multi-start runs: which oracle eigenvalues were
hit (within `match_tol`) by at least one converged run, plus whether the
eigenvalue with the largest real part was found.
"""
function coverage_summary(results::Vector{Tuple{String,RQIResult}}, F, match_tol,
                          total_times::Dict{String,Float64})
    labels = unique(first.(results))
    ilargest = argmax(real.(F.values))
    out = Dict{String,Any}()
    for lab in labels
        idxs = Int[]
        for (l, r) in results
            (l == lab && r.converged) || continue
            for (i, λ) in enumerate(F.values)
                if abs(r.eigenvalue - λ) <= match_tol
                    push!(idxs, i)
                end
            end
        end
        idxs = sort!(unique!(idxs))
        out[lab] = Dict{String,Any}(
            "n_eigenvalues" => length(F.values),
            "n_found" => length(idxs),
            "coverage" => length(idxs) / length(F.values),
            "oracle_indices_found" => idxs,
            "eigenvalues_found" => _sanitize(F.values[idxs]),
            "found_largest" => ilargest in idxs,
            "total_runtime_ns_median" => get(total_times, lab, NaN),
        )
    end
    return out
end

function main()
    args = parse_cli()
    T = get(ELTYPE_MAP, args["eltype"], nothing)
    T === nothing && error("unknown --eltype '$(args["eltype"])'; choices: $(join(keys(ELTYPE_MAP), ", "))")
    methods = build_methods(args["methods"], args["taus"])
    files = matrix_files(args["matrices"])
    isempty(files) && error("no matrix files found under: $(args["matrices"])")

    if !args["skip-sanity"]
        println(stderr, "sanity check (T=$T) ...")
        sanity_check(T = T)
        println(stderr, "sanity check passed")
    end

    out = Dict{String,Any}(
        "meta" => Dict{String,Any}(
            "timestamp" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
            "julia_version" => string(VERSION),
            "os" => string(Sys.MACHINE),
            "blas_config" => string(BLAS.get_config()),
            "blas_threads" => BLAS.get_num_threads(),
            "args" => args,
            "eltype" => string(T),
            "methods" => [method_label(m) for m in methods],
        ),
        "matrices" => Any[],
    )
    mkpath(dirname(abspath(args["out"])))

    for file in files
        println(stderr, "== $file")
        A = load_matrix(file, T)
        n = size(A, 1)
        normA = opnorm(A, Inf)
        # Search phase: compute the starts once per matrix (shared by all
        # methods) and time it — for `largest` this is the shifted power
        # iteration, a real cost the oracle comparison should include.
        # CLI args are unpacked into concrete values here so build_starts
        # is type-stable in the timed path.
        strategy = args["start-strategy"]
        shift_arg = isnan(args["shift"]) ? nothing : T(args["shift"])
        starts = build_starts(strategy, A;
                              seed = args["seed"],
                              nstarts = args["nstarts"],
                              power_iters = args["power-iters"],
                              shift = shift_arg)
        starts_trial = run(@benchmarkable(
            build_starts($strategy, $A;
                         seed = $(args["seed"]),
                         nstarts = $(args["nstarts"]),
                         power_iters = $(args["power-iters"]),
                         shift = $shift_arg)
        ); samples = args["samples"], evals = 1)
        match_tol = isnan(args["match-tol"]) ?
            max(10 * args["tol"], 10 * sqrt(eps(real(T)))) * normA :
            args["match-tol"]

        Toracle = oracle_eltype(T)
        Aoracle = Matrix{Toracle}(A)
        # Deterministic run (doubles as JIT warmup) + BenchmarkTools trial,
        # same as the method runs — a fair O(n³) vs O(n³) comparison.
        F = run_oracle(Aoracle)
        rss0 = Sys.maxrss()
        oracle_trial = run(@benchmarkable(run_oracle($Aoracle));
                           samples = args["samples"], evals = 1)
        oracle_dict = Dict{String,Any}(
            "eltype" => string(Toracle),
            "runtime_ns_median" => median(oracle_trial.times),
            "runtime_ns_min" => minimum(oracle_trial.times),
            "runtime_ns_samples" => oracle_trial.times,
            "gc_time_ns_max" => maximum(oracle_trial.gctimes),
            "bytes_allocated" => oracle_trial.memory,
            "alloc_count" => oracle_trial.allocs,
            "maxrss_delta_bytes" => Sys.maxrss() - rss0,
            "eigenvalues" => _sanitize(collect(F.values)),
            "eigenvectors" => args["skip-oracle-vectors"] ? nothing :
                             _sanitize(Matrix(F.vectors)),
        )

        entry = Dict{String,Any}(
            "file" => file,
            "n" => n,
            "symmetric" => issymmetric(A),
            "norm_inf" => normA,
            "eltype" => string(T),
            "match_tol" => match_tol,
            "n_starts" => length(starts),
            "starts_benchmark" => _sanitize(Dict{String,Any}(
                "runtime_ns_median" => median(starts_trial.times),
                "runtime_ns_min" => minimum(starts_trial.times),
                "runtime_ns_samples" => starts_trial.times,
                "bytes_allocated" => starts_trial.memory,
                "alloc_count" => starts_trial.allocs,
            )),
            "oracle" => oracle_dict,
            "runs" => Any[],
            "coverage" => Dict{String,Any}(),
        )

        results = Tuple{String,RQIResult}[]
        total_times = Dict{String,Float64}()
        σ0_init = T(starts[1][2])
        v0_init = normalize(Vector{T}(starts[1][1]))

        for m in methods
            label = method_label(m)
            print(stderr, "   $label ... ")
            inner_max = args["inner-maxiter"] <= 0 ? n : args["inner-maxiter"]
            params = Dict{String,Any}(
                "tol" => args["tol"], "maxiter" => args["maxiter"],
                "inner_reltol" => args["inner-reltol"],
                "inner_abstol" => args["inner-abstol"],
                "inner_maxiter" => inner_max,
                "gmres_restart" => args["gmres-restart"],
                "power_iters" => args["start-strategy"] == "largest" ?
                                 args["power-iters"] : 0,
                "seed" => args["seed"],
            )
            m isa GMRESPrevPrec && (params["tau"] = m.tau)

            # Deterministic pass: one init_state, then ALL starts on the
            # same state (reset between starts) — mirrors the timed loop.
            rss0 = Sys.maxrss()
            st = init_state(m, A, σ0_init, v0_init, normA;
                            inner_reltol = args["inner-reltol"],
                            inner_abstol = args["inner-abstol"],
                            inner_maxiter = args["inner-maxiter"],
                            gmres_restart = args["gmres-restart"])
            init_cnt = (n_factorizations = st.counters.n_factorizations,
                        n_perturbations = st.counters.n_perturbations)
            mresults = RQIResult[]
            for (v0, σ0) in starts
                reset_state!(st, A, σ0)
                push!(mresults, run_rqi!(st, A, v0, σ0;
                                        tol = args["tol"],
                                        maxiter = args["maxiter"]))
            end
            maxrss_delta = Sys.maxrss() - rss0

            # Init phase: state + first factorization.
            init_trial = run(@benchmarkable(
                init_state($m, $A, $σ0_init, $v0_init, $normA;
                           inner_reltol = $(args["inner-reltol"]),
                           inner_abstol = $(args["inner-abstol"]),
                           inner_maxiter = $(args["inner-maxiter"]),
                           gmres_restart = $(args["gmres-restart"]))
            ); samples = args["samples"], evals = 1)
            # Iterate phase: ALL starts on one reused SolverState, incl.
            # reset_state! between them; setup= rebuilds the state untimed.
            trial = run(@benchmarkable(begin
                    for (v0_j, σ0_j) in $starts
                        reset_state!(st_, $A, σ0_j)
                        run_rqi!(st_, $A, v0_j, σ0_j;
                                 tol = $(args["tol"]),
                                 maxiter = $(args["maxiter"]))
                    end
                end;
                setup = (st_ = init_state($m, $A, $σ0_init, $v0_init, $normA;
                           inner_reltol = $(args["inner-reltol"]),
                           inner_abstol = $(args["inner-abstol"]),
                           inner_maxiter = $(args["inner-maxiter"]),
                           gmres_restart = $(args["gmres-restart"])))
            ); samples = args["samples"], evals = 1)

            starts_med = median(starts_trial.times)
            push!(entry["runs"],
                  run_entry_dict(label, params, mresults, starts, init_cnt,
                                 trial, init_trial, starts_med, maxrss_delta,
                                 F))
            append!(results, ((label, r) for r in mresults))
            total_times[label] =
                starts_med + median(init_trial.times) + median(trial.times)
            nconv = count(r -> r.converged, mresults)
            tot_facs = init_cnt.n_factorizations +
                       sum(r -> r.counters.n_factorizations, mresults)
            println(stderr, "$nconv/$(length(mresults)) converged " *
                          "facs=$tot_facs λ=$(mresults[1].eigenvalue)")
        end

        entry["coverage"] = _sanitize(coverage_summary(results, F, match_tol,
                                                     total_times))
        push!(out["matrices"], entry)
        open(args["out"], "w") do io
            JSON.print(io, _sanitize(out), 4)
        end
    end
    println(stderr, "wrote $(args["out"])")
end

main()
