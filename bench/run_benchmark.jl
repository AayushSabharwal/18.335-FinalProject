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
            help = "rq: single start at σ0=RQ(v0); grid: nstarts shifts spread over the Gershgorin range; largest: shifted power iterations build v0 near the rightmost eigenvector, σ0=RQ(v0); spectrum: deflated runs find all eigenpairs"
            arg_type = String
            default = "rq"
            range_tester = x -> x in ("rq", "grid", "largest", "spectrum")
        "--nstarts"
            help = "Number of starts for --start-strategy=grid"
            arg_type = Int
            default = 10
        "--power-iters"
            help = "Shifted power iterations used to build v0 for --start-strategy=largest"
            arg_type = Int
            default = 10
        "--max-attempts"
            help = "Maximum deflated attempts for --start-strategy=spectrum (0 → 3n)"
            arg_type = Int
            default = 0
        "--skip-spectrum-vectors"
            help = "Do not serialize the per-method spectrum eigenvectors"
            action = :store_true
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
    build_starts(strategy, A, args) -> Vector{Tuple{Vector{T}, T}}

Initial (v0, σ0) pairs. `grid` spreads `--nstarts` shifts evenly across the
Gershgorin real-axis range; `largest` builds `v0` from `--power-iters`
shifted power iterations on `A - lo*I` (whose dominant eigenpair is the one
with the largest real part of `A`) and starts at `σ0 = RQ(v0)`. An explicit
`--shift` always forces a single start.
"""
function build_starts(strategy, A::AbstractMatrix{T}, args) where {T}
    n = size(A, 1)
    explicit = !isnan(args["shift"])
    if explicit || strategy == "rq"
        v0 = normalize(rand(MersenneTwister(args["seed"]), T, n))
        σ0 = explicit ? T(args["shift"]) : dot(v0, A * v0)
        return [(v0, T(σ0))]
    end
    lo, hi = gershgorin_bounds(A)
    if strategy == "largest"
        # Power iteration on A - lo·I: its eigenvalue of largest modulus is
        # λmax(A) - lo, so v0 concentrates on the rightmost eigenvector.
        v0 = normalize(rand(MersenneTwister(args["seed"]), T, n))
        B = A - T(lo) * I
        for _ in 1:args["power-iters"]
            v0 = normalize(B * v0)
        end
        return [(v0, T(dot(v0, A * v0)))]
    end
    # grid
    K = args["nstarts"]
    K < 1 && error("--nstarts must be ≥ 1")
    σs = K == 1 ? [(lo + hi) / 2] : collect(range(lo, hi, K))
    return [(normalize(rand(MersenneTwister(args["seed"] + 1000j), T, n)), T(σ))
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

function run_entry_dict(r::RQIResult, label, start_index, trial,
                        maxrss_delta, oracle_F, params)
    i, λo, u = nearest_oracle(oracle_F, r.eigenvalue)
    c = r.counters
    d = Dict{String,Any}(
        "method" => r.method,
        "label" => label,
        "start_index" => start_index,
        "params" => params,
        "converged" => r.converged,
        "fail_reason" => r.fail_reason,
        "outer_iterations" => c.outer_iters,
        "n_shifts" => c.n_shifts,
        "n_factorizations" => c.n_factorizations,
        "n_perturbations" => c.n_perturbations,
        "gmres_iterations_total" => c.gmres_iters_total,
        "gmres_stalled_solves" => c.gmres_stalled,
        "gmres_iterations_per_step" => c.gmres_iters_per_step,
        "runtime_ns_median" => median(trial.times),
        "runtime_ns_min" => minimum(trial.times),
        "runtime_ns_samples" => trial.times,
        "gc_time_ns_max" => maximum(trial.gctimes),
        "bytes_allocated" => trial.memory,
        "alloc_count" => trial.allocs,
        "maxrss_delta_bytes" => maxrss_delta,
        "eigenvalue" => r.eigenvalue,
        "eigenvector" => r.eigenvector,
        "final_residual" => r.final_residual,
        "residual_history" => c.residual_history,
        "shift_history" => c.shift_history,
        "oracle_nearest_eigenvalue" => _sanitize(λo),
        "eigenvalue_abs_error" => _sanitize(abs(r.eigenvalue - λo)),
        "eigenvector_error" => _sanitize(eigenvector_error(r.eigenvector, u)),
    )
    return _sanitize(d)
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

"""Serialize a full-spectrum run: totals, per-attempt rows, coverage vs oracle."""
function spectrum_entry_dict(sr::SpectrumResult, label, trial, maxrss_delta,
                             oracle_F, match_tol, store_vectors)
    t = sr.totals
    matched = Int[]
    errs = real(eltype(sr.eigenvalues))[]
    for λ in sr.eigenvalues
        i, λo, _ = nearest_oracle(oracle_F, λ)
        push!(matched, i)
        push!(errs, abs(λ - λo))
    end
    ilargest = argmax(real.(oracle_F.values))
    attempts = [Dict{String,Any}(
        "eigenvalue" => _sanitize(r.eigenvalue),
        "converged" => r.converged,
        "accepted" => acc,
        "outer_iterations" => r.counters.outer_iters,
        "n_shifts" => r.counters.n_shifts,
        "n_factorizations" => r.counters.n_factorizations,
        "gmres_iterations_total" => r.counters.gmres_iters_total,
        "final_residual" => _sanitize(r.final_residual),
    ) for (r, acc) in zip(sr.attempts, sr.accepted)]
    return _sanitize(Dict{String,Any}(
        "method" => sr.method,
        "label" => label,
        "eigenvalues" => sr.eigenvalues,
        "eigenvectors" => store_vectors ? sr.eigenvectors : nothing,
        "n_target" => sr.n_target,
        "n_found" => length(sr.eigenvalues),
        "n_attempts" => length(sr.attempts),
        "converged_all" => length(sr.eigenvalues) == sr.n_target,
        "totals" => Dict{String,Any}(
            "outer_iterations" => t.outer_iters,
            "n_shifts" => t.n_shifts,
            "n_factorizations" => t.n_factorizations,
            "n_perturbations" => t.n_perturbations,
            "gmres_iterations_total" => t.gmres_iters_total,
            "gmres_stalled_solves" => t.gmres_stalled,
        ),
        "attempts" => attempts,
        "runtime_ns_median" => median(trial.times),
        "runtime_ns_min" => minimum(trial.times),
        "runtime_ns_samples" => trial.times,
        "gc_time_ns_max" => maximum(trial.gctimes),
        "bytes_allocated" => trial.memory,
        "alloc_count" => trial.allocs,
        "maxrss_delta_bytes" => maxrss_delta,
        "coverage" => Dict{String,Any}(
            "n_eigenvalues" => length(oracle_F.values),
            "n_found" => length(sr.eigenvalues),
            "coverage" => length(sr.eigenvalues) / length(oracle_F.values),
            "oracle_indices_found" => sort!(matched),
            "found_largest" => ilargest in matched,
            "eigenvalue_abs_errors" => errs,
            "max_eigenvalue_abs_error" => isempty(errs) ? NaN : maximum(errs),
        ),
    ))
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
        spectrum_mode = args["start-strategy"] == "spectrum"
        starts = spectrum_mode ? Tuple{Vector{T},T}[] :
                 build_starts(args["start-strategy"], A, args)
        match_tol = isnan(args["match-tol"]) ?
            max(10 * args["tol"], 10 * sqrt(eps(real(T)))) * normA :
            args["match-tol"]

        Toracle = oracle_eltype(T)
        Aoracle = Matrix{Toracle}(A)
        rss0 = Sys.maxrss()
        oracle_t = @timed F = run_oracle(Aoracle)
        oracle_dict = Dict{String,Any}(
            "eltype" => string(Toracle),
            "runtime_ns" => oracle_t.time * 1e9,
            "bytes_allocated" => oracle_t.bytes,
            "gc_time_ns" => oracle_t.gctime * 1e9,
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
            "oracle" => oracle_dict,
            "runs" => Any[],
            "coverage" => Dict{String,Any}(),
        )

        results = Tuple{String,RQIResult}[]
        total_times = Dict{String,Float64}()

        if spectrum_mode
            entry["spectrum_runs"] = Any[]
            for m in methods
                label = method_label(m)
                print(stderr, "   spectrum $label ... ")
                sr = run_rqi_spectrum(m, A; seed = args["seed"],
                                      tol = args["tol"],
                                      maxiter = args["maxiter"],
                                      inner_reltol = args["inner-reltol"],
                                      inner_abstol = args["inner-abstol"],
                                      inner_maxiter = args["inner-maxiter"],
                                      gmres_restart = args["gmres-restart"],
                                      match_tol = match_tol,
                                      max_attempts = args["max-attempts"])
                rss0 = Sys.maxrss()
                trial = run(@benchmarkable(
                    run_rqi_spectrum($m, $A; seed = $(args["seed"]),
                            tol = $(args["tol"]),
                            maxiter = $(args["maxiter"]),
                            inner_reltol = $(args["inner-reltol"]),
                            inner_abstol = $(args["inner-abstol"]),
                            inner_maxiter = $(args["inner-maxiter"]),
                            gmres_restart = $(args["gmres-restart"]),
                            match_tol = $match_tol,
                            max_attempts = $(args["max-attempts"]))
                ); samples = args["samples"], evals = 1)
                push!(entry["spectrum_runs"],
                      spectrum_entry_dict(sr, label, trial, Sys.maxrss() - rss0,
                                          F, match_tol,
                                          !args["skip-spectrum-vectors"]))
                println(stderr, "$(length(sr.eigenvalues))/$(sr.n_target) found, " *
                              "attempts=$(length(sr.attempts)) " *
                              "facs=$(sr.totals.n_factorizations) " *
                              "gmres=$(sr.totals.gmres_iters_total)")
            end
        else
        for (si, (v0, σ0)) in enumerate(starts), m in methods
            label = method_label(m)
            print(stderr, "   start $si/$(length(starts)) $label ... ")
            inner_max = args["inner-maxiter"] <= 0 ? n : args["inner-maxiter"]
            params = Dict{String,Any}(
                "tol" => args["tol"], "maxiter" => args["maxiter"],
                "inner_reltol" => args["inner-reltol"],
                "inner_abstol" => args["inner-abstol"],
                "inner_maxiter" => inner_max,
                "gmres_restart" => args["gmres-restart"],
                "power_iters" => args["start-strategy"] == "largest" ?
                                 args["power-iters"] : 0,
                "seed" => args["seed"], "sigma0" => σ0,
            )
            m isa GMRESPrevPrec && (params["tau"] = m.tau)
            # Deterministic run: captures the eigenpair + counters.
            result = run_rqi(m, A; v0 = v0, σ0 = σ0, tol = args["tol"],
                             maxiter = args["maxiter"],
                             inner_reltol = args["inner-reltol"],
                             inner_abstol = args["inner-abstol"],
                             inner_maxiter = args["inner-maxiter"],
                             gmres_restart = args["gmres-restart"])
            rss0 = Sys.maxrss()
            trial = run(@benchmarkable(
                run_rqi($m, $A; v0 = $v0, σ0 = $σ0, tol = $(args["tol"]),
                        maxiter = $(args["maxiter"]),
                        inner_reltol = $(args["inner-reltol"]),
                        inner_abstol = $(args["inner-abstol"]),
                        inner_maxiter = $(args["inner-maxiter"]),
                        gmres_restart = $(args["gmres-restart"]))
            ); samples = args["samples"], evals = 1)
            push!(entry["runs"],
                  run_entry_dict(result, label, si - 1, trial,
                                 Sys.maxrss() - rss0, F, params))
            push!(results, (label, result))
            total_times[label] =
                get(total_times, label, 0.0) + median(trial.times)
            println(stderr, "$(result.converged ? "converged" : "FAILED") " *
                          "iters=$(result.counters.outer_iters) " *
                          "facs=$(result.counters.n_factorizations) " *
                          "λ=$(result.eigenvalue)")
        end
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
