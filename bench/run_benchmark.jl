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
        "--shift"
            help = "Initial shift σ0 (NaN → Rayleigh quotient of v0)"
            arg_type = Float64
            default = NaN
        "--seed"
            help = "Seed for the initial vector v0"
            arg_type = Int
            default = 1234
        "--samples"
            help = "BenchmarkTools samples per (matrix, method)"
            arg_type = Int
            default = 5
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

# JSON.jl chokes on NaN/Inf; serialize them as null (full finite precision is
# preserved — Float64 prints shortest-round-trip).
_sanitize(x) = x
_sanitize(x::AbstractFloat) = isfinite(x) ? x : nothing
_sanitize(x::Complex) = Dict("re" => _sanitize(real(x)), "im" => _sanitize(imag(x)))
_sanitize(x::AbstractVector) = [_sanitize(v) for v in x]
_sanitize(x::AbstractMatrix) = [_sanitize(collect(r)) for r in eachrow(x)]
_sanitize(d::AbstractDict) = Dict(k => _sanitize(v) for (k, v) in pairs(d))

complex_vector(v::AbstractVector{<:Complex}) =
    [Dict("re" => real(z), "im" => imag(z)) for z in v]
complex_vector(v::AbstractVector{<:Real}) = collect(v)

function run_entry_dict(r::RQIResult, method::RQIMethod, trial,
                        maxrss_delta, oracle_F, params)
    i, λo, u = nearest_oracle(oracle_F, r.eigenvalue)
    c = r.counters
    d = Dict{String,Any}(
        "method" => r.method,
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
    method isa GMRESPrevPrec && (d["params"]["tau"] = method.tau)
    return _sanitize(d)
end

function main()
    args = parse_cli()
    methods = build_methods(args["methods"], args["taus"])
    files = matrix_files(args["matrices"])
    isempty(files) && error("no matrix files found under: $(args["matrices"])")

    if !args["skip-sanity"]
        println(stderr, "sanity check ...")
        sanity_check()
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
            "methods" => [method_tag(m) for m in methods],
            "taus" => [m.tau for m in methods if m isa GMRESPrevPrec],
        ),
        "matrices" => Any[],
    )
    mkpath(dirname(abspath(args["out"])))

    for file in files
        println(stderr, "== $file")
        A = load_matrix(file)
        n = size(A, 1)
        normA = opnorm(A, Inf)
        v0 = normalize(rand(MersenneTwister(args["seed"]), n))
        σ0 = isnan(args["shift"]) ? dot(v0, A * v0) : args["shift"]

        rss0 = Sys.maxrss()
        oracle_t = @timed F = run_oracle(A)
        oracle_dict = Dict{String,Any}(
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
            "v0_seed" => args["seed"],
            "sigma0" => σ0,
            "oracle" => oracle_dict,
            "runs" => Any[],
        )

        for m in methods
            tag = method_tag(m) * (m isa GMRESPrevPrec ? " (tau=$(m.tau))" : "")
            print(stderr, "   $tag ... ")
            inner_max = args["inner-maxiter"] <= 0 ? n : args["inner-maxiter"]
            params = Dict{String,Any}(
                "tol" => args["tol"], "maxiter" => args["maxiter"],
                "inner_reltol" => args["inner-reltol"],
                "inner_abstol" => args["inner-abstol"],
                "inner_maxiter" => inner_max,
                "gmres_restart" => args["gmres-restart"],
                "seed" => args["seed"], "sigma0" => σ0,
            )
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
                  run_entry_dict(result, m, trial, Sys.maxrss() - rss0, F, params))
            println(stderr, "$(result.converged ? "converged" : "FAILED") " *
                          "iters=$(result.counters.outer_iters) " *
                          "facs=$(result.counters.n_factorizations) " *
                          "λ=$(result.eigenvalue)")
        end

        push!(out["matrices"], entry)
        open(args["out"], "w") do io
            JSON.print(io, _sanitize(out), 4)
        end
    end
    println(stderr, "wrote $(args["out"])")
end

main()
