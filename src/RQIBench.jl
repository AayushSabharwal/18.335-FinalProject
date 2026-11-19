"""
    RQIBench

Benchmark infrastructure for variants of (inexact) Rayleigh quotient
iteration. All variants share one outer iteration; they differ only in how
the inner shifted solve `(A - σI) w = v` is performed. Inner solves go
through LinearSolve.jl (dense LU via LAPACK, GMRES via Krylov.jl).
"""
module RQIBench

using LinearAlgebra
using LinearSolve
import SciMLBase
using DelimitedFiles
using Printf
using Random

export RQIMethod, FixedShiftLU, RefactorLU, GMRESInputPrec, GMRESPrevPrec,
       RQICounters, RQIResult, run_rqi, run_oracle, load_matrix,
       method_tag, nearest_oracle, eigenvector_error, sanity_check

# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

abstract type RQIMethod end

"""Inverse iteration: factorize `A - σ0 I` once and reuse it for every solve."""
struct FixedShiftLU <: RQIMethod end

"""Classic RQI: refactorize `A - σ I` (dense LU) every time the shift moves."""
struct RefactorLU <: RQIMethod end

"""RQI with GMRES inner solves, left-preconditioned by `lu(A - σ0 I)` (fixed)."""
struct GMRESInputPrec <: RQIMethod end

"""
RQI with GMRES inner solves, left-preconditioned by `lu(A - σfac I)` where
`σfac` is the most recently *factorized* shift. A new preconditioner is
factorized only when `|σ - σfac| > tau * max(|σfac|, ||A||_∞)` or when the
previous GMRES solve exhausted its iteration budget. `tau = 0` refreshes the
preconditioner every iteration (the "previous factorization" lag-one scheme);
`tau = Inf` is equivalent to [`GMRESInputPrec`](@ref).
"""
struct GMRESPrevPrec <: RQIMethod
    tau::Float64
end

method_tag(::FixedShiftLU) = "fixed_shift_lu"
method_tag(::RefactorLU) = "rqi_lu"
method_tag(::GMRESInputPrec) = "rqi_gmres_input_prec"
method_tag(::GMRESPrevPrec) = "rqi_gmres_prev_prec"

# ---------------------------------------------------------------------------
# Counters and results
# ---------------------------------------------------------------------------

mutable struct RQICounters
    outer_iters::Int
    n_shifts::Int                    # times the solve matrix (A - σI) changed
    n_factorizations::Int
    n_perturbations::Int             # times σ was nudged to dodge a singular matrix
    gmres_iters_total::Int
    gmres_stalled::Int               # GMRES solves that hit inner_maxiter
    gmres_iters_per_step::Vector{Int}
    shift_history::Vector{Float64}
    residual_history::Vector{Float64}
end

RQICounters() = RQICounters(0, 0, 0, 0, 0, 0, Int[], Float64[], Float64[])

struct RQIResult{T}
    method::String
    eigenvalue::T
    eigenvector::Vector{T}
    final_residual::T
    converged::Bool
    fail_reason::Union{String,Nothing}
    counters::RQICounters
end

# ---------------------------------------------------------------------------
# Internal solver state
# ---------------------------------------------------------------------------

const DenseLU = LU{Float64,Matrix{Float64},Vector{Int64}}

mutable struct SolverState{M,C}
    method::M
    cache::C                    # LinearSolve LinearCache
    S::Matrix{Float64}          # workspace for the shifted matrix A - σI
    prec::Ref{Union{Nothing,DenseLU}}  # current preconditioner factorization (GMRES)
    σ_solve::Float64            # shift used by the most recent solve
    σ_fac::Float64              # shift of the most recent factorization
    normA::Float64
    inner_maxiter::Int
    counters::RQICounters
    stall_next::Bool            # previous GMRES hit maxiters → refactorize prec
end

shift_into!(S, A, σ) = (copyto!(S, A); S[diagind(S)] .-= σ; S)

_singular_perturb(σ, normA, k = 1) =
    σ + sign(σ == 0 ? 1 : σ) * eps(normA) * max(abs(σ), normA) * 8.0^(k - 1)

"""Factorize `S` with dense LU, nudging the shift if the matrix is singular."""
function _factorize_shifted!(cnt, S, A, σ, normA)
    F = lu(S; check = false)
    attempts = 0
    while !issuccess(F) && attempts < 3
        σ = _singular_perturb(σ, normA, attempts + 1)
        shift_into!(S, A, σ)
        F = lu(S; check = false)
        cnt.n_perturbations += 1
        attempts += 1
    end
    return F, σ
end

function _new_cache(Ashift, b, ::Union{FixedShiftLU,RefactorLU}; kwargs...)
    prob = LinearProblem(Ashift, b)
    init(prob, LUFactorization(); verbose = false)
end

function _new_cache(Ashift, b, ::Union{GMRESInputPrec,GMRESPrevPrec}, prec;
                    inner_reltol, inner_abstol, inner_maxiter, gmres_restart)
    prec_ref = Ref{Union{Nothing,DenseLU}}(prec)
    alg = KrylovJL_GMRES(gmres_restart = gmres_restart;
                         precs = (A_, p_) -> (prec_ref[], LinearAlgebra.I))
    prob = LinearProblem(Ashift, b)
    cache = init(prob, alg; reltol = inner_reltol, abstol = inner_abstol,
                 maxiters = inner_maxiter, verbose = false)
    return cache, prec_ref
end

function init_state(method::Union{FixedShiftLU,RefactorLU}, A, σ0, v, normA;
                    kwargs...)
    cnt = RQICounters()
    n = size(A, 1)
    S = similar(A, Float64)
    shift_into!(S, A, σ0)
    cache = _new_cache(copy(S), copy(v), method)
    cnt.n_factorizations = 1
    return SolverState(method, cache, S, Ref{Union{Nothing,DenseLU}}(nothing), σ0, σ0, normA,
                       get(kwargs, :inner_maxiter, 0), cnt, false)
end

function init_state(method::Union{GMRESInputPrec,GMRESPrevPrec}, A, σ0, v, normA;
                    inner_reltol, inner_abstol, inner_maxiter, gmres_restart)
    cnt = RQICounters()
    n = size(A, 1)
    S = similar(A, Float64)
    F, σf = _factorize_shifted!(cnt, shift_into!(S, A, σ0), A, σ0, normA)
    cache, prec_ref = _new_cache(copy(S), copy(v), method, F;
                                 inner_reltol, inner_abstol,
                                 inner_maxiter = inner_maxiter <= 0 ? n : inner_maxiter,
                                 gmres_restart)
    cnt.n_factorizations = 1
    return SolverState(method, cache, S, prec_ref, σ0, σf, normA,
                       inner_maxiter <= 0 ? n : inner_maxiter, cnt, false)
end

# ---------------------------------------------------------------------------
# Inner solve dispatch
# ---------------------------------------------------------------------------

# --- direct LU -------------------------------------------------------------

function solve_step!(st::SolverState{<:Union{FixedShiftLU,RefactorLU}}, A, σ, v)
    cnt = st.counters
    # FixedShiftLU keeps solving at the (possibly perturbed) initial shift and
    # never touches cache.A, so the init-time factorization is reused. Any
    # other variant refactorizes whenever the solve shift moved.
    σsolve = st.method isa FixedShiftLU ? st.σ_solve : σ
    if σsolve != st.σ_solve
        cnt.n_shifts += 1
        shift_into!(st.S, A, σsolve)
        st.cache.A = st.S          # flags isfresh → refactorizes on solve
        cnt.n_factorizations += 1
    end
    for attempt in 1:3
        st.cache.b = v
        sol = solve!(st.cache)
        if SciMLBase.successful_retcode(sol.retcode) && all(isfinite, sol.u)
            st.σ_solve = σsolve
            return sol.u, 0, true
        end
        σsolve = _singular_perturb(σsolve, st.normA, attempt)
        cnt.n_perturbations += 1
        cnt.n_shifts += 1
        shift_into!(st.S, A, σsolve)
        st.cache.A = st.S
        cnt.n_factorizations += 1
    end
    return nothing, 0, false
end

# --- GMRES -----------------------------------------------------------------

"""
Factorization policy for the GMRES preconditioner. GMRESInputPrec never
refactorizes; GMRESPrevPrec refactorizes when the shift moved more than the
relative threshold `tau`, or after a stalled solve.
"""
function _maybe_refactor_prec!(st::SolverState{<:GMRESPrevPrec}, A, σ)
    cnt = st.counters
    scale = max(abs(st.σ_fac), st.normA)
    if st.stall_next || abs(σ - st.σ_fac) > st.method.tau * scale
        F, σf = _factorize_shifted!(cnt, shift_into!(st.S, A, σ), A, σ, st.normA)
        st.prec[] = F
        st.σ_fac = σf
        cnt.n_factorizations += 1
        st.stall_next = false
        return true
    end
    return false
end

_maybe_refactor_prec!(st::SolverState{GMRESInputPrec}, A, σ) = false

function solve_step!(st::SolverState{<:Union{GMRESInputPrec,GMRESPrevPrec}}, A, σ, v)
    cnt = st.counters
    _maybe_refactor_prec!(st, A, σ)
    if σ != st.σ_solve
        cnt.n_shifts += 1
        shift_into!(st.S, A, σ)
        st.cache.A = st.S          # fresh A → rebuild Krylov workspace, refresh prec
    end
    st.cache.b = v
    sol = solve!(st.cache)
    st.σ_solve = σ
    u = sol.u
    inner = Int(sol.iters)
    ok = all(isfinite, u)
    if !SciMLBase.successful_retcode(sol.retcode) || inner >= st.inner_maxiter
        cnt.gmres_stalled += 1
        st.stall_next = true
    end
    return ok ? u : nothing, inner, ok
end

# ---------------------------------------------------------------------------
# Outer iteration (shared by every method)
# ---------------------------------------------------------------------------

"""
    run_rqi(method, A; v0, σ0, tol, maxiter, inner_reltol, inner_abstol,
            inner_maxiter, gmres_restart) -> RQIResult

Run one eigenpair iteration. `tol` is relative to `||A||_∞`: the loop stops
when `||Av - σv|| ≤ tol·||A||_∞`. For [`FixedShiftLU`](@ref) the solve shift
is frozen at `σ0` while `σ` still tracks the Rayleigh quotient for reporting.
"""
function run_rqi(method::RQIMethod, A::AbstractMatrix{<:Real};
                 v0::AbstractVector, σ0::Real,
                 tol::Real = 1e-10, maxiter::Integer = 50,
                 inner_reltol::Real = 1e-10, inner_abstol::Real = 0.0,
                 inner_maxiter::Integer = 0, gmres_restart::Integer = 0)
    A = Matrix{Float64}(A)
    n = size(A, 1)
    normA = opnorm(A, Inf)
    v = collect(Float64, v0)
    v ./= norm(v)
    σ = Float64(σ0)
    st = init_state(method, A, σ, v, normA;
                    inner_reltol, inner_abstol = Float64(inner_abstol),
                    inner_maxiter = Int(inner_maxiter),
                    gmres_restart = Int(gmres_restart))
    cnt = st.counters

    converged = false
    fail_reason = nothing
    res = Inf
    Av = similar(v)
    for it in 1:maxiter
        cnt.outer_iters = it
        w, inner_iters, ok = solve_step!(st, A, σ, v)
        cnt.gmres_iters_total += inner_iters
        inner_iters > 0 && push!(cnt.gmres_iters_per_step, inner_iters)
        if !ok || w === nothing
            fail_reason = "linear solve failed"
            break
        end
        nw = norm(w, Inf)
        if !(isfinite(nw) && nw > 0)
            fail_reason = "diverged (nonfinite or zero iterate)"
            break
        end
        v = normalize(w ./ nw)
        mul!(Av, A, v)
        σ = dot(v, Av)
        res = norm(Av .- σ .* v)
        push!(cnt.shift_history, σ)
        push!(cnt.residual_history, res)
        if res <= tol * normA
            converged = true
            break
        end
    end

    return RQIResult(method_tag(method), σ, v, res, converged, fail_reason, cnt)
end

# ---------------------------------------------------------------------------
# Oracle and I/O
# ---------------------------------------------------------------------------

"""Full eigendecomposition baseline. Returns the `Eigen` factorization."""
run_oracle(A::AbstractMatrix{<:Real}) = eigen(Matrix{Float64}(A))

"""
    load_matrix(path) -> Matrix{Float64}

Read a dense matrix from a delimited text file. The delimiter is sniffed from
the first non-blank line: comma if one appears, else whitespace.
"""
function load_matrix(path::AbstractString)
    firstline = ""
    open(path) do io
        for line in eachline(io)
            s = strip(line)
            if !isempty(s) && !startswith(s, "#")
                firstline = s
                break
            end
        end
    end
    isempty(firstline) && error("empty matrix file: $path")
    dlm = occursin(',', firstline) ? ',' : nothing
    M = dlm === nothing ? readdlm(path, Float64, comments = true) :
                        readdlm(path, dlm, Float64, comments = true)
    M isa Matrix || (M = reshape(M, :, 1))
    size(M, 1) == size(M, 2) || error("matrix in $path is not square: $(size(M))")
    return Matrix{Float64}(M)
end

"""Nearest oracle eigenpair to `λ`; returns (index, λ_oracle, eigenvector)."""
function nearest_oracle(F::Eigen, λ)
    i = argmin(abs.(F.values .- λ))
    return i, F.values[i], F.vectors[:, i]
end

"""Eigenvector distance, sign-insensitive, relative to unit vectors."""
function eigenvector_error(v, u)
    d = min(norm(v .- u), norm(v .+ u))
    return d / norm(v)
end

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

"""
    sanity_check() -> Bool

Run every method on a small symmetric matrix with the initial shift placed
near a true eigenvalue, and assert they all converge to it.
"""
function sanity_check(; seed = 0, n = 30, tol = 1e-10)
    rng = MersenneTwister(seed)
    A = randn(rng, n, n)
    A = (A + A') / 2
    F = run_oracle(A)
    v0 = normalize(rand(rng, n))
    λtarget = F.values[argmin(abs.(F.values .- dot(v0, A * v0)))]
    σ0 = λtarget + 1e-3
    for m in (FixedShiftLU(), RefactorLU(), GMRESInputPrec(), GMRESPrevPrec(0.05))
        r = run_rqi(m, A; v0 = v0, σ0 = σ0, tol = tol)
        i, λo, u = nearest_oracle(F, r.eigenvalue)
        converged_ok = r.converged && abs(r.eigenvalue - λo) < sqrt(tol) &&
                       eigenvector_error(r.eigenvector, u) < sqrt(tol)
        converged_ok ||
            error("sanity check failed for $(r.method): converged=$(r.converged) " *
                  "λ=$(r.eigenvalue) vs oracle $(λo) (res=$(r.final_residual))")
    end
    return true
end

end # module
