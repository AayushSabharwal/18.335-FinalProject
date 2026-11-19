"""
    RQIBench

Benchmark infrastructure for variants of (inexact) Rayleigh quotient
iteration. All variants share one outer iteration; they differ only in how
the inner shifted solve `(A - σI) w = v` is performed. Inner solves go
through LinearSolve.jl (dense LU, GMRES via Krylov.jl).

Each [`run_rqi`](@ref) call computes ONE eigenpair and stops at the first
eigenvalue it converges to. Generic over the element type `T` of `A`
(Float32/64, BigFloat, ComplexF32/64, ...).
"""
module RQIBench

using LinearAlgebra
using LinearSolve
import SciMLBase
using DelimitedFiles
using Printf
using Random

export RQIMethod, FixedShiftLU, RefactorLU, GMRESInputPrec, GMRESPrevPrec,
       RQICounters, RQIResult, SpectrumResult, run_rqi, run_rqi_spectrum,
       run_oracle, load_matrix,
       method_tag, nearest_oracle, eigenvector_error, gershgorin_bounds,
       sanity_check

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

mutable struct RQICounters{T,R<:Real}
    outer_iters::Int
    n_shifts::Int                    # times the solve matrix (A - σI) changed
    n_factorizations::Int
    n_perturbations::Int             # times σ was nudged to dodge a singular matrix
    gmres_iters_total::Int
    gmres_stalled::Int               # GMRES solves that hit inner_maxiter
    gmres_iters_per_step::Vector{Int}
    shift_history::Vector{T}
    residual_history::Vector{R}
end

RQICounters{T}() where {T} =
    RQICounters{T,real(T)}(0, 0, 0, 0, 0, 0, Int[], T[], real(T)[])

struct RQIResult{T,R<:Real}
    method::String
    eigenvalue::T
    eigenvector::Vector{T}
    final_residual::R
    converged::Bool
    fail_reason::Union{String,Nothing}
    counters::RQICounters{T,R}
end

"""Result of a full-spectrum computation: the accepted eigenpairs plus one
`RQIResult` per attempt (accepted or not) and aggregated totals."""
struct SpectrumResult{T,R<:Real}
    method::String
    eigenvalues::Vector{T}
    eigenvectors::Vector{Vector{T}}
    attempts::Vector{RQIResult{T,R}}
    accepted::Vector{Bool}
    totals::RQICounters{T,R}
    n_target::Int                 # number of eigenvalues sought (size(A,1))
end

# ---------------------------------------------------------------------------
# Internal solver state
# ---------------------------------------------------------------------------

const DenseLU{T} = LU{T,Matrix{T},Vector{Int64}}

mutable struct SolverState{M,C,T,R<:Real}
    method::M
    cache::C                            # LinearSolve LinearCache
    S::Matrix{T}                        # workspace for the shifted matrix A - σI
    prec::Ref{Union{Nothing,DenseLU{T}}}  # current preconditioner (GMRES methods)
    σ_solve::T                          # shift used by the most recent solve
    σ_fac::T                            # shift of the most recent factorization
    normA::R
    inner_maxiter::Int
    counters::RQICounters{T,R}
    stall_next::Bool                    # previous GMRES hit maxiters → refactorize
end

shift_into!(S::AbstractMatrix{T}, A, σ::T) where {T} =
    (copyto!(S, A); S[diagind(S)] .-= σ; S)

function _singular_perturb(σ::T, normA::Real, k = 1) where {T<:Number}
    s = sign(σ)
    s == 0 && (s = one(T))
    return T(σ + s * eps(normA) * max(abs(σ), normA) * 8^(k - 1))
end

"""Factorize `S` with dense LU, nudging the shift if the matrix is singular."""
function _factorize_shifted!(cnt, S::Matrix{T}, A, σ::T, normA) where {T}
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

function _new_cache(Ashift::Matrix{T}, b, ::Union{GMRESInputPrec,GMRESPrevPrec},
                    prec::DenseLU{T}; inner_reltol, inner_abstol,
                    inner_maxiter, gmres_restart) where {T}
    prec_ref = Ref{Union{Nothing,DenseLU{T}}}(prec)
    alg = KrylovJL_GMRES(gmres_restart = gmres_restart;
                         precs = (A_, p_) -> (prec_ref[], LinearAlgebra.I))
    prob = LinearProblem(Ashift, b)
    cache = init(prob, alg; reltol = inner_reltol, abstol = inner_abstol,
                 maxiters = inner_maxiter, verbose = false)
    return cache, prec_ref
end

function init_state(method::Union{FixedShiftLU,RefactorLU}, A::Matrix{T}, σ0::T,
                    v::Vector{T}, normA; kwargs...) where {T}
    cnt = RQICounters{T}()
    S = similar(A)
    shift_into!(S, A, σ0)
    cache = _new_cache(copy(S), copy(v), method)
    cnt.n_factorizations = 1
    return SolverState(method, cache, S, Ref{Union{Nothing,DenseLU{T}}}(nothing),
                       σ0, σ0, normA, get(kwargs, :inner_maxiter, 0), cnt, false)
end

function init_state(method::Union{GMRESInputPrec,GMRESPrevPrec}, A::Matrix{T},
                    σ0::T, v::Vector{T}, normA;
                    inner_reltol, inner_abstol, inner_maxiter, gmres_restart) where {T}
    cnt = RQICounters{T}()
    n = size(A, 1)
    S = similar(A)
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

"""Twice-repeated classical Gram-Schmidt orthogonalization of `v` against `V`."""
function _deflate!(v, V)
    for _ in 1:2, u in V
        v .-= u .* dot(u, v)
    end
    return v
end

"""
    run_rqi(method, A; v0, σ0, tol, maxiter, inner_reltol, inner_abstol,
            inner_maxiter, gmres_restart, deflate) -> RQIResult

Run one eigenpair iteration: solves for a single eigenpair and stops when
`||Av - σv|| ≤ tol·||A||_∞` (or `maxiter` is hit). For [`FixedShiftLU`](@ref)
the solve shift is frozen at `σ0` while `σ` still tracks the Rayleigh
quotient for reporting. Generic over the element type `T` of `A`.

`deflate` may be a vector of already-converged eigenvectors; the iterate is
then reorthogonalized against them after every solve. For symmetric `A` this
keeps the iteration in the orthogonal complement of the found eigenvectors.
For nonsymmetric `A` it is only heuristic (right eigenvectors are not
mutually orthogonal), and re-converging to a found eigenpair is possible.
"""
function run_rqi(method::RQIMethod, A::AbstractMatrix{T};
                 v0::AbstractVector, σ0::Number,
                 tol::Real = 1e-10, maxiter::Integer = 50,
                 inner_reltol::Real = 1e-10, inner_abstol::Real = 0,
                 inner_maxiter::Integer = 0, gmres_restart::Integer = 0,
                 deflate::Union{Nothing,Vector{Vector{T}}} = nothing,
                 state::Union{Nothing,SolverState} = nothing) where {T<:Number}
    A = Matrix{T}(A)
    n = size(A, 1)
    normA = opnorm(A, Inf)
    v = normalize(Vector{T}(v0))
    if deflate !== nothing
        _deflate!(v, deflate)
        v ./= norm(v)
    end
    σ = T(σ0)
    rt = real(T)
    st = state === nothing ?
        init_state(method, A, σ, v, normA;
                   inner_reltol = rt(inner_reltol),
                   inner_abstol = rt(inner_abstol),
                   inner_maxiter = Int(inner_maxiter),
                   gmres_restart = Int(gmres_restart)) : state
    cnt = st.counters

    converged = false
    fail_reason = nothing
    res = normA
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
        if deflate !== nothing
            _deflate!(v, deflate)
            nv = norm(v)
            if !(isfinite(nv) && nv > 0)
                fail_reason = "iterate collapsed onto deflated subspace"
                break
            end
            v ./= nv
        end
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

"""
    run_rqi_spectrum(method, A; tol, maxiter, ..., seed, σ0, match_tol,
                     max_attempts) -> SpectrumResult

Compute all `n` eigenpairs of `A` by repeated deflated [`run_rqi`](@ref):
each attempt starts from a deterministic random `v0` orthogonalized against
the eigenvectors already found, and the iteration itself is reorthogonalized
(see the `deflate` keyword). An attempt is *accepted* if it converged and its
eigenvalue is new (no found eigenvalue within `match_tol`). Attempts continue
until `n` eigenvalues are found or `max_attempts` (default `3n`) is hit.

Aggregate cost is the headline metric: `FixedShiftLU` reuses a SINGLE LU
factorization for the whole spectrum (one `init_state`, shared across all
attempts), while `RefactorLU` pays `~k` factorizations per eigenpair (O(n⁴)
overall). For `FixedShiftLU`, `σ0` (default: the Rayleigh quotient of the
first attempt's `v0`) fixes the solve shift, so eigenvalues are collected
roughly in order of distance from it; the other methods get a fresh
`σ0 = RQ(v0)` per attempt.
"""
function run_rqi_spectrum(method::RQIMethod, A::AbstractMatrix{T};
                          tol::Real = 1e-10, maxiter::Integer = 50,
                          inner_reltol::Real = 1e-10, inner_abstol::Real = 0,
                          inner_maxiter::Integer = 0, gmres_restart::Integer = 0,
                          σ0::Union{Nothing,Number} = nothing, seed::Integer = 1234,
                          match_tol::Union{Nothing,Real} = nothing,
                          max_attempts::Integer = 0) where {T<:Number}
    A = Matrix{T}(A)
    n = size(A, 1)
    normA = opnorm(A, Inf)
    rt = real(T)
    mtol = something(match_tol, max(10 * tol, 10 * sqrt(eps(rt))) * normA)
    max_attempts <= 0 && (max_attempts = 3n)

    V = Vector{T}[]
    λs = T[]
    attempts = RQIResult{T,rt}[]
    accepted = Bool[]
    totals = RQICounters{T}()

    # FixedShiftLU: one factorization for the entire spectrum. The state is
    # built once and shared across attempts; the per-attempt counters are
    # reset so each attempt reports its own numbers while the single init
    # factorization is charged to `totals`.
    shared_state = nothing
    if method isa FixedShiftLU
        v_init = normalize(rand(MersenneTwister(seed + 1000), T, n))
        σinit = σ0 === nothing ? T(dot(v_init, A * v_init)) : T(σ0)
        shared_state = init_state(method, A, σinit, v_init, normA;
                                  inner_reltol = rt(inner_reltol),
                                  inner_abstol = rt(inner_abstol),
                                  inner_maxiter = Int(inner_maxiter),
                                  gmres_restart = Int(gmres_restart))
        totals.n_factorizations += 1
    end

    for att in 1:max_attempts
        length(λs) >= n && break
        v0 = normalize(rand(MersenneTwister(seed + 1000att), T, n))
        _deflate!(v0, V)
        nv0 = norm(v0)
        if !(isfinite(nv0) && nv0 > sqrt(eps(rt)))
            break   # found vectors already span the space
        end
        v0 ./= nv0
        σstart = σ0 === nothing ? T(dot(v0, A * v0)) : T(σ0)
        if shared_state !== nothing
            shared_state.counters = RQICounters{T}()
        end
        r = run_rqi(method, A; v0 = v0, σ0 = σstart, tol = tol,
                    maxiter = maxiter, inner_reltol = inner_reltol,
                    inner_abstol = inner_abstol, inner_maxiter = inner_maxiter,
                    gmres_restart = gmres_restart, deflate = V,
                    state = shared_state)
        push!(attempts, r)
        c = r.counters
        totals.outer_iters += c.outer_iters
        totals.n_shifts += c.n_shifts
        totals.n_factorizations += c.n_factorizations
        totals.n_perturbations += c.n_perturbations
        totals.gmres_iters_total += c.gmres_iters_total
        totals.gmres_stalled += c.gmres_stalled
        isnew = r.converged && all(λ -> abs(λ - r.eigenvalue) > mtol, λs)
        push!(accepted, isnew)
        if isnew
            push!(λs, r.eigenvalue)
            push!(V, r.eigenvector)
        end
    end

    return SpectrumResult(method_tag(method), λs, V, attempts, accepted,
                          totals, n)
end

# ---------------------------------------------------------------------------
# Oracle and I/O
# ---------------------------------------------------------------------------

"""Full eigendecomposition baseline. Returns the `Eigen` factorization."""
run_oracle(A::AbstractMatrix) = eigen(Matrix(A))

"""
    load_matrix(path, T = Float64) -> Matrix{T}

Read a dense matrix from a delimited text file and convert to eltype `T`.
The delimiter is sniffed from the first non-blank line: comma if one
appears, else whitespace.
"""
function load_matrix(path::AbstractString, ::Type{T} = Float64) where {T<:Number}
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
    return Matrix{T}(M)
end

"""Nearest oracle eigenpair to `λ`; returns (index, λ_oracle, eigenvector)."""
function nearest_oracle(F::Eigen, λ)
    i = argmin(abs.(F.values .- λ))
    return i, F.values[i], F.vectors[:, i]
end

"""
Eigenvector distance as `sin θ` between the two one-dimensional subspaces,
`sqrt(1 - |⟨u,v⟩|²/(‖u‖‖v‖))`. Sign/phase-insensitive, so it works for both
real and complex eigenvectors.
"""
eigenvector_error(v, u) =
    sqrt(max(0, 1 - abs2(dot(u, v) / (norm(u) * norm(v)))))

"""
    gershgorin_bounds(A) -> (lo, hi)

Real-axis bounds on the spectrum from the Gershgorin discs
`[Re(a_ii) - r_i, Re(a_ii) + r_i]`. For symmetric `A` these bound
`[λmin, λmax]`; otherwise they bound the real parts.
"""
function gershgorin_bounds(A::AbstractMatrix)
    d = diag(A)
    lo, hi = Inf, -Inf
    for i in axes(A, 1)
        r = sum(abs, A[i, :]) - abs(d[i])
        lo = min(lo, real(d[i]) - r)
        hi = max(hi, real(d[i]) + r)
    end
    return lo, hi
end

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

"""
    sanity_check(; T = Float64) -> Bool

Run every method on a small symmetric matrix of eltype `T` with the initial
shift placed near a true eigenvalue, and assert they all converge to it.
"""
function sanity_check(; seed = 0, n = 30, T::Type = Float64)
    rt = real(T)
    tol = max(1e-10, 10 * eps(rt))
    rng = MersenneTwister(seed)
    A = randn(rng, n, n)
    A = T.((A + A') / 2)
    F = run_oracle(A)
    v0 = normalize(rand(rng, T, n))
    λtarget = F.values[argmin(abs.(F.values .- dot(v0, A * v0)))]
    σ0 = T(real(λtarget) + rt(1e-3))
    for m in (FixedShiftLU(), RefactorLU(), GMRESInputPrec(), GMRESPrevPrec(0.05))
        r = run_rqi(m, A; v0 = v0, σ0 = σ0, tol = tol)
        i, λo, u = nearest_oracle(F, r.eigenvalue)
        converged_ok = r.converged && abs(r.eigenvalue - λo) < sqrt(tol) &&
                       eigenvector_error(r.eigenvector, u) < sqrt(tol)
        converged_ok ||
            error("sanity check failed for $(r.method) (T=$T): converged=$(r.converged) " *
                  "λ=$(r.eigenvalue) vs oracle $(λo) (res=$(r.final_residual))")
    end
    return true
end

end # module
