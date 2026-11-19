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

using EnumX
using LinearAlgebra
using LinearSolve
import SciMLBase
using DelimitedFiles
using Printf
using Random

export RQIMethod, FixedShiftLU, RefactorLU, GMRESInputPrec, GMRESPrevPrec,
       RQICounters, RQIResult, RQIFail, run_rqi, run_rqi!, init_state,
       reset_state!, run_oracle, load_matrix, load_problem, method_tag,
       nearest_oracle, eigenvector_error, gershgorin_bounds, build_starts,
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

"""Why a [`run_rqi!`](@ref) call stopped without (or with) convergence."""
@enumx RQIFail begin
    None              # converged, or nothing abnormal to report
    SolveFailed       # linear solve failed / nonfinite solution
    Diverged          # nonfinite or zero iterate
    MaxIterations     # iteration cap hit without converging
end

struct RQIResult{T,R<:Real}
    method::String
    eigenvalue::T
    eigenvector::Vector{T}
    final_residual::R
    converged::Bool
    fail_reason::RQIFail.T
    counters::RQICounters{T,R}
end

# ---------------------------------------------------------------------------
# Internal solver state
# ---------------------------------------------------------------------------

const DenseLU{T} = LU{T,Matrix{T},Vector{LinearAlgebra.BlasInt}}

mutable struct SolverState{M,C,T,R<:Real}
    method::M
    cache::C                            # LinearSolve LinearCache
    S::Matrix{T}                        # workspace for the shifted matrix A - σB
    prec::Union{Nothing,DenseLU{T}}     # preconditioner; refactored in place
    σ_solve::T                          # shift used by the most recent solve
    σ_fac::T                            # shift of the most recent factorization
    normA::R
    inner_maxiter::Int
    counters::RQICounters{T,R}
    stall_next::Bool                    # previous GMRES hit maxiters → refactorize
    v_work::Vector{T}                   # current iterate (normalize! in place)
    Av::Vector{T}                       # A*v, then overwritten as the residual
    Bv::Vector{T}                       # B*v; also the solve right-hand side
end

# S = A - σ B.  B = I keeps the cheap diagonal update for standard problems.
shift_into!(S::AbstractMatrix{T}, A, B::UniformScaling, σ::T) where {T} =
    (copyto!(S, A); S[diagind(S)] .-= σ; S)
shift_into!(S::AbstractMatrix{T}, A, B::AbstractMatrix, σ::T) where {T} =
    (copyto!(S, A); axpy!(-σ, B, S); S)

function _singular_perturb(σ::T, normA::Real, k = 1) where {T<:Number}
    s = sign(σ)
    s == 0 && (s = one(T))
    return T(σ + s * eps(normA) * max(abs(σ), normA) * 8^(k - 1))
end

"""Factorize `A - σI` in place into `F.factors`/`F.ipiv` via `getrf!`,
nudging the shift if the matrix is singular. `F` keeps its identity (so
references held elsewhere, e.g. a preconditioner slot, see the new
contents); returns the possibly perturbed σ."""
function _factorize_shifted!(cnt, F::DenseLU{T}, A, B, σ::T, normA) where {T}
    info = zero(LinearAlgebra.BlasInt)
    attempts = 0
    while attempts == 0 || (info > 0 && attempts < 4)
        shift_into!(F.factors, A, B, σ)
        _, _, info = LAPACK.getrf!(F.factors, F.ipiv; check = false)
        attempts += 1
        info > 0 || break
        σ = _singular_perturb(σ, normA, attempts)
        cnt.n_perturbations += 1
    end
    return σ
end

function _new_cache(Ashift, b, ::Union{FixedShiftLU,RefactorLU}; kwargs...)
    prob = LinearProblem(Ashift, b)
    init(prob, LUFactorization(); verbose = false)
end

# The precs closure captures `prec` itself: the factorization is updated
# in place by _factorize_shifted!, so the reference LinearSolve stores as
# Pl always sees current contents — no indirection needed.
function _new_cache(Ashift::Matrix{T}, b, ::Union{GMRESInputPrec,GMRESPrevPrec},
                    prec::DenseLU{T}; inner_reltol, inner_abstol,
                    inner_maxiter, gmres_restart) where {T}
    alg = KrylovJL_GMRES(gmres_restart = gmres_restart;
                         precs = (A_, p_) -> (prec, LinearAlgebra.I))
    prob = LinearProblem(Ashift, b)
    return init(prob, alg; reltol = inner_reltol, abstol = inner_abstol,
                maxiters = inner_maxiter, verbose = false)
end

function init_state(method::Union{FixedShiftLU,RefactorLU}, A::Matrix{T}, B,
                    σ0::T, v::Vector{T}, normA; kwargs...) where {T}
    cnt = RQICounters{T}()
    S = similar(A)
    shift_into!(S, A, B, σ0)
    cache = _new_cache(copy(S), copy(v), method)
    cnt.n_factorizations = 1
    return SolverState(method, cache, S, nothing, σ0, σ0, normA,
                       get(kwargs, :inner_maxiter, 0), cnt, false,
                       similar(v), similar(v), similar(v))
end

function init_state(method::Union{GMRESInputPrec,GMRESPrevPrec}, A::Matrix{T}, B,
                    σ0::T, v::Vector{T}, normA;
                    inner_reltol, inner_abstol, inner_maxiter, gmres_restart) where {T}
    cnt = RQICounters{T}()
    n = size(A, 1)
    S = similar(A)
    F = lu(shift_into!(S, A, B, σ0); check = false)
    σf = issuccess(F) ? σ0 : _factorize_shifted!(cnt, F, A, B, σ0, normA)
    cache = _new_cache(copy(S), copy(v), method, F;
                       inner_reltol, inner_abstol,
                       inner_maxiter = inner_maxiter <= 0 ? n : inner_maxiter,
                       gmres_restart)
    cnt.n_factorizations = 1
    return SolverState(method, cache, S, F, σ0, σf, normA,
                       inner_maxiter <= 0 ? n : inner_maxiter, cnt, false,
                       similar(v), similar(v), similar(v))
end

# ---------------------------------------------------------------------------
# Inner solve dispatch
# ---------------------------------------------------------------------------

# --- direct LU -------------------------------------------------------------

# Shared direct-LU solve loop with singular-shift perturbation. `cache.A =`
# assignment marks the cache isfresh → the next solve! refactorizes; no
# reinit! needed.
function _solve_direct!(st, A, B, σsolve, Bv)
    cnt = st.counters
    for attempt in 1:3
        st.cache.b = Bv
        sol = solve!(st.cache)
        if SciMLBase.successful_retcode(sol.retcode) && all(isfinite, sol.u)
            st.σ_solve = σsolve
            return sol.u, 0, true
        end
        σsolve = _singular_perturb(σsolve, st.normA, attempt)
        cnt.n_perturbations += 1
        cnt.n_shifts += 1
        shift_into!(st.S, A, B, σsolve)
        st.cache.A = st.S
        cnt.n_factorizations += 1
    end
    return nothing, 0, false
end

# FixedShiftLU keeps solving at the (possibly perturbed) initial shift and
# never touches cache.A outside the perturbation path — the init-time
# factorization is reused.
solve_step!(st::SolverState{<:FixedShiftLU}, A, B, σ, Bv) =
    _solve_direct!(st, A, B, st.σ_solve, Bv)

# RefactorLU refactorizes whenever the solve shift moved.
function solve_step!(st::SolverState{<:RefactorLU}, A, B, σ, Bv)
    cnt = st.counters
    if σ != st.σ_solve
        cnt.n_shifts += 1
        shift_into!(st.S, A, B, σ)
        st.cache.A = st.S
        cnt.n_factorizations += 1
    end
    return _solve_direct!(st, A, B, σ, Bv)
end

# --- GMRES -----------------------------------------------------------------

"""
Factorization policy for the GMRES preconditioner. GMRESInputPrec never
refactorizes; GMRESPrevPrec refactorizes when the shift moved more than the
relative threshold `tau`, or after a stalled solve.
"""
function _maybe_refactor_prec!(st::SolverState{<:GMRESPrevPrec}, A, B, σ)
    cnt = st.counters
    scale = max(abs(st.σ_fac), st.normA)
    if st.stall_next || abs(σ - st.σ_fac) > st.method.tau * scale
        st.σ_fac = _factorize_shifted!(cnt, st.prec, A, B, σ, st.normA)
        cnt.n_factorizations += 1
        st.stall_next = false
        return true
    end
    return false
end

_maybe_refactor_prec!(st::SolverState{GMRESInputPrec}, A, B, σ) = false

function solve_step!(st::SolverState{<:Union{GMRESInputPrec,GMRESPrevPrec}}, A, B, σ, Bv)
    cnt = st.counters
    _maybe_refactor_prec!(st, A, B, σ)
    if σ != st.σ_solve
        cnt.n_shifts += 1
        shift_into!(st.S, A, B, σ)
        st.cache.A = st.S          # fresh A → rebuild Krylov workspace, refresh prec
    end
    st.cache.b = Bv
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
    reset_state!(st::SolverState, A, σ0)

Reset the per-run counters so `st` can be reused for another [`run_rqi!`](@ref)
call — benchmarks amortize one `init_state` over several starts.

Per-method semantics for the new start's σ0:

- [`FixedShiftLU`](@ref): the factorization IS the solver, so a different
  start needs its own — refactorize `A - σ0 I` whenever σ0 moved (each
  start then targets the eigenvalue nearest *its* σ0, not the init one).
- [`RefactorLU`](@ref): invalidate `σ_solve`; the first solve refactorizes
  lazily through the normal shift-change path.
- GMRES methods: invalidate `σ_solve` (fresh `cache.A`), but `st.prec` /
  `σ_fac` carry over — for `GMRESInputPrec` the preconditioner stays the
  init factorization; for `GMRESPrevPrec` the previous start's last
  factorization is the initial preconditioner and τ decides refreshes.
"""
function reset_state!(st::SolverState{<:FixedShiftLU,<:Any,T}, A, B, σ0) where {T}
    st.counters = RQICounters{T}()
    σ = T(σ0)
    if σ != st.σ_solve
        st.counters.n_shifts += 1
        shift_into!(st.S, A, B, σ)
        st.cache.A = st.S      # isfresh → refactorizes on next solve
        st.counters.n_factorizations += 1
        st.σ_solve = σ
    end
    return st
end

function reset_state!(st::SolverState{<:Any,<:Any,T}, A, B, σ0) where {T}
    st.counters = RQICounters{T}()
    st.σ_solve = T(NaN)     # force cache.A refresh on next solve
    return st
end

"""
    run_rqi!(st::SolverState, A, B, v0, σ0; tol, maxiter) -> RQIResult

The outer eigenpair iteration on a prepared [`init_state`](@ref): solves
for a single eigenpair of the pencil `(A, B)` and stops when
`||Av - σBv|| ≤ tol·||A||_∞` (or `maxiter` is hit). Each step solves
`(A - σB) w = Bv` and updates `σ` to the generalized Rayleigh quotient
`(v*Av)/(v*Bv)`. `B = I` recovers the standard problem. Splitting this
from `init_state` lets the benchmark time state/factorization setup
separately from the iteration itself.
"""
function run_rqi!(st::SolverState{<:Any,<:Any,T}, A::Matrix{T}, B,
                  v0::Vector{T}, σ0::Number;
                  tol::Real = 1e-10, maxiter::Integer = 50) where {T<:Number}
    normA = st.normA
    v = normalize!(copyto!(st.v_work, v0))
    Av = st.Av
    Bv = st.Bv
    mul!(Bv, B, v)              # RHS of the first shifted solve
    σ = T(σ0)
    cnt = st.counters
    sizehint!(cnt.shift_history, maxiter)
    sizehint!(cnt.residual_history, maxiter)
    sizehint!(cnt.gmres_iters_per_step, maxiter)

    converged = false
    fail_reason = RQIFail.None
    res = normA
    for it in 1:maxiter
        cnt.outer_iters = it
        w, inner_iters, ok = solve_step!(st, A, B, σ, Bv)
        cnt.gmres_iters_total += inner_iters
        inner_iters > 0 && push!(cnt.gmres_iters_per_step, inner_iters)
        if !ok || w === nothing
            fail_reason = RQIFail.SolveFailed
            break
        end
        nw = norm(w, Inf)
        if !(isfinite(nw) && nw > 0)
            fail_reason = RQIFail.Diverged
            break
        end
        copyto!(v, w)
        normalize!(v)
        mul!(Av, A, v)
        mul!(Bv, B, v)
        σ = dot(v, Av) / dot(v, Bv)
        axpy!(-σ, Bv, Av)        # Av -= σ·Bv → Av now holds the residual
        res = norm(Av)
        push!(cnt.shift_history, σ)
        push!(cnt.residual_history, res)
        if res <= tol * normA
            converged = true
            break
        end
    end
    !converged && fail_reason == RQIFail.None &&
        (fail_reason = RQIFail.MaxIterations)

    return RQIResult(method_tag(st.method), σ, copy(v), res, converged,
                     fail_reason, cnt)
end

"""
    run_rqi(method, A; v0, σ0, tol, maxiter, inner_reltol, inner_abstol,
            inner_maxiter, gmres_restart) -> RQIResult

One-call convenience wrapper: [`init_state`](@ref) + [`run_rqi!`](@ref).
For [`FixedShiftLU`](@ref) the solve shift is frozen at `σ0` while `σ`
still tracks the Rayleigh quotient for reporting. Generic over the element
type `T` of `A`.
"""
function run_rqi(method::RQIMethod, A::AbstractMatrix{T};
                 v0::AbstractVector, σ0::Number, B = I,
                 tol::Real = 1e-10, maxiter::Integer = 50,
                 inner_reltol::Real = 1e-10, inner_abstol::Real = 0,
                 inner_maxiter::Integer = 0, gmres_restart::Integer = 0) where {T<:Number}
    A = Matrix{T}(A)
    Bm = B isa UniformScaling ? B : Matrix{T}(B)
    normA = opnorm(A, Inf)
    v = normalize(Vector{T}(v0))
    σ = T(σ0)
    rt = real(T)
    st = init_state(method, A, Bm, σ, v, normA;
                    inner_reltol = rt(inner_reltol),
                    inner_abstol = rt(inner_abstol),
                    inner_maxiter = Int(inner_maxiter),
                    gmres_restart = Int(gmres_restart))
    return run_rqi!(st, A, Bm, v, σ; tol = tol, maxiter = maxiter)
end

# ---------------------------------------------------------------------------
# Oracle and I/O
# ---------------------------------------------------------------------------

"""Full eigendecomposition baseline. Returns the `Eigen` factorization."""
run_oracle(A::AbstractMatrix) = eigen(Matrix(A))

"""Generalized eigendecomposition baseline (`Ax = λBx`, QZ). Returns the
`GeneralizedEigen` factorization; `B` singular yields infinite eigenvalues."""
run_oracle(A::AbstractMatrix, B::AbstractMatrix) = eigen(Matrix(A), Matrix(B))

"""
    load_matrix(path, T = Float64) -> Matrix{T}

Read a dense matrix from a delimited text file and convert to eltype `T`.
The delimiter is sniffed from the first non-blank line: comma if one
appears, else whitespace.
"""
function _read_delim(path::AbstractString)
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
    return M isa Matrix ? M : reshape(M, :, 1)
end

function load_matrix(path::AbstractString, ::Type{T} = Float64) where {T<:Number}
    M = _read_delim(path)
    size(M, 1) == size(M, 2) || error("matrix in $path is not square: $(size(M))")
    return Matrix{T}(M)
end

"""
    load_problem(path, T = Float64) -> (A, B)

Read a problem file. A square `n×n` file is the standard problem `(A, nothing)`;
a `2n×n` file is a stacked pencil `[A; B]` and returns the two `n×n` blocks.
"""
function load_problem(path::AbstractString, ::Type{T} = Float64) where {T<:Number}
    M = _read_delim(path)
    m, n = size(M)
    if m == n
        return Matrix{T}(M), nothing
    elseif m == 2n
        return Matrix{T}(M[1:n, :]), Matrix{T}(M[n+1:2n, :])
    end
    error("matrix in $path must be n×n (standard) or 2n×n stacked [A; B] " *
          "(generalized pencil): got $(size(M))")
end

"""Nearest finite oracle eigenpair to `λ`; returns (index, λ_oracle, eigvec).
Non-finite oracle eigenvalues (singular `B`) never match."""
function nearest_oracle(F, λ)
    i = argmin(i -> (d = abs(F.values[i] - λ); isfinite(d) ? d : Inf),
               eachindex(F.values))
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
    lo, hi = Inf, -Inf
    for (i, row) in enumerate(eachrow(A))
        r = sum(abs, row) - abs(A[i, i])
        lo = min(lo, real(A[i, i]) - r)
        hi = max(hi, real(A[i, i]) + r)
    end
    return lo, hi
end

"""
    build_starts(strategy, A, B = I; seed, nstarts, power_iters, shift) -> Vector{Tuple{Vector{T}, T}}

Initial (v0, σ0) pairs. `grid` spreads `nstarts` shifts evenly across the
Gershgorin real-axis range; `largest` builds `v0` from `power_iters`
shifted power iterations on `A - lo*I` (whose dominant eigenpair is the one
with the largest real part of `A`) and starts at `σ0 = RQ(v0)`. An explicit
`shift` always forces a single start. Takes only concrete values — the CLI
Dict is unpacked once at the call site so the (timed) search phase is
type-stable.
"""
function build_starts(strategy::String, A::AbstractMatrix{T}, B = I;
                      seed::Integer, nstarts::Integer, power_iters::Integer,
                      shift::Union{Nothing,T} = nothing) where {T}
    n = size(A, 1)
    hasB = !(B isa UniformScaling && B.λ == one(T))
    if shift !== nothing || strategy == "rq"
        v0 = normalize(rand(Xoshiro(seed), T, n))
        σ0 = if shift !== nothing
            T(shift)
        elseif hasB
            dot(v0, A * v0) / dot(v0, B * v0)
        else
            dot(v0, A * v0)
        end
        return [(v0, T(σ0))]
    end
    # Shift placement and power iteration act on the standard-form problem
    # Â = B\A (one dense solve in the timed search phase when B ≠ I).
    Â = hasB ? B \ A : A
    lo, hi = gershgorin_bounds(Â)
    if strategy == "largest"
        # Power iteration on Â - lo·I: its eigenvalue of largest modulus is
        # λmax(Â) - lo, so v0 concentrates on the rightmost eigenvector.
        # Iterate until the Rayleigh quotient stagnates (a converged
        # dominant eigenvector) or the cap is hit — a fixed small count
        # under-converges when the spectral gap is small. For a nonsymmetric
        # matrix whose rightmost eigenvalues are a complex pair, v0 can only
        # reach the invariant 2-plane and the RQ oscillates; a complex
        # --eltype is needed for RQI to converge to a complex λ.
        v0 = rand(Xoshiro(seed), T, n)
        normalize!(v0)
        w = similar(v0)
        Av = similar(v0)
        scale = max(real(hi - lo), eps(real(T)))
        mul!(Av, Â, v0)
        rq = dot(v0, Av)
        for _ in 1:power_iters
            # (Â - lo·I)·v0 without materializing the shifted matrix
            mul!(w, Â, v0)
            axpy!(-T(lo), v0, w)
            nw = norm(w)
            nw < eps(real(T)) && break   # λmax == lo, degenerate
            @. v0 = w / nw
            mul!(Av, Â, v0)
            rq_new = dot(v0, Av)
            abs(rq_new - rq) <= 1e-8 * scale && (rq = rq_new; break)
            rq = rq_new
        end
        return [(v0, T(rq))]
    end
    strategy != "grid" && error("unknown start strategy: $strategy")
    K = nstarts
    K < 1 && error("nstarts must be ≥ 1")
    σs = K == 1 ? [(lo + hi) / 2] : collect(range(lo, hi, K))
    return [(normalize!(rand(Xoshiro(seed + 1000j), T, n)), T(σ))
            for (j, σ) in enumerate(σs)]
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

    # Generalized pencil (A, B) with a well-conditioned B.
    Bm = T.(diagm(0 => 1 .+ rand(rng, n)))
    Fg = run_oracle(A, Bm)
    ig = argmax(real.(Fg.values))
    λg, ug = Fg.values[ig], Fg.vectors[:, ig]
    for m in (FixedShiftLU(), RefactorLU(), GMRESInputPrec(), GMRESPrevPrec(0.05))
        r = run_rqi(m, A; v0 = ug, σ0 = real(λg) + rt(1e-3), B = Bm, tol = tol)
        converged_ok = r.converged && abs(r.eigenvalue - λg) < sqrt(tol)
        converged_ok ||
            error("generalized sanity check failed for $(r.method) (T=$T): " *
                  "converged=$(r.converged) λ=$(r.eigenvalue) vs oracle $(λg)")
    end
    return true
end

end # module
