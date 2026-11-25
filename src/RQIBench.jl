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

using Arpack
using ArnoldiMethod
using EnumX
using LinearAlgebra
using LinearSolve
import SciMLBase
using DelimitedFiles
using Printf
using Random
using SparseArrays
import SuiteSparse

export RQIMethod, FixedShiftLU, RefactorLU, GMRESInputPrec, GMRESPrevPrec,
       RQICounters, RQIResult, RQIFail, run_rqi, run_rqi!, init_state,
       reset_state!, run_oracle, run_oracle_eigs, run_oracle_hp,
       run_oracle_arnoldi, load_matrix, load_problem, method_tag,
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

const HISTORY_CAPACITY = 1000

# Pre-size the history vectors at construction so `run_rqi!` sees no
# capacity effects; covers init_state and every reset_state! rebuild.
function RQICounters{T}() where {T}
    cnt = RQICounters{T,real(T)}(0, 0, 0, 0, 0, 0, Int[], T[], real(T)[])
    sizehint!(cnt.gmres_iters_per_step, HISTORY_CAPACITY)
    sizehint!(cnt.shift_history, HISTORY_CAPACITY)
    sizehint!(cnt.residual_history, HISTORY_CAPACITY)
    return cnt
end

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

mutable struct SolverState{M,C,T,R<:Real,W,P}
    method::M
    cache::C                            # LinearSolve LinearCache
    S::W                                # shifted matrix A - σB (dense or
                                        # sparse union-pattern workspace)
    prec::P                             # preconditioner LU (dense/UMFPACK)
                                        # or Nothing for direct-LU methods
    σ_solve::T                          # shift used by the most recent solve
    σ_fac::T                            # shift of the most recent factorization
    normA::R
    inner_maxiter::Int
    counters::RQICounters{T,R}
    stall_next::Bool                    # previous GMRES hit maxiters → refactorize
    v_work::Vector{T}                   # current iterate (normalize! in place)
    Av::Vector{T}                       # A*v, then overwritten as the residual
    Bv::Vector{T}                       # B*v; also the solve right-hand side
    work::Vector{T}                     # column buffer for sparse shift_into!
end

# S = A - σ B.  B = I keeps the cheap diagonal update for standard problems.
shift_into!(S::AbstractMatrix{T}, A, B::UniformScaling, σ::T,
            work = nothing) where {T} =
    (copyto!(S, A); S[diagind(S)] .-= σ; S)
shift_into!(S::AbstractMatrix{T}, A, B::AbstractMatrix, σ::T,
            work = nothing) where {T} =
    (copyto!(S, A); axpy!(-σ, B, S); S)

# Sparse shifts write through a dense column buffer into a union-pattern
# workspace S (pattern(S) must contain pattern(A) ∪ pattern(B)); `work` is
# returned to zero as each column is copied out.
function shift_into!(S::SparseMatrixCSC{Tv}, A, B::UniformScaling,
                     σ::Tv, work) where {Tv}
    fill!(work, zero(Tv))
    rvA, nzA = rowvals(A), nonzeros(A)
    rvS, nzS = rowvals(S), nonzeros(S)
    for j in axes(S, 2)
        for p in nzrange(A, j); work[rvA[p]] += nzA[p]; end
        work[j] -= σ * B.λ
        for q in nzrange(S, j)
            i = rvS[q]; nzS[q] = work[i]; work[i] = zero(Tv)
        end
    end
    return S
end

function shift_into!(S::SparseMatrixCSC{Tv}, A, B::SparseMatrixCSC,
                     σ::Tv, work) where {Tv}
    fill!(work, zero(Tv))
    rvA, nzA = rowvals(A), nonzeros(A)
    rvB, nzB = rowvals(B), nonzeros(B)
    rvS, nzS = rowvals(S), nonzeros(S)
    for j in axes(S, 2)
        for p in nzrange(A, j); work[rvA[p]] += nzA[p]; end
        for p in nzrange(B, j); work[rvB[p]] -= σ * nzB[p]; end
        for q in nzrange(S, j)
            i = rvS[q]; nzS[q] = work[i]; work[i] = zero(Tv)
        end
    end
    return S
end

# Zero-valued sparse matrix whose pattern is the column-merge union of
# pattern(A) ∪ pattern(B): `A - B` is NOT a safe workspace pattern because
# sparse subtraction drops exactly-cancelling entries that a later shift
# might need to write.
function _union_pattern(A::SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC) where {Tv,Ti}
    n, m = size(A)
    I = Ti[]; J = Ti[]
    sizehint!(I, nnz(A) + nnz(B)); sizehint!(J, nnz(A) + nnz(B))
    rvA, rvB = rowvals(A), rowvals(B)
    for j in 1:m
        pa, pb = A.colptr[j], B.colptr[j]
        ea, eb = A.colptr[j+1], B.colptr[j+1]
        while pa < ea || pb < eb
            ra = pa < ea ? rvA[pa] : typemax(Ti)
            rb = pb < eb ? rvB[pb] : typemax(Ti)
            i = min(ra, rb)
            push!(I, i); push!(J, Ti(j))
            ra == i && (pa += 1)
            rb == i && (pb += 1)
        end
    end
    return sparse(I, J, zeros(Tv, length(I)), n, m)
end

# Workspace for A - σB with the union shift pattern baked in. Dense gets
# a plain uninitialized buffer; sparse gets the merged pattern (for B = I
# that means union with the diagonal).
_shift_pattern(A::AbstractMatrix, B) = similar(A)
_shift_pattern(A::SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC) where {Tv,Ti} =
    _union_pattern(A, B)
function _shift_pattern(A::SparseMatrixCSC{Tv,Ti}, B::UniformScaling) where {Tv,Ti}
    n = size(A, 1)
    return _union_pattern(A, sparse(Ti.(1:n), Ti.(1:n), ones(Tv, n), n, n))
end

function _singular_perturb(σ::T, normA::Real, k = 1) where {T<:Number}
    s = sign(σ)
    s == 0 && (s = one(T))
    return T(σ + s * eps(normA) * max(abs(σ), normA) * 8^(k - 1))
end

"""Factorize `A - σB` in place into `F.factors`/`F.ipiv` via `getrf!`,
nudging the shift if the matrix is singular. `F` keeps its identity (so
references held elsewhere, e.g. a preconditioner slot, see the new
contents); returns the possibly perturbed σ."""
function _factorize_shifted!(cnt, F::DenseLU{T}, A, B, σ::T, normA,
                             S = nothing, work = nothing) where {T}
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

# Sparse analog: `lu!(F, S)` reuses F's symbolic factorization (the union
# pattern of S never changes) and updates the numeric factors in place —
# the UMFPACK version of the identity-preserving in-place update above.
function _factorize_shifted!(cnt, F::SuiteSparse.UMFPACK.UmfpackLU, A, B,
                             σ, normA, S, work)
    attempts = 0
    ok = false
    while attempts == 0 || (!ok && attempts < 4)
        shift_into!(S, A, B, σ, work)
        lu!(F, S; check = false)
        ok = issuccess(F)
        attempts += 1
        ok && break
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
function _new_cache(Ashift::AbstractMatrix{T}, b,
                    ::Union{GMRESInputPrec,GMRESPrevPrec}, prec;
                    inner_reltol, inner_abstol,
                    inner_maxiter, gmres_restart) where {T}
    alg = KrylovJL_GMRES(gmres_restart = gmres_restart;
                         precs = (A_, p_) -> (prec, LinearAlgebra.I))
    prob = LinearProblem(Ashift, b)
    return init(prob, alg; reltol = inner_reltol, abstol = inner_abstol,
                maxiters = inner_maxiter, verbose = false)
end

function init_state(method::Union{FixedShiftLU,RefactorLU}, A::AbstractMatrix{T}, B,
                    σ0::T, v::Vector{T}, normA; kwargs...) where {T}
    cnt = RQICounters{T}()
    S = _shift_pattern(A, B)
    work = Vector{T}(undef, size(A, 1))
    shift_into!(S, A, B, σ0, work)
    cache = _new_cache(copy(S), copy(v), method)
    cnt.n_factorizations = 1
    return SolverState(method, cache, S, nothing, σ0, σ0, normA,
                       get(kwargs, :inner_maxiter, 0), cnt, false,
                       similar(v), similar(v), similar(v), work)
end

function init_state(method::Union{GMRESInputPrec,GMRESPrevPrec}, A::AbstractMatrix{T}, B,
                    σ0::T, v::Vector{T}, normA;
                    inner_reltol, inner_abstol, inner_maxiter, gmres_restart) where {T}
    cnt = RQICounters{T}()
    n = size(A, 1)
    S = _shift_pattern(A, B)
    work = Vector{T}(undef, n)
    F = lu(shift_into!(S, A, B, σ0, work); check = false)
    σf = issuccess(F) ? σ0 : _factorize_shifted!(cnt, F, A, B, σ0, normA, S, work)
    cache = _new_cache(copy(S), copy(v), method, F;
                       inner_reltol, inner_abstol,
                       inner_maxiter = inner_maxiter <= 0 ? n : inner_maxiter,
                       gmres_restart)
    cnt.n_factorizations = 1
    return SolverState(method, cache, S, F, σ0, σf, normA,
                       inner_maxiter <= 0 ? n : inner_maxiter, cnt, false,
                       similar(v), similar(v), similar(v), work)
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
        shift_into!(st.S, A, B, σsolve, st.work)
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
        shift_into!(st.S, A, B, σ, st.work)
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
        st.σ_fac = _factorize_shifted!(cnt, st.prec, A, B, σ, st.normA,
                                       st.S, st.work)
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
        shift_into!(st.S, A, B, σ, st.work)
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
        shift_into!(st.S, A, B, σ, st.work)
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
function run_rqi!(st::SolverState{<:Any,<:Any,T}, A::AbstractMatrix{T}, B,
                  v0::Vector{T}, σ0::Number;
                  tol::Real = 1e-10, maxiter::Integer = 50) where {T<:Number}
    normA = st.normA
    v = normalize!(copyto!(st.v_work, v0))
    Av = st.Av
    Bv = st.Bv
    mul!(Bv, B, v)              # RHS of the first shifted solve
    σ = T(σ0)
    cnt = st.counters

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
    if A isa SparseMatrixCSC || B isa SparseMatrixCSC
        # sparse path: keep sparse operands; UMFPACK does the solves
        A = SparseMatrixCSC{T,Int}(A)
        Bm = B isa UniformScaling ? B : SparseMatrixCSC{T,Int}(B)
    else
        A = Matrix{T}(A)
        Bm = B isa UniformScaling ? B : Matrix{T}(B)
    end
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
    run_oracle_hp(A[, B]) -> (; values, vectors = nothing)

High-precision oracle for generic element types (BigFloat, MultiFloats
`Float64xN`, ...). Eigenvalue-only: `eigen!`/`eigvals(A,B)` have no
generic-precision implementation (QZ is LAPACK-only), so a pencil is
reduced to the standard problem `B\\A`, which requires `B` nonsingular.
Returns a NamedTuple stand-in with `vectors = nothing` — downstream
consumers only need `.values`. Requires GenericLinearAlgebra.jl loaded.
"""
function run_oracle_hp(A::AbstractMatrix, B::Union{AbstractMatrix,Nothing}=nothing)
    λ = if B === nothing
        eigvals(A)
    else
        F = lu(B; check = false)
        issuccess(F) || error("run_oracle_hp: singular B — generic-precision " *
            "generalized eigendecomposition (QZ) is unavailable; use " *
            "--oracle arpack with --search-shift or a BlasFloat oracle eltype")
        eigvals(Matrix(B) \ Matrix(A))
    end
    return (values = λ, vectors = nothing)
end

"""
    ShiftInvertOp(F, B, n)

Lazy `AbstractMatrix` for the shift-invert operator `C = (A - σB)⁻¹B`,
where `F` is a factorization of `A - σB`. `eigs` only needs `size` and
`mul!`, so this lets ARPACK run in *standard* mode on a generalized or
shifted problem — crucially, ARPACK's own generalized mode (bmat="G") uses
`B`-inner-products that are degenerate for a singular `B` and returns
garbage; running standard-mode on `C` sidesteps that entirely: the
infinite eigenpairs of the pencil are the kernel of `C` and deflate
automatically (the Krylov space stays in `range(C)`).
"""
struct ShiftInvertOp{Tv,F<:Factorization,BM} <: AbstractMatrix{Tv}
    F::F
    B::BM
    n::Int
    work::Vector{Tv}
end
Base.size(o::ShiftInvertOp) = (o.n, o.n)
LinearAlgebra.issymmetric(::ShiftInvertOp) = false
LinearAlgebra.ishermitian(::ShiftInvertOp) = false
function LinearAlgebra.mul!(y::AbstractVector, o::ShiftInvertOp,
                            x::AbstractVector)
    mul!(o.work, o.B, x)            # work .= B*x
    ldiv!(y, o.F, o.work)           # y .= (A - σB) \ work
    return y
end

"""
    run_oracle_eigs(A[, B]; nev, sigma, v0) -> (Eigen, nconv, niter, nmult)

ARPACK partial oracle via `eigs`:

* `sigma === nothing` (standard problems only): the `nev` eigenvalues with
  the largest real part, plain Arnoldi on `A`.
* `sigma` given: shift-invert mode — eigenvalues nearest `sigma` are found
  as the largest-`|`μ`|` eigenpairs `μ = 1/(λ - σ)` of `C = (A - σB)⁻¹B`
  (see [`ShiftInvertOp`](@ref); mandatory for generalized pencils and the
  only mode that tolerates a singular `B`). Returned pencil eigenvalues
  are `λ = σ + 1/μ`.

Double precision only; `nev` is clamped to `n - 2`. The returned `Eigen`
is restricted to the `nconv ≤ nev` converged pairs so it can stand in for
a dense `eigen`/`eigen(A, B)` result downstream. A deterministic `v0`
(default `ones`) keeps the benchmark reproducible.
"""
function run_oracle_eigs(A::AbstractMatrix, B::Union{AbstractMatrix,Nothing}=nothing;
                         nev::Integer=20, sigma=nothing,
                         v0::Union{AbstractVector,Nothing}=nothing,
                         tol::Real=0.0)
    n = size(A, 1)
    nev = min(nev, n - 2)
    nev >= 1 || error("run_oracle_eigs needs n >= 3, got n=$n")
    if sigma === nothing
        B === nothing || error(
            "run_oracle_eigs: generalized pencils require sigma " *
            "(ARPACK's own bmat=\"G\" mode is unreliable for singular B)")
        kw = (; nev, which = :LR, ritzvec = true,
               v0 = v0 === nothing ? ones(eltype(A), n) : v0)
        λ, X, nconv, niter, nmult, _ = eigs(A; kw...)
        nconv = min(nconv, length(λ))
    else
        To = eltype(A)
        Tσ = promote_type(To, typeof(sigma))
        F = factorize(Tσ.(A) - sigma * (B === nothing ? I : B))
        op = ShiftInvertOp{Tσ,typeof(F),typeof(B === nothing ? I : B)}(
            F, B === nothing ? I : B, n, zeros(Tσ, n))
        kw = (; nev, which = :LM, tol, ritzvec = true,
               v0 = v0 === nothing ? ones(Tσ, n) : v0)
        μ, X, nconv, niter, nmult, _ = eigs(op; kw...)
        nconv = min(nconv, length(μ))
        μc = μ[1:nconv]
        # |μ| ≲ eps·|μ|max ⇔ pencil eigenvalues at ∞ (kernel of B) or
        # unconverged garbage: keep only λ = σ + 1/μ that stay finite
        keep = abs.(μc) .> eps(real(Tσ)) * maximum(abs, μc)
        λ = sigma .+ inv.(μc[keep])
        X = X[:, findall(keep)]
        nconv = length(λ)
    end
    nconv < nev &&
        @warn "eigs converged only $nconv of $nev requested eigenvalues"
    nconv == 0 && error("eigs converged no eigenvalues")
    return Eigen(collect(λ[1:nconv]), X[:, 1:nconv]), nconv, niter, nmult
end

"""
    run_oracle_arnoldi(A[, B]; nev, sigma, v0) -> ((; values, vectors), nconv, niter, nmult)

Generic-precision partial oracle via ArnoldiMethod.jl `partialschur`
(pure-Julia Krylov–Schur): the MultiFloats/BigFloat counterpart of
`run_oracle_eigs`, since ARPACK is a compiled Fortran library and
BlasFloat-only. Same contract:

* `sigma === nothing` (standard problems only): the `nev` eigenvalues of
  `A` with largest real part (`which = :LR`).
* `sigma` given: the `nev` eigenvalues nearest `sigma` as the largest-|μ|
  pairs of `C = (A - σB)⁻¹B` (see [`ShiftInvertOp`](@ref)). The `A - σB`
  factorization is a *dense* generic `lu` — sparse LU is BlasFloat-only —
  but it is paid once; singular `B` is tolerated exactly as in
  `run_oracle_eigs` (ker B ⊂ ker C deflates). Mandatory for pencils.

Eigenvalue-only: `partialeigen` (Schur → eigenvectors) has no
generic-eltype implementation, so the returned stand-in carries
`vectors = nothing`. `niter`/`nmult` both report `History.mvproducts`
(ArnoldiMethod tracks matvecs, not aupd iterations). `nev` is clamped to
`n - 2` to match the ARPACK path's bound.
"""
function run_oracle_arnoldi(A::AbstractMatrix,
                            B::Union{AbstractMatrix,Nothing}=nothing;
                            nev::Integer=20, sigma=nothing,
                            v0::Union{AbstractVector,Nothing}=nothing,
                            tol::Real=0)
    n = size(A, 1)
    nev = min(nev, n - 2)
    nev >= 1 || error("run_oracle_arnoldi needs n >= 3, got n=$n")
    To = eltype(A)
    Tσ = sigma === nothing ? To : promote_type(To, typeof(sigma))
    if sigma === nothing
        B === nothing || error(
            "run_oracle_arnoldi: generalized pencils require sigma " *
            "(shift-invert mode; this also handles singular B)")
        op = A
        which = :LR
    else
        # lu(A - σB) is dense: sparse factorization is BlasFloat-only
        F = lu(Matrix(Tσ.(A) - Tσ(sigma) * (B === nothing ? I : B)))
        op = ShiftInvertOp{Tσ,typeof(F),typeof(B === nothing ? I : B)}(
            F, B === nothing ? I : B, n, zeros(Tσ, n))
        which = :LM
    end
    kw = (; nev, which, v1 = v0 === nothing ? ones(Tσ, n) : v0)
    tol != 0 && (kw = merge(kw, (; tol)))
    ps, hist = partialschur(op; kw...)
    nconv = min(hist.nconverged, length(ps.eigenvalues))
    μ = collect(ps.eigenvalues[1:nconv])
    if sigma !== nothing
        # |μ| ≲ eps·|μ|max ⇔ pencil eigenvalues at ∞ (kernel of B)
        isempty(μ) && error("partialschur converged no eigenvalues")
        keep = abs.(μ) .> eps(real(Tσ)) * maximum(abs, μ)
        λ = Tσ(sigma) .+ inv.(μ[keep])
        nconv = length(λ)
    else
        λ = μ
        nconv == 0 && error("partialschur converged no eigenvalues")
    end
    hist.nconverged < nev &&
        @warn "partialschur converged only $(hist.nconverged) of $nev " *
              "requested eigenvalues"
    return (values = λ, vectors = nothing), nconv, hist.mvproducts,
           hist.mvproducts
end

"""
    load_matrix(path, T = Float64) -> Matrix{T} or SparseMatrixCSC{T}

Read a matrix file and convert to eltype `T`. Delimited text files give
dense matrices; Matrix Market (`.mtx`) coordinate files give
`SparseMatrixCSC`. The text delimiter is sniffed from the first non-blank
line: comma if one appears, else whitespace.
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

_as_eltype(M::SparseMatrixCSC, ::Type{T}) where {T} = SparseMatrixCSC{T,Int}(M)
_as_eltype(M::AbstractMatrix, ::Type{T}) where {T} = Matrix{T}(M)

# Matrix Market coordinate files (.mtx). Supports real/integer/complex/
# pattern fields and general/symmetric/hermitian/skew-symmetric storage;
# symmetric entries are mirrored explicitly. A `2n×n` file is a stacked
# pencil [A; B] by the same convention as the dense text format.
function _read_mtx(path::AbstractString)
    open(path) do io
        header = split(strip(readline(io)))
        (length(header) >= 5 && header[1] == "%%MatrixMarket" &&
         header[2] == "matrix" && header[3] == "coordinate") ||
            error("$path is not a Matrix Market coordinate-format file")
        field = lowercase(header[4]); symmetry = lowercase(header[5])
        function nextline()
            for line in eachline(io)
                s = strip(line)
                isempty(s) || startswith(s, '%') || return s
            end
            error("unexpected EOF in $path")
        end
        dims = split(nextline())
        m, n = parse(Int, dims[1]), parse(Int, dims[2])
        I = Int[]; J = Int[]; V = ComplexF64[]
        for line in eachline(io)
            s = strip(line)
            (isempty(s) || startswith(s, '%')) && continue
            p = split(s)
            i = parse(Int, p[1]); j = parse(Int, p[2])
            v = if field == "complex"
                    parse(Float64,p[3]) + im*parse(Float64,p[4])
                elseif field == "pattern"
                    1.0
                else
                    parse(Float64, p[3])
                end
            push!(I, i); push!(J, j); push!(V, v)
            i == j && continue
            if symmetry == "symmetric" || symmetry == "hermitian"
                push!(I, j); push!(J, i); push!(V, conj(v))
            elseif symmetry == "skew-symmetric"
                push!(I, j); push!(J, i); push!(V, -v)
            elseif symmetry != "general"
                error("unsupported Matrix Market symmetry: $symmetry")
            end
        end
        return sparse(I, J, V, m, n)
    end
end

_read_any(path::AbstractString) =
    endswith(lowercase(path), ".mtx") ? _read_mtx(path) : _read_delim(path)

function load_matrix(path::AbstractString, ::Type{T} = Float64) where {T<:Number}
    M = _read_any(path)
    size(M, 1) == size(M, 2) || error("matrix in $path is not square: $(size(M))")
    return _as_eltype(M, T)
end

"""
    load_problem(path, T = Float64) -> (A, B)

Read a problem file (dense delimited text or sparse Matrix Market `.mtx`).
A square `n×n` file is the standard problem `(A, nothing)`; a `2n×n` file
is a stacked pencil `[A; B]` and returns the two `n×n` blocks.
"""
function load_problem(path::AbstractString, ::Type{T} = Float64) where {T<:Number}
    M = _read_any(path)
    m, n = size(M)
    if m == n
        return _as_eltype(M, T), nothing
    elseif m == 2n
        return _as_eltype(M[1:n, :], T), _as_eltype(M[n+1:2n, :], T)
    end
    error("matrix in $path must be n×n (standard) or 2n×n stacked [A; B] " *
          "(generalized pencil): got $(size(M))")
end

"""Nearest finite oracle eigenpair to `λ`; returns (index, λ_oracle, eigvec).
Non-finite oracle eigenvalues (singular `B`) never match."""
function nearest_oracle(F, λ)
    i = argmin(i -> (d = abs(F.values[i] - λ); isfinite(d) ? d : Inf),
               eachindex(F.values))
    # high-precision oracles (run_oracle_hp) carry no eigenvectors
    u = F.vectors === nothing ? nothing : F.vectors[:, i]
    return i, F.values[i], u
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

# O(nnz) version: single pass over stored entries splitting diagonal
# (disc center) from off-diagonal (radius).
function gershgorin_bounds(A::SparseMatrixCSC{Tv}) where {Tv}
    n = size(A, 1)
    c = zeros(Tv, n)
    r = zeros(real(Tv), n)
    for j in axes(A, 2), p in nzrange(A, j)
        i = rowvals(A)[p]; v = nonzeros(A)[p]
        i == j ? (c[i] = v) : (r[i] += abs(v))
    end
    lo, hi = Inf, -Inf
    for i in 1:n
        lo = min(lo, real(c[i]) - r[i])
        hi = max(hi, real(c[i]) + r[i])
    end
    return lo, hi
end

"""
    build_starts(strategy, A, B = I; seed, nstarts, power_iters, shift, search_shift) -> (starts, info)

Initial `(v0, σ0)` pairs for the pencil `(A, B)`.

- `rq`: a single start at `σ0 = RQ(v0)` (generalized Rayleigh quotient
  when `B ≠ I`). An explicit `shift` overrides `σ0` and forces a single
  start for any strategy.
- `grid`: `nstarts` shifts spread evenly across the Gershgorin real-axis
  range of the standard-form problem `Â = B\\A` (requires nonsingular `B`).
- `largest`: `v0` built from `power_iters` shift-invert power iterations
  on `(A - sB)\\B`, whose dominant eigenpair is the pencil eigenpair
  nearest `s`; `σ0 = RQ(v0)`. `s` defaults to the right Gershgorin bound
  (of `A`, or of `Â` when `B` is nonsingular); a singular `B` requires
  `search_shift`. `search_shift` may be a vector — one start is built per
  shift (start `j` uses `seed + 1000(j-1)`).

Returns the `(v0, σ0)` pairs plus a `Dict{String,Any}` recording the
resolved search parameters (`search_shift` for `largest`, `(lo, hi)` for
`grid`). Takes only concrete values — the CLI Dict is unpacked once at the
call site so the (timed) search phase is type-stable.
"""
function build_starts(strategy::String, A::AbstractMatrix{T}, B = I;
                      seed::Integer, nstarts::Integer, power_iters::Integer,
                      shift::Union{Nothing,T} = nothing,
                      search_shift::Union{Nothing,T,AbstractVector} = nothing) where {T}
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
        return [(v0, T(σ0))], Dict{String,Any}()
    end
    if strategy == "largest"
        # Shift-invert power iteration on C = (A - sB)\B: its dominant
        # eigenvalue μ = 1/(λ - s) corresponds to the pencil eigenvalue
        # nearest s, so s is the targeting knob — at/right of the spectral
        # edge it picks out rightmost-λ. Singular B is fine: the infinite
        # eigenvalues of the pencil lie in the kernel of C and deflate
        # automatically. Iterate until the eigenvalue estimate stagnates
        # (a converged dominant eigenvector) or the cap is hit. If the
        # nearest eigenvalue is a complex pair and T is real, v0 can only
        # reach the invariant 2-plane and the estimate oscillates; a
        # complex --eltype is needed for RQI to converge to a complex λ.
        #
        # `search_shift` may be a vector: each entry gets its own
        # factorization + power iteration and produces one start.
        shifts = if search_shift !== nothing
            search_shift isa AbstractVector ? T.(search_shift) :
                                              [T(search_shift)]
        elseif hasB
            FB = lu(B; check = false)
            issuccess(FB) ||
                error("start-strategy largest on a pencil with singular B " *
                      "requires --search-shift")
            [T(gershgorin_bounds(FB \ A)[2])]
        else
            [T(gershgorin_bounds(A)[2])]
        end
        S = _shift_pattern(A, B)
        work = Vector{T}(undef, n)
        normA = opnorm(A, Inf)
        starts = Tuple{Vector{T},T}[]
        s_used = T[]
        for (j, s0) in enumerate(shifts)
            s = T(s0)
            F = lu(shift_into!(S, A, B, s, work); check = false)
            for k in 1:3
                issuccess(F) && break
                s = _singular_perturb(s, normA, k)
                F = lu(shift_into!(S, A, B, s, work); check = false)
            end
            issuccess(F) ||
                error("largest: could not factorize A - sB near s=$s")
            v0 = rand(Xoshiro(seed + 1000 * (j - 1)), T, n)
            normalize!(v0)
            w = similar(v0)
            Bv = similar(v0)
            scale = max(abs(s), one(real(T)))
            λest = s
            for _ in 1:power_iters
                mul!(Bv, B, v0)
                ldiv!(w, F, Bv)          # w = C·v0
                nw = norm(w)
                nw < eps(real(T)) && break   # v0 in ker C ⊇ infinite eigenpairs
                μ = dot(v0, w)              # RQ of C at v0
                λest_new = s + inv(μ)
                @. v0 = w / nw
                if isfinite(λest_new) && abs(λest_new - λest) <= 1e-8 * scale
                    λest = λest_new
                    break
                end
                λest = λest_new
            end
            # σ0 = generalized Rayleigh quotient of the final vector
            mul!(w, A, v0)
            mul!(Bv, B, v0)
            push!(starts, (v0, T(dot(v0, w) / dot(v0, Bv))))
            push!(s_used, s)
        end
        return starts,
               Dict{String,Any}("search_shift" =>
                                length(s_used) == 1 ? s_used[1] : s_used)
    end
    strategy != "grid" && error("unknown start strategy: $strategy")
    # Shift placement for `grid` acts on the standard-form problem
    # Â = B\A (one dense solve in the timed search phase when B ≠ I).
    Â = if hasB
        FB = lu(B; check = false)
        issuccess(FB) ||
            error("start-strategy grid requires a nonsingular B")
        FB \ A
    else
        A
    end
    lo, hi = gershgorin_bounds(Â)
    K = nstarts
    K < 1 && error("nstarts must be ≥ 1")
    σs = K == 1 ? [(lo + hi) / 2] : collect(range(lo, hi, K))
    return [(normalize!(rand(Xoshiro(seed + 1000j), T, n)), T(σ))
            for (j, σ) in enumerate(σs)],
           Dict{String,Any}("lo" => lo, "hi" => hi)
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
