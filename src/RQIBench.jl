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
       reset_state!, run_oracle, load_matrix, method_tag, nearest_oracle,
       eigenvector_error, gershgorin_bounds, sanity_check

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
    S::Matrix{T}                        # workspace for the shifted matrix A - σI
    prec::Union{Nothing,DenseLU{T}}     # preconditioner; refactored in place
    σ_solve::T                          # shift used by the most recent solve
    σ_fac::T                            # shift of the most recent factorization
    normA::R
    inner_maxiter::Int
    counters::RQICounters{T,R}
    stall_next::Bool                    # previous GMRES hit maxiters → refactorize
    v_work::Vector{T}                   # current iterate (normalize! in place)
    Av::Vector{T}                       # A*v, then overwritten as the residual
end

shift_into!(S::AbstractMatrix{T}, A, σ::T) where {T} =
    (copyto!(S, A); S[diagind(S)] .-= σ; S)

function _singular_perturb(σ::T, normA::Real, k = 1) where {T<:Number}
    s = sign(σ)
    s == 0 && (s = one(T))
    return T(σ + s * eps(normA) * max(abs(σ), normA) * 8^(k - 1))
end

"""Factorize `A - σI` in place into `F.factors`/`F.ipiv` via `getrf!`,
nudging the shift if the matrix is singular. `F` keeps its identity (so
references held elsewhere, e.g. a preconditioner slot, see the new
contents); returns the possibly perturbed σ."""
function _factorize_shifted!(cnt, F::DenseLU{T}, A, σ::T, normA) where {T}
    info = zero(LinearAlgebra.BlasInt)
    attempts = 0
    while attempts == 0 || (info > 0 && attempts < 4)
        shift_into!(F.factors, A, σ)
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

function init_state(method::Union{FixedShiftLU,RefactorLU}, A::Matrix{T}, σ0::T,
                    v::Vector{T}, normA; kwargs...) where {T}
    cnt = RQICounters{T}()
    S = similar(A)
    shift_into!(S, A, σ0)
    cache = _new_cache(copy(S), copy(v), method)
    cnt.n_factorizations = 1
    return SolverState(method, cache, S, nothing, σ0, σ0, normA,
                       get(kwargs, :inner_maxiter, 0), cnt, false,
                       similar(v), similar(v))
end

function init_state(method::Union{GMRESInputPrec,GMRESPrevPrec}, A::Matrix{T},
                    σ0::T, v::Vector{T}, normA;
                    inner_reltol, inner_abstol, inner_maxiter, gmres_restart) where {T}
    cnt = RQICounters{T}()
    n = size(A, 1)
    S = similar(A)
    F = lu(shift_into!(S, A, σ0); check = false)
    σf = issuccess(F) ? σ0 : _factorize_shifted!(cnt, F, A, σ0, normA)
    cache = _new_cache(copy(S), copy(v), method, F;
                       inner_reltol, inner_abstol,
                       inner_maxiter = inner_maxiter <= 0 ? n : inner_maxiter,
                       gmres_restart)
    cnt.n_factorizations = 1
    return SolverState(method, cache, S, F, σ0, σf, normA,
                       inner_maxiter <= 0 ? n : inner_maxiter, cnt, false,
                       similar(v), similar(v))
end

# ---------------------------------------------------------------------------
# Inner solve dispatch
# ---------------------------------------------------------------------------

# --- direct LU -------------------------------------------------------------

# Shared direct-LU solve loop with singular-shift perturbation. `cache.A =`
# assignment marks the cache isfresh → the next solve! refactorizes; no
# reinit! needed.
function _solve_direct!(st, A, σsolve, v)
    cnt = st.counters
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

# FixedShiftLU keeps solving at the (possibly perturbed) initial shift and
# never touches cache.A outside the perturbation path — the init-time
# factorization is reused.
solve_step!(st::SolverState{<:FixedShiftLU}, A, σ, v) =
    _solve_direct!(st, A, st.σ_solve, v)

# RefactorLU refactorizes whenever the solve shift moved.
function solve_step!(st::SolverState{<:RefactorLU}, A, σ, v)
    cnt = st.counters
    if σ != st.σ_solve
        cnt.n_shifts += 1
        shift_into!(st.S, A, σ)
        st.cache.A = st.S
        cnt.n_factorizations += 1
    end
    return _solve_direct!(st, A, σ, v)
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
        st.σ_fac = _factorize_shifted!(cnt, st.prec, A, σ, st.normA)
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
function reset_state!(st::SolverState{<:FixedShiftLU,<:Any,T}, A, σ0) where {T}
    st.counters = RQICounters{T}()
    σ = T(σ0)
    if σ != st.σ_solve
        st.counters.n_shifts += 1
        shift_into!(st.S, A, σ)
        st.cache.A = st.S      # isfresh → refactorizes on next solve
        st.counters.n_factorizations += 1
        st.σ_solve = σ
    end
    return st
end

function reset_state!(st::SolverState{<:Any,<:Any,T}, A, σ0) where {T}
    st.counters = RQICounters{T}()
    st.σ_solve = T(NaN)     # force cache.A refresh on next solve
    return st
end

"""
    run_rqi!(st::SolverState, A, v0, σ0; tol, maxiter) -> RQIResult

The outer eigenpair iteration on a prepared [`init_state`](@ref): solves
for a single eigenpair and stops when `||Av - σv|| ≤ tol·||A||_∞` (or
`maxiter` is hit). Splitting this from `init_state` lets the benchmark time
state/factorization setup separately from the iteration itself.
"""
function run_rqi!(st::SolverState{<:Any,<:Any,T}, A::Matrix{T}, v0::Vector{T},
                  σ0::Number;
                  tol::Real = 1e-10, maxiter::Integer = 50) where {T<:Number}
    normA = st.normA
    v = normalize!(copyto!(st.v_work, v0))
    Av = st.Av
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
        w, inner_iters, ok = solve_step!(st, A, σ, v)
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
        σ = dot(v, Av)
        axpy!(-σ, v, Av)         # Av -= σ·v → Av now holds the residual
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
                 v0::AbstractVector, σ0::Number,
                 tol::Real = 1e-10, maxiter::Integer = 50,
                 inner_reltol::Real = 1e-10, inner_abstol::Real = 0,
                 inner_maxiter::Integer = 0, gmres_restart::Integer = 0) where {T<:Number}
    A = Matrix{T}(A)
    normA = opnorm(A, Inf)
    v = normalize(Vector{T}(v0))
    σ = T(σ0)
    rt = real(T)
    st = init_state(method, A, σ, v, normA;
                    inner_reltol = rt(inner_reltol),
                    inner_abstol = rt(inner_abstol),
                    inner_maxiter = Int(inner_maxiter),
                    gmres_restart = Int(gmres_restart))
    return run_rqi!(st, A, v, σ; tol = tol, maxiter = maxiter)
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
