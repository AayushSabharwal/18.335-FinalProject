# Standalone sampling-profile breakdown per method.
#
#   julia --project=. bench/profile.jl [n] [matrixfile]
#
# Runs each method's run_rqi on a loop for a few seconds under Profile's
# statistical sampler, then attributes every sampled backtrace to a cost
# bucket by walking frames leaf->root for the first recognizable Julia
# wrapper (BLAS/LAPACK kernel symbols are the leaves; the wrapper above
# them says which operation they belong to). No source instrumentation.

include(joinpath(@__DIR__, "..", "src", "RQIBench.jl"))
using .RQIBench
using LinearAlgebra, Random, Profile, Printf

const StackFrame = Base.StackTraces.StackFrame

const CATS = [:factorize, :prec_apply, :lu_solve, :blas_gemm, :blas_matvec,
              :gmres_inner, :init_overhead, :shift_matrix, :alloc_copy,
              :vector_math, :linearsolve, :rqi_outer, :gc_alloc,
              :runtime_sys, :eigen_lapack, :blas_other, :idle, :other]

# (regex-on-func-name, category) checked in order; file-based rules handled
# in frame_cat. First matching frame on the leaf->root walk wins; pure
# BLAS/LAPACK kernel names only mark the fallback :blas_leaf bucket.
const RULES = [
    (r"psynch|cvwait|semaphore_wait|__ulock|nanosleep|usleep|swtch|mach_msg", :idle),
    (r"gc_|maybe_collect|jl_alloc|alloc_generic|sweep|mark_obj|mark_outrefs|claim_and_push|heap_snapshot|try_setmark|pool_alloc|jl_gc|malloc", :gc_alloc),
    (r"getrf|getf2|getc2|factorize|(^|#)lu(!|#|$)", :factorize),
    (r"getrs|trtrs|latrs|ldiv|lusolve", :lu_solve),
    (r"init_state|_new_cache|LinearProblem|init_cacheval|(^|#)init(!|#|$)", :init_overhead),
    (r"shift_into", :shift_matrix),
    (r"copyto|(^|#)copy(!|#|$)|similar|resize|sizehint|GenericMemory|memory", :alloc_copy),
    (r"mul|matmul|(^|#)dot|norm|axpy|normalize|map|reduce|sum|maximum|isfinite|diagind|(^|#)all(!|#|$)", :vector_math),
    (r"solve!|SciML", :linearsolve),
    (r"run_rqi|solve_step|_solve_direct|_maybe_refactor|reset_state|run_oracle|eigen|sanity|workload", :rqi_outer),
    (r"kevent|pthread|ijl_task|(^|_)wait|readnb|pump|jl_apply|start_task|safepoint|sleep|uv_|kqueue|lock|unlock|signal|yield|schedule|jl_|exec_blas|ws_queue|basic_thread", :runtime_sys),
    (r"geev|hseqr|gehrd|orghr|trevc|hsein|gebal|steqr|sytrd|ormtr|trexc|balanc|gesdd|gebrd|gehd2|lascl|lange|labad|larfg|lartg|laset|lassq|dlapy|dlar|dhqr|dors|dorgh|dtrev|dhseq", :eigen_lapack),
]

const BLAS_LEAF = r"kernel|_k_|dgemm|gemv|trsm|laswp|iamax|ddot|scal|axpy|nrm2|exec_blas|dot_compute|ger_|gemm|memmove|memset|larf|gemv_|copy_"

"""Map a stack frame to a category, or :other/:blas_leaf (non-decisive)."""
function frame_cat(sf)
    fn, f = String(sf.func), String(sf.file)
    occursin("Krylov", f) && return :gmres_inner
    for (re, c) in RULES
        occursin(re, fn) && return c
    end
    occursin("LinearSolve", f) && return :linearsolve
    occursin(BLAS_LEAF, fn) && return :blas_leaf
    return :other
end

# OpenBLAS pool threads produce stacks with no Julia context — the kernel
# symbol is the only signal. Classify by name: laswp/iamax only happen
# inside getrf (pivoting), trsm/trtrs inside solves, gemv = matvec.
function blas_cat(frames)
    for sf in frames
        fn = String(sf.func)
        occursin(r"laswp|iamax|getrf|getf2|larft|larfb", fn) && return :factorize
        occursin(r"trsm|trtrs|latrs|getrs|trsv|tpsv", fn)  && return :lu_solve
        occursin(r"geev|hseqr|gehrd|orghr|trevc|hsein|gebal|steqr|trexc|balanc|sytrd|ormtr|gesdd|gebrd|dhqr|dlahqr", fn) && return :eigen_lapack
        occursin(r"gemv|matvec", fn)                            && return :blas_matvec
        occursin(r"dot|axpy|scal|nrm2|asum", fn)                && return :vector_math
        occursin(r"gemm|dgemm", fn)                             && return :blas_gemm
    end
    return :blas_other
end

"""Attribute one backtrace to a single bucket."""
function stack_cat(frames)
    has_krylov = false
    saw_blas = false
    cat = :other
    for sf in frames
        c = frame_cat(sf)
        c === :gmres_inner && (has_krylov = true)
        c === :blas_leaf && (saw_blas = true)
        if cat === :other && c !== :other && c !== :blas_leaf
            cat = c
        end
    end
    if cat === :other && saw_blas
        cat = blas_cat(frames)
    elseif cat === :lu_solve && has_krylov
        cat = :prec_apply
    end
    return cat
end

"""Profile `reps` calls to `thunk()`; return (counts, n_stacks, wall_s)."""
function profile_loop(thunk, reps)
    Profile.clear()
    t = @elapsed Profile.@profile for _ in 1:reps
        thunk()
    end
    data, lidict = Profile.retrieve()
    counts = Dict(c => 0 for c in CATS)
    total = 0
    frames = StackFrame[]
    function flush!()
        isempty(frames) && return
        total += 1
        counts[stack_cat(frames)] += 1
        empty!(frames)
    end
    for ip in data
        if ip == 0
            flush!()
        else
            sfs = get(lidict, ip, nothing)   # an ip may be several inlined frames
            sfs !== nothing && append!(frames, sfs)
        end
    end
    flush!()
    return counts, total, t
end

function report(tag, counts, total, t, reps)
    work = total - counts[:idle]          # exclude parked BLAS pool threads
    @printf("\n== %s  (reps=%d, wall=%.3fs, %d work samples [%d idle], %.3g ms/run)\n",
            tag, reps, t, work, counts[:idle], 1000t / reps)
    for c in CATS
        c === :idle && continue
        counts[c] == 0 && continue
        @printf("   %-15s %5.1f%%  (%.0f µs/run)\n", c,
                100 * counts[c] / work, 1e6 * t * counts[c] / work / reps)
    end
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 300
    rng = MersenneTwister(7)
    A = if length(ARGS) >= 2
        load_matrix(ARGS[2])
    else
        M = randn(rng, n, n); (M + M') / 2       # symmetric random
    end
    n = size(A, 1)
    v0 = normalize(rand(rng, n))
    σ0 = dot(v0, A * v0)
    println("n=$n, σ0=$σ0 (symmetric=$(issymmetric(A)))")

    methods = Any[FixedShiftLU(), RefactorLU(), GMRESInputPrec(),
                  GMRESPrevPrec(0.0), GMRESPrevPrec(0.1)]
    for m in methods
        tag = method_tag(m) * (m isa GMRESPrevPrec ? " (tau=$(m.tau))" : "")
        r = run_rqi(m, A; v0 = v0, σ0 = σ0)          # warmup + measure
        t1 = @elapsed run_rqi(m, A; v0 = v0, σ0 = σ0)
        reps = clamp(ceil(Int, 2.5 / max(t1, 1e-4)), 3, 20000)
        counts, total, t = profile_loop(() -> run_rqi(m, A; v0 = v0, σ0 = σ0),
                                        reps)
        report("$tag [conv=$(r.converged) iters=$(r.counters.outer_iters) " *
               "facs=$(r.counters.n_factorizations) " *
               "gmres=$(r.counters.gmres_iters_total)]",
               counts, total, t, reps)
    end

    run_oracle(A)
    t1 = @elapsed run_oracle(A)
    reps = clamp(ceil(Int, 1.5 / max(t1, 1e-4)), 3, 500)
    counts, total, t = profile_loop(() -> run_oracle(A), reps)
    report("oracle_eig", counts, total, t, reps)
end

main()
