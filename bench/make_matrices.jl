# Generate deterministic test matrices as whitespace-delimited text files
# with full (round-trip) precision entries.
#
#   julia --project=. bench/make_matrices.jl [outdir]
#
# Or convert a FreeFEM pencil dump into a stacked Matrix Market file:
#
#   julia --project=. bench/make_matrices.jl fem <fem_dir> <out.mtx>
#
# where fem_dir contains J.txt / M.txt (FreeFEM `ofstream << matrix` COO
# triples, 0-based) and optionally bnd.txt (per-DOF boundary indicator —
# flagged DOFs are eliminated, giving homogeneous Dirichlet rows/cols).

using LinearAlgebra
using Printf
using Random
using SparseArrays

posargs = filter(a -> !startswith(a, "--"), ARGS)

if !isempty(posargs) && posargs[1] == "fem"
    length(posargs) == 3 || error("usage: make_matrices.jl fem <fem_dir> <out.mtx>")
    dir, out = posargs[2], posargs[3]

    readcoo(f) = begin
        lines = filter(l -> !startswith(l, '#'), readlines(f))
        n = parse(Int, split(lines[1])[1])
        I = Int[]; J = Int[]; V = Float64[]
        for l in lines[2:end]
            p = split(l); length(p) < 3 && continue
            push!(I, parse(Int, p[1]) + 1)
            push!(J, parse(Int, p[2]) + 1)
            push!(V, parse(Float64, p[3]))
        end
        sparse(I, J, V, n, n)
    end

    J = readcoo(joinpath(dir, "J.txt"))
    M = readcoo(joinpath(dir, "M.txt"))
    n = size(J, 1)

    keep = if isfile(joinpath(dir, "bnd.txt"))
        bnd = [parse(Float64, l) for l in readlines(joinpath(dir, "bnd.txt"))
               if !isempty(strip(l))]
        findall(==(0.0), bnd)
    else
        collect(1:n)
    end
    println(stderr, "eliminating $(n - length(keep)) boundary DOFs ($n -> $(length(keep)))")
    J = J[keep, keep]; M = M[keep, keep]
    n = length(keep)

    open(out, "w") do io
        nnzJ, nnzM = nnz(J), nnz(M)
        println(io, "%%MatrixMarket matrix coordinate real general")
        println(io, "% stacked pencil [A; B] from $dir (boundary DOFs eliminated)")
        println(io, "$(2n) $n $(nnzJ + nnzM)")
        IJ, JJ, VJ = findnz(J)
        IM, JM, VM = findnz(M)
        for k in eachindex(IJ)
            @printf(io, "%d %d %.17g\n", IJ[k], JJ[k], VJ[k])
        end
        for k in eachindex(IM)
            @printf(io, "%d %d %.17g\n", IM[k] + n, JM[k], VM[k])
        end
    end
    println(stderr, "wrote $out (n=$n, nnz=$(nnz(J)+nnz(M)))")
    exit(0)
end

outdir = length(posargs) >= 1 ? posargs[1] : joinpath(@__DIR__, "..", "matrices")
mkpath(outdir)

function write_matrix(path, A)
    open(path, "w") do io
        for i in axes(A, 1)
            println(io, join((@sprintf("%.17e", A[i, j]) for j in axes(A, 2)), " "))
        end
    end
    println(stderr, "wrote $path")
end

rng = MersenneTwister(2024)

# 1. Random symmetric matrix.
let n = 20
    B = randn(rng, n, n)
    write_matrix(joinpath(outdir, "symmetric_random.txt"), (B + B') / 2)
end

# 2. Symmetric matrix with a cluster of eigenvalues near 1 plus outliers.
let n = 25
    Q, _ = qr(randn(rng, n, n))
    λ = [1.0 .+ 1e-3 .* (1:20); collect(range(5.0, 20.0, 5))]
    write_matrix(joinpath(outdir, "clustered.txt"), Matrix(Q) * Diagonal(λ) * Matrix(Q)')
end

# 3. Hilbert matrix (very ill conditioned, eigenvalues clustered near 0).
let n = 12
    write_matrix(joinpath(outdir, "hilbert.txt"),
                 [1.0 / (i + j - 1) for i in 1:n, j in 1:n])
end

# 4. Nonsymmetric matrix with real spectrum (diagonalizable but non-normal).
let n = 15
    X = randn(rng, n, n) + I
    λ = collect(range(1.0, 3.0, n))
    write_matrix(joinpath(outdir, "nonsymmetric.txt"), X * Diagonal(λ) * inv(X))
end
