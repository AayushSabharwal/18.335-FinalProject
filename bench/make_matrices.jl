# Generate deterministic test matrices as whitespace-delimited text files
# with full (round-trip) precision entries.
#
#   julia --project=. bench/make_matrices.jl [outdir]

using LinearAlgebra
using Printf
using Random

posargs = filter(a -> !startswith(a, "--"), ARGS)
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
