# Convert FreeFEM savemtx dumps (bench/stability_*.edp output) into benchmark
# problem files for run_benchmark.jl.
#
# The FreeFEM scripts emit the generalized pencil  J x = λ M x  on the full
# DOF set plus a boundary-DOF indicator. Two output modes:
#
#   pencil (default)  - eliminate flagged (Dirichlet) velocity DOFs and write
#                       the stacked 2n x n pencil [J; M] as a Matrix Market
#                       (.mtx) sparse file; this is what the generalized RQI
#                       benchmark consumes.
#   shiftinvert       - additionally form the dense standard-EVP matrix
#                       A = (J - σM)\M  (μ = 1/(λ-σ) ⟺ λ = σ + 1/μ) and
#                       write it as whitespace-delimited dense text.
#
#   julia --project=. bench/make_fem_matrices.jl \
#       --in TestEnv/fem/cavity_N16_Re400 [--mode pencil] [--out matrices/x.mtx]
#
# Callable from other scripts: include this file and use reduced_pencil,
# write_pencil_mtx.

using ArgParse
using DelimitedFiles
using LinearAlgebra
using Printf
using SparseArrays

include(joinpath(@__DIR__, "..", "src", "RQIBench.jl"))
using .RQIBench

function parse_cli()
    s = ArgParseSettings(description = "FreeFEM pencil -> benchmark problem file (.mtx pencil or dense shift-invert).")
    @add_arg_table! s begin
        "--in", "-i"
            help = "Directory with J.mtx, M.mtx, bnd.txt (and optionally info.txt, U.txt)"
            arg_type = String
            required = true
        "--out", "-o"
            help = "Output path (default: matrices/fem_<dirname>.mtx for --mode=pencil)"
            arg_type = String
            default = ""
        "--mode"
            help = "pencil: sparse stacked [J; M] .mtx; shiftinvert: dense (J-σM)\\M .txt"
            arg_type = String
            default = "pencil"
            range_tester = x -> x in ("pencil", "shiftinvert")
        "--sigma"
            help = "Shift σ for --mode=shiftinvert (NaN → auto, right of spectrum)"
            arg_type = Float64
            default = NaN
        "--sigma-margin"
            help = "Auto-shift margin as a fraction of the top-10 finite real-part spread"
            arg_type = Float64
            default = 0.05
        "--bnd-tol"
            help = "Threshold on the boundary indicator for DOF elimination"
            arg_type = Float64
            default = 1e-14
        "--no-verify"
            help = "Skip the dense generalized-eigenvalue validation pass"
            action = :store_true
    end
    return parse_args(s)
end

"""Read a FreeFEM `savemtx` Matrix Market dump into a SparseMatrixCSC."""
read_fem_matrix(path) = RQIBench._read_mtx(path)

read_vector(path) = vec(readdlm(path, Float64, comments = true))

"""Parse `key value` pairs and trailing eigenvalue lines from info.txt."""
function read_info(path)
    d = Dict{String,String}()
    λff = ComplexF64[]
    isfile(path) || return d, λff
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        f = split(s)
        if length(f) >= 2 &&
           (re = tryparse(Float64, f[1])) !== nothing &&
           (im = tryparse(Float64, f[2])) !== nothing &&
           !(f[1] in ("Re", "U0", "epsP", "ndof", "sigma_ref", "N", "L"))
            push!(λff, ComplexF64(re, im))
        elseif length(f) >= 2
            d[f[1]] = f[2]
        end
    end
    return d, λff
end

"""Reduced pencil (Jr, Mr) after eliminating flagged Dirichlet DOFs."""
function reduced_pencil(indir; bnd_tol = 1e-14)
    J = read_fem_matrix(joinpath(indir, "J.mtx"))
    M = read_fem_matrix(joinpath(indir, "M.mtx"))
    bnd = read_vector(joinpath(indir, "bnd.txt"))
    n = size(J, 1)
    size(J) == (n, n) && size(M) == (n, n) || error("pencil not square")
    length(bnd) == n || error("bnd.txt length $(length(bnd)) != matrix size $n")
    keep = findall(<=(bnd_tol), abs.(bnd))
    Jr, Mr = J[keep, keep], M[keep, keep]
    @printf(stderr, "pencil: n=%d nnz(J)=%d nnz(M)=%d; eliminated %d Dirichlet DOFs -> n=%d\n",
            n, nnz(J), nnz(M), n - length(keep), length(keep))
    return Jr, Mr, n
end

"""Finite part of a generalized spectrum, sorted by decreasing real part."""
function finite_spectrum(F)
    λ = filter(isfinite, F.values)
    return sort(λ, by = real, rev = true)
end

"""Sparse LU validate: rightmost finite eigenvalues of the reduced pencil."""
function reference_spectrum(Jr, Mr; quiet = false)
    F = eigen(Matrix(Jr), Matrix(Mr))
    λ = finite_spectrum(F)
    quiet || @printf(stderr, "finite eigenvalues of reduced pencil: %d / %d\n",
                     length(λ), size(Jr, 1))
    return λ
end

"""Write a sparse matrix in Matrix Market coordinate format (1-based)."""
function write_mtx(io::IO, S::SparseMatrixCSC, comments::Vector{<:AbstractString})
    I, J, V = findnz(S)
    println(io, "%%MatrixMarket matrix coordinate real general")
    for c in comments
        println(io, "% ", c)
    end
    println(io, size(S, 1), " ", size(S, 2), " ", nnz(S))
    for k in eachindex(I)
        println(io, I[k], " ", J[k], " ", @sprintf("%.17e", V[k]))
    end
    return nothing
end

"""Write the reduced pencil as a stacked [J; M] Matrix Market file."""
function write_pencil_mtx(indir::AbstractString, outpath::AbstractString;
                          bnd_tol = 1e-14)
    Jr, Mr, n = reduced_pencil(indir; bnd_tol)
    info, λff = read_info(joinpath(indir, "info.txt"))
    S = [Jr; Mr]
    dropzeros!(S)
    mkpath(dirname(abspath(outpath)))
    comments = [
        "stacked pencil [A; B] from $indir (boundary DOFs eliminated)",
        "Re=$(get(info, "Re", "?")) epsP=$(get(info, "epsP", "?")) " *
        "ndof_full=$n ndof_reduced=$(size(Jr, 1))",
        "eigenproblem J x = lambda M x; ARPACK eigenvalues nearest " *
        "sigma_ref=$(get(info, "sigma_ref", "?")) (re im):",
    ]
    for l in λff[1:min(8, end)]
        push!(comments, "  $l")
    end
    open(outpath, "w") do io
        write_mtx(io, S, comments)
    end
    println(stderr, "wrote ", outpath)
    return Jr, Mr
end

function main()
    args = parse_cli()
    indir = rstrip(args["in"], '/')
    out = args["out"]

    if args["mode"] == "pencil"
        isempty(out) &&
            (out = joinpath("matrices", "fem_" * basename(indir) * ".mtx"))
        Jr, Mr = write_pencil_mtx(indir, out; bnd_tol = args["bnd-tol"])
        if !args["no-verify"]
            λ = reference_spectrum(Jr, Mr)
            println(stderr, "  rightmost: ", λ[1:min(5, end)])
        end
        return
    end

    # --mode shiftinvert: dense A = (J - σM)\M
    Jr, Mr, n = reduced_pencil(indir; bnd_tol = args["bnd-tol"])
    info, λff = read_info(joinpath(indir, "info.txt"))

    λref = ComplexF64[]
    if !args["no-verify"]
        λref = reference_spectrum(Jr, Mr)
        println(stderr, "  rightmost: ", λref[1:min(5, end)])
    end

    σ = args["sigma"]
    if isnan(σ)
        isempty(λref) && error("need --sigma when --no-verify is set")
        ktop = min(10, length(λref))
        spread = real(λref[1]) - real(λref[ktop])
        σ = real(λref[1]) + max(args["sigma-margin"] * spread, 1e-3)
    end
    println(stderr, "shift σ = ", σ)

    F = lu(Jr - σ * Mr)
    A = Matrix(F \ Matrix(Mr))

    isempty(out) &&
        (out = joinpath("matrices", "fem_" * basename(indir) * "_si.txt"))
    mkpath(dirname(abspath(out)))

    μ = eigvals(A)
    λhat = sort(σ .+ inv.(filter(!iszero, μ)), by = real, rev = true)
    println(stderr, "rightmost mapped-back eigenvalues of A: ", λhat[1:min(5, end)])
    if !isempty(λref)
        k = min(5, length(λref))
        err = [minimum(abs, λhat[i] .- λref) for i in 1:k]
        println(stderr, "top-$k match vs reduced pencil: ", err)
    end

    open(out, "w") do io
        println(io, "# shift-invert matrix A = (J - σM)⁻¹M from FreeFEM pencil")
        println(io, "# source_dir=", indir,
                " Re=", get(info, "Re", "?"),
                " epsP=", get(info, "epsP", "?"),
                " ndof_full=", n, " ndof_reduced=", size(Jr, 1))
        println(io, "# sigma=", σ, " ; lambda = sigma + 1/mu")
        for l in λref[1:min(8, end)]
            println(io, "#   ", l)
        end
        for row in eachrow(A)
            println(io, join((@sprintf("%.17e", x) for x in row), " "))
        end
    end
    println(stderr, "wrote ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
