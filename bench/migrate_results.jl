# One-off migration: rewrite an old-format results file in the current
# output format —
#   * drop `null` entries from oracle.eigenvalues (non-finite eigenvalues
#     from singular-B pencils),
#   * renumber coverage.*.oracle_indices_found to positions in the
#     filtered (serialized) array,
#   * drop oracle.eigenvectors and per-start eigenvector/eigenvector_error.
#
#   julia --project=. bench/migrate_results.jl <in.json> [out.json]
#
# With no out.json the file is rewritten in place.

using JSON

length(ARGS) in (1, 2) ||
    error("usage: migrate_results.jl <in.json> [out.json]")
inpath = ARGS[1]
outpath = length(ARGS) == 2 ? ARGS[2] : inpath

isfinite_scalar(v::Number) = isfinite(v)
isfinite_scalar(v::AbstractDict) =
    get(v, "re", nothing) isa Number && get(v, "im", nothing) isa Number &&
    isfinite(v["re"]) && isfinite(v["im"])
isfinite_scalar(::Any) = false

d = JSON.parsefile(inpath)

function migrate!(d)
    n_dropped = 0
    for entry in get(d, "matrices", Any[])
        oracle = get(entry, "oracle", nothing)
        oracle === nothing && continue
        delete!(oracle, "eigenvectors")
        evs = get(oracle, "eigenvalues", Any[])
        keep = isfinite_scalar.(evs)
        n_dropped += count(!, keep)
        # old index -> new index in the filtered array
        remap = Dict{Int,Int}()
        k = 0
        for (i, ok) in enumerate(keep)
            ok && (remap[i] = (k += 1))
        end
        oracle["eigenvalues"] = evs[keep]

        for cov in values(get(entry, "coverage", Dict{String,Any}()))
            cov isa AbstractDict || continue
            cov["oracle_indices_found"] =
                [remap[i] for i in get(cov, "oracle_indices_found", Int[])
                 if haskey(remap, i)]
        end

        for run in get(entry, "runs", Any[])
            run isa AbstractDict || continue
            for st in get(run, "starts", Any[])
                st isa AbstractDict || continue
                delete!(st, "eigenvector")
                delete!(st, "eigenvector_error")
            end
        end
    end
    return n_dropped
end

n_dropped = migrate!(d)
tmp = outpath * ".tmp"
open(tmp, "w") do io
    JSON.print(io, d, 4)
end
mv(tmp, outpath; force = true)
println(stderr, "wrote $outpath (dropped $n_dropped non-finite eigenvalues)")
