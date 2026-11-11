# Helpers shared by the figure drivers: `key=value` command-line options,
# emitter-number lists, and merging rows into an existing TSV file.

using SuperradiantLMC
using Printf

"""
    option(name, default)

Value of the command-line option `name=value`, parsed to the type of
`default`.  Vectors are given comma separated, e.g. `n0=200,500,1000`.
"""
function option(name::AbstractString, default)
    for a in ARGS
        startswith(a, name * "=") || continue
        value = a[length(name) + 2:end]
        default isa AbstractVector && return parse.(eltype(default), split(value, ','))
        default isa AbstractString && return value
        return parse(typeof(default), value)
    end
    return default
end

"""
    emitter_list(n0_max)

1, 2, ..., 20 followed by ten log-spaced even and ten log-spaced odd emitter
numbers up to `n0_max` (both parities are needed, since g2 alternates
between even and odd n0).
"""
function emitter_list(n0_max::Integer)
    small = collect(1:min(20, n0_max))
    n0_max <= 20 && return small
    function log_spaced(lo, hi, parity)
        values = Int[]
        for r in exp10.(range(log10(lo), log10(hi), length=10))
            v = round(Int, r)
            v % 2 == parity || (v = v < hi ? v + 1 : v - 1)
            (isempty(values) || v > values[end]) && push!(values, v)
        end
        return values
    end
    hi_even = iseven(n0_max) ? n0_max : n0_max - 1
    hi_odd = isodd(n0_max) ? n0_max : n0_max - 1
    large = sort(unique(vcat(log_spaced(26, hi_even, 0), log_spaced(25, hi_odd, 1))))
    return vcat(small, large)
end

"Read a TSV written by `write_tsv`: returns (comment, columns, rows of strings)."
function read_tsv(path)
    lines = readlines(path)
    comment = startswith(lines[1], "#") ? lines[1][3:end] : ""
    body = filter(l -> !startswith(l, "#") && !isempty(strip(l)), lines)
    columns = split(body[1], '\t')
    rows = [Tuple(split(l, '\t')) for l in body[2:end]]
    return comment, columns, rows
end

"""
    merge_method_rows(path, comment, columns, methods, new_rows)

Write `new_rows` to the long-format file `path` (first column = method),
keeping the rows of all methods not in `methods` that are already there.
This lets every method of a figure be (re)computed in a separate run.
"""
function merge_method_rows(path, comment, columns, methods, new_rows)
    kept = []
    if isfile(path)
        _, old_columns, old_rows = read_tsv(path)
        old_columns == columns || error("column mismatch in $path")
        kept = [r for r in old_rows if !(r[1] in methods)]
    end
    write_tsv(path, comment, columns, vcat(kept, new_rows))
end

"Run `f()`; return `nothing` (and print the error) if it throws, e.g. for a cutoff or memory failure."
function attempt(f, label)
    try
        return f()
    catch err
        err isa InterruptException && rethrow()
        println("  $label failed: ", sprint(showerror, err)[1:min(end, 200)])
        return nothing
    end
end

value_or_nan(result, field) = result === nothing ? NaN : getfield(result, field)
