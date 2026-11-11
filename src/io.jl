# Plain tab-separated output shared by all drivers.  Every data file starts
# with one '#' comment line describing its content, followed by a header row.

"Absolute path of a file in the repository's `data/` folder."
data_path(name...) = joinpath(dirname(@__DIR__), "data", name...)

format_cell(x::AbstractFloat) = isnan(x) ? "NaN" : @sprintf("%.10g", x)
format_cell(x) = string(x)

"""
    write_tsv(path, comment, columns, rows)

Write `rows` (an iterable of tuples) as a TSV file with a `# comment` line and
a header row.
"""
function write_tsv(path, comment, columns, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# ", comment)
        println(io, join(columns, '\t'))
        for r in rows
            println(io, join(map(format_cell, r), '\t'))
        end
    end
    println("wrote ", path)
    return path
end
