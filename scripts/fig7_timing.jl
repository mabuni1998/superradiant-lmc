# Fig. 7 (top row): wall-clock time of a converged steady state versus n0,
# for the direct solvers, below (gamma_P = 1e-4) and above (gamma_P = 2)
# threshold.  The bottom row (error in g2) is computed from the same files.
#
#   julia --project=. scripts/fig7_timing.jl gamma_P=1e-4 [methods=me,pp,lmc,conv]
#       [me_n0=...] [pp_n0=...] [lmc_n0=...] [conv_n0=...] [dim_cap=60000000]
#
# Writes data/fig7_timing_P<gamma_P>.tsv (method, n0, dim, walltime, n, g2).
# Every point is timed once, after a warm-up solve at the first n0 of the
# list (compilation excluded).  A scan stops at the first failure.
#
# The paper's numbers were measured sequentially on one HPC node (Intel Xeon
# Gold 6342, up to 300 GB of memory), one method per job, single-threaded
# BLAS (OPENBLAS_NUM_THREADS=1).  The largest points need that memory: e.g.
# the population--polarization model at n0 = 41 above threshold has 2.4e6
# variables, and the conventional LMC at n0 = 1256 has 4e7 states.
#
# GFRM timings: scripts/gfrm_timing.jl.  PIQS: python/piqs_reference.py.
# Cumulant expansion: cumulant/run_cumulant.jl.

include("common.jl")

gamma_P = option("gamma_P", P_BELOW)
above = gamma_P > 0.1
methods = option("methods", ["me", "pp", "lmc", "conv"])
log25(points...) = vcat(1:20, collect(points))
me_n0 = option("me_n0", above ? collect(1:24) : collect(1:30))
pp_n0 = option("pp_n0", above ? log25(25, 28, 32, 36, 41) :
                                log25(25, 29, 33, 38, 44, 51, 59, 68, 78, 90, 104))
lmc_n0 = option("lmc_n0", above ? log25(25, 27, 29, 32, 34, 37, 40, 44, 47, 51, 55, 60) :
                                  log25(25, 29, 33, 38, 44, 51, 59, 68, 78, 90, 104, 120, 300, 500))
conv_n0 = option("conv_n0", above ? log25(25, 39, 60, 92, 143, 220, 340, 526, 813, 1256) :
                                    log25(25, 35, 49, 68, 96, 134, 187, 261, 366, 511, 715, 1000))
dim_cap = option("dim_cap", 60_000_000)

solvers = Dict(
    "lmc" => (p -> (r = solve_dicke_lmc(p); (dim=r.nstates, n=r.n, g2=r.g2)), lmc_n0),
    "conv" => (p -> (r = solve_conventional_lmc(p); (dim=r.nstates, n=r.n, g2=r.g2)), conv_n0),
    "pp" => (p -> solve_population_polarization(p; dim_cap=dim_cap), pp_n0),
    "me" => (p -> solve_exact_me(p; dim_cap=dim_cap), me_n0),
)

rows = []
for key in methods
    solve, n0s = solvers[key]
    solve(LaserParams(first(n0s); gamma_P=gamma_P))            # warm-up (compilation)
    for n0 in n0s
        GC.gc()
        local r
        t = @elapsed r = attempt(() -> solve(LaserParams(n0; gamma_P=gamma_P)), "$key n0=$n0")
        r === nothing && break
        push!(rows, (key, n0, r.dim, t, r.n, r.g2))
        @printf("%-4s n0=%-6d dim=%-10d t=%.4g s  g2=%.8g\n", key, n0, r.dim, t, r.g2)
        flush(stdout)
    end
end

merge_method_rows(data_path("fig7_timing_P$(gamma_P).tsv"),
                  "wall-clock time (s) of converged steady states, gamma_P=$gamma_P, gamma_A=gamma_D=0",
                  ["method", "n0", "dim", "walltime", "n", "g2"], methods, rows)
