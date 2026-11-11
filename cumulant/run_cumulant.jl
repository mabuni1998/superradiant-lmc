# Cumulant-expansion results for every figure of the paper.
#
#   julia --project=cumulant cumulant/run_cumulant.jl [order=4] [modes...]
#
# modes (default: pump emitters dephasing g2tau timing):
#   orders     Fig. 2   test case (n0=10, gamma_P=1e-4) at orders M = 4..8
#                       (orders=4,5,6 by default; M=7 takes ~30 min and M=8
#                       ~15 h to derive, single core)
#   pump       Fig. 3   n0=10, the 50 pump rates of the sweep
#   emitters   Fig. 4   gamma_P=1e-4, n0 = 1..34 and up to 1e4
#   dephasing  Fig. 5   n0=10, gamma_P=1e-4, the gamma_D grid of the sweep
#   g2tau      Fig. 6   n0=10, gamma_P=1e-4, on the tau grid of data/fig6_g2_tau_n010.tsv
#   timing     Fig. 7   wall time per steady state (median of 3), gamma_P=1e-4 and 2
#
# The symbolic derivation (~30 s at M=4, plus compilation) is cached in
# data/cumulant_cache/ and reused for every parameter point.  Output columns:
# `status` 0 = certified fixed point, 2 = <a'a'aa> < 0, 4 = none found;
# `growth` = largest real part of the Jacobian spectrum (> 0: unstable fixed
# point, found by Newton instead of time integration); `route` = 0
# integration, 1 Newton from the trajectory, 2 homotopy + Newton.  Wall times
# exclude the one-off derivation and compilation.

using Printf

include(joinpath(@__DIR__, "cumulant_solver.jl"))

const DATA_DIR = normpath(joinpath(@__DIR__, "..", "data"))
const CACHE_DIR = joinpath(DATA_DIR, "cumulant_cache")
const G, KAPPA = 0.1, 0.04                                   # g, gamma_c (1/ps)
const PUMPS = exp10.(range(-4, log10(20), length=50))
const DEPHASING = vcat(0.0, KAPPA .* exp10.(range(log10(0.01), log10(30.0), length=16)))
const COLS = ["x", "n", "g2", "status", "residual", "growth", "walltime", "route"]

function write_tsv(path, comment, columns, rows)
    open(path, "w") do io
        println(io, "# ", comment)
        println(io, join(columns, '\t'))
        for r in rows
            println(io, join([x isa AbstractFloat ? @sprintf("%.10g", x) : string(x) for x in r], '\t'))
        end
    end
    println("wrote ", path)
end

function point_row(cs, x, ss)
    growth = ss.status == 4 ? NaN : max_growth_rate(cs, ss.p, ss.u, ss.active)
    @printf("  x=%-10.4g n=%-12.6g g2=%-10.5g status=%d route=%d growth=%.2e  %.3f s\n",
            x, ss.n, ss.g2, ss.status, ss.route, growth, ss.walltime)
    flush(stdout)
    return (x, ss.n, ss.g2, ss.status, ss.residual, growth, ss.walltime, ss.route)
end

median3(v) = sort(v)[(length(v) + 1) ÷ 2]

option(name, default) = (a = findfirst(s -> startswith(s, name * "="), ARGS);
                         a === nothing ? default : parse.(Int, split(ARGS[a][length(name) + 2:end], ',')))

function main()
    order = only(option("order", [4]))
    modes = filter(a -> !occursin('=', a), ARGS)
    isempty(modes) && (modes = ["pump", "emitters", "dephasing", "g2tau", "timing"])

    if "orders" in modes
        orders = option("orders", [4, 5, 6])
        path = joinpath(DATA_DIR, "cumulant_orders_n010.tsv")
        rows = []        # keep stored rows of orders that are not recomputed
        if isfile(path)
            for l in filter(l -> !startswith(l, "#"), readlines(path))[2:end]
                f = split(l, '\t')
                parse(Int, f[1]) in orders || push!(rows, Tuple(f))
            end
        end
        for M in orders
            cs = load_cumulant_system(M; cache_dir=CACHE_DIR)
            ss = steady_state(cs, 10, G, KAPPA, 0.0, 1e-4, 0.0)
            growth = max_growth_rate(cs, ss.p, ss.u, ss.active)
            @printf("M=%d: %d equations, n=%.8g g2=%.8g growth=%.3g derivation %.0f s\n",
                    M, cs.n, ss.n, ss.g2, growth, cs.build_time)
            push!(rows, (M, cs.n, ss.n, ss.g2, ss.status, growth, cs.build_time))
        end
        sort!(rows, by=r -> parse(Int, string(r[1])))
        write_tsv(path,
                  "cumulant expansion at order M, n0=10, gamma_P=1e-4, gamma_A=gamma_D=0 " *
                  "(M=2,3 give g2=2 exactly); growth>0 marks an unstable fixed point; derivation time in s " *
                  "(0 if loaded from the cache)",
                  ["M", "nequations", "n", "g2", "status", "growth", "derivation_time"], rows)
        modes = filter(!=("orders"), modes)
        isempty(modes) && return
    end

    cs = load_cumulant_system(order; cache_dir=CACHE_DIR)
    println("order $order: $(cs.n) equations")
    ss = steady_state(cs, 10, G, KAPPA, 0.0, 1e-4, 0.0)          # warm-up
    @printf("test case n0=10, gamma_P=1e-4: n=%.8g g2=%.8g\n", ss.n, ss.g2)
    order == 4 && @assert isapprox(ss.g2, 29.936442; rtol=1e-5)

    if "pump" in modes
        rows = [point_row(cs, P, steady_state(cs, 10, G, KAPPA, 0.0, P, 0.0)) for P in PUMPS]
        write_tsv(joinpath(DATA_DIR, "cumulant_pump_n010.tsv"),
                  "cumulant expansion order $order, n0=10, gA=gD=0, x = pump rate P (1/ps)", COLS, rows)
    end
    if "emitters" in modes
        n0s = vcat(1:34, [40, 50, 60, 80, 100, 150, 200, 300, 500, 1000, 2000, 5000, 10000])
        rows = [point_row(cs, n0, steady_state(cs, n0, G, KAPPA, 0.0, 1e-4, 0.0)) for n0 in n0s]
        write_tsv(joinpath(DATA_DIR, "cumulant_emitters_P0.0001.tsv"),
                  "cumulant expansion order $order, P=1e-4, gA=gD=0, x = n0", COLS, rows)
    end
    if "dephasing" in modes
        rows = [point_row(cs, gD, steady_state(cs, 10, G, KAPPA, 0.0, 1e-4, gD)) for gD in DEPHASING]
        write_tsv(joinpath(DATA_DIR, "cumulant_dephasing_n010.tsv"),
                  "cumulant expansion order $order, n0=10, P=1e-4, gA=0, x = gamma_D (1/ps)", COLS, rows)
    end
    if "g2tau" in modes
        lines = filter(l -> !startswith(l, "#"), readlines(joinpath(DATA_DIR, "fig6_g2_tau_n010.tsv")))
        taus = [parse(Float64, split(l, '\t')[1]) for l in lines[2:end]]
        g2s, rc = g2_tau(cs, ss, taus)
        @printf("g2(tau): g2(0)=%.6g, minimum %.4g at tau=%.4g ps (%s)\n",
                g2s[1], minimum(g2s), taus[argmin(g2s)], rc)
        write_tsv(joinpath(DATA_DIR, "cumulant_g2_tau_n010.tsv"),
                  "cumulant expansion order $order, conditional-state regression, n0=10, P=1e-4, gA=gD=0",
                  ["tau", "g2"], collect(zip(taus, g2s)))
    end
    if "timing" in modes
        for (P, n0s) in ((1e-4, vcat(1:34, [40, 50, 60, 80, 100, 150, 200, 300, 500, 1000])),
                         (2.0, vcat(1:20, [25, 30, 40, 50, 60, 80, 100])))
            rows = []
            for n0 in n0s
                runs = [steady_state(cs, n0, G, KAPPA, 0.0, P, 0.0) for _ in 1:3]
                r = runs[end]
                wall = median3([x.walltime for x in runs])
                push!(rows, (n0, r.n, r.g2, r.status, r.residual, wall, r.route))
                @printf("  P=%g n0=%-6d g2=%-10.5g status=%d  %.4f s\n", P, n0, r.g2, r.status, wall)
                flush(stdout)
            end
            write_tsv(joinpath(DATA_DIR, "cumulant_timing_P$(P).tsv"),
                      "cumulant expansion order $order, per-point wall time (median of 3, derivation excluded), " *
                      "P=$P, gA=gD=0", ["n0", "n", "g2", "status", "residual", "walltime", "route"], rows)
        end
    end
end

main()
