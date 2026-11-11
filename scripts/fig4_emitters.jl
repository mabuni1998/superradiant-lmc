# Fig. 4: g2(0) versus emitter number at gamma_P = 1e-4.
#
#   julia --project=. scripts/fig4_emitters.jl [methods=me,pp,lmc,conv,gfrm]
#       [me_n0=...] [pp_n0=...] [lmc_n0=...] [conv_n0=...] [gfrm_n0=...]
#       [gfrm_steps=1000000000] [dim_cap=20000000]
#
# Writes data/fig4_emitters_P<gamma_P>.tsv (long format: method, x = n0, n, g2).
# Each method can be run on its own (e.g. `methods=gfrm`); rows of the other
# methods already in the file are kept.  A scan stops at the first n0 that
# fails (dimension cap, memory), since larger n0 will not fit either.
#
# Cost of the default (paper) ranges on one core: exact ME up to n0 = 34
# ~10 min per point at the top and tens of GB of memory; population--
# polarization model up to n0 = 104 ~1 h in total; Dicke LMC < 1 min in
# total; conventional LMC ~minutes; GFRM a few minutes per point for 1e9
# events, independent of n0.

include("common.jl")

gamma_P = option("gamma_P", P_BELOW)
methods = option("methods", ["me", "pp", "lmc", "conv", "gfrm"])
me_n0 = option("me_n0", collect(1:34))
pp_n0 = option("pp_n0", vcat(1:20, [25, 29, 33, 38, 44, 51, 59, 68, 78, 90, 104]))
lmc_n0 = option("lmc_n0", emitter_list(100))
conv_n0 = option("conv_n0", vcat(1:20, [25, 35, 49, 68, 96, 134, 187, 261, 366, 511, 715, 1000]))
gfrm_n0 = option("gfrm_n0", [200, 500, 1000, 2000, 5000, 10000])
gfrm_steps = option("gfrm_steps", 1_000_000_000)
dim_cap = option("dim_cap", 20_000_000)

rows = []
function scan!(rows, key, n0s, solve)
    for n0 in n0s
        t = @elapsed s = attempt(() -> solve(LaserParams(n0; gamma_P=gamma_P)), "$key n0=$n0")
        s === nothing && break
        push!(rows, (key, n0, s.n, s.g2))
        @printf("%-4s n0=%-6d n=%.6g g2=%.6g [%.2f s]\n", key, n0, s.n, s.g2, t)
        flush(stdout)
    end
end

"lmc" in methods && scan!(rows, "lmc", lmc_n0, solve_dicke_lmc)
"conv" in methods && scan!(rows, "conv", conv_n0, solve_conventional_lmc)
"pp" in methods && scan!(rows, "pp", pp_n0, p -> solve_population_polarization(p; dim_cap=dim_cap))
"me" in methods && scan!(rows, "me", me_n0, p -> solve_exact_me(p; dim_cap=dim_cap))
"gfrm" in methods && scan!(rows, "gfrm", gfrm_n0, p -> gfrm_dicke_lmc(p, gfrm_steps))

merge_method_rows(data_path("fig4_emitters_P$(gamma_P).tsv"),
                  "emitter scan, gamma_P=$gamma_P, gamma_A=gamma_D=0, x = n0",
                  ["method", "x", "n", "g2"], methods, rows)
