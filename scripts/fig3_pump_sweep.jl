# Fig. 3: photon number and g2(0) versus pump rate for n0 = 10.
#
#   julia --project=. scripts/fig3_pump_sweep.jl [n0=10] [methods=me,pp,lmc,conv]
#
# Writes data/fig3_pump_sweep_n0<n0>.tsv (long format: method, x = gamma_P, n, g2).
# Runtime: a few minutes; the exact ME dominates near and above threshold,
# where it may exceed `me_dim_cap` (written as NaN).

include("common.jl")

n0 = option("n0", 10)
methods = option("methods", ["me", "pp", "lmc", "conv"])
me_dim_cap = option("me_dim_cap", 400_000)

rows = []
for gamma_P in pump_grid()
    p = LaserParams(n0; gamma_P=gamma_P)
    lmc = solve_dicke_lmc(p)        # also provides the initial cutoff for the ME
    line = @sprintf("gamma_P=%.4g:", gamma_P)
    if "lmc" in methods
        push!(rows, ("lmc", gamma_P, lmc.n, lmc.g2))
        line *= @sprintf(" lmc %.5g", lmc.g2)
    end
    if "conv" in methods
        c = solve_conventional_lmc(p)
        push!(rows, ("conv", gamma_P, c.n, c.g2))
        line *= @sprintf(" conv %.5g", c.g2)
    end
    if "pp" in methods
        s = attempt(() -> solve_population_polarization(p), "pp")
        push!(rows, ("pp", gamma_P, value_or_nan(s, :n), value_or_nan(s, :g2)))
        line *= @sprintf(" pp %.5g", value_or_nan(s, :g2))
    end
    if "me" in methods
        s = attempt(() -> solve_exact_me(p; initial_cutoff=max(29, lmc.cutoff),
                                         dim_cap=me_dim_cap), "me")
        push!(rows, ("me", gamma_P, value_or_nan(s, :n), value_or_nan(s, :g2)))
        line *= @sprintf(" me %.5g", value_or_nan(s, :g2))
    end
    println(line)
    flush(stdout)
end

merge_method_rows(data_path("fig3_pump_sweep_n0$(n0).tsv"),
                  "pump sweep, n0=$n0, gamma_A=gamma_D=0, x = gamma_P (1/ps)",
                  ["method", "x", "n", "g2"], methods, rows)
