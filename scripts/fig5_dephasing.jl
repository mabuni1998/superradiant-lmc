# Fig. 5: g2(0) versus the pure dephasing rate gamma_D (n0 = 10, gamma_P = 1e-4).
#
#   julia --project=. scripts/fig5_dephasing.jl [n0=10] [methods=me,pp,lmc,conv]
#
# Writes data/fig5_dephasing_n0<n0>.tsv (long format: method, x = gamma_D, n, g2).
# Runtime: a few minutes.

include("common.jl")

n0 = option("n0", 10)
gamma_P = option("gamma_P", P_BELOW)
methods = option("methods", ["me", "pp", "lmc", "conv"])
me_dim_cap = option("me_dim_cap", 400_000)

rows = []
for gamma_D in dephasing_grid()
    p = LaserParams(n0; gamma_P=gamma_P, gamma_D=gamma_D)
    lmc = solve_dicke_lmc(p)
    line = @sprintf("gamma_D=%.4g:", gamma_D)
    if "lmc" in methods
        push!(rows, ("lmc", gamma_D, lmc.n, lmc.g2))
        line *= @sprintf(" lmc %.5g", lmc.g2)
    end
    if "conv" in methods
        c = solve_conventional_lmc(p)
        push!(rows, ("conv", gamma_D, c.n, c.g2))
        line *= @sprintf(" conv %.5g", c.g2)
    end
    if "pp" in methods
        s = attempt(() -> solve_population_polarization(p; initial_cutoff=lmc.cutoff), "pp")
        push!(rows, ("pp", gamma_D, value_or_nan(s, :n), value_or_nan(s, :g2)))
        line *= @sprintf(" pp %.5g", value_or_nan(s, :g2))
    end
    if "me" in methods
        s = attempt(() -> solve_exact_me(p; initial_cutoff=lmc.cutoff, dim_cap=me_dim_cap), "me")
        push!(rows, ("me", gamma_D, value_or_nan(s, :n), value_or_nan(s, :g2)))
        line *= @sprintf(" me %.5g", value_or_nan(s, :g2))
    end
    println(line)
    flush(stdout)
end

merge_method_rows(data_path("fig5_dephasing_n0$(n0).tsv"),
                  "dephasing sweep, n0=$n0, gamma_P=$gamma_P, gamma_A=0, x = gamma_D (1/ps)",
                  ["method", "x", "n", "g2"], methods, rows)
