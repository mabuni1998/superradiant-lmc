# Fig. 6: delayed correlation g2(tau) in the collective regime
# (n0 = 10, gamma_P = 1e-4, gamma_A = gamma_D = 0).
#
#   julia --project=. scripts/fig6_g2_tau.jl [n0=10]
#
# Writes data/fig6_g2_tau_n0<n0>.tsv with columns tau, me, pp, lmc, conv.
# Runtime: ~10 min, dominated by the exact ME (Crank--Nicolson on the full
# symmetry-reduced Liouvillian).

include("common.jl")

n0 = option("n0", 10)
p = LaserParams(n0; gamma_P=option("gamma_P", P_BELOW))

curves = Dict{String, Vector{Float64}}()
taus = nothing
for (key, f) in (("lmc", dicke_lmc_g2_tau), ("conv", conventional_lmc_g2_tau),
                 ("pp", population_polarization_g2_tau), ("me", exact_me_g2_tau))
    t = @elapsed (tk, gk) = f(p)
    @printf("%-4s g2(0)=%.6g  g2(end)=%.6g  [%.1f s]\n", key, gk[1], gk[end], t)
    flush(stdout)
    global taus
    taus === nothing ? (taus = tk) : (taus == tk || error("time grids differ"))
    curves[key] = gk
end

write_tsv(data_path("fig6_g2_tau_n0$(n0).tsv"),
          "g2(tau), n0=$n0, gamma_P=$(p.gamma_P), gamma_A=gamma_D=0, tau in ps",
          ["tau", "me", "pp", "lmc", "conv"],
          [(taus[i], curves["me"][i], curves["pp"][i], curves["lmc"][i], curves["conv"][i])
           for i in eachindex(taus)])
