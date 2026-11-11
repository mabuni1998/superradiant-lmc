# Fig. 2: polarization-order hierarchy for the collective test case.
#
# The master equation is truncated at polarization order B = 1, ..., n0
# (B = n0 is exact) at the photon cutoff of the converged exact solve.
#
#   julia --project=. scripts/fig2_polarization_order.jl [n0=10] [gamma_P=1e-4]
#
# Writes data/fig2_polarization_order_n0<n0>.tsv.  Runtime: ~1 min.
# The cumulant-expansion points of the figure come from cumulant/run_cumulant.jl.

include("common.jl")

n0 = option("n0", 10)
gamma_P = option("gamma_P", P_BELOW)
p = LaserParams(n0; gamma_P=gamma_P)

exact = solve_exact_me(p; verbose=true)
println("converged photon cutoff $(exact.cutoff), exact g2 = $(exact.g2)")

rows = []
for B in 1:n0
    t = @elapsed s = solve_polarization_order(p, exact.cutoff, B)
    @printf("B=%2d  dim=%7d  n=%.8g  g2=%.8g  [%.2f s]\n", B, s.dim, s.n, s.g2, t)
    push!(rows, (B, s.dim, s.n, s.g2))
end

write_tsv(data_path("fig2_polarization_order_n0$(n0).tsv"),
          "ME truncated at polarization order B, n0=$n0, gamma_P=$gamma_P, gamma_A=gamma_D=0, " *
          "photon cutoff $(exact.cutoff); B=n0 is the exact ME",
          ["B", "dim", "n", "g2"], rows)
