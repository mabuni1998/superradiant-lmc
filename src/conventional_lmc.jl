# Conventional LMC: classical Markov chain over (n_p, n_e), i.e. the
# birth--death process of the standard laser rate equations.  It follows from
# the Dicke LMC by replacing C_-(j,m) -> n_e and C_+(j,m) -> n0 - n_e
# (uncorrelated emitters).  Pure dephasing enters only through gamma_r.

"""
    conventional_generator(p, cutoff) -> (M, states)

Sparse generator over the states `(n, n_e)`, `0 <= n <= cutoff`,
`0 <= n_e <= n0`, with a reflecting photon wall at `n == cutoff`.
"""
function conventional_generator(p::LaserParams, cutoff::Integer)
    n0 = p.n0
    gr = gamma_r(p)
    states = [(n, ne) for n in 0:cutoff for ne in 0:n0]
    index = Dict(s => i for (i, s) in enumerate(states))
    rows = Int[]; cols = Int[]; vals = Float64[]
    for (src, (n, ne)) in enumerate(states)
        events = (
            (p.gamma_P * (n0 - ne), (n, ne + 1)),     # incoherent pump
            (p.gamma_A * ne, (n, ne - 1)),            # background decay
            (p.gamma_c * n, (n - 1, ne)),             # cavity decay
            (gr * ne * n, (n + 1, ne - 1)),           # stimulated emission
            (gr * ne, (n + 1, ne - 1)),               # spontaneous emission
            (gr * (n0 - ne) * n, (n - 1, ne + 1)),    # stimulated absorption
        )
        out = 0.0
        for (rate, dst_state) in events
            rate > 0 || continue
            dst_state[1] > cutoff && continue
            dst = index[dst_state]
            push!(rows, dst); push!(cols, src); push!(vals, rate)
            out += rate
        end
        if out > 0
            push!(rows, src); push!(cols, src); push!(vals, -out)
        end
    end
    return sparse(rows, cols, vals, length(states), length(states)), states
end

"""
    solve_conventional_lmc(p; initial_cutoff=29, edge_tol=1e-10, max_cutoff=100_000)

Stationary state of the conventional LMC with an adaptive photon cutoff.
Returns a NamedTuple `(n, g2, ne, cutoff, nstates, probability, states)`.
"""
function solve_conventional_lmc(p::LaserParams; initial_cutoff=29, edge_tol=1e-10,
                                max_cutoff=100_000, atol=1e-11)
    cutoff = initial_cutoff
    while true
        M, states = conventional_generator(p, cutoff)
        prob, residual = stationary_probability(M; atol=atol)
        edge = sum(prob[i] for (i, s) in enumerate(states) if s[1] == cutoff)
        if edge <= edge_tol
            n = sum(w * s[1] for (w, s) in zip(prob, states))
            f2 = sum(w * s[1] * (s[1] - 1) for (w, s) in zip(prob, states))
            ne = sum(w * s[2] for (w, s) in zip(prob, states))
            return (n=n, g2=f2 / n^2, ne=ne, cutoff=cutoff, nstates=length(states),
                    residual=residual, edge=edge, probability=prob, states=states)
        end
        cutoff >= max_cutoff && error("photon cutoff did not converge (edge $edge)")
        cutoff = next_cutoff(cutoff, max_cutoff)
    end
end

"""
    conventional_lmc_g2_tau(p; kwargs...) -> (taus, g2)

Delayed correlation of the conventional LMC, by the same classical
regression as `dicke_lmc_g2_tau`.
"""
function conventional_lmc_g2_tau(p::LaserParams; kwargs...)
    ss = solve_conventional_lmc(p)
    M, states = conventional_generator(p, ss.cutoff)
    index = Dict(s => i for (i, s) in enumerate(states))
    pc = zeros(length(states))
    for (i, (n, ne)) in enumerate(states)
        n == 0 && continue
        pc[index[(n - 1, ne)]] += n * ss.probability[i] / ss.n
    end
    photons = [s[1] for s in states]
    taus, nvals = propagate(M, pc, v -> dot(photons, v); kwargs...)
    return taus, nvals ./ ss.n
end
