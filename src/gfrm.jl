# Trajectory sampling of the Dicke LMC with Gillespie's first reaction method
# (GFRM).  The cost per event is independent of n0 and n_p, so the chain can
# be sampled for arbitrarily many emitters.

"""
    gfrm_dicke_lmc(p, nsteps; rng=Random.default_rng(), burnin=nsteps ÷ 10)

Sample one long trajectory of the Dicke LMC with Gillespie's first reaction
method: from the current state, an exponential waiting time is drawn for
every event with nonzero rate, and the earliest event is executed.  The
trajectory starts in the cavity vacuum with all emitters in the ground state,
`(n, 2j, 2m) = (0, n0, -n0)`; the first `burnin` events are discarded, and
the stationary moments are time averages over the remaining events.

Returns `(n, g2, ne, time)` with `g2 = (<n^2> - <n>) / <n>^2`.
"""
function gfrm_dicke_lmc(p::LaserParams, nsteps::Integer;
                        rng::AbstractRNG=Random.default_rng(), burnin::Integer=nsteps ÷ 10)
    gr = gamma_r(p)
    a = zeros(length(DICKE_EVENTS))
    n, j2, m2 = 0, p.n0, -p.n0
    T = 0.0; S1 = 0.0; S2 = 0.0; Sm = 0.0
    for step in 1:nsteps
        dicke_rates!(a, n, j2, m2, p, gr)
        dt = Inf
        k = 0
        @inbounds for e in eachindex(a)
            a[e] > 0 || continue
            tau = randexp(rng) / a[e]
            if tau < dt
                dt = tau
                k = e
            end
        end
        k == 0 && error("absorbing state reached: all rates vanish")
        if step > burnin
            T += dt
            S1 += n * dt
            S2 += n^2 * dt
            Sm += m2 * dt
        end
        dn, dj2, dm2 = DICKE_EVENTS[k]
        n += dn; j2 += dj2; m2 += dm2
    end
    nbar = S1 / T
    return (n=nbar, g2=(S2 / T - nbar) / nbar^2, ne=(p.n0 + Sm / T) / 2, time=T)
end
