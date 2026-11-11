# Dicke LMC: classical Markov chain over (n_p, j, m).
#
# States are stored as integer triples (n, 2j, 2m).  The event table is the
# one of the paper (Table "events_dicke"): three radiative events with the
# collective Dicke factors C_-(j,m) and C_+(j,m), cavity decay, and the
# j-branching local decay, pump, and dephasing events.

"""
Change of `(n, 2j, 2m)` for each event, in the order used by `dicke_rates!`:

     1 cavity decay               7 decay  j -> j+1
     2 stimulated absorption      8 pump   j -> j
     3 stimulated emission        9 pump   j -> j-1
     4 spontaneous emission      10 pump   j -> j+1
     5 decay  j -> j             11 dephasing j -> j-1
     6 decay  j -> j-1           12 dephasing j -> j+1
"""
const DICKE_EVENTS = (
    (-1, 0, 0),
    (-1, 0, 2),
    (1, 0, -2),
    (1, 0, -2),
    (0, 0, -2),
    (0, -2, -2),
    (0, 2, -2),
    (0, 0, 2),
    (0, -2, 2),
    (0, 2, 2),
    (0, -2, 0),
    (0, 2, 0),
)

valid_dicke(j2, m2, n0) = 0 <= j2 <= n0 && abs(m2) <= j2 && iseven(j2 - m2)

"""
    dicke_rates!(a, n, j2, m2, p::LaserParams, gr=gamma_r(p))

Fill `a` (length 12) with the rates of the events in `DICKE_EVENTS` for the
state `(n, 2j, 2m) = (n, j2, m2)`.  Rates of transitions that would leave the
set of valid Dicke states are zero.  The local rates obey the sum rules
sum(a[5:7]) = gamma_A n_e and sum(a[8:10]) = gamma_P (n0 - n_e).
"""
function dicke_rates!(a, n, j2, m2, p::LaserParams, gr=gamma_r(p))
    n0 = p.n0
    j = j2 / 2
    m = m2 / 2
    S = n0 / 2
    Cminus = (j + m) * (j - m + 1)
    Cplus = (j - m) * (j + m + 1)

    fill!(a, 0.0)
    a[1] = p.gamma_c * n
    a[2] = gr * n * Cplus
    a[3] = gr * n * Cminus
    a[4] = gr * Cminus

    # Local decay sigma_i: m -> m-1, j -> j, j-1, j+1.
    if valid_dicke(j2, m2 - 2, n0) && j > 0
        a[5] = p.gamma_A / 2 * (S + 1) / (j * (j + 1)) * (j + m) * (j - m + 1)
    end
    if valid_dicke(j2 - 2, m2 - 2, n0) && j > 0
        a[6] = p.gamma_A / 2 * (S + j + 1) / (j * (2j + 1)) * (j + m) * (j + m - 1)
    end
    if valid_dicke(j2 + 2, m2 - 2, n0)
        a[7] = p.gamma_A / 2 * (S - j) / ((j + 1) * (2j + 1)) * (j - m + 1) * (j - m + 2)
    end

    # Local pump sigma_i^dagger: m -> m+1, j -> j, j-1, j+1.
    if valid_dicke(j2, m2 + 2, n0) && j > 0
        a[8] = p.gamma_P / 2 * (S + 1) / (j * (j + 1)) * (j - m) * (j + m + 1)
    end
    if valid_dicke(j2 - 2, m2 + 2, n0) && j > 0
        a[9] = p.gamma_P / 2 * (S + j + 1) / (j * (2j + 1)) * (j - m) * (j - m - 1)
    end
    if valid_dicke(j2 + 2, m2 + 2, n0)
        a[10] = p.gamma_P / 2 * (S - j) / ((j + 1) * (2j + 1)) * (j + m + 1) * (j + m + 2)
    end

    # Local pure dephasing sigma_i^dagger sigma_i: m -> m, j -> j-1, j+1.
    if valid_dicke(j2 - 2, m2, n0) && j > 0
        a[11] = p.gamma_D / 2 * (S + j + 1) / (j * (2j + 1)) * (j^2 - m^2)
    end
    if valid_dicke(j2 + 2, m2, n0)
        a[12] = p.gamma_D / 2 * (S - j) / ((j + 1) * (2j + 1)) * ((j + 1)^2 - m^2)
    end
    return a
end

"All `(n, 2j, 2m)` states with `0 <= n <= cutoff`, photon number outermost."
function dicke_states(n0::Integer, cutoff::Integer)
    states = NTuple{3, Int}[]
    for n in 0:cutoff, j2 in n0:-2:0, m2 in -j2:2:j2
        push!(states, (n, j2, m2))
    end
    return states
end

"""
    dicke_generator(p, cutoff) -> (M, states)

Sparse generator of the Dicke LMC, `dP/dt = M P`, columns are source states.
Photon-creating events are suppressed at `n == cutoff` (reflecting wall); the
adaptive solvers check that the stationary probability on that edge is
negligible.
"""
function dicke_generator(p::LaserParams, cutoff::Integer)
    states = dicke_states(p.n0, cutoff)
    index = Dict(s => i for (i, s) in enumerate(states))
    gr = gamma_r(p)
    a = zeros(length(DICKE_EVENTS))
    rows = Int[]; cols = Int[]; vals = Float64[]
    for (src, (n, j2, m2)) in enumerate(states)
        dicke_rates!(a, n, j2, m2, p, gr)
        out = 0.0
        for (k, (dn, dj2, dm2)) in enumerate(DICKE_EVENTS)
            a[k] > 0 || continue
            n + dn > cutoff && continue
            dst = index[(n + dn, j2 + dj2, m2 + dm2)]
            push!(rows, dst); push!(cols, src); push!(vals, a[k])
            out += a[k]
        end
        if out > 0
            push!(rows, src); push!(cols, src); push!(vals, -out)
        end
    end
    M = sparse(rows, cols, vals, length(states), length(states))
    return M, states
end

"""
    stationary_probability(M; atol=1e-11)

Stationary distribution of a classical generator.  Round-off negatives above
`-atol` are clipped; anything more negative is an error.
"""
function stationary_probability(M::SparseMatrixCSC{Float64}; atol=1e-11)
    nstates = size(M, 1)
    tol = max(atol, 100eps(Float64))
    p, _ = stationary_state(M, ones(nstates), fill(1.0 / nstates, nstates);
                            atol=tol, maxiter=50)
    minimum(p) >= -atol || error("stationary solve produced a negative probability $(minimum(p))")
    p .= max.(p, 0.0)
    p ./= sum(p)
    residual = norm(M * p, Inf)
    residual <= tol || error("stationary residual $residual exceeds tolerance $tol")
    return p, residual
end

"""
    solve_dicke_lmc(p; initial_cutoff=29, edge_tol=1e-10, max_cutoff=100_000)

Stationary state of the Dicke LMC by a direct sparse solve.  The photon cutoff
is enlarged until the stationary probability at `n == cutoff` is below
`edge_tol`.  Returns a NamedTuple with `n` (mean photon number), `g2`
(= g2(0)), `ne` (excited emitters), `Cminus` (<J_+J_->), `eta_sp`
(= <C_->/<n_e>), the converged `cutoff`, `nstates`, the stationary
`probability`, and the matching `states`.
"""
function solve_dicke_lmc(p::LaserParams; initial_cutoff=29, edge_tol=1e-10,
                         max_cutoff=100_000, atol=1e-11)
    cutoff = initial_cutoff
    while true
        M, states = dicke_generator(p, cutoff)
        prob, residual = stationary_probability(M; atol=atol)
        edge = sum(prob[i] for (i, s) in enumerate(states) if s[1] == cutoff)
        if edge <= edge_tol
            n = 0.0; f2 = 0.0; ne = 0.0; Cm = 0.0
            for (w, (np, j2, m2)) in zip(prob, states)
                j = j2 / 2; m = m2 / 2
                n += w * np
                f2 += w * np * (np - 1)
                ne += w * (p.n0 + m2) / 2
                Cm += w * (j + m) * (j - m + 1)
            end
            return (n=n, g2=f2 / n^2, ne=ne, Cminus=Cm, eta_sp=Cm / ne, cutoff=cutoff,
                    nstates=length(states), residual=residual, edge=edge,
                    probability=prob, states=states)
        end
        cutoff >= max_cutoff && error("photon cutoff did not converge (edge $edge)")
        cutoff = next_cutoff(cutoff, max_cutoff)
    end
end

"""
    dicke_lmc_g2_tau(p; kwargs...) -> (taus, g2)

Delayed correlation of the Dicke LMC.  A photon detection at tau = 0 maps the
stationary distribution to P_c(n-1, j, m) = n P_ss(n, j, m) / <n>, which is
propagated with the generator: g2(tau) = <n(tau)>_c / <n>.
"""
function dicke_lmc_g2_tau(p::LaserParams; kwargs...)
    ss = solve_dicke_lmc(p)
    M, states = dicke_generator(p, ss.cutoff)
    index = Dict(s => i for (i, s) in enumerate(states))
    pc = zeros(length(states))
    for (i, (n, j2, m2)) in enumerate(states)
        n == 0 && continue
        pc[index[(n - 1, j2, m2)]] += n * ss.probability[i] / ss.n
    end
    photons = [s[1] for s in states]
    taus, nvals = propagate(M, pc, v -> dot(photons, v); kwargs...)
    return taus, nvals ./ ss.n
end
