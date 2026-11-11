# Population--polarization model (second order of the elimination).
#
# The populations (b = 0) and the optical polarization (b = 1) of the
# symmetry-reduced Liouvillian are kept exactly, and the two-photon
# coherences (b = 2) are adiabatically eliminated with their collective
# eigenvalue, L22 -> -gamma_perp I: (a^dagger)^2 J_-^2 is an exact left
# eigenoperator of the dissipative Liouvillian with eigenvalue -gamma_perp,
# the two-photon analog of the closure that gives the Dicke LMC.  The
# polarization block is thereby folded to F = L11 + L12 L21 / gamma_perp, and
# the stationary problem is solved in the sparse augmented form
#
#     [L00 L01; L10 F] (P; Q) = 0,   sum(P) = 1,
#
# so the dense effective population generator is never formed.

"""
    population_polarization_generator(p, cutoff) -> (A, keys, trace, npop)

Augmented generator of the population--polarization model.  Variables are
the populations followed by the b = 1 polarization elements, labeled by
`keys`; the first `npop` entries are populations.
"""
function population_polarization_generator(p::LaserParams, cutoff::Integer)
    B = min(2, p.n0)
    L, basis, _ = symmetry_reduced_liouvillian(p, cutoff, B)
    order = [abs(k[1] - k[4]) for k in basis]
    i0 = findall(==(0), order)
    i1 = findall(==(1), order)
    i2 = findall(==(2), order)
    F = L[i1, i1] + (1 / gamma_perp(p)) * (L[i1, i2] * L[i2, i1])
    A = [L[i0, i0] L[i0, i1]; L[i1, i0] F]
    keys = vcat(basis[i0], basis[i1])
    trace = zeros(ComplexF64, size(A, 1))
    trace[1:length(i0)] .= 1.0
    return A, keys, trace, length(i0)
end

"""
    solve_population_polarization(p; initial_cutoff=29, edge_tol=1e-10,
                                  dim_cap=2_000_000, max_cutoff=100_000)

Stationary state of the population--polarization model with an adaptive
photon cutoff.  Returns `nothing` above `dim_cap` variables, otherwise
`(n, g2, cutoff, dim, residual, state, keys, A)`.  The stationary populations
can be negative at the 1e-6 level (signed effective rates); this does not
affect the low-order moments.
"""
function solve_population_polarization(p::LaserParams; initial_cutoff=29, edge_tol=1e-10,
                                       dim_cap=2_000_000, max_cutoff=100_000)
    cutoff = initial_cutoff
    while true
        length(density_basis(p.n0, cutoff, min(2, p.n0))) > dim_cap && return nothing
        A, keys, trace, npop = population_polarization_generator(p, cutoff)
        start = copy(trace)
        x, residual = stationary_state(A, trace, start; atol=1e-13)
        n, g2, edge, _ = photon_observables(real.(x[1:npop]), keys[1:npop], cutoff)
        if edge <= edge_tol
            return (n=n, g2=g2, cutoff=cutoff, dim=size(A, 1), residual=residual,
                    state=x, keys=keys, A=A)
        end
        cutoff >= max_cutoff && error("photon cutoff did not converge (edge $edge)")
        cutoff = next_cutoff(cutoff, max_cutoff)
    end
end

"""
    population_polarization_g2_tau(p; kwargs...) -> (taus, g2)

Delayed correlation of the population--polarization model by the quantum
regression theorem in the augmented space: the detection map
rho -> a rho a^dagger preserves the polarization order, so it acts on (P, Q)
directly, and the result is propagated with the augmented generator.
"""
function population_polarization_g2_tau(p::LaserParams; kwargs...)
    ss = solve_population_polarization(p)
    nvec = photon_number_vector(ss.keys)
    w = detection_map(ss.state, ss.keys)
    taus, vals = propagate(ss.A, w, v -> real(dot(nvec, v)); kwargs...)
    return taus, vals ./ ss.n^2
end
