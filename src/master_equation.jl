# Exact master equation in the symmetry-reduced representation, and its
# polarization-order hierarchy.
#
# The density matrix is expanded in elements rho[n, j, m; n', j, m'] (j is
# conserved by the permutation-symmetric dynamics).  The Liouvillian conserves
# the charge c = (n + m) - (n' + m'), and the unique steady state lies in the
# c = 0 sector, so only elements with n + m = n' + m' are kept.  Within that
# sector the polarization order b = |n - n'| = |m' - m| labels the elements:
#   b = 0  populations
#   b = 1  optical polarization (first order)
#   b = 2  two-photon coherences (second order) ...
# Truncating at order B gives the hierarchy of the paper; B = min(cutoff, n0)
# keeps every element of the sector, i.e. the exact permutationally invariant
# master equation.
#
# Keys are (n, 2j, 2m, n', 2m').

const DensityKey = NTuple{5, Int}

"""
    density_basis(n0, cutoff, B)

All zero-charge density-matrix elements with photon numbers `<= cutoff` and
polarization order `|n - n'| <= B`.
"""
function density_basis(n0::Integer, cutoff::Integer, B::Integer)
    B >= 0 || throw(ArgumentError("polarization order must be non-negative"))
    basis = DensityKey[]
    for j2 in n0:-2:0, m2 in -j2:2:j2, n in 0:cutoff
        charge2 = 2n + m2
        for mp2 in -j2:2:j2
            iseven(charge2 - mp2) || continue
            np = (charge2 - mp2) ÷ 2
            0 <= np <= cutoff || continue
            abs(n - np) <= B || continue
            push!(basis, (n, j2, m2, np, mp2))
        end
    end
    return basis
end

safe_sqrt(x) = sqrt(max(float(x), 0.0))

# Matrix elements of the collective operators J_+, J_-, J_z between Dicke
# states for the three j-branches of a local jump (A: j -> j, B: j -> j-1,
# D: j -> j+1), cf. Shammah et al., PRA 98, 063815 (2018).
function dicke_amplitude(channel::Symbol, branch::Symbol, j, m)
    if branch == :A
        channel == :plus && return safe_sqrt((j - m) * (j + m + 1))
        channel == :minus && return safe_sqrt((j + m) * (j - m + 1))
        channel == :z && return m
    elseif branch == :B
        channel == :plus && return safe_sqrt((j - m) * (j - m - 1))
        channel == :minus && return -safe_sqrt((j + m) * (j + m - 1))
        channel == :z && return safe_sqrt((j + m) * (j - m))
    elseif branch == :D
        channel == :plus && return -safe_sqrt((j + m + 1) * (j + m + 2))
        channel == :minus && return safe_sqrt((j - m + 1) * (j - m + 2))
        channel == :z && return safe_sqrt((j + m + 1) * (j - m + 1))
    end
    error("unknown channel=$channel or branch=$branch")
end

# Coefficient of sum_i L_i rho L_i^dagger for the local jump branch, between
# |j,m><j,m'| and the corresponding element of the target ladder.
function local_jump_coefficient(channel::Symbol, branch::Symbol, n0, j, m, mp)
    S = n0 / 2
    if branch == :A
        j == 0 && return 0.0
        eta = (S + 1) / (2j * (j + 1))
    elseif branch == :B
        j == 0 && return 0.0
        eta = (S + j + 1) / (2j * (2j + 1))
    elseif branch == :D
        eta = (S - j) / (2 * (j + 1) * (2j + 1))
    else
        error("unknown branch $branch")
    end
    return eta * dicke_amplitude(channel, branch, j, m) * dicke_amplitude(channel, branch, j, mp)
end

function push_entry!(rows, cols, vals, index, target, src, value)
    abs(value) <= 1e-15 && return
    dst = get(index, target, 0)
    dst == 0 && return            # outside the truncated basis
    push!(rows, dst); push!(cols, src); push!(vals, ComplexF64(value))
    return
end

# Local dissipator rate * sum_i D[L_i] for L_i = sigma_i (:minus),
# sigma_i^dagger (:plus), or sigma_i^dagger sigma_i (:z).
function add_local_dissipator!(rows, cols, vals, index, src, key, n0, rate, channel)
    rate == 0 && return
    n, j2, m2, np, mp2 = key
    j = j2 / 2; m = m2 / 2; mp = mp2 / 2
    S = n0 / 2
    shift2 = channel == :plus ? 2 : channel == :minus ? -2 : 0

    for (branch, jshift2) in ((:A, 0), (:B, -2), (:D, 2))
        c = local_jump_coefficient(channel, branch, n0, j, m, mp)
        c == 0 && continue
        push_entry!(rows, cols, vals, index, (n, j2 + jshift2, m2 + shift2, np, mp2 + shift2),
                    src, rate * c)
    end
    if channel == :z
        # sigma^dagger sigma = (1 + sigma_z)/2, so sum_i n_i rho n_i
        # = (S/2 + (m + m')/2) rho + (1/4) sum_i sigma_z,i rho sigma_z,i.  The
        # second piece is the (:A) branch above; the first is added here.
        push_entry!(rows, cols, vals, index, key, src, rate * (S / 2 + (m + mp) / 2))
    end
    occupation = channel == :plus ? (S - m) + (S - mp) : (S + m) + (S + mp)
    push!(rows, src); push!(cols, src); push!(vals, ComplexF64(-rate * occupation / 2))
    return
end

"""
    symmetry_reduced_liouvillian(p, cutoff, B) -> (L, basis, trace)

Sparse Liouvillian of the laser master equation in the zero-charge sector,
truncated at photon number `cutoff` and polarization order `B`.  `trace` is
the vector with `dot(trace, rho) == Tr(rho)`.
"""
function symmetry_reduced_liouvillian(p::LaserParams, cutoff::Integer, B::Integer)
    n0 = p.n0
    g = p.g
    basis = density_basis(n0, cutoff, B)
    index = Dict(k => i for (i, k) in enumerate(basis))
    rows = Int[]; cols = Int[]; vals = ComplexF64[]

    for (src, key) in enumerate(basis)
        n, j2, m2, np, mp2 = key
        j = j2 / 2; m = m2 / 2; mp = mp2 / 2

        # -i [H, rho], H = g (a^dagger J_- + a J_+); action on the ket ...
        if n > 0 && m2 < j2
            amp = g * sqrt(n) * dicke_amplitude(:plus, :A, j, m)
            push_entry!(rows, cols, vals, index, (n - 1, j2, m2 + 2, np, mp2), src, -im * amp)
        end
        if n < cutoff && m2 > -j2
            amp = g * sqrt(n + 1) * dicke_amplitude(:minus, :A, j, m)
            push_entry!(rows, cols, vals, index, (n + 1, j2, m2 - 2, np, mp2), src, -im * amp)
        end
        # ... and on the bra.
        if np > 0 && mp2 < j2
            amp = g * sqrt(np) * dicke_amplitude(:plus, :A, j, mp)
            push_entry!(rows, cols, vals, index, (n, j2, m2, np - 1, mp2 + 2), src, im * amp)
        end
        if np < cutoff && mp2 > -j2
            amp = g * sqrt(np + 1) * dicke_amplitude(:minus, :A, j, mp)
            push_entry!(rows, cols, vals, index, (n, j2, m2, np + 1, mp2 - 2), src, im * amp)
        end

        # Cavity loss gamma_c D[a].
        if n > 0 && np > 0
            push_entry!(rows, cols, vals, index, (n - 1, j2, m2, np - 1, mp2), src,
                        p.gamma_c * sqrt(n * np))
        end
        push!(rows, src); push!(cols, src); push!(vals, ComplexF64(-p.gamma_c * (n + np) / 2))

        add_local_dissipator!(rows, cols, vals, index, src, key, n0, p.gamma_A, :minus)
        add_local_dissipator!(rows, cols, vals, index, src, key, n0, p.gamma_P, :plus)
        add_local_dissipator!(rows, cols, vals, index, src, key, n0, p.gamma_D, :z)
    end

    L = sparse(rows, cols, vals, length(basis), length(basis))
    dropzeros!(L)
    trace = ComplexF64[k[1] == k[4] && k[3] == k[5] ? 1.0 : 0.0 for k in basis]
    return L, basis, trace
end

"Mean photon number, g2(0), stationary probability at the cutoff, and smallest population."
function photon_observables(rho, basis, cutoff)
    n = 0.0; f2 = 0.0; edge = 0.0; minpop = Inf
    for (i, (np1, _, m2, np2, mp2)) in enumerate(basis)
        (np1 == np2 && m2 == mp2) || continue
        w = real(rho[i])
        minpop = min(minpop, w)
        n += np1 * w
        f2 += np1 * (np1 - 1) * w
        np1 == cutoff && (edge += w)
    end
    return n, f2 / n^2, edge, minpop
end

"""
    solve_polarization_order(p, cutoff, B)

Stationary state of the master equation truncated at polarization order `B`
and photon cutoff `cutoff` (no adaptivity).  `B = 1` is close to the Dicke
LMC, `B = min(cutoff, n0)` is exact.  Returns `(n, g2, edge, dim, residual, rho, basis)`.
"""
function solve_polarization_order(p::LaserParams, cutoff::Integer, B::Integer; atol=1e-13)
    L, basis, trace = symmetry_reduced_liouvillian(p, cutoff, B)
    start = copy(trace)                    # maximally mixed diagonal state
    rho, residual = stationary_state(L, trace, start; atol=atol)
    n, g2, edge, _ = photon_observables(rho, basis, cutoff)
    return (n=n, g2=g2, edge=edge, dim=length(basis), residual=residual, rho=rho, basis=basis)
end

"""
    solve_exact_me(p; initial_cutoff=29, rtol=1e-6, edge_tol=1e-10,
                   dim_cap=400_000, max_cutoff=1000, verbose=false)

Numerically exact steady state of the permutationally invariant master
equation.  The polarization order is raised through B = 1, 2, 4, 6, ... until
`<n>` and `g2(0)` change by less than `rtol` between consecutive solves; if
that never happens the sequence ends at B = min(cutoff, n0), which is exact.
The photon cutoff is then verified through the stationary population at the
cutoff (`<= edge_tol`) and enlarged if necessary.  Returns `nothing` if the
basis would exceed `dim_cap` elements, otherwise
`(n, g2, B, cutoff, dim, edge, residual)`.
"""
function solve_exact_me(p::LaserParams; initial_cutoff=29, rtol=1e-6, edge_tol=1e-10,
                        dim_cap=400_000, max_cutoff=1000, verbose=false)
    cutoff = initial_cutoff
    while true
        Bmax = min(cutoff, p.n0)
        orders = unique(vcat([b for b in (1, 2:2:Bmax...) if b < Bmax], Bmax))
        result = nothing
        prev_n = NaN; prev_g2 = NaN
        for B in orders
            dim = length(density_basis(p.n0, cutoff, B))
            if dim > dim_cap
                verbose && println("  dim $dim exceeds cap at B=$B")
                return nothing
            end
            s = solve_polarization_order(p, cutoff, B)
            verbose && @printf("  cutoff=%d B=%d dim=%d n=%.10g g2=%.10g edge=%.2e\n",
                               cutoff, B, dim, s.n, s.g2, s.edge)
            result = (n=s.n, g2=s.g2, B=B, cutoff=cutoff, dim=dim, edge=s.edge,
                      residual=s.residual)
            abs(s.g2 - prev_g2) <= rtol * abs(s.g2) && abs(s.n - prev_n) <= rtol * abs(s.n) && break
            prev_n, prev_g2 = s.n, s.g2
        end
        result.edge <= edge_tol && return result
        cutoff >= max_cutoff && error("photon cutoff did not converge")
        cutoff = next_cutoff(cutoff, max_cutoff)
        verbose && println("  edge $(result.edge) too large; new cutoff $cutoff")
    end
end

"""
    detection_map(state, keys)

Apply the photon-detection superoperator rho -> a rho a^dagger to a vector of
density-matrix elements labeled by `keys` (it preserves charge and polarization order).
"""
function detection_map(state, keys)
    index = Dict(k => i for (i, k) in enumerate(keys))
    w = zeros(ComplexF64, length(state))
    for (i, (n, j2, m2, np, mp2)) in enumerate(keys)
        (n == 0 || np == 0) && continue
        t = get(index, (n - 1, j2, m2, np - 1, mp2), 0)
        t == 0 && continue
        w[t] += sqrt(n * np) * state[i]
    end
    return w
end

photon_number_vector(keys) = [k[1] == k[4] && k[3] == k[5] ? k[1] : 0 for k in keys]

"""
    exact_me_g2_tau(p; cutoff=29, kwargs...) -> (taus, g2)

Delayed correlation of the exact master equation by the quantum regression
theorem, g2(tau) = Tr[n exp(L tau)(a rho_ss a^dagger)] / <n>^2, with the
full polarization order at a fixed photon cutoff (the stationary edge
population is checked).
"""
function exact_me_g2_tau(p::LaserParams; cutoff=29, edge_tol=1e-10, kwargs...)
    B = min(cutoff, p.n0)
    L, basis, trace = symmetry_reduced_liouvillian(p, cutoff, B)
    rho, _ = stationary_state(L, trace, copy(trace); atol=1e-13)
    n, _, edge, _ = photon_observables(rho, basis, cutoff)
    edge <= edge_tol || error("photon cutoff $cutoff too small (edge $edge)")
    nvec = photon_number_vector(basis)
    taus, vals = propagate(L, detection_map(rho, basis), v -> real(dot(nvec, v)); kwargs...)
    return taus, vals ./ n^2
end
