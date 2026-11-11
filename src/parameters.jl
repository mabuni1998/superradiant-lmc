# Physical parameters shared by every calculation in the paper.
# All rates are in ps^-1.

"Light--matter coupling g used throughout the paper (ps^-1)."
const G_DEFAULT = 0.1
"Cavity loss rate gamma_c used throughout the paper (ps^-1)."
const GAMMA_C_DEFAULT = 0.04
"Pump rate of the below-threshold (collective) test case (ps^-1)."
const P_BELOW = 1e-4
"Pump rate of the above-threshold timing benchmark (ps^-1)."
const P_ABOVE = 2.0

"""
    LaserParams(n0; g, gamma_c, gamma_P, gamma_A, gamma_D)

Parameters of the laser master equation for `n0` emitters.  Defaults are the
paper's: `g = 0.1`, `gamma_c = 0.04`, `gamma_P = 1e-4`, `gamma_A = gamma_D = 0`
(all in ps^-1).  Pure dephasing uses the Lindblad operator
`sigma_i^dagger sigma_i` (the `gamma_D` convention of the paper).
"""
struct LaserParams
    n0::Int
    g::Float64
    gamma_c::Float64
    gamma_P::Float64
    gamma_A::Float64
    gamma_D::Float64
    function LaserParams(n0, g, gamma_c, gamma_P, gamma_A, gamma_D)
        n0 >= 1 || throw(ArgumentError("n0 must be positive"))
        all(>=(0), (g, gamma_c, gamma_P, gamma_A, gamma_D)) ||
            throw(ArgumentError("rates must be non-negative"))
        new(n0, g, gamma_c, gamma_P, gamma_A, gamma_D)
    end
end

LaserParams(n0::Integer; g=G_DEFAULT, gamma_c=GAMMA_C_DEFAULT, gamma_P=P_BELOW,
            gamma_A=0.0, gamma_D=0.0) =
    LaserParams(Int(n0), Float64(g), Float64(gamma_c), Float64(gamma_P),
                Float64(gamma_A), Float64(gamma_D))

"Decay rate of the collective polarization, gamma_perp = gamma_c + gamma_A + gamma_P + gamma_D."
gamma_perp(p::LaserParams) = p.gamma_c + p.gamma_A + p.gamma_P + p.gamma_D

"Radiative rate of the adiabatically eliminated polarization, gamma_r = 4 g^2 / gamma_perp."
gamma_r(p::LaserParams) = 4 * p.g^2 / gamma_perp(p)

"The 50 log-spaced pump rates of the pump sweep, 1e-4 ... 20 ps^-1."
pump_grid() = exp10.(range(-4, log10(20), length=50))

"Pure-dephasing rates of the dephasing sweep: 0 and 16 log-spaced values in 0.01 ... 30 gamma_c."
dephasing_grid(gamma_c=GAMMA_C_DEFAULT) =
    vcat(0.0, gamma_c .* exp10.(range(log10(0.01), log10(30.0), length=16)))

"""
    next_cutoff(cutoff, max_cutoff)

Photon-cutoff ladder used by every adaptive solver: the number of photon
states grows by ~50% per retry (30 -> 45 -> 68 -> ...).
"""
next_cutoff(cutoff, max_cutoff) =
    min(max_cutoff, max(cutoff + 1, ceil(Int, 1.5 * (cutoff + 1)) - 1))
