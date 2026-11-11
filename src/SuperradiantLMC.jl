"""
    SuperradiantLMC

Solvers for the superradiant nanolaser studied in the paper "Superradiant
photon bunching in nanolasers: from a polarization-order hierarchy of the
master equation to Dicke Markov chains".

All methods solve (approximations of) the same laser master equation for
`n0` identical two-level emitters coupled to one cavity mode, with cavity
loss `gamma_c`, local incoherent pump `gamma_P`, local decay `gamma_A`, and
local pure dephasing `gamma_D`:

| Method (paper name)            | Entry point                        |
|:-------------------------------|:-----------------------------------|
| Exact ME (symmetry-reduced)    | `solve_exact_me`                   |
| ME truncated at order `B`      | `solve_polarization_order`         |
| Population--polarization model | `solve_population_polarization`    |
| Dicke LMC, direct              | `solve_dicke_lmc`                  |
| Dicke LMC, GFRM sampling       | `gfrm_dicke_lmc`                   |
| Conventional LMC               | `solve_conventional_lmc`           |

Delayed correlations `g2(tau)` are available for all direct solvers through
the corresponding `*_g2_tau` functions.  The cumulant expansion and the PIQS
cross-check live outside this package (see `cumulant/` and `python/`).

Rates are in ps^-1 and times in ps throughout.
"""
module SuperradiantLMC

using LinearAlgebra
using SparseArrays
using Printf
using Random

include("parameters.jl")
include("linalg.jl")
include("dicke_lmc.jl")
include("conventional_lmc.jl")
include("gfrm.jl")
include("master_equation.jl")
include("population_polarization.jl")
include("io.jl")

export LaserParams, gamma_perp, gamma_r
export G_DEFAULT, GAMMA_C_DEFAULT, P_BELOW, P_ABOVE, pump_grid, dephasing_grid
export stationary_state, propagate, G2TAU_SEGMENTS
export dicke_states, dicke_rates!, DICKE_EVENTS, dicke_generator, solve_dicke_lmc, dicke_lmc_g2_tau
export conventional_generator, solve_conventional_lmc, conventional_lmc_g2_tau
export gfrm_dicke_lmc
export density_basis, symmetry_reduced_liouvillian, solve_polarization_order, solve_exact_me,
       exact_me_g2_tau
export solve_population_polarization, population_polarization_g2_tau
export write_tsv, data_path

end # module
