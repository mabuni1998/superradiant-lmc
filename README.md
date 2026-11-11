# SuperradiantLMC

Code and data for the paper

> M. Bundgaard-Nielsen and J. Mørk, *Superradiant photon bunching in nanolasers: from a polarization-order hierarchy of the master equation to Dicke Markov chains* (2026). <!-- TODO: journal / arXiv identifier -->

The repository contains every solver compared in the paper and one driver script per figure. The drivers regenerate the data files in `data/`, and a single Python script turns these into the figures. The results used in the paper are included, so the figures can be reproduced without rerunning any calculation.

## Model and methods

All methods treat the same laser master equation: $n_0$ identical two-level emitters coupled to one cavity mode with coupling $g$, cavity loss $\gamma_c$, and local incoherent pump $\gamma_P$, decay $\gamma_A$, and pure dephasing $\gamma_D$ (Lindblad operator $\sigma_i^\dagger\sigma_i$). Rates are in ps$^{-1}$ and times in ps. The paper uses $g = 0.1$ and $\gamma_c = 0.04$ throughout.

| Method (Table I of the paper) | State space | Solution | Code |
|:--|:--|:--|:--|
| Exact ME, symmetry-reduced | zero-charge sector $\rho_{n_p j m;\,n_p' j m'}$, all polarization orders | direct | `solve_exact_me`, [src/master_equation.jl](src/master_equation.jl) |
| ME truncated at polarization order $B$ | as above, $\lvert n_p - n_p'\rvert \le B$ | direct | `solve_polarization_order`, [src/master_equation.jl](src/master_equation.jl) |
| Population–polarization model | populations + polarization ($b \le 1$), $b = 2$ eliminated | direct | `solve_population_polarization`, [src/population_polarization.jl](src/population_polarization.jl) |
| Dicke LMC | $(n_p, j, m)$, classical | direct | `solve_dicke_lmc`, [src/dicke_lmc.jl](src/dicke_lmc.jl) |
| Dicke LMC | $(n_p, j, m)$, classical | GFRM | `gfrm_dicke_lmc`, [src/gfrm.jl](src/gfrm.jl) |
| Conventional LMC | $(n_p, n_e)$, classical | direct | `solve_conventional_lmc`, [src/conventional_lmc.jl](src/conventional_lmc.jl) |
| Cumulant expansion (QuantumCumulants.jl) | moments up to order $M$ | time integration / Newton | [cumulant/](cumulant/) |
| Exact ME, PIQS (QuTiP) | full Dicke + photon density matrix | direct | [python/piqs_reference.py](python/piqs_reference.py) |

"Direct" means that the stationary state is found from the sparse generator by shifted inverse iteration (`stationary_state` in [src/linalg.jl](src/linalg.jl)). The photon cutoff of every direct solver is enlarged until the stationary population at the cutoff is below $10^{-10}$. Delayed correlations $g^{(2)}(\tau)$ are available for all direct solvers (`*_g2_tau`), by classical regression for the Markov chains and by the quantum regression theorem for the master equation and the population–polarization model.

## Installation

The core package (`src/`) depends only on Julia's standard library.

```
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Tested with Julia 1.12.2. The cumulant expansion has its own environment, pinned to the versions used for the paper (QuantumCumulants 0.4.3, ModelingToolkit 10.32.1, OrdinaryDiffEq 6.103.0), because its symbolic derivation is sensitive to the versions of the symbolic stack:

```
julia --project=cumulant -e "using Pkg; Pkg.instantiate()"
```

Plotting needs Python 3 with NumPy and Matplotlib; the PIQS cross-check also needs QuTiP 5:

```
pip install -r python/requirements.txt
```

## Quick start

```julia
using SuperradiantLMC

p = LaserParams(10)                      # n0 = 10, gamma_P = 1e-4, gamma_A = gamma_D = 0
solve_exact_me(p).g2                     # 20.04, numerically exact
solve_population_polarization(p).g2      # 20.10
solve_dicke_lmc(p).g2                    # 16.07
solve_conventional_lmc(p).g2             # 1.66
gfrm_dicke_lmc(p, 10^8).g2               # samples the Dicke LMC: 16.07 within sampling error

q = LaserParams(10; gamma_P=2.0, gamma_D=0.01)   # any other operating point
```

`julia --project=. -e "using Pkg; Pkg.test()"` checks these numbers and a few sum rules (about 30 s).

## Reproducing the figures

Each driver writes one tab-separated file to `data/`. Long-format files have the columns `method, x, n, g2`, with the method keys `me` (exact ME), `pp` (population–polarization model), `lmc` (Dicke LMC, direct), `gfrm` (Dicke LMC, GFRM), and `conv` (conventional LMC). Options are passed as `key=value`, e.g. `methods=lmc,conv` or `n0=8`, and a driver run for a subset of methods keeps the rows of the other methods.

| Figure | Command | Data file(s) | Cost |
|:--|:--|:--|:--|
| 2, polarization-order hierarchy | `julia --project=. scripts/fig2_polarization_order.jl` | `fig2_polarization_order_n010.tsv` | ~1 min |
| 3, pump sweep | `julia --project=. scripts/fig3_pump_sweep.jl` | `fig3_pump_sweep_n010.tsv` | ~20 min |
| 4, emitter number | `julia --project=. scripts/fig4_emitters.jl` | `fig4_emitters_P0.0001.tsv` | hours, see below |
| 5, pure dephasing | `julia --project=. scripts/fig5_dephasing.jl` | `fig5_dephasing_n010.tsv` | ~3 min |
| 6, $g^{(2)}(\tau)$ | `julia --project=. scripts/fig6_g2_tau.jl` | `fig6_g2_tau_n010.tsv` | ~4 min |
| 7, cost and accuracy | `julia --project=. scripts/fig7_timing.jl gamma_P=1e-4` (and `gamma_P=2.0`) | `fig7_timing_P*.tsv` | HPC, see below |
| 7, GFRM series | `julia --project=. scripts/gfrm_timing.jl gamma_P=1e-4` (and `gamma_P=2.0`) | `gfrm_timing_P*.tsv` | hours |
| 7, PIQS series | `python python/piqs_reference.py --gamma-p 1e-4` (and `--gamma-p 2 --n0-max 5`) | `piqs_timing_P*.tsv` | ~10 min |
| 2–7, cumulant expansion | `julia --project=cumulant cumulant/run_cumulant.jl [orders] [pump] ...` | `cumulant_*.tsv` | see [cumulant/run_cumulant.jl](cumulant/run_cumulant.jl) |

Then

```
python python/plot_figures.py            # writes figures/fig2_*.pdf ... fig7_*.pdf
```

Figure 1 is a schematic and is not generated here. Runtimes are for one laptop core (Julia 1.12, after compilation). Rerunning the drivers of Figs. 2, 3, 5, and 6 reproduces the shipped data to all ten stored digits.

**Large systems.** The default ranges of Figs. 4 and 7 are those of the paper, and the largest points need a large machine. The paper's Fig. 7 timings were measured on one HPC node (Intel Xeon Gold 6342, up to 300 GB of memory), one method per job, with `OPENBLAS_NUM_THREADS=1`. For instance, the exact ME at $n_0 = 34$ (below threshold) and $n_0 = 24$ (above) is limited by memory, the population–polarization model reaches $n_0 = 104$ and $41$, and the conventional LMC at $n_0 = 1256$ above threshold has $4\times10^7$ states. Restrict the ranges on smaller machines, e.g.

```
julia --project=. scripts/fig4_emitters.jl me_n0=1,2,3,4,5,6,7,8,9,10 methods=me
julia --project=. scripts/fig7_timing.jl gamma_P=2.0 methods=lmc,pp lmc_n0=1,2,5,10,20
```

A scan stops at the first point that exceeds the dimension cap (`dim_cap`) or fails, since larger $n_0$ will not fit either. Wall times are machine dependent; all other numbers in the data files are not, up to the sampling error of GFRM.

**Cumulant expansion.** The fourth-order equations are derived once (about 30 s plus compilation) and cached in `data/cumulant_cache/`. Orders $M = 5$ and $6$ take minutes, $M = 7$ about 30 min, and $M = 8$ about 15 h to derive on one core (`orders=7,8`). For $M \ge 5$ the fixed point at the test case is unstable and is found with Newton's method.

## Repository layout

```
src/            SuperradiantLMC package (Julia, standard library only)
test/           regression tests against the paper's numbers
scripts/        one driver per figure (+ GFRM timing)
cumulant/       cumulant expansion (separate, pinned Julia environment)
python/         plotting, plot styles, PIQS cross-check
data/           results used in the paper (tab-separated)
figures/        figures generated from data/
```

## Conventions

* Dicke states are stored as integers $(n_p, 2j, 2m)$, so that odd $n_0$ needs no half-integers; the number of excited emitters is $n_e = n_0/2 + m$.
* The events of the Dicke LMC and their order are listed in `DICKE_EVENTS` ([src/dicke_lmc.jl](src/dicke_lmc.jl)) and match the event table of the paper. The radiative rate is $\gamma_r = 4g^2/\gamma_\perp$ with $\gamma_\perp = \gamma_c + \gamma_A + \gamma_P + \gamma_D$.
* Polarization order $b = \lvert n_p - n_p'\rvert$: $b = 0$ populations, $b = 1$ optical polarization, $b = 2$ two-photon coherences. Truncating at $B = \min(n_{\max}, n_0)$ is exact.
* The stationary populations of the population–polarization model can be negative at the $10^{-6}$ level (signed effective rates, see the paper); the low-order moments are unaffected.


## Citation

If you use this code, please cite the paper above (see [CITATION.cff](CITATION.cff)).

## License

MIT, see [LICENSE](LICENSE).
