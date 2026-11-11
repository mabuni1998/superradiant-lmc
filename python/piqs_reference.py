"""Independent cross-check of the exact ME with PIQS (QuTiP), and its cost.

Builds the full permutationally invariant Liouvillian with qutip.piqs (local
pump, decay, and dephasing via the Dicke class), tensors it with the cavity,
adds the Tavis--Cummings coupling, and solves for the steady state by the
same shifted inverse iteration (sparse LU) as the Julia solvers, so the
timing comparison isolates the size of the formulation.  (qutip's direct
steady-state solver hits an exactly singular factor here, and its iterative
solvers fail silently deep below threshold.)

    python python/piqs_reference.py [--n0-max 10] [--gamma-p 1e-4] [--nph 30]

The number of Fock states per n0 is taken from the converged photon cutoff
of the Dicke-LMC rows in data/fig7_timing_P<gamma_P>.tsv when available
(this favors PIQS, which gets the converged cutoff for free), else --nph.
Writes data/piqs_timing_P<gamma_P>.tsv.  The sweep stops at the first
MemoryError.

At the paper's test case (n0 = 10, gamma_P = 1e-4) PIQS reproduces the
symmetry-reduced exact ME to 2e-8 in g2(0).
"""

import argparse
import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from qutip import (Qobj, destroy, qeye, tensor, spre, spost, to_super, super_tensor,
                   expect, liouvillian, operator_to_vector, vector_to_operator)
from qutip.piqs.piqs import Dicke, jspin, num_dicke_states

G = 0.1         # light-matter coupling g (1/ps)
GAMMA_C = 0.04  # cavity loss (1/ps)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def julia_float(x):
    """Format a float like Julia's string(x), as used in the data file names."""
    s = repr(float(x))
    return s if ("." in s or "e" in s) else s + ".0"


def lmc_photon_states(gamma_p):
    """Converged number of Fock states per n0 from the Dicke-LMC timing rows."""
    path = os.path.join(DATA, f"fig7_timing_P{julia_float(gamma_p)}.tsv")
    if not os.path.isfile(path):
        return {}
    nph = {}
    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip("\n").split("\t") for l in f if not l.startswith("#")]
    cols = lines[0]
    for row in lines[1:]:
        r = dict(zip(cols, row))
        if r["method"] != "lmc":
            continue
        n0, dim = int(r["n0"]), int(r["dim"])
        dicke_dim = sum(j2 + 1 for j2 in range(n0, -1, -2))   # number of (j, m) pairs
        nph[n0] = dim // dicke_dim
    return nph


def solve_piqs(n0, nph, gamma_p):
    """PIQS steady state with local pump only (gamma_A = gamma_D = 0, as in the paper)."""
    t0 = time.perf_counter()
    ensemble = Dicke(n0)
    ensemble.pumping = gamma_p
    L_tls = ensemble.liouvillian()

    nds = num_dicke_states(n0)
    a = destroy(nph)
    L_cav = liouvillian(0 * a.dag() * a, [np.sqrt(GAMMA_C) * a])
    L = super_tensor(L_cav, to_super(qeye(nds))) + super_tensor(to_super(qeye(nph)), L_tls)
    H = G * (tensor(a.dag(), jspin(n0, "-")) + tensor(a, jspin(n0, "+")))
    L += -1j * spre(H) + 1j * spost(H)

    # shifted inverse iteration on the vectorized Liouvillian
    M = L.data.as_scipy().tocsc()
    dim = M.shape[0]
    sigma = 1e-10 * np.abs(M.diagonal()).max()
    lu = splu(sigma * sp.identity(dim, format="csc") - M)
    ident = tensor(qeye(nph), qeye(nds))
    template = operator_to_vector(ident / (nph * nds))
    trace = operator_to_vector(ident).full().ravel().conj()
    x = template.full().ravel()
    residual = np.inf
    for _ in range(200):
        x = lu.solve(x)
        x /= trace @ x
        residual = np.abs(M @ x).max()
        if residual < 1e-12:
            break
    if residual > 1e-9:
        raise RuntimeError(f"stationary residual {residual} too large")
    rho = vector_to_operator(Qobj(x[:, None], dims=template.dims))

    nbar = expect(tensor(a.dag() * a, qeye(nds)), rho)
    g2 = expect(tensor(a.dag() * a.dag() * a * a, qeye(nds)), rho) / nbar**2
    edge = np.real(rho.ptrace(0).diag())[-1]     # photon population at the cutoff
    return time.perf_counter() - t0, dim, nbar, g2, edge


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n0-max", type=int, default=10)
    parser.add_argument("--gamma-p", type=float, default=1e-4)
    parser.add_argument("--nph", type=int, default=30, help="Fock states if no LMC cutoff is stored")
    args = parser.parse_args()

    nph_by_n0 = lmc_photon_states(args.gamma_p)
    solve_piqs(1, 10, args.gamma_p)                    # warm-up (imports, BLAS)
    out = os.path.join(DATA, f"piqs_timing_P{julia_float(args.gamma_p)}.tsv")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# PIQS (full permutationally invariant Liouvillian) steady state, "
                f"gamma_P={args.gamma_p}, gamma_A=gamma_D=0\n")
        f.write("method\tn0\tdim\twalltime\tn\tg2\tedge\n")
        for n0 in range(1, args.n0_max + 1):
            nph = nph_by_n0.get(n0, args.nph)
            try:
                walltime, dim, nbar, g2, edge = solve_piqs(n0, nph, args.gamma_p)
            except MemoryError:
                print(f"n0={n0}: MemoryError, stopping")
                break
            f.write(f"piqs\t{n0}\t{dim}\t{walltime:.4g}\t{nbar:.8g}\t{g2:.8g}\t{edge:.3g}\n")
            f.flush()
            print(f"n0={n0}: dim={dim} nph={nph} t={walltime:.3g} s n={nbar:.6g} g2={g2:.8g} "
                  f"edge={edge:.2g}", flush=True)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
