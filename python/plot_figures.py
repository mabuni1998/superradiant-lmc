"""Plot the result figures of the paper from the TSV files in ../data.

    python python/plot_figures.py            # all figures
    python python/plot_figures.py fig3 fig6  # selected figures

Figures are written to ../figures as PDF and PNG.  Figure numbers follow the
manuscript (Fig. 1 is the schematic and is not generated here).
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from figstyle import (RCPARAMS, HALFWIDTH, COLOR, MARKER, LINESTYLE, LABEL, style,
                      common_legend, guide_lines, finish_axes)

mpl.rcParams.update(RCPARAMS)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGDIR = os.path.join(HERE, "..", "figures")


# ---------------------------------------------------------------- data access

def load(name):
    """Structured array from a TSV file with '#' comment lines and a header."""
    path = os.path.join(DATA, name)
    with open(path, encoding="utf-8") as f:
        skip = 0
        while f.readline().startswith("#"):
            skip += 1
    return np.atleast_1d(np.genfromtxt(path, skip_header=skip, names=True, dtype=None,
                                       encoding="utf-8", delimiter="\t"))


def method(table, key, x="x"):
    """Rows of one method of a long-format table, sorted along `x`."""
    rows = table[table["method"] == key]
    return rows[np.argsort(rows[x])]


def load_cumulant(name):
    """Cumulant results, keeping only certified fixed points (status 0)."""
    d = load(name)
    return d[d["status"] == 0] if "status" in d.dtype.names else d


def save(fig, stem):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, stem + ".pdf"))
    fig.savefig(os.path.join(FIGDIR, stem + ".png"), dpi=250)
    plt.close(fig)
    print("saved", stem)


def panel_label(ax, text, x=0.02, y=0.92):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10, fontweight="bold")


# ---------------------------------------------------------------- figures

def fig2():
    """Polarization-order hierarchy versus cumulant order, n0 = 10, gamma_P = 1e-4."""
    me = load("fig2_polarization_order_n010.tsv")
    B, g2 = me["B"], me["g2"]
    exact = g2[-1]                          # B = n0 keeps every element: exact
    cum = load("cumulant_orders_n010.tsv")
    orders = np.concatenate(([2, 3], cum["M"]))
    cum_g2 = np.concatenate(([2.0, 2.0], cum["g2"]))   # M <= 3: g2 = 2 exactly
    unstable = np.concatenate(([False, False], cum["growth"] > 0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(HALFWIDTH, 4.4), sharex=True,
                                   constrained_layout=True,
                                   gridspec_kw={"height_ratios": [1.6, 1.0]})
    cum_kw = dict(color=COLOR["cumulant"], marker=MARKER["cumulant"], ms=6, mew=1.4,
                  ls=LINESTYLE["cumulant"], lw=1.0)
    ring_kw = dict(color=COLOR["cumulant"], marker="o", ms=11, mfc="none", mew=0.9, ls="none")

    ax1.plot(B, g2, "o-", color=COLOR["me"], ms=5, label=r"ME truncated at order $B$")
    ax1.plot(orders, cum_g2, label="Cumulant expansion", **cum_kw)
    ax1.plot(orders[unstable], cum_g2[unstable], label="Unstable fixed point", **ring_kw)
    ax1.axhline(exact, color=COLOR["piqs"], lw=1.8, label="Exact ME")
    ax1.set_ylabel(r"$g^{(2)}(0)$")
    ax1.set_ylim(0, 34)
    ax1.legend(frameon=True, fontsize=8.5, loc="lower right")

    rel = np.abs(g2 - exact) / exact
    keep = rel > 0
    ax2.semilogy(B[keep], rel[keep], "o-", color=COLOR["me"], ms=5)
    cum_rel = np.abs(cum_g2 - exact) / exact
    ax2.semilogy(orders, cum_rel, **cum_kw)
    ax2.semilogy(orders[unstable], cum_rel[unstable], **ring_kw)
    ax2.set_xlabel(r"Polarization order $B$ / cumulant order $M$")
    ax2.set_ylabel(r"Rel. error in $g^{(2)}(0)$")
    ax2.set_xticks(B)
    panel_label(ax1, "(a)", 0.025, 0.85)
    panel_label(ax2, "(b)", 0.025, 0.08)
    finish_axes(ax1, ax2)
    save(fig, "fig2_polarization_order")


def fig3():
    """Pump sweep, n0 = 10."""
    d = load("fig3_pump_sweep_n010.tsv")
    cum = load_cumulant("cumulant_pump_n010.tsv")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(HALFWIDTH, 5.4), sharex=True,
                                   constrained_layout=True)
    for key in ("me", "pp", "lmc", "conv"):
        r = method(d, key)
        ax1.loglog(r["x"], r["n"], **style(key, line=(key == "me")))
        ax2.semilogx(r["x"], r["g2"], **style(key, line=(key == "me"), label=False))
    ax1.loglog(cum["x"], cum["n"], **style("cumulant", line=False))
    ax2.semilogx(cum["x"], cum["g2"], **style("cumulant", line=False, label=False))
    ax1.set_ylabel(r"Photon number $\langle n_p \rangle$")
    common_legend(ax1, loc="lower right")
    guide_lines(ax2, x_text=1, thermal=True, coherent=True, dy_thermal=1.45, dy_coherent=0.62)
    ax2.set_yscale("log")
    ax2.set_ylim(0.5, 90)
    ax2.set_xlabel(r"Pump rate $\gamma_P$ (ps$^{-1}$)")
    ax2.set_ylabel(r"$g^{(2)}(0)$")
    panel_label(ax1, "(a)")
    panel_label(ax2, "(b)")
    finish_axes(ax1, ax2)
    save(fig, "fig3_pump_sweep")


def fig4():
    """g2(0) versus emitter number, gamma_P = 1e-4."""
    d = load("fig4_emitters_P0.0001.tsv")
    cum = load_cumulant("cumulant_emitters_P0.0001.tsv")
    fig, ax = plt.subplots(figsize=(HALFWIDTH, 3.4), constrained_layout=True)
    me = method(d, "me")
    ax.plot(me["x"], me["g2"], "-", color=COLOR["me"], lw=1.8)
    for key in ("pp", "lmc", "gfrm", "conv"):
        r = method(d, key)
        ax.plot(r["x"], r["g2"], **style(key, line=False, label=False))
    ax.plot(cum["x"], cum["g2"], **style("cumulant", line=False, label=False))
    ax.set_xscale("log")
    ax.set_yscale("log")
    guide_lines(ax, x_text=400, thermal=True, coherent=True, dy_thermal=1.2, dy_coherent=0.7)
    ax.set_ylim(0.5, 102)
    ax.set_xlabel(r"Number of emitters $n_0$")
    ax.set_ylabel(r"$g^{(2)}(0)$")
    common_legend(ax, extra=("gfrm",), loc="upper right")
    finish_axes(ax)
    save(fig, "fig4_emitters")


def fig5():
    """g2(0) versus pure dephasing, n0 = 10, gamma_P = 1e-4 (broken x axis at 0)."""
    d = load("fig5_dephasing_n010.tsv")
    cum = load_cumulant("cumulant_dephasing_n010.tsv")
    gamma_c = 0.04
    fig, (ax0, ax) = plt.subplots(1, 2, figsize=(HALFWIDTH, 3.6), sharey=True,
                                  constrained_layout=True,
                                  gridspec_kw=dict(width_ratios=[1, 11]))
    fig.get_layout_engine().set(w_pad=0.0, wspace=0.0)

    for key in ("me", "pp", "lmc", "conv"):
        r = method(d, key)
        pos, zero = r["x"] > 0, r["x"] == 0
        ax.plot(r["x"][pos], r["g2"][pos], **style(key, line=(key == "me"), label=False))
        if key == "me":     # the gamma_D = 0 value as a short red dash
            kw = style("me", label=False)
            ax0.plot(np.zeros(zero.sum()), r["g2"][zero], ls="none", marker="_", ms=11,
                     mew=kw["lw"], color=kw["color"])
        else:
            ax0.plot(np.zeros(zero.sum()), r["g2"][zero], **style(key, line=False, label=False))
    pos, zero = cum["x"] > 0, cum["x"] == 0
    ax.plot(cum["x"][pos], cum["g2"][pos], **style("cumulant", line=False, label=False))
    ax0.plot(np.zeros(zero.sum()), cum["g2"][zero], **style("cumulant", line=False, label=False))

    ax.axvline(gamma_c, color="black", lw=1.0, ls="--", zorder=0)
    guide_lines(ax, x_text=0.1, thermal=True, coherent=True, dy_thermal=1.05, dy_coherent=1.05)
    for y in (1.0, 2.0):
        ax0.axhline(y, color="black", lw=1.0, ls="--", zorder=0)
    ax.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(-1.0, 1.0)
    ax0.set_xticks([0.0])
    ax0.set_xticklabels(["0"])
    ax.set_xlim(left=2e-4)
    ax.set_xlabel(r"Pure dephasing rate $\gamma_D$ (ps$^{-1}$)")
    ax0.set_ylabel(r"$g^{(2)}(0)$")
    ax.set_title(r"$n_0 = 10$, $\gamma_P = 10^{-4}\,\mathrm{ps}^{-1}$", fontsize=10)
    for s in ("top", "right"):
        ax0.spines[s].set_visible(False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", which="both", left=False)
    brk = dict(marker=[(-1, -1.5), (1, 1.5)], markersize=7, linestyle="none", color="black",
               mec="black", mew=0.8, clip_on=False)
    ax0.plot([1], [0], transform=ax0.transAxes, **brk)
    ax.plot([0], [0], transform=ax.transAxes, **brk)
    common_legend(ax, loc="upper right")
    save(fig, "fig5_dephasing")


def fig6():
    """Delayed correlation g2(tau), n0 = 10, gamma_P = 1e-4."""
    d = load("fig6_g2_tau_n010.tsv")
    cum = load_cumulant("cumulant_g2_tau_n010.tsv")
    assert np.allclose(cum["tau"], d["tau"]), "cumulant g2(tau) grid differs"
    sel = d["tau"] > 0
    taus = d["tau"][sel]
    # log-spaced markers on the piecewise-uniform grid
    marks = np.unique(np.searchsorted(taus, np.geomspace(taus[0], taus[-1], 60)))
    marks = list(marks[marks < len(taus)])

    fig, ax = plt.subplots(figsize=(HALFWIDTH, 3.4), constrained_layout=True)
    ax.semilogx(taus, d["me"][sel], **style("me"))
    for key in ("pp", "lmc", "conv"):
        ax.semilogx(taus, d[key][sel], **style(key, line=False, markevery=marks))
    ax.semilogx(taus, cum["g2"][sel], **style("cumulant", line=False, markevery=marks))
    guide_lines(ax, x_text=0.06, thermal=True, coherent=True, dy_thermal=1.25, dy_coherent=-0.7)
    ax.set_xlabel(r"Delay $\tau$ (ps)")
    ax.set_ylabel(r"$g^{(2)}(\tau)$")
    ax.set_xlim(0.05, 5e4)
    ax.set_ylim(-4.5, 32)
    common_legend(ax, loc="upper right")
    finish_axes(ax)
    save(fig, "fig6_g2_tau")


def fig7(gfrm_tolerance=1e-3):
    """Wall-clock time (top) and error in g2(0) (bottom), below and above threshold."""
    fig, axes = plt.subplots(2, 2, figsize=(2 * HALFWIDTH, 5.6), constrained_layout=True,
                             sharex="col")
    for col, (P, title) in enumerate((
            ("0.0001", r"Below threshold, $\gamma_P = 10^{-4}\,\mathrm{ps}^{-1}$"),
            ("2.0", r"Above threshold, $\gamma_P = 2\,\mathrm{ps}^{-1}$"))):
        t = load(f"fig7_timing_P{P}.tsv")
        gfrm = load(f"gfrm_timing_P{P}.tsv")
        piqs = load(f"piqs_timing_P{P}.tsv")
        cum = load_cumulant(f"cumulant_timing_P{P}.tsv")
        series = {key: method(t, key, x="n0") for key in ("lmc", "conv", "me", "pp")}

        ax = axes[0, col]
        for key, fmt in (("lmc", "o-"), ("conv", "+-."), ("pp", "^-")):
            r = series[key]
            ax.loglog(r["n0"], r["walltime"], fmt, color=COLOR[key], ms=4, mfc="none")
        ax.loglog(series["me"]["n0"], series["me"]["walltime"], "-", color=COLOR["me"], lw=1.8)
        ax.loglog(gfrm["n0"], gfrm["walltime"], "s-", color=COLOR["gfrm"], ms=4, mfc="none")
        ax.loglog(piqs["n0"], piqs["walltime"], "d-", color=COLOR["piqs"], ms=4, mfc="none")
        ax.loglog(cum["n0"], cum["walltime"], **style("cumulant"), marker=MARKER["cumulant"],
                  ms=5, mew=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Wall-clock time (s)" if col == 0 else "")
        ax.grid(True, which="major", alpha=0.2)
        ax.tick_params(axis="x", which="both", labelbottom=True)
        panel_label(ax, f"({'ab'[col]})", 0.03, 0.94)

        exact = {int(n0): g2 for n0, g2 in zip(series["me"]["n0"], series["me"]["g2"])}

        def deviation(n0s, g2s, extra=0.0):
            pts = [(n0, abs(g2 - exact[int(n0)]) / exact[int(n0)] + extra)
                   for n0, g2 in zip(n0s, g2s) if int(n0) in exact]
            return np.array(pts).T if pts else (np.array([]), np.array([]))

        ax = axes[1, col]
        for key, fmt in (("lmc", "o-"), ("conv", "+-."), ("pp", "^-")):
            x, y = deviation(series[key]["n0"], series[key]["g2"])
            ax.loglog(x, y, fmt, color=COLOR[key], ms=4, mfc="none")
        # GFRM samples the Dicke LMC: model error plus the sampling tolerance
        x, y = deviation(gfrm["n0"], gfrm["g2_ref"], gfrm_tolerance)
        ax.loglog(x, y, "s-", color=COLOR["gfrm"], ms=4, mfc="none")
        x, y = deviation(cum["n0"], cum["g2"])
        ax.loglog(x, y, **style("cumulant"), marker=MARKER["cumulant"], ms=5, mew=1.0)
        ax.set_xlabel(r"Number of emitters $n_0$")
        ax.set_ylabel(r"$|g^{(2)} - g^{(2)}_{\rm ME}| / g^{(2)}_{\rm ME}$" if col == 0 else "")
        ax.grid(True, which="major", alpha=0.2)
        if col == 1:
            ax.set_ylim(top=2.0)
        panel_label(ax, f"({'cd'[col]})", 0.03, 0.94)

    if axes[0, 1].get_xlim()[1] < 100:
        axes[0, 1].set_xlim(right=115)
    common_legend(axes[1, 0], extra=("gfrm", "piqs"), loc="lower right",
                  bbox_to_anchor=(1.0, 0.08))
    for ax in axes.flat:
        finish_axes(ax)
    save(fig, "fig7_timing")


FIGURES = {"fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7}

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGURES):
        FIGURES[name]()
