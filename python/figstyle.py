"""Shared per-method plot styles: one method = one color = one marker.

    me        Exact ME                        red         thick solid line
    pp        Population-polarization model   green       open triangles
    lmc       Dicke LMC                       black       open circles
    conv      Conventional LMC                blue        plus markers
    gfrm      Dicke LMC (GFRM)                darkorange  open squares
    piqs      Exact ME (PIQS)                 purple      open diamonds
    cumulant  Cumulant expansion              brown       crosses

Except in the timing figure, approximate methods are drawn as symbols only.
"""

RCPARAMS = {
    "font.size": 10,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
}

HALFWIDTH = 4.5

COLOR = {"me": "tab:red", "pp": "tab:green", "lmc": "black", "conv": "tab:blue",
         "gfrm": "darkorange", "piqs": "tab:purple", "cumulant": "tab:brown"}
MARKER = {"me": None, "pp": "^", "lmc": "o", "conv": "+", "gfrm": "s", "piqs": "d",
          "cumulant": "x"}
LINESTYLE = {"me": "-", "pp": "-", "lmc": "--", "conv": "-.", "gfrm": "-", "piqs": ":",
             "cumulant": (0, (5, 1))}
LABEL = {"me": "Exact ME", "pp": "Population–polarization model", "lmc": "Dicke LMC",
         "conv": "Conventional LMC", "gfrm": "Dicke LMC (GFRM)", "piqs": "Exact ME (PIQS)",
         "cumulant": "Cumulant expansion"}

LEGEND_METHODS = ("me", "pp", "lmc", "conv", "cumulant")


def style(method, line=True, markevery=None, label=True):
    """ax.plot keyword arguments for a method."""
    kw = dict(color=COLOR[method])
    if label:
        kw["label"] = LABEL[method]
    marker = MARKER[method]
    if line:
        kw.update(ls=LINESTYLE[method], lw=1.8 if method == "me" else 1.2)
        if marker is not None and markevery is not None:
            kw.update(marker=marker, markevery=markevery)
    else:
        kw["ls"] = "none"
        if marker is not None:
            kw["marker"] = marker
            if markevery is not None:
                kw["markevery"] = markevery
    if kw.get("marker") is not None:
        if kw["marker"] in ("+", "x"):
            kw.update(ms=6.0, mew=1.2)
        else:
            kw.update(ms=4.2, mfc="none", mew=1.0)
    return kw


def common_legend(ax, extra=(), **kwargs):
    """Legend shared by all result figures; `extra` appends more methods."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], **style(m, line=(m == "me")))
               for m in LEGEND_METHODS + tuple(extra)]
    kwargs.setdefault("frameon", True)
    kwargs.setdefault("fontsize", 8)
    return ax.legend(handles=handles, **kwargs)


def guide_lines(ax, x_text, thermal=True, coherent=False, fontsize=10,
                dy_thermal=1.15, dy_coherent=0.8):
    """Thermal (g2 = 2) and coherent (g2 = 1) reference lines."""
    if thermal:
        ax.axhline(2.0, color="black", lw=1.0, ls="--", zorder=0)
        ax.text(x_text, 2.0 * dy_thermal, "Thermal", fontsize=fontsize)
    if coherent:
        ax.axhline(1.0, color="black", lw=1.0, ls="--", zorder=0)
        ax.text(x_text, 1.0 * dy_coherent, "Coherent", fontsize=fontsize)


def finish_axes(*axes):
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
