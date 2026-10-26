"""Figura 8.4: penalizar más quedarse corto (LightGBM con pérdida cuantil y corrimientos del modelo de la media).

Media en gris; los τ de naranja a rojo oscuro. Cuantil: marcador lleno; corrimiento: hueco.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .plots_exp import INK2, WHITE, WIDTH, Lang
from .plots_pred import MODEL_STYLE, _legend_top, _pct

TAU_COLOR = {50: "#f5b48c", 60: "#ee8a55", 70: "#de6128", 80: "#b4461a", 90: "#7a2c10"}
CLF_COLOR = MODEL_STYLE["lightgbm"]["color"]


def style(key: str, lw: float = 1.3, ms: float = 3.8) -> dict:
    if key == "media":
        return {"color": INK2, "ls": "-", "lw": lw, "marker": "o", "ms": ms, "mfc": INK2, "mec": WHITE, "mew": 0.5}
    tau = int(key[1:])
    color = TAU_COLOR[tau]
    if key[0] == "q":
        return {"color": color, "ls": "-", "lw": lw, "marker": "o", "ms": ms, "mfc": color, "mec": WHITE, "mew": 0.5}
    return {"color": color, "ls": (0, (3, 1.5)), "lw": lw, "marker": "o", "ms": ms, "mfc": WHITE, "mec": color, "mew": 0.9}


def _order(keys) -> list:
    rank = lambda k: (0, 0) if k == "media" else ((1, int(k[1:])) if k[0] == "q" else (2, int(k[1:])))  # noqa: E731
    return sorted(keys, key=rank)


def fig_threshold(thr: pd.DataFrame, trade: pd.DataFrame, L: Lang, name, leads=(5, 15, 30)):
    """Anticipación contra minutos de alarma en falso: cada variante es un punto sobre la curva del clasificador."""
    keys = _order(thr["variante"].unique())
    clf = trade[trade["modelo"] == "lightgbm_clasif"].sort_values("p_min")
    fig, axes = plt.subplots(1, len(leads), figsize=(WIDTH, 3.2), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.7, bottom=0.15, wspace=0.08)
    for ax, lead in zip(axes, leads):
        ax.plot(clf["minutos_falsos_por_dia"], clf[f"anticipados_{lead}"], color=CLF_COLOR, lw=1.3, zorder=2)
        for p in (0.3, 0.5, 0.8):
            r = clf[np.isclose(clf["p_min"], p)]
            ax.annotate(f"p ≥ {L.g(p)}", (r["minutos_falsos_por_dia"].iloc[0], r[f"anticipados_{lead}"].iloc[0]), xytext=(3, -9), textcoords="offset points", fontsize=6.3, color=CLF_COLOR)
        for k in keys:
            d = thr[thr["variante"] == k]
            st = style(k, ms=5.2)
            ax.plot(d["minutos_falsos_por_dia"], d[f"anticipados_{lead}"], ls="", marker=st["marker"], ms=st["ms"], mfc=st["mfc"], mec=st["mec"] if k[0] != "c" else st["color"], mew=0.8, color=st["color"], zorder=4)
        ax.set_title(f"Anticipados con al menos {lead} min")
        ax.set_xlabel("Minutos de alarma sin cruce, por día")
        ax.set_ylim(0, 1)
        ax.set_xlim(left=0)
        _pct(ax)
        ax.xaxis.set_major_formatter(L.tick)
    axes[0].set_ylabel("Cruces")
    handles = [Line2D([], [], color=CLF_COLOR, lw=1.3, label="clasificador, barriendo p")] + [Line2D([], [], ls="", marker="o", ms=5.2, color=style(k)["color"], mfc=style(k)["mfc"], mec=style(k)["color"], label=name(k)) for k in keys]
    _legend_top(fig, handles, ncol=4, y=0.995, fontsize=6.8)
    return fig
