"""Figuras de la penalización asimétrica: LightGBM con pérdida cuantil y corrimientos conformales.

Color: el modelo actual (media) en gris; los niveles τ en una escala de naranja
a rojo oscuro, de 0,5 a 0,9. Los modelos cuantil van con línea llena y marcador
lleno; los corrimientos del modelo actual, con línea cortada y marcador hueco.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator

from .plots_exp import INK, INK2, WHITE, WIDTH, Lang, _row_tag
from .plots_pred import MODEL_STYLE, _day_series, _hours_axis, _legend_top, _pct, _threshold
from .prediccion import HORIZONS

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


def _handles(keys, name) -> list:
    return [Line2D([], [], label=name(k), **style(k)) for k in keys]


def fig_day(det: pd.DataFrame, mids: dict, grid_day: pd.DataFrame, title: str, L: Lang, name, horizons=(15, 30, 60)):
    """Un día: lo medido y lo pronosticado para cada instante con el modelo actual y con penalizaciones asimétricas."""
    keys = _order(mids)
    n = len(horizons)
    height = 0.8 + 1.35 * n
    fig, axes = plt.subplots(n, 1, figsize=(WIDTH, height), sharex=True, squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.99, top=1 - 0.7 / height, bottom=0.4 / height, hspace=0.14)
    hx, hy = _day_series(grid_day)
    win = (hx >= 7.0) & (hx <= 18.6)
    tmin = det["minuto"].to_numpy(float)
    for r, h in enumerate(horizons):
        ax = axes[r, 0]
        j = HORIZONS.index(h)
        x = (tmin + h) / 60.0
        y = det["co2"].to_numpy(float) + det[f"y{h}"].to_numpy(float)
        ax.plot(hx[win], hy[win], color=INK, lw=1.0, zorder=3)
        texts = []
        for k in keys:
            st = style(k, lw=1.0)
            ax.plot(x, mids[k][:, j], color=st["color"], ls=st["ls"], lw=1.0, zorder=4)
            e = mids[k][:, j] - y
            texts.append(f"{name(k).split(' (')[0]}: MAE {L.num(np.mean(np.abs(e)), 0)}, corto el {L.num(100 * np.mean(e < 0), 0)} %")
        _threshold(ax, label=(r == 0))
        _row_tag(ax, f"Pronóstico hecho {h} min antes")
        ax.text(0.985, 0.95, "\n".join(texts), transform=ax.transAxes, ha="right", va="top", fontsize=6.2, color=INK2, linespacing=1.3, bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.0, "alpha": 0.85})
        _hours_axis(ax)
        ax.yaxis.set_major_formatter(L.tick)
        ax.set_ylabel("CO₂ (ppm)", fontsize=7)
        ax.set_ylim(min(350.0, float(np.nanmin(hy[win])) - 50), 1.3 * float(np.nanmax(hy[win])))
    handles = [Line2D([], [], color=INK, lw=1.0, label="CO₂ medido")] + [Line2D([], [], color=style(k)["color"], ls=style(k)["ls"], lw=1.0, label=name(k)) for k in keys]
    _legend_top(fig, handles, ncol=len(handles), y=0.995, fontsize=7)
    fig.suptitle(title, x=0.085, ha="left", y=1 - 0.36 / height, fontsize=8.5, fontweight="semibold")
    return fig


def fig_metrics(metrics: pd.DataFrame, L: Lang, name):
    keys = _order(metrics["variante"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.1))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.7, bottom=0.15, wspace=0.32)
    for k in keys:
        d = metrics[metrics["variante"] == k].sort_values("horizonte")
        st = style(k, lw=1.1, ms=3.2)
        axes[0].plot(d["horizonte"], d["mae"], **st)
        axes[1].plot(d["horizonte"], d["frac_debajo"], **st)
        axes[2].plot(d["horizonte"], d["sesgo"], **st)
    for ax, title in zip(axes, ("Error absoluto medio (ppm)", "Veces que se queda corto", "Sesgo medio (ppm)")):
        ax.set_title(title)
        ax.set_xlim(-1, 62)
        ax.xaxis.set_major_locator(FixedLocator([1, 15, 30, 45, 60]))
        ax.set_xlabel("Horizonte (min)")
    axes[0].set_ylim(bottom=0)
    axes[0].yaxis.set_major_formatter(L.tick)
    axes[1].axhline(0.5, color=INK2, lw=0.7)
    axes[1].set_ylim(0, 1)
    _pct(axes[1])
    axes[2].axhline(0, color=INK2, lw=0.7)
    axes[2].yaxis.set_major_formatter(L.tick)
    _legend_top(fig, _handles(keys, name), ncol=3, y=0.995, fontsize=6.8)
    return fig


def fig_threshold(thr: pd.DataFrame, trade: pd.DataFrame, L: Lang, name, leads=(5, 15, 30)):
    """Anticipación contra minutos de alarma en falso: cada pronóstico como un punto, sobre la curva del clasificador."""
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


def fig_timing(timing: pd.DataFrame, L: Lang, name, keys=("media", "q50", "q70", "q80", "q90", "c70")):
    keys = [k for k in _order(keys) if k in set(timing["variante"])]
    blocks = sorted(set(timing["bloque"]), key=lambda s: int(s.split("-")[0]))
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.76, bottom=0.15)
    width = 0.8 / len(keys)
    for i, k in enumerate(keys):
        d = timing[timing["variante"] == k].set_index("bloque").reindex(blocks)
        x = np.arange(len(blocks)) - 0.4 + width * (i + 0.5)
        st = style(k, ms=4.0)
        ax.vlines(x, d["q1"], d["q3"], color=st["color"], lw=2.2)
        ax.plot(x, d["mediana"], ls="", marker="o", ms=4.0, mfc=st["mfc"], mec=st["color"], mew=0.9, color=st["color"])
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels(blocks)
    ax.set_xlabel("Faltaban para el cruce (min)")
    ax.set_ylabel("Pronosticado − medido (min)")
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    _legend_top(fig, _handles(keys, name), ncol=3, y=0.995, fontsize=6.8)
    return fig
