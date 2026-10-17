"""Figuras de la importancia de las variables de LightGBM, en español y con el estilo de las de la predicción.

Color: cada grupo de variables tiene siempre el mismo tono (el orden validado
de la paleta del proyecto). En las contribuciones de un pronóstico, rojo empuja
la concentración hacia arriba y azul hacia abajo.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from .importancia import variable_group, variable_name
from .plots_exp import GRID, INK, INK2, MUTED, SEQ_BLUE, WHITE, WIDTH, Lang
from .prediccion import HORIZONS

GROUP_ORDER = ("CO₂ reciente", "hora y día", "temperatura y humedad", "historia del salón", "salón")
GROUP_COLOR = dict(zip(GROUP_ORDER, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")))
UP, DOWN = "#c43a3a", "#1c5cab"
VALUE_CMAP = LinearSegmentedColormap.from_list("valor", ["#1c5cab", "#86b6ef", "#d9d7d0", "#f2a9a8", "#c43a3a"])
METHOD_NAMES = {"ganancia": "Ganancia en los árboles", "permutacion": "Permutación en la prueba", "shap": "SHAP medio en la prueba"}


def _clock(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60}:{m % 60:02d}"


WEEKDAY_NAMES = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")


def _value_text(col: str, v: float, L: Lang, rooms=None) -> str:
    if col == "minuto":
        return _clock(v)
    if col == "salon":
        return rooms[int(v)] if rooms is not None else f"{int(v)}"
    if col == "escuela":
        return "A" if int(v) == 0 else "B"
    if col == "dia_semana":
        return WEEKDAY_NAMES[int(v)]
    if col in ("hora_sen", "hora_cos"):
        return L.num(v, 2).replace("−0,00", "0,00")
    if col.startswith(("temp", "dtemp")):
        return L.num(v, 1) + " °C"
    if col.startswith(("hum", "dhum")):
        return L.num(v, 0) + " %"
    if col.startswith(("pend", "curv")):
        return L.num(v, 1) + " ppm/min"
    return L.num(v, 0) + " ppm"


def _group_handles() -> list:
    return [Patch(facecolor=GROUP_COLOR[g], label=g) for g in GROUP_ORDER]


def _legend_top(fig, handles, ncol: int, y: float = 0.995) -> None:
    fig.legend(handles=handles, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y), frameon=False, fontsize=7.2, handlelength=1.6, columnspacing=1.4)


# ------------------------------------------------------------------ un pronóstico explicado


def fig_case(cases: list, L: Lang, top: int = 8, rooms=None):
    """Contribuciones SHAP al cambio pronosticado a 30 min desde varios orígenes de un día.

    `cases`: lista de dicts con `titulo`, `shap` (variables,), `valores` (variables,),
    `cols`, `base`, `pronostico` y `medido` (cambios en ppm).
    """
    n = len(cases)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    height = 2.45 * nrow + 0.5
    fig, axes = plt.subplots(nrow, ncol, figsize=(WIDTH, height), squeeze=False)
    fig.subplots_adjust(left=0.2, right=0.985, top=1 - 0.75 / height, bottom=0.45 / height, hspace=0.75, wspace=0.95)
    for ax, c in zip(axes.ravel(), cases):
        s = np.asarray(c["shap"], float)
        order = np.argsort(-np.abs(s))
        keep, rest = order[:top], order[top:]
        vals = list(s[keep]) + [s[rest].sum()]
        labels = [f"{variable_name(c['cols'][i])} = {_value_text(c['cols'][i], c['valores'][i], L, rooms)}" for i in keep] + [f"otras {len(rest)} variables"]
        y = np.arange(len(vals))[::-1]
        ax.barh(y, vals, color=[UP if v > 0 else DOWN for v in vals], height=0.7, edgecolor=WHITE, linewidth=0.6)
        ax.axvline(0, color=INK2, lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(L.tick)
        ax.set_xlabel("Contribución al cambio pronosticado (ppm)", fontsize=7)
        sign = lambda v: f"{v:+.0f}".replace("-", "−")  # noqa: E731
        ax.set_title(c["titulo"] + "\n" + f"pronostica {sign(c['pronostico'])} ppm, medido {sign(c['medido'])} ppm", fontsize=7.3, linespacing=1.4)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    handles = [Patch(facecolor=UP, label="empuja hacia arriba"), Patch(facecolor=DOWN, label="empuja hacia abajo")]
    _legend_top(fig, handles, ncol=2)
    return fig


# ------------------------------------------------------------------ métodos comparados


def method_table(gain: pd.DataFrame, perm: pd.DataFrame, shap: pd.DataFrame, h: int) -> pd.DataFrame:
    """Importancia de cada variable con los tres métodos, en fracción del total de cada método."""
    g = gain[(gain["horizonte"] == h) & (gain["tipo"] == "mid")].set_index("variable")["ganancia"]
    p = perm[perm["horizonte"] == h].set_index("variable")["aumento_mae"].clip(lower=0)
    s = shap[(shap["horizonte"] == h) & (shap["tipo"] == "mid")].set_index("variable")["shap_medio"]
    t = pd.DataFrame({"ganancia": g / g.sum(), "permutacion": p / p.sum(), "shap": s / s.sum()})
    return t.sort_values("shap", ascending=False)


def fig_methods(gain, perm, shap, L: Lang, h: int = 30, top: int = 20):
    t = method_table(gain, perm, shap, h).head(top)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 0.2 * top + 1.2), sharey=True)
    fig.subplots_adjust(left=0.27, right=0.99, top=1 - 0.75 / (0.2 * top + 1.2), bottom=0.45 / (0.2 * top + 1.2), wspace=0.08)
    y = np.arange(len(t))[::-1]
    colors = [GROUP_COLOR[variable_group(v)] for v in t.index]
    for ax, m in zip(axes, ("ganancia", "permutacion", "shap")):
        ax.barh(y, t[m], color=colors, height=0.72, edgecolor=WHITE, linewidth=0.5)
        ax.set_title(METHOD_NAMES[m], fontsize=7.8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f} %"))
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(0.05, t[m].max() * 1.12))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([variable_name(v) for v in t.index], fontsize=6.8)
    axes[0].tick_params(axis="y", length=0)
    fig.supxlabel("Fracción de la importancia total", fontsize=7.5, color=INK2, y=0.005)
    _legend_top(fig, _group_handles(), ncol=5)
    return fig


def fig_heatmap(shap: pd.DataFrame, L: Lang, top: int = 20, kind: str = "mid"):
    d = shap[shap["tipo"] == kind].pivot_table(index="variable", columns="horizonte", values="shap_frac")
    d = d.loc[d.mean(axis=1).sort_values(ascending=False).index[:top]]
    fig, ax = plt.subplots(figsize=(WIDTH, 0.21 * top + 0.9))
    fig.subplots_adjust(left=0.3, right=0.92, top=1 - 0.35 / (0.21 * top + 0.9), bottom=0.5 / (0.21 * top + 0.9))
    im = ax.imshow(d.to_numpy(), aspect="auto", cmap=SEQ_BLUE, vmin=0, vmax=float(d.to_numpy().max()))
    for (r, c), v in np.ndenumerate(d.to_numpy()):
        ax.text(c, r, f"{v * 100:.0f}", ha="center", va="center", fontsize=6.4, color=WHITE if v > 0.55 * d.to_numpy().max() else INK)
    ax.set_xticks(range(d.shape[1]))
    ax.set_xticklabels([str(h) for h in d.columns])
    ax.set_yticks(range(d.shape[0]))
    ax.set_yticklabels([variable_name(v) for v in d.index], fontsize=6.8)
    for lab, v in zip(ax.get_yticklabels(), d.index):
        lab.set_color(GROUP_COLOR[variable_group(v)] if variable_group(v) != "CO₂ reciente" else INK)
    ax.set_xlabel("Horizonte (min)")
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = fig.add_axes([0.935, 1 - 0.35 / (0.21 * top + 0.9) - 0.6, 0.012, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f} %"))
    return fig


# ------------------------------------------------------------------ SHAP sobre muchos orígenes


def fig_beeswarm(s: np.ndarray, X: pd.DataFrame, cols, L: Lang, top: int = 15, seed: int = 0, max_points: int = 4000):
    """Cada punto es un origen: su contribución SHAP (horizontal) coloreada por el valor de la variable."""
    rng = np.random.default_rng(seed)
    order = np.argsort(-np.abs(s).mean(axis=0))[:top]
    fig, ax = plt.subplots(figsize=(WIDTH, 0.26 * top + 1.0))
    fig.subplots_adjust(left=0.27, right=0.9, top=1 - 0.25 / (0.26 * top + 1.0), bottom=0.5 / (0.26 * top + 1.0))
    idx = rng.choice(len(s), min(max_points, len(s)), replace=False)
    for k, j in enumerate(order):
        y = top - 1 - k
        v = X.iloc[idx, j].to_numpy(float)
        lo, hi = np.nanpercentile(v, [5, 95])
        colour = np.clip((v - lo) / (hi - lo) if hi > lo else np.full(len(v), 0.5), 0, 1)
        ax.scatter(s[idx, j], y + rng.normal(0, 0.12, len(idx)), c=colour, cmap=VALUE_CMAP, vmin=0, vmax=1, s=2.2, lw=0, alpha=0.7, rasterized=True)
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_yticks(range(top))
    ax.set_yticklabels([variable_name(cols[j]) for j in order][::-1], fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Contribución SHAP al cambio pronosticado (ppm)")
    ax.xaxis.set_major_formatter(L.tick)
    sm = plt.cm.ScalarMappable(cmap=VALUE_CMAP)
    cax = fig.add_axes([0.915, 0.25, 0.012, 0.5])
    cb = fig.colorbar(sm, cax=cax)
    cb.outline.set_visible(False)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["bajo", "alto"])
    cb.ax.tick_params(labelsize=7, length=0)
    cb.set_label("valor de la variable", fontsize=7, color=INK2)
    return fig


def fig_dependence(s: np.ndarray, X: pd.DataFrame, cols, variables, L: Lang, seed: int = 0, max_points: int = 15000):
    """Contribución SHAP de una variable según su valor."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(s), min(max_points, len(s)), replace=False)
    n = len(variables)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(WIDTH, 2.1 * nrow + 0.3), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=1 - 0.25 / (2.1 * nrow + 0.3), bottom=0.12, hspace=0.6, wspace=0.3)
    for ax, var in zip(axes.ravel(), variables):
        j = list(cols).index(var)
        x = X.iloc[idx, j].to_numpy(float)
        ax.scatter(x, s[idx, j], s=2, lw=0, alpha=0.25, color=GROUP_COLOR[variable_group(var)], rasterized=True)
        ax.axhline(0, color=INK2, lw=0.7)
        ax.set_title(variable_name(var), fontsize=7.8)
        if var == "minuto":
            ax.xaxis.set_major_locator(FixedLocator([480, 600, 720, 840, 960]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _clock(v)))
        else:
            lo, hi = np.nanpercentile(x, [1, 99])
            ax.set_xlim(lo, hi)
            ax.xaxis.set_major_formatter(L.tick)
        ax.yaxis.set_major_formatter(L.tick)
    for ax in axes[:, 0]:
        ax.set_ylabel("SHAP (ppm)")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return fig


def fig_by_hour(s: np.ndarray, hours: np.ndarray, cols, L: Lang):
    """Media de |SHAP| de cada grupo de variables según la hora del origen."""
    groups = np.array([variable_group(c) for c in cols])
    fig, ax = plt.subplots(figsize=(WIDTH, 2.7))
    fig.subplots_adjust(left=0.08, right=0.975, top=0.8, bottom=0.17)
    hs = np.sort(np.unique(hours))
    for g in GROUP_ORDER:
        cols_g = groups == g
        tot = np.abs(s[:, cols_g]).sum(axis=1)
        y = [tot[hours == h].mean() for h in hs]
        ax.plot(hs + 0.5, y, color=GROUP_COLOR[g], lw=1.4, marker="o", ms=3.4, mec=WHITE, mew=0.5, label=g)
    ax.set_xlim(7, 18)
    ax.xaxis.set_major_locator(FixedLocator(range(8, 19, 2)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}:00"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Hora del origen")
    ax.set_ylabel("Media de |SHAP| (ppm)")
    ax.yaxis.set_major_formatter(L.tick)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.22), ncol=5, fontsize=7.2, frameon=False)
    return fig


# ------------------------------------------------------------------ permutación contra quitar y reentrenar


def fig_drop_vs_perm(drop: pd.DataFrame, perm: pd.DataFrame, L: Lang):
    hs = sorted(drop["horizonte"].unique())
    fig, axes = plt.subplots(1, len(hs), figsize=(WIDTH, 3.4))
    fig.subplots_adjust(left=0.2, right=0.99, top=0.82, bottom=0.14, wspace=0.85)
    for ax, h in zip(np.atleast_1d(axes), hs):
        d = drop[drop["horizonte"] == h].set_index("variable")
        p = perm[perm["horizonte"] == h].set_index("variable")["aumento_mae"].reindex(d.index)
        order = p.sort_values().index
        y = np.arange(len(order))
        ax.barh(y + 0.19, p[order], height=0.36, color=MUTED, label="permutación")
        ax.barh(y - 0.19, d.loc[order, "aumento_mae"], height=0.36, color=[GROUP_COLOR[variable_group(v)] for v in order], label="quitar y reentrenar")
        ax.axvline(0, color=INK2, lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([variable_name(v) for v in order], fontsize=6.6)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"A {h} min (MAE con todas: {L.num(d['mae_base'].iloc[0], 0)} ppm)", fontsize=7.8)
        ax.set_xlabel("Aumento del MAE (ppm)")
        ax.xaxis.set_major_formatter(L.tick)
    handles = [Patch(facecolor=MUTED, label="desordenar la variable (permutación)"), Patch(facecolor=INK2, label="entrenar sin la variable (color: su grupo)")]
    _legend_top(fig, handles, ncol=2)
    return fig


def fig_classifier(shap_clf: pd.DataFrame, L: Lang, horizons=(15, 30, 60), top: int = 12):
    fig, axes = plt.subplots(1, len(horizons), figsize=(WIDTH, 0.22 * top + 1.2))
    fig.subplots_adjust(left=0.2, right=0.99, top=1 - 0.7 / (0.22 * top + 1.2), bottom=0.12, wspace=1.25)
    for ax, h in zip(axes, horizons):
        d = shap_clf[(shap_clf["tipo"] == "clasif") & (shap_clf["horizonte"] == h)].sort_values("shap_frac", ascending=False).head(top)
        y = np.arange(len(d))[::-1]
        ax.barh(y, d["shap_frac"], color=[GROUP_COLOR[variable_group(v)] for v in d["variable"]], height=0.72, edgecolor=WHITE, linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([variable_name(v) for v in d["variable"]], fontsize=6.5)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f} %"))
        ax.set_title(f"Cruce en {h} min", fontsize=7.8)
    fig.supxlabel("Fracción del SHAP total (log-odds)", fontsize=7.5, color=INK2, y=0.005)
    _legend_top(fig, _group_handles(), ncol=5)
    return fig
