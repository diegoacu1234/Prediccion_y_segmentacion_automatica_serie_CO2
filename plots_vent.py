"""Figuras del informe de ventilación de un salón (`ventilacion`). Textos en español."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, MultipleLocator, NullLocator

from .plots_exp import AXIS, BLUE, BLUE_DARK, BLUE_LIGHT, INK, INK2, MUTED, RED, SEQ_BLUE, WHITE, WIDTH, Lang
from .plots_med import CASE_COLORS, EMPTY, MONTHS, MONTHS_LONG, WEEKDAYS
from .ventilacion import CATEGORIES, EN_EXCESS, EN_FLOW, GROUPS

GROUP_NAMES = {"clase": "con la clase adentro", "pausa": "en recreos y mediodía", "salida": "al irse la clase", "noche": "de noche, cerrado"}
GROUP_COLORS = {"clase": "#2a78d6", "pausa": "#eb6834", "salida": "#1baf7a", "noche": "#eda100"}
CATEGORY_COLORS = ("#cde2fb", "#86b6ef", "#3987e5", "#123f78")  # de la categoría I a la IV
ACH_AXIS = (0.05, 30.0)
NO_ACH = "#e3e1da"  # día con clase del que no sale un ACH


def _log_ach(axis, L: Lang, ticks=(0.1, 0.3, 1, 3, 10)) -> None:
    axis.set_major_locator(FixedLocator(list(ticks)))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(L.tick)


def _dates(report: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.to_datetime(report.index)


def _thin(ticks, labels, min_gap: float) -> list:
    """Deja en blanco la etiqueta de una marca que queda a menos de `min_gap` de la siguiente."""
    return ["" if k + 1 < len(ticks) and ticks[k + 1] - ticks[k] < min_gap else text for k, text in enumerate(labels)]


def _month_ticks(ax, dates: pd.DatetimeIndex, values, min_gap: float = 15.0) -> None:
    """Marcas en el primer día de cada mes sobre un eje cuyas posiciones son `values` (una por fecha)."""
    months = sorted(set(zip(dates.year, dates.month)))
    ticks, labels = [], []
    for year, month in months:
        k = int(np.flatnonzero((dates.year == year) & (dates.month == month))[0])
        ticks.append(values[k])
        labels.append(MONTHS[month - 1])
    labels = _thin(ticks, labels, min_gap)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(NullLocator())


# ------------------------------------------------------------------ el año


def fig_calendar(report: pd.DataFrame, L: Lang):
    """El año como calendario: una columna por semana y una fila por día hábil, para cuatro indicadores."""
    dates = _dates(report)
    first = (dates.min() - pd.Timedelta(days=int(dates.min().dayofweek))).normalize()
    week = ((dates - first).days // 7).to_numpy()
    n_weeks = int(week.max()) + 1
    panels = (
        ("co2_medio_clase", "CO₂ medio en horario de clase (ppm)", None),
        ("pct_sobre_1000", "Tiempo de clase por encima de 1000 ppm (%)", None),
        ("ach_clase", "ACH con la clase adentro, caso medio (1/h)", LogNorm(0.2, 5.0)),
        ("ach_pausa", "ACH en recreos y mediodía (1/h)", LogNorm(0.5, 12.0)),
    )
    row_h, top_in, bottom_in, gap = 1.02, 0.45, 0.3, 0.42
    height = len(panels) * row_h + (len(panels) - 1) * gap + top_in + bottom_in
    fig = plt.figure(figsize=(WIDTH, height))
    cmap = SEQ_BLUE.copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    month_start = {m: int(week[np.flatnonzero(dates.month == m)[0]]) for m in sorted(set(dates.month))}
    for k, (col, title, norm) in enumerate(panels):
        y0 = (bottom_in + (len(panels) - 1 - k) * (row_h + gap)) / height
        ax = fig.add_axes([0.07, y0, 0.83, row_h / height])
        grid = np.full((5, n_weeks), np.nan)
        grid[report["dia_semana"].to_numpy(), week] = report[col].to_numpy(dtype=float)
        has_class = np.zeros((5, n_weeks), dtype=bool)
        has_class[report["dia_semana"].to_numpy(), week] = True
        missing = has_class & ~np.isfinite(grid)
        ax.imshow(np.where(missing, 1.0, np.nan), cmap=plt.matplotlib.colors.ListedColormap([NO_ACH]), aspect="auto", interpolation="nearest", extent=(-0.5, n_weeks - 0.5, 4.5, -0.5))
        image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest", extent=(-0.5, n_weeks - 0.5, 4.5, -0.5))
        for x in range(n_weeks + 1):  # separación entre celdas
            ax.axvline(x - 0.5, color=WHITE, lw=1.0, zorder=3)
        for y in range(6):
            ax.axhline(y - 0.5, color=WHITE, lw=1.0, zorder=3)
        ax.set_yticks(range(5))
        ax.set_yticklabels([w[:3] for w in WEEKDAYS[:5]], fontsize=6.8)
        ax.set_xticks(list(month_start.values()))
        ax.set_xticklabels(_thin(list(month_start.values()), [MONTHS[m - 1] for m in month_start], 3), fontsize=7)
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left")
        cax = fig.add_axes([0.915, y0, 0.012, row_h / height])
        cb = fig.colorbar(image, cax=cax)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6.6, length=0)
        if norm is not None:
            cb.set_ticks([t for t in (0.2, 0.5, 1, 2, 5, 10) if norm.vmin <= t <= norm.vmax])
            cb.ax.yaxis.set_minor_locator(NullLocator())
        cb.ax.yaxis.set_major_formatter(L.tick)
    fig.legend(handles=[Patch(color=NO_ACH, lw=0, label="día con clase del que no sale un ACH"), Patch(facecolor=WHITE, edgecolor=AXIS, lw=0.6, label="sin clase")], loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=2, handlelength=1.2, columnspacing=1.6)
    return fig


def fig_year(report: pd.DataFrame, cases, L: Lang):
    """El ACH de cada día a lo largo del año: con la clase adentro (con su rango) y con el salón vacío, y el CO2 medio en clase."""
    lo, hi = cases[0], cases[-1]
    dates = _dates(report)
    x = (dates - dates.min()).days.to_numpy(dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 6.6), sharex=True, gridspec_kw={"height_ratios": (1.25, 1.0, 0.8)})
    fig.subplots_adjust(left=0.085, right=0.985, top=0.9, bottom=0.06, hspace=0.3)

    ax = axes[0]
    ok = report["ach_clase"].notna().to_numpy()
    low = report[f"ach_clase_{lo}"].fillna(ACH_AXIS[0]).to_numpy()
    ax.vlines(x[ok], low[ok], report[f"ach_clase_{hi}"].to_numpy()[ok], color=BLUE_LIGHT, lw=1.0, zorder=2)
    ax.vlines(x[ok], report["ach_clase_min"].to_numpy()[ok], report["ach_clase_max"].to_numpy()[ok], color=BLUE, lw=2.2, zorder=3)
    ax.scatter(x[ok], report["ach_clase"].to_numpy()[ok], s=9, color=WHITE, edgecolor=BLUE_DARK, lw=0.9, zorder=4)
    for th, text in ((1000, "para no pasar de 1000 ppm"), (1500, "para no pasar de 1500 ppm")):
        need = float(report[f"ach_necesario_{th}"].median())
        ax.axhline(need, color=INK2, lw=0.8, zorder=1)
        ax.text(0.004, need, f"{text}: {L.num(need, 1)}", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=6.8, color=INK2, zorder=5, bbox=dict(facecolor=WHITE, edgecolor="none", alpha=0.85, pad=0.8))
    ax.set_yscale("log")
    ax.set_ylim(*ACH_AXIS)
    _log_ach(ax.yaxis, L)
    ax.set_ylabel("ACH (1/h)")
    ax.set_title("Con la clase adentro, mientras el CO₂ sube", loc="left")

    ax = axes[1]
    for group in GROUPS[1:]:
        ax.scatter(x, report[f"ach_{group}"], s=9, color=GROUP_COLORS[group], lw=0, zorder=3, label=GROUP_NAMES[group])
    ax.set_yscale("log")
    ax.set_ylim(*ACH_AXIS)
    _log_ach(ax.yaxis, L)
    ax.set_ylabel("ACH (1/h)")
    ax.set_title("Con el salón vacío", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.2, 1.17), ncol=3, handletextpad=0.2, columnspacing=1.2, markerscale=1.4)

    ax = axes[2]
    ax.vlines(x, report["co2_medio_clase"], report["co2_p95_clase"], color=AXIS, lw=0.9, zorder=2)
    ax.scatter(x, report["co2_medio_clase"], s=8, color=INK, lw=0, zorder=3)
    ax.axhline(1000, color=INK2, lw=0.8, zorder=1)
    ax.set_ylabel("CO₂ (ppm)")
    ax.yaxis.set_major_formatter(L.tick)
    ax.set_title("CO₂ en horario de clase: media del día y, hacia arriba, su percentil 95", loc="left")
    _month_ticks(ax, dates, x)
    ax.set_xlim(x.min() - 3, x.max() + 3)
    handles = [
        Line2D([], [], color=BLUE_DARK, lw=0, marker="o", ms=4, mfc=WHITE, mew=0.9, label="caso medio de generación"),
        Line2D([], [], color=BLUE, lw=2.2, label="entre métodos (mínimo a máximo)"),
        Line2D([], [], color=BLUE_LIGHT, lw=1.0, label=f"del caso {lo} al {hi} de generación"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=3, handlelength=1.4, columnspacing=1.3, handletextpad=0.5)
    return fig


def fig_monthly(report: pd.DataFrame, L: Lang):
    """Por mes: CO2, tiempo sobre 1000 ppm, temperatura y ACH con la clase adentro, en las pausas y de noche."""
    panels = (
        ("co2_medio_clase", "CO₂ medio en clase (ppm)", INK),
        ("pct_sobre_1000", "Tiempo de clase sobre 1000 ppm (%)", INK),
        ("temp_clase", "Temperatura en clase (°C)", INK),
        ("ach_clase", "ACH con la clase adentro (1/h)", GROUP_COLORS["clase"]),
        ("ach_pausa", "ACH en recreos y mediodía (1/h)", GROUP_COLORS["pausa"]),
        ("ach_noche", "ACH de noche, cerrado (1/h)", GROUP_COLORS["noche"]),
    )
    months = [m for m in sorted(report["mes"].unique()) if (report["mes"] == m).sum() >= 5]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.3), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.085, hspace=0.3, wspace=0.26)
    for ax, (col, title, color) in zip(axes.ravel(), panels):
        for k, month in enumerate(months):
            v = report.loc[report["mes"] == month, col].dropna()
            if len(v) < 3:
                continue
            q = v.quantile([0.25, 0.5, 0.75]).to_numpy()
            ax.plot([k, k], [q[0], q[2]], color=color, lw=3.0, alpha=0.45, solid_capstyle="butt", zorder=3)
            ax.scatter([k], [q[1]], s=20, color=WHITE, edgecolor=color, lw=1.2, zorder=4)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels([MONTHS[m - 1] for m in months], fontsize=6.8)
        ax.set_xlim(-0.6, len(months) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(L.tick)
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left", fontsize=7.8)
    return fig


def fig_categories(report: pd.DataFrame, L: Lang):
    """Reparto del horario de clase entre las categorías de EN 16798-1, por mes y en el año."""
    months = [m for m in sorted(report["mes"].unique()) if (report["mes"] == m).sum() >= 5]
    rows = [(MONTHS[m - 1], report[report["mes"] == m]) for m in months] + [("año", report)]
    fig, ax = plt.subplots(figsize=(WIDTH, 0.3 * len(rows) + 1.25))
    fig.subplots_adjust(left=0.07, right=0.985, top=1.0 - 0.38 / fig.get_figheight(), bottom=0.5 / fig.get_figheight())
    for y, (_, g) in enumerate(rows):
        left = 0.0
        y = y + (0.35 if y == len(rows) - 1 else 0.0)  # el total del año, separado
        for name, color in zip(CATEGORIES, CATEGORY_COLORS):
            value = float(g[f"pct_cat_{name}"].mean())
            ax.barh(y, value, left=left, height=0.72, color=color, edgecolor=WHITE, lw=1.2, zorder=3)
            if value >= 6:
                ax.text(left + value / 2, y, L.num(value, 0), ha="center", va="center", fontsize=6.8, color=WHITE if name in ("III", "IV") else INK, zorder=4)
            left += value
    ax.set_yticks([k + (0.35 if k == len(rows) - 1 else 0.0) for k in range(len(rows))])
    ax.set_yticklabels([name for name, _ in rows])
    ax.set_ylim(len(rows) - 0.1, -0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Horario de clase (%)")
    ax.grid(axis="y", visible=False)
    edges = [L.g(v) for v in EN_EXCESS]
    labels = (f"I: hasta {edges[0]} ppm sobre el exterior", f"II: {edges[0]} a {edges[1]}", f"III: {edges[1]} a {edges[2]}", f"IV: más de {edges[2]}")
    fig.legend(handles=[Patch(color=c, lw=0, label=t) for c, t in zip(CATEGORY_COLORS, labels)], loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=4, handlelength=1.1, columnspacing=1.2, handletextpad=0.4)
    return fig


# ------------------------------------------------------------------ el día


def hourly_profile(df: pd.DataFrame, report: pd.DataFrame, bin_min: float = 5.0) -> pd.DataFrame:
    """Mediana y cuartiles del CO2 según la hora, sobre los días del informe."""
    t = df["tiempo"]
    sub = df[t.dt.normalize().isin(_dates(report)).to_numpy()]
    hour = sub["tiempo"].dt.hour + sub["tiempo"].dt.minute / 60.0
    g = sub.groupby(np.floor(hour * 60.0 / bin_min) * bin_min / 60.0)["co2"]
    return pd.DataFrame({"q25": g.quantile(0.25), "mediana": g.median(), "q75": g.quantile(0.75)})


def fig_blocks(df: pd.DataFrame, report: pd.DataFrame, by_block: pd.DataFrame, blocks: pd.DataFrame, L: Lang, hours=(6.5, 21.0), levels=(1000.0, 1500.0)):
    """El día tipo: CO2 según la hora y, debajo, el ACH de cada bloque del horario.

    `levels` son los niveles de CO2 [ppm] que se marcan como referencia, los
    mismos con los que se mide la exposición.
    """
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.0), sharex=True, gridspec_kw={"height_ratios": (1.0, 1.0)})
    fig.subplots_adjust(left=0.085, right=0.985, top=0.9, bottom=0.085, hspace=0.24)
    ax = axes[0]
    prof = hourly_profile(df, report)
    prof = prof[(prof.index >= hours[0]) & (prof.index <= hours[1])]
    for b in blocks[blocks["tipo"] == "vacio"].itertuples():
        for axis in axes:
            axis.axvspan(max(b.desde, hours[0]), min(b.hasta, hours[1]), color=EMPTY, lw=0, zorder=0)
    ax.fill_between(prof.index, prof["q25"], prof["q75"], color=MUTED, alpha=0.3, lw=0, zorder=2)
    ax.plot(prof.index, prof["mediana"], color=INK, lw=1.4, zorder=3)
    for level in levels:
        ax.axhline(level, color=INK2, lw=0.7, ls=(0, (4, 2)), zorder=1)
        ax.text(hours[1] - 0.08, level, f"{L.num(level, 0)} ppm", ha="right", va="bottom", fontsize=6.6, color=INK2)
    ax.set_ylabel("CO₂ (ppm)")
    ax.yaxis.set_major_formatter(L.tick)
    ax.set_ylim(bottom=350)
    ax.set_title("CO₂ en un día con clase: mediana y rango intercuartil", loc="left")

    ax = axes[1]
    for b in blocks.itertuples():
        v = by_block.loc[by_block["bloque"] == b.bloque, "ach"].dropna()
        x0, x1 = max(b.desde, hours[0]), min(b.hasta, hours[1])
        if len(v) < 5 or x1 <= x0:
            continue
        q = v.quantile([0.25, 0.5, 0.75]).to_numpy()
        color = GROUP_COLORS[b.grupo]
        pad = 0.04
        ax.fill_between([x0 + pad, x1 - pad], q[0], q[2], color=color, alpha=0.3, lw=0, zorder=2)
        ax.plot([x0 + pad, x1 - pad], [q[1], q[1]], color=color, lw=2.2, solid_capstyle="butt", zorder=3)
        ax.text(0.5 * (x0 + x1), q[2] * 1.12, L.num(q[1], 2 if q[1] < 1 else 1), ha="center", va="bottom", fontsize=6.8, color=INK, zorder=4)
        ax.text(0.5 * (x0 + x1), ACH_AXIS[0] * 1.15, f"{len(v)} d", ha="center", va="bottom", fontsize=6.0, color=INK2, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(*ACH_AXIS)
    _log_ach(ax.yaxis, L)
    ax.set_ylabel("ACH (1/h)")
    ax.set_xlabel("Hora del día")
    ax.set_xlim(*hours)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_title("ACH de cada bloque del horario: mediana entre días (línea, con su valor) y rango intercuartil", loc="left")
    handles = [Patch(color=GROUP_COLORS[g], alpha=0.6, lw=0, label=GROUP_NAMES[g]) for g in GROUPS] + [Patch(color=EMPTY, lw=0, label="salón vacío según el horario")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=5, handlelength=1.1, columnspacing=1.1, handletextpad=0.4)
    return fig


def fig_matrix(by_block: pd.DataFrame, blocks: pd.DataFrame, L: Lang, min_days: int = 3):
    """Mediana del ACH por mes y por bloque del horario."""
    data = by_block.assign(mes=pd.to_datetime(by_block["fecha"]).dt.month)
    months = sorted(data["mes"].unique())
    names = list(blocks["bloque"])
    mat = np.full((len(names), len(months)), np.nan)
    for i, name in enumerate(names):
        for j, month in enumerate(months):
            v = data.loc[(data["bloque"] == name) & (data["mes"] == month), "ach"].dropna()
            if len(v) >= min_days:
                mat[i, j] = v.median()
    keep = np.isfinite(mat).any(axis=0)
    mat, months = mat[:, keep], [m for m, k in zip(months, keep) if k]
    fig, ax = plt.subplots(figsize=(WIDTH, 0.33 * len(names) + 1.0))
    fig.subplots_adjust(left=0.27, right=0.9, top=1.0 - 0.3 / fig.get_figheight(), bottom=0.4 / fig.get_figheight())
    cmap = SEQ_BLUE.copy()
    cmap.set_bad(WHITE)
    norm = LogNorm(0.15, 8.0)
    image = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(names)):
        for j in range(len(months)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, L.num(mat[i, j], 2 if mat[i, j] < 1 else 1), ha="center", va="center", fontsize=7, color=WHITE if mat[i, j] > 1.6 else INK)
    kind = dict(zip(blocks["bloque"], blocks["grupo"]))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n}  ·  {GROUP_NAMES[kind[n]]}" for n in names], fontsize=7)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([MONTHS[m - 1] for m in months])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in range(len(months) + 1):
        ax.axvline(x - 0.5, color=WHITE, lw=1.2)
    for y in range(len(names) + 1):
        ax.axhline(y - 0.5, color=WHITE, lw=1.2)
    cax = fig.add_axes([0.915, 0.4 / fig.get_figheight(), 0.014, 1.0 - 0.7 / fig.get_figheight()])
    cb = fig.colorbar(image, cax=cax)
    cb.outline.set_visible(False)
    cb.set_ticks([0.2, 0.5, 1, 2, 5])
    cb.ax.yaxis.set_minor_locator(NullLocator())
    cb.ax.yaxis.set_major_formatter(L.tick)
    cb.ax.tick_params(labelsize=6.8, length=0)
    cb.set_label("ACH (1/h)", fontsize=7.4, color=INK2)
    return fig


# ------------------------------------------------------------------ relaciones


def _spearman(x, y) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    return float(pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman")) if ok.sum() > 5 else np.nan


def fig_co2_vs_ach(report: pd.DataFrame, cases_table: pd.DataFrame, L: Lang):
    """Cuánto CO2 hay en clase según el ACH con la clase adentro, con el equilibrio que predice cada caso de generación."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.25), sharex=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.82, bottom=0.14, wspace=0.26)
    x = report["ach_clase"].to_numpy(dtype=float)
    grid = np.geomspace(0.1, 10.0, 80)
    c_ext = float(report["c_ext"].median())
    ax = axes[0]
    for case, color in zip(cases_table.itertuples(), CASE_COLORS):
        ax.plot(grid, c_ext + case.fuente_ppm_h / grid, color=color, lw=1.2, zorder=2, label=f"equilibrio, caso {case.caso}")
    ax.scatter(x, report["co2_p95_clase"], s=10, color=WHITE, edgecolor=MUTED, lw=0.7, zorder=3, label="percentil 95 del día")
    ax.scatter(x, report["co2_medio_clase"], s=10, color=INK, lw=0, zorder=4, label="media del día")
    ax.set_ylim(300, 5500)
    ax.set_ylabel("CO₂ en horario de clase (ppm)")
    ax.yaxis.set_major_formatter(L.tick)
    ax.text(0.97, 0.95, f"ρ de Spearman con la media: {L.num(_spearman(x, report['co2_medio_clase'].to_numpy(dtype=float)), 2)}", transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=INK2)
    ax = axes[1]
    ax.scatter(x, report["pct_sobre_1000"], s=10, color=INK, lw=0, zorder=4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Tiempo de clase sobre 1000 ppm (%)")
    ax.text(0.97, 0.95, f"ρ de Spearman: {L.num(_spearman(x, report['pct_sobre_1000'].to_numpy(dtype=float)), 2)}", transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=INK2)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(0.1, 10.0)
        _log_ach(ax.xaxis, L)
        ax.set_xlabel("ACH con la clase adentro, caso medio (1/h)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=3, handlelength=1.4, columnspacing=1.4, handletextpad=0.5)
    return fig


def fig_flow(report: pd.DataFrame, cases, L: Lang, references=(("ASHRAE 62.1", 7.5),)):
    """Aire exterior por persona de cada día, para los tres casos de generación, contra los caudales de referencia."""
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.14)
    for case, color in zip(cases, CASE_COLORS):
        v = np.sort(report[f"caudal_{case}"].dropna().to_numpy())
        ax.step(v, 100.0 * np.arange(1, len(v) + 1) / len(v), where="post", color=color, lw=1.6, zorder=3, label=f"caso {case} ({len(v)} días)")
    for name, flow in references:
        ax.axvline(flow, color=INK2, lw=0.8, zorder=1)
        ax.text(flow, 101.5, f"{name}: {L.g(flow)} L/s", ha="center", va="bottom", fontsize=6.8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 30.0)
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10, 30]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(L.tick)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Aire exterior por persona con la clase adentro (L/s)")
    ax.set_ylabel("Días con ese caudal o menos (%)")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=3, handlelength=1.4, columnspacing=1.6)
    return fig


def fig_temperature(report: pd.DataFrame, L: Lang):
    """ACH y CO2 contra la temperatura del salón en horario de clase."""
    panels = (
        ("ach_clase", "ACH con la clase adentro (1/h)", GROUP_COLORS["clase"], True),
        ("ach_pausa", "ACH en recreos y mediodía (1/h)", GROUP_COLORS["pausa"], True),
        ("co2_medio_clase", "CO₂ medio en clase (ppm)", INK, False),
    )
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.9), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.9, bottom=0.155, wspace=0.3)
    temp = report["temp_clase"].to_numpy(dtype=float)
    edges = np.arange(np.floor(np.nanmin(temp) / 2) * 2, np.nanmax(temp) + 2, 2.0)
    for ax, (col, title, color, log) in zip(axes, panels):
        y = report[col].to_numpy(dtype=float)
        ax.scatter(temp, y, s=9, color=color, alpha=0.45, lw=0, zorder=2)
        mids, meds = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            v = y[(temp >= a) & (temp < b) & np.isfinite(y)]
            if len(v) >= 5:
                mids.append(0.5 * (a + b))
                meds.append(np.median(v))
        ax.plot(mids, meds, color=INK, lw=1.4, marker="o", ms=3.6, mec=WHITE, mew=0.5, zorder=4)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(*ACH_AXIS)
            _log_ach(ax.yaxis, L)
        else:
            ax.yaxis.set_major_formatter(L.tick)
        ax.set_title(title, loc="left", fontsize=7.8)
        ax.set_xlabel("Temperatura en clase (°C)")
        ax.text(0.96, 0.95, f"ρ = {L.num(_spearman(temp, y), 2)}", transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=INK2)
    return fig


# ------------------------------------------------------------------ días


def day_readings(df: pd.DataFrame, date: str, max_gap_min: float = 10.0):
    """Hora y CO2 de las lecturas de un día, con un NaN en cada hueco para que la línea no lo cruce."""
    day = pd.Timestamp(date)
    t = df["tiempo"].to_numpy()
    i0, i1 = np.searchsorted(t, [np.datetime64(day), np.datetime64(day + pd.Timedelta(days=1))])
    t_h = (t[i0:i1] - np.datetime64(day)) / np.timedelta64(1, "h")
    co2 = df["co2"].to_numpy(dtype=float)[i0:i1]
    cut = np.flatnonzero(np.diff(t_h) > max_gap_min / 60.0) + 1
    return np.insert(t_h, cut, np.nan), np.insert(co2, cut, np.nan)


def _mini_day(ax, df, daily, date: str, blocks: pd.DataFrame, hours, top: float, L: Lang) -> None:
    t_h, co2 = day_readings(df, date)
    show = np.isnan(t_h) | ((t_h >= hours[0]) & (t_h <= hours[1]))  # los NaN marcan los huecos: se conservan
    for b in blocks[blocks["tipo"] == "vacio"].itertuples():
        if min(b.hasta, hours[1]) > max(b.desde, hours[0]):
            ax.axvspan(max(b.desde, hours[0]), min(b.hasta, hours[1]), color=EMPTY, lw=0, zorder=0)
    ax.axhline(1000, color=AXIS, lw=0.7, zorder=1)
    ax.plot(t_h[show], np.minimum(co2[show], top), color=INK, lw=0.8, zorder=3)
    ax.set_xlim(*hours)
    ax.set_ylim(300, top)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([1000, 2000, 3000]))
    ax.yaxis.set_major_formatter(L.tick)
    ax.tick_params(labelsize=6.2, length=2)


def _ach_text(row, L: Lang) -> str:
    parts = []
    for col, name in (("ach_clase", "clase"), ("ach_pausa", "pausas")):
        if np.isfinite(row[col]):
            parts.append(f"{name} {L.num(row[col], 2 if row[col] < 1 else 1)}")
    return " · ".join(parts) if parts else "sin ACH"


def fig_extreme_days(df: pd.DataFrame, daily: pd.DataFrame, report: pd.DataFrame, blocks: pd.DataFrame, L: Lang, n: int = 4, hours=(7.0, 19.0)):
    """Los días con más y con menos ACH con la clase adentro."""
    ranked = report.dropna(subset=["ach_clase"]).sort_values("ach_clase")
    rows = (("Los de más ACH con la clase adentro", ranked.index[::-1][:n]), ("Los de menos", ranked.index[:n]))
    fig, axes = plt.subplots(2, n, figsize=(WIDTH, 3.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.11, hspace=0.62, wspace=0.08)
    for r, (title, dates) in enumerate(rows):
        for c, date in enumerate(dates):
            ax = axes[r, c]
            _mini_day(ax, df, daily, date, blocks, hours, 4200.0, L)
            d = pd.Timestamp(date)
            ax.set_title(f"{WEEKDAYS[d.dayofweek][:3]} {d.day} de {MONTHS[d.month - 1]}", fontsize=7.4, pad=11)
            ax.text(0.0, 1.03, f"ACH: {_ach_text(report.loc[date], L)}", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.4, color=INK2)
        axes[r, 0].set_ylabel("CO₂ (ppm)", fontsize=7.2)
        fig.text(0.075, axes[r, 0].get_position().y1 + 0.085, title, ha="left", va="bottom", fontsize=8.2, fontweight="semibold", color=INK)
    for ax in axes[1]:
        ax.set_xlabel("Hora del día", fontsize=7.2)
    return fig


def fig_month_board(df: pd.DataFrame, daily: pd.DataFrame, report: pd.DataFrame, blocks: pd.DataFrame, year: int, month: int, L: Lang, hours=(7.0, 19.0)):
    """Tablero de un mes: un panel por día hábil con el CO2 de 7 a 19 h y el ACH del día."""
    dates = _dates(report)
    sub = report[(dates.year == year) & (dates.month == month)]
    first = pd.Timestamp(year=year, month=month, day=1)
    monday = first - pd.Timedelta(days=int(first.dayofweek))
    last = first + pd.offsets.MonthEnd(0)
    n_weeks = int((last - monday).days // 7) + 1
    row_h, top_in, bottom_in = 1.12, 1.45, 0.4
    height = n_weeks * row_h + top_in + bottom_in
    fig, axes = plt.subplots(n_weeks, 5, figsize=(WIDTH, height), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=1.0 - top_in / height, bottom=bottom_in / height, hspace=0.5, wspace=0.07)
    for w in range(n_weeks):
        for d in range(5):
            ax = axes[w, d]
            day = monday + pd.Timedelta(days=7 * w + d)
            key = day.strftime("%Y-%m-%d")
            if day.month != month:
                ax.set_visible(False)
                continue
            ax.text(0.0, 1.03, f"{WEEKDAYS[d][:3]} {day.day}", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0, fontweight="semibold", color=INK)
            if key not in sub.index:
                ax.text(0.5, 0.5, "sin clase", transform=ax.transAxes, ha="center", va="center", fontsize=6.6, color=MUTED)
                ax.set_xlim(*hours)
                ax.grid(False)
                continue
            _mini_day(ax, df, daily, key, blocks, hours, 4200.0, L)
            ax.text(1.0, 1.03, _ach_text(sub.loc[key], L), transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=INK2)
            share = sub.loc[key, "pct_sobre_1000"]
            if np.isfinite(share):
                ax.text(0.97, 0.93, f"{L.num(share, 0)} %", transform=ax.transAxes, ha="right", va="top", fontsize=6.4, color=RED if share >= 50 else INK2, zorder=5)
        shown = [ax for ax in axes[w] if ax.get_visible()]
        if shown:
            shown[0].set_ylabel("CO₂ (ppm)", fontsize=7.0)
            shown[0].tick_params(labelleft=True)
    for ax in axes[-1]:
        ax.set_xlabel("Hora del día", fontsize=7.0)
    fig.text(0.07, 1.0 - 0.38 / height, f"{MONTHS_LONG[month - 1].capitalize()} de {year}", ha="left", va="bottom", fontsize=10, fontweight="semibold", color=INK)
    if len(sub):
        med = sub.median(numeric_only=True)
        parts = [f"CO₂ medio en clase: {L.num(med['co2_medio_clase'], 0)} ppm", f"{L.num(med['pct_sobre_1000'], 0)} % del horario de clase sobre 1000 ppm"]
        ach = []
        if np.isfinite(med["ach_clase"]):
            ach.append(f"con la clase adentro: {L.num(med['ach_clase'], 2)}")
        if np.isfinite(med["ach_pausa"]):
            ach.append(f"en recreos y mediodía: {L.num(med['ach_pausa'], 1)}")
        fig.text(0.07, 1.0 - 0.58 / height, f"{len(sub)} días con clase. Medianas del mes: " + " · ".join(parts), ha="left", va="bottom", fontsize=7.2, color=INK2)
        if ach:
            fig.text(0.07, 1.0 - 0.72 / height, "ACH (1/h), mediana del mes: " + " · ".join(ach), ha="left", va="bottom", fontsize=7.2, color=INK2)
    fig.text(0.07, 1.0 - 0.94 / height, "En cada día: arriba a la derecha, ACH (1/h) con la clase adentro y en recreos y mediodía;", ha="left", va="bottom", fontsize=6.6, color=INK2)
    fig.text(0.07, 1.0 - 1.07 / height, "adentro, porcentaje del horario de clase por encima de 1000 ppm (línea gris). Fondo gris: salón vacío según el horario.", ha="left", va="bottom", fontsize=6.6, color=INK2)
    return fig
