"""Figuras de la ventilación del aula (6.10 a 6.14)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, MultipleLocator, NullLocator

from .plots_exp import AXIS, BLUE, BLUE_DARK, BLUE_LIGHT, INK, INK2, MUTED, SEQ_BLUE, WHITE, WIDTH, Lang
from .plots_med import CASE_COLORS, EMPTY, MONTHS, WEEKDAYS
from .ventilacion import GROUP_NAMES, GROUPS

GROUP_COLORS = {"clase": "#2a78d6", "pausa": "#eb6834", "salida": "#1baf7a", "noche": "#eda100"}
ACH_AXIS = (0.05, 30.0)
NO_ACH = "#e3e1da"  # día con clase sin ACH


def _log_ach(axis, L: Lang, ticks=(0.1, 0.3, 1, 3, 10)) -> None:
    axis.set_major_locator(FixedLocator(list(ticks)))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(L.tick)


def _dates(report: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.to_datetime(report.index)


def _thin(ticks, labels, min_gap: float) -> list:
    """Borra la etiqueta de una marca que queda a menos de `min_gap` de la siguiente."""
    return ["" if k + 1 < len(ticks) and ticks[k + 1] - ticks[k] < min_gap else text for k, text in enumerate(labels)]


def _month_ticks(ax, dates: pd.DatetimeIndex, values, min_gap: float = 15.0) -> None:
    """Marca el primer día de cada mes; `values` es la posición de cada fecha."""
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


# ------------------------------------------------------------------ el año (6.11 a 6.13)


def fig_calendar(report: pd.DataFrame, L: Lang):
    """Calendario: una columna por semana y una fila por día hábil, para cuatro indicadores."""
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
        for x in range(n_weeks + 1):  # bordes de las celdas
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
    """ACH de cada día con la clase adentro (con su rango) y con el salón vacío, y el CO2 en clase."""
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
    """Por mes: CO2, tiempo sobre 1000 ppm, temperatura y ACH (clase, pausas y noche)."""
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


# ------------------------------------------------------------------ el día (6.10)


def hourly_profile(df: pd.DataFrame, report: pd.DataFrame, bin_min: float = 5.0) -> pd.DataFrame:
    """Mediana y cuartiles del CO2 según la hora, en los días del informe."""
    t = df["tiempo"]
    sub = df[t.dt.normalize().isin(_dates(report)).to_numpy()]
    hour = sub["tiempo"].dt.hour + sub["tiempo"].dt.minute / 60.0
    g = sub.groupby(np.floor(hour * 60.0 / bin_min) * bin_min / 60.0)["co2"]
    return pd.DataFrame({"q25": g.quantile(0.25), "mediana": g.median(), "q75": g.quantile(0.75)})


def fig_blocks(df: pd.DataFrame, report: pd.DataFrame, by_block: pd.DataFrame, blocks: pd.DataFrame, L: Lang, hours=(6.5, 21.0), levels=(1000.0, 1500.0)):
    """Día tipo: CO2 según la hora y, abajo, la ACH de cada bloque. `levels`: niveles de referencia [ppm]."""
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


# ------------------------------------------------------------------ caudal por persona (6.14)


def fig_flow(report: pd.DataFrame, cases, L: Lang, references=(("ASHRAE 62.1", 7.5),)):
    """Aire exterior por persona con los tres casos de generación, contra la referencia de 7,5 L/s."""
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
