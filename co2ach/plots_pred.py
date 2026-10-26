"""Figuras del pronóstico y del cruce de 1000 ppm (caps. 7 y 8, apéndice D).

Cada modelo tiene siempre el mismo color y marcador; las referencias van en gris
y cortadas. Los conjuntos de información van de azul claro (sólo CO2) a oscuro
(todo). El umbral de 1000 ppm, en rojo.
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullLocator

from .plots_exp import GRID, INK, INK2, MUTED, SEQ_BLUE, WHITE, WIDTH, Lang, _row_tag
from .plots_med import MONTHS, MONTHS_LONG
from .prediccion import HORIZONS, INFO_NAMES, MODEL_NAMES, THRESHOLD, time_to_threshold

CRITICAL = "#d03b3b"
MODEL_KEYS = ("persistencia", "perfil", "fisico", "ridge", "mlp", "lightgbm", "chronos")
MODEL_STYLE = {
    "persistencia": {"color": INK2, "ls": (0, (4, 2)), "marker": "o", "mfc": WHITE},
    "perfil": {"color": MUTED, "ls": (0, (1, 1.5)), "marker": "s", "mfc": WHITE},
    "fisico": {"color": "#1baf7a", "ls": "-", "marker": "^"},
    "ridge": {"color": "#eda100", "ls": "-", "marker": "D"},
    "mlp": {"color": "#e87ba4", "ls": "-", "marker": "v"},
    "lightgbm": {"color": "#2a78d6", "ls": "-", "marker": "o"},
    "chronos": {"color": "#eb6834", "ls": "-", "marker": "s"},
}
APPROACH_STYLE = {
    "global_con_salon": ("#2a78d6", "o", "Global, con el salón"),
    "global_sin_salon": ("#1baf7a", "^", "Global, sin el salón"),
    "por_salon": ("#eb6834", "s", "Uno por salón"),
}
INFO_COLORS = {"I0": "#86b6ef", "I1": "#5598e7", "I2": "#2a78d6", "I3": "#1c5cab", "I4": "#0d366b"}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a9a8", "#c43a3a"])
SPLIT_COLORS = {"sin_datos": WHITE, "sin_clase": "#eceae3", "train": "#2a78d6", "es": "#86b6ef", "cal": "#1baf7a", "test": "#eb6834"}
SPLIT_NAMES = {
    "train": "entrenamiento (días 1–15)",
    "es": "detención temprana (días 16–19)",
    "cal": "calibración (días 20–23)",
    "test": "prueba (del 24 a fin de mes)",
    "sin_clase": "sin clase",
    "sin_datos": "sin datos",
}


# ------------------------------------------------------------------ auxiliares


def model_name(key: str) -> str:
    return MODEL_NAMES.get(key, key)


def _style(key: str, lw: float = 1.3, ms: float = 4.0) -> dict:
    s = MODEL_STYLE[key]
    return {"color": s["color"], "ls": s["ls"], "lw": lw, "marker": s["marker"], "ms": ms, "mfc": s.get("mfc", s["color"]), "mec": s["color"] if "mfc" in s else WHITE, "mew": 0.8 if "mfc" in s else 0.5}


def _handles(keys, lw: float = 1.3) -> list:
    return [Line2D([], [], label=model_name(k), **_style(k, lw)) for k in keys]


def _threshold(ax, label: bool = True, x: float = 0.995) -> None:
    ax.axhline(THRESHOLD, color=CRITICAL, lw=0.9, ls=(0, (5, 2)), zorder=1)
    if label:
        ax.text(x, THRESHOLD, f"{THRESHOLD:,.0f} ppm".replace(",", "."), transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.8, color=CRITICAL)


def _clock(h: float) -> str:
    m = int(round(h * 60))
    return f"{m // 60}:{m % 60:02d}"


def _hours_axis(ax, lo: float = 7.0, hi: float = 18.5, step: float = 2.0) -> None:
    ax.set_xlim(lo, hi)
    ticks = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _clock(v)))
    ax.xaxis.set_minor_locator(NullLocator())


def _horizon_axis(ax, L: Lang, label: bool = True) -> None:
    ax.set_xlim(-1, 62)
    ax.xaxis.set_major_locator(FixedLocator([1, 5, 10, 15, 20, 25, 30, 45, 60]))
    ax.xaxis.set_major_formatter(L.tick)
    ax.xaxis.set_minor_locator(NullLocator())
    if label:
        ax.set_xlabel("Horizonte (min)")


def _pct(ax, axis: str = "y") -> None:
    fmt = FuncFormatter(lambda v, _: f"{v * 100:.0f} %".replace("-", "−"))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _legend_top(fig, handles, ncol: int, y: float = 0.995, fontsize: float = 7.3) -> None:
    fig.legend(handles=handles, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y), frameon=False, fontsize=fontsize, handlelength=2.4, columnspacing=1.4)


def _day_series(grid_day: pd.DataFrame):
    """Hora del día y CO2 de la grilla de un día."""
    t = grid_day.index
    return (t.hour + t.minute / 60.0).to_numpy(), grid_day["co2"].to_numpy(float)


# ------------------------------------------------------------------ figura 7.1


def pick_schematic_origin(det: pd.DataFrame, fc, lead: int = 20) -> int:
    """Un origen de A02 con alarma y un cruce a `lead` min: el de error mediano en el instante."""
    t_hat = time_to_threshold(det["co2"].to_numpy(float), fc.mid)
    ok = (det["sala"] == "A02").to_numpy() & det["habilitado"].to_numpy(bool) & (det["min_al_cruce"].to_numpy() == lead) & np.isfinite(t_hat)
    idx = np.flatnonzero(ok)
    if not len(idx):
        idx = np.flatnonzero(det["habilitado"].to_numpy(bool) & np.isfinite(t_hat) & (np.abs(det["min_al_cruce"].to_numpy() - lead) <= 5))
    err = np.abs(t_hat[idx] - det["min_al_cruce"].to_numpy()[idx])
    return int(idx[np.argsort(err)[len(err) // 2]])


def fig_schematic(det: pd.DataFrame, fc, grid_room: pd.DataFrame, i: int, L: Lang):
    """Qué se pronostica desde un origen: historia, horizontes con su intervalo, umbral e instante del cruce."""
    row = det.iloc[i]
    t0 = pd.Timestamp(row["tiempo"])
    seg = grid_room.loc[t0 - pd.Timedelta(minutes=90) : t0 + pd.Timedelta(minutes=66), "co2"]
    x = ((seg.index - t0) / pd.Timedelta(minutes=1)).to_numpy()
    c0 = float(row["co2"])
    mid, lo, hi = fc.mid[i], fc.lo[i], fc.hi[i]
    t_mid = time_to_threshold([c0], mid[None])[0]
    t_early = time_to_threshold([c0], hi[None])[0]
    t_late = time_to_threshold([c0], lo[None])[0]
    lead = float(row["min_al_cruce"])

    fig, ax = plt.subplots(figsize=(WIDTH, 3.5))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.8, bottom=0.13)
    past = x <= 0
    ax.plot(x[past], seg.to_numpy()[past], color=INK, lw=1.4, zorder=4)
    ax.plot(x[~past], seg.to_numpy()[~past], color=INK, lw=1.0, ls=(0, (1, 1.4)), zorder=4)
    knots = np.r_[0, HORIZONS]
    color = MODEL_STYLE["lightgbm"]["color"]
    ax.fill_between(knots, np.r_[c0, lo], np.r_[c0, hi], color=color, alpha=0.16, lw=0, zorder=2)
    ax.plot(knots, np.r_[c0, mid], color=color, lw=1.4, zorder=5)
    ax.plot(HORIZONS, mid, ls="", marker="o", ms=4.2, color=color, mec=WHITE, mew=0.6, zorder=6)
    _threshold(ax)
    ax.axvline(0, color=INK2, lw=0.8, zorder=3)
    ax.axvspan(-60, 0, color="#f3f2ee", lw=0, zorder=0)
    if np.isfinite(t_mid):
        late = t_late if np.isfinite(t_late) else 62
        early = t_early if np.isfinite(t_early) else 0
        ax.errorbar([t_mid], [THRESHOLD], xerr=[[t_mid - early], [late - t_mid]], fmt="D", ms=5.5, color=color, mec=WHITE, mew=0.7, ecolor=color, elinewidth=2.2, capsize=0, zorder=7)
    ax.plot([lead], [THRESHOLD], marker="|", ms=13, mew=1.8, color=INK, zorder=8)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.12 * (ymax - ymin))
    top = ymax + 0.1 * (ymax - ymin)
    for h in HORIZONS:
        ax.plot([h, h], [top - 0.03 * (ymax - ymin), top], color=INK2, lw=0.8, zorder=3)
    ax.text(31, top + 0.015 * (ymax - ymin), "horizontes: 1, 5, 10, 15, 20, 25, 30, 45 y 60 min", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.text(-30, top - 0.02 * (ymax - ymin), "historia que usan las variables\n(la última hora)", ha="center", va="top", fontsize=7, color=INK2)
    ax.annotate("origen t", xy=(0, c0), xytext=(-24, c0 + 0.22 * (ymax - ymin)), fontsize=7.2, color=INK, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.7})
    ax.annotate("pronóstico e intervalo del 80 %", xy=(45, mid[7]), xytext=(38, mid[7] + 0.2 * (ymax - ymin)), fontsize=7.2, color=color, ha="center", arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7})
    if np.isfinite(t_mid):
        ax.annotate("instante pronosticado del cruce\ny su banda", xy=(t_mid, THRESHOLD), xytext=(63, THRESHOLD - 0.32 * (ymax - ymin)), fontsize=7.2, color=color, ha="right", arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7})
    ax.annotate("cruce medido", xy=(lead, THRESHOLD), xytext=(lead - 22, THRESHOLD + 0.2 * (ymax - ymin)), fontsize=7.2, color=INK, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.7})
    ax.set_xlim(-90, 64)
    ax.xaxis.set_major_locator(MultipleLocator(15))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"t{'+' if v > 0 else '−' if v < 0 else ''}{abs(v):.0f}" if v else "t"))
    ax.set_xlabel("Minutos respecto del origen")
    ax.set_ylabel("CO₂ (ppm)")
    ax.yaxis.set_major_formatter(L.tick)
    fig.suptitle(f"Salón {row['sala']}, {t0.day} de {MONTHS_LONG[t0.month - 1]}, origen a las {t0.hour}:{t0.minute:02d} (LightGBM)", x=0.08, ha="left", y=0.975, fontsize=8.5, fontweight="semibold")
    handles = [
        Line2D([], [], color=INK, lw=1.4, label="CO₂ medido hasta t"),
        Line2D([], [], color=INK, lw=1.0, ls=(0, (1, 1.4)), label="lo que pasó después"),
        Line2D([], [], color=color, lw=1.4, marker="o", ms=4, mec=WHITE, label="pronóstico"),
    ]
    fig.legend(handles=handles, loc="upper left", fontsize=7, ncol=3, bbox_to_anchor=(0.075, 0.925), frameon=False)
    return fig


# ------------------------------------------------------------------ figura 7.2


def split_matrix(rows: pd.DataFrame, days: dict, rooms) -> tuple:
    """Matriz salón × día de 2024 con la partición de cada día."""
    cats = ["sin_datos", "sin_clase", "train", "es", "cal", "test"]
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    mat = np.zeros((len(rooms), len(dates)), dtype=int)
    used = rows.assign(fecha=rows["tiempo"].dt.normalize()).groupby(["sala", "fecha"])["split"].first()
    for r, sala in enumerate(rooms):
        d = days[sala].reindex(dates)
        has = d["lecturas"].fillna(0).to_numpy() > 0
        mat[r] = np.where(has, 1, 0)
        if sala in used.index.get_level_values(0):
            u = used.loc[sala]
            pos = dates.get_indexer(u.index)
            mat[r, pos[pos >= 0]] = [cats.index(s) for s in u.to_numpy()[pos >= 0]]
    return mat, dates, cats


def fig_split(rows: pd.DataFrame, days: dict, rooms, L: Lang):
    mat, dates, cats = split_matrix(rows, days, rooms)
    fig, ax = plt.subplots(figsize=(WIDTH, 4.6))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.07)
    cmap = ListedColormap([SPLIT_COLORS[c] for c in cats])
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(cats) - 0.5, interpolation="nearest")
    ax.set_yticks(range(len(rooms)))
    ax.set_yticklabels(rooms, fontsize=6.2)
    starts = pd.date_range(dates[0], dates[-1], freq="MS")
    ax.set_xticks([(d - dates[0]).days for d in starts])
    ax.set_xticklabels([MONTHS[d.month - 1] for d in starts])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    n = rows.groupby("split").size()
    handles = [Patch(facecolor=SPLIT_COLORS[c], edgecolor=GRID if c in ("sin_datos", "sin_clase") else "none", label=SPLIT_NAMES[c] + (f": {n[c]:,} orígenes".replace(",", ".") if c in n.index else "")) for c in ("train", "es", "cal", "test", "sin_clase", "sin_datos")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, fontsize=7, handlelength=1.2, columnspacing=1.2)
    return fig


# ------------------------------------------------------------------ un día (figuras 7.3, D.2 y D.3)


def fig_day(det_day: pd.DataFrame, fcs_day: dict, grid_day: pd.DataFrame, title: str, L: Lang, horizons=(15, 60), models=MODEL_KEYS):
    """Una fila por modelo: lo medido y lo pronosticado 15 y 60 min antes, con el intervalo del 80 %."""
    rows = [(m, model_name(m), MODEL_STYLE[m]["color"]) for m in models if m in fcs_day]
    return fig_day_grid(det_day, fcs_day, grid_day, title, L, rows, horizons)


def fig_day_grid(det_day: pd.DataFrame, fcs_day: dict, grid_day: pd.DataFrame, title: str, L: Lang, rows, horizons=(15, 60)):
    """Una fila por pronóstico (`rows`: clave, nombre, color) y una columna por horizonte."""
    nrow = len(rows)
    height = 0.55 + 0.5 + nrow * 0.98
    fig, axes = plt.subplots(nrow, len(horizons), figsize=(WIDTH, height), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.99, top=1 - 0.62 / height, bottom=0.42 / height, hspace=0.12, wspace=0.05)
    narrow = len(horizons) > 2
    hx, hy = _day_series(grid_day)
    win = (hx >= 7.0) & (hx <= 18.6)
    top = 1.35 * float(np.nanmax(hy[win]))  # el eje es el de lo medido
    y_meas = det_day["co2"].to_numpy(float)[:, None] + det_day[[f"y{h}" for h in HORIZONS]].to_numpy(float)
    tmin = det_day["minuto"].to_numpy(float)
    for c, h in enumerate(horizons):
        j = HORIZONS.index(h)
        x = (tmin + h) / 60.0
        for r, (key, name, color) in enumerate(rows):
            ax = axes[r, c]
            fc = fcs_day[key]
            ax.plot(hx[win], hy[win], color=INK, lw=0.9, zorder=3)
            if fc.lo is not None:
                ax.fill_between(x, fc.lo[:, j], fc.hi[:, j], color=color, alpha=0.18, lw=0, zorder=1)
            ax.plot(x, fc.mid[:, j], color=color, lw=1.1 if not narrow else 0.9, zorder=4)
            _threshold(ax, label=(r == 0 and c == len(horizons) - 1 and not narrow))
            mae = np.nanmean(np.abs(fc.mid[:, j] - y_meas[:, j]))
            ax.text(0.985, 0.06 if narrow else 0.94, f"MAE {L.num(mae, 0)} ppm", transform=ax.transAxes, ha="right", va="bottom" if narrow else "top", fontsize=6.3 if narrow else 6.8, color=INK2, bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.8, "alpha": 0.85})
            if c == 0:
                _row_tag(ax, name)
            if r == 0:
                ax.set_title(f"A {h} min" if narrow else f"Pronóstico hecho {h} min antes", loc="left")
            _hours_axis(ax, step=4.0 if narrow else 2.0)
            ax.yaxis.set_major_formatter(L.tick)
            ax.yaxis.set_major_locator(MultipleLocator(1000 if top > 2200 else 500))
            ax.set_ylim(min(350.0, float(np.nanmin(hy[win])) - 50), top)
    fig.text(0.012, 0.5, "CO₂ (ppm)", rotation=90, va="center", ha="left", fontsize=8, color=INK2)
    fig.suptitle(title, x=0.085, ha="left", y=1 - 0.12 / height, fontsize=8.5, fontweight="semibold")
    return fig


# ------------------------------------------------------------------ figura 7.4


FAN_ROWS = (("persistencia", "perfil", "fisico"), ("ridge", "mlp", "lightgbm", "chronos"))


def fig_fans(det_day: pd.DataFrame, fcs_day: dict, grid_day: pd.DataFrame, origins, title: str, L: Lang):
    """Desde cuatro orígenes: la hora siguiente según cada modelo, contra lo medido."""
    fig, axes = plt.subplots(2, len(origins), figsize=(WIDTH, 4.7), sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.79, bottom=0.1, hspace=0.3, wspace=0.07)
    times = det_day["tiempo"].reset_index(drop=True)
    knots = np.r_[0, HORIZONS]
    for c, minute in enumerate(origins):
        hit = np.flatnonzero(det_day["minuto"].to_numpy() == minute)
        if not len(hit):
            continue
        i = int(hit[0])
        t0 = pd.Timestamp(times[i])
        seg = grid_day.loc[t0 - pd.Timedelta(minutes=45) : t0 + pd.Timedelta(minutes=62), "co2"]
        x = ((seg.index - t0) / pd.Timedelta(minutes=1)).to_numpy()
        c0 = float(det_day["co2"].iloc[i])
        for r, keys in enumerate(FAN_ROWS):
            ax = axes[r, c]
            ax.axvspan(-45, 0, color="#f3f2ee", lw=0, zorder=0)
            ax.plot(x, seg.to_numpy(), color=INK, lw=1.1, zorder=3)
            for key in keys:
                if key not in fcs_day:
                    continue
                fc = fcs_day[key]
                st = _style(key, lw=1.1, ms=3.2)
                if key == "lightgbm":
                    ax.fill_between(knots, np.r_[c0, fc.lo[i]], np.r_[c0, fc.hi[i]], color=st["color"], alpha=0.14, lw=0, zorder=1)
                ax.plot(knots, np.r_[c0, fc.mid[i]], zorder=4, **st)
            _threshold(ax, label=False)
            ax.axvline(0, color=INK2, lw=0.7)
            ax.set_xlim(-45, 62)
            ax.xaxis.set_major_locator(FixedLocator([-30, 0, 30, 60]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "t" if v == 0 else f"{v:+.0f}".replace("-", "−")))
            ax.yaxis.set_major_formatter(L.tick)
            if r == 0:
                ax.set_title(f"Origen a las {t0.hour}:{t0.minute:02d}")
            if r == 1:
                ax.set_xlabel("Minutos")
    for r in range(2):
        axes[r, 0].set_ylabel("CO₂ (ppm)")
    handles = [Line2D([], [], color=INK, lw=1.1, label="CO₂ medido")] + _handles(MODEL_KEYS, lw=1.1)
    _legend_top(fig, handles, ncol=4, y=0.995)
    fig.text(0.085, 0.855, title, fontsize=8.5, fontweight="semibold", ha="left")
    return fig


# ------------------------------------------------------------------ error (7.5, 7.6, 7.7, 7.9, 7.10 y D.1)


def fig_horizon(metrics: pd.DataFrame, L: Lang, models=MODEL_KEYS):
    """MAE, RMSE y mejora sobre la persistencia, por horizonte."""
    models = [m for m in models if m in set(metrics["modelo"])]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.9))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.78, bottom=0.16, wspace=0.32)
    for ax, col, title in zip(axes, ("mae", "rmse", "mejora_sobre_persistencia"), ("MAE (ppm)", "RMSE (ppm)", "Mejora sobre la persistencia")):
        for key in models:
            if col == "mejora_sobre_persistencia" and key == "persistencia":
                continue
            d = metrics[metrics["modelo"] == key].sort_values("horizonte")
            ax.plot(d["horizonte"], d[col], **_style(key, lw=1.2, ms=3.4))
        _horizon_axis(ax, L)
        ax.set_title(title)
        if col == "mejora_sobre_persistencia":
            ax.axhline(0, color=INK2, lw=0.7)
            _pct(ax)
        else:
            ax.yaxis.set_major_formatter(L.tick)
            ax.set_ylim(bottom=0)
    _legend_top(fig, _handles(models, lw=1.2), ncol=7, y=0.99, fontsize=7)
    return fig


def fig_by_hour(by_hour: pd.DataFrame, L: Lang, horizons=(15, 60), models=MODEL_KEYS):
    models = [m for m in models if m in set(by_hour["modelo"])]
    fig, axes = plt.subplots(1, len(horizons), figsize=(WIDTH, 2.8), sharey=False)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.78, bottom=0.17, wspace=0.18)
    for ax, h in zip(axes, horizons):
        d = by_hour[by_hour["horizonte"] == h]
        for key in models:
            e = d[d["modelo"] == key].sort_values("hora")
            ax.plot(e["hora"] + 0.5, e["mae"], **_style(key, lw=1.2, ms=3.4))
        _hours_axis(ax, 7.0, 18.0, step=2.0)
        ax.set_title(f"A {h} min")
        ax.set_xlabel("Hora del origen")
        ax.yaxis.set_major_formatter(L.tick)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("MAE (ppm)")
    _legend_top(fig, _handles(models, lw=1.2), ncol=7, y=0.99, fontsize=7)
    return fig


def fig_global_vs_room(table: pd.DataFrame, L: Lang, h: int = 30):
    """LightGBM global con y sin el salón, y uno por salón."""
    d = table[(table["modelo"] == "lightgbm") & (table["horizonte"] == h)].pivot_table(index="sala", columns="enfoque", values="mae")
    d = d.sort_values("global_con_salon")
    fig = plt.figure(figsize=(WIDTH, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.25, 1.0), left=0.075, right=0.99, top=0.86, bottom=0.1, wspace=0.28)
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(d))
    for key, (color, marker, label) in APPROACH_STYLE.items():
        if key in d:
            ax.plot(d[key], y, ls="", marker=marker, ms=4.2, color=color, mec=WHITE, mew=0.5, label=label, zorder=3)
    lo = d.min(axis=1)
    hi = d.max(axis=1)
    ax.hlines(y, lo, hi, color=GRID, lw=1.6, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d.index, fontsize=6.4)
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"MAE a {h} min (ppm)")
    ax.xaxis.set_major_formatter(L.tick)
    ax.set_title(f"Por salón, a {h} min")

    ax = fig.add_subplot(gs[1])
    all_h = table[table["modelo"] == "lightgbm"]
    common = all_h.groupby(["sala", "horizonte"])["enfoque"].nunique()
    common = common[common == all_h["enfoque"].nunique()].reset_index()[["sala", "horizonte"]]
    all_h = all_h.merge(common, on=["sala", "horizonte"])
    for key, (color, marker, label) in APPROACH_STYLE.items():
        e = all_h[all_h["enfoque"] == key].groupby("horizonte")["mae"].median()
        if len(e):
            ax.plot(e.index, e.to_numpy(), color=color, marker=marker, ms=3.6, mec=WHITE, mew=0.5, lw=1.2, label=label)
    _horizon_axis(ax, L)
    ax.set_title("Mediana entre salones")
    ax.set_ylabel("MAE (ppm)")
    ax.yaxis.set_major_formatter(L.tick)
    ax.set_ylim(bottom=0)
    handles = [Line2D([], [], color=c, marker=m, ms=4.2, mec=WHITE, lw=1.2, label=lab) for c, m, lab in APPROACH_STYLE.values()]
    _legend_top(fig, handles, ncol=3, y=0.995)
    return fig


def fig_by_month(by_month: pd.DataFrame, L: Lang, horizons=(15, 60), models=MODEL_KEYS):
    models = [m for m in models if m in set(by_month["modelo"])]
    fig, axes = plt.subplots(1, len(horizons), figsize=(WIDTH, 2.6))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.76, bottom=0.12, wspace=0.18)
    for ax, h in zip(axes, horizons):
        d = by_month[by_month["horizonte"] == h]
        for key in models:
            e = d[d["modelo"] == key].sort_values("mes")
            ax.plot(e["mes"], e["mae"], **_style(key, lw=1.2, ms=3.6))
        months = sorted(d["mes"].unique())
        ax.xaxis.set_major_locator(FixedLocator(months))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: MONTHS[int(v) - 1]))
        ax.set_xlim(months[0] - 0.3, months[-1] + 0.3)
        ax.set_title(f"A {h} min")
        ax.yaxis.set_major_formatter(L.tick)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("MAE (ppm)")
    _legend_top(fig, _handles(models, lw=1.2), ncol=7, y=0.99, fontsize=7)
    return fig


def fig_density(hist: dict, edges, key: str, L: Lang, horizons=(5, 15, 30, 60)):
    """Pronosticado contra medido (densidad) para un modelo."""
    fig, axes = plt.subplots(1, len(horizons), figsize=(WIDTH, 2.45), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.9, top=0.86, bottom=0.2, wspace=0.08)
    vmax = max(hist[f"{key}_h{h}"].max() for h in horizons)
    cmap = SEQ_BLUE.copy()
    cmap.set_bad(WHITE)
    for ax, h in zip(axes, horizons):
        H = hist[f"{key}_h{h}"].T  # filas: pronosticado, columnas: medido
        im = ax.imshow(np.ma.masked_equal(H, 0), origin="lower", extent=(edges[0], edges[-1], edges[0], edges[-1]), cmap=cmap, norm=LogNorm(1, vmax), aspect="equal", interpolation="nearest")
        ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], color=INK2, lw=0.7)
        ax.axvline(THRESHOLD, color=CRITICAL, lw=0.7, ls=(0, (4, 2)))
        ax.axhline(THRESHOLD, color=CRITICAL, lw=0.7, ls=(0, (4, 2)))
        ax.set_title(f"A {h} min")
        ax.set_xlabel("CO₂ medido (ppm)")
        ax.xaxis.set_major_locator(FixedLocator([1000, 2000, 3000]))
        ax.yaxis.set_major_locator(FixedLocator([1000, 2000, 3000]))
        ax.xaxis.set_major_formatter(L.tick)
        ax.yaxis.set_major_formatter(L.tick)
        ax.grid(False)
    axes[0].set_ylabel("Pronosticado (ppm)")
    cax = fig.add_axes([0.915, 0.2, 0.012, 0.66])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("orígenes", fontsize=7.5, color=INK2)
    cb.ax.tick_params(labelsize=7, length=0)
    return fig


def fig_coverage(cover: pd.DataFrame, cover_month: pd.DataFrame, L: Lang, models=MODEL_KEYS, h_month: int = 30):
    models = [m for m in models if m in set(cover["modelo"])]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.9))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.76, bottom=0.16, wspace=0.34)
    extra = [k for k in set(cover["modelo"]) if k.endswith("_crudo")]
    for key in models + extra:
        base = key.replace("_crudo", "")
        st = _style(base, lw=1.2, ms=3.4)
        if key in extra:
            st.update(ls=(0, (2, 1.5)), mfc=WHITE, mec=st["color"], mew=0.8)
        d = cover[cover["modelo"] == key].sort_values("horizonte")
        axes[0].plot(d["horizonte"], d["cobertura"], **st)
        axes[1].plot(d["horizonte"], d["ancho"], **st)
        m = cover_month[(cover_month["modelo"] == key) & (cover_month["horizonte"] == h_month)].sort_values("mes")
        axes[2].plot(m["mes"], m["cobertura"], **st)
    for ax in (axes[0], axes[2]):
        ax.axhline(0.8, color=INK, lw=0.8)
        ax.text(0.98, 0.8, "80 % nominal", transform=ax.get_yaxis_transform(), fontsize=6.8, color=INK2, va="center", ha="right", zorder=9, bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.6})
        _pct(ax)
    _horizon_axis(axes[0], L)
    _horizon_axis(axes[1], L)
    axes[0].set_title("Cobertura del intervalo")
    axes[1].set_title("Ancho medio (ppm)")
    axes[1].yaxis.set_major_formatter(L.tick)
    axes[1].set_ylim(bottom=0)
    months = sorted(cover_month["mes"].unique())
    axes[2].xaxis.set_major_locator(FixedLocator(months))
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda v, _: MONTHS[int(v) - 1]))
    axes[2].set_title(f"Cobertura por mes, a {h_month} min")
    handles = _handles(models, lw=1.2)
    if extra:
        handles.append(Line2D([], [], color=MODEL_STYLE["chronos"]["color"], ls=(0, (2, 1.5)), marker="s", ms=3.4, mfc=WHITE, lw=1.2, label="Chronos sin calibrar"))
    _legend_top(fig, handles, ncol=4, y=0.995, fontsize=7)
    return fig


# ------------------------------------------------------------------ el umbral (8.1, 8.2, 8.3, 8.5 y 8.6)


def _spans(x_min, on) -> list:
    """(inicio, largo) en horas de las rachas de `on`; `x_min` en minutos del día."""
    out = []
    idx = np.flatnonzero(on)
    if not len(idx):
        return out
    breaks = np.flatnonzero(np.diff(x_min[idx]) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    for s, e in zip(starts, ends):
        out.append((x_min[s] / 60.0, (x_min[e] - x_min[s] + 1) / 60.0))
    return out


def fig_threshold_day(det_day: pd.DataFrame, fcs_day: dict, grid_day: pd.DataFrame, cross_min, title: str, L: Lang, models=MODEL_KEYS):
    """Un día con sus cruces y las alarmas de cada modelo con las dos reglas."""
    models = [m for m in models if m in fcs_day and m != "persistencia"]
    n = len(models)
    height = 2.0 + 0.36 * n + 0.85
    fig = plt.figure(figsize=(WIDTH, height))
    gs = fig.add_gridspec(2, 1, height_ratios=(1.75, 0.36 * n), left=0.15, right=0.985, top=1 - 0.3 / height, bottom=0.78 / height, hspace=0.08)
    ax = fig.add_subplot(gs[0])
    hx, hy = _day_series(grid_day)
    win = (hx >= 7.0) & (hx <= 18.6)
    ax.plot(hx[win], hy[win], color=INK, lw=1.1)
    _threshold(ax)
    for m in cross_min:
        ax.axvline(m / 60.0, color=CRITICAL, lw=0.8, alpha=0.6)
    ax.set_ylabel("CO₂ (ppm)")
    ax.yaxis.set_major_formatter(L.tick)
    _hours_axis(ax)
    ax.tick_params(labelbottom=False)
    ax.set_title(title)

    ax2 = fig.add_subplot(gs[1], sharex=ax)
    c0 = det_day["co2"].to_numpy(float)
    hab = det_day["habilitado"].to_numpy(bool)
    xm = det_day["minuto"].to_numpy()
    for r, key in enumerate(models):
        fc = fcs_day[key]
        y = n - 1 - r
        color = MODEL_STYLE[key]["color"]
        prob = hab & np.isfinite(time_to_threshold(c0, fc.mid))
        sure = hab & np.isfinite(time_to_threshold(c0, fc.lo))
        ax2.broken_barh(_spans(xm, prob), (y - 0.32, 0.64), facecolors=color, alpha=0.35, lw=0)
        ax2.broken_barh(_spans(xm, sure), (y - 0.32, 0.64), facecolors=color, lw=0)
    for m in cross_min:
        ax2.axvline(m / 60.0, color=CRITICAL, lw=0.8, alpha=0.6)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([model_name(k) for k in models[::-1]])
    ax2.set_ylim(-0.6, n - 0.4)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("Hora")
    _hours_axis(ax2)
    handles = [Patch(facecolor=INK2, alpha=0.35, label="alarma probable: el pronóstico llega a 1000 ppm en la próxima hora"), Patch(facecolor=INK2, label="alarma segura: también el extremo inferior del 80 %"), Line2D([], [], color=CRITICAL, lw=0.8, label="cruce medido")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.15, 0.0), ncol=2, fontsize=6.8, frameon=False)
    return fig


def fig_anticipation(leads: pd.DataFrame, summary: pd.DataFrame, L: Lang, models=MODEL_KEYS):
    """% de cruces anticipados con al menos L min y minutos de alarma en falso por día."""
    models = [m for m in models if m in set(leads["modelo"]) and m != "persistencia"]
    fig = plt.figure(figsize=(WIDTH, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.9), left=0.075, right=0.99, top=0.78, bottom=0.16, wspace=0.3)
    grid = np.arange(0, 61)
    for c, rule in enumerate(("probable", "segura")):
        ax = fig.add_subplot(gs[c])
        for key in models:
            a = leads[(leads["modelo"] == key) & (leads["regla"] == rule)]["anticipacion"].to_numpy()
            frac = [(a >= g).mean() for g in grid[1:]]
            st = _style(key, lw=1.2, ms=0)
            st["marker"] = ""
            ax.plot(grid[1:], frac, **st)
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 1.0)
        ax.xaxis.set_major_locator(FixedLocator([5, 10, 15, 30, 45, 60]))
        _pct(ax)
        ax.set_xlabel("Anticipación de al menos (min)")
        ax.set_title(f"Cruces anticipados, alarma {rule}")
    ax = fig.add_subplot(gs[2])
    y = np.arange(len(models))
    for r, (rule, mfc) in enumerate((("probable", WHITE), ("segura", None))):
        for k, key in enumerate(models):
            s = summary[(summary["modelo"] == key) & (summary["regla"] == rule)]
            if not len(s):
                continue
            color = MODEL_STYLE[key]["color"]
            ax.plot(s["minutos_falsos_por_dia"], [len(models) - 1 - k + (0.15 if r == 0 else -0.15)], ls="", marker="o", ms=4.6, color=color, mfc=mfc or color, mec=color, mew=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([model_name(k) for k in models[::-1]])
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(L.tick)
    ax.set_xlabel("Minutos por día")
    ax.set_title("Alarma sin cruce en la hora")
    handles = _handles(models, lw=1.2) + [Line2D([], [], ls="", marker="o", color=INK2, mfc=WHITE, label="probable"), Line2D([], [], ls="", marker="o", color=INK2, label="segura")]
    _legend_top(fig, handles, ncol=8, y=0.995, fontsize=6.8)
    return fig


def fig_timing(timing: pd.DataFrame, L: Lang, models=MODEL_KEYS):
    """Error del instante pronosticado del cruce según cuánto faltaba."""
    models = [m for m in models if m in set(timing["modelo"]) and m != "persistencia"]
    blocks = sorted(set(timing["bloque"]), key=lambda s: int(s.split("-")[0]))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.95), gridspec_kw={"width_ratios": (1.5, 1.0)})
    fig.subplots_adjust(left=0.085, right=0.99, top=0.78, bottom=0.17, wspace=0.25)
    ax = axes[0]
    width = 0.8 / len(models)
    for k, key in enumerate(models):
        d = timing[timing["modelo"] == key].set_index("bloque").reindex(blocks)
        x = np.arange(len(blocks)) - 0.4 + width * (k + 0.5)
        color = MODEL_STYLE[key]["color"]
        ax.vlines(x, d["p10"], d["p90"], color=color, lw=0.8, alpha=0.6)
        ax.vlines(x, d["q1"], d["q3"], color=color, lw=2.4)
        ax.plot(x, d["mediana"], ls="", marker=MODEL_STYLE[key]["marker"], ms=3.6, color=color, mec=WHITE, mew=0.5)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels(blocks)
    ax.set_xlabel("Faltaban para el cruce (min)")
    ax.set_ylabel("Pronosticado − medido (min)")
    ax.set_title("Error del instante del cruce")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    for key in models:
        d = timing[timing["modelo"] == key].set_index("bloque").reindex(blocks)
        ax.plot(np.arange(len(blocks)), d["en_banda"], **_style(key, lw=1.2, ms=3.4))
    ax.axhline(0.8, color=INK, lw=0.8)
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels(blocks, fontsize=6.6)
    ax.set_xlabel("Faltaban para el cruce (min)")
    ax.set_ylim(0, 1.0)
    _pct(ax)
    ax.set_title("Cruce medido dentro de la banda")
    _legend_top(fig, _handles(models, lw=1.2), ncol=6, y=0.99, fontsize=7)
    return fig


def fig_probability(probs: pd.DataFrame, rel: pd.DataFrame, L: Lang, horizons=(15, 30, 60), models=("lightgbm_clasif", "lightgbm", "chronos", "fisico", "ridge")):
    """Confiabilidad de P(cruce en h min) y mejora de Brier."""
    models = [m for m in models if m in set(probs["modelo"])]
    fig, axes = plt.subplots(1, len(horizons) + 1, figsize=(WIDTH, 2.75))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.76, bottom=0.17, wspace=0.36)

    def st(key):
        if key == "lightgbm_clasif":
            return {"color": MODEL_STYLE["lightgbm"]["color"], "ls": "-", "lw": 1.4, "marker": "o", "ms": 3.4, "mec": WHITE, "mew": 0.5}
        s = _style(key, lw=1.1, ms=3.2)
        if key == "lightgbm":
            s.update(ls=(0, (2, 1.5)), mfc=WHITE, mec=s["color"], mew=0.8)
        return s

    for ax, h in zip(axes, horizons):
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.7)
        for key in models:
            d = rel[(rel["modelo"] == key) & (rel["horizonte"] == h) & (rel["n"] >= 30)]
            ax.plot(d["p_media"], d["frecuencia"], **st(key))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        _pct(ax, "x")
        _pct(ax)
        ax.xaxis.set_major_locator(FixedLocator([0, 0.5, 1]))
        ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 1]))
        ax.set_title(f"Cruce en {h} min")
        ax.set_xlabel("Probabilidad pronosticada")
    axes[0].set_ylabel("Frecuencia observada")
    ax = axes[-1]
    for key in models:
        d = probs[probs["modelo"] == key].sort_values("horizonte")
        ax.plot(d["horizonte"], 1 - d["brier"] / d["brier_ref"], **st(key))
    _horizon_axis(ax, L)
    ax.xaxis.set_major_locator(FixedLocator([1, 15, 30, 45, 60]))
    ax.set_title("Mejora de Brier")
    _pct(ax)
    handles = [Line2D([], [], label="LightGBM (clasificador)" if k == "lightgbm_clasif" else "LightGBM (de los residuos)" if k == "lightgbm" else model_name(k), **st(k)) for k in models]
    _legend_top(fig, handles, ncol=5, y=0.995, fontsize=7)
    return fig


# ------------------------------------------------------------------ información (8.12 a 8.14, D.11 y D.12)

NESTED = ("I0", "I1", "I2", "I3", "I4")
STEP_NAMES = {"I1": "+ hora y día", "I2": "+ temperatura y humedad", "I3": "+ historia del salón", "I4": "+ salón"}
DROP_NAMES = {"sin_G1": "sin hora y día", "sin_G2": "sin temperatura y humedad", "sin_G3": "sin historia del salón", "I3": "sin salón"}


def fig_info_horizon(info: pd.DataFrame, metrics: pd.DataFrame, L: Lang):
    """MAE por horizonte de los conjuntos anidados, con LightGBM y ridge."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.78, bottom=0.16, wspace=0.08)
    pers = metrics[metrics["modelo"] == "persistencia"].sort_values("horizonte")
    for ax, kind, title in zip(axes, ("lightgbm", "ridge"), ("LightGBM", "Ridge")):
        ax.plot(pers["horizonte"], pers["mae"], **_style("persistencia", lw=1.0, ms=0))
        for key in NESTED:
            d = info[(info["conjunto"] == key) & (info["modelo"] == kind)].sort_values("horizonte")
            ax.plot(d["horizonte"], d["mae"], color=INFO_COLORS[key], lw=1.3, marker="o", ms=3.3, mec=WHITE, mew=0.5)
        _horizon_axis(ax, L)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(L.tick)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("MAE (ppm)")
    handles = [Line2D([], [], color=INFO_COLORS[k], lw=1.3, marker="o", ms=3.3, mec=WHITE, label=INFO_NAMES[k]) for k in NESTED] + [Line2D([], [], label="Persistencia", **_style("persistencia", lw=1.0, ms=0))]
    _legend_top(fig, handles, ncol=3, y=0.995, fontsize=7)
    return fig


def info_changes(info: pd.DataFrame, kind: str = "lightgbm") -> tuple:
    """Cambio relativo del MAE al agregar cada grupo y al quitarlo del modelo completo."""
    d = info[info["modelo"] == kind].pivot_table(index="conjunto", columns="horizonte", values="mae")
    add = pd.DataFrame({STEP_NAMES[b]: d.loc[b] / d.loc[a] - 1 for a, b in zip(NESTED[:-1], NESTED[1:])}).T
    drop = pd.DataFrame({DROP_NAMES[k]: d.loc[k] / d.loc["I4"] - 1 for k in DROP_NAMES if k in d.index}).T
    return add, drop


def fig_info_changes(info: pd.DataFrame, L: Lang, kind: str = "lightgbm"):
    """Agregar y quitar cada grupo de variables, por horizonte."""
    add, drop = info_changes(info, kind)
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 3.9), sharex=True)
    fig.subplots_adjust(left=0.22, right=0.87, top=0.9, bottom=0.11, hspace=0.38)
    lim = max(np.nanmax(np.abs(add.to_numpy())), np.nanmax(np.abs(drop.to_numpy())), 0.01)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    for ax, d, title in zip(axes, (add, drop), ("Al agregar el grupo, sobre el conjunto anterior", "Al quitar el grupo del modelo completo")):
        im = ax.imshow(d.to_numpy(), aspect="auto", cmap=DIVERGING, norm=norm)
        for (r, c), v in np.ndenumerate(d.to_numpy()):
            ax.text(c, r, f"{v * 100:+.0f} %".replace("-", "−"), ha="center", va="center", fontsize=6.8, color=WHITE if abs(v) > 0.55 * lim else INK)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d.index)
        ax.set_xticks(range(d.shape[1]))
        ax.set_xticklabels([str(h) for h in d.columns])
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title)
    axes[1].set_xlabel("Horizonte (min)")
    cax = fig.add_axes([0.885, 0.11, 0.012, 0.79])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:+.0f} %".replace("-", "−")))
    cb.set_label("cambio del MAE (azul: mejora)", fontsize=7.5, color=INK2)
    fig.suptitle(f"Cambio del MAE de {'LightGBM' if kind == 'lightgbm' else 'ridge'}", x=0.22, ha="left", y=0.985, fontsize=8.5, fontweight="semibold")
    return fig


def fig_info_history(info: pd.DataFrame, L: Lang, horizons=(1, 5, 15, 30, 60)):
    """MAE según los minutos de historia de CO2, y relativo a una hora."""
    d = info[info["modelo"] == "lightgbm"].pivot_table(index="conjunto", columns="horizonte", values="mae")
    lengths = {15: "H15", 30: "H30", 60: "I4", 120: "H120", 240: "H240"}
    lengths = {k: v for k, v in lengths.items() if v in d.index}
    ramp = ["#b7d3f6", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.78, bottom=0.17, wspace=0.28)
    x = list(lengths)
    for k, h in enumerate(horizons):
        y = np.array([d.loc[lengths[L_], h] for L_ in x])
        axes[0].plot(x, y, color=ramp[k], lw=1.3, marker="o", ms=3.4, mec=WHITE, mew=0.5, label=f"a {h} min")
        axes[1].plot(x, y / d.loc["I4", h] - 1, color=ramp[k], lw=1.3, marker="o", ms=3.4, mec=WHITE, mew=0.5)
    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(x))
        ax.xaxis.set_major_formatter(L.tick)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Historia de CO₂ (min)")
    axes[0].set_title("MAE (ppm)")
    axes[0].yaxis.set_major_formatter(L.tick)
    axes[0].set_ylim(bottom=0)
    axes[1].axhline(0, color=INK2, lw=0.7)
    axes[1].set_title("Cambio respecto de una hora")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:+.0f} %".replace("-", "−")))
    handles = [Line2D([], [], color=ramp[k], lw=1.3, marker="o", ms=3.4, mec=WHITE, label=f"a {h} min") for k, h in enumerate(horizons)]
    _legend_top(fig, handles, ncol=5, y=0.99)
    return fig


def fig_info_threshold(thr: pd.DataFrame, L: Lang, lead: int = 15):
    """Cruces anticipados con al menos `lead` min y falsas alarmas según la información."""
    keys = [k for k in NESTED if k in set(thr["conjunto"])]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.76, bottom=0.27, wspace=0.25)
    x = np.arange(len(keys))
    for kind, ls, mfc in (("lightgbm", "-", None), ("ridge", (0, (3, 1.5)), WHITE)):
        for rule, alpha in (("probable", 0.5), ("segura", 1.0)):
            d = thr[(thr["modelo"] == kind) & (thr["regla"] == rule)].set_index("conjunto").reindex(keys)
            color = INFO_COLORS["I4"] if kind == "lightgbm" else MUTED
            for ax, col in zip(axes, (f"anticipados_{lead}", "falsas_por_dia")):
                ax.plot(x, d[col], color=color, ls=ls, lw=1.2, alpha=alpha, marker="o", ms=3.6, mfc=mfc or color, mec=color)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([INFO_NAMES[k].replace(" (completo)", "") for k in keys], rotation=25, ha="right", fontsize=6.8)
        ax.grid(axis="x", visible=False)
    axes[0].set_title(f"Cruces anticipados con al menos {lead} min")
    _pct(axes[0])
    axes[0].set_ylim(0, 1)
    axes[1].set_title("Falsas alarmas por día")
    axes[1].yaxis.set_major_formatter(L.tick)
    axes[1].set_ylim(bottom=0)
    handles = [
        Line2D([], [], color=INFO_COLORS["I4"], lw=1.2, marker="o", ms=3.6, label="LightGBM, alarma segura"),
        Line2D([], [], color=INFO_COLORS["I4"], lw=1.2, alpha=0.5, marker="o", ms=3.6, label="LightGBM, probable"),
        Line2D([], [], color=MUTED, ls=(0, (3, 1.5)), lw=1.2, marker="o", ms=3.6, mfc=WHITE, label="ridge, segura"),
        Line2D([], [], color=MUTED, ls=(0, (3, 1.5)), lw=1.2, alpha=0.5, marker="o", ms=3.6, mfc=WHITE, label="ridge, probable"),
    ]
    _legend_top(fig, handles, ncol=4, y=0.99, fontsize=7)
    return fig


def fig_info_hour(by_hour: pd.DataFrame, L: Lang, h: int = 30, keys=("I0", "I1", "I3", "I4")):
    d = by_hour[(by_hour["modelo"] == "lightgbm") & (by_hour["horizonte"] == h)]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.8, bottom=0.17)
    for key in keys:
        e = d[d["conjunto"] == key].sort_values("hora")
        ax.plot(e["hora"] + 0.5, e["mae"], color=INFO_COLORS[key], lw=1.3, marker="o", ms=3.4, mec=WHITE, mew=0.5, label=INFO_NAMES[key])
    _hours_axis(ax, 7.0, 18.0, step=2.0)
    ax.set_xlabel("Hora del origen")
    ax.set_ylabel(f"MAE a {h} min (ppm)")
    ax.yaxis.set_major_formatter(L.tick)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.2), ncol=len(keys), fontsize=7)
    return fig


def fig_tradeoff(trade: pd.DataFrame, L: Lang, leads=(5, 15, 30), models=("perfil", "fisico", "ridge", "mlp", "lightgbm", "chronos", "lightgbm_clasif"), label_p=(0.2, 0.5, 0.8)):
    """Alarma con P(cruce en la próxima hora) > p: anticipación contra falsas alarmas al barrer p."""
    models = [m for m in models if m in set(trade["modelo"])]
    fig, axes = plt.subplots(1, len(leads), figsize=(WIDTH, 2.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.76, bottom=0.16, wspace=0.08)

    def st(key):
        if key == "lightgbm_clasif":
            return {"color": MODEL_STYLE["lightgbm"]["color"], "ls": "-", "lw": 1.5, "marker": "o", "ms": 3.6, "mec": WHITE, "mew": 0.5}
        s = _style(key, lw=1.1, ms=3.2)
        if key == "lightgbm":
            s.update(ls=(0, (2, 1.5)), mfc=WHITE, mec=s["color"], mew=0.8)
        return s

    for ax, lead in zip(axes, leads):
        for key in models:
            d = trade[trade["modelo"] == key].sort_values("p_min")
            ax.plot(d["minutos_falsos_por_dia"], d[f"anticipados_{lead}"], **st(key))
            if key == "lightgbm_clasif":
                for p in label_p:
                    r = d[np.isclose(d["p_min"], p)]
                    if len(r):
                        ax.annotate(f"p ≥ {L.g(p)}", (r["minutos_falsos_por_dia"].iloc[0], r[f"anticipados_{lead}"].iloc[0]), xytext=(4, -9), textcoords="offset points", fontsize=6.6, color=MODEL_STYLE["lightgbm"]["color"])
        ax.set_title(f"Anticipados con al menos {lead} min")
        ax.set_xlabel("Minutos de alarma sin cruce, por día")
        ax.set_ylim(0, 1)
        ax.set_xlim(left=0)
        _pct(ax)
        ax.xaxis.set_major_formatter(L.tick)
    axes[0].set_ylabel("Cruces")
    handles = [Line2D([], [], label="LightGBM (clasificador)" if k == "lightgbm_clasif" else "LightGBM (de los residuos)" if k == "lightgbm" else model_name(k), **st(k)) for k in models]
    _legend_top(fig, handles, ncol=4, y=0.995, fontsize=7)
    return fig


# ------------------------------------------------------------------ situaciones (7.8, D.4 a D.10)


def _trajectory_panel(ax, case, keys_styles, band=None, L=None) -> None:
    """Lo medido alrededor del origen y la trayectoria de cada clave."""
    x, seg, c0 = case["x"], case["medido"], case["c0"]
    knots = np.r_[0, HORIZONS]
    ax.axvspan(x[0], 0, color="#f3f2ee", lw=0, zorder=0)
    ax.plot(x, seg, color=INK, lw=1.3, zorder=3)
    if band is not None:
        lo, hi, color = band
        ax.fill_between(knots, np.r_[c0, lo], np.r_[c0, hi], color=color, alpha=0.13, lw=0, zorder=1)
    for key, color, ls, marker in keys_styles:
        ax.plot(knots, np.r_[c0, case["pron"][key]], color=color, ls=ls, lw=1.1, marker=marker, ms=2.8, mec=WHITE, mew=0.4, zorder=4)
    _threshold(ax, label=False)
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_xlim(x[0], 62)
    ax.xaxis.set_major_locator(FixedLocator([-30, 0, 30, 60]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "t" if v == 0 else f"{v:+.0f}".replace("-", "−")))
    if L is not None:
        ax.yaxis.set_major_formatter(L.tick)


def fig_situation(name: str, cases: list, L: Lang, info_rows):
    """Una situación en varios casos: todos los modelos y LightGBM con distinta información."""
    groups = (("persistencia", "perfil", "fisico"), ("ridge", "mlp", "lightgbm", "chronos"))
    n = len(cases)
    fig, axes = plt.subplots(3, n, figsize=(WIDTH, 7.2 if name else 6.7), sharex=True, squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.78 if name else 0.845, bottom=0.07, hspace=0.32, wspace=0.18)
    for c, case in enumerate(cases):
        for r, keys in enumerate(groups):
            ks = [(k, MODEL_STYLE[k]["color"], MODEL_STYLE[k]["ls"], MODEL_STYLE[k]["marker"]) for k in keys if k in case["pron"]]
            band = (case["lo"]["lightgbm"], case["hi"]["lightgbm"], MODEL_STYLE["lightgbm"]["color"]) if r == 1 else None
            _trajectory_panel(axes[r, c], case, ks, band, L)
        ks = [(k, col, "-", "o") for k, _, col in info_rows]
        _trajectory_panel(axes[2, c], case, ks, None, L)
        axes[0, c].set_title(case["titulo"], fontsize=7.4)
        axes[2, c].set_xlabel("Minutos respecto del origen")
    for r, tag in enumerate(("Referencias y\nbalance de masa", "Estadísticos\ny Chronos", "LightGBM con\notra información")):
        axes[r, 0].set_ylabel(tag + " (ppm)", fontsize=7, fontweight="semibold")
    handles = [Line2D([], [], color=INK, lw=1.3, label="CO₂ medido")] + _handles(MODEL_KEYS, lw=1.1) + [Line2D([], [], color=col, lw=1.1, marker="o", ms=2.8, mec=WHITE, label=lab) for _, lab, col in info_rows]
    _legend_top(fig, handles, ncol=5, y=0.995, fontsize=6.6)
    if not name:  # sin título (en la tesis va en el pie)
        return fig
    title, _, detail = name.partition(": ")
    fig.text(0.085, 0.905, title, fontsize=8.6, fontweight="semibold", ha="left")
    if detail:
        fig.text(0.085, 0.89, textwrap.fill(detail[0].upper() + detail[1:], 140), fontsize=7.2, color=INK2, ha="left", va="top", linespacing=1.3)
    return fig


def fig_composite(comp: dict, L: Lang, rows, title: str | None = None):
    """Trayectoria media alrededor del inicio de subidas y bajadas, con el origen antes, al inicio y después.

    `comp[(situación, desfase)]`: dict con x, medido, pron (clave -> media) y n.
    `rows`: [(clave, nombre, color, estilo de línea), ...].
    """
    situations = list(dict.fromkeys(s for s, _ in comp))
    offsets = sorted(set(o for _, o in comp))
    fig, axes = plt.subplots(len(situations), len(offsets), figsize=(WIDTH, 2.3 * len(situations) + 1.1), sharey="row", squeeze=False)
    height = 2.3 * len(situations) + (1.1 if title else 0.85)
    fig.set_size_inches(WIDTH, height)
    fig.subplots_adjust(left=0.085, right=0.99, top=1 - (1.0 if title else 0.72) / height, bottom=0.55 / height, hspace=0.42, wspace=0.07)
    knots = np.r_[0, HORIZONS]
    for r, s in enumerate(situations):
        for c, off in enumerate(offsets):
            ax = axes[r, c]
            d = comp.get((s, off))
            if d is None:
                ax.set_visible(False)
                continue
            ax.axvspan(d["x"][0], 0, color="#f3f2ee", lw=0, zorder=0)
            ax.axvline(-off, color=MUTED, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
            ax.plot(d["x"], d["medido"], color=INK, lw=1.5, zorder=3)
            for key, _, color, ls in rows:
                if key in d["pron"]:
                    ax.plot(knots, np.r_[d["c0"], d["pron"][key]], color=color, ls=ls, lw=1.1, marker="o", ms=2.6, mec=WHITE, mew=0.4, zorder=4)
            ax.axvline(0, color=INK2, lw=0.7)
            ax.set_xlim(d["x"][0], 62)
            ax.xaxis.set_major_locator(FixedLocator([-30, 0, 30, 60]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "t" if v == 0 else f"{v:+.0f}".replace("-", "−")))
            ax.yaxis.set_major_formatter(L.tick)
            when = "al empezar" if off == 0 else (f"{abs(off)} min antes" if off < 0 else f"{off} min después")
            ax.set_title(f"{s}: origen {when} (n = {d['n']})", fontsize=7.3)
            if r == len(situations) - 1:
                ax.set_xlabel("Minutos respecto del origen")
        axes[r, 0].set_ylabel("CO₂ medio (ppm)", fontsize=7)
    handles = [Line2D([], [], color=INK, lw=1.5, label="CO₂ medido (media)"), Line2D([], [], color=MUTED, lw=0.8, ls=(0, (1, 1.5)), label="inicio del evento")] + [Line2D([], [], color=col, ls=ls, lw=1.1, marker="o", ms=2.6, mec=WHITE, label=lab) for _, lab, col, ls in rows]
    _legend_top(fig, handles, ncol=5, y=0.995, fontsize=6.8)
    if title:
        fig.text(0.085, 1 - 0.78 / height, title, fontsize=8.5, fontweight="semibold", ha="left")
    return fig
