"""Figuras del análisis de mediciones reales, en español y con el estilo de las de los experimentos."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, MaxNLocator, MultipleLocator, NullLocator

from .mediciones import CONFIGS, METHODS, Selection, events, hourly_profile, known_occupancy
from .plots_exp import AXIS, BLUE, BLUE_DARK, BLUE_LIGHT, INK, INK2, METHOD_COLOR, METHOD_MARKER, MUTED, RED, SEQ_BLUE, WHITE, WIDTH, Lang, _grid
from .segmenters.base import RISE
from .ventilacion import GROUP_NAMES

METHOD_NAMES = {"M1": "Hyun fiel", "M2": "Hyun + decaimientos", "M3a": "Hyun + C″ en K-means", "M3b": "Hyun + quiebres", "M4": "Derivada", "M5": "PELT", "M6": "HMM", "manual": "Marcado a mano (referencia)"}
FAMILIES = {"familia_hyun": ("M1", "M2", "M3a", "M3b"), "alternativas": ("M4", "M5", "M6")}
CONFIG_NAMES = {"tal_cual": "tal cual", "ruido_efectivo": "con el ruido efectivo"}
MONTHS = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")
MONTHS_LONG = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre")
WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
ACH_LIM = (0.03, 40.0)
RED_LIGHT = "#f2a9a8"
EMPTY = "#eceae3"  # fondo del horario en que el salón está vacío


def rule_text(sel: Selection, L: Lang) -> str:
    """La regla de reporte, en palabras."""
    return f"RMSE de hasta {L.g(100 * sel.max_rmse_rel)} % del recorrido y al menos {L.g(sel.min_lam_t)} constante de tiempo"


def _month_axis(ax, index) -> None:
    starts = [d for d in pd.date_range(index[0], index[-1], freq="MS")]
    ax.set_xticks([(d - index[0]).days for d in starts])
    ax.set_xticklabels([MONTHS[d.month - 1] for d in starts])
    ax.xaxis.set_minor_locator(NullLocator())


# ------------------------------------------------------------------ datos y ruido


def fig_data(df: pd.DataFrame, daily: pd.DataFrame, days: pd.DataFrame, L: Lang):
    """El año del salón: mapa día × hora, perfil horario de los días con clase y máximo diario."""
    fig = plt.figure(figsize=(WIDTH, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.25, 1.0), left=0.08, right=0.935, top=0.94, bottom=0.085, hspace=0.42, wspace=0.26)

    ax = fig.add_subplot(gs[0, :])
    t = df["tiempo"]
    quarter = t.dt.hour * 4 + t.dt.minute // 15
    grid = df.groupby([t.dt.normalize(), quarter])["co2"].median().unstack(0).reindex(index=range(96), columns=daily.index)
    cmap = SEQ_BLUE.copy()
    cmap.set_bad(WHITE)
    vmax = float(np.nanpercentile(grid.to_numpy(), 99.5))
    im = ax.imshow(np.ma.masked_invalid(grid.to_numpy()), origin="lower", aspect="auto", cmap=cmap, vmin=400, vmax=vmax, extent=(-0.5, len(daily) - 0.5, 0, 24), interpolation="nearest")
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_ylabel("Hora del día")
    ax.grid(False)
    _month_axis(ax, daily.index)
    ax.set_title("CO₂ medido: cada columna es un día")
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03)
    cb.outline.set_visible(False)
    cb.set_label("CO₂ (ppm)", fontsize=7.5, color=INK2)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.ax.yaxis.set_major_formatter(L.tick)

    ax = fig.add_subplot(gs[1, 0])
    prof = hourly_profile(df, daily.index[daily["ocupado"]])
    ax.fill_between(prof.index, prof["q25"], prof["q75"], color=BLUE, alpha=0.22, lw=0)
    ax.plot(prof.index, prof["mediana"], color=BLUE_DARK, lw=1.5)
    ax.set_xlim(0, 24)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("CO₂ (ppm)")
    ax.yaxis.set_major_formatter(L.tick)
    ax.set_title("Un día con clase: mediana y rango intercuartil")

    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(len(daily))
    used = daily.index.isin(days.index)
    ax.scatter(x[~used], daily["maximo"][~used], s=5, color=MUTED, alpha=0.55, lw=0, label="otros días")
    ax.scatter(x[used], daily["maximo"][used], s=7, color=BLUE_DARK, lw=0, label="días analizados")
    _month_axis(ax, daily.index)
    ax.set_xlim(-3, len(daily) + 2)
    ax.set_ylabel("Máximo del día (ppm)")
    ax.yaxis.set_major_formatter(L.tick)
    ax.set_title("Máximo de CO₂ de cada día")
    ax.legend(loc="upper left", handletextpad=0.2, markerscale=1.6)
    return fig


def fig_noise(info: pd.DataFrame, L: Lang):
    """Fluctuación de la señal según la escala de tiempo: con gente en el salón y en el resto del día."""
    fig, axes, _ = _grid(1, 1, row_h=2.7, top_in=0.35, bottom_in=0.5, left=0.1, right=0.72, width=WIDTH)
    ax = axes[0, 0]
    lags = sorted(float(c.split("_")[1]) for c in info.columns if c.startswith("ocupado_"))
    for prefix, color, label in (("ocupado", BLUE_DARK, "con gente en el salón"), ("resto", MUTED, "resto del día")):
        q = np.array([[info[f"{prefix}_{lag:g}"].quantile(p) for lag in lags] for p in (0.25, 0.5, 0.75)])
        ax.fill_between(lags, q[0], q[2], color=color, alpha=0.2, lw=0)
        ax.plot(lags, q[1], color=color, lw=1.5, marker="o", ms=3.8, mec=WHITE, mew=0.5, label=label)
    sensor = float(info["sensor"].median())
    ax.axhline(sensor, color=INK, lw=0.9, zorder=1)
    ax.text(lags[-1], sensor * 1.12, f"ruido que estiman los métodos: {L.num(sensor, 1)} ppm", ha="right", va="bottom", fontsize=7, color=INK2)
    ef = float(info["ruido_efectivo"].median())
    ax.scatter([5.0], [ef], s=46, facecolor=WHITE, edgecolor=INK, lw=1.2, zorder=6)
    ax.annotate(f"ruido efectivo: {L.num(ef, 0)} ppm", xy=(5.0, ef), xytext=(-8, 6), textcoords="offset points", ha="right", va="bottom", fontsize=7.2, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(lags))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(L.tick)
    ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(L.tick)
    ax.set_xlabel("Escala de tiempo (min)")
    ax.set_ylabel("Fluctuación del CO₂ (ppm)")
    fig.legend(loc="center left", bbox_to_anchor=(0.735, 0.55), handlelength=1.4, fontsize=7.2)
    return fig


# ------------------------------------------------------------------ resúmenes


def fig_counts(fits: pd.DataFrame, segs: pd.DataFrame, n_days: int, sel: Selection, L: Lang):
    """Tramos, ajustes aceptados y eventos por día, tal cual y con el ruido efectivo."""
    panels = (
        ("Tramos detectados por día", segs.groupby(["config", "metodo"]).size() / n_days),
        ("Ajustes aceptados por día", fits[fits["aceptado"]].groupby(["config", "metodo"]).size() / n_days),
        ("Eventos por día", events(fits, sel).groupby(["config", "metodo"]).size() / n_days),
    )
    fig, axes, _ = _grid(1, 3, row_h=2.25, top_in=0.5, bottom_in=0.3, left=0.2, wspace=0.12, sharex=False)
    colors = {"tal_cual": BLUE_LIGHT, "ruido_efectivo": BLUE_DARK}
    for ax, (title, values) in zip(axes[0], panels):
        top = 0.0
        for y, key in enumerate(METHODS):
            xs = [float(values.get((c, key), 0.0)) for c in CONFIGS]
            ax.plot(xs, [y, y], color=AXIS, lw=1.3, zorder=2, solid_capstyle="butt")
            for c, x in zip(CONFIGS, xs):
                ax.scatter([x], [y], s=28, color=colors[c], edgecolor=WHITE, lw=0.7, zorder=4 if c == "ruido_efectivo" else 3)
            ax.text(max(xs), y, "  " + " → ".join(L.num(x, 1 if x < 20 else 0) for x in xs), ha="left", va="center", fontsize=6.8, color=INK2)
            top = max(top, max(xs))
        ax.set_xlim(0, 1.75 * top)
        ax.set_ylim(len(METHODS) - 0.5, -0.6)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([METHOD_NAMES[k] for k in METHODS], fontsize=7.4)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(L.tick)
        ax.set_title(title)
    for ax in axes[0, 1:]:
        ax.tick_params(labelleft=False)
    handles = [Line2D([], [], marker="o", ls="", ms=5, mfc=colors[c], mec=WHITE, label=CONFIG_NAMES[c]) for c in CONFIGS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.19, 1.0), ncol=2, handlelength=1.0, columnspacing=1.6)
    return fig


def fig_ach(fits: pd.DataFrame, sel: Selection, L: Lang):
    """Cómo se reparte el ACH de los eventos de cada método, en subidas y bajadas."""
    ev = events(fits, sel)
    fig, axes, _ = _grid(1, 2, row_h=2.9, top_in=0.5, bottom_in=0.45, left=0.2, right=0.91, wspace=0.34)
    for ax, phase in zip(axes[0], ("subida", "bajada")):
        sub = ev[ev["fase"] == phase]
        for y, key in enumerate(METHODS):
            g = sub[sub["metodo"] == key]
            for dy, data, color in ((-0.17, g, BLUE_LIGHT), (0.17, g[g["reportado"]], BLUE_DARK)):
                if len(data) < 5:
                    continue
                q = data["ach"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_numpy()
                ax.plot([q[0], q[4]], [y + dy] * 2, color=color, lw=1.0, solid_capstyle="butt", zorder=2)
                ax.plot([q[1], q[3]], [y + dy] * 2, color=color, lw=4.0, solid_capstyle="butt", zorder=3)
                ax.scatter([q[2]], [y + dy], s=20, color=WHITE, edgecolor=color, lw=1.2, zorder=4)
            ax.text(1.02, y, f"{int(g['reportado'].sum())} de {len(g)}" if len(g) else "no ajusta", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.8, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(*ACH_LIM)
        ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(L.tick)
        ax.set_ylim(len(METHODS) - 0.5, -0.6)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([METHOD_NAMES[k] for k in METHODS], fontsize=7.4)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("ACH (1/h)")
        ax.set_title("Subidas" if phase == "subida" else "Bajadas")
    handles = [
        Line2D([], [], color=BLUE_LIGHT, lw=4.0, label="todos los eventos"),
        Line2D([], [], color=BLUE_DARK, lw=4.0, label=f"reportados ({rule_text(sel, L)})"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.19, 1.0), ncol=2, handlelength=1.4, columnspacing=1.6)
    return fig


def fig_agreement(agree_all: pd.DataFrame, agree_reported: pd.DataFrame, L: Lang, vmax: float = 150.0):
    """Diferencia mediana de ACH entre parejas de métodos sobre los eventos que comparten: todos y sólo los reportados."""
    keys = list(METHODS)
    names = [METHOD_NAMES[k] for k in keys]
    tables = (("todos los eventos", agree_all), ("sólo los reportados", agree_reported))
    fig, axes, _ = _grid(2, 2, row_h=2.75, top_in=0.3, bottom_in=1.0, left=0.2, right=0.87, wspace=0.08, hspace=0.1)
    image = None
    for r, (row_name, agree) in enumerate(tables):
        for c, phase in enumerate(("subida", "bajada")):
            ax = axes[r, c]
            sub = agree[agree["fase"] == phase].set_index(["a", "b"])
            mat = np.full((len(keys), len(keys)), np.nan)
            for i, a in enumerate(keys):
                for j, b in enumerate(keys[:i]):
                    if (a, b) in sub.index and sub.loc[(a, b), "emparejamientos"] >= 10:
                        mat[i, j] = sub.loc[(a, b), "desacuerdo_mediano_pct"]
            image = ax.imshow(np.ma.masked_invalid(mat), cmap=SEQ_BLUE, vmin=0, vmax=vmax, aspect="auto")
            for i in range(len(keys)):
                for j in range(i):
                    if np.isfinite(mat[i, j]):
                        ax.text(j, i, L.num(mat[i, j], 0), ha="center", va="center", fontsize=7.2, color=WHITE if mat[i, j] > 0.58 * vmax else INK)
            ax.set_xticks(range(len(keys) - 1))
            ax.set_xticklabels(names[:-1], rotation=35, ha="right", fontsize=7.2)
            ax.set_yticks(range(1, len(keys)))
            ax.set_yticklabels(names[1:], fontsize=7.4)
            ax.set_xlim(-0.5, len(keys) - 1.5)
            ax.set_ylim(len(keys) - 0.5, 0.5)
            ax.grid(False)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title("Subidas" if phase == "subida" else "Bajadas")
            if c == 1:
                ax.text(1.03, 0.5, row_name, transform=ax.transAxes, rotation=270, ha="left", va="center", fontsize=7.6, color=INK2)
    cax = fig.add_axes([0.935, 0.35, 0.012, 0.4])
    cb = fig.colorbar(image, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("Diferencia mediana de ACH (%)", fontsize=7.5, color=INK2)
    cb.ax.tick_params(labelsize=7, length=0)
    return fig


THRESHOLD_AXES = {
    "r2": ("R² mínimo", lambda v, L: L.g(v)),
    "rmse_rel": ("RMSE máximo\n(% del recorrido)", lambda v, L: "sin" if v >= 1 else L.g(100 * v)),
    "se_rel": ("Error estándar relativo\nmáximo (%)", lambda v, L: "sin" if v >= 5 else L.g(100 * v)),
    "lam_t": ("Duración mínima\n(constantes de tiempo)", lambda v, L: "sin" if v <= 0 else L.g(v)),
}


def fig_thresholds(study: pd.DataFrame, sel: Selection, L: Lang):
    """Efecto de exigir más calidad a los ajustes: eventos que quedan y diferencia entre métodos, por criterio."""
    criteria = list(THRESHOLD_AXES)
    fig, axes, _ = _grid(2, len(criteria), row_h=1.9, top_in=0.5, bottom_in=0.7, left=0.085, hspace=0.16, wspace=0.12, sharex=False, sharey=False)
    chosen = {"rmse_rel": sel.max_rmse_rel, "lam_t": sel.min_lam_t}
    top_diff = 1.08 * float(study["diferencia_mediana_pct"].max())
    for c, name in enumerate(criteria):
        label, fmt = THRESHOLD_AXES[name]
        sub = study[study["criterio"] == name]
        values = list(dict.fromkeys(sub["umbral"]))
        x = np.arange(len(values))
        for r, col in enumerate(("quedan_pct", "diferencia_mediana_pct")):
            ax = axes[r, c]
            for phase, color in (("subida", RED), ("bajada", BLUE_DARK)):
                y = sub[sub["fase"] == phase].set_index("umbral")[col].reindex(values).to_numpy()
                ax.plot(x, y, color=color, lw=1.5, marker="o", ms=4, mec=WHITE, mew=0.5, zorder=4)
            if name in chosen and chosen[name] in values:
                ax.axvline(values.index(chosen[name]), color=INK2, lw=0.8, zorder=1)
            ax.set_xticks(x)
            ax.set_xticklabels([fmt(v, L) for v in values] if r == 1 else [], fontsize=7)
            ax.set_xlim(-0.4, len(values) - 0.6)
            ax.set_ylim(0, 105 if r == 0 else top_diff)
            ax.yaxis.set_major_formatter(L.tick)
            if c > 0:
                ax.tick_params(labelleft=False)
        axes[1, c].set_xlabel(label, fontsize=7.2)
    axes[0, 0].set_ylabel("Eventos que quedan (%)", fontsize=7.4)
    axes[1, 0].set_ylabel("Diferencia mediana\nentre métodos (%)", fontsize=7.4)
    handles = [
        Line2D([], [], color=RED, lw=1.5, marker="o", ms=4, mec=WHITE, label="subidas"),
        Line2D([], [], color=BLUE_DARK, lw=1.5, marker="o", ms=4, mec=WHITE, label="bajadas"),
        Line2D([], [], color=INK2, lw=0.8, label="umbral que se usa para reportar"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=3, handlelength=1.5, columnspacing=1.6)
    return fig


# ------------------------------------------------------------------ un día


def _fit_info(series, f, sel: Selection) -> dict:
    curve = f.predict(series.t_h[[f.i0, f.i1 - 1]])
    excursion = abs(float(curve[1] - curve[0]))
    event = excursion >= sel.min_excursion_ppm
    good = event and f.rmse <= sel.max_rmse_rel * excursion and f.lam * f.duration_h >= sel.min_lam_t
    return {"evento": event, "firme": bool(good)}


def _label_bars(fig, ax, items, fontsize: float = 6.3) -> None:
    """Valor de ACH sobre cada barra; si no entra arriba prueba abajo, y si tampoco, lo omite. Primero las barras firmes y largas."""
    renderer = fig.canvas.get_renderer()
    placed = []
    for it in sorted(items, key=lambda d: (not d["firme"], d["x0"] - d["x1"])):
        for dy, va in ((3.0, "bottom"), (-3.0, "top")):
            t = ax.annotate(it["text"], xy=(0.5 * (it["x0"] + it["x1"]), it["y"]), xytext=(0, dy), textcoords="offset points", ha="center", va=va, fontsize=fontsize, color=INK if it["firme"] else INK2, zorder=7, annotation_clip=False)
            box = t.get_window_extent(renderer)
            inside = ax.get_window_extent(renderer)
            if not any(box.overlaps(other) for other in placed) and box.y1 <= inside.y1 + 2 and box.y0 >= inside.y0 - 1:
                placed.append(box)
                break
            t.remove()


def fig_day(series, results: dict, keys, config: str, noise: float, sel: Selection, L: Lang, hours=(6.0, 24.0)):
    """Un día: para cada método, la serie con sus tramos y ajustes y, debajo, el ACH de cada ajuste aceptado."""
    n = len(keys)
    height = 1.72 * n + 1.3
    fig = plt.figure(figsize=(WIDTH, height))
    outer = fig.add_gridspec(n, 1, left=0.085, right=0.985, top=1.0 - 1.2 / height, bottom=0.42 / height, hspace=0.34)
    t = series.t_h
    show = (t >= hours[0]) & (t <= hours[1])
    span = series.c_obs[show].max() - series.c_obs[show].min()
    for r, key in enumerate(keys):
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[r], height_ratios=(1.0, 0.62), hspace=0.1)
        top, bot = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        res = results[key]
        segs = res.segmentation.transients
        for s in segs:
            top.axvspan(t[s.i0], t[s.i1 - 1], color=RED if s.phase == RISE else BLUE, alpha=0.16, lw=0, zorder=0)
        top.scatter(t[show], series.c_obs[show], s=1.4, color=MUTED, lw=0, zorder=2, rasterized=True)
        items, n_events = [], 0
        for f in res.fits:
            if not (f.ok and np.isfinite(f.lam)):
                continue
            tt = t[f.i0 : f.i1]
            top.plot(tt, f.predict(tt), color=INK if f.accepted else MUTED, lw=1.2 if f.accepted else 0.7, zorder=5 if f.accepted else 3)
            if not f.accepted:
                continue
            kind = _fit_info(series, f, sel)
            n_events += int(kind["evento"])
            x0, x1, y = max(float(tt[0]), hours[0]), min(float(tt[-1]), hours[1]), float(np.clip(f.lam, ACH_LIM[0] * 1.15, ACH_LIM[1] / 1.15))
            if x1 <= x0:  # el ajuste cae fuera de las horas que se dibujan
                continue
            color = (BLUE_DARK if f.phase != RISE else RED) if kind["firme"] else (BLUE_LIGHT if f.phase != RISE else RED_LIGHT)
            if np.isfinite(f.lam_se):
                lo, hi = max(f.lam - 1.96 * f.lam_se, ACH_LIM[0]), min(f.lam + 1.96 * f.lam_se, ACH_LIM[1])
                if hi > lo:
                    bot.fill_between([x0, x1], lo, hi, color=color, alpha=0.22, lw=0, zorder=2)
            bot.plot([x0, x1], [y, y], color=color, lw=2.0, solid_capstyle="butt", zorder=5 if kind["firme"] else 4)
            items.append({"x0": x0, "x1": x1, "y": y, "firme": kind["firme"], "text": L.num(f.lam, 2 if f.lam < 1 else 1)})
        top.set_title(METHOD_NAMES[key], fontsize=8.2, pad=3)
        top.text(1.0, 1.03, f"{len(segs)} tramos · {n_events} eventos", transform=top.transAxes, ha="right", va="bottom", fontsize=6.9, color=INK2)
        top.set_ylim(series.c_obs[show].min() - 0.06 * span, series.c_obs[show].max() + 0.08 * span)
        top.yaxis.set_major_locator(MaxNLocator(nbins=3))
        top.yaxis.set_major_formatter(L.tick)
        top.tick_params(labelbottom=False)
        bot.set_yscale("log")
        bot.set_ylim(*ACH_LIM)
        bot.yaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
        bot.yaxis.set_minor_locator(NullLocator())
        bot.yaxis.set_major_formatter(L.tick)
        for ax in (top, bot):
            ax.set_xlim(*hours)
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(NullLocator())
        if r < n - 1:
            bot.tick_params(labelbottom=False)
        else:
            bot.set_xlabel("Hora del día")
        top.set_ylabel("CO₂ (ppm)", fontsize=7.2)
        bot.set_ylabel("ACH (1/h)", fontsize=7.2)
        fig.align_ylabels([top, bot])
        _label_bars(fig, bot, items)
    day = pd.Timestamp(series.name)
    fig.text(0.085, 1.0 - 0.72 / height, f"{WEEKDAYS[day.dayofweek].capitalize()} {day.day} de {MONTHS_LONG[day.month - 1]} de {day.year}", ha="left", va="bottom", fontsize=9, fontweight="semibold", color=INK)
    detail = f"ruido efectivo como piso: {L.num(noise, 0)} ppm" if config == "ruido_efectivo" else f"tal cual, con el ruido que estiman los métodos: {L.num(noise, 1)} ppm"
    fig.text(0.085, 1.0 - 0.92 / height, f"{detail} · CO₂ exterior supuesto: {L.num(series.c_ext_nominal, 0)} ppm", ha="left", va="bottom", fontsize=7.6, color=INK2)
    handles = [
        Patch(color=RED, alpha=0.3, lw=0, label="subida"),
        Patch(color=BLUE, alpha=0.3, lw=0, label="bajada"),
        Line2D([], [], color=INK, lw=1.2, label="ajuste aceptado"),
        Line2D([], [], color=MUTED, lw=0.7, label="ajuste descartado"),
        Line2D([], [], color=INK2, lw=2.0, label="ACH reportado"),
        Line2D([], [], color=AXIS, lw=2.0, label="ACH de los demás ajustes aceptados"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=3, handlelength=1.3, columnspacing=1.3, handletextpad=0.5)
    return fig


def _curve(f, t):
    """Curva de una fila de la tabla de ajustes."""
    return f.c_inf + (f.c0 - f.c_inf) * np.exp(-f.ach * (t - f.t0_h))


def _label_curves(fig, ax, items, fontsize: float = 6.0) -> None:
    """Valor de ACH junto a cada curva, del lado en que no hay datos; si se pisa con otro o se sale, prueba del otro lado y, si tampoco, lo omite."""
    renderer = fig.canvas.get_renderer()
    inside = ax.get_window_extent(renderer)
    placed = []
    for it in sorted(items, key=lambda d: d["x0"] - d["x1"]):
        above = ("right", -2.5) if it["rise"] else ("left", 2.5)
        below = ("left", 2.5) if it["rise"] else ("right", -2.5)
        for (ha, dx), dy, va in ((above, 2.5, "bottom"), (below, -2.5, "top")):
            t = ax.annotate(it["text"], xy=(it["x"], it["y"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=fontsize, color=INK, zorder=7, annotation_clip=False)
            box = t.get_window_extent(renderer)
            if not any(box.overlaps(other) for other in placed) and inside.x0 <= box.x0 and box.x1 <= inside.x1 and inside.y0 <= box.y0 and box.y1 <= inside.y1:
                placed.append(box)
                break
            t.remove()


def fig_day_compact(series, fits_day: pd.DataFrame, segs_day: pd.DataFrame, noise: float, sel: Selection, L: Lang, hours=(7.0, 21.0), methods=METHODS, occupancy=None, cases=()):
    """Un día en una sola figura: la serie arriba, una fila por método con sus tramos, sus ajustes y su ACH, y todos los métodos juntos abajo.

    `fits_day` y `segs_day` son las filas de ese día de las tablas de ajustes y
    de tramos. En cada fila, el sombreado son los tramos que detecta el método
    y cada curva, un ajuste aceptado sobre el CO2 medido: gruesa y con su ACH si
    se reporta, fina si no. Abajo se superponen los ACH de todos los métodos
    para ver si coinciden.

    Con `occupancy` (y las columnas de `refit_empty` en `fits_day`) se marca el
    horario en que el salón está vacío, se dibuja en trazos el reajuste de las
    bajadas que caen ahí y se agrega un último panel con los ACH de ese reajuste.
    Con `cases` (los casos de generación de `refit_occupied`, de menor a mayor)
    se hace lo mismo con los tramos con gente: en trazos, el ajuste con la
    generación del caso del medio y, en el último panel, su ACH con una línea
    vertical que va del caso menor al mayor.
    """
    day = pd.Timestamp(series.name)
    if day.dayofweek >= 5 or "ach_puro" not in fits_day:
        occupancy = None
    cases = tuple(cases) if occupancy is not None and all(f"ach_{c}" in fits_day for c in cases) else ()
    mid = cases[len(cases) // 2] if cases else None
    ev = events(fits_day, sel)
    accepted = fits_day[fits_day["aceptado"] & fits_day["ach"].notna()]
    reported = set(ev.index[ev["reportado"]])
    n = len(methods)
    row_h, gap, sep = 0.82, 0.2, 0.36
    heights = [1.2, n * row_h + (n - 1) * gap] + [1.15] * (2 if occupancy is not None else 1)
    top_in, bottom_in = 1.45, 0.75
    height = sum(heights) + sep * (len(heights) - 1) + top_in + bottom_in
    fig = plt.figure(figsize=(WIDTH, height))
    outer = fig.add_gridspec(len(heights), 1, height_ratios=heights, left=0.085, right=0.985, top=1.0 - top_in / height, bottom=bottom_in / height, hspace=sep / np.mean(heights))
    t, c = series.t_h, series.c_obs
    show = (t >= hours[0]) & (t <= hours[1])
    span = c[show].max() - c[show].min()
    windows = [(max(a, hours[0]), min(b, hours[1])) for a, b in occupancy.empty if min(b, hours[1]) > max(a, hours[0])] if occupancy is not None else []

    def time_axis(ax, labels=False):
        ax.set_xlim(*hours)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelbottom=labels)

    def shade_empty(ax):
        for a, b in windows:
            ax.axvspan(a, b, color=EMPTY, lw=0, zorder=0)

    def ach_axis(ax):
        ax.set_yscale("log")
        ax.set_ylim(*ACH_LIM)
        ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(L.tick)
        ax.set_ylabel("ACH (1/h)", fontsize=7.4)

    def tag(ax, text, y=1.04):
        ax.text(0.0, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2, fontweight="semibold", color=INK)

    ax = fig.add_subplot(outer[0])
    shade_empty(ax)
    ax.plot(t[show], c[show], color=INK, lw=0.9)
    ax.set_ylabel("CO₂ (ppm)", fontsize=7.4)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(L.tick)
    time_axis(ax)

    inner = GridSpecFromSubplotSpec(n, 1, subplot_spec=outer[1], hspace=gap / row_h)
    for r, key in enumerate(methods):
        ax = fig.add_subplot(inner[r])
        segs = segs_day[segs_day["metodo"] == key]
        for s_ in segs.itertuples():
            ax.axvspan(s_.t0_h, s_.t1_h, color=RED if s_.fase == "subida" else BLUE, alpha=0.12, lw=0, zorder=0)
        ax.plot(t[show], c[show], color=AXIS, lw=2.0, zorder=2)
        items = []
        for f in accepted[accepted["metodo"] == key].itertuples():
            m = show & (t >= f.t0_h - 1e-9) & (t <= f.t1_h + 1e-9)
            if m.sum() < 2:
                continue
            firm, rise = f.Index in reported, f.fase == "subida"
            y = _curve(f, t[m])
            ax.plot(t[m], y, color=RED if rise else BLUE_DARK, lw=1.3 if firm else 0.65, zorder=5 if firm else 4)
            if occupancy is not None:
                for suffix in ("puro", mid) if mid else ("puro",):  # reajuste con lo que se sabe de la ocupación
                    lam, c0, c_inf = (getattr(f, f"{name}_{suffix}") for name in ("ach", "c0", "c_inf"))
                    if np.isfinite(lam):
                        ax.plot(t[m], c_inf + (c0 - c_inf) * np.exp(-lam * (t[m] - f.t0_h)), color=INK, lw=0.8, ls=(0, (2.2, 1.6)), zorder=6)
            if firm:
                k = int(m.sum()) // 2
                items.append({"x": float(t[m][k]), "y": float(y[k]), "x0": f.t0_h, "x1": f.t1_h, "rise": rise, "text": L.num(f.ach, 2 if f.ach < 1 else 1)})
        for a, b in windows:  # el horario vacío, como una barra en el borde de abajo: el fondo ya lo usan los tramos
            ax.plot([a, b], [0.0, 0.0], transform=ax.get_xaxis_transform(), color=INK2, lw=2.4, solid_capstyle="butt", clip_on=False, zorder=6)
        ax.set_ylim(c[show].min() - 0.08 * span, c[show].max() + 0.2 * span)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(L.tick)
        ax.tick_params(axis="y", labelsize=6.6)
        time_axis(ax)
        tag(ax, METHOD_NAMES[key])
        ax.text(1.0, 1.04, f"{len(segs)} tramos · {len(items)} ACH reportados", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6, color=INK2)
        _label_curves(fig, ax, items)
        if r == n // 2:
            ax.set_ylabel("CO₂ (ppm)", fontsize=7.4)

    shown = [k for k in methods if k != "M1"]  # las subidas de Hyun fiel son las de Hyun + decaimientos

    def overlay(ax, table):
        shade_empty(ax)
        for key in shown:
            for f in table[table["metodo"] == key].itertuples():
                x0, x1 = max(f.t0_h, hours[0]), min(f.t1_h, hours[1])
                if x1 > x0 and np.isfinite(f.ach):
                    y = float(np.clip(f.ach, ACH_LIM[0] * 1.15, ACH_LIM[1] / 1.15))
                    ax.plot([x0, x1], [y, y], color=METHOD_COLOR[key], lw=2.0 if f.reportado else 0.8, alpha=1.0 if f.reportado else 0.45, solid_capstyle="butt", zorder=4 if f.reportado else 3)
        ach_axis(ax)
        time_axis(ax)

    ax = fig.add_subplot(outer[2])
    overlay(ax, ev)
    tag(ax, "Todos los métodos: ACH de cada ajuste", 1.02)
    if occupancy is not None:
        known = known_occupancy(ev, sel, case=mid)
        used = known["en_vacio"] & known["ach_puro"].notna()
        if mid:
            busy = known["con_gente"].astype(bool) & known[f"ach_{mid}"].notna() & known["metodo"].isin(shown)
            used = used | busy
        ax = fig.add_subplot(outer[3])
        overlay(ax, known[used])
        if mid:  # de qué ACH a qué ACH va cada tramo con gente entre el caso de menor y el de mayor generación
            for f in known[busy].itertuples():
                x = 0.5 * (max(f.t0_h, hours[0]) + min(f.t1_h, hours[1]))
                lo, hi = getattr(f, f"ach_{cases[0]}"), getattr(f, f"ach_{cases[-1]}")
                lo = ACH_LIM[0] if not np.isfinite(lo) and getattr(f, f"cota_{cases[0]}") == "baja" else lo
                hi = ACH_LIM[1] if not np.isfinite(hi) and getattr(f, f"cota_{cases[-1]}") == "alta" else hi
                if hours[0] < x < hours[1] and np.isfinite(lo) and np.isfinite(hi):
                    ax.plot([x, x], np.clip([lo, hi], *ACH_LIM), color=METHOD_COLOR[f.metodo], lw=0.9, alpha=1.0 if f.reportado else 0.45, solid_capstyle="butt", zorder=2)
            tag(ax, f"Con la ocupación conocida: vacío, sin generación; con gente, caso {mid} (línea vertical: de {cases[0]} a {cases[-1]})", 1.02)
        else:
            tag(ax, "Bajadas en horario vacío, reajustadas con la asíntota en el CO₂ exterior", 1.02)
    time_axis(ax, labels=True)
    ax.set_xlabel("Hora del día")

    fig.text(0.085, 1.0 - 0.9 / height, f"{WEEKDAYS[day.dayofweek].capitalize()} {day.day} de {MONTHS_LONG[day.month - 1]} de {day.year}", ha="left", va="bottom", fontsize=9, fontweight="semibold", color=INK)
    fig.text(0.085, 1.0 - 1.1 / height, f"ruido efectivo: {L.num(noise, 0)} ppm · CO₂ exterior supuesto: {L.num(series.c_ext_nominal, 0)} ppm · se reporta con {rule_text(sel, L)}", ha="left", va="bottom", fontsize=7.2, color=INK2)
    handles = [
        Patch(color=RED, alpha=0.25, lw=0, label="tramo de subida"),
        Patch(color=BLUE, alpha=0.25, lw=0, label="tramo de bajada"),
        Line2D([], [], color=AXIS, lw=2.0, label="CO₂ medido"),
        Line2D([], [], color=INK2, lw=1.3, label="ajuste que se reporta, con su ACH"),
        Line2D([], [], color=INK2, lw=0.65, label="ajuste aceptado que no se reporta"),
    ]
    if occupancy is not None:
        handles += [
            Line2D([], [], color=INK, lw=0.8, ls=(0, (2.2, 1.6)), label="reajuste con la ocupación conocida" if mid else "reajuste con el salón vacío"),
            Patch(color=EMPTY, lw=0, label="salón vacío según el horario"),
        ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=3, handlelength=1.3, columnspacing=1.6, handletextpad=0.5)
    method_handles = [Line2D([], [], color=METHOD_COLOR[k], lw=2.0, label=METHOD_NAMES[k]) for k in shown]
    fig.legend(handles=method_handles, loc="lower left", bbox_to_anchor=(0.075, 0.0), ncol=len(shown), handlelength=1.3, columnspacing=1.2 if len(shown) <= 6 else 0.8, handletextpad=0.5, fontsize=7 if len(shown) <= 6 else 6.2)
    return fig


def fig_occupancy(check: pd.DataFrame, effect: pd.DataFrame, occupancy, L: Lang, hours=(7.0, 19.0)):
    """El horario contra los datos y lo que cambia en las bajadas en horario vacío al fijar la asíntota en el exterior."""
    top_h, low_h, top_in, mid_in, bottom_in = 1.9, 2.5, 0.4, 0.95, 0.45
    height = top_h + low_h + top_in + mid_in + bottom_in
    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes([0.085, (bottom_in + low_h + mid_in) / height, 0.9, top_h / height])
    for a, b in occupancy.empty:
        if min(b, hours[1]) > max(a, hours[0]):
            ax.axvspan(max(a, hours[0]), min(b, hours[1]), color=EMPTY, lw=0, zorder=0)
    sub = check[(check["hora"] >= hours[0]) & (check["hora"] < hours[1])]
    x = sub["hora"].to_numpy() + 0.5 * float(np.diff(check["hora"].to_numpy()[:2])[0])
    ax.plot(x, sub["sube_pct"], color=RED, lw=1.5, zorder=4)
    ax.plot(x, sub["baja_pct"], color=BLUE_DARK, lw=1.5, zorder=4)
    ax.set_xlim(*hours)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Días hábiles (%)", fontsize=7.4)

    names = list(effect["ventana"])
    rows = np.arange(len(names))
    left = fig.add_axes([0.17, bottom_in / height, 0.36, low_h / height])
    right = fig.add_axes([0.57, bottom_in / height, 0.3, low_h / height])
    for y, f in zip(rows, effect.itertuples()):
        for dy, pre, color in ((-0.17, "ach_libre", BLUE_LIGHT), (0.17, "ach_vacio", BLUE_DARK)):
            left.plot([getattr(f, f"{pre}_p25"), getattr(f, f"{pre}_p75")], [y + dy] * 2, color=color, lw=4.0, solid_capstyle="butt", zorder=3)
            left.scatter([getattr(f, f"{pre}_mediana")], [y + dy], s=20, color=WHITE, edgecolor=color, lw=1.2, zorder=4)
        a, b = f.diferencia_entre_metodos_libre_pct, f.diferencia_entre_metodos_vacio_pct
        right.plot([a, b], [y, y], color=AXIS, lw=1.2, zorder=2)
        right.scatter([a], [y], s=26, color=BLUE_LIGHT, edgecolor=WHITE, lw=0.6, zorder=4)
        right.scatter([b], [y], s=26, color=BLUE_DARK, edgecolor=WHITE, lw=0.6, zorder=4)
        right.text(1.03, y, f"{f.bajadas} bajadas", transform=right.get_yaxis_transform(), ha="left", va="center", fontsize=6.8, color=INK2)
    left.set_xscale("log")
    left.set_xlim(0.1, 12.0)
    left.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10]))
    left.xaxis.set_minor_locator(NullLocator())
    left.xaxis.set_major_formatter(L.tick)
    left.set_xlabel("ACH (1/h)")
    left.set_title("ACH de las bajadas en horario vacío")
    right.set_xlim(0, 1.15 * float(effect[["diferencia_entre_metodos_libre_pct", "diferencia_entre_metodos_vacio_pct"]].max().max()))
    right.xaxis.set_major_formatter(L.tick)
    right.set_xlabel("Diferencia mediana de ACH (%)")
    right.set_title("Diferencia entre métodos")
    for axis in (left, right):
        axis.set_ylim(len(names) - 0.5, -0.6)
        axis.set_yticks(rows)
        axis.grid(axis="y", visible=False)
    left.set_yticklabels(names, fontsize=7.4)
    right.tick_params(labelleft=False)
    handles = [
        Line2D([], [], color=RED, lw=1.5, label="el CO₂ sube"),
        Line2D([], [], color=BLUE_DARK, lw=1.5, label="el CO₂ baja"),
        Patch(color=EMPTY, lw=0, label="salón vacío según el horario"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=3, handlelength=1.5, columnspacing=1.6)
    low = [
        Line2D([], [], color=BLUE_LIGHT, lw=4.0, label="asíntota libre, como sale de cada método"),
        Line2D([], [], color=BLUE_DARK, lw=4.0, label="asíntota en el CO₂ exterior (salón vacío)"),
    ]
    fig.legend(handles=low, loc="upper left", bbox_to_anchor=(0.075, (bottom_in + low_h + 0.62) / height), ncol=2, handlelength=1.4, columnspacing=1.6)
    return fig


def fig_manual_scatter(comp: pd.DataFrame, L: Lang, methods=METHODS, lim=(0.05, 30.0)):
    """ACH de cada método contra el del evento marcado a mano que le corresponde, un panel por método."""
    ncols = 4
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH, 1.95 * nrows + 0.95), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.985, top=1.0 - 0.55 / fig.get_figheight(), bottom=0.5 / fig.get_figheight(), hspace=0.3, wspace=0.08)
    grid = np.array(lim)
    for ax, key in zip(axes.ravel(), methods):
        sub = comp[(comp["metodo"] == key) & comp["dlog"].notna()]
        ax.fill_between(grid, grid / 1.2, grid * 1.2, color=EMPTY, lw=0, zorder=0)
        ax.plot(grid, grid, color=AXIS, lw=0.9, zorder=1)
        for phase, color in (("subida", RED), ("bajada", BLUE_DARK)):
            g = sub[sub["fase"] == phase]
            ax.scatter(g["ach_manual"], g["ach_metodo"], s=11, color=color, alpha=0.75, lw=0, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator([0.1, 1, 10]))
            axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(L.tick)
        ax.set_title(METHOD_NAMES[key], fontsize=7.8, loc="left")
        total = int(((comp["metodo"] == key) & comp["ach_manual"].notna()).sum())
        note = f"{len(sub)} de {total} eventos" + (f"\ndiferencia mediana {L.num(100.0 * (np.exp(sub['dlog'].abs().median()) - 1.0), 0)} %" if len(sub) else "")
        ax.text(0.97, 0.04, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4, color=INK2, linespacing=1.35, zorder=5, bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.2, "alpha": 0.85})
    for ax in axes.ravel()[len(methods) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("ACH del evento manual (1/h)", fontsize=7.2)
    for row in axes:
        row[0].set_ylabel("ACH del método (1/h)", fontsize=7.2)
    for ax in axes.ravel()[: len(methods)]:
        ax.tick_params(labelbottom=True)
    handles = [
        Line2D([], [], color=RED, lw=0, marker="o", ms=4, label="subidas"),
        Line2D([], [], color=BLUE_DARK, lw=0, marker="o", ms=4, label="bajadas"),
        Patch(color=EMPTY, lw=0, label="a menos de 20 % del valor manual"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=3, handlelength=1.2, columnspacing=1.6)
    return fig


def fig_manual_detection(summary: pd.DataFrame, L: Lang, methods=METHODS):
    """Cuánto de lo marcado a mano detecta cada método, en cuántos fragmentos, con qué desfase y cuántos eventos agrega."""
    panels = (
        ("detectados_pct", "Eventos manuales\ndetectados (%)", (0, 100)),
        ("fragmentos_medios", "Tramos del método\npor evento manual", None),
        ("desfase_inicio_abs_mediano_min", "Desfase del inicio\n(min, mediana)", None),
        ("eventos_de_mas_por_dia", "Eventos de más\npor día", None),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH, 0.3 * len(methods) + 1.45), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.985, top=1.0 - 0.8 / fig.get_figheight(), bottom=0.42 / fig.get_figheight(), wspace=0.12)
    for ax, (col, title, xlim) in zip(axes, panels):
        for y, key in enumerate(methods):
            for dy, phase, color in ((-0.18, "subida", RED), (0.18, "bajada", BLUE_DARK)):
                row = summary[(summary["metodo"] == key) & (summary["fase"] == phase)]
                if len(row) and np.isfinite(row[col].iloc[0]):
                    value = float(row[col].iloc[0])
                    ax.barh(y + dy, value, height=0.3, color=color, lw=0, zorder=3)
                    ax.text(value, y + dy, " " + L.num(value, 0 if col == "detectados_pct" else 1), ha="left", va="center", fontsize=6.2, color=INK2, zorder=4)
        ax.set_ylim(len(methods) - 0.45, -0.6)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_NAMES[k] for k in methods], fontsize=7.2)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=7.4, loc="left")
        ax.xaxis.set_major_formatter(L.tick)
        if xlim:
            ax.set_xlim(xlim[0], xlim[1] * 1.18)
        else:
            ax.set_xlim(0, max(ax.get_xlim()[1] * 1.2, 1e-6))
    handles = [Patch(color=RED, lw=0, label="subidas"), Patch(color=BLUE_DARK, lw=0, label="bajadas")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.19, 1.0), ncol=2, handlelength=1.2, columnspacing=1.6)
    return fig


LESS_MORE = LinearSegmentedColormap.from_list("menos_mas", ["#0d366b", "#3987e5", "#a9c9f2", "#dedcd5", "#f2bb97", "#dd6a2c", "#8a3206"])  # el método da menos (azul) o más (naranja) que la referencia; gris, igual
PHASES = (("subida", "Subidas"), ("bajada", "Bajadas"))


def _firm(comp: pd.DataFrame, sel: Selection):
    """La referencia cumple el criterio de reporte: abarca lo suficiente y se parece a una exponencial."""
    return (comp["lam_t_manual"] >= sel.min_lam_t) & (comp["rmse_rel_manual"] <= sel.max_rmse_rel)


def _period(dates) -> str:
    """"octubre de 2024" si todas las fechas son de un mismo mes; si no, vacío."""
    stamps = pd.to_datetime(pd.Series(sorted(set(dates))))
    months = {(d.year, d.month) for d in stamps}
    if len(months) != 1:
        return ""
    year, month = months.pop()
    return f"{MONTHS_LONG[month - 1]} de {year}"


def _method_handles(methods, lw: float = 1.4, marker: bool = True):
    return [Line2D([], [], color=METHOD_COLOR[k], lw=lw, marker=METHOD_MARKER[k] if marker else None, ms=4, mec=WHITE, mew=0.4, label=METHOD_NAMES[k]) for k in methods]


def fig_manual_month(comp: pd.DataFrame, L: Lang, methods=METHODS, clip: float = 4.0, sel: Selection = Selection()):
    """Todos los eventos marcados a mano, uno por columna: ACH de la referencia y cuánto más o menos da cada método."""
    ref_h, map_h, gap_in, top_in, bottom_in = 0.8, 0.2 * len(methods), 0.7, 1.15, 0.5
    height = top_in + bottom_in + len(PHASES) * (ref_h + map_h + 0.06) + (len(PHASES) - 1) * gap_in
    fig = plt.figure(figsize=(WIDTH, height))
    left, width = 0.2, 0.785
    span = np.log2(clip)
    period = _period(comp["fecha"])
    y_in = height - top_in
    image = None
    for phase, title in PHASES:
        sub = comp[(comp["fase"] == phase) & comp["ach_manual"].notna()]
        marks = sub.drop_duplicates(["fecha", "t0_h", "t1_h"]).sort_values(["fecha", "t0_h"]).reset_index(drop=True)
        day = pd.factorize(marks["fecha"])[0]
        x = np.arange(len(marks)) + day  # una columna vacía entre un día y el siguiente
        where = {(r.fecha, r.t0_h): xi for r, xi in zip(marks.itertuples(), x)}
        grid = np.full((len(methods), int(x.max()) + 1 if len(x) else 1), np.nan)
        for r in sub.itertuples():
            if np.isfinite(r.dlog) and r.metodo in methods:
                grid[methods.index(r.metodo), where[(r.fecha, r.t0_h)]] = np.clip(r.dlog / np.log(2.0), -span, span)

        ax = fig.add_axes([left, (y_in - ref_h) / height, width, ref_h / height])
        firm = _firm(marks, sel).to_numpy()
        ax.scatter(x[firm], marks["ach_manual"][firm], s=9, color=INK, lw=0, zorder=3)
        ax.scatter(x[~firm], marks["ach_manual"][~firm], s=8, facecolor=WHITE, edgecolor=INK, lw=0.6, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(*ACH_LIM)
        ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(L.tick)
        ax.set_xlim(-0.5, grid.shape[1] - 0.5)
        ax.tick_params(labelbottom=False, bottom=False)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("ACH de la\nreferencia (1/h)", fontsize=6.8)
        ax.set_title(f"{title}: {len(marks)} eventos marcados a mano", loc="left")
        y_in -= ref_h + 0.06

        ax = fig.add_axes([left, (y_in - map_h) / height, width, map_h / height])
        cmap = LESS_MORE.copy()
        cmap.set_bad(WHITE)
        image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-span, vmax=span, aspect="auto", interpolation="nearest", extent=(-0.5, grid.shape[1] - 0.5, len(methods) - 0.5, -0.5))
        for k in range(1, len(methods)):
            ax.axhline(k - 0.5, color=WHITE, lw=1.0)
        for k in np.flatnonzero(~np.isfinite(grid).any(axis=1)):
            ax.text(0.5, k, f"no ajusta {title.lower()}", transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=6.6, color=INK2)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_NAMES[k] for k in methods], fontsize=7.0)
        centers = [x[day == d].mean() for d in range(day.max() + 1)] if len(x) else []
        ax.set_xticks(centers)
        ax.set_xticklabels([str(pd.Timestamp(d).day) for d in pd.unique(marks["fecha"])], fontsize=6.6)
        ax.tick_params(length=0)
        ax.grid(visible=False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        y_in -= map_h + gap_in
    ax.set_xlabel(f"Día de {period}" if period else "Día", fontsize=7.4)

    cax = fig.add_axes([0.63, 1.0 - 0.52 / height, 0.34, 0.09 / height])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal")
    ticks = [t for t in (0.25, 0.5, 1.0, 2.0, 4.0) if 1.0 / clip <= t <= clip]
    names = {0.25: "la cuarta parte\no menos", 0.5: "la mitad", 1.0: "igual", 2.0: "el doble", 4.0: "4 veces\no más"}
    bar.set_ticks(np.log2(ticks))
    bar.set_ticklabels([names[t] for t in ticks], fontsize=6.2)
    bar.outline.set_visible(False)
    cax.tick_params(length=0)
    cax.set_title("ACH del método respecto de la referencia", fontsize=6.8, loc="left", pad=3, fontweight="normal", color=INK2)
    handles = [
        Line2D([], [], color=INK, lw=0, marker="o", ms=3.4, label="referencia firme: una constante de tiempo o más y buen ajuste"),
        Line2D([], [], color=INK, lw=0, marker="o", ms=3.4, mfc=WHITE, mew=0.6, label="referencia poco determinada"),
        Patch(facecolor=WHITE, edgecolor=AXIS, lw=0.6, label="en blanco: el método no da un ACH para ese evento"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=1, handlelength=1.2, fontsize=6.8, labelspacing=0.35)
    return fig


def _spread(ax, y: float, dlog, color) -> None:
    """Mediana, rango intercuartil y percentiles 10 a 90 del cociente con la referencia."""
    q = np.exp(np.quantile(dlog, [0.1, 0.25, 0.5, 0.75, 0.9]))
    ax.plot([q[0], q[4]], [y, y], color=color, lw=0.9, solid_capstyle="butt", zorder=3)
    ax.plot([q[1], q[3]], [y, y], color=color, lw=3.4, solid_capstyle="butt", zorder=3)
    ax.scatter([q[2]], [y], s=16, color=WHITE, edgecolor=color, lw=1.1, zorder=4)


def fig_manual_ratio(comp: pd.DataFrame, comp_known, L: Lang, methods=METHODS, lim=(0.02, 50.0), min_pairs: int = 5):
    """Cuánto más o menos que la referencia da cada método: reparto del cociente, por fase, con la asíntota libre y con la ocupación conocida."""
    sets = [(comp, BLUE_LIGHT, "asíntota libre en los dos lados")] + ([(comp_known, BLUE_DARK, "con la ocupación conocida en los dos lados")] if comp_known is not None else [])
    step = 0.36 if len(sets) > 1 else 0.0
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 0.46 * len(methods) + 1.5), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.955, top=1.0 - 0.85 / fig.get_figheight(), bottom=0.48 / fig.get_figheight(), wspace=0.16)
    for ax, (phase, title) in zip(axes, PHASES):
        ax.axvspan(1 / 1.2, 1.2, color=EMPTY, lw=0, zorder=0)
        ax.axvline(1.0, color=AXIS, lw=0.9, zorder=1)
        for y, key in enumerate(methods):
            for k, (table, color, _) in enumerate(sets):
                d = table.loc[(table["metodo"] == key) & (table["fase"] == phase), "dlog"].dropna().to_numpy()
                yy = y + (k - 0.5) * step
                if len(d) >= min_pairs:
                    _spread(ax, yy, d, color)
                ax.text(1.012, yy, str(len(d)) if len(d) else "–", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.0, color=INK2)
        ax.text(1.012, -0.62, "eventos", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.0, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(*lim)
        ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(L.tick)
        ax.set_ylim(len(methods) - 0.4, -0.85)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_NAMES[k] for k in methods], fontsize=7.4)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("ACH del método / ACH de la referencia")
        ax.set_title(title)
    handles = [Line2D([], [], color=color, lw=3.4, label=label) for _, color, label in sets]
    handles += [Line2D([], [], color=MUTED, lw=0.9, label="percentiles 10 a 90"), Patch(color=EMPTY, lw=0, label="a menos de 20 % de la referencia")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.19, 1.0), ncol=2, handlelength=1.4, columnspacing=1.6)
    return fig


def fig_manual_hits(comp: pd.DataFrame, comp_known, L: Lang, methods=METHODS, lim=(5.0, 300.0)):
    """Porcentaje de los eventos manuales en que el ACH del método queda a menos de cierta diferencia de la referencia."""
    sets = [(comp, "asíntota libre")] + ([(comp_known, "ocupación conocida")] if comp_known is not None else [])
    fig, axes, height = _grid(len(sets), 2, row_h=2.05, top_in=0.82, bottom_in=0.5, hspace=0.34, wspace=0.08)
    x = np.geomspace(lim[0], lim[1], 160)
    for row, (table, name) in zip(axes, sets):
        for ax, (phase, title) in zip(row, PHASES):
            sub = table[(table["fase"] == phase) & table["ach_manual"].notna()]
            total = len(sub.drop_duplicates(["fecha", "t0_h", "t1_h"]))
            ax.axvline(20.0, color=AXIS, lw=0.9, zorder=1)
            for key in methods:
                d = sub.loc[sub["metodo"] == key, "dlog"].abs().to_numpy()
                if not len(d) or not np.isfinite(d).any():
                    continue
                share = 100.0 * (d[None, :] <= np.log1p(x[:, None] / 100.0)).mean(axis=1)  # sin ACH del método: nunca acierta
                ax.plot(x, share, color=METHOD_COLOR[key], lw=1.4, ls=(0, (3, 1.6)) if key == "M1" else "-", zorder=4 if key == "M1" else 3)
            ax.set_xscale("log")
            ax.set_xlim(*lim)
            ax.set_ylim(0, 100)
            ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 200]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(L.tick)
            ax.yaxis.set_major_locator(MultipleLocator(20))
            ax.set_title(f"{title} · {name}", loc="left")
            ax.text(0.03, 0.95, f"{total} eventos manuales", transform=ax.transAxes, ha="left", va="top", fontsize=6.6, color=INK2)
    for ax in axes[-1]:
        ax.set_xlabel("Diferencia admitida con la referencia (%)")
    for row in axes:
        row[0].set_ylabel("Eventos manuales (%)", fontsize=7.4)
    handles = [Line2D([], [], color=METHOD_COLOR[k], lw=1.6, ls=(0, (3, 1.6)) if k == "M1" else "-", label=METHOD_NAMES[k]) for k in methods]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=4, handlelength=1.7, columnspacing=1.4)
    return fig


def fig_manual_by_duration(table: pd.DataFrame, L: Lang, methods=METHODS, min_pairs: int = 5, sel: Selection = Selection()):
    """Diferencia con la referencia según cuántas constantes de tiempo abarca el evento manual."""
    fig, axes, height = _grid(1, 2, row_h=2.5, top_in=0.82, bottom_in=0.82, wspace=0.08, sharex=False)
    for ax, (phase, title) in zip(axes[0], PHASES):
        sub = table[table["fase"] == phase]
        bins = sub.drop_duplicates(["desde", "hasta"])
        x = np.arange(len(bins))
        edges = bins["desde"].to_numpy()
        firm = x[edges >= sel.min_lam_t]
        if len(firm):
            ax.axvspan(firm[0] - 0.5, len(bins) - 0.5, color=EMPTY, lw=0, zorder=0)
        for k, key in enumerate(methods):
            g = sub[sub["metodo"] == key].set_index("desde").reindex(edges)
            y = g["diferencia_mediana_pct"].where(g["con_ach"] >= min_pairs).to_numpy(dtype=float)
            dx = 0.06 * (k - 0.5 * (len(methods) - 1))
            ax.plot(x + dx, y, color=METHOD_COLOR[key], lw=1.1, marker=METHOD_MARKER[key], ms=4.2, mec=WHITE, mew=0.4, ls=(0, (3, 1.6)) if key == "M1" else "-", zorder=3)
        labels = [f"menos de {L.g(b)}" if a <= 0 else (f"{L.g(a)} o más" if not np.isfinite(b) else f"{L.g(a)} a {L.g(b)}") for a, b in zip(bins["desde"], bins["hasta"])]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{label}\n{n} eventos" for label, n in zip(labels, bins["eventos_manuales"])], fontsize=6.8)
        ax.set_xlim(-0.5, len(bins) - 0.5)
        ax.set_yscale("log")
        ax.set_ylim(5.0, 3000.0)
        ax.yaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 300, 1000]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(L.tick)
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
    fig.text(0.5 * (0.09 + 0.985), 0.1 / height, "Constantes de tiempo que abarca el evento manual (ACH de referencia × duración)", ha="center", va="bottom", fontsize=7.6, color=INK2)
    axes[0][0].set_ylabel("Diferencia mediana con la referencia (%)", fontsize=7.4)
    handles = _method_handles(methods) + [Patch(color=EMPTY, lw=0, label="una constante de tiempo o más")]
    for h, key in zip(handles, methods):
        if key == "M1":
            h.set_linestyle((0, (3, 1.6)))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=4, handlelength=1.7, columnspacing=1.4)
    return fig


DAILY_TITLES = {
    "clase": "Con la clase adentro: subidas, con la generación conocida",
    "pausa": "En recreos y mediodía: bajadas hacia el CO₂ exterior",
    "salida": "Al irse la clase: bajadas hacia el CO₂ exterior",
    "noche": "De noche, cerrado: bajadas hacia el CO₂ exterior",
}


def fig_manual_daily(per: pd.DataFrame, dates, L: Lang, methods=METHODS, groups=("clase", "pausa", "salida"), reference: str = "manual", min_days: int = 3):
    """ACH de cada día por grupo de bloques del horario: la referencia manual y cada método.

    Sólo se dibujan los grupos en que la referencia tiene ACH en al menos `min_days` días.
    """
    dates = sorted(dates)
    x = np.arange(len(dates))
    with_ref = per[per["metodo"] == reference].groupby("grupo")["fecha"].nunique()
    groups = [g for g in groups if with_ref.get(g, 0) >= min_days] or list(groups[:1])
    fig, axes, height = _grid(len(groups), 1, row_h=1.75, top_in=0.85, bottom_in=0.62, hspace=0.3, sharey=True)
    for ax, group in zip(axes[:, 0], groups):
        sub = per[per["grupo"] == group].pivot_table(index="fecha", columns="metodo", values="ach").reindex(dates)
        both = [k for k in methods if k in sub.columns]
        values = sub[both].to_numpy(dtype=float) if both else np.full((len(dates), 0), np.nan)
        has = np.isfinite(values).any(axis=1)
        if has.any():
            ax.vlines(x[has], np.nanmin(values[has], axis=1), np.nanmax(values[has], axis=1), color=AXIS, lw=0.8, zorder=1)
        for k, key in enumerate(both):
            dx = 0.09 * (k - 0.5 * (len(both) - 1))
            ax.scatter(x + dx, sub[key], s=13, color=METHOD_COLOR[key], marker=METHOD_MARKER[key], lw=0.3, edgecolor=WHITE, zorder=3)
        if reference in sub.columns:
            ax.plot(x, sub[reference], color=INK, lw=1.2, zorder=4)
            ax.scatter(x, sub[reference], s=20, color=INK, edgecolor=WHITE, lw=0.5, zorder=5)
        ax.set_yscale("log")
        ax.set_ylim(*ACH_LIM)
        ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(L.tick)
        ax.set_ylabel("ACH (1/h)")
        ax.set_title(DAILY_TITLES.get(group, GROUP_NAMES.get(group, group)), loc="left")
        ax.grid(axis="x", visible=False)
    stamps = [pd.Timestamp(d) for d in dates]
    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels([f"{WEEKDAYS[d.dayofweek][:2]}\n{d.day}" for d in stamps], fontsize=6.6)
    axes[-1, 0].set_xlim(-0.6, len(dates) - 0.4)
    period = _period(dates)
    axes[-1, 0].set_xlabel(f"Día de {period}" if period else "Día")
    handles = [Line2D([], [], color=INK, lw=1.2, marker="o", ms=4.2, mec=WHITE, mew=0.5, label=METHOD_NAMES[reference])]
    handles += [Line2D([], [], color=METHOD_COLOR[k], lw=0, marker=METHOD_MARKER[k], ms=4.2, mec=WHITE, mew=0.3, label=METHOD_NAMES[k]) for k in methods]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=4, handlelength=1.5, columnspacing=1.4)
    return fig


def fig_manual_example(series, ref_day: pd.DataFrame, L: Lang, hours=(7.5, 18.5), occupancy=None):
    """Un día con los eventos marcados a mano: cada uno sombreado según su fase, con el exponencial de referencia y su ACH."""
    fig, ax = plt.subplots(figsize=(WIDTH, 2.75))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.83, bottom=0.17)
    t = series.t_h
    keep = (t >= hours[0]) & (t <= hours[1])
    if occupancy is not None:
        for a, b in occupancy.empty:
            if b > hours[0] and a < hours[1]:
                ax.axvspan(max(a, hours[0]), min(b, hours[1]), ymin=0.0, ymax=0.035, color=INK2, lw=0, zorder=1)
    ax.plot(t[keep], series.c_obs[keep], color=MUTED, lw=1.6, zorder=2)
    items = []
    for f in ref_day.sort_values("t0_h").itertuples():
        rise = f.fase == "subida"
        ax.axvspan(f.t0_h, f.t1_h, color=RED_LIGHT if rise else BLUE_LIGHT, alpha=0.28, lw=0, zorder=0)
        if not (f.aceptado and np.isfinite(f.ach)):
            continue
        grid = np.linspace(f.t0_h, f.t1_h, 60)
        curve = _curve(f, grid)
        ax.plot(grid, curve, color=RED if rise else BLUE_DARK, lw=1.3, zorder=4)
        k = len(grid) // 2
        items.append({"text": L.num(f.ach, 2 if f.ach < 1 else 1), "x": grid[k], "y": curve[k], "x0": f.t0_h, "x1": f.t1_h, "rise": rise})
    ax.set_xlim(*hours)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(L.tick)
    ax.yaxis.set_major_formatter(L.tick)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("CO₂ (ppm)")
    fig.canvas.draw()
    _label_curves(fig, ax, items, fontsize=6.2)
    day = pd.Timestamp(series.name)
    ax.set_title(f"{WEEKDAYS[day.dayofweek].capitalize()} {day.day} de {MONTHS_LONG[day.month - 1]} de {day.year}: {len(ref_day)} eventos marcados a mano", loc="left", fontsize=7.8)
    handles = [
        Line2D([], [], color=MUTED, lw=1.6, label="CO₂ medido"),
        Patch(color=RED_LIGHT, alpha=0.5, lw=0, label="subida marcada"),
        Patch(color=BLUE_LIGHT, alpha=0.5, lw=0, label="bajada marcada"),
        Line2D([], [], color=INK, lw=1.3, label="ajuste de referencia, con su ACH (1/h)"),
    ] + ([Patch(color=INK2, lw=0, label="aula vacía según el horario")] if occupancy is not None else [])
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=5, handlelength=1.3, columnspacing=1.2, fontsize=6.8)
    return fig


def fig_marks_changes(changes: pd.DataFrame, L: Lang, lim=(0.03, 40.0)):
    """Qué cambia al corregir la puntería de las marcas: cuánto se recorta cada borde y cómo queda el ACH de referencia."""
    fig = plt.figure(figsize=(WIDTH, 2.95))
    ax = fig.add_axes([0.075, 0.17, 0.33, 0.63])
    edges = np.arange(-0.25, 3.3, 0.5)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for dx, col, color, label in ((-0.11, "recorte_inicio_min", INK2, "inicio"), (0.11, "recorte_fin_min", AXIS, "fin")):
        counts, _ = np.histogram(changes[col], bins=edges)
        ax.bar(centers + dx, counts, width=0.2, color=color, lw=0, zorder=3, label=label)
    ax.set_xticks(centers[::2])
    ax.xaxis.set_major_formatter(L.tick)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Minutos recortados")
    ax.set_ylabel("Eventos")
    ax.set_title("Cuánto se corre cada borde")
    ax.legend(loc="upper right", handlelength=1.0)

    ax = fig.add_axes([0.52, 0.17, 0.27, 0.63])
    grid = np.array(lim)
    ax.fill_between(grid, grid / 1.2, grid * 1.2, color=EMPTY, lw=0, zorder=0)
    ax.plot(grid, grid, color=AXIS, lw=0.9, zorder=1)
    for phase, color in (("subida", RED), ("bajada", BLUE_DARK)):
        g = changes[changes["fase"] == phase]
        ax.scatter(g["ach_antes"], g["ach_despues"], s=10, color=color, alpha=0.75, lw=0, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator([0.1, 1, 10]))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(L.tick)
    ax.set_xlabel("ACH con la marca original (1/h)")
    ax.set_ylabel("ACH con la marca corregida (1/h)", fontsize=7.4)
    ax.set_title("ACH de referencia")
    handles = [Line2D([], [], color=RED, lw=0, marker="o", ms=4, label="subidas"), Line2D([], [], color=BLUE_DARK, lw=0, marker="o", ms=4, label="bajadas"), Patch(color=EMPTY, lw=0, label="cambia menos de 20 %")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.51, 1.0), ncol=3, handlelength=1.1, columnspacing=1.2, fontsize=7.0)

    ax = fig.add_axes([0.875, 0.17, 0.115, 0.63])
    firm = {tag: (changes[f"lam_t_{tag}"] >= 1.0) & (changes[f"rmse_rel_{tag}"] <= 0.05) for tag in ("antes", "despues")}
    for x, (phase, color) in enumerate((("subida", RED), ("bajada", BLUE_DARK))):
        m = changes["fase"] == phase
        a, b = int(firm["antes"][m].sum()), int(firm["despues"][m].sum())
        ax.bar([x - 0.19, x + 0.19], [a, b], width=0.34, color=[RED_LIGHT if phase == "subida" else BLUE_LIGHT, color], lw=0, zorder=3)
        for xx, v in ((x - 0.19, a), (x + 0.19, b)):
            ax.text(xx, v, str(v), ha="center", va="bottom", fontsize=6.4, color=INK2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["subidas", "bajadas"], fontsize=6.8)
    ax.grid(axis="x", visible=False)
    ax.set_title("Referencias\nfirmes", fontsize=7.8)
    ax.text(0.5, -0.2, "antes → después", transform=ax.transAxes, ha="center", va="top", fontsize=6.6, color=INK2)
    return fig


def fig_start_sensitivity(summary: pd.DataFrame, L: Lang, limit: float = 0.2):
    """Cuánto cambia el ACH de referencia si el inicio del evento se corre unos minutos."""
    fig, axes, height = _grid(1, 2, row_h=2.3, top_in=0.62, bottom_in=0.5, wspace=0.26, sharey=False)
    ax = axes[0][0]
    ax.axhspan(1.0 / (1.0 + limit), 1.0 + limit, color=EMPTY, lw=0, zorder=0)
    ax.axhline(1.0, color=AXIS, lw=0.9, zorder=1)
    for phase, color in (("subida", RED), ("bajada", BLUE_DARK)):
        g = summary[summary["fase"] == phase].sort_values("corrimiento_min")
        ax.fill_between(g["corrimiento_min"], g["cociente_p25"], g["cociente_p75"], color=color, alpha=0.18, lw=0, zorder=2)
        ax.plot(g["corrimiento_min"], g["cociente_mediano"], color=color, lw=1.5, marker="o", ms=4, mec=WHITE, mew=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.4, 3.2)
    ax.yaxis.set_major_locator(FixedLocator([0.5, 0.7, 1, 1.5, 2, 3]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(L.tick)
    ax.set_ylabel("ACH / ACH con el inicio marcado", fontsize=7.4)
    ax.set_title("Cuánto cambia el ACH de referencia")
    ax = axes[0][1]
    col = f"cambian_mas_de_{round(100 * limit)}_pct"
    for phase, color in (("subida", RED), ("bajada", BLUE_DARK)):
        g = summary[summary["fase"] == phase].sort_values("corrimiento_min")
        ax.plot(g["corrimiento_min"], g[col], color=color, lw=1.5, marker="o", ms=4, mec=WHITE, mew=0.5, zorder=3)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_ylabel(f"Eventos que cambian más de {round(100 * limit)} % (%)", fontsize=7.4)
    ax.set_title("A cuántos eventos les cambia")
    for ax in axes[0]:
        ax.axvline(0.0, color=AXIS, lw=0.9, zorder=1)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlabel("Minutos que se corre el inicio (negativo: antes; positivo: después)", fontsize=7.0)
    handles = [
        Line2D([], [], color=RED, lw=1.5, marker="o", ms=4, mec=WHITE, label="subidas"),
        Line2D([], [], color=BLUE_DARK, lw=1.5, marker="o", ms=4, mec=WHITE, label="bajadas"),
        Patch(color=MUTED, alpha=0.3, lw=0, label="rango intercuartil"),
        Patch(color=EMPTY, lw=0, label=f"cambia menos de {round(100 * limit)} %"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=4, handlelength=1.5, columnspacing=1.4)
    return fig


def fig_marks_day(series, clean, changes_day: pd.DataFrame, fit, L: Lang, ncols: int = 5, context_min: float = 10.0):
    """Las marcas de un día, una por panel: los clics originales (trazos), los corregidos (línea llena) y el ajuste de referencia que queda.

    `fit(i0, i1, fase)` devuelve el ajuste de un tramo.
    """
    t = series.t_h
    per_hour = 1.0 / (t[1] - t[0])
    n = len(changes_day)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH, 1.42 * nrows + 0.75), squeeze=False)
    fig.subplots_adjust(left=0.055, right=0.99, top=1.0 - 0.62 / fig.get_figheight(), bottom=0.22 / fig.get_figheight(), hspace=0.62, wspace=0.34)
    pad = int(round(context_min / 60.0 * per_hour))
    for ax, ev in zip(axes.ravel(), changes_day.itertuples()):
        a0, a1 = int(round(ev.t0_h * per_hour)), int(round(ev.t1_h * per_hour)) + 1
        b0, b1 = int(round(ev.t0_nuevo_h * per_hour)), int(round(ev.t1_nuevo_h * per_hour)) + 1
        lo, hi = max(a0 - pad, 0), min(a1 + pad, len(t))
        color = RED if ev.fase == "subida" else BLUE_DARK
        ax.plot(t[lo:hi], series.c_obs[lo:hi], color=MUTED, lw=0.9, zorder=2)
        ax.plot(t[b0:b1], fit(b0, b1, ev.fase).predict(t[b0:b1]), color=color, lw=1.5, zorder=3)
        for x in (t[a0], t[a1 - 1]):
            ax.axvline(x, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
        for x in (t[b0], t[b1 - 1]):
            ax.axvline(x, color=INK, lw=0.8, zorder=4)
        same = abs(ev.ach_despues / ev.ach_antes - 1.0) < 0.005
        ach = L.num(ev.ach_despues, 2) if same else f"{L.num(ev.ach_antes, 2)} → {L.num(ev.ach_despues, 2)}"
        warn = [name for name, flag in (("va en contra", ev.va_en_contra), ("mal ajuste", ev.mal_ajuste), ("se pisa", ev.se_pisa)) if flag]
        ax.set_title(f"{_clock(ev.t0_h)} · ACH {ach}" + (f"\n{', '.join(warn)}" if warn else ""), fontsize=6.4, loc="left", pad=2.5, color=INK if not warn else RED)
        ax.tick_params(labelsize=5.6, length=1.5)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.xaxis.set_major_formatter(lambda v, _: _clock(v))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.grid(visible=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    day = pd.Timestamp(series.name)
    fig.text(0.055, 1.0 - 0.2 / fig.get_figheight(), f"{WEEKDAYS[day.dayofweek].capitalize()} {day.day} de {MONTHS_LONG[day.month - 1]} de {day.year}", ha="left", va="top", fontsize=8.5, fontweight="semibold", color=INK)
    handles = [
        Line2D([], [], color=INK2, lw=0.7, ls=(0, (3, 2)), label="clic original"),
        Line2D([], [], color=INK, lw=0.8, label="borde corregido"),
        Line2D([], [], color=RED, lw=1.5, label="ajuste de una subida"),
        Line2D([], [], color=BLUE_DARK, lw=1.5, label="ajuste de una bajada"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=4, handlelength=1.6, columnspacing=1.3, fontsize=6.8)
    return fig


def _clock(h: float) -> str:
    m = int(round(h * 60.0))
    return f"{m // 60:02d}:{m % 60:02d}"


def fig_before_after(before: pd.DataFrame, after: pd.DataFrame, L: Lang, methods=METHODS, columns=(("diferencia_mediana_pct", "Diferencia mediana\ncon la referencia (%)"), ("aciertos_20_pct", "Eventos manuales a menos\nde 20 % de la referencia (%)"))):
    """Los resúmenes de la comparación con las marcas originales y con las corregidas, por método y fase."""
    fig, axes = plt.subplots(1, 2 * len(columns), figsize=(WIDTH, 0.3 * len(methods) + 1.6), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.985, top=1.0 - 0.95 / fig.get_figheight(), bottom=0.42 / fig.get_figheight(), wspace=0.14)
    k = 0
    for col, title in columns:
        for phase, name in PHASES:
            ax = axes[k]
            k += 1
            color = RED if phase == "subida" else BLUE_DARK
            for y, key in enumerate(methods):
                a = before[(before["metodo"] == key) & (before["fase"] == phase)][col]
                b = after[(after["metodo"] == key) & (after["fase"] == phase)][col]
                if not len(a) or not len(b) or not np.isfinite(a.iloc[0]) or not np.isfinite(b.iloc[0]):
                    continue
                va, vb = float(a.iloc[0]), float(b.iloc[0])
                ax.plot([va, vb], [y, y], color=AXIS, lw=1.2, zorder=2)
                ax.scatter([va], [y], s=16, color=WHITE, edgecolor=color, lw=1.0, zorder=3)
                ax.scatter([vb], [y], s=16, color=color, lw=0, zorder=4)
            ax.set_ylim(len(methods) - 0.5, -0.5)
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels([METHOD_NAMES[m] for m in methods], fontsize=7.2)
            ax.grid(axis="y", visible=False)
            ax.set_xlim(left=0)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_formatter(L.tick)
            ax.set_title(f"{title}\n{name.lower()}", fontsize=7.0, loc="left")
    handles = [
        Line2D([], [], color=INK2, lw=0, marker="o", ms=4.2, mfc=WHITE, mew=1.0, label="marcas originales"),
        Line2D([], [], color=INK2, lw=0, marker="o", ms=4.2, label="marcas con la puntería corregida"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.19, 1.0), ncol=2, handlelength=1.2, columnspacing=1.6)
    return fig


CASE_COLORS = ("#9cc4f2", BLUE, "#123f78")  # de menor a mayor generación


def fig_generation(effect: pd.DataFrame, by_duration: pd.DataFrame, sources: dict, L: Lang):
    """Qué pasa al conocer la generación con un caso bajo, uno medio y uno alto: ACH por método, rango de ACH y generación que implican los ajustes libres."""
    cases = list(sources)
    lo, hi = cases[0], cases[-1]
    colors = dict(zip(cases, CASE_COLORS))
    row_h, low_h, top_in, mid_in, bottom_in = 3.0, 2.3, 0.75, 0.95, 0.45
    height = row_h + low_h + top_in + mid_in + bottom_in
    fig = plt.figure(figsize=(WIDTH, height))
    y_top = (bottom_in + low_h + mid_in) / height
    keys = [k for k in METHODS if k in set(effect["metodo"])]
    for c, (phase, title) in enumerate((("subida", "Subidas con la clase adentro"), ("bajada", "Bajadas con la clase adentro"))):
        ax = fig.add_axes([0.2 + 0.4 * c, y_top, 0.36, row_h / height])
        sub = effect[effect["fase"] == phase].set_index("metodo")
        for y, key in enumerate(keys):
            if key not in sub.index or sub.loc[key, "con_los_tres_casos"] < 5:
                ax.text(0.5, y, "no ajusta" if key not in sub.index else "", transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=6.8, color=INK2)
                continue
            f = sub.loc[key]
            for dy, pre, color in ((-0.3, "ach_libre", MUTED), *((-0.1 + 0.2 * k, f"ach_{case}", colors[case]) for k, case in enumerate(cases))):
                ax.plot([f[f"{pre}_p25"], f[f"{pre}_p75"]], [y + dy] * 2, color=color, lw=3.2, solid_capstyle="butt", zorder=3)
                ax.scatter([f[f"{pre}_mediana"]], [y + dy], s=16, color=WHITE, edgecolor=color, lw=1.1, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(*ACH_LIM)
        ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(L.tick)
        ax.set_ylim(len(keys) - 0.45, -0.6)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([METHOD_NAMES[k] for k in keys] if c == 0 else [], fontsize=7.4)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("ACH (1/h)")
        ax.set_title(title)

    ax = fig.add_axes([0.085, bottom_in / height, 0.33, low_h / height])
    bins = by_duration[by_duration["fase"] == "subida"]
    labels = [f"menos de {L.g(b)}" if a <= 0 else (f"{L.g(a)} o más" if not np.isfinite(b) else f"{L.g(a)} a {L.g(b)}") for a, b in zip(bins["desde"], bins["hasta"])]
    x = np.arange(len(labels))
    for dx, phase, color in ((-0.12, "subida", RED), (0.12, "bajada", BLUE_DARK)):
        sub = by_duration[by_duration["fase"] == phase]
        ax.plot(np.c_[x + dx, x + dx].T, np.c_[sub["rango_p25"], sub["rango_p75"]].T, color=color, lw=1.4, solid_capstyle="butt", zorder=3)
        ax.plot(x + dx, sub["rango_mediana"], color=color, lw=0, marker="o", ms=4.5, mec=WHITE, mew=0.5, zorder=4)
    ratio = sources[hi] / sources[lo]
    ax.axhline(ratio, color=INK2, lw=0.8, zorder=1)
    ax.text(0.99, ratio, f"generación: {L.num(ratio, 1)} veces", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.8, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(1.0, 20.0)
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(L.tick)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_xlabel("Duración del tramo (constantes de tiempo)")
    ax.set_ylabel(f"ACH del caso {hi} / ACH del caso {lo}", fontsize=7.4)
    ax.set_title("Cuánto se abre el ACH")

    ax = fig.add_axes([0.62, bottom_in / height, 0.36, low_h / height])
    sub = effect[effect["fase"] == "subida"].set_index("metodo")
    ax.axvspan(sources[lo] / 3600.0, sources[hi] / 3600.0, color=EMPTY, lw=0, zorder=0)
    for case in cases:
        ax.axvline(sources[case] / 3600.0, color=colors[case], lw=1.2, zorder=1)
    for y, key in enumerate(keys):
        if key in sub.index:
            f = sub.loc[key]
            ax.plot([f["fuente_libre_p25"] / 3600.0, f["fuente_libre_p75"] / 3600.0], [y, y], color=MUTED, lw=3.2, solid_capstyle="butt", zorder=3)
            ax.scatter([f["fuente_libre_mediana"] / 3600.0], [y], s=16, color=WHITE, edgecolor=MUTED, lw=1.1, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 8.0)
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(L.tick)
    ax.set_ylim(len(keys) - 0.45, -0.6)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([METHOD_NAMES[k] for k in keys], fontsize=7.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Generación por unidad de volumen, G/V (ppm/s)")
    ax.set_title("Generación que implican las subidas libres")

    handles = [Line2D([], [], color=MUTED, lw=3.2, label="generación libre, como sale de cada método")]
    handles += [Line2D([], [], color=colors[case], lw=3.2, label=f"caso {case}: G/V = {L.num(sources[case] / 3600.0, 2)} ppm/s") for case in cases]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=2, handlelength=1.4, columnspacing=1.6)
    low = [
        Line2D([], [], color=RED, lw=1.4, marker="o", ms=4.5, mec=WHITE, label="subidas"),
        Line2D([], [], color=BLUE_DARK, lw=1.4, marker="o", ms=4.5, mec=WHITE, label="bajadas"),
        Patch(color=EMPTY, lw=0, label="generación entre el caso más bajo y el más alto"),
    ]
    fig.legend(handles=low, loc="upper left", bbox_to_anchor=(0.075, (bottom_in + low_h + 0.62) / height), ncol=3, handlelength=1.5, columnspacing=1.6)
    return fig
