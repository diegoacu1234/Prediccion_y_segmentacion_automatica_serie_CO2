"""Figuras del análisis de la simulación CFD, con el estilo de las de los experimentos."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, MultipleLocator, NullLocator

from .cfd import CLOSED, OPEN, CfdCase, CfdResults, Schedule, fit_reference, fit_segments, match_fits, mean_series, schedule_segmentation
from .plots_exp import AXIS, BLUE, BLUE_DARK, BLUE_LIGHT, INK, INK2, MUTED, RED, SEQ_BLUE, WHITE, WIDTH, Lang, _grid
from .segmenters.base import RISE

WALL_COLOR = {"oeste": "#2a78d6", "este": "#eb6834", "norte": "#1baf7a", "sur": "#eda100", "centro": "#4a3aa7"}
WALL_ORDER = ("oeste", "norte", "centro", "sur", "este")
PERSON_LITRES_MIN = 0.254
DETAIL_METHODS = ("M0", "M1", "M2", "M3a", "M3b", "M4", "M5", "M6")

TXT = {
    "es": {
        "methods": {"M1": "Hyun fiel", "M2": "Hyun + decaimientos", "M3a": "Hyun + C″ en K-means", "M3b": "Hyun + quiebres", "M4": "Derivada", "M5": "PELT", "M6": "HMM", "M0": "Cortes del esquema"},
        "setups": {"estudio": "escalas del estudio", "corta": "escalas para eventos de 5 min"},
        "walls": {"oeste": "pared oeste", "este": "pared este", "norte": "pared norte", "sur": "pared sur", "centro": "centro"},
        "groups": {"todos": "todos los sensores"},
        "minute": "Tiempo (min)",
        "co2": "CO₂ (ppm)",
        "mean": "media",
        "open": "ventilación abierta",
        "sensors_at": "Sensores a {z} m",
        "n_sensors": "z = {z} m · {n} sensores",
        "x_m": "x (m)",
        "y_m": "y (m)",
        "ach": "ACH (1/h)",
        "ach_open": "abierto",
        "ach_closed": "cerrado",
        "source": "Generación supuesta, G/V (ppm/s)",
        "people": "personas cada 100 m³ (a {g} L/min por persona)",
        "rmse": "Error del ajuste, RMSE (ppm)",
        "same_fit": "Generaciones distintas, ajustes parecidos",
        "mixing": "{v} min de mezcla, sin ajustar",
        "free_fit": "G libre, cada ciclo",
        "truth_short": "simulación",
        "with_mixing": "con los minutos de mezcla",
        "without_mixing": "sin ellos",
        "ach_vs_source": "ACH de cada estado según la generación",
        "fit_error": "Error del ajuste según la generación",
        "curve": "G/V = {s} ppm/s: ACH {c} cerrado, {o} abierto",
        "data": "media de los sensores",
        "delta": "ΔACH ≈ {v}",
        "truth": "valor de la simulación",
        "simulation": "simulación",
        "calibrated": "generación calibrada",
        "all_short": " todos ",
        "dev_truth": "Desvío respecto de la simulación (%)",
        "panel_closed": "ACH cerrado",
        "panel_open": "ACH abierto",
        "panel_delta": "ΔACH, abierto − cerrado",
        "gen_free": "generación libre",
        "gen_bounded": "generación acotada",
        "gen_known": "generación conocida",
        "gen_known_s": "generación conocida (G/V = {s} ppm/s)",
        "gen_occupancy": "generación de la ocupación",
        "exact_cuts": "cortes exactos",
        "ach_est": "ACH estimado e intervalo de 95 %",
        "ach_extra": "estimación de más (no cuenta)",
        "ach_real": "ACH de la simulación ({o} abierto; {c} cerrado)",
        "within": "a ±20 %",
        "each_sensor": "cada sensor",
        "room_mean": "promedio de todos los sensores",
        "gen_series": "generación que mejor ajusta la serie",
        "gen_calibrated": "generación calibrada",
        "rise": "subida",
        "decay": "bajada",
        "fit_ok": "ajuste aceptado",
        "fit_no": "ajuste descartado",
        "cut": "corte del método",
        "change": "cambio del esquema",
        "onsets": "inicios {a} de {b}",
        "dach": "ΔACH {v}",
        "no_dach": "sin ΔACH",
        "detected": "Sensores que detectan el cambio (%)",
        "tol": "tolerancia ±{v} min",
        "close": "cierre {t}′",
        "opening": "apertura {t}′",
        "delta_ach": "ΔACH, abierto − cerrado (1/h)",
        "delta_at": "ΔACH a {z} m",
        "by_wall": "Por pared y altura",
        "all": "todos los sensores",
        "k_sensors": "Sensores promediados",
        "dev": "Desvío respecto de todos los sensores (%)",
        "median": "mediana",
        "iqr": "25–75 %",
        "p90": "5–95 %",
        "p90dev": "percentil 90",
        "schedule_cuts": "cortes del esquema",
        "own_cuts": "cortes del método",
        "height": "{z} m",
    },
    "en": {
        "methods": {"M1": "Hyun (as published)", "M2": "Hyun + decays", "M3a": "Hyun + C″ feature", "M3b": "Hyun + kink split", "M4": "Derivative", "M5": "PELT", "M6": "HMM", "M0": "Schedule breakpoints"},
        "setups": {"estudio": "time scales of the study", "corta": "time scales for 5-min events"},
        "walls": {"oeste": "west wall", "este": "east wall", "norte": "north wall", "sur": "south wall", "centro": "centre"},
        "groups": {"todos": "all sensors"},
        "minute": "Time (min)",
        "co2": "CO₂ (ppm)",
        "mean": "mean",
        "open": "ventilation open",
        "sensors_at": "Sensors at {z} m",
        "n_sensors": "z = {z} m · {n} sensors",
        "x_m": "x (m)",
        "y_m": "y (m)",
        "ach": "ACH (1/h)",
        "ach_open": "open",
        "ach_closed": "closed",
        "source": "Assumed generation, G/V (ppm/s)",
        "people": "people per 100 m³ (at {g} L/min per person)",
        "rmse": "Fit error, RMSE (ppm)",
        "same_fit": "Different sources, similar fits",
        "mixing": "{v} min of mixing, not fitted",
        "free_fit": "free G, each cycle",
        "truth_short": "simulation",
        "with_mixing": "with the mixing minutes",
        "without_mixing": "without them",
        "ach_vs_source": "ACH of each state against the source",
        "fit_error": "Fit error against the source",
        "curve": "G/V = {s} ppm/s: ACH {c} closed, {o} open",
        "data": "mean of the sensors",
        "delta": "ΔACH ≈ {v}",
        "truth": "value in the simulation",
        "simulation": "simulation",
        "calibrated": "calibrated generation",
        "all_short": " all ",
        "dev_truth": "Deviation from the simulation (%)",
        "panel_closed": "ACH, closed",
        "panel_open": "ACH, open",
        "panel_delta": "ΔACH, open − closed",
        "gen_free": "free generation",
        "gen_bounded": "bounded generation",
        "gen_known": "known generation",
        "gen_known_s": "known generation (G/V = {s} ppm/s)",
        "gen_occupancy": "generation from occupancy",
        "exact_cuts": "exact breakpoints",
        "ach_est": "estimated ACH and 95 % interval",
        "ach_extra": "extra estimate (not counted)",
        "ach_real": "ACH in the simulation ({o} open; {c} closed)",
        "within": "within ±20 %",
        "each_sensor": "each sensor",
        "room_mean": "mean of all sensors",
        "gen_series": "source that best fits the series",
        "gen_calibrated": "calibrated generation",
        "rise": "rise",
        "decay": "decay",
        "fit_ok": "accepted fit",
        "fit_no": "rejected fit",
        "cut": "method breakpoint",
        "change": "schedule change",
        "onsets": "onsets {a} of {b}",
        "dach": "ΔACH {v}",
        "no_dach": "no ΔACH",
        "detected": "Sensors detecting the change (%)",
        "tol": "tolerance ±{v} min",
        "close": "closing {t}′",
        "opening": "opening {t}′",
        "delta_ach": "ΔACH, open − closed (1/h)",
        "delta_at": "ΔACH at {z} m",
        "by_wall": "By wall and height",
        "all": "all sensors",
        "k_sensors": "Sensors averaged",
        "dev": "Deviation from all sensors (%)",
        "median": "median",
        "iqr": "25–75 %",
        "p90": "5–95 %",
        "p90dev": "90th percentile",
        "schedule_cuts": "schedule breakpoints",
        "own_cuts": "method breakpoints",
        "height": "{z} m",
    },
}


def _t(L: Lang, key: str, **kw) -> str:
    text = TXT[L.code][key]
    return text.format(**kw) if kw else text


def _name(L: Lang, name: str) -> str:
    """Nombre de un promedio en el idioma de la figura."""
    if name in TXT[L.code]["groups"]:
        return TXT[L.code]["groups"][name]
    for wall, label in TXT[L.code]["walls"].items():
        if name in (wall, f"pared {wall}"):
            return label
    if name.startswith("z ="):
        return f"z = {L.num(float(name[3:].replace('m', '').replace(',', '.')), 1)} m"
    return name


def _series_title(L: Lang, case: CfdCase, name: str) -> str:
    """Nombre de la serie para un encabezado; a un sensor le agrega dónde está."""
    if not name.startswith("sensor "):
        return _name(L, name)
    row = case.sensors[case.sensors["sensor"] == int(name.split()[1])].iloc[0]
    where = f"x = {L.num(row['x'], 1)} m, y = {L.num(row['y'], 1)} m, z = {L.num(row['z'], 1)} m"
    return f"{name} ({where}; {TXT[L.code]['walls'][row['pared']]})"


def _fmt_ach(L: Lang, value: float) -> str:
    return L.num(value, 2 if value < 3.0 else 1)


def _fmt_err(L: Lang, error: float) -> str:
    return ("+" if error >= 0 else "−") + L.num(abs(100.0 * error), 0) + " %"


def _place_ach_labels(fig, ax, items, top: float, fontsize: float = 6.5) -> None:
    """Etiqueta cada barra de ACH estimado con su valor y su error.

    La etiqueta va arriba de la barra, o arriba de la línea del ACH real si
    quedaría tachada por ella: cuando la barra queda apenas por debajo, o cuando
    es corta y la etiqueta, más ancha que la barra, cruzaría el escalón. Se
    colocan primero las barras largas; si la de una barra corta pisa a otra,
    queda sólo con el valor y, si hace falta, sube. Hay que llamarla con los
    límites de los ejes ya puestos.
    """
    renderer = fig.canvas.get_renderer()
    placed = []
    for it in sorted(items, key=lambda d: d["x0"] - d["x1"]):
        x0, x1, y, real = it["x0"], it["x1"], it["y"], it["real"]
        if x1 - x0 >= 14.0:
            xt, ha = 0.5 * (x0 + x1), "center"
        elif x0 > 44.0:
            xt, ha = x1, "right"
        else:
            xt, ha = x0 + 0.4, "left"
        if y > 0.8 * top:
            yt, va, step = y - 0.45, "top", -0.9
        else:
            below_real = real is not None and y < real
            base = real if (below_real and (real - y <= 1.9 or x1 - x0 < 14.0)) else y
            yt, va, step = base + 0.4, "bottom", 0.9
        t = ax.text(xt, yt, it["full"], ha=ha, va=va, fontsize=fontsize, color=INK, zorder=7, bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.5, "alpha": 0.8})
        box = t.get_window_extent(renderer)
        if any(box.overlaps(other) for other in placed):
            t.set_text(it["short"])
            box = t.get_window_extent(renderer)
            for _ in range(6):
                if not any(box.overlaps(other) for other in placed):
                    break
                t.set_y(t.get_position()[1] + step)
                box = t.get_window_extent(renderer)
        placed.append(box)


def _shade_open(ax, schedule: Schedule, mixing: bool = False) -> None:
    for k, (a, b, state) in enumerate(zip(schedule.edges_min[:-1], schedule.edges_min[1:], schedule.states)):
        if state == OPEN:
            ax.axvspan(a, b, color=MUTED, alpha=0.16, lw=0, zorder=0)
        elif mixing and k > 0 and schedule.mixing_min > 0:
            ax.axvspan(a, a + schedule.mixing_min, facecolor="none", edgecolor=MUTED, hatch="////", alpha=0.45, lw=0, zorder=0)


def _minute_axis(ax, schedule: Schedule) -> None:
    ax.set_xlim(schedule.edges_min[0] - 0.8, schedule.edges_min[-1] + 0.8)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(NullLocator())


# ------------------------------------------------------------------ datos


def fig_data(case: CfdCase, schedule: Schedule, L: Lang, room_xy=(8.0, 6.0)):
    """Series de todos los sensores por altura y dónde está cada sensor."""
    heights = sorted(case.sensors["z"].unique())
    fig = plt.figure(figsize=(WIDTH, 5.9))
    gs = fig.add_gridspec(2, len(heights), height_ratios=(1.0, 0.86), left=0.085, right=0.985, top=0.9, bottom=0.075, hspace=0.36, wspace=0.12)
    t_min = case.t_h * 60.0
    top = None
    for j, z in enumerate(heights):
        ax = fig.add_subplot(gs[0, j], sharey=top)
        top = top or ax
        idx = np.flatnonzero(np.isclose(case.sensors["z"], z))
        _shade_open(ax, schedule)
        for i in idx:
            ax.plot(t_min, case.data[:, i], color=WALL_COLOR[case.sensors["pared"].iloc[i]], lw=0.6, alpha=0.75, zorder=2)
        ax.plot(t_min, case.data[:, idx].mean(axis=1), color=INK, lw=1.7, zorder=4)
        ax.set_title(_t(L, "n_sensors", z=L.num(z, 1), n=len(idx)))
        _minute_axis(ax, schedule)
        ax.set_xlabel(_t(L, "minute"))
        ax.yaxis.set_major_formatter(L.tick)
        if j == 0:
            ax.set_ylabel(_t(L, "co2"))
        else:
            ax.tick_params(labelleft=False)

        m = fig.add_subplot(gs[1, j])
        m.add_patch(Rectangle((0, 0), room_xy[0], room_xy[1], facecolor="none", edgecolor=AXIS, lw=0.9, zorder=1))
        for i in idx:
            row = case.sensors.iloc[i]
            m.scatter([row["x"]], [row["y"]], s=58, color=WALL_COLOR[row["pared"]], edgecolor=WHITE, lw=0.6, zorder=3)
            m.text(row["x"], row["y"], str(int(row["sensor"])), ha="center", va="center", fontsize=5.4, color=WHITE, fontweight="semibold", zorder=4)
        m.set_aspect("equal")
        m.set_xlim(-0.5, room_xy[0] + 0.5)
        m.set_ylim(-0.5, room_xy[1] + 0.5)
        m.grid(False)
        m.set_title(_t(L, "sensors_at", z=L.num(z, 1)))
        m.set_xlabel(_t(L, "x_m"))
        m.xaxis.set_major_locator(MultipleLocator(2))
        m.yaxis.set_major_locator(MultipleLocator(2))
        if j == 0:
            m.set_ylabel(_t(L, "y_m"))
        else:
            m.tick_params(labelleft=False)
    walls = [w for w in WALL_ORDER if (case.sensors["pared"] == w).any()]
    handles = [Line2D([], [], color=WALL_COLOR[w], lw=1.2, label=TXT[L.code]["walls"][w]) for w in walls]
    handles += [Line2D([], [], color=INK, lw=1.7, label=_t(L, "mean")), Patch(color=MUTED, alpha=0.3, lw=0, label=_t(L, "open"))]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 1.0), ncol=len(handles), handlelength=1.4, columnspacing=1.2)
    return fig


# ------------------------------------------------------------------ identificabilidad


def fig_identifiability(case: CfdCase, schedule: Schedule, scan, scan_all_minutes, free, setup, L: Lang, examples=(0.4, 1.0, 1.6), reference=None, truth=None, calibrated=None):
    """La serie fija mal la generación: generaciones muy distintas ajustan parecido y sólo queda determinada la diferencia de ACH.

    La generación se muestra como G/V en ppm/s, igual que en el texto; las
    tablas la guardan en ppm/h (`fuente_ppm_h`). `examples` va en ppm/s.

    `scan` es el barrido de la fuente salteando los minutos de mezcla y
    `scan_all_minutes` el que los incluye; `free` son los ajustes con la
    generación libre de los tramos cerrados.
    """
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.25))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.72, bottom=0.15, wspace=0.34)
    series = mean_series(case, np.arange(case.data.shape[1]), "todos")
    t_min = series.t_h * 60.0
    seg = schedule_segmentation(series, schedule)

    ax = axes[0]
    _shade_open(ax, schedule, mixing=True)
    ax.scatter(t_min, series.c_obs, s=7, color=MUTED, lw=0, zorder=2, label=_t(L, "data"))
    for g_s, color in zip(examples, (BLUE_LIGHT, BLUE, BLUE_DARK)):
        fits = fit_segments(series, "M0", seg, setup, source=float(g_s) * 3600.0)
        closed, opened = (float(np.interp(g_s * 3600.0, scan["fuente_ppm_h"], scan[f"ach_{k}"])) for k in (CLOSED, OPEN))
        label = _t(L, "curve", s=L.g(g_s), c=L.num(closed, 1), o=L.num(opened, 1))
        for k, f in enumerate(fits):
            if f.ok:
                tt = series.t_h[f.i0 : f.i1]
                ax.plot(tt * 60.0, f.predict(tt), color=color, lw=1.1, zorder=4, label=label if k == 0 else None)
    _minute_axis(ax, schedule)
    ax.set_xlabel(_t(L, "minute"))
    ax.set_ylabel(_t(L, "co2"))
    ax.yaxis.set_major_formatter(L.tick)
    ax.set_title(_t(L, "same_fit"))
    handles, labels = ax.get_legend_handles_labels()
    if schedule.mixing_min > 0:
        handles.append(Patch(facecolor="none", edgecolor=MUTED, hatch="////", alpha=0.6, lw=0))
        labels.append(_t(L, "mixing", v=L.g(schedule.mixing_min)))
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.07, 1.0), ncol=2, handlelength=1.4, columnspacing=1.4)

    ax = axes[1]
    src = scan["fuente_ppm_h"].to_numpy() / 3600.0
    ax.plot(src, scan[f"ach_{OPEN}"], color=BLUE, lw=1.5)
    ax.plot(src, scan[f"ach_{CLOSED}"], color=RED, lw=1.5)
    span = src[-1] - src[0]
    x_lab = src[0] + 0.25 * span
    ax.text(x_lab, float(np.interp(x_lab, src, scan[f"ach_{OPEN}"])) + 0.55, _t(L, "ach_open"), color=INK2, fontsize=7.5, ha="right", va="bottom")
    x_lab = src[0] + 0.56 * span
    ax.text(x_lab, float(np.interp(x_lab, src, scan[f"ach_{CLOSED}"].fillna(0.0))) - 0.45, _t(L, "ach_closed"), color=INK2, fontsize=7.5, ha="left", va="top")
    j = len(src) * 3 // 4
    ax.annotate("", xy=(src[j], scan[f"ach_{OPEN}"].iloc[j]), xytext=(src[j], scan[f"ach_{CLOSED}"].iloc[j]), arrowprops={"arrowstyle": "<->", "color": INK2, "lw": 0.8})
    ax.text(src[j] - 0.02 * span, 0.5 * (scan[f"ach_{OPEN}"].iloc[j] + scan[f"ach_{CLOSED}"].iloc[j]) - 0.45, _t(L, "delta", v=L.num(float(np.nanmedian(scan["delta_ach"])), 1)), fontsize=7.5, color=INK, va="center", ha="right")
    ok = free.dropna(subset=["ach"])
    if len(ok):
        ax.scatter(ok["fuente_ppm_h"] / 3600.0, ok["ach"], s=26, facecolor=WHITE, edgecolor=INK, lw=1.0, zorder=6, label=_t(L, "free_fit"))
    if truth:
        for key, color in ((OPEN, BLUE), (CLOSED, RED)):
            if truth.get(key) is not None:
                ax.axhline(truth[key], color=color, lw=0.8, zorder=1)
        ax.plot([], [], color=INK2, lw=0.8, label=_t(L, "truth_short"))
    if calibrated is not None and np.isfinite(calibrated):
        ax.axvline(calibrated / 3600.0, color=INK2, lw=0.8, zorder=1)
        ax.text(calibrated / 3600.0, 0.03, " " + _t(L, "calibrated"), transform=ax.get_xaxis_transform(), fontsize=6.6, color=INK2, ha="left", va="bottom")
    ax.set_ylim(0, None)
    ax.set_xlim(src[0], src[-1])
    ax.set_xlabel(_t(L, "source"))
    ax.set_ylabel(_t(L, "ach"))
    ax.xaxis.set_major_formatter(L.tick)
    ax.yaxis.set_major_formatter(L.tick)
    ax.legend(loc="upper left", handlelength=1.0, fontsize=6.8, borderaxespad=0.2, frameon=True, facecolor=WHITE, edgecolor="none", framealpha=0.94)
    top = ax.secondary_xaxis("top", functions=(lambda v: v * 6.0 / PERSON_LITRES_MIN, lambda v: v * PERSON_LITRES_MIN / 6.0))  # G/V en ppm/s → personas cada 100 m³
    top.set_xlabel(_t(L, "people", g=L.g(PERSON_LITRES_MIN)), fontsize=7, color=INK2)
    top.tick_params(labelsize=7, colors=MUTED, labelcolor=INK2)
    top.xaxis.set_major_formatter(L.tick)

    ax = axes[2]
    ax.plot(scan_all_minutes["fuente_ppm_h"] / 3600.0, scan_all_minutes["rmse"].where(scan_all_minutes[f"ach_{CLOSED}"].notna()), color=MUTED, lw=1.2, zorder=3, label=_t(L, "with_mixing"))
    ax.plot(src, scan["rmse"].where(scan[f"ach_{CLOSED}"].notna()), color=INK, lw=1.6, zorder=4, label=_t(L, "without_mixing"))
    if reference and np.isfinite(reference.get("fuente", np.nan)):
        y = float(np.interp(reference["fuente"] / 3600.0, src, scan["rmse"]))
        ax.plot([reference["lo"] / 3600.0, reference["hi"] / 3600.0], [y * 0.86] * 2, color=INK, lw=1.0, zorder=5)
        ax.scatter([reference["fuente"] / 3600.0], [y * 0.86], s=18, color=INK, zorder=6)
    ax.set_ylim(0, None)
    ax.set_xlim(src[0], src[-1])
    ax.set_xlabel(_t(L, "source"))
    ax.set_ylabel(_t(L, "rmse"))
    ax.xaxis.set_major_formatter(L.tick)
    ax.yaxis.set_major_formatter(L.tick)
    ax.set_title(_t(L, "fit_error"))
    ax.legend(loc="lower right", handlelength=1.4, fontsize=6.8)
    return fig


# ------------------------------------------------------------------ segmentación


def fig_segmentation(res: CfdResults, columns, L: Lang, methods=("M1", "M2", "M3a", "M3b", "M4", "M5", "M6"), gen: str = "conocida"):
    """Cómo segmentó y ajustó cada método. `columns` es una lista de (serie, escalas)."""
    fig, axes, _ = _grid(len(methods), len(columns), row_h=1.06, top_in=0.78, bottom_in=0.42, hspace=0.42, wspace=0.06, left=0.075)
    sch = res.schedule
    n_onsets = len(sch.onsets())
    for c, (name, sk) in enumerate(columns):
        ser = res.series[name]
        t_min = ser.t_h * 60.0
        span = ser.c_obs.max() - ser.c_obs.min()
        for r, key in enumerate(methods):
            ax = axes[r, c]
            seg = res.segmentations[(name, sk, key)]
            fits = res.fits[(name, sk, key, gen)]
            for s in seg.transients:
                ax.axvspan(t_min[s.i0], t_min[s.i1 - 1], color=RED if s.phase == RISE else BLUE, alpha=0.16, lw=0, zorder=0)
            for edge in sch.edges_min[1:-1]:
                ax.axvline(edge, color=AXIS, lw=0.7, zorder=1)
            ax.scatter(t_min, ser.c_obs, s=4.5, color=MUTED, lw=0, zorder=2)
            ax.vlines([t_min[s.i0] for s in seg.transients], 0, 0.07, transform=ax.get_xaxis_transform(), color=INK2, lw=0.8, zorder=4)
            for f in fits:
                if f.ok:
                    tt = ser.t_h[f.i0 : f.i1]
                    ax.plot(tt * 60.0, f.predict(tt), color=INK if f.accepted else MUTED, lw=1.25 if f.accepted else 0.7, zorder=5 if f.accepted else 3)
            row = res.onsets[(res.onsets["serie"] == name) & (res.onsets["escalas"] == sk) & (res.onsets["metodo"] == key)].iloc[0]
            st = res.states[(res.states["serie"] == name) & (res.states["escalas"] == sk) & (res.states["metodo"] == key) & (res.states["gen"] == gen)]
            delta = float(st["delta_ach"].iloc[0]) if len(st) else np.nan
            note = _t(L, "onsets", a=int(row["inicios_acertados"]), b=n_onsets) + " · " + (_t(L, "dach", v=L.num(delta, 1)) if np.isfinite(delta) else _t(L, "no_dach"))
            ax.text(1.0, 1.03, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8, color=INK2)
            ax.set_title(TXT[L.code]["methods"][key], fontsize=8, pad=3)
            ax.set_ylim(ser.c_obs.min() - 0.1 * span, ser.c_obs.max() + 0.1 * span)
            ax.yaxis.set_major_formatter(L.tick)
            _minute_axis(ax, sch)
        head = _name(L, name) + " · " + TXT[L.code]["setups"][sk]
        axes[0, c].text(0.0, 1.4, head, transform=axes[0, c].transAxes, ha="left", va="bottom", fontsize=8.6, fontweight="semibold", color=INK)
        axes[-1, c].set_xlabel(_t(L, "minute"))
    fig.text(0.012, 0.5, _t(L, "co2"), rotation=90, ha="left", va="center", fontsize=8, color=INK2)
    handles = [
        Patch(color=RED, alpha=0.3, lw=0, label=_t(L, "rise")),
        Patch(color=BLUE, alpha=0.3, lw=0, label=_t(L, "decay")),
        Line2D([], [], color=INK, lw=1.25, label=_t(L, "fit_ok")),
        Line2D([], [], color=MUTED, lw=0.7, label=_t(L, "fit_no")),
        Line2D([], [], color=INK2, lw=0.8, marker="|", ms=5, ls="", label=_t(L, "cut")),
        Line2D([], [], color=AXIS, lw=0.7, label=_t(L, "change")),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 1.0), ncol=6, handlelength=1.2, columnspacing=1.1, handletextpad=0.4)
    return fig


def _sensor_order(case: CfdCase) -> list:
    """Sensores ordenados por altura y, dentro de cada una, de la pared oeste a la este."""
    rank = {w: k for k, w in enumerate(WALL_ORDER)}
    s = case.sensors.assign(_w=case.sensors["pared"].map(rank))
    return list(s.sort_values(["z", "_w", "x", "y"]).index)


def fig_raster(res: CfdResults, L: Lang, methods=("M2", "M4", "M5", "M6"), setups=("estudio", "corta")):
    """Tramos que detecta cada método en cada sensor: una fila por sensor."""
    case, sch = res.case, res.schedule
    order = _sensor_order(case)
    heights = sorted(case.sensors["z"].unique())
    fig, axes, _ = _grid(len(methods), len(setups), row_h=1.9, top_in=0.62, bottom_in=0.42, hspace=0.2, wspace=0.05, left=0.115)
    t_min = case.t_h * 60.0
    ypos, y, groups = {}, 0.0, []
    for z in heights:
        start = y
        for i in order:
            if np.isclose(case.sensors["z"].iloc[i], z):
                ypos[i] = y
                y += 1.0
        groups.append((z, start, y - 1.0))
        y += 1.6
    for r, key in enumerate(methods):
        for c, sk in enumerate(setups):
            ax = axes[r, c]
            for i, yy in ypos.items():
                seg = res.segmentations[(f"sensor {i + 1}", sk, key)]
                for s in seg.transients:
                    ax.add_patch(Rectangle((t_min[s.i0], yy - 0.4), t_min[s.i1 - 1] - t_min[s.i0], 0.8, facecolor=RED if s.phase == RISE else BLUE, alpha=0.75, lw=0, zorder=3))
            for edge in sch.edges_min[1:-1]:
                ax.axvline(edge, color=INK2, lw=0.7, zorder=4)
            ax.set_ylim(y - 1.6 + 0.2, -0.8)
            ax.grid(False)
            ax.set_yticks([0.5 * (a + b) for _, a, b in groups])
            ax.set_yticklabels([_t(L, "height", z=L.num(z, 1)) for z, _, _ in groups])
            ax.tick_params(axis="y", length=0, pad=9)
            _minute_axis(ax, sch)
            ax.set_title(TXT[L.code]["methods"][key], fontsize=8, pad=3)
            if r == 0:
                ax.text(0.0, 1.14, TXT[L.code]["setups"][sk], transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, fontweight="semibold", color=INK)
    for ax in axes[-1]:
        ax.set_xlabel(_t(L, "minute"))
    walls = [w for w in WALL_ORDER if (case.sensors["pared"] == w).any()]
    for (z, a, b) in groups:
        idx = [i for i in order if np.isclose(case.sensors["z"].iloc[i], z)]
        for w in walls:
            ys = [ypos[i] for i in idx if case.sensors["pared"].iloc[i] == w]
            if ys:
                for ax in axes[:, 0]:
                    ax.plot([-0.012, -0.012], [min(ys) - 0.35, max(ys) + 0.35], transform=ax.get_yaxis_transform(), color=WALL_COLOR[w], lw=2.2, clip_on=False, solid_capstyle="butt")
    handles = [Patch(color=RED, alpha=0.75, lw=0, label=_t(L, "rise")), Patch(color=BLUE, alpha=0.75, lw=0, label=_t(L, "decay")), Line2D([], [], color=INK2, lw=0.7, label=_t(L, "change"))]
    handles += [Line2D([], [], color=WALL_COLOR[w], lw=2.2, label=TXT[L.code]["walls"][w]) for w in walls]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.1, 1.0), ncol=len(handles), handlelength=1.2, columnspacing=1.1, handletextpad=0.4)
    return fig


def fig_onsets(res: CfdResults, L: Lang, methods=("M1", "M2", "M3a", "M3b", "M4", "M5", "M6"), setups=("estudio", "corta")):
    """Porcentaje de sensores que detecta cada cambio del esquema, por método, altura y tolerancia."""
    case, sch = res.case, res.schedule
    heights = sorted(case.sensors["z"].unique())
    z_of = {f"sensor {int(n)}": z for n, z in zip(case.sensors["sensor"], case.sensors["z"])}
    data = res.onsets[res.onsets["tipo"] == "sensor"].assign(z=lambda d: d["serie"].map(z_of))
    changes = sch.onsets()
    tols = [("acierto", res.setups[setups[0]].onset_tol_min), ("acierto_amplio", res.setups[setups[0]].onset_tol_loose_min)]
    fig, axes, _ = _grid(len(tols), len(setups), row_h=2.05, top_in=0.38, bottom_in=0.68, left=0.2, right=0.9, hspace=0.1, wspace=0.07)
    image = None
    for r, (prefix, tol) in enumerate(tols):
        for c, sk in enumerate(setups):
            ax = axes[r, c]
            mat = np.full((len(methods), len(changes) * len(heights)), np.nan)
            for i, key in enumerate(methods):
                sub = data[(data["metodo"] == key) & (data["escalas"] == sk)]
                for j, (t_true, _) in enumerate(changes):
                    for h, z in enumerate(heights):
                        v = sub[np.isclose(sub["z"], z)][f"{prefix}_{t_true:g}"]
                        mat[i, j * len(heights) + h] = 100.0 * v.mean() if len(v) else np.nan
            image = ax.imshow(mat, cmap=SEQ_BLUE, vmin=0, vmax=100, aspect="auto")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if np.isfinite(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7, color=WHITE if mat[i, j] > 58 else INK)
            ax.set_xticks(range(mat.shape[1]))
            ax.set_xticklabels([L.num(z, 1) for _ in changes for z in heights], fontsize=7)
            if r == len(tols) - 1:
                for j, (t_true, phase) in enumerate(changes):
                    x = j * len(heights) + 0.5 * (len(heights) - 1)
                    ax.plot([x - 0.45 * len(heights), x + 0.45 * len(heights)], [-0.15, -0.15], transform=ax.get_xaxis_transform(), color=AXIS, lw=0.8, clip_on=False)
                    ax.text(x, -0.18, _t(L, "close" if phase == RISE else "opening", t=L.g(t_true)), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color=INK2)
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels([TXT[L.code]["methods"][k] for k in methods], fontsize=7.4)
            ax.grid(False)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks(np.arange(-0.5, mat.shape[1]), minor=True)
            ax.set_yticks(np.arange(-0.5, len(methods)), minor=True)
            ax.grid(which="minor", color=WHITE, linewidth=1.6)
            if r == 0:
                ax.set_title(TXT[L.code]["setups"][sk])
            if c == len(setups) - 1:
                ax.text(1.03, 0.5, _t(L, "tol", v=L.g(tol)), transform=ax.transAxes, rotation=270, ha="left", va="center", fontsize=7.6, color=INK2)
    cax = fig.add_axes([0.94, 0.3, 0.012, 0.42])
    cb = fig.colorbar(image, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label(_t(L, "detected"), fontsize=7.5, color=INK2)
    cb.ax.tick_params(labelsize=7, length=0)
    fig.text(0.2, 0.012, "z (m)", ha="left", va="bottom", fontsize=7.3, color=INK2)
    return fig


# ------------------------------------------------------------------ ACH por sensor


def fig_delta_map(case: CfdCase, sensors, L: Lang, room_xy=(8.0, 6.0), all_value: float | None = None, truth: float | None = None):
    """Diferencia de ACH entre abierto y cerrado en cada sensor, en planta y por pared."""
    heights = sorted(case.sensors["z"].unique())
    fig = plt.figure(figsize=(WIDTH, 3.45))
    gs = fig.add_gridspec(1, len(heights) + 2, width_ratios=[1.0] * len(heights) + [0.56, 0.72], left=0.065, right=0.985, top=0.86, bottom=0.27, wspace=0.14)
    values = sensors["delta_ach"].to_numpy(dtype=float)
    vmax = float(np.ceil(np.nanmax(values)))
    image = None
    maps = []
    for j, z in enumerate(heights):
        ax = fig.add_subplot(gs[0, j])
        maps.append(ax)
        ax.add_patch(Rectangle((0, 0), room_xy[0], room_xy[1], facecolor="none", edgecolor=AXIS, lw=0.9, zorder=1))
        sub = sensors[np.isclose(sensors["z"], z)]
        image = ax.scatter(sub["x"], sub["y"], c=sub["delta_ach"], cmap=SEQ_BLUE, vmin=0, vmax=vmax, s=150, edgecolor=AXIS, lw=0.5, zorder=3)
        for row in sub.itertuples():
            if np.isfinite(row.delta_ach):
                ax.text(row.x, row.y, L.num(row.delta_ach, 1), ha="center", va="center", fontsize=5.6, color=WHITE if row.delta_ach > 0.58 * vmax else INK, zorder=4)
        ax.set_aspect("equal")
        ax.set_xlim(-0.6, room_xy[0] + 0.6)
        ax.set_ylim(-0.6, room_xy[1] + 0.6)
        ax.grid(False)
        ax.set_title(_t(L, "delta_at", z=L.num(z, 1)))
        ax.set_xlabel(_t(L, "x_m"))
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(2))
        if j == 0:
            ax.set_ylabel(_t(L, "y_m"))
        else:
            ax.tick_params(labelleft=False)
    cax = fig.add_axes([maps[0].get_position().x0, 0.115, maps[-1].get_position().x1 - maps[0].get_position().x0, 0.022])
    cb = fig.colorbar(image, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_label(_t(L, "delta_ach"), fontsize=7.5, color=INK2, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.ax.xaxis.set_major_formatter(L.tick)

    ax = fig.add_subplot(gs[0, -1])
    walls = [w for w in WALL_ORDER if (sensors["pared"] == w).any()]
    rows = [(w, z) for w in walls for z in heights if len(sensors[(sensors["pared"] == w) & np.isclose(sensors["z"], z)])]
    for y, (w, z) in enumerate(rows):
        sub = sensors[(sensors["pared"] == w) & np.isclose(sensors["z"], z)]["delta_ach"].dropna()
        ax.scatter(sub, np.full(len(sub), y), s=16, color=WALL_COLOR[w], edgecolor=WHITE, lw=0.4, alpha=0.9, zorder=3)
        if len(sub):
            ax.plot([sub.mean()] * 2, [y - 0.32, y + 0.32], color=INK, lw=1.4, zorder=4)
    if all_value is not None and np.isfinite(all_value):
        ax.axvline(all_value, color=INK2, lw=0.8, zorder=1)
        side = "center" if truth is None else ("right" if truth > all_value else "left")
        ax.text(all_value, 1.01, _t(L, "all_short"), transform=ax.get_xaxis_transform(), ha=side, va="bottom", fontsize=6.6, color=INK2)
    if truth is not None:
        ax.axvline(truth, color=INK, lw=1.1, zorder=2)
        side = "left" if (all_value is None or truth > all_value) else "right"
        ax.text(truth, 1.01, " " + _t(L, "simulation") + " ", transform=ax.get_xaxis_transform(), ha=side, va="bottom", fontsize=6.6, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{TXT[L.code]['walls'][w]}, {L.num(z, 1)} m" for w, z in rows], fontsize=7)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlim(0, None)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(_t(L, "delta_ach"))
    ax.xaxis.set_major_formatter(L.tick)
    ax.set_title(_t(L, "by_wall"), pad=12)
    return fig


def fig_subsets(sub, L: Lang, all_value: float, truth: float | None = None):
    """Qué se gana al promediar más sensores antes de ajustar."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.84, bottom=0.17, wspace=0.28)
    g = sub.groupby("k")["delta_ach"]
    ks = np.array(sorted(sub["k"].unique()), dtype=float)
    q = {p: g.quantile(p).reindex(ks).to_numpy() for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
    ax = axes[0]
    ax.fill_between(ks, q[0.05], q[0.95], color=BLUE, alpha=0.13, lw=0, label=_t(L, "p90"))
    ax.fill_between(ks, q[0.25], q[0.75], color=BLUE, alpha=0.3, lw=0, label=_t(L, "iqr"))
    ax.plot(ks, q[0.5], color=BLUE_DARK, lw=1.5, marker="o", ms=4, mec=WHITE, mew=0.5, label=_t(L, "median"))
    ax.axhline(all_value, color=INK2, lw=0.8, zorder=1, label=_t(L, "all"))
    if truth is not None:
        ax.axhline(truth, color=INK, lw=1.1, zorder=1, label=_t(L, "truth"))
    ax.set_ylabel(_t(L, "delta_ach"))
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", ncol=2, handlelength=1.4, columnspacing=1.2)

    target = truth if truth is not None else all_value
    dev = sub.assign(dev=100.0 * (sub["delta_ach"] - target).abs() / target).groupby("k")["dev"]
    ax = axes[1]
    ax.plot(ks, dev.median().reindex(ks), color=BLUE_DARK, lw=1.5, marker="o", ms=4, mec=WHITE, mew=0.5, label=_t(L, "median"))
    ax.plot(ks, dev.quantile(0.9).reindex(ks), color=BLUE_LIGHT, lw=1.5, marker="o", ms=4, mec=WHITE, mew=0.5, label=_t(L, "p90dev"))
    ax.set_ylabel(_t(L, "dev_truth" if truth is not None else "dev"))
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", handlelength=1.4)
    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(list(ks)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(L.tick)
        ax.yaxis.set_major_formatter(L.tick)
        ax.set_xlabel(_t(L, "k_sensors"))
    return fig


def fig_methods_delta(res: CfdResults, L: Lang, names, methods=("M2", "M3a", "M3b", "M4", "M5", "M6"), setup: str = "corta", truth: float | None = None):
    """ΔACH con los cortes de cada método frente a los del esquema, en los promedios."""
    fig, axes, _ = _grid(1, len(names), row_h=2.1, top_in=0.5, bottom_in=0.45, left=0.2, wspace=0.08)
    st = res.states[(res.states["gen"] == "conocida") & (res.states["escalas"] == setup)]
    shown = st[st["serie"].isin(names) & st["metodo"].isin((*methods, "M0"))]["delta_ach"]
    x_max = 1.08 * float(np.nanmax([shown.max(), truth if truth is not None else np.nan]))
    for ax, name in zip(axes[0], names):
        sub = st[st["serie"] == name].set_index("metodo")["delta_ach"]
        ref = float(sub.get("M0", np.nan))
        if np.isfinite(ref):
            ax.axvline(ref, color=INK2, lw=0.9, zorder=1)
        if truth is not None:
            ax.axvline(truth, color=INK, lw=1.1, zorder=1)
        for y, key in enumerate(methods):
            v = float(sub.get(key, np.nan))
            if np.isfinite(v):
                ax.scatter([v], [y], s=30, color=BLUE_DARK, edgecolor=WHITE, lw=0.6, zorder=3)
        ax.set_ylim(len(methods) - 0.5, -0.6)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([TXT[L.code]["methods"][k] for k in methods], fontsize=7.4)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, x_max)
        ax.xaxis.set_major_formatter(L.tick)
        ax.set_title(_name(L, name))
    axes[0, len(names) // 2].set_xlabel(_t(L, "delta_ach"))
    handles = [Line2D([], [], marker="o", ls="", ms=5, mfc=BLUE_DARK, mec=WHITE, label=_t(L, "own_cuts")), Line2D([], [], color=INK2, lw=0.9, label=_t(L, "schedule_cuts"))]
    if truth is not None:
        handles.append(Line2D([], [], color=INK, lw=1.1, label=_t(L, "truth")))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.19, 1.0), ncol=len(handles), handlelength=1.2, columnspacing=1.4)
    return fig


def fig_truth(table, L: Lang, truth: dict, names, generations=("libre", "calibrada")):
    """ACH de cada estado y su diferencia frente a los de la simulación, según lo que se suponga de la generación."""
    panels = ((f"ach_{CLOSED}", truth.get(CLOSED), "panel_closed"), (f"ach_{OPEN}", truth.get(OPEN), "panel_open"), ("delta_ach", truth.get(OPEN) - truth.get(CLOSED), "panel_delta"))
    fig, axes, _ = _grid(1, len(panels), row_h=2.5, top_in=0.74, bottom_in=0.45, left=0.17, wspace=0.1, sharex=False)
    style = {
        "libre": dict(facecolor=WHITE, edgecolor=INK2, lw=1.0),
        "serie": dict(facecolor=BLUE_LIGHT, edgecolor=WHITE, lw=0.6),
        "calibrada": dict(facecolor=BLUE_DARK, edgecolor=WHITE, lw=0.6),
        "ocupacion": dict(facecolor=BLUE_DARK, edgecolor=WHITE, lw=0.6),
    }
    for ax, (col, ref, title) in zip(axes[0], panels):
        ax.axvline(ref, color=INK, lw=1.1, zorder=1)
        ax.axvspan(0.8 * ref, 1.2 * ref, color=MUTED, alpha=0.14, lw=0, zorder=0)
        top = 0.0
        for y, name in enumerate(names):
            sub = table[table["serie"] == name].set_index("clave")
            for gen in generations:
                if gen in sub.index and np.isfinite(sub.loc[gen, col]):
                    ax.scatter([sub.loc[gen, col]], [y], s=30, zorder=3 + generations.index(gen), **style[gen])
                    top = max(top, float(sub.loc[gen, col]))
        ax.set_ylim(len(names) - 0.5, -0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([_name(L, n) for n in names], fontsize=7.4)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 1.08 * max(top, 1.25 * ref))
        ax.xaxis.set_major_formatter(L.tick)
        ax.set_title(_t(L, title))
        ax.set_xlabel(_t(L, "ach"))
    for ax in axes[0, 1:]:
        ax.tick_params(labelleft=False)
    marks = {
        "libre": Line2D([], [], marker="o", ls="", ms=5, mfc=WHITE, mec=INK2, mew=1.0, label=_t(L, "gen_free")),
        "serie": Line2D([], [], marker="o", ls="", ms=5, mfc=BLUE_LIGHT, mec=WHITE, label=_t(L, "gen_series")),
        "calibrada": Line2D([], [], marker="o", ls="", ms=5, mfc=BLUE_DARK, mec=WHITE, label=_t(L, "gen_calibrated")),
        "ocupacion": Line2D([], [], marker="o", ls="", ms=5, mfc=BLUE_DARK, mec=WHITE, label=_t(L, "gen_occupancy")),
    }
    handles = [
        *(marks[g] for g in generations),
        Line2D([], [], color=INK, lw=1.1, label=_t(L, "truth")),
        Patch(color=MUTED, alpha=0.3, lw=0, label="±20 %"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.16, 1.0), ncol=len(handles), handlelength=1.2, columnspacing=1.6)
    return fig


# ------------------------------------------------------------------ detalle por serie


def fig_detail(res: CfdResults, name: str, sk: str, gen: str, L: Lang, truth: dict | None = None, methods=DETAIL_METHODS, ach_max: float = 10.0):
    """La serie, cómo la segmenta cada método y qué ACH estima en cada tramo frente al de la simulación.

    Una fila por método. Izquierda: CO2, tramos y ajustes. Derecha: ACH de la
    simulación (escalón) y ACH estimado en cada ajuste aceptado, con su
    intervalo de 95 %, su valor y su error. En azul oscuro, el ajuste que cuenta
    para cada tramo del esquema (ver `cfd.match_fits`); en azul claro, los
    aceptados de más. Los ACH mayores que `ach_max` se dibujan en el borde
    superior.
    """
    case, sch = res.case, res.schedule
    ser = res.series[name]
    t_min = ser.t_h * 60.0
    n = len(methods)
    row_h, top_in, bottom_in = 0.9, 1.12, 0.42
    height = n * row_h + top_in + bottom_in
    fig, axes = plt.subplots(n, 2, figsize=(WIDTH, height), sharex=True, sharey="col", squeeze=False)
    fig.subplots_adjust(left=0.078, right=0.988, top=1.0 - top_in / height, bottom=bottom_in / height, hspace=0.5, wspace=0.15)
    span = ser.c_obs.max() - ser.c_obs.min()
    n_onsets = len(sch.onsets())
    real_x, real_y = [], []
    if truth:
        for a, b, state in zip(sch.edges_min[:-1], sch.edges_min[1:], sch.states):
            real_x += [a, b]
            real_y += [truth[state]] * 2

    for r, key in enumerate(methods):
        seg = res.segmentations[(name, sk, key)]
        fits = res.fits[(name, sk, key, gen)]

        ax = axes[r, 0]
        for s_ in seg.transients:
            ax.axvspan(t_min[s_.i0], t_min[s_.i1 - 1], color=RED if s_.phase == RISE else BLUE, alpha=0.16, lw=0, zorder=0)
        for edge in sch.edges_min[1:-1]:
            ax.axvline(edge, color=AXIS, lw=0.7, zorder=1)
        ax.scatter(t_min, ser.c_obs, s=4.0, color=MUTED, lw=0, zorder=2)
        ax.vlines([t_min[s_.i0] for s_ in seg.transients], 0, 0.08, transform=ax.get_xaxis_transform(), color=INK2, lw=0.8, zorder=4)
        for f in fits:
            if f.ok:
                tt = ser.t_h[f.i0 : f.i1]
                ax.plot(tt * 60.0, f.predict(tt), color=INK if f.accepted else MUTED, lw=1.25 if f.accepted else 0.7, zorder=5 if f.accepted else 3)
        ax.set_title(TXT[L.code]["methods"][key], fontsize=8, pad=3)
        if key == "M0":
            note = _t(L, "exact_cuts")
        else:
            row = res.onsets[(res.onsets["serie"] == name) & (res.onsets["escalas"] == sk) & (res.onsets["metodo"] == key)].iloc[0]
            note = _t(L, "onsets", a=int(row["inicios_acertados"]), b=n_onsets)
        ax.text(1.0, 1.04, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8, color=INK2)
        ax.set_ylim(ser.c_obs.min() - 0.12 * span, ser.c_obs.max() + 0.12 * span)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(L.tick)
        _minute_axis(ax, sch)

        ax = axes[r, 1]
        for edge in sch.edges_min[1:-1]:
            ax.axvline(edge, color=AXIS, lw=0.7, zorder=1)
        if truth:
            ax.plot(real_x, real_y, color=INK, lw=1.0, zorder=3, solid_joinstyle="miter")
        items = []
        owner = {j: k for k, j in match_fits(ser, fits, sch).items()}
        for j, f in enumerate(fits):
            if not (f.ok and f.accepted and np.isfinite(f.lam)):
                continue
            x0, x1 = float(t_min[f.i0]), float(t_min[f.i1 - 1])
            clipped = f.lam > ach_max
            y = min(f.lam, 0.975 * ach_max)
            counted = j in owner
            if np.isfinite(f.lam_se) and not clipped:
                lo, hi = max(f.lam - 1.96 * f.lam_se, 0.0), min(f.lam + 1.96 * f.lam_se, ach_max)
                ax.fill_between([x0, x1], lo, hi, color=BLUE, alpha=0.2 if counted else 0.1, lw=0, zorder=2)
            ax.plot([x0, x1], [y, y], color=BLUE_DARK if counted else BLUE_LIGHT, lw=2.0, solid_capstyle="butt", zorder=5 if counted else 4)
            ref = fit_reference(ser, f, sch, truth, owner.get(j))
            value = _fmt_ach(L, f.lam)
            full = value + (f" ({_fmt_err(L, ref['error'])})" if np.isfinite(ref["error"]) else "")
            items.append({"x0": x0, "x1": x1, "y": y, "real": ref["ach_real"], "full": full, "short": value})
        st = res.states[(res.states["serie"] == name) & (res.states["escalas"] == sk) & (res.states["metodo"] == key) & (res.states["gen"] == gen)]
        parts = []
        for state, word in ((OPEN, "ach_open"), (CLOSED, "ach_closed")):
            value = float(st[f"ach_{state}"].iloc[0]) if len(st) else np.nan
            if not np.isfinite(value):
                parts.append(f"{_t(L, word)} —")
            elif truth:
                parts.append(f"{_t(L, word)} {_fmt_ach(L, value)} ({_fmt_err(L, value / truth[state] - 1.0)})")
            else:
                parts.append(f"{_t(L, word)} {_fmt_ach(L, value)}")
        ax.text(1.0, 1.04, " · ".join(parts), transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8, color=INK2)
        ax.set_ylim(0.0, ach_max)
        ax.yaxis.set_major_locator(FixedLocator([0.0, 0.5 * ach_max, ach_max]))
        ax.yaxis.set_major_formatter(L.tick)
        _minute_axis(ax, sch)
        _place_ach_labels(fig, ax, items, ach_max)

    for ax in axes[-1]:
        ax.set_xlabel(_t(L, "minute"))
    fig.text(0.012, 0.5, _t(L, "co2"), rotation=90, ha="left", va="center", fontsize=8, color=INK2)
    fig.text(axes[0, 1].get_position().x0 - 0.04, 0.5, _t(L, "ach"), rotation=90, ha="right", va="center", fontsize=8, color=INK2)
    gen_text = {"conocida": _t(L, "gen_known_s", s=L.num(res.source / 3600.0, 2)), "acotada": _t(L, "gen_bounded")}.get(gen, _t(L, "gen_free"))
    axes[0, 0].text(0.0, 1.8, _series_title(L, case, name), transform=axes[0, 0].transAxes, ha="left", va="bottom", fontsize=8.6, fontweight="semibold", color=INK)
    axes[0, 0].text(0.0, 1.5, f"{TXT[L.code]['setups'][sk]} · {gen_text}", transform=axes[0, 0].transAxes, ha="left", va="bottom", fontsize=7.6, color=INK2)
    handles = [
        Patch(color=RED, alpha=0.3, lw=0, label=_t(L, "rise")),
        Patch(color=BLUE, alpha=0.3, lw=0, label=_t(L, "decay")),
        Line2D([], [], color=INK, lw=1.25, label=_t(L, "fit_ok")),
        Line2D([], [], color=MUTED, lw=0.7, label=_t(L, "fit_no")),
        Line2D([], [], color=AXIS, lw=0.7, label=_t(L, "change")),
        Line2D([], [], color=BLUE_DARK, lw=2.0, label=_t(L, "ach_est")),
        Line2D([], [], color=BLUE_LIGHT, lw=2.0, label=_t(L, "ach_extra")),
    ]
    if truth:
        handles.append(Line2D([], [], color=INK, lw=1.0, label=_t(L, "ach_real", o=L.g(truth[OPEN]), c=L.g(truth[CLOSED]))))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 1.0), ncol=4, handlelength=1.3, columnspacing=1.2, handletextpad=0.5)
    return fig


def fig_methods_ach(res: CfdResults, L: Lang, truth: dict, setup: str = "corta", methods=DETAIL_METHODS, band: float = 0.2):
    """ACH que estima cada método en cada sensor frente al de la simulación, por estado y según la generación."""
    st = res.states[res.states["escalas"] == setup]
    sens = st[st["tipo"] == "sensor"]
    room = st[st["serie"] == "todos"]
    names = [f"sensor {int(k)}" for k in res.case.sensors["sensor"]]
    offsets = dict(zip(names, np.random.default_rng(0).permutation(np.linspace(-0.3, 0.3, len(names)))))
    gens = tuple((g, k) for g, k in (("conocida", "gen_known"), ("acotada", "gen_bounded"), ("libre", "gen_free")) if (st["gen"] == g).any())
    states = ((CLOSED, "panel_closed"), (OPEN, "panel_open"))
    fig, axes, _ = _grid(len(gens), len(states), row_h=2.3, top_in=0.5, bottom_in=0.45, left=0.2, right=0.94, hspace=0.26, wspace=0.2)
    for r, (gen, gen_key) in enumerate(gens):
        for c, (state, title) in enumerate(states):
            ax = axes[r, c]
            ref = truth[state]
            ax.axvspan((1.0 - band) * ref, (1.0 + band) * ref, color=MUTED, alpha=0.16, lw=0, zorder=0)
            ax.axvline(ref, color=INK, lw=1.1, zorder=1)
            for y, key in enumerate(methods):
                sub = sens[(sens["metodo"] == key) & (sens["gen"] == gen)].set_index("serie")[f"ach_{state}"].reindex(names)
                ok = sub.dropna()
                ax.scatter(ok.to_numpy(), [y + offsets[k] for k in ok.index], s=9, color=BLUE_LIGHT, edgecolor=WHITE, lw=0.3, zorder=3)
                value = room[(room["metodo"] == key) & (room["gen"] == gen)][f"ach_{state}"]
                if len(value) and np.isfinite(value.iloc[0]):
                    ax.scatter([value.iloc[0]], [y], s=30, marker="D", color=BLUE_DARK, edgecolor=WHITE, lw=0.6, zorder=5)
                hit = 100.0 * float(np.mean(np.abs(np.nan_to_num(sub.to_numpy(dtype=float) / ref - 1.0, nan=np.inf)) <= band))
                ax.text(1.02, y, f"{hit:.0f} %", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.8, color=INK2)
            ax.text(1.02, -0.72, _t(L, "within"), transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.6, color=INK2)
            ax.set_xscale("log")
            ax.set_xlim(0.02, 40.0)
            ax.xaxis.set_major_locator(FixedLocator([0.1, 1.0, 10.0]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(L.tick)
            ax.set_ylim(len(methods) - 0.5, -0.6)
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels([TXT[L.code]["methods"][k] for k in methods], fontsize=7.4)
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{_t(L, title)} · {_t(L, gen_key)}")
    for ax in axes[-1]:
        ax.set_xlabel(_t(L, "ach"))
    handles = [
        Line2D([], [], marker="o", ls="", ms=3.6, mfc=BLUE_LIGHT, mec=WHITE, label=_t(L, "each_sensor")),
        Line2D([], [], marker="D", ls="", ms=5, mfc=BLUE_DARK, mec=WHITE, label=_t(L, "room_mean")),
        Line2D([], [], color=INK, lw=1.1, label=_t(L, "truth")),
        Patch(color=MUTED, alpha=0.3, lw=0, label="±20 %"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.19, 1.0), ncol=4, handlelength=1.2, columnspacing=1.4)
    return fig


# ------------------------------------------------------------------ sensor por sensor

DIVERGING = LinearSegmentedColormap.from_list("diverging_cfd", ["#1c5cab", "#86b6ef", "#f2f1ec", "#f4a582", "#c4441c"])


def _room_map(ax, room_xy, L: Lang, left: bool) -> None:
    ax.add_patch(Rectangle((0, 0), room_xy[0], room_xy[1], facecolor="none", edgecolor=AXIS, lw=0.9, zorder=1))
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, room_xy[0] + 0.6)
    ax.set_ylim(-0.6, room_xy[1] + 0.6)
    ax.grid(False)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    if left:
        ax.set_ylabel(_t(L, "y_m"))
    else:
        ax.tick_params(labelleft=False)


def fig_sensor_ach(case: CfdCase, sensors, L: Lang, truth: dict, room_all: dict, room_xy=(8.0, 6.0), vmax: float = 0.6):
    """ACH de cada estado en cada sensor, en planta y por pared, frente al de la simulación.

    Cortes del esquema y generación conocida. Color: error con signo; texto: ACH.
    `room_all` es el ACH de cada estado con la media de todos los sensores.
    """
    heights = sorted(case.sensors["z"].unique())
    states = ((CLOSED, "panel_closed"), (OPEN, "panel_open"))
    fig = plt.figure(figsize=(WIDTH, 5.3))
    gs = fig.add_gridspec(2, len(heights) + 2, width_ratios=[1.0] * len(heights) + [0.42, 0.85], left=0.065, right=0.985, top=0.94, bottom=0.17, wspace=0.14, hspace=0.42)
    image = None
    walls = [w for w in WALL_ORDER if (sensors["pared"] == w).any()]
    groups = [(w, z) for w in walls for z in heights if len(sensors[(sensors["pared"] == w) & np.isclose(sensors["z"], z)])]
    for r, (state, title) in enumerate(states):
        word = "abierto" if state == OPEN else "cerrado"
        col, val = f"error_{word}", f"ach_{word}"
        for j, z in enumerate(heights):
            ax = fig.add_subplot(gs[r, j])
            _room_map(ax, room_xy, L, j == 0)
            sub = sensors[np.isclose(sensors["z"], z)]
            image = ax.scatter(sub["x"], sub["y"], c=sub[col], cmap=DIVERGING, vmin=-vmax, vmax=vmax, s=150, edgecolor=AXIS, lw=0.5, zorder=3)
            for row in sub.itertuples():
                v, e = getattr(row, val), getattr(row, col)
                if np.isfinite(v):
                    ax.text(row.x, row.y, L.num(v, 1), ha="center", va="center", fontsize=5.4, color=WHITE if abs(e) > 0.4 else INK, zorder=4)
            ax.set_title(f"{_t(L, title)} · {L.num(z, 1)} m")
            if r == len(states) - 1:
                ax.set_xlabel(_t(L, "x_m"))
            else:
                ax.tick_params(labelbottom=False)
        ax = fig.add_subplot(gs[r, -1])
        ref = truth[state]
        ax.axvspan(0.8 * ref, 1.2 * ref, color=MUTED, alpha=0.16, lw=0, zorder=0)
        ax.axvline(ref, color=INK, lw=1.1, zorder=2)
        if np.isfinite(room_all.get(state, np.nan)):
            ax.axvline(room_all[state], color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=2)
        for y, (w, z) in enumerate(groups):
            sub = sensors[(sensors["pared"] == w) & np.isclose(sensors["z"], z)][val].dropna()
            ax.scatter(sub, np.full(len(sub), y), s=16, color=WALL_COLOR[w], edgecolor=WHITE, lw=0.4, alpha=0.9, zorder=3)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([f"{TXT[L.code]['walls'][w]}, {L.num(z, 1)} m" for w, z in groups], fontsize=6.8)
        ax.set_ylim(len(groups) - 0.5, -0.6)
        ax.set_xlim(0, 3.0 if state == CLOSED else 11.0)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(L.tick)
        ax.set_title(f"{_t(L, title)}, por pared")
        if r == len(states) - 1:
            ax.set_xlabel(_t(L, "ach"))
    cax = fig.add_axes([0.065, 0.075, 0.42, 0.018])
    cb = fig.colorbar(image, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_label("Error del ACH del sensor frente al de la simulación (%)", fontsize=7.4, color=INK2, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: L.g(round(100 * v))))
    handles = [Line2D([], [], color=INK, lw=1.1, label=_t(L, "truth")), Line2D([], [], color=INK2, lw=0.8, ls=(0, (3, 2)), label="media de todos los sensores"), Patch(color=MUTED, alpha=0.3, lw=0, label="±20 %")]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.985, 0.0), ncol=1, handlelength=1.4)
    return fig


def fig_sensor_why(case: CfdCase, sensors, dilution, L: Lang, room_xy=(8.0, 6.0)):
    """Por qué unos sensores estiman mejor el ACH abierto: cuándo les llega el aire nuevo."""
    data = sensors if "dilucion_rel" in sensors else sensors.merge(dilution, on="sensor")
    heights = sorted(case.sensors["z"].unique())
    fig = plt.figure(figsize=(WIDTH, 3.0))
    gs = fig.add_gridspec(1, len(heights) + 1, width_ratios=[1.0] * len(heights) + [1.25], left=0.065, right=0.985, top=0.88, bottom=0.33, wspace=0.3)
    image = None
    for j, z in enumerate(heights):
        ax = fig.add_subplot(gs[0, j])
        _room_map(ax, room_xy, L, j == 0)
        sub = data[np.isclose(data["z"], z)]
        image = ax.scatter(sub["x"], sub["y"], c=np.log2(sub["dilucion_rel"].clip(0.05, 4.0)), cmap=DIVERGING, vmin=-2, vmax=2, s=150, edgecolor=AXIS, lw=0.5, zorder=3)
        for row in sub.itertuples():
            dark = abs(np.log2(max(row.dilucion_rel, 0.05))) > 1.3
            ax.text(row.x, row.y, L.num(max(row.dilucion_rel, 0.0), 1), ha="center", va="center", fontsize=5.4, color=WHITE if dark else INK, zorder=4)
        ax.set_title(f"Dilución relativa · {L.num(z, 1)} m")
        ax.set_xlabel(_t(L, "x_m"))
    cax = fig.add_axes([0.065, 0.12, 0.38, 0.025])
    cb = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[-2, -1, 0, 1, 2])
    cb.outline.set_visible(False)
    cb.ax.set_xticklabels([L.g(v) for v in (0.25, 0.5, 1, 2, 4)])
    cb.set_label("Dilución en los 3 primeros minutos de la apertura, respecto de la media", fontsize=6.8, color=INK2, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=0)
    ax = fig.add_subplot(gs[0, -1])
    ax.axhspan(-20, 20, color=MUTED, alpha=0.16, lw=0, zorder=0)
    ax.axhline(0, color=INK, lw=0.8, zorder=1)
    ax.axvline(1, color=AXIS, lw=0.8, zorder=1)
    for w in WALL_ORDER:
        for z, marker in zip(heights, ("o", "^")):
            sub = data[(data["pared"] == w) & np.isclose(data["z"], z)]
            if len(sub):
                ax.scatter(sub["dilucion_rel"].clip(0.05, 4.0), 100 * sub["error_abierto"], s=22, marker=marker, color=WALL_COLOR[w], edgecolor=WHITE, lw=0.5, zorder=3)
    rho = data["dilucion_rel"].corr(data["error_abierto"], method="spearman")
    ax.text(0.03, 0.95, f"ρ de Spearman = {L.num(rho, 2)}", transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK2)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator([0.0625, 0.25, 1, 4]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(L.tick)
    ax.yaxis.set_major_formatter(L.tick)
    ax.set_xlabel("Dilución relativa en la apertura")
    ax.set_ylabel("Error del ACH abierto (%)")
    ax.set_title("ACH abierto según la dilución local")
    walls = [w for w in WALL_ORDER if (data["pared"] == w).any()]
    handles = [Line2D([], [], marker="o", ls="", ms=4.5, mfc=WALL_COLOR[w], mec=WHITE, label=TXT[L.code]["walls"][w]) for w in walls]
    handles += [Line2D([], [], marker=m, ls="", ms=4.5, mfc=MUTED, mec=WHITE, label=f"{L.num(z, 1)} m") for z, m in zip(heights, ("o", "^"))]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.985, 0.0), ncol=3, handlelength=1.0, columnspacing=0.9, fontsize=6.8)
    return fig


def fig_sensor_grid(res: CfdResults, z: float, L: Lang, truth: dict, gen: str = "conocida", sk: str = "corta", ncols: int = 4):
    """Un panel por sensor de una altura: la serie, los cortes del esquema, los ajustes y el ACH de cada estado."""
    case, sch = res.case, res.schedule
    rows_s = case.sensors[np.isclose(case.sensors["z"], z)].copy()
    rows_s["_w"] = rows_s["pared"].map({w: i for i, w in enumerate(WALL_ORDER)})
    rows_s = rows_s.sort_values(["_w", "sensor"])
    names = [f"sensor {int(k)}" for k in rows_s["sensor"]]
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes, height = _grid(nrows, ncols, row_h=1.3, top_in=0.7, bottom_in=0.45, left=0.075, right=0.99, hspace=0.66, wspace=0.12, sharex=True, sharey=True)
    lo = min(res.series[n].c_obs.min() for n in names)
    hi = max(res.series[n].c_obs.max() for n in names)
    for ax, name, row in zip(axes.flat, names, rows_s.itertuples()):
        ser = res.series[name]
        t_min = ser.t_h * 60.0
        _shade_open(ax, sch)
        ax.scatter(t_min, ser.c_obs, s=2.5, color=MUTED, lw=0, zorder=2)
        for f in res.fits[(name, sk, "M0", gen)]:
            if f.ok and f.accepted:
                tt = ser.t_h[f.i0 : f.i1]
                ax.plot(tt * 60.0, f.predict(tt), color=INK, lw=1.0, zorder=4)
        st = res.states[(res.states["serie"] == name) & (res.states["escalas"] == sk) & (res.states["metodo"] == "M0") & (res.states["gen"] == gen)].iloc[0]
        parts = []
        for state, word in ((CLOSED, "c"), (OPEN, "a")):
            v = float(st[f"ach_{state}"])
            parts.append(f"{word} {L.num(v, 1)} ({_fmt_err(L, v / truth[state] - 1.0)})" if np.isfinite(v) else f"{word} —")
        ax.set_title(f"{int(row.sensor)} · {TXT[L.code]['walls'][row.pared]}", fontsize=7.2, pad=11)
        ax.text(0.0, 1.03, " · ".join(parts), transform=ax.transAxes, ha="left", va="bottom", fontsize=6.0, color=INK2)
        ax.set_ylim(lo - 50, hi + 50)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(L.tick)
        _minute_axis(ax, sch)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(_t(L, "minute"))
    fig.text(0.012, 0.5, _t(L, "co2"), rotation=90, ha="left", va="center", fontsize=8, color=INK2)
    note = f"Sensores a {L.num(z, 1)} m. Sobre cada panel, ACH del estado cerrado (c) y del abierto (a), en h⁻¹, y su error frente a la simulación ({L.g(truth[CLOSED])} y {L.g(truth[OPEN])})"
    fig.text(0.075, 1.0 - 0.1 / height, note, ha="left", va="top", fontsize=7.4, color=INK2)
    return fig
