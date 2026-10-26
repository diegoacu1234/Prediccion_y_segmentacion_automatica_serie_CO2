"""Figuras de la simulación CFD (5.8 a 5.12, C.1 y C.2)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, MultipleLocator, NullLocator

from .cfd import CLOSED, OPEN, CfdCase, CfdResults, Schedule
from .plots_exp import AXIS, BLUE_DARK, BLUE_LIGHT, INK, INK2, MUTED, WHITE, WIDTH, Lang, _grid

WALL_COLOR = {"oeste": "#2a78d6", "este": "#eb6834", "norte": "#1baf7a", "sur": "#eda100", "centro": "#4a3aa7"}
WALL_ORDER = ("oeste", "norte", "centro", "sur", "este")
DETAIL_METHODS = ("M0", "M1", "M2", "M3a", "M3b", "M4", "M5", "M6")

TXT = {
    "methods": {"M1": "Hyun fiel", "M2": "Hyun + decaimientos", "M3a": "Hyun + C″ en K-means", "M3b": "Hyun + quiebres", "M4": "Derivada", "M5": "PELT", "M6": "HMM", "M0": "Cortes del esquema"},
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
    "truth": "valor de la simulación",
    "panel_closed": "ACH cerrado",
    "panel_open": "ACH abierto",
    "panel_delta": "ΔACH, abierto − cerrado",
    "gen_free": "generación libre",
    "gen_bounded": "generación acotada",
    "gen_known": "generación conocida",
    "within": "a ±20 %",
    "each_sensor": "cada sensor",
    "room_mean": "promedio de todos los sensores",
    "gen_calibrated": "generación calibrada",
}


def _t(L: Lang, key: str, **kw) -> str:
    text = TXT[key]
    return text.format(**kw) if kw else text


def _name(L: Lang, name: str) -> str:
    """Nombre de un promedio para la figura."""
    if name in TXT["groups"]:
        return TXT["groups"][name]
    for wall, label in TXT["walls"].items():
        if name in (wall, f"pared {wall}"):
            return label
    if name.startswith("z ="):
        return f"z = {L.num(float(name[3:].replace('m', '').replace(',', '.')), 1)} m"
    return name


def _fmt_err(L: Lang, error: float) -> str:
    return ("+" if error >= 0 else "−") + L.num(abs(100.0 * error), 0) + " %"


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
    """Figura 5.8: los sensores por altura y su ubicación en planta."""
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
    handles = [Line2D([], [], color=WALL_COLOR[w], lw=1.2, label=TXT["walls"][w]) for w in walls]
    handles += [Line2D([], [], color=INK, lw=1.7, label=_t(L, "mean")), Patch(color=MUTED, alpha=0.3, lw=0, label=_t(L, "open"))]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 1.0), ncol=len(handles), handlelength=1.4, columnspacing=1.2)
    return fig


# ------------------------------------------------------------------ promedios (figura 5.9)


def fig_truth(table, L: Lang, truth: dict, names, generations=("libre", "calibrada")):
    """ACH de cada estado y su diferencia frente a la simulación, con la generación libre y la calibrada."""
    panels = ((f"ach_{CLOSED}", truth.get(CLOSED), "panel_closed"), (f"ach_{OPEN}", truth.get(OPEN), "panel_open"), ("delta_ach", truth.get(OPEN) - truth.get(CLOSED), "panel_delta"))
    fig, axes, _ = _grid(1, len(panels), row_h=2.5, top_in=0.74, bottom_in=0.45, left=0.17, wspace=0.1, sharex=False)
    style = {
        "libre": dict(facecolor=WHITE, edgecolor=INK2, lw=1.0),
        "calibrada": dict(facecolor=BLUE_DARK, edgecolor=WHITE, lw=0.6),
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
        "calibrada": Line2D([], [], marker="o", ls="", ms=5, mfc=BLUE_DARK, mec=WHITE, label=_t(L, "gen_calibrated")),
    }
    handles = [
        *(marks[g] for g in generations),
        Line2D([], [], color=INK, lw=1.1, label=_t(L, "truth")),
        Patch(color=MUTED, alpha=0.3, lw=0, label="±20 %"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.16, 1.0), ncol=len(handles), handlelength=1.2, columnspacing=1.6)
    return fig


# ------------------------------------------------------------------ método por método (figura 5.12)


def fig_methods_ach(res: CfdResults, L: Lang, truth: dict, setup: str = "corta", methods=DETAIL_METHODS, band: float = 0.2):
    """ACH de cada método en cada sensor y en el promedio, por estado y según la generación."""
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
            ax.set_yticklabels([TXT["methods"][k] for k in methods], fontsize=7.4)
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


# ------------------------------------------------------------------ sensor por sensor (5.10, 5.11, C.1 y C.2)

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
    """Figura 5.10: ACH de cada estado en cada sensor, en planta (color = error, texto = ACH) y por pared.

    `room_all`: ACH de cada estado con la media de todos los sensores.
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
        ax.set_yticklabels([f"{TXT['walls'][w]}, {L.num(z, 1)} m" for w, z in groups], fontsize=6.8)
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
    """Figura 5.11: dilución relativa de cada sensor y error de su ACH abierta."""
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
    handles = [Line2D([], [], marker="o", ls="", ms=4.5, mfc=WALL_COLOR[w], mec=WHITE, label=TXT["walls"][w]) for w in walls]
    handles += [Line2D([], [], marker=m, ls="", ms=4.5, mfc=MUTED, mec=WHITE, label=f"{L.num(z, 1)} m") for z, m in zip(heights, ("o", "^"))]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.985, 0.0), ncol=3, handlelength=1.0, columnspacing=0.9, fontsize=6.8)
    return fig


def fig_sensor_grid(res: CfdResults, z: float, L: Lang, truth: dict, gen: str = "conocida", sk: str = "corta", ncols: int = 4):
    """Figuras C.1 y C.2: un panel por sensor de la altura `z`, con los ajustes y la ACH de cada estado."""
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
        ax.set_title(f"{int(row.sensor)} · {TXT['walls'][row.pared]}", fontsize=7.2, pad=11)
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
