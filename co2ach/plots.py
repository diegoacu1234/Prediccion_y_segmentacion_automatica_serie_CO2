"""Figuras de las series sintéticas (5.1 y 5.2) y del apéndice B (B.1 a B.3).

Subida en rojo, bajada en azul y meseta en gris; los métodos van uno por fila.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullFormatter

from .evaluate import truth_for_span
from .fitting import lam_se_theory
from .segmenters.base import DECAY, RISE, STEADY
from .signal_utils import runs

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
RED = "#e34948"
ORANGE = "#eb6834"
PHASE_COLOR = {RISE: RED, DECAY: BLUE, STEADY: MUTED}
PHASE_ALPHA = {RISE: 0.16, DECAY: 0.16, STEADY: 0.13}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 160,
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 9,
            "text.color": INK,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK2,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
        }
    )


def _finish(fig, path):
    """Con `path` la guarda y la cierra; si no, la devuelve abierta."""
    if path is not None:
        fig.savefig(path)
        plt.close(fig)
    return fig


def es(x: float, nd: int = 1) -> str:
    """Número con coma decimal."""
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def pct(x: float, nd: int = 0, sign: bool = True) -> str:
    """Fracción -> "+6 %"."""
    s = f"{100 * x:+.{nd}f}" if sign else f"{100 * x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−") + " %"


_COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def _time_axis(ax, series) -> None:
    span = series.t_h[-1] - series.t_h[0]
    ax.xaxis.set_major_locator(MultipleLocator(2 if span > 12 else 1))
    ax.set_xlim(series.t_h[0], series.t_h[-1] + series.dt_h)
    ax.set_xlabel("Hora (h)")


def _true_changes(ax, series) -> None:
    for i in series.change_idx:
        ax.axvline(series.t_h[i], color=AXIS, lw=0.8, zorder=1)


def _shade(ax, series, labels, steady: bool) -> None:
    for i0, i1, v in runs(labels):
        if v == STEADY and not steady:
            continue
        ax.axvspan(series.t_h[i0], series.t_h[i1 - 1] + series.dt_h, color=PHASE_COLOR[v], alpha=PHASE_ALPHA[v], lw=0, zorder=0)


# ----------------------------------------------------------------------------- figura 5.1


def fig_scenarios(series_list, titles, path=None):
    """Los escenarios, uno debajo del otro: CO2 con cada régimen sombreado entero y su ACH."""
    rows = len(series_list)
    height = 3.05 * rows + 0.75
    fig = plt.figure(figsize=(13, height))
    outer = fig.add_gridspec(rows, 1, left=0.06, right=0.985, top=1.0 - 0.75 / height, bottom=0.45 / height, hspace=0.42)
    for k, (series, title) in enumerate(zip(series_list, titles)):
        ax, ax2 = outer[k].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.08).subplots(sharex=True)
        t = series.t_h
        for row in series.regimes.itertuples():
            sign = int(np.sign(series.phase[int(row.i0) : int(row.i1)].sum()))  # fase dominante (con rampa, el régimen arranca siguiendo al anterior)
            if sign:
                ax.axvspan(row.t0_h, row.t1_h, color=PHASE_COLOR[sign], alpha=PHASE_ALPHA[sign], lw=0, zorder=0)
        _true_changes(ax, series)
        ax.scatter(t, series.c_obs, s=2.5, color=MUTED, alpha=0.55, lw=0, rasterized=True, zorder=2)
        ax.plot(t, series.c_true, color=INK, lw=1.3, zorder=3)
        ymax = max(series.c_obs.max(), series.c_true.max())
        ymin = min(series.c_obs.min(), 400.0)
        ax.set_ylim(ymin - 30, ymax + 0.16 * (ymax - ymin))
        for row in series.regimes.itertuples():
            if row.t1_h - row.t0_h >= 0.045 * (t[-1] - t[0]):
                txt = "vacío" if row.personas == 0 else f"{es(row.personas, 0)} pers."
                ax.text(0.5 * (row.t0_h + row.t1_h), 0.975, txt, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color=INK2)
        ax.set_ylabel("CO₂ (ppm)")
        ax.set_title(title, loc="left", fontsize=11, fontweight="semibold")
        _true_changes(ax2, series)
        ax2.step(np.r_[t, t[-1] + series.dt_h], np.r_[series.ach, series.ach[-1]], where="post", color=INK, lw=1.6)
        for row in series.regimes.itertuples():
            if row.t1_h - row.t0_h >= 0.022 * (t[-1] - t[0]):
                ax2.text(0.5 * (row.t0_h + row.t1_h), row.ach, es(row.ach, 1), ha="center", va="bottom", fontsize=8, color=INK2)
        ax2.set_ylim(0, series.ach.max() * 1.45)
        ax2.set_ylabel("ACH (1/h)")
        ax2.yaxis.set_major_formatter(_COMMA)
        _time_axis(ax2, series)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", ms=3.5, color=MUTED, label="CO₂ observado"),
            Line2D([], [], color=INK, lw=1.3, label="CO₂ sin ruido"),
            Patch(color=RED, alpha=0.3, lw=0, label="régimen de subida"),
            Patch(color=BLUE, alpha=0.3, lw=0, label="régimen de bajada"),
            Line2D([], [], color=AXIS, lw=0.8, label="cambio de régimen"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.055, 1.0),
        ncol=5,
        handlelength=1.6,
        columnspacing=1.6,
        frameon=False,
    )
    return _finish(fig, path)


# ----------------------------------------------------------------------------- figura 5.2


def _place_labels(fig, ax, items, lanes=(0.975, 0.9, 0.825)) -> None:
    """Etiquetas sobre cada ajuste, en carriles para que no se pisen."""
    renderer = fig.canvas.get_renderer()
    last_right = [-np.inf] * len(lanes)
    for x, text in sorted(items):
        t = ax.text(x, lanes[0], text, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.6, color=INK, zorder=6)
        box = t.get_window_extent(renderer)
        lane = next((k for k in range(len(lanes)) if box.x0 > last_right[k] + 4), int(np.argmin(last_right)))
        t.set_y(lanes[lane])
        last_right[lane] = box.x1


def fig_panel(series, results, keys, path=None, title="", summaries=None):
    """Una fila por método: tramos, ajustes, ACH estimada y error."""
    n = len(keys)
    height = 2.3 * n + 1.3
    fig, axes = plt.subplots(n, 1, figsize=(13, height), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.985, top=1.0 - 1.2 / height, bottom=0.6 / height, hspace=0.27)
    t = series.t_h
    ymax, ymin = series.c_obs.max(), min(series.c_obs.min(), 400.0)
    _time_axis(axes[-1, 0], series)  # fijar límites antes de medir las etiquetas
    for ax, key in zip(axes[:, 0], keys):
        res = results[key]
        ax.set_ylim(ymin - 30, ymax + 0.52 * (ymax - ymin))
        _shade(ax, series, res.segmentation.labels, steady=not key.startswith(("M1", "M2", "M3")))
        _true_changes(ax, series)
        ax.scatter(t, series.c_obs, s=2.0, color=MUTED, alpha=0.5, lw=0, rasterized=True, zorder=2)
        cuts = sorted({s.i0 for s in res.segmentation.segments if s.i0 > 0} | {s.i1 for s in res.segmentation.transients if s.i1 < series.n})
        ax.vlines(t[cuts], 0, 0.055, transform=ax.get_xaxis_transform(), color=INK2, lw=1.0, zorder=4)
        items = []
        for f in res.fits:
            if not f.ok:
                continue
            tt = t[f.i0 : f.i1]
            ax.plot(tt, f.predict(tt), color=INK if f.accepted else MUTED, lw=1.9 if f.accepted else 1.0, zorder=5 if f.accepted else 3)
            if f.accepted:
                true = truth_for_span(series, f.i0, f.i1)["ach_true"]
                items.append((0.5 * (tt[0] + tt[-1]), f"{es(f.lam, 2)} ({pct((f.lam - true) / true)})"))
        _place_labels(fig, ax, items)
        ax.set_title(f"{key} · {res.label}", loc="left", fontsize=9.5, pad=4)
        if summaries is not None:
            sm = summaries[key]
            if sm["regimenes_cubiertos"]:
                note = f"ACH aceptado en {sm['regimenes_cubiertos']} de {sm['regimenes_con_evento']} regímenes · error mediano por régimen {pct(sm['error_abs_mediano_por_regimen'], 1, sign=False)}"
            else:
                note = f"ACH aceptado en 0 de {sm['regimenes_con_evento']} regímenes"
            ax.set_title(note, loc="right", fontsize=8.2, fontweight="normal", color=INK2, pad=4)
    axes[n // 2, 0].set_ylabel("CO₂ (ppm)")
    handles = [
        Patch(color=RED, alpha=0.3, lw=0, label="subida"),
        Patch(color=BLUE, alpha=0.3, lw=0, label="bajada"),
        *([Patch(color=MUTED, alpha=0.25, lw=0, label="meseta")] if any(not k.startswith(("M1", "M2", "M3")) for k in keys) else []),
        Line2D([], [], color=INK, lw=1.9, label="ajuste aceptado"),
        Line2D([], [], color=MUTED, lw=1.0, label="ajuste descartado"),
        Line2D([], [], color=INK2, lw=1.0, marker="|", ms=7, ls="", label="corte del método"),
        Line2D([], [], color=AXIS, lw=0.8, label="cambio de régimen verdadero"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 1.0 - 0.5 / height), ncol=7, handlelength=1.5, columnspacing=1.5)
    fig.text(0.06, 1.0 - 0.1 / height, title, fontsize=12, fontweight="semibold", ha="left", va="top")
    fig.text(0.985, 1.0 - 0.14 / height, "Sobre cada ajuste: ACH estimado en 1/h (error frente al verdadero)", fontsize=8.5, color=INK2, ha="right", va="top")
    return _finish(fig, path)


# ----------------------------------------------------------------------------- figura B.2


def fig_derivatives(series, result, path=None):
    """C, C′, C″ y salto de pendiente, con los umbrales y los quiebres."""
    info = result.segmentation.info
    cfg = info["config"]
    t = series.t_h
    fig, axes = plt.subplots(4, 1, figsize=(13, 9.2), sharex=True, gridspec_kw={"height_ratios": [1.5, 1, 1, 1], "hspace": 0.22})
    fig.subplots_adjust(left=0.07, right=0.985, top=0.9, bottom=0.065)
    kt = t[info["kinks"]]

    ax = axes[0]
    _shade(ax, series, result.segmentation.labels, steady=False)
    ax.scatter(t, series.c_obs, s=2.2, color=MUTED, alpha=0.5, lw=0, rasterized=True)
    ax.plot(t, series.c_true, color=INK, lw=1.1)
    ax.set_ylabel("CO₂ (ppm)")
    ax.set_title("Serie, con la subida y la bajada que marca el método")

    def band(ax, level, label):
        for sgn in (1, -1):
            ax.axhline(sgn * level, color=INK2, lw=0.8)
        ax.text(1.0, level, label, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7.8, color=INK2)

    ax = axes[1]
    z1 = info["z1"]
    ax.plot(t, z1, color=BLUE, lw=1.1)
    band(ax, cfg.z_enter, f"entrada ±{es(cfg.z_enter, 0)}")
    ax.axhline(0, color=AXIS, lw=0.8)
    lim = float(np.percentile(np.abs(z1), 99.5)) * 1.15
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("C′ / σ(C′)")
    ax.set_title(f"Derivada primera suavizada ({info['w1']} muestras), en desvíos de su ruido: decide subida, meseta o bajada")

    ax = axes[2]
    z2 = info["d2"] / info["d2_sd"]
    ax.plot(t, z2, color=BLUE, lw=1.1)
    ax.axhline(0, color=AXIS, lw=0.8)
    lim = float(np.percentile(np.abs(z2), 99.5)) * 1.15
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("C″ / σ(C″)")
    ax.set_title(f"Derivada segunda suavizada ({info['w2']} muestras): mezcla los quiebres con la curvatura propia de cada exponencial")

    ax = axes[3]
    zj = info["jump"] / info["jump_sd"]
    ax.plot(t, zj, color=BLUE, lw=1.1)
    band(ax, cfg.kink_z, f"umbral ±{es(cfg.kink_z, 0)}")
    ax.axhline(0, color=AXIS, lw=0.8)
    if len(kt):
        ax.scatter(kt, info["kink_z"], s=42, facecolor=INK, edgecolor=SURFACE, lw=1.2, zorder=5, label="quiebre detectado")
        ax.legend(loc="upper left", handletextpad=0.3)
    lim = max(float(np.max(np.abs(zj))) * 1.15, cfg.kink_z * 1.6)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("salto de C′ / σ")
    ax.set_title(f"Salto de pendiente (C′ a la derecha menos C′ a la izquierda, {info['half']} muestras por lado): la derivada segunda concentrada en el quiebre")
    for a in axes:
        _true_changes(a, series)
    _time_axis(axes[-1], series)
    for a in axes[:-1]:
        a.set_xlabel("")
    fig.text(0.07, 0.975, f"{series.scenario.title}: las derivadas y los quiebres", fontsize=12, fontweight="semibold", ha="left", va="top")
    fig.text(0.07, 0.945, "Las líneas verticales son los cambios de régimen verdaderos", fontsize=8.5, color=INK2, ha="left", va="top")
    return _finish(fig, path)


# ----------------------------------------------------------------------------- figura B.1


def _hyun_profile(t_h, y, c_ext, lam_v):
    """Para cada λ: mejor S y SSE del modelo de Hyun con C0 fijo (lineal en S)."""
    tau = np.asarray(t_h, dtype=float) - t_h[0]
    c0 = float(y[0])
    w = np.exp(-np.outer(lam_v, tau))
    u = 1.0 - w
    r = y[None, :] - c_ext - (c0 - c_ext) * w
    k = np.sum(u * r, axis=1) / np.sum(u * u, axis=1)  # k = S/λ
    sse = np.sum((r - k[:, None] * u) ** 2, axis=1)
    return lam_v * k, sse


def fig_identifiability(series_a, tables_general, path=None, hyun_fits=(), scenarios="A, B y C"):
    """Acople entre E y λ, cota teórica y error estándar contra error real."""
    fig = plt.figure(figsize=(13, 9.4))
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.975, top=0.885, bottom=0.07, hspace=0.36, wspace=0.2)
    c_ext = series_a.c_ext_nominal
    sigma_a = series_a.sensor.noise_sd
    rng = np.random.default_rng(3)
    regs = series_a.regimes

    def data(reg_id):
        r = regs[regs["regimen"] == reg_id].iloc[0]
        i0, i1 = int(r.i0), int(r.i1)
        y = series_a.c_true[i0:i1] + rng.normal(0, sigma_a, i1 - i0)
        return series_a.t_h[i0:i1], y, float(r.ach), float(r.fuente_ppm_h), float(r.n_tau)

    cases = [(5, BLUE, "subida sin meseta"), (1, ORANGE, "subida con su meseta")]
    ax_v = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    lam_v = np.linspace(0.15, 3.3, 420)
    rel = np.geomspace(0.4, 2.5, 500)
    for reg_id, color, name in cases:
        t_h, y, lam_true, s_true, n_tau = data(reg_id)
        label = f"{name} ({es(n_tau, 1)} constantes de tiempo)"
        s_best, _ = _hyun_profile(t_h, y, c_ext, lam_v)
        ax_v.plot(lam_v, s_best, color=color, lw=1.8, label=label)
        ax_v.scatter([lam_true], [s_true], marker="+", s=170, color=INK, lw=1.8, zorder=5)
        for f in hyun_fits:
            if f.accepted and int(np.bincount(series_a.regime_id[f.i0 : f.i1]).argmax()) == reg_id:
                ax_v.scatter([f.lam], [f.s], s=64, facecolor=SURFACE, edgecolor=INK, lw=1.6, zorder=6)
        _, sse_rel = _hyun_profile(t_h, y, c_ext, rel * lam_true)
        ax_p.plot(rel, np.sqrt(sse_rel / len(y)), color=color, lw=2.0, label=label)
    ax_v.scatter([], [], marker="+", s=110, color=INK, lw=1.8, label="valor verdadero")
    if len(hyun_fits):
        ax_v.scatter([], [], s=64, facecolor=SURFACE, edgecolor=INK, lw=1.6, label="estimación de Hyun fiel")
    ax_v.set_xlim(lam_v[0], lam_v[-1])
    ax_v.set_ylim(0, 4600)
    ax_v.xaxis.set_major_formatter(_COMMA)
    ax_v.set_xlabel("ACH, λ (1/h)")
    ax_v.set_ylabel("Fuente S = E/V (ppm/h)")
    ax_v.set_title("La mejor fuente para cada ACH es casi una recta")
    ax_v.legend(loc="upper left", fontsize=7.8)

    ax_p.axhline(sigma_a, color=INK2, lw=0.8)
    ax_p.text(rel[-1], sigma_a, f"ruido del sensor, σ = {es(sigma_a, 0)} ppm ", ha="right", va="bottom", fontsize=7.8, color=INK2)
    ax_p.set_xscale("log")
    ax_p.set_xlim(rel[0], rel[-1])
    ax_p.set_ylim(0, 110)
    ax_p.xaxis.set_major_locator(FixedLocator([0.5, 0.7, 1, 1.4, 2]))
    ax_p.xaxis.set_minor_formatter(NullFormatter())
    ax_p.xaxis.set_major_formatter(_COMMA)
    ax_p.set_xlabel("ACH supuesto / ACH verdadero")
    ax_p.set_ylabel("Residuo RMS del mejor ajuste con ese ACH (ppm)")
    ax_p.set_title("Cuánto empeora el ajuste al imponer otro ACH")
    ax_p.legend(loc="upper center", fontsize=7.8)

    ax = fig.add_subplot(gs[1, 0])
    lam, sigma, amp, dt = 1.0, 5.0, 1000.0, 1.0 / 120.0
    lt = np.geomspace(0.15, 8.0, 60)
    rise = [lam_se_theory(np.arange(0, x / lam, dt), 420.0, 420.0 + amp, lam, sigma) / lam for x in lt]
    decay = [lam_se_theory(np.arange(0, x / lam, dt), 420.0 + amp, 420.0, lam, sigma, free=("c0", "lam")) / lam for x in lt]
    ax.plot(lt, 100 * np.array(rise), color=BLUE, lw=2.0, label="asíntota libre (subida, o bajada con generación)")
    ax.plot(lt, 100 * np.array(decay), color=ORANGE, lw=2.0, label="asíntota conocida (decaimiento puro hacia el exterior)")
    ax.axhline(25, color=INK2, lw=0.8)
    ax.text(lt[-1], 25, "25 % ", ha="right", va="bottom", fontsize=7.8, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lt[0], lt[-1])
    ax.set_ylim(0.05, 3000)
    ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 3, 5]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(_COMMA)
    ax.yaxis.set_major_formatter(_COMMA)
    ax.set_xlabel("Duración del tramo en constantes de tiempo, λ·T")
    ax.set_ylabel("Error estándar relativo del ACH (%)")
    ax.set_title("Cota teórica: escalón de 1000 ppm, σ = 5 ppm, λ = 1 1/h")
    ax.legend(loc="upper right", fontsize=7.8)

    ax = fig.add_subplot(gs[1, 1])
    tb = tables_general[np.isfinite(tables_general["ach_se"]) & np.isfinite(tables_general["error_rel"]) & (tables_general["ach_se"] > 0)]
    x = np.clip(100 * (tb["ach_se"] / tb["ach_est"]).to_numpy(dtype=float), 0.02, 5e4)
    y = np.clip(100 * np.abs(tb["error_rel"]).to_numpy(dtype=float), 0.02, 5e4)
    ax.plot([0.02, 5e4], [0.02, 5e4], color=AXIS, lw=0.9, zorder=1)
    ax.scatter(x, y, s=30, facecolor=BLUE, edgecolor=SURFACE, lw=0.9, alpha=0.9, zorder=3)
    ax.axvline(25, color=INK2, lw=0.8)
    ax.text(25, 4e4, " 25 %", ha="left", va="top", fontsize=7.8, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.02, 5e4)
    ax.set_ylim(0.02, 5e4)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator([0.1, 1, 10, 100, 1000, 10000]))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.set_xlabel("Error estándar relativo que informa el ajuste (%)")
    ax.set_ylabel("Error real del ACH (%)")
    ax.set_title("El error estándar anticipa el error real")
    ax.text(0.985, 0.04, f"{len(tb)} ajustes del oráculo, la derivada, PELT y el HMM (escenarios {scenarios})", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.8, color=INK2)
    fig.text(0.07, 0.975, "Identificabilidad: el ACH sale de la curvatura", fontsize=12, fontweight="semibold", ha="left", va="top")
    fig.text(0.07, 0.945, "Arriba: ajuste (E, λ) de Hyun sobre dos tramos del escenario A. Sin la meseta, el ajuste empeora más despacio al alejarse del ACH verdadero", fontsize=8.5, color=INK2, ha="left", va="top")
    return _finish(fig, path)


# ----------------------------------------------------------------------------- figura B.3


def fig_pelt_penalty(sweeps, path=None, default_factor: float = 4.0):
    """Regímenes que encuentra PELT según el factor de penalización."""
    n = len(sweeps)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 3.9), squeeze=False)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.76, bottom=0.15, wspace=0.22)
    for ax, (title, factors, counts, n_true) in zip(axes[0], sweeps):
        ax.plot(factors, counts, color=BLUE, lw=2.0, marker="o", ms=6, mec=SURFACE, mew=1.2)
        ax.axhline(n_true, color=INK2, lw=0.9)
        ax.text(factors[-1], n_true * 1.04, f"verdaderos: {n_true}", ha="right", va="bottom", fontsize=8, color=INK2)
        ax.axvline(default_factor, color=AXIS, lw=0.9)
        ax.text(default_factor, 0.97, " valor usado", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=8, color=INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.25, 1, 4, 16, 64, 256]))
        ax.xaxis.set_major_formatter(_COMMA)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100]))
        ax.yaxis.set_major_formatter(_COMMA)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Factor de penalización (β = factor · σ² · ln n)")
        ax.set_title(title)
    axes[0, 0].set_ylabel("Regímenes que encuentra PELT")
    fig.text(0.065, 0.965, "PELT con costo exponencial: sensibilidad a la penalización", fontsize=12, fontweight="semibold", ha="left", va="top")
    fig.text(0.065, 0.895, "Si la cantidad de regímenes no cambia en un rango amplio del factor, el resultado no depende de afinarlo", fontsize=8.5, color=INK2, ha="left", va="top")
    return _finish(fig, path)
