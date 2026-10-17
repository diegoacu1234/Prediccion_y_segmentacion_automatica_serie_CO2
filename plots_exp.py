"""Figuras de los experimentos, con formato de publicación.

Ancho de página completa (7,2 pulgadas), fondo blanco, PNG a 300 ppp y PDF
vectorial. Los textos salen en español o en inglés (`Lang("es")`, `Lang("en")`).

Color: cada método tiene siempre el mismo tono y la misma forma de marcador, y
el oráculo va en negro porque es la referencia. Cuando se comparan variantes de
un mismo método, el tono indica la generación (azul libre, naranja conocida) y
el relleno del marcador indica el tramo (hueco sólo transitorio, lleno con
meseta).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullFormatter, NullLocator

from .evaluate import truth_for_span
from .experiments import make_series
from .fitting import fit_exponential, fit_known_source, theory_free, theory_known
from .physics import exp_curve
from .segmenters.base import DECAY, RISE
from .variants import METHOD_KEYS, fit_variant, segments_for

WHITE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"
BLUE_DARK = "#1c5cab"
ORANGE = "#eb6834"
RED = "#e34948"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#eef5fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

METHOD_COLOR = {"M1": "#2a78d6", "M2": "#eb6834", "M3a": "#1baf7a", "M3b": "#eda100", "M4": "#e87ba4", "M5": "#008300", "M6": "#4a3aa7", "M0": INK, "manual": INK}
METHOD_MARKER = {"M1": "o", "M2": "s", "M3a": "^", "M3b": "D", "M4": "v", "M5": "P", "M6": "X", "M0": ""}
FAMILIES = (("fam_hyun", ("M1", "M2", "M3a", "M3b")), ("fam_alt", ("M4", "M5", "M6")))
WIDTH = 7.2

TXT = {
    "es": {
        "methods": {"M1": "Hyun fiel", "M2": "Hyun + decaimientos", "M3a": "Hyun + C″ en K-means", "M3b": "Hyun + quiebres", "M4": "Derivada", "M5": "PELT", "M6": "HMM", "M0": "Oráculo"},
        "scen": {"A": "A · día de aula", "B": "B · quiebres en una fase", "C": "C · caso difícil"},
        "fam_hyun": "Familia Hyun",
        "fam_alt": "Alternativas",
        "sigma": "Ruido del sensor, σ (ppm)",
        "hour": "Hora (h)",
        "co2": "CO₂ (ppm)",
        "err_med": "Error mediano de ACH (%)",
        "err_abs": "Error de ACH en valor absoluto (%)",
        "err_signed": "Error de ACH (%)",
        "exito20": "Regímenes con ACH dentro de ±20 % (%)",
        "f1": "F1 de los inicios detectados",
        "falsos": "Estimaciones espurias por serie",
        "T": "sólo transitorio",
        "TM": "transitorio + meseta",
        "T_short": "T",
        "TM_short": "T+M",
        "libre": "generación libre",
        "conocida": "generación conocida",
        "libre_short": "G libre",
        "conocida_short": "G conocida",
        "sigma_eq": "σ = {v} ppm",
        "n_tau": "Duración del régimen en constantes de tiempo, λ·T",
        "three_tau": "3 τ",
        "rise": "subida",
        "decay": "bajada",
        "plateau": "meseta",
        "obs": "CO₂ observado",
        "truth": "CO₂ sin ruido",
        "fit_ok": "ajuste aceptado",
        "fit_no": "ajuste descartado",
        "cut": "corte del método",
        "change": "cambio de régimen verdadero",
        "window": "ventana observada, en constantes de tiempo",
        "rel_se": "Error estándar relativo del ACH (%)",
        "corr": "|correlación| entre ACH y C∞ estimados",
        "th_rise": "Subida desde la línea de base",
        "th_decay": "Bajada con generación",
        "th_corr": "Acople ACH–asíntota (G libre)",
        "th_note": "escalón de 1000 ppm, σ = 5 ppm, λ = 1 1/h",
        "win_1": "ventana corta (1 τ)",
        "win_3": "transitorio completo (3 τ)",
        "win_full": "transitorio + meseta",
        "ex_rise": "Subida (20 personas, ACH real {v})",
        "ex_decay": "Bajada con generación (15 personas, ACH real {v})",
        "used": "datos usados",
        "g_error": "Error en la generación supuesta (%)",
        "cls_rise_plateau": "Subida que llega a la meseta (λ·T ≥ 3)",
        "cls_rise_short": "Subida sin meseta (λ·T < 3)",
        "cls_decay_gen": "Bajada con generación",
        "ref_free": "generación libre (mediana y rango intercuartil)",
        "diag": "error de ACH = error de G",
        "breakeven": "Error en G que iguala a no conocerla (%)",
        "never": "≥ 40 %",
        "filt_hyun": "filtros de Hyun (duración, R², C − C_ext)",
        "filt_se": "filtro por error estándar (< 25 %)",
        "wrong": "Aceptados con error > 20 %, por serie",
        "pooled": "los tres escenarios juntos",
        "success_pct": "éxito (%)",
        "variants_note": "T: sólo transitorio · T+M: transitorio + meseta",
        "ach_label": "ACH {a} ± {b}",
        "real": "real {v}",
    },
    "en": {
        "methods": {"M1": "Hyun (as published)", "M2": "Hyun + decays", "M3a": "Hyun + C″ feature", "M3b": "Hyun + kink split", "M4": "Derivative", "M5": "PELT", "M6": "HMM", "M0": "Oracle"},
        "scen": {"A": "A · classroom day", "B": "B · same-phase changes", "C": "C · hard case"},
        "fam_hyun": "Hyun family",
        "fam_alt": "Alternatives",
        "sigma": "Sensor noise, σ (ppm)",
        "hour": "Time (h)",
        "co2": "CO₂ (ppm)",
        "err_med": "Median ACH error (%)",
        "err_abs": "Absolute ACH error (%)",
        "err_signed": "ACH error (%)",
        "exito20": "Regimes with ACH within ±20 % (%)",
        "f1": "F1 of detected onsets",
        "falsos": "Spurious estimates per series",
        "T": "transient only",
        "TM": "transient + plateau",
        "T_short": "T",
        "TM_short": "T+P",
        "libre": "generation free",
        "conocida": "generation known",
        "libre_short": "G free",
        "conocida_short": "G known",
        "sigma_eq": "σ = {v} ppm",
        "n_tau": "Regime duration in time constants, λ·T",
        "three_tau": "3 τ",
        "rise": "rise",
        "decay": "decay",
        "plateau": "plateau",
        "obs": "observed CO₂",
        "truth": "noise-free CO₂",
        "fit_ok": "accepted fit",
        "fit_no": "rejected fit",
        "cut": "method breakpoint",
        "change": "true regime change",
        "window": "observed window, in time constants",
        "rel_se": "Relative standard error of ACH (%)",
        "corr": "|correlation| between estimated ACH and C∞",
        "th_rise": "Rise from baseline",
        "th_decay": "Decay with generation",
        "th_corr": "ACH–asymptote coupling (G free)",
        "th_note": "1000 ppm step, σ = 5 ppm, λ = 1 1/h",
        "win_1": "short window (1 τ)",
        "win_3": "full transient (3 τ)",
        "win_full": "transient + plateau",
        "ex_rise": "Rise (20 people, true ACH {v})",
        "ex_decay": "Decay with generation (15 people, true ACH {v})",
        "used": "data used",
        "g_error": "Error in the assumed generation (%)",
        "cls_rise_plateau": "Rise reaching the plateau (λ·T ≥ 3)",
        "cls_rise_short": "Rise without plateau (λ·T < 3)",
        "cls_decay_gen": "Decay with generation",
        "ref_free": "generation free (median and interquartile range)",
        "diag": "ACH error = G error",
        "breakeven": "G error that matches not knowing it (%)",
        "never": "≥ 40 %",
        "filt_hyun": "Hyun's filters (duration, R², C − C_ext)",
        "filt_se": "standard-error filter (< 25 %)",
        "wrong": "Accepted with error > 20 %, per series",
        "pooled": "all three scenarios pooled",
        "success_pct": "success (%)",
        "variants_note": "T: transient only · T+P: transient + plateau",
        "ach_label": "ACH {a} ± {b}",
        "real": "true {v}",
    },
}


class Lang:
    """Textos y formato de números en un idioma."""

    def __init__(self, code: str = "es"):
        self.code = code
        self.t = TXT[code]

    def __call__(self, key: str, **kw) -> str:
        return self.t[key].format(**kw) if kw else self.t[key]

    def num(self, x: float, nd: int = 1) -> str:
        s = f"{x:.{nd}f}"
        if self.code == "es":
            s = s.replace(".", ",")
        return s.replace("-", "−")

    def g(self, x: float) -> str:
        s = f"{x:g}"
        if self.code == "es":
            s = s.replace(".", ",")
        return s.replace("-", "−")

    @property
    def tick(self) -> FuncFormatter:
        return FuncFormatter(lambda v, _: self.g(v))

    def method(self, key: str) -> str:
        return self.t["methods"][key]

    def scen(self, key: str) -> str:
        return self.t["scen"][key]


def apply_pub_style() -> None:
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.dpi": 300,
            "figure.dpi": 110,
            "pdf.fonttype": 42,
            "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "text.color": INK,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK2,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 4,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.3,
            "lines.markersize": 4,
            "lines.solid_capstyle": "round",
        }
    )


def save(fig, out_dir, name: str) -> None:
    """Guarda la figura en PNG (300 ppp) y PDF y la cierra."""
    out_dir = Path(out_dir)
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)


def _grid(nrows, ncols, row_h=1.75, top_in=0.5, bottom_in=0.45, left=0.09, right=0.985, hspace=0.32, wspace=0.1, sharex=True, sharey=True, width=WIDTH):
    height = nrows * row_h + top_in + bottom_in
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), sharex=sharex, sharey=sharey, squeeze=False)
    fig.subplots_adjust(left=left, right=right, top=1.0 - top_in / height, bottom=bottom_in / height, hspace=hspace, wspace=wspace)
    return fig, axes, height


def _row_tag(ax, text, bottom: bool = False) -> None:
    """Nombre de la fila dentro del primer panel; abajo cuando las curvas van pegadas al borde superior."""
    y, va = (0.06, "bottom") if bottom else (0.94, "top")
    ax.text(0.03, y, text, transform=ax.transAxes, ha="left", va=va, fontsize=7.6, fontweight="semibold", color=INK2, zorder=8, bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.0, "alpha": 0.85})


def _stat(df, value, keys) -> pd.DataFrame:
    g = df.groupby(list(keys), sort=True)[value]
    return pd.DataFrame({"med": g.median(), "mean": g.mean(), "q1": g.quantile(0.25), "q3": g.quantile(0.75), "n": g.count()}).reset_index()


def _sigma_axis(ax, sigmas, L) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(sigmas)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(L.tick)
    ax.set_xlim(min(sigmas) * 0.8, max(sigmas) * 1.25)


def _pct_log_axis(axis, L, ticks=(0.1, 1, 10, 100)) -> None:
    axis.set_major_locator(FixedLocator(list(ticks)))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(L.tick)


def _method_handles(L, keys=METHOD_KEYS) -> list:
    out = []
    for k in keys:
        if k == "M0":
            out.append(Line2D([], [], color=INK, lw=1.3, label=L.method(k)))
        else:
            out.append(Line2D([], [], color=METHOD_COLOR[k], lw=1.3, marker=METHOD_MARKER[k], ms=4.2, mec=WHITE, mew=0.5, label=L.method(k)))
    return out


# Tramo de las figuras que no comparan tramos: transitorio + meseta. "T" queda sólo como contraste.
BASE_TRAMO = "TM"


def base_runs(runs, tramo=BASE_TRAMO, gen="libre", g_error=0.0):
    return runs[(runs["tramo"] == tramo) & (runs["gen"] == gen) & np.isclose(runs["g_error"], g_error)]


# ------------------------------------------------------------------ contexto y teoría


def fig_noise_examples(L: Lang, key: str = "A", sigmas=(1.0, 5.0, 20.0, 40.0), window=(6.0, 18.0)):
    """La misma serie con cuatro niveles de ruido."""
    fig, axes, _ = _grid(len(sigmas), 1, row_h=1.25, top_in=0.12, bottom_in=0.42, hspace=0.2, left=0.085)
    for ax, sigma in zip(axes[:, 0], sigmas):
        s = make_series(key, sigma, 0)
        m = (s.t_h >= window[0]) & (s.t_h <= window[1])
        ax.scatter(s.t_h[m], s.c_obs[m], s=1.6, color=MUTED, alpha=0.7, lw=0, rasterized=True)
        ax.plot(s.t_h[m], s.c_true[m], color=INK, lw=0.9)
        ax.text(0.012, 0.9, L("sigma_eq", v=L.g(sigma)), transform=ax.transAxes, ha="left", va="top", fontsize=8, fontweight="semibold")
        ax.set_xlim(*window)
        ax.xaxis.set_major_locator(MultipleLocator(2))
    axes[len(sigmas) // 2, 0].set_ylabel(L("co2"))
    axes[-1, 0].set_xlabel(L("hour"))
    fig.legend(
        handles=[Line2D([], [], marker="o", ls="", ms=2.5, color=MUTED, label=L("obs")), Line2D([], [], color=INK, lw=0.9, label=L("truth"))],
        loc="lower right",
        bbox_to_anchor=(0.985, 0.0),
        ncol=2,
        handlelength=1.4,
    )
    return fig


def fig_theory(L: Lang, sigma: float = 5.0, dt_h: float = 1.0 / 120.0):
    """Cota teórica: qué aporta alargar la ventana, con la generación libre y conocida."""
    fig, axes, _ = _grid(1, 3, row_h=2.25, top_in=0.5, bottom_in=0.5, wspace=0.34, sharex=True, sharey=False, left=0.08)
    lam, c_ext = 1.0, 420.0
    counts = np.unique(np.round(np.geomspace(0.3, 8.0, 60) / (lam * dt_h)).astype(int))
    windows = counts * dt_h * lam
    cases = [("th_rise", c_ext, c_ext + 1000.0), ("th_decay", c_ext + 1500.0, c_ext + 500.0)]
    for ax, (title, c0, css) in zip(axes[0, :2], cases):
        s = lam * (css - c_ext)
        free = np.array([theory_free(np.arange(m) * dt_h, c0, css, lam, sigma)[0] for m in counts])
        known = np.array([theory_known(np.arange(m) * dt_h, c0, s, c_ext, lam, sigma) for m in counts])
        ax.plot(windows, 100 * free / lam, color=BLUE, lw=1.6, label=L("libre"))
        ax.plot(windows, 100 * known / lam, color=ORANGE, lw=1.6, label=L("conocida"))
        ax.set_yscale("log")
        ax.set_ylim(0.01, 300)
        _pct_log_axis(ax.yaxis, L, (0.01, 0.1, 1, 10, 100))
        ax.set_title(L(title))
    axes[0, 0].set_ylabel(L("rel_se"))
    ax = axes[0, 2]
    _, c0, css = cases[0]
    corr = np.array([abs(theory_free(np.arange(m) * dt_h, c0, css, lam, sigma)[1]) for m in counts])
    ax.plot(windows, corr, color=INK, lw=1.6)
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(L.tick)
    ax.set_ylabel(L("corr"))
    ax.set_title(L("th_corr"))
    for ax in axes[0]:
        ax.set_xscale("log")
        ax.set_xlim(windows[0], windows[-1])
        ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 3, 5, 8]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(L.tick)
        ax.axvline(3.0, color=AXIS, lw=0.8, zorder=1)
        ax.text(3.0, 0.985, " " + L("three_tau"), transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=7, color=INK2)
    axes[0, 1].set_xlabel(L("window").capitalize())
    axes[0, 0].legend(loc="lower left", handlelength=1.4)
    fig.text(0.985, 0.985, L("th_note"), ha="right", va="top", fontsize=7.5, color=INK2)
    return fig


def fig_variants_example(L: Lang, sigma: float = 5.0, seed: int = 0):
    """Dos regímenes del escenario A ajustados con tres ventanas y con la generación libre o conocida."""
    s = make_series("A", sigma, seed)
    c_ext = s.c_ext_nominal
    fig, axes, _ = _grid(2, 3, row_h=2.05, top_in=0.52, bottom_in=0.45, hspace=0.42, wspace=0.11, sharex=False, sharey="row", left=0.085)
    for row, (reg_id, title) in enumerate(((1, "ex_rise"), (6, "ex_decay"))):
        reg = s.regimes[s.regimes["regimen"] == reg_id].iloc[0]
        i0, i1 = int(reg.i0), int(reg.i1)
        tau_n = int(round(1.0 / reg.ach / s.dt_h))
        lengths = [tau_n, min(3 * tau_n, i1 - i0), i1 - i0]
        t = s.t_h[i0:i1]
        for col, (n_use, wkey) in enumerate(zip(lengths, ("win_1", "win_3", "win_full"))):
            ax = axes[row, col]
            sl = slice(i0, i0 + n_use)
            ax.axvspan(s.t_h[i0], s.t_h[i0 + n_use - 1] + s.dt_h, color=MUTED, alpha=0.13, lw=0, zorder=0)
            ax.scatter(t, s.c_obs[i0:i1], s=2.0, color=MUTED, alpha=0.75, lw=0, rasterized=True, zorder=2)
            free = fit_exponential(s.t_h[sl], s.c_obs[sl], c_ext=c_ext, css_min=c_ext, i0=i0)
            known = fit_known_source(s.t_h[sl], s.c_obs[sl], c_ext=c_ext, s=float(reg.fuente_ppm_h), i0=i0)
            y0, step = (0.075, 0.105) if row == 0 else (0.925, -0.105)
            for k, (f, color) in enumerate(((free, BLUE), (known, ORANGE))):
                ax.plot(t, f.predict(t), color=color, lw=0.9, alpha=0.6, zorder=3 + k)
                ax.plot(s.t_h[sl], f.predict(s.t_h[sl]), color=color, lw=2.9 if k == 0 else 1.2, zorder=4 + k)
                half = 1.96 * f.lam_se if np.isfinite(f.lam_se) else np.nan
                nd = 3 if np.isfinite(half) and half < 0.05 else 2
                yy = y0 + step * ((1 - k) if row == 0 else k)
                ax.plot([0.43, 0.49], [yy, yy], transform=ax.transAxes, color=color, lw=2.0, solid_capstyle="butt", zorder=7)
                ax.text(0.52, yy, L("ach_label", a=L.num(f.lam, nd), b=L.num(half, nd)), transform=ax.transAxes, ha="left", va="center", fontsize=7.2, color=INK, zorder=7, bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.0, "alpha": 0.9})
            ax.set_title(L(wkey) if row == 0 else "", fontsize=8.2)
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_major_formatter(L.tick)
            ax.set_xlim(t[0], t[-1] + s.dt_h)
            if row == 1:
                ax.set_xlabel(L("hour"))
        axes[row, 0].set_ylabel(L("co2"))
        axes[row, 0].text(0.0, 1.16 if row == 0 else 1.05, L(title, v=L.g(reg.ach)), transform=axes[row, 0].transAxes, ha="left", va="bottom", fontsize=8.5, fontweight="semibold")
    handles = [
        Line2D([], [], color=BLUE, lw=1.8, label=L("libre")),
        Line2D([], [], color=ORANGE, lw=1.8, label=L("conocida")),
        Patch(color=MUTED, alpha=0.25, lw=0, label=L("used")),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 1.0), ncol=3, handlelength=1.5, columnspacing=1.4)
    return fig


# ------------------------------------------------------------------ paneles de segmentación


def _place_labels(fig, ax, items, lanes=(0.97, 0.87, 0.77), fontsize=6.6) -> None:
    renderer = fig.canvas.get_renderer()
    last_right = [-np.inf] * len(lanes)
    for x, text in sorted(items):
        t = ax.text(x, lanes[0], text, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=fontsize, color=INK, zorder=6)
        box = t.get_window_extent(renderer)
        lane = next((k for k in range(len(lanes)) if box.x0 > last_right[k] + 3), int(np.argmin(last_right)))
        t.set_y(lanes[lane])
        last_right[lane] = box.x1


def fig_panels_tramo(series, bases, L: Lang, methods=("M2", "M4", "M6"), gen: str = "libre"):
    """Cómo cambia cada método al tomar la meseta como asíntota: una fila por método y tramo."""
    rows = [(m, tramo) for m in methods for tramo in ("T", "TM")]
    fig, axes, _ = _grid(len(rows), 1, row_h=1.28, top_in=0.42, bottom_in=0.42, hspace=0.36, left=0.075)
    t = series.t_h
    ymax, ymin = series.c_obs.max(), min(series.c_obs.min(), 400.0)
    axes[-1, 0].set_xlim(t[0], t[-1] + series.dt_h)
    span = t[-1] - t[0]
    for ax, (key, tramo) in zip(axes[:, 0], rows):
        segs = segments_for(bases, key, tramo)
        fits = fit_variant(bases, key, segs, gen)
        ax.set_ylim(ymin - 40, ymax + 0.62 * (ymax - ymin))
        for seg in segs:
            end_t = seg.i_trans if (tramo == "TM" and seg.i_trans) else seg.i1
            end_t = min(end_t, seg.i1)
            ax.axvspan(t[seg.i0], t[end_t - 1] + series.dt_h, color=RED if seg.phase == RISE else BLUE, alpha=0.16, lw=0, zorder=0)
            if end_t < seg.i1:
                ax.axvspan(t[end_t], t[seg.i1 - 1] + series.dt_h, color=MUTED, alpha=0.15, lw=0, zorder=0)
        for i in series.change_idx:
            ax.axvline(t[i], color=AXIS, lw=0.6, zorder=1)
        ax.scatter(t, series.c_obs, s=0.8, color=MUTED, alpha=0.55, lw=0, rasterized=True, zorder=2)
        ax.vlines([t[seg.i0] for seg in segs], 0, 0.06, transform=ax.get_xaxis_transform(), color=INK2, lw=0.8, zorder=4)
        items = []
        for f in fits:
            if not f.ok:
                continue
            tt = t[f.i0 : f.i1]
            ax.plot(tt, f.predict(tt), color=INK if f.accepted else MUTED, lw=1.3 if f.accepted else 0.7, zorder=5 if f.accepted else 3)
            if f.accepted:
                true = truth_for_span(series, f.i0, f.i1)["ach_true"]
                err = (f.lam - true) / true
                items.append((0.5 * (tt[0] + tt[-1]), f"{L.num(f.lam, 2)} ({'+' if err >= 0 else '−'}{L.num(abs(100 * err), 0)} %)"))
        _place_labels(fig, ax, items)
        ax.set_title(f"{L.method(key)} · {L(tramo)}", fontsize=8.2, pad=3)
        ax.xaxis.set_major_locator(MultipleLocator(2 if span > 12 else 1))
    axes[len(rows) // 2, 0].set_ylabel(L("co2"))
    axes[-1, 0].set_xlabel(L("hour"))
    handles = [
        Patch(color=RED, alpha=0.3, lw=0, label=L("rise")),
        Patch(color=BLUE, alpha=0.3, lw=0, label=L("decay")),
        Patch(color=MUTED, alpha=0.28, lw=0, label=L("plateau")),
        Line2D([], [], color=INK, lw=1.3, label=L("fit_ok")),
        Line2D([], [], color=MUTED, lw=0.7, label=L("fit_no")),
        Line2D([], [], color=INK2, lw=0.8, marker="|", ms=5, ls="", label=L("cut")),
        Line2D([], [], color=AXIS, lw=0.6, label=L("change")),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 1.0), ncol=7, handlelength=1.2, columnspacing=1.0, handletextpad=0.4)
    return fig


# ------------------------------------------------------------------ ruido


def fig_noise_lines(runs, L: Lang, value: str, ylabel: str, log_y: bool = True, use_mean: bool = False, ylim=None, scale: float = 100.0, tag_bottom: bool = False):
    """Una métrica contra el nivel de ruido: familias en filas, escenarios en columnas."""
    data = base_runs(runs)
    scenarios = sorted(data["escenario"].unique())
    sigmas = sorted(data["sigma"].unique())
    fig, axes, _ = _grid(len(FAMILIES), len(scenarios), row_h=1.95, top_in=0.62, bottom_in=0.45, hspace=0.3)
    centre = "mean" if use_mean else "med"
    for r, (fam, keys) in enumerate(FAMILIES):
        for c, sc in enumerate(scenarios):
            ax = axes[r, c]
            for key in (*keys, "M0"):
                st = _stat(data[(data["metodo"] == key) & (data["escenario"] == sc)], value, ["sigma"])
                if st.empty:
                    continue
                color = METHOD_COLOR[key]
                if key != "M0":
                    ax.fill_between(st["sigma"], scale * st["q1"], scale * st["q3"], color=color, alpha=0.11, lw=0, zorder=2)
                ax.plot(st["sigma"], scale * st[centre], color=color, lw=1.2 if key == "M0" else 1.3, marker=METHOD_MARKER[key] or None, ms=4, mec=WHITE, mew=0.5, zorder=4 if key != "M0" else 3)
            _sigma_axis(ax, sigmas, L)
            if log_y:
                ax.set_yscale("log")
                _pct_log_axis(ax.yaxis, L)
            else:
                ax.yaxis.set_major_formatter(L.tick)
            if ylim:
                ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(L.scen(sc))
            if c == 0:
                _row_tag(ax, L(fam), tag_bottom)
    axes[-1, len(scenarios) // 2].set_xlabel(L("sigma"))
    fig.text(0.012, 0.5, ylabel, rotation=90, ha="left", va="center", fontsize=8, color=INK2)
    fig.legend(handles=_method_handles(L), loc="upper left", bbox_to_anchor=(0.08, 1.0), ncol=4, handlelength=1.6, columnspacing=1.3)
    return fig


def fig_noise_bias(reg, L: Lang):
    """Distribución del error con signo según el ruido: una fila por método."""
    data = reg[(reg["tramo"] == BASE_TRAMO) & (reg["gen"] == "libre") & np.isclose(reg["g_error"], 0.0) & reg["error_rel"].notna()]
    scenarios = sorted(data["escenario"].unique())
    sigmas = sorted(data["sigma"].unique())
    fig, axes, _ = _grid(len(METHOD_KEYS), len(scenarios), row_h=0.98, top_in=0.3, bottom_in=0.45, hspace=0.22)
    for r, key in enumerate(METHOD_KEYS):
        for c, sc in enumerate(scenarios):
            ax = axes[r, c]
            sub = data[(data["metodo"] == key) & (data["escenario"] == sc)]
            g = sub.groupby("sigma")["error_rel"]
            q = pd.DataFrame({p: 100 * g.quantile(p) for p in (0.1, 0.25, 0.5, 0.75, 0.9)})
            color = METHOD_COLOR[key] if key != "M0" else INK2
            if len(q):
                ax.fill_between(q.index, q[0.1], q[0.9], color=color, alpha=0.13, lw=0)
                ax.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.28, lw=0)
                ax.plot(q.index, q[0.5], color=color, lw=1.3)
            ax.axhline(0, color=INK2, lw=0.6, zorder=1)
            _sigma_axis(ax, sigmas, L)
            ax.set_yscale("symlog", linthresh=10.0, linscale=1.0)
            ax.set_ylim(-300, 300)
            ax.yaxis.set_major_locator(FixedLocator([-100, -10, 0, 10, 100]))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(L.tick)
            if r == 0:
                ax.set_title(L.scen(sc))
            if c == 0:
                ax.text(0.03, 0.93, L.method(key), transform=ax.transAxes, ha="left", va="top", fontsize=7.5, fontweight="semibold", bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.0, "alpha": 0.85})
    axes[-1, len(scenarios) // 2].set_xlabel(L("sigma"))
    fig.text(0.012, 0.5, L("err_signed"), rotation=90, ha="left", va="center", fontsize=8, color=INK2)
    return fig


def filter_table(fits, reg) -> pd.DataFrame:
    """Éxito y aceptados erróneos por corrida con los filtros de Hyun y con el filtro por error estándar."""
    methods = ["M4", "M5", "M6", "M0"]
    f = fits[(fits["tramo"] == BASE_TRAMO) & (fits["gen"] == "libre") & fits["metodo"].isin(methods)].copy()
    rel = f["ach_se"] / f["ach_est"]
    usable = f["motivo"].fillna("").isin(["", "R²", "duración", "C ≤ C_ext"])
    masks = {"hyun": f["aceptado"].astype(bool), "se": usable & np.isfinite(rel) & (rel < 0.25) & (f["dur_min"] >= 15.0 - 1e-9)}
    events = reg[(reg["tramo"] == BASE_TRAMO) & (reg["gen"] == "libre") & np.isclose(reg["g_error"], 0.0) & reg["metodo"].isin(methods)]
    n_events = events.groupby(["escenario", "sigma", "semilla", "metodo"])["regimen"].count().rename("n")
    truth = events.drop_duplicates(["escenario", "regimen"]).set_index(["escenario", "regimen"])["ach_real"]
    run_keys = ["escenario", "sigma", "semilla", "metodo"]
    out = []
    for name, mask in masks.items():
        sel = f[mask]
        wrong = sel[np.abs(sel["error_rel"]) > 0.2].groupby(run_keys).size().rename("wrong")
        pure = sel[sel["pureza"] >= 0.8].copy()
        pure["w"] = pure["dur_min"] * pure["ach_est"]
        agg = pure.groupby(run_keys + ["regimen"]).agg(w=("w", "sum"), d=("dur_min", "sum")).reset_index()
        agg["est"] = agg["w"] / agg["d"]
        agg["real"] = [truth.get((e, r), np.nan) for e, r in zip(agg["escenario"], agg["regimen"])]
        agg = agg[agg["real"].notna()]
        agg["hit"] = np.abs(agg["est"] / agg["real"] - 1.0) <= 0.2
        hits = agg.groupby(run_keys)["hit"].sum().rename("hits")
        tb = pd.concat([n_events, hits, wrong], axis=1).fillna({"hits": 0, "wrong": 0}).reset_index()
        tb["exito20"] = tb["hits"] / tb["n"]
        tb["filtro"] = name
        out.append(tb)
    return pd.concat(out, ignore_index=True)


def fig_filters(fits, reg, L: Lang):
    """Filtros de Hyun contra filtro por error estándar, según el ruido."""
    tb = filter_table(fits, reg)
    methods = ["M4", "M5", "M6", "M0"]
    sigmas = sorted(tb["sigma"].unique())
    fig, axes, _ = _grid(2, len(methods), row_h=1.85, top_in=0.5, bottom_in=0.45, sharey="row", hspace=0.22)
    for c, key in enumerate(methods):
        for name, color in (("hyun", BLUE), ("se", ORANGE)):
            sub = tb[(tb["metodo"] == key) & (tb["filtro"] == name)]
            for r, (value, scale) in enumerate((("exito20", 100.0), ("wrong", 1.0))):
                st = _stat(sub, value, ["sigma"])
                axes[r, c].fill_between(st["sigma"], scale * st["q1"], scale * st["q3"], color=color, alpha=0.12, lw=0)
                axes[r, c].plot(st["sigma"], scale * st["mean"], color=color, lw=1.4, marker="o", ms=3.8, mec=WHITE, mew=0.5)
        axes[0, c].set_title(L.method(key))
        for r in range(2):
            _sigma_axis(axes[r, c], sigmas, L)
            axes[r, c].yaxis.set_major_formatter(L.tick)
    axes[0, 0].set_ylim(0, 105)
    axes[1, 0].set_ylim(bottom=0)
    axes[0, 0].set_ylabel(L("exito20"), fontsize=7.4)
    axes[1, 0].set_ylabel(L("wrong"), fontsize=7.4)
    fig.text(0.54, 0.012, L("sigma"), ha="center", va="bottom", fontsize=8, color=INK2)
    handles = [Line2D([], [], color=BLUE, lw=1.4, marker="o", ms=3.8, label=L("filt_hyun")), Line2D([], [], color=ORANGE, lw=1.4, marker="o", ms=3.8, label=L("filt_se"))]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 1.0), ncol=2, handlelength=1.6)
    fig.text(0.985, 0.985, L("pooled"), ha="right", va="top", fontsize=7.5, color=INK2)
    return fig


# ------------------------------------------------------------------ comparaciones de a pares


def fig_dumbbell(data, L: Lang, value, pair_col, pair_vals, pair_labels, row_col, row_vals, row_labels, xlabel, log_x=True, use_mean=False, scale=100.0, xlim=None):
    """Antes y después por método: dos valores de `pair_col` unidos por una línea."""
    scenarios = sorted(data["escenario"].unique())
    fig, axes, _ = _grid(len(row_vals), len(scenarios), row_h=1.95, top_in=0.5, bottom_in=0.45, left=0.2, hspace=0.26, wspace=0.08)
    centre = "mean" if use_mean else "med"
    colors = (BLUE_LIGHT, BLUE_DARK)
    for r, rv in enumerate(row_vals):
        for c, sc in enumerate(scenarios):
            ax = axes[r, c]
            sub = data[(data[row_col] == rv) & (data["escenario"] == sc)]
            st = _stat(sub, value, ["metodo", pair_col])
            for y, key in enumerate(METHOD_KEYS):
                pts = [st[(st["metodo"] == key) & (st[pair_col] == pv)] for pv in pair_vals]
                xs = [scale * p[centre].iloc[0] if len(p) and np.isfinite(p[centre].iloc[0]) else np.nan for p in pts]
                if np.all(np.isfinite(xs)):
                    ax.plot(xs, [y, y], color=AXIS, lw=1.3, zorder=2, solid_capstyle="butt")
                for k, (p, x) in enumerate(zip(pts, xs)):
                    if not np.isfinite(x):
                        continue
                    ax.plot([scale * p["q1"].iloc[0], scale * p["q3"].iloc[0]], [y + (0.16 if k else -0.16)] * 2, color=colors[k], lw=0.8, zorder=3, solid_capstyle="butt")
                    ax.scatter([x], [y], s=26, facecolor=colors[k], edgecolor=WHITE, lw=0.7, zorder=4 + k)
            if log_x:
                ax.set_xscale("log")
                _pct_log_axis(ax.xaxis, L)
            else:
                ax.xaxis.set_major_formatter(L.tick)
            if xlim:
                ax.set_xlim(*xlim)
            ax.set_ylim(len(METHOD_KEYS) - 0.5, -0.6)
            ax.set_yticks(range(len(METHOD_KEYS)))
            ax.set_yticklabels([L.method(k) for k in METHOD_KEYS], fontsize=7.4)
            ax.grid(axis="y", visible=False)
            if r == 0:
                ax.set_title(L.scen(sc))
            if c == len(scenarios) - 1:
                ax.text(1.0, 1.015, row_labels[r], transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
    axes[-1, len(scenarios) // 2].set_xlabel(xlabel)
    handles = [Line2D([], [], marker="o", ls="", ms=5, mfc=colors[k], mec=WHITE, label=pair_labels[k]) for k in range(2)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.19, 1.0), ncol=2, handlelength=1.0, columnspacing=1.6)
    return fig


def fig_by_regime(reg, L: Lang, sigma: float = 5.0, methods=("M0", "M5", "M4", "M6", "M2", "M3b"), gen: str = "libre"):
    """Efecto de la meseta régimen por régimen, según cuánto dura el régimen."""
    data = reg[np.isclose(reg["sigma"], sigma) & (reg["gen"] == gen) & np.isclose(reg["g_error"], 0.0)]
    fig, axes, _ = _grid(2, 3, row_h=2.0, top_in=0.5, bottom_in=0.45, hspace=0.3)
    colors = {"T": BLUE_LIGHT, "TM": BLUE_DARK}
    cap = 300.0  # tope del eje: los errores mayores se dibujan ahí
    n_seeds = max(data["semilla"].nunique(), 1)
    for ax, key in zip(axes.ravel(), methods):
        sub = data[data["metodo"] == key].copy()
        sub["abs"] = np.abs(sub["error_rel"])
        g = sub.groupby(["escenario", "regimen", "tramo"]).agg(err=("abs", "median"), n=("abs", "count"), n_tau=("n_tau", "first"), tipo=("tipo", "first")).reset_index()
        g = g[g["n"] >= 0.5 * n_seeds]
        for (sc, rid), grp in g.groupby(["escenario", "regimen"]):
            x = float(grp["n_tau"].iloc[0])
            marker = "^" if int(grp["tipo"].iloc[0]) == RISE else "v"
            vals = {row.tramo: 100 * row.err for row in grp.itertuples()}
            if len(vals) == 2:
                ax.plot([x, x], [min(vals["T"], cap), min(vals["TM"], cap)], color=AXIS, lw=1.0, zorder=2)
            for tramo, v in vals.items():
                ax.scatter([x], [min(max(v, 0.03), cap)], s=24, marker=marker, facecolor=colors[tramo], edgecolor=WHITE, lw=0.6, zorder=3 if tramo == "T" else 4)
        ax.axvline(3.0, color=AXIS, lw=0.8, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.2, 8)
        ax.set_ylim(0.03, 450)
        ax.xaxis.set_major_locator(FixedLocator([0.3, 1, 3, 6]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(L.tick)
        _pct_log_axis(ax.yaxis, L, (0.1, 1, 10, 100))
        ax.set_title(L.method(key))
    axes[0, 0].text(3.0, 0.985, " " + L("three_tau"), transform=axes[0, 0].get_xaxis_transform(), ha="left", va="top", fontsize=7, color=INK2)
    axes[-1, 1].set_xlabel(L("n_tau"))
    fig.text(0.012, 0.5, L("err_med"), rotation=90, ha="left", va="center", fontsize=8, color=INK2)
    handles = [
        Line2D([], [], marker="o", ls="", ms=5, mfc=colors["T"], mec=WHITE, label=L("T")),
        Line2D([], [], marker="o", ls="", ms=5, mfc=colors["TM"], mec=WHITE, label=L("TM")),
        Line2D([], [], marker="^", ls="", ms=5, mfc=MUTED, mec=WHITE, label=L("rise")),
        Line2D([], [], marker="v", ls="", ms=5, mfc=MUTED, mec=WHITE, label=L("decay")),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 1.0), ncol=4, handlelength=1.0, columnspacing=1.5)
    fig.text(0.985, 0.985, L("sigma_eq", v=L.g(sigma)), ha="right", va="top", fontsize=7.5, color=INK2)
    return fig


# ------------------------------------------------------------------ generación


def fig_factorial(runs, L: Lang, value: str, ylabel: str, methods=("M0", "M5", "M4", "M2"), log_y=True, use_mean=False, ylim=None, scale=100.0, tag_bottom: bool = False):
    """Las cuatro variantes (tramo × generación) contra el ruido, para algunos métodos."""
    data = runs[np.isclose(runs["g_error"], 0.0)]
    scenarios = sorted(data["escenario"].unique())
    sigmas = sorted(data["sigma"].unique())
    fig, axes, _ = _grid(len(methods), len(scenarios), row_h=1.55, top_in=0.7, bottom_in=0.45, hspace=0.28)
    centre = "mean" if use_mean else "med"
    for r, key in enumerate(methods):
        for c, sc in enumerate(scenarios):
            ax = axes[r, c]
            for gen, color in (("libre", BLUE), ("conocida", ORANGE)):
                for tramo in ("T", "TM"):
                    st = _stat(data[(data["metodo"] == key) & (data["escenario"] == sc) & (data["gen"] == gen) & (data["tramo"] == tramo)], value, ["sigma"])
                    if st.empty:
                        continue
                    filled = tramo == "TM"
                    ax.plot(st["sigma"], scale * st[centre], color=color, lw=1.5 if filled else 0.9, marker="o", ms=4, mfc=color if filled else WHITE, mec=color, mew=0.9, zorder=4 if filled else 3)
            _sigma_axis(ax, sigmas, L)
            if log_y:
                ax.set_yscale("log")
                _pct_log_axis(ax.yaxis, L)
            else:
                ax.yaxis.set_major_formatter(L.tick)
            if ylim:
                ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(L.scen(sc))
            if c == 0:
                _row_tag(ax, L.method(key), tag_bottom)
    axes[-1, len(scenarios) // 2].set_xlabel(L("sigma"))
    fig.text(0.012, 0.5, ylabel, rotation=90, ha="left", va="center", fontsize=8, color=INK2)
    handles = [
        Line2D([], [], color=BLUE, lw=0.9, marker="o", ms=4, mfc=WHITE, mec=BLUE, label=f"{L('libre')}, {L('T')}"),
        Line2D([], [], color=BLUE, lw=1.5, marker="o", ms=4, mfc=BLUE, mec=BLUE, label=f"{L('libre')}, {L('TM')}"),
        Line2D([], [], color=ORANGE, lw=0.9, marker="o", ms=4, mfc=WHITE, mec=ORANGE, label=f"{L('conocida')}, {L('T')}"),
        Line2D([], [], color=ORANGE, lw=1.5, marker="o", ms=4, mfc=ORANGE, mec=ORANGE, label=f"{L('conocida')}, {L('TM')}"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 1.0), ncol=2, handlelength=1.8, columnspacing=1.6)
    return fig


CLASSES = (
    ("cls_rise_plateau", lambda d: (d["tipo"] == RISE) & (d["n_tau"] >= 3.0)),
    ("cls_rise_short", lambda d: (d["tipo"] == RISE) & (d["n_tau"] < 3.0)),
    ("cls_decay_gen", lambda d: (d["tipo"] == DECAY) & (d["personas"] > 0)),
)


def fig_g_error(reg, L: Lang, rows, row_labels):
    """Sensibilidad del ACH a un error en la generación supuesta.

    `rows` es una lista de (método, sigma); cada fila muestra las tres clases de régimen.
    """
    fig, axes, _ = _grid(len(rows), len(CLASSES), row_h=1.75, top_in=0.62, bottom_in=0.45, hspace=0.26)
    colors = {"T": BLUE_LIGHT, "TM": BLUE_DARK}
    for r, (key, sigma) in enumerate(rows):
        base = reg[(reg["metodo"] == key) & np.isclose(reg["sigma"], sigma) & reg["error_rel"].notna()]
        for c, (cls, rule) in enumerate(CLASSES):
            ax = axes[r, c]
            sub = base[rule(base)]
            free = 100 * sub[(sub["gen"] == "libre") & (sub["tramo"] == "TM")]["error_rel"]
            if len(free):
                ax.axhspan(free.quantile(0.25), free.quantile(0.75), color=MUTED, alpha=0.2, lw=0, zorder=1)
                ax.axhline(free.median(), color=MUTED, lw=1.0, zorder=2)
            ax.plot([-45, 45], [-45, 45], color=AXIS, lw=0.9, zorder=1)
            known = sub[sub["gen"] == "conocida"]
            for tramo in ("T", "TM"):
                g = known[known["tramo"] == tramo].groupby("g_error")["error_rel"]
                q = pd.DataFrame({"q1": 100 * g.quantile(0.25), "med": 100 * g.median(), "q3": 100 * g.quantile(0.75)})
                if q.empty:
                    continue
                x = 100 * q.index.to_numpy()
                ax.fill_between(x, q["q1"], q["q3"], color=colors[tramo], alpha=0.2, lw=0, zorder=2)
                ax.plot(x, q["med"], color=colors[tramo], lw=1.4, marker="o", ms=3.6, mec=WHITE, mew=0.5, zorder=4)
            ax.set_xlim(-44, 44)
            ax.set_ylim(-100, 100)
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.yaxis.set_major_locator(MultipleLocator(50))
            ax.xaxis.set_major_formatter(L.tick)
            ax.yaxis.set_major_formatter(L.tick)
            if r == 0:
                ax.set_title(L(cls), fontsize=8)
            if c == 0:
                _row_tag(ax, row_labels[r])
    axes[-1, 1].set_xlabel(L("g_error"))
    fig.text(0.012, 0.5, L("err_signed"), rotation=90, ha="left", va="center", fontsize=8, color=INK2)
    handles = [
        Line2D([], [], color=colors["T"], lw=1.4, marker="o", ms=3.6, label=f"{L('conocida')}, {L('T')}"),
        Line2D([], [], color=colors["TM"], lw=1.4, marker="o", ms=3.6, label=f"{L('conocida')}, {L('TM')}"),
        Patch(color=MUTED, alpha=0.35, lw=0, label=L("ref_free")),
        Line2D([], [], color=AXIS, lw=0.8, label=L("diag")),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 1.0), ncol=2, handlelength=1.6, columnspacing=1.6)
    return fig


def fig_g_success(runs, L: Lang, methods=("M0", "M5", "M4", "M2"), tramo: str = "TM"):
    """Éxito cuando la generación supuesta tiene un error, frente a dejarla libre."""
    data = runs[(runs["tramo"] == tramo) & runs["metodo"].isin(methods)]
    swept = sorted(data.loc[~np.isclose(data["g_error"], 0.0), "sigma"].unique())
    scenarios = sorted(data["escenario"].unique())
    shades = dict(zip(swept, (BLUE_LIGHT, BLUE_DARK) if len(swept) == 2 else [BLUE] * len(swept)))
    fig, axes, _ = _grid(len(methods), len(scenarios), row_h=1.55, top_in=0.7, bottom_in=0.45, hspace=0.28)
    for r, key in enumerate(methods):
        for c, sc in enumerate(scenarios):
            ax = axes[r, c]
            for sigma in swept:
                sub = data[(data["metodo"] == key) & (data["escenario"] == sc) & np.isclose(data["sigma"], sigma)]
                free = sub[(sub["gen"] == "libre") & np.isclose(sub["g_error"], 0.0)]["exito20"].mean()
                known = sub[sub["gen"] == "conocida"].groupby("g_error")["exito20"].mean()
                if np.isfinite(free):
                    ax.axhline(100 * free, color=shades[sigma], lw=0.9, zorder=2)
                if len(known):
                    ax.plot(100 * known.index.to_numpy(), 100 * known.to_numpy(), color=shades[sigma], lw=1.4, marker="o", ms=3.6, mec=WHITE, mew=0.5, zorder=4)
            ax.set_xlim(-44, 44)
            ax.set_ylim(-3, 103)
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.xaxis.set_major_formatter(L.tick)
            ax.yaxis.set_major_formatter(L.tick)
            if r == 0:
                ax.set_title(L.scen(sc))
            if c == 0:
                _row_tag(ax, L.method(key), bottom=True)
    axes[-1, len(scenarios) // 2].set_xlabel(L("g_error"))
    fig.text(0.012, 0.5, L("exito20"), rotation=90, ha="left", va="center", fontsize=8, color=INK2)
    handles = []
    for sigma in swept:
        tag = L("sigma_eq", v=L.g(sigma))
        handles.append(Line2D([], [], color=shades[sigma], lw=1.4, marker="o", ms=3.6, mec=WHITE, label=f"{L('conocida')}, {tag}"))
        handles.append(Line2D([], [], color=shades[sigma], lw=0.9, label=f"{L('libre')}, {tag}"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 1.0), ncol=2, handlelength=1.8, columnspacing=1.6)
    return fig


def breakeven_table(reg, method: str = "M0") -> pd.DataFrame:
    """Para cada régimen con generación: error en G a partir del cual conocerla ya no ayuda."""
    data = reg[(reg["metodo"] == method) & (reg["personas"] > 0) & reg["error_rel"].notna()].copy()
    data["abs"] = np.abs(data["error_rel"])
    rows = []
    for (sigma, sc, rid, tramo), grp in data.groupby(["sigma", "escenario", "regimen", "tramo"]):
        known = grp[grp["gen"] == "conocida"].groupby("g_error")["abs"].median().sort_index()
        free = grp[grp["gen"] == "libre"]["abs"]
        if len(known) < 3:
            continue
        ref = float(free.median()) if len(free) else np.inf
        eps = known.index.to_numpy()
        val = known.to_numpy()

        def cross(side):
            m = eps * side >= -1e-12
            e, v = np.abs(eps[m]), val[m]
            order = np.argsort(e)
            e, v = e[order], v[order]
            if v[0] >= ref:
                return 0.0
            for k in range(1, len(e)):
                if v[k] >= ref:
                    return float(e[k - 1] + (ref - v[k - 1]) * (e[k] - e[k - 1]) / (v[k] - v[k - 1]))
            return np.inf

        rows.append({"sigma": sigma, "escenario": sc, "regimen": rid, "tramo": tramo, "n_tau": grp["n_tau"].iloc[0], "tipo": grp["tipo"].iloc[0], "umbral": min(cross(1), cross(-1)), "err_libre": ref, "err_conocida": float(known.get(0.0, np.nan))})
    return pd.DataFrame(rows)


def fig_breakeven(reg, L: Lang, method: str = "M0", cap: float = 40.0):
    """Cuánto error en G se puede tolerar antes de que convenga dejarla libre."""
    tb = breakeven_table(reg, method)
    sigmas = sorted(tb["sigma"].unique())
    fig, axes, _ = _grid(1, len(sigmas), row_h=2.5, top_in=0.5, bottom_in=0.48, wspace=0.08)
    colors = {"T": BLUE_LIGHT, "TM": BLUE_DARK}
    for ax, sigma in zip(axes[0], sigmas):
        sub = tb[np.isclose(tb["sigma"], sigma)]
        for (sc, rid), grp in sub.groupby(["escenario", "regimen"]):
            x = float(grp["n_tau"].iloc[0])
            ys = {row.tramo: min(100 * row.umbral, cap * 1.4) for row in grp.itertuples()}
            if len(ys) == 2:
                ax.plot([x, x], [ys["T"], ys["TM"]], color=AXIS, lw=1.0, zorder=2)
            marker = "^" if int(grp["tipo"].iloc[0]) == RISE else "v"
            for tramo, y in ys.items():
                ax.scatter([x], [y], s=26, marker=marker, facecolor=colors[tramo], edgecolor=WHITE, lw=0.6, zorder=3 if tramo == "T" else 4)
        ax.axhline(cap, color=AXIS, lw=0.8)
        ax.text(0.985, cap * 1.06, L("never"), transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color=INK2)
        ax.axvline(3.0, color=AXIS, lw=0.8, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(0.2, 8)
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
        ax.set_ylim(-0.12, cap * 1.75)
        ax.xaxis.set_major_locator(FixedLocator([0.3, 1, 3, 6]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(L.tick)
        ax.yaxis.set_major_locator(FixedLocator([0, 1, 3, 10, 30]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(L.tick)
        ax.set_title(L("sigma_eq", v=L.g(sigma)))
    axes[0, 0].set_ylabel(L("breakeven"))
    fig.text(0.54, 0.012, L("n_tau"), ha="center", va="bottom", fontsize=8, color=INK2)
    handles = [
        Line2D([], [], marker="o", ls="", ms=5, mfc=colors["T"], mec=WHITE, label=L("T")),
        Line2D([], [], marker="o", ls="", ms=5, mfc=colors["TM"], mec=WHITE, label=L("TM")),
        Line2D([], [], marker="^", ls="", ms=5, mfc=MUTED, mec=WHITE, label=L("rise")),
        Line2D([], [], marker="v", ls="", ms=5, mfc=MUTED, mec=WHITE, label=L("decay")),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 1.0), ncol=4, handlelength=1.0, columnspacing=1.5)
    fig.text(0.985, 0.985, L.method(method), ha="right", va="top", fontsize=7.5, color=INK2)
    return fig


def fig_summary_heatmap(runs, L: Lang, sigmas=(5.0, 20.0)):
    """Resumen: porcentaje de regímenes con ACH dentro de ±20 %, por método y variante."""
    data = runs[np.isclose(runs["g_error"], 0.0)]
    scenarios = sorted(data["escenario"].unique())
    variants = [("T", "libre"), ("TM", "libre"), ("T", "conocida"), ("TM", "conocida")]
    fig, axes, _ = _grid(len(sigmas), len(scenarios), row_h=2.25, top_in=0.38, bottom_in=0.78, left=0.2, right=0.9, hspace=0.12, wspace=0.07, sharex=True, sharey=True)
    image = None
    for r, sigma in enumerate(sigmas):
        for c, sc in enumerate(scenarios):
            ax = axes[r, c]
            sub = data[np.isclose(data["sigma"], sigma) & (data["escenario"] == sc)]
            mat = np.full((len(METHOD_KEYS), len(variants)), np.nan)
            for i, key in enumerate(METHOD_KEYS):
                for j, (tramo, gen) in enumerate(variants):
                    v = sub[(sub["metodo"] == key) & (sub["tramo"] == tramo) & (sub["gen"] == gen)]["exito20"]
                    if len(v):
                        mat[i, j] = 100 * v.mean()
            image = ax.imshow(mat, cmap=SEQ_BLUE, vmin=0, vmax=100, aspect="auto")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if np.isfinite(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7.2, color=WHITE if mat[i, j] > 58 else INK)
            ax.set_xticks(range(len(variants)))
            ax.set_xticklabels([L(t + "_short") for t, _ in variants], fontsize=7)
            if r == len(sigmas) - 1:
                for x, gen in ((0.5, "libre"), (2.5, "conocida")):
                    ax.plot([x - 0.9, x + 0.9], [-0.115, -0.115], transform=ax.get_xaxis_transform(), color=AXIS, lw=0.8, clip_on=False)
                    ax.text(x, -0.14, L(gen + "_short"), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.2, color=INK2)
            ax.set_yticks(range(len(METHOD_KEYS)))
            ax.set_yticklabels([L.method(k) for k in METHOD_KEYS], fontsize=7.4)
            ax.grid(False)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks(np.arange(-0.5, len(variants)), minor=True)
            ax.set_yticks(np.arange(-0.5, len(METHOD_KEYS)), minor=True)
            ax.grid(which="minor", color=WHITE, linewidth=1.6)
            if r == 0:
                ax.set_title(L.scen(sc))
            if c == len(scenarios) - 1:
                ax.text(1.03, 0.5, L("sigma_eq", v=L.g(sigma)), transform=ax.transAxes, rotation=270, ha="left", va="center", fontsize=7.6, color=INK2)
    cax = fig.add_axes([0.935, 0.25, 0.012, 0.5])
    cb = fig.colorbar(image, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label(L("success_pct"), fontsize=7.5, color=INK2)
    cb.ax.tick_params(labelsize=7, length=0)
    fig.text(0.2, 0.01, L("variants_note"), ha="left", va="bottom", fontsize=7.3, color=INK2)
    return fig


# ------------------------------------------------------------------ generación acotada


def _config_label(name: str) -> str:
    """"conocida ×0,7" → "×0,7"; "acotada [0,7; 1,4]" → "0,7–1,4"."""
    if name.startswith("conocida"):
        return name.split(" ", 1)[1]
    if name.startswith("acotada"):
        lo, hi = name[name.index("[") + 1 : name.index("]")].split(";")
        return f"{lo.strip()}–{hi.strip()}"
    return name


def fig_bounded_success(summary: pd.DataFrame, groups: list, sigma: float = 5.0):
    """Éxito (±20 %) en los regímenes con gente, por método y por lo que se supone de la generación.

    `groups` es una lista de (título, [configuraciones]); cada grupo es un bloque de columnas.
    """
    order = [c for _, cfgs in groups for c in cfgs]
    data = summary[np.isclose(summary["sigma"], sigma)]
    scenarios = sorted(data["escenario"].unique())
    fig, axes, height = _grid(len(scenarios), 1, row_h=1.75, top_in=0.12, bottom_in=0.82, left=0.19, right=0.87, hspace=0.2, sharex=True, sharey=True)
    image = None
    edges = np.cumsum([len(cfgs) for _, cfgs in groups])[:-1] - 0.5
    for r, sc in enumerate(scenarios):
        ax = axes[r, 0]
        sub = data[data["escenario"] == sc]
        mat = np.full((len(METHOD_KEYS), len(order)), np.nan)
        for i, key in enumerate(METHOD_KEYS):
            for j, cfg in enumerate(order):
                v = sub[(sub["metodo"] == key) & (sub["config"] == cfg)]["exito20"]
                if len(v):
                    mat[i, j] = float(v.iloc[0])
        image = ax.imshow(mat, cmap=SEQ_BLUE, vmin=0, vmax=100, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=6.6, color=WHITE if mat[i, j] > 58 else INK)
        ax.set_yticks(range(len(METHOD_KEYS)))
        ax.set_yticklabels([TXT["es"]["methods"][k] for k in METHOD_KEYS], fontsize=7.2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([_config_label(c) for c in order], fontsize=6.8, rotation=0)
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(order)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(METHOD_KEYS)), minor=True)
        ax.grid(which="minor", color=WHITE, linewidth=1.4)
        for x in edges:
            ax.axvline(x, color=WHITE, lw=4.0)
        ax.text(1.02, 0.5, TXT["es"]["scen"][sc], transform=ax.transAxes, rotation=270, ha="left", va="center", fontsize=7.4, color=INK2)
    ax = axes[-1, 0]
    start = 0
    for title, cfgs in groups:
        a, b = start, start + len(cfgs) - 1
        start += len(cfgs)
        if not title:
            continue
        ax.plot([a - 0.4, b + 0.4], [-0.2, -0.2], transform=ax.get_xaxis_transform(), color=AXIS, lw=0.8, clip_on=False)
        ax.text((a + b) / 2, -0.24, title, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.9, color=INK2, linespacing=1.15)
    cax = fig.add_axes([0.925, 0.3, 0.012, 0.45])
    cb = fig.colorbar(image, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("Regímenes con ACH a menos de 20 % (%)", fontsize=7.0, color=INK2, labelpad=3)
    cb.ax.tick_params(labelsize=7, length=0)
    return fig


def fig_bounded_mechanism(fits: pd.DataFrame, by_dur: pd.DataFrame, cfg: str = "acotada [0,7; 1,4]", methods=("M2", "M3a", "M3b", "M4", "M5", "M6")):
    """Dónde queda la generación acotada y qué error deja, según cuántas constantes de tiempo abarca el tramo."""
    bins = [0, 0.5, 1, 2, 3, np.inf]
    labels = ["< 0,5", "0,5–1", "1–2", "2–3", "> 3"]
    sel = fits[fits["aceptado"] & (fits["s_nominal"] > 0) & fits["metodo"].isin(methods) & (fits["pureza"] >= 0.8)].copy()
    sel["dur"] = pd.cut(sel["lam_T"], bins, labels=labels, right=False)
    sigmas = sorted(sel["sigma"].unique())
    fig, axes = plt.subplots(1, 1 + len(sigmas), figsize=(WIDTH, 2.55))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.74, bottom=0.2, wspace=0.32)
    x = np.arange(len(labels))
    ax = axes[0]
    for sigma, color in zip(sigmas, (BLUE, BLUE_LIGHT)):
        d = by_dur[(by_dur["config"] == cfg) & np.isclose(by_dur["sigma"], sigma)].set_index("dur").reindex(labels)
        ax.plot(x, d["en_cota"], color=color, marker="o", ms=4, mec=WHITE, mew=0.5, lw=1.4, label=f"σ = {sigma:g} ppm")
    ax.set_ylim(-3, 103)
    ax.set_title("Ajustes con S en una cota (%)")
    ax.legend(loc="upper right", handlelength=1.4)
    series = [("libre", "generación libre", BLUE_LIGHT, "-"), ("conocida ×1", "conocida, valor exacto", ORANGE, "-"), (cfg, "acotada, 0,7–1,4", INK, "-"), ("conocida ×1,4", "conocida, ×1,4", ORANGE, (0, (3, 2)))]
    for ax, sigma in zip(axes[1:], sigmas):
        for name, label, color, ls in series:
            g = sel[(sel["config"] == name) & np.isclose(sel["sigma"], sigma)].groupby("dur", observed=False)["error_rel"]
            med = 100 * g.apply(lambda e: float(np.nanmedian(np.abs(e))) if len(e) else np.nan).reindex(labels)
            ax.plot(x, med, color=color, ls=ls, marker="o", ms=3.6, mec=WHITE, mew=0.5, lw={"libre": 3.2, cfg: 1.3}.get(name, 1.1), label=label, zorder=4 if name == cfg else 3)
        ax.set_yscale("log")
        _pct_log_axis(ax.yaxis, Lang("es"), ticks=(0.1, 1, 10, 100))
        ax.set_ylim(0.05, 300)
        ax.set_title(f"Error mediano (%), σ = {sigma:g} ppm")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.985, 1.0), ncol=4, handlelength=2.0, fontsize=6.8)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.yaxis.set_major_formatter(Lang("es").tick)
    fig.text(0.53, 0.02, "Duración del tramo en constantes de tiempo, λ·T", ha="center", va="bottom", fontsize=7.6, color=INK2)
    fig.text(0.075, 0.885, "Ajustes aceptados de seis métodos automáticos en regímenes con gente", ha="left", va="top", fontsize=7.2, color=INK2)
    return fig
