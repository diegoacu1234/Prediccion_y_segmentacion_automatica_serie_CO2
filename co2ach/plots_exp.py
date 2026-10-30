"""Estilo común de las figuras (paleta, `Lang`, `save`) y la figura 4.7.

Ancho de página (7,2 in), fondo blanco, PNG a 300 ppp y PDF. Cada método tiene
siempre el mismo color y marcador; el oráculo va en negro.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

from .evaluate import truth_for_span
from .segmenters.base import RISE
from .variants import fit_variant, segments_for

WHITE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"
BLUE_DARK = "#1c5cab"
RED = "#e34948"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#eef5fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

METHOD_COLOR = {"M1": "#2a78d6", "M2": "#eb6834", "M3a": "#1baf7a", "M3b": "#eda100", "M4": "#e87ba4", "M5": "#008300", "M6": "#4a3aa7", "M0": INK, "manual": INK}
METHOD_MARKER = {"M1": "o", "M2": "s", "M3a": "^", "M3b": "D", "M4": "v", "M5": "P", "M6": "X", "M0": ""}
WIDTH = 7.2

TXT = {
    "methods": {"M1": "Hyun fiel", "M2": "Hyun + decaimientos", "M3a": "Hyun + C″ en K-means", "M3b": "Hyun + quiebres", "M4": "Derivada", "M5": "PELT", "M6": "HMM", "M0": "Oráculo"},
    "hour": "Hora (h)",
    "co2": "CO₂ (ppm)",
    "T": "sólo transitorio",
    "TM": "transitorio + meseta",
    "rise": "subida",
    "decay": "bajada",
    "plateau": "meseta",
    "fit_ok": "ajuste aceptado",
    "fit_no": "ajuste descartado",
    "cut": "corte del método",
    "change": "cambio de régimen verdadero",
}


class Lang:
    """Textos de las figuras y números con coma decimal."""

    def __call__(self, key: str, **kw) -> str:
        return TXT[key].format(**kw) if kw else TXT[key]

    def num(self, x: float, nd: int = 1) -> str:
        return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")

    def g(self, x: float) -> str:
        return f"{x:g}".replace(".", ",").replace("-", "−")

    @property
    def tick(self) -> FuncFormatter:
        return FuncFormatter(lambda v, _: self.g(v))

    def method(self, key: str) -> str:
        return TXT["methods"][key]


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
    """Guarda en PNG (300 ppp) y PDF y cierra."""
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
    """Nombre de la fila dentro del primer panel (abajo si las curvas van arriba)."""
    y, va = (0.06, "bottom") if bottom else (0.94, "top")
    ax.text(0.03, y, text, transform=ax.transAxes, ha="left", va=va, fontsize=7.6, fontweight="semibold", color=INK2, zorder=8, bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.0, "alpha": 0.85})


# ------------------------------------------------------------------ figura 4.7


def _place_labels(fig, ax, items, lanes=(0.97, 0.87, 0.77), fontsize=6.6) -> None:
    renderer = fig.canvas.get_renderer()
    last_right = [-np.inf] * len(lanes)
    for x, text in sorted(items):
        t = ax.text(x, lanes[0], text, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=fontsize, color=INK, zorder=6)
        box = t.get_window_extent(renderer)
        lane = next((k for k in range(len(lanes)) if box.x0 > last_right[k] + 3), int(np.argmin(last_right)))
        t.set_y(lanes[lane])
        last_right[lane] = box.x1


def fig_panels_tramo(series, bases, L: Lang, methods=("M2", "M4", "M6")):
    """Cada método ajustando sólo el transitorio y el régimen completo: una fila por método y tramo."""
    rows = [(m, tramo) for m in methods for tramo in ("T", "TM")]
    fig, axes, _ = _grid(len(rows), 1, row_h=1.28, top_in=0.42, bottom_in=0.42, hspace=0.36, left=0.075)
    t = series.t_h
    ymax, ymin = series.c_obs.max(), min(series.c_obs.min(), 400.0)
    axes[-1, 0].set_xlim(t[0], t[-1] + series.dt_h)
    span = t[-1] - t[0]
    for ax, (key, tramo) in zip(axes[:, 0], rows):
        segs = segments_for(bases, key, tramo)
        fits = fit_variant(bases, key, segs)
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
