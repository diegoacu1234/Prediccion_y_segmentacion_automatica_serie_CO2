"""Estilo común de las figuras (paleta, `Lang`, `save`) y las del Monte Carlo (4.2, 5.3, 5.4 y 5.5).

Ancho de página (7,2 in), fondo blanco, PNG a 300 ppp y PDF. Cada método tiene
siempre el mismo color y marcador; el oráculo va en negro.
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
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullLocator

from .evaluate import truth_for_span
from .segmenters.base import RISE
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
RED = "#e34948"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#eef5fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

METHOD_COLOR = {"M1": "#2a78d6", "M2": "#eb6834", "M3a": "#1baf7a", "M3b": "#eda100", "M4": "#e87ba4", "M5": "#008300", "M6": "#4a3aa7", "M0": INK, "manual": INK}
METHOD_MARKER = {"M1": "o", "M2": "s", "M3a": "^", "M3b": "D", "M4": "v", "M5": "P", "M6": "X", "M0": ""}
FAMILIES = (("fam_hyun", ("M1", "M2", "M3a", "M3b")), ("fam_alt", ("M4", "M5", "M6")))
WIDTH = 7.2

TXT = {
    "methods": {"M1": "Hyun fiel", "M2": "Hyun + decaimientos", "M3a": "Hyun + C″ en K-means", "M3b": "Hyun + quiebres", "M4": "Derivada", "M5": "PELT", "M6": "HMM", "M0": "Oráculo"},
    "scen": {"A": "A · día de aula", "B": "B · quiebres en una fase", "C": "C · caso difícil"},
    "fam_hyun": "Familia Hyun",
    "fam_alt": "Alternativas",
    "sigma": "Ruido del sensor, σ (ppm)",
    "hour": "Hora (h)",
    "co2": "CO₂ (ppm)",
    "exito20": "Regímenes con ACH dentro de ±20 % (%)",
    "T": "sólo transitorio",
    "TM": "transitorio + meseta",
    "T_short": "T",
    "TM_short": "T+M",
    "libre_short": "G libre",
    "conocida_short": "G conocida",
    "sigma_eq": "σ = {v} ppm",
    "rise": "subida",
    "decay": "bajada",
    "plateau": "meseta",
    "fit_ok": "ajuste aceptado",
    "fit_no": "ajuste descartado",
    "cut": "corte del método",
    "change": "cambio de régimen verdadero",
    "success_pct": "éxito (%)",
    "variants_note": "T: sólo transitorio · T+M: transitorio + meseta",
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

    def scen(self, key: str) -> str:
        return TXT["scen"][key]


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


# las figuras que no comparan tramos usan el régimen completo
BASE_TRAMO = "TM"


def base_runs(runs, tramo=BASE_TRAMO, gen="libre", g_error=0.0):
    return runs[(runs["tramo"] == tramo) & (runs["gen"] == gen) & np.isclose(runs["g_error"], g_error)]


# ------------------------------------------------------------------ figura 4.2


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
    """Cada método ajustando sólo el transitorio y el régimen completo: una fila por método y tramo."""
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


# ------------------------------------------------------------------ figura 5.3


def fig_noise_lines(runs, L: Lang, value: str, ylabel: str, log_y: bool = True, use_mean: bool = False, ylim=None, scale: float = 100.0, tag_bottom: bool = False):
    """Una métrica contra el ruido: familias en filas, escenarios en columnas."""
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


# ------------------------------------------------------------------ figura 5.4


def fig_summary_heatmap(runs, L: Lang, sigmas=(5.0, 20.0)):
    """% de regímenes con ACH a menos de 20 %, por método y variante de ajuste."""
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


# ------------------------------------------------------------------ figura 5.5


def _config_label(name: str) -> str:
    """"conocida ×0,7" → "×0,7"; "acotada [0,7; 1,4]" → "0,7–1,4"."""
    if name.startswith("conocida"):
        return name.split(" ", 1)[1]
    if name.startswith("acotada"):
        lo, hi = name[name.index("[") + 1 : name.index("]")].split(";")
        return f"{lo.strip()}–{hi.strip()}"
    return name


def fig_bounded_success(summary: pd.DataFrame, groups: list, sigma: float = 5.0):
    """Éxito en los regímenes con gente según lo que se supone de la generación.

    `groups`: [(título, [configuraciones]), ...]; cada grupo es un bloque de columnas.
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
        ax.set_yticklabels([TXT["methods"][k] for k in METHOD_KEYS], fontsize=7.2)
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
        ax.text(1.02, 0.5, TXT["scen"][sc], transform=ax.transAxes, rotation=270, ha="left", va="center", fontsize=7.4, color=INK2)
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
