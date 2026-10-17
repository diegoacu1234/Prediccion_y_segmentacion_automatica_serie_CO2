"""Diagramas de los métodos, para la tesis: qué hace cada uno, paso a paso, con un ejemplo calculado donde ayuda."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import MultipleLocator

from .fitting import fit_exponential
from .plots_exp import AXIS, BLUE, BLUE_DARK, INK, INK2, METHOD_COLOR, MUTED, RED, WHITE, WIDTH, Lang
from .regimes import Joiner
from .segmenters.derivative import DerivConfig, derivative_diagnostics, hysteresis_labels, segment_derivative
from .segmenters.hmm import HmmConfig, _initial_model
from .segmenters.pelt_exp import PeltConfig, make_cost, penalty
from .signal_utils import despike, n_samples, noise_sd, sg_derivative, sg_gain

SOFT_BLUE = "#eef5fe"
SOFT_RED = "#fbe9e8"
EMPTY = "#eceae3"


def _canvas(height: float, width: float = WIDTH):
    """Lienzo sin ejes, con 100 unidades de ancho y la misma escala en vertical."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 100.0 * height / width)
    ax.axis("off")
    return fig, ax, 100.0 * height / width


def _box(ax, x: float, y: float, w: float, h: float, title: str, detail: str = "", strong: bool = False, title_size: float = 7.4, detail_size: float = 6.2, face=None, edge=None):
    """Caja con título y, debajo, una aclaración. (x, y) es la esquina inferior izquierda."""
    face = face if face is not None else (SOFT_BLUE if strong else WHITE)
    edge = edge if edge is not None else (BLUE_DARK if strong else AXIS)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0", lw=1.0 if strong else 0.8, edgecolor=edge, facecolor=face, zorder=2))
    if detail:
        ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top", fontsize=title_size, fontweight="semibold", color=INK, linespacing=1.12, zorder=3)
        ax.text(x + w / 2, y + 1.4, detail, ha="center", va="bottom", fontsize=detail_size, color=INK2, linespacing=1.2, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size, fontweight="semibold", color=INK, linespacing=1.12, zorder=3)


def _arrow(ax, p0, p1, color=INK2, lw: float = 0.9, style: str = "-|>", curve: float = 0.0, scale: float = 8.0) -> None:
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=scale, lw=lw, color=color, connectionstyle=f"arc3,rad={curve}", shrinkA=0, shrinkB=0, zorder=4))


def _chain(ax, steps, y: float, h: float, x0: float = 1.0, x1: float = 99.0, gap: float = 3.2, strong=(), **kwargs):
    """Fila de cajas unidas por flechas. Devuelve el centro en x de cada una y el ancho."""
    w = (x1 - x0 - gap * (len(steps) - 1)) / len(steps)
    centers = []
    for k, (title, detail) in enumerate(steps):
        x = x0 + k * (w + gap)
        _box(ax, x, y, w, h, title, detail, strong=k in strong, **kwargs)
        centers.append(x + w / 2)
        if k:
            _arrow(ax, (x - gap + 0.45, y + h / 2), (x - 0.45, y + h / 2))
    return centers, w


def _note(ax, x: float, y_from: float, y: float, name: str, text: str, color, width: float = 22.0, size: float = 6.5) -> None:
    """Aclaración colgada de una caja: una línea hacia abajo, el nombre con su color y el texto."""
    ax.plot([x, x], [y_from, y + 0.6], color=color, lw=0.9, zorder=1)
    ax.add_patch(Circle((x - width / 2 + 0.9, y - 1.05), 0.55, color=color, lw=0, zorder=3))
    ax.text(x - width / 2 + 2.2, y, name, ha="left", va="top", fontsize=size + 0.5, fontweight="semibold", color=INK, zorder=3)
    ax.text(x - width / 2, y - 3.1, text, ha="left", va="top", fontsize=size, color=INK2, linespacing=1.22, zorder=3)


# ------------------------------------------------------------------ Hyun y sus variantes


def fig_hyun(L: Lang):
    """Los pasos del método de Hyun y en cuál interviene cada variante."""
    fig, ax, top = _canvas(2.95)
    steps = (
        ("Mediana\nmóvil", "25 muestras\n(12,5 min)"),
        ("Dos variables\npor muestra", "signo de la pendiente\ny posición relativa\n(ventana de 5)"),
        ("K-means", "tres grupos: subida,\nmeseta y bajada"),
        ("Separación\nen eventos", "suma acumulada\ny DBSCAN: al menos\n10 muestras"),
        ("Ajuste", "G y λ a la vez,\ncon C₀ fijo"),
        ("Filtros", "15 min o más,\nR² ≥ 0,1\ny C > C$_e$"),
    )
    y, h = top - 17.6, 14.0
    centers, w = _chain(ax, steps, y, h, gap=2.6, strong=(2, 3), title_size=7.4, detail_size=6.2)
    ax.text(1.0, top - 1.0, "Método de Hyun, con la configuración que publica para escuelas", ha="left", va="top", fontsize=6.8, color=INK2)
    ax.text(1.0, y - 3.0, "Variantes: cambian un paso por vez", ha="left", va="top", fontsize=6.8, color=INK2)
    yn = y - 8.2
    _note(ax, 0.5 * (centers[1] + centers[2]), y - 0.3, yn, "Hyun + C″ en K-means", "tercera variable: la derivada segunda\nsuavizada, en unidades de su ruido;\nsepara el comienzo de un transitorio\nde su llegada a la meseta", METHOD_COLOR["M3a"], width=27.0)
    _note(ax, centers[3] - 5.2, y - 0.3, yn, "Hyun + decaimientos", "separa en eventos también\nel grupo de bajada, que el\nmétodo original descarta", METHOD_COLOR["M2"], width=19.0)
    _note(ax, centers[3] + 15.5, y - 0.3, yn, "Hyun + quiebres", "además corta cada evento donde\nla pendiente salta: un cambio de\nrégimen que no cambia la fase", METHOD_COLOR["M3b"], width=22.0)
    ax.plot([centers[3] + 4.0, centers[3] + 15.5], [y - 0.3, y - 0.3], color=METHOD_COLOR["M3b"], lw=0.9, zorder=1)
    ax.text(centers[4] + 8.5, y - 3.0, "en las tres variantes, las bajadas\nse ajustan con la asíntota libre", ha="center", va="top", fontsize=6.3, color=INK2, linespacing=1.22, style="italic")
    return fig


# ------------------------------------------------------------------ derivada


def _rise_example(seed: int, noise: float, tau_min: float, dt_s: float = 30.0, hours: float = 3.0):
    """Una subida sintética: una exponencial que arranca a los 15 min, con ruido blanco y lecturas enteras."""
    t = np.arange(0.0, hours, dt_s / 3600.0)
    c_true = np.where(t < 0.25, 420.0, 1500.0 - 1080.0 * np.exp(-(t - 0.25) * 60.0 / tau_min))
    c_obs = np.round(c_true + np.random.default_rng(seed).normal(0.0, noise, len(t)))
    return t, c_true, c_obs


def _runs(mask) -> list[tuple[int, int]]:
    m = np.diff(np.r_[0, np.asarray(mask, dtype=int), 0])
    return list(zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1)))


def fig_derivative(L: Lang, seed: int = 42, noise: float = 5.0, tau_min: float = 40.0, zoom=(1.6, 3.0), cfg: DerivConfig | None = None):
    """Los pasos del método de la derivada y, al final de una subida sintética, qué cambia al decidir con uno o con dos umbrales y qué tramo se ajusta."""
    cfg = cfg or DerivConfig()
    fig, ax, top = _canvas(4.05)
    steps = (
        ("Derivada\nsuavizada", "Savitzky–Golay de\ngrado 2, ventana\nde 10 min"),
        ("En unidades\nde su ruido", "z = C′ / σ$_{C′}$, con el\nruido σ̂ estimado\nde la propia serie"),
        ("Dos umbrales", f"empieza cuando |z| > {L.g(cfg.z_enter)},\ntermina cuando\n|z| < {L.g(cfg.z_exit)}"),
        ("Limpieza", f"une cortes de menos\nde {L.g(cfg.gap_min)} min, descarta\ntramos de menos de {L.g(cfg.min_dur_min)}"),
        ("Quiebres", f"lleva cada borde al\nquiebre más cercano\n(hasta {L.g(cfg.snap_min)} min) y divide\ndonde hay uno adentro"),
    )
    _chain(ax, steps, top - 16.0, 15.0, strong=(2,), title_size=7.5, detail_size=6.3)

    t, c_true, c_obs = _rise_example(seed, noise, tau_min, hours=3.5)
    dt_h = float(t[1] - t[0])
    sigma = noise_sd(c_obs)
    clean, _ = despike(c_obs, sigma)
    dg = derivative_diagnostics(clean, dt_h, sigma, cfg)
    z = dg["d1"] / dg["d1_sd"]
    z_true = derivative_diagnostics(c_true, dt_h, sigma, cfg)["d1"] / dg["d1_sd"]
    decisions = (
        (f"Un umbral: transitorio mientras z > {L.g(cfg.z_enter)}", hysteresis_labels(z, cfg.z_enter, cfg.z_enter)),
        (f"Dos umbrales: empieza al superar {L.g(cfg.z_enter)} y termina al bajar de {L.g(cfg.z_exit)}", hysteresis_labels(z, cfg.z_enter, cfg.z_exit)),
    )
    # el tramo que se ajusta: el método completo y, después, el régimen hasta el inicio del transitorio siguiente
    seg = segment_derivative(t, c_obs, cfg)
    regimes = Joiner(t, seg.info["c_clean"], seg.info["sigma"]).regimes(seg)

    height = fig.get_figheight()
    bottom, gap = 0.40 / height, 0.10 / height
    h_z, h_bar = 1.1 / height, 0.92 / height
    x_left, w_left, x_right = 0.075, 0.25, 0.40
    w_right = 0.985 - x_right
    # (a) la subida entera, con la ventana que se amplía
    ax_c = fig.add_axes([x_left, bottom, w_left, h_z + gap + h_bar])
    for r in regimes:
        ax_c.axvspan(t[r.i0], t[r.i1 - 1] + dt_h, color=SOFT_RED, lw=0, zorder=0)
    ax_c.plot(t, c_obs, color=INK, lw=0.8)
    ax_c.axvspan(*zoom, facecolor="none", edgecolor=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax_c.text(np.mean(zoom), 700.0, "ampliado\na la derecha", ha="center", va="center", fontsize=6.0, color=INK2, linespacing=1.15)
    ax_c.set_xlim(t[0], t[-1])
    ax_c.set_ylabel("CO₂ (ppm)", fontsize=7.0)
    ax_c.set_xlabel("Tiempo (h)", fontsize=7.0)
    ax_c.yaxis.set_major_locator(MultipleLocator(400))
    ax_c.xaxis.set_major_locator(MultipleLocator(1.0))
    ax_c.xaxis.set_major_formatter(L.tick)
    ax_c.tick_params(labelsize=6.6)
    ax_c.set_title("(a) La subida y el tramo que se ajusta", fontsize=7.0, color=INK, loc="left")
    # (b) la derivada al final de la subida, con los dos umbrales
    sel = (t >= zoom[0]) & (t <= zoom[1]) & (np.arange(len(t)) < len(t) - dg["w1"] // 2)  # sin el borde, donde la ventana queda incompleta
    ax_z = fig.add_axes([x_right, bottom + h_bar + gap, w_right, h_z])
    ax_z.axhspan(cfg.z_exit, cfg.z_enter, color=SOFT_BLUE, lw=0, zorder=0)
    ax_z.plot(t[sel], z_true[sel], color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=2, label="sin ruido")
    ax_z.plot(t[sel], z[sel], color=BLUE_DARK, lw=1.1, zorder=3, label="estimada")
    for level, style in ((cfg.z_enter, "-"), (cfg.z_exit, (0, (4, 2)))):
        ax_z.axhline(level, color=INK2, lw=0.7, ls=style, zorder=2)
    ax_z.text(zoom[0] + 0.008, cfg.z_enter + 0.25, f"entra: {L.g(cfg.z_enter)}", ha="left", va="bottom", fontsize=6.2, color=INK2)
    ax_z.text(zoom[0] + 0.008, cfg.z_exit - 0.25, f"sale: {L.g(cfg.z_exit)}", ha="left", va="top", fontsize=6.2, color=INK2)
    ax_z.axhline(0.0, color=AXIS, lw=0.6, zorder=1)
    ax_z.set_xlim(*zoom)
    ax_z.set_ylim(-2.5, 9.0)
    ax_z.yaxis.set_major_locator(MultipleLocator(2))
    ax_z.set_ylabel("z = C′ / σ$_{C′}$", fontsize=7.0)
    ax_z.tick_params(labelbottom=False, labelsize=6.6)
    ax_z.legend(loc="upper right", ncol=2, fontsize=6.2, handlelength=1.8, columnspacing=1.2, frameon=False, borderaxespad=0.2)
    ax_z.set_title("(b) La derivada al final de la subida", fontsize=7.0, color=INK, loc="left")
    # (c) lo que decide cada regla en la misma ventana
    ax_b = fig.add_axes([x_right, bottom, w_right, h_bar], sharex=ax_z)
    for row, (label, lab) in enumerate(decisions):
        y = 1.0 - 1.1 * row
        pieces = [(a, b) for a, b in _runs(lab == 1) if t[a] > 0.1 and t[a] < zoom[1] and t[b - 1] > zoom[0]]  # los que se ven en la ventana
        for a, b in pieces:
            x0, x1 = max(t[a], zoom[0]), min(t[b - 1] + dt_h, zoom[1])
            if x1 > x0:
                ax_b.add_patch(Rectangle((x0, y - 0.17), x1 - x0, 0.34, facecolor=SOFT_RED, edgecolor=RED, lw=0.6, zorder=2))
        n = len(pieces)
        ax_b.text(zoom[0] + 0.005, y + 0.24, label, ha="left", va="bottom", fontsize=6.3, color=INK)
        ax_b.text(zoom[1] - 0.005, y + 0.24, f"{n} {'tramo' if n == 1 else 'tramos'}", ha="right", va="bottom", fontsize=6.3, color=INK2)
    y = 1.0 - 1.1 * len(decisions)
    for r in regimes:
        a, b = max(t[r.i0], zoom[0]), min(t[r.i1 - 1] + dt_h, zoom[1])
        ax_b.add_patch(Rectangle((a, y - 0.17), b - a, 0.34, facecolor=SOFT_RED, edgecolor=RED, lw=0.6, zorder=2))
    ax_b.text(zoom[0] + 0.005, y + 0.24, "Tramo que se ajusta: transitorio y meseta, hasta que empieza otro transitorio", ha="left", va="bottom", fontsize=6.3, color=INK)
    ax_b.text(zoom[1] - 0.005, y + 0.24, f"{len(regimes)} {'tramo' if len(regimes) == 1 else 'tramos'}", ha="right", va="bottom", fontsize=6.3, color=INK2)
    ax_b.set_ylim(y - 0.35, 1.75)
    ax_b.set_yticks([])
    for side in ("left", "right", "top"):
        ax_b.spines[side].set_visible(False)
    ax_b.set_xlabel("Tiempo (h)", fontsize=7.0)
    ax_b.tick_params(axis="x", labelsize=6.6)
    ax_b.xaxis.set_major_formatter(L.tick)
    return fig


# ------------------------------------------------------------------ PELT


def _partition_cost(t, y, breaks):
    """Residuos (RSS) de ajustar un exponencial en cada tramo de la partición, y las curvas ajustadas."""
    rss, curves = 0.0, []
    edges = [0, *breaks, len(t)]
    for i0, i1 in zip(edges[:-1], edges[1:]):
        f = fit_exponential(t[i0:i1], y[i0:i1], c_ext=0.0, i0=i0)
        curve = f.predict(t[i0:i1]) if f.ok else np.full(i1 - i0, y[i0:i1].mean())
        rss += float(np.sum((y[i0:i1] - curve) ** 2))
        curves.append((i0, i1, curve))
    return rss, curves


def _units(value: float, L: Lang) -> str:
    """Un costo en unidades de β: con un decimal si es chico, en miles si es grande."""
    return f"{L.num(value / 1000.0, 0)} mil" if value >= 10000 else L.num(value, 0 if value >= 100 else 1)


def fig_pelt(series, L: Lang, hours=(6.0, 13.5), every_min: float = 30.0, cfg: PeltConfig | None = None):
    """El criterio de PELT sobre un ejemplo: tres particiones de la misma serie y lo que cuesta cada una."""
    cfg = cfg or PeltConfig()
    keep = (series.t_h >= hours[0]) & (series.t_h <= hours[1])
    t, c = series.t_h[keep], series.c_obs[keep]
    dt_h = float(t[1] - t[0])
    sigma = noise_sd(c)
    clean, _ = despike(c, sigma)
    _, algo = make_cost(clean, dt_h, cfg)
    beta = penalty(sigma, len(t), cfg.pen_factor)
    chosen = algo.predict(pen=beta)[:-1]
    step = n_samples(every_min, dt_h)
    dense = list(range(step, len(t) - step // 2, step))
    options = (("Sin cambios", []), ("La partición que elige PELT", chosen), (f"Un cambio cada {L.g(every_min)} min", dense))

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.0), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.41, wspace=0.07)
    for ax, (title, breaks) in zip(axes, options):
        rss, curves = _partition_cost(t, clean, breaks)
        ax.plot(t, c, color=MUTED, lw=1.6, zorder=2)
        for i0, i1, curve in curves:
            ax.plot(t[i0:i1], curve, color=BLUE_DARK, lw=1.1, zorder=4)
        for b in breaks:
            ax.axvline(t[b], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=3)
        ax.set_title(title, fontsize=7.6, loc="left")
        ax.set_xlabel("Hora (h)", fontsize=7.0)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_major_formatter(L.tick)
        ax.tick_params(labelsize=6.6)
        k, total = len(breaks), rss / beta + len(breaks)
        text = f"residuos: {_units(rss / beta, L)}\ncambios: {k}\ncosto total: {_units(total, L)}"
        ax.text(0.5, -0.3, text, transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color=INK, linespacing=1.35)
    axes[0].set_ylabel("CO₂ (ppm)", fontsize=7.0)
    fig.text(0.075, 0.012, "Costo total = residuos de ajustar un exponencial en cada tramo + β por cada cambio. Todo en unidades de β, la penalización por cambio.", ha="left", va="bottom", fontsize=6.4, color=INK2)
    handles = [Line2D([], [], color=MUTED, lw=1.6, label="serie"), Line2D([], [], color=BLUE_DARK, lw=1.1, label="exponencial de cada tramo"), Line2D([], [], color=INK2, lw=0.6, ls=(0, (3, 2)), label="cambio")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=3, fontsize=6.4, handlelength=1.6, columnspacing=1.2, frameon=False)
    return fig


# ------------------------------------------------------------------ HMM


def _hmm_emissions(series, cfg: HmmConfig):
    """Media y varianza de los estados de subida y bajada que estima el HMM sobre la serie."""
    t = series.t_h
    dt_h = float(t[1] - t[0])
    sigma = noise_sd(series.c_obs)
    clean, _ = despike(series.c_obs, sigma)
    win = n_samples(cfg.win_min, dt_h, odd=True, minimum=5)
    z = sg_derivative(clean, dt_h, win, cfg.poly, 1) / (sigma * sg_gain(win, cfg.poly, 1, dt_h))
    model = _initial_model(z, cfg, cfg.dwell_min / 60.0 / dt_h)
    try:
        model.fit(z.reshape(-1, 1))
    except ValueError:
        pass
    return float(model.means_[2, 0]), float(model._covars_[2, 0])


def fig_hmm(series, L: Lang, cfg: HmmConfig | None = None):
    """Los tres estados del HMM, sus transiciones y qué derivada se espera en cada uno."""
    cfg = cfg or HmmConfig()
    mean, var = _hmm_emissions(series, cfg)
    fig, ax, top = _canvas(3.55)
    states = (("Bajada", 17.0, BLUE_DARK, SOFT_BLUE), ("Meseta", 50.0, INK2, "#f3f2ee"), ("Subida", 83.0, RED, SOFT_RED))
    y, r = top - 15.6, 6.0
    for name, x, color, face in states:
        ax.add_patch(Circle((x, y), r, facecolor=face, edgecolor=color, lw=1.2, zorder=3))
        ax.text(x, y, name, ha="center", va="center", fontsize=8.0, fontweight="semibold", color=INK, zorder=4)
        _arrow(ax, (x - 2.6, y + r - 0.5), (x + 2.6, y + r - 0.5), color=color, lw=1.3, curve=-1.7, scale=7.0)  # se queda
    ax.text(50.0, top - 1.0, f"cada estado tiende a durar: su permanencia esperada es de {L.g(cfg.dwell_min)} min", ha="center", va="top", fontsize=7.0, color=INK2)
    for (xa, xb) in ((17.0, 50.0), (50.0, 83.0)):
        _arrow(ax, (xa + r + 0.4, y + 1.4), (xb - r - 0.4, y + 1.4), lw=0.8, scale=7.0)
        _arrow(ax, (xb - r - 0.4, y - 1.4), (xa + r + 0.4, y - 1.4), lw=0.8, scale=7.0)
    _arrow(ax, (17.0 + 3.6, y - r + 0.9), (83.0 - 3.6, y - r + 0.9), lw=0.6, color=MUTED, curve=0.16, style="<|-|>", scale=6.0)
    ax.text(50.0, y - r - 4.2, "paso directo entre subida y bajada: posible, pero menos probable", ha="center", va="top", fontsize=6.6, color=INK2)

    height = fig.get_figheight()
    ax_e = fig.add_axes([0.1, 0.56 / height, 0.8, 0.9 / height])
    ax_e.patch.set_alpha(0.0)
    sd = np.sqrt(var)
    grid = np.linspace(-mean - 3.2 * sd, mean + 3.2 * sd, 800)

    def density(m, s):
        return np.exp(-0.5 * ((grid - m) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))

    peak = density(0.0, 1.0).max()
    for (name, x, color, face), (m, s) in zip(states, ((-mean, sd), (0.0, 1.0), (mean, sd))):
        d = density(m, s)
        shown = d / d.max() * (1.0 if name == "Meseta" else 0.55) * peak  # alturas a escala libre: importa dónde está cada una y cuánto se abre
        ax_e.fill_between(grid, 0, shown, color=face, lw=0, zorder=2)
        ax_e.plot(grid, shown, color=color, lw=1.2, zorder=3)
    ax_e.set_yticks([])
    ax_e.set_ylim(0, peak * 1.5)
    ax_e.set_xlim(grid[0], grid[-1])
    for side in ("left", "top", "right"):
        ax_e.spines[side].set_visible(False)
    ax_e.grid(visible=False)
    ax_e.tick_params(labelsize=6.6)
    ax_e.xaxis.set_major_formatter(L.tick)
    ax_e.set_xlabel("Lo que se observa: la derivada en unidades de su ruido, z", fontsize=7.0)
    ax_e.text(-mean, peak * 0.62, "normal de media −m\ny varianza v", ha="center", va="bottom", fontsize=6.8, color=INK2, linespacing=1.2)
    ax_e.text(mean, peak * 0.62, "normal de media +m\ny la misma varianza v", ha="center", va="bottom", fontsize=6.8, color=INK2, linespacing=1.2)
    ax_e.text(0.0, peak * 1.06, "media 0 y varianza 1, fijas: es el ruido conocido de z", ha="center", va="bottom", fontsize=6.8, color=INK2)
    ax.text(50.0, 1.0, "m, v y las probabilidades de transición se estiman de la serie (Baum–Welch); la secuencia de estados más probable sale de Viterbi", ha="center", va="bottom", fontsize=6.6, color=INK2)
    return fig


# ------------------------------------------------------------------ qué ajuste corresponde a cada tramo


def fig_fit_decision(L: Lang):
    """Del tramo detectado al ACH que se reporta: qué modelo se ajusta según lo que se sabe de la ocupación, y los criterios que siguen."""
    fig, ax, top = _canvas(3.7)
    _box(ax, 37.0, top - 8.5, 26.0, 7.0, "Tramo detectado", "subida o bajada", strong=True, title_size=8.0, detail_size=6.9)
    ax.text(50.0, top - 12.4, "¿qué se sabe de la ocupación durante el tramo?", ha="center", va="top", fontsize=7.3, color=INK2)
    branches = (
        (2.0, "Nada", "Asíntota libre", "libres C₀, C∞ ≥ C$_e$ y λ"),
        (35.5, "El aula está vacía\n(bajada con al menos 80 % en horario vacío)", "Decaimiento puro", "C∞ = C$_e$; libres C₀ y λ"),
        (69.0, "La clase está adentro\n(a lo sumo 20 % en horario vacío)", "Generación conocida", "C∞ = C$_e$ + G / (λV); libres C₀ y λ"),
    )
    y_fit, w = top - 34.0, 29.0
    for x, case, title, detail in branches:
        ax.text(x + w / 2, top - 17.0, case, ha="center", va="top", fontsize=6.9, color=INK, linespacing=1.2)
        _arrow(ax, (x + w / 2, top - 22.6), (x + w / 2, y_fit + 9.2))
        _box(ax, x, y_fit, w, 9.0, title, detail, title_size=8.0, detail_size=6.9)
        _arrow(ax, (x + w / 2, y_fit - 0.3), (x + w / 2, y_fit - 4.6))
    ax.plot([2.0 + w / 2, 69.0 + w / 2], [top - 15.2, top - 15.2], color=AXIS, lw=0.8, zorder=1)
    ax.plot([50.0, 50.0], [top - 8.7, top - 11.6], color=AXIS, lw=0.8, zorder=1)
    ax.plot([2.0 + w / 2, 69.0 + w / 2], [y_fit - 4.6, y_fit - 4.6], color=INK2, lw=0.9, zorder=1)
    steps = (
        ("Filtros de aceptación", "15 min o más, R² ≥ 0,1, C > C$_e$\ny λ fuera de las cotas"),
        ("Evento", "la curva ajustada recorre\nal menos 100 ppm"),
        ("Se reporta su ACH", "RMSE ≤ 5 % del recorrido y el tramo\nabarca una constante de tiempo o más"),
    )
    y_low = y_fit - 16.4
    _arrow(ax, (50.0, y_fit - 4.6), (50.0, y_low + 9.4))
    _chain(ax, steps, y_low, 9.2, x0=2.0, x1=98.0, gap=4.0, strong=(2,), title_size=8.0, detail_size=6.7)
    return fig


# ------------------------------------------------------------------ comparación con la referencia manual


def fig_manual_scheme(L: Lang):
    """Qué se mide al comparar los tramos de un método con un evento marcado a mano."""
    fig, ax, top = _canvas(2.3)
    x0, x1 = 26.0, 96.0

    def at(minute):  # minutos del ejemplo → coordenada
        return x0 + (x1 - x0) * minute / 70.0

    y_ref, y_met, h = top - 8.6, top - 18.6, 4.2
    ax.text(1.0, y_ref + h / 2, "Evento marcado a mano", ha="left", va="center", fontsize=7.8, fontweight="semibold", color=INK)
    ax.text(1.0, y_met + h / 2, "Tramos del método,\nde la misma fase", ha="left", va="center", fontsize=7.8, fontweight="semibold", color=INK, linespacing=1.15)
    ax.add_patch(Rectangle((at(12), y_ref), at(52) - at(12), h, facecolor=INK, lw=0, zorder=3))
    for a, b, color in ((8, 30, BLUE), (33, 46, BLUE), (58, 68, "#9cc4f2")):
        ax.add_patch(Rectangle((at(a), y_met), at(b) - at(a), h, facecolor=color, lw=0, zorder=3))
    for m in (12, 52):
        ax.plot([at(m), at(m)], [y_met - 1.5, y_ref + h + 1.5], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(at(32), y_ref + h + 2.0, "ACH de referencia: el exponencial ajustado sobre todo el evento", ha="center", va="bottom", fontsize=6.9, color=INK2)
    ax.annotate("", xy=(at(8), y_met + h + 1.6), xytext=(at(12), y_met + h + 1.6), arrowprops={"arrowstyle": "<|-|>", "lw": 0.7, "color": INK, "mutation_scale": 5, "shrinkA": 0, "shrinkB": 0})
    ax.text(at(7.2), y_met + h + 1.6, "desfase\ndel inicio", ha="right", va="center", fontsize=6.6, color=INK, linespacing=1.1)
    ax.text(at(19), y_met + h / 2, "1", ha="center", va="center", fontsize=6.6, color=WHITE, fontweight="semibold", zorder=4)
    ax.text(at(39.5), y_met + h / 2, "2", ha="center", va="center", fontsize=6.6, color=WHITE, fontweight="semibold", zorder=4)
    ax.text(at(63), y_met - 1.6, "evento de más:\nno corresponde a\nninguno marcado", ha="center", va="top", fontsize=6.6, color=INK2, linespacing=1.15)
    y_txt = y_met - 3.4
    ax.text(at(12), y_txt, "Cobertura: parte del evento marcado que\ncubren los tramos (aquí, 31 de 40 min).\nDetectado si llega a la mitad.", ha="left", va="top", fontsize=6.8, color=INK2, linespacing=1.22)
    ax.text(1.0, y_txt, "Fragmentos: cuántos tramos\nlo tocan (aquí, 2).", ha="left", va="top", fontsize=6.8, color=INK2, linespacing=1.22)
    ax.text(1.0, y_txt - 8.2, "ACH del método: el del ajuste aceptado, de la misma fase, que más se solapa con el evento marcado (aquí, el del tramo 1).", ha="left", va="top", fontsize=6.8, color=INK2)
    return fig
