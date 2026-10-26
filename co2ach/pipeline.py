"""Un método = segmentador + ajuste + filtros de aceptación (tabla 4.1).

Todos comparten el ajuste y los filtros de Hyun (duración, R², C − C_ext > 0);
la familia de Hyun mantiene su propio ajuste (E, λ) con C0 fijo. Además se
descarta un λ̂ que queda en la cota del optimizador.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .fitting import Fit, fit_exponential, fit_hyun
from .regimes import Joiner, as_regimes, transient_part, with_transient_end
from .segmenters.base import RISE, Segmentation
from .segmenters.derivative import DerivConfig, segment_derivative
from .segmenters.hmm import HmmConfig, segment_hmm
from .segmenters.hyun import SCHOOL, HyunConfig, segment_hyun, variant
from .segmenters.oracle import segment_oracle
from .segmenters.pelt_exp import PeltConfig, segment_pelt
from .signal_utils import despike, median_filter, n_samples, noise_sd


@dataclass(frozen=True)
class Filters:
    """Filtros de aceptación (los de Hyun para escuelas)."""

    min_duration_h: float = 0.25
    min_r2: float = 0.1
    # tolerancia de C − C_ext > 0: lo mayor entre c_tol_sigmas·σ y c_tol_ppm
    c_tol_sigmas: float = 3.0
    c_tol_ppm: float = 25.0


@dataclass
class Context:
    """Lo que se calcula una vez por serie."""

    series: object
    c_ext: float
    sigma: float
    c_clean: np.ndarray  # sin picos: lo que ajustan los métodos de ajuste general
    c_med: np.ndarray  # mediana de 12,5 min: sobre esta se evalúa C − C_ext

    @classmethod
    def build(cls, series, sigma_min: float = 0.0) -> "Context":
        sigma = max(noise_sd(series.c_obs), sigma_min)
        c_clean, _ = despike(series.c_obs, sigma)
        window = n_samples(12.5, series.dt_h, odd=True)
        return cls(series=series, c_ext=series.c_ext_nominal, sigma=sigma, c_clean=c_clean, c_med=median_filter(series.c_obs, window))


@dataclass
class MethodResult:
    key: str
    label: str
    segmentation: Segmentation
    fits: list = field(default_factory=list)
    filters: Filters = field(default_factory=Filters)


def apply_filters(fit: Fit, c_ref, c_ext: float, flt: Filters, sigma: float = 0.0) -> Fit:
    """Marca el ajuste como aceptado o anota por qué no.

    C − C_ext > 0 se mira sobre `c_ref` (filtrada) y con tolerancia: tomado al
    pie de la letra descarta toda subida que arranca de la línea de base.
    """
    if not fit.ok:
        fit.reason = fit.reason or "el ajuste falló"
    elif fit.at_bound:
        fit.reason = "λ en la cota"
    elif fit.duration_h < flt.min_duration_h - 1e-9:
        fit.reason = "duración"
    elif not np.isfinite(fit.r2) or fit.r2 < flt.min_r2:
        fit.reason = "R²"
    elif np.min(c_ref) - c_ext <= -max(flt.c_tol_sigmas * sigma, flt.c_tol_ppm):
        fit.reason = "C ≤ C_ext"
    else:
        fit.accepted = True
    return fit


def fit_general(ctx: Context, segmentation: Segmentation, flt: Filters) -> list:
    """Exponencial general (C0, C_ss ≥ C_ext, λ) sobre la serie sin picos."""
    t = ctx.series.t_h
    fits = []
    for seg in segmentation.transients:
        sl = slice(seg.i0, seg.i1)
        f = fit_exponential(t[sl], ctx.c_clean[sl], c_ext=ctx.c_ext, css_min=ctx.c_ext, i0=seg.i0, model="general", phase=seg.phase)
        fits.append(apply_filters(f, ctx.c_med[sl], ctx.c_ext, flt, ctx.sigma))
    return fits


def fit_hyun_family(ctx: Context, segmentation: Segmentation, flt: Filters) -> list:
    """Ajuste de Hyun (E, λ) con C0 fijo en la primera muestra, sobre la serie filtrada.

    Las bajadas usan el mismo modelo, con E ≥ 0 libre.
    """
    t = ctx.series.t_h
    data = segmentation.info["c_filt"]
    fits = []
    for seg in segmentation.transients:
        sl = slice(seg.i0, seg.i1)
        y = data[sl]
        model = "Hyun (E, λ)" if seg.phase == RISE else "decaimiento con generación"
        f = fit_hyun(t[sl], y, c_ext=ctx.c_ext, c0=y[0], i0=seg.i0, model=model, phase=seg.phase)
        fits.append(apply_filters(f, y, ctx.c_ext, flt, ctx.sigma))
    return fits


def regime_tramos(ctx: Context, seg: Segmentation, tramo: str, by_regime: bool = False, rises_only: bool = False) -> Segmentation:
    """Los tramos que se ajustan: el régimen completo ("TM") o sólo el transitorio ("T").

    Con "T", PELT y el oráculo (`by_regime`) recortan cada régimen al 95 % del
    escalón; el resto ajusta lo que detecta.
    """
    if tramo == "TM":
        joiner = Joiner(ctx.series.t_h, ctx.c_clean, ctx.sigma)
        return as_regimes(seg, joiner.regimes(seg, by_regime=by_regime, rises_only=rises_only))
    if by_regime:
        return as_regimes(seg, transient_part(with_transient_end(seg.transients, seg.labels)))
    if rises_only:
        return as_regimes(seg, [s for s in seg.transients if s.phase == RISE])
    return seg


def _hyun(key: str, label: str, cfg: HyunConfig, tramo: str = "TM"):
    def run(ctx: Context) -> MethodResult:
        flt = Filters(cfg.min_duration_h, cfg.min_r2)
        rises_only = not cfg.decays  # Hyun fiel: las bajadas sólo delimitan las subidas
        seg = segment_hyun(ctx.series.t_h, ctx.series.c_obs, replace(cfg, decays=True) if rises_only else cfg)
        seg = regime_tramos(ctx, seg, tramo, rises_only=rises_only)
        return MethodResult(key, label, seg, fit_hyun_family(ctx, seg, flt), flt)

    return run


def _general(key: str, label: str, segmenter, tramo: str = "TM", by_regime: bool = False):
    def run(ctx: Context) -> MethodResult:
        flt = Filters()
        seg = regime_tramos(ctx, segmenter(ctx), tramo, by_regime=by_regime)
        return MethodResult(key, label, seg, fit_general(ctx, seg, flt), flt)

    return run


def build_methods(sigma_min: float = 0.0, tramo: str = "TM") -> dict:
    """Los siete métodos y el oráculo. `sigma_min`: piso del ruido de los segmentadores; `tramo`: "TM" o "T"."""
    n, k = sigma_min, tramo
    return {
        "M0": ("Oráculo", _general("M0", "Oráculo", lambda ctx: segment_oracle(ctx.series), k, by_regime=True)),
        "M1": ("Hyun fiel (configuración de escuela)", _hyun("M1", "Hyun fiel (configuración de escuela)", SCHOOL, tramo=k)),
        "M2": ("Hyun + decaimiento con generación", _hyun("M2", "Hyun + decaimiento con generación", variant(decays=True), tramo=k)),
        "M3a": ("Hyun + C″ como variable de K-means", _hyun("M3a", "Hyun + C″ como variable de K-means", variant(decays=True, d2_feature=True, sigma_min=n), tramo=k)),
        "M3b": ("Hyun + corte en quiebres de C″", _hyun("M3b", "Hyun + corte en quiebres de C″", variant(decays=True, kink_split=True, sigma_min=n), tramo=k)),
        "M4": ("Derivada: umbral en C′ + quiebres", _general("M4", "Derivada: umbral en C′ + quiebres", lambda ctx: segment_derivative(ctx.series.t_h, ctx.series.c_obs, DerivConfig(sigma_min=n)), k)),
        "M5": ("PELT con costo exponencial", _general("M5", "PELT con costo exponencial", lambda ctx: segment_pelt(ctx.series.t_h, ctx.series.c_obs, PeltConfig(sigma_min=n)), k, by_regime=True)),
        "M6": ("HMM de 3 estados", _general("M6", "HMM de 3 estados", lambda ctx: segment_hmm(ctx.series.t_h, ctx.series.c_obs, HmmConfig(sigma_min=n)), k)),
    }


METHODS = build_methods()

# filas de los paneles de segmentación
FAMILY_HYUN = ["M1", "M2", "M3a", "M3b"]
FAMILY_ALT = ["M0", "M4", "M5", "M6"]


def run_all(series, keys=None, sigma_min: float = 0.0, tramo: str = "TM") -> dict:
    """Corre los métodos pedidos (todos si `keys` es None) sobre una serie."""
    methods = METHODS if (sigma_min <= 0 and tramo == "TM") else build_methods(sigma_min, tramo)
    ctx = Context.build(series, sigma_min)
    return {key: methods[key][1](ctx) for key in (keys or methods)}
