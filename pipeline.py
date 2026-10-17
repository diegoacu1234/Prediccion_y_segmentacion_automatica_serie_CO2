"""Un método = un segmentador + una estrategia de ajuste + los filtros de aceptación.

Todos los métodos comparten el motor de ajuste y los filtros de Hyun (duración,
R² y C − C_ext > 0), así la comparación aísla la segmentación. Se agrega un
cuarto criterio: un λ̂ pegado a una cota del optimizador no es una estimación.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .fitting import Fit, fit_exponential, fit_hyun
from .regimes import Joiner, as_regimes, transient_part, with_transient_end
from .segmenters.base import DECAY, RISE, Segmentation
from .segmenters.derivative import DerivConfig, segment_derivative
from .segmenters.hmm import HmmConfig, segment_hmm
from .segmenters.hyun import PRESETS, HyunConfig, segment_hyun, variant
from .segmenters.oracle import segment_oracle
from .segmenters.pelt_exp import PeltConfig, segment_pelt
from .signal_utils import despike, median_filter, n_samples, noise_sd


@dataclass(frozen=True)
class Filters:
    """Umbrales de aceptación (por defecto, el preset School de Hyun)."""

    min_duration_h: float = 0.25
    min_r2: float = 0.1
    # tolerancia del criterio C − C_ext > 0: el mayor entre estos desvíos de ruido y este margen en ppm
    c_tol_sigmas: float = 3.0
    c_tol_ppm: float = 25.0


@dataclass
class Context:
    """Lo que se calcula una vez por serie."""

    series: object
    c_ext: float
    sigma: float
    c_clean: np.ndarray  # sin outliers: lo que ajustan los métodos de ajuste general
    c_med: np.ndarray  # mediana de 12,5 min (25 muestras cada 30 s): sobre ella se evalúa el criterio C − C_ext

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
    """Marca el ajuste como aceptado o deja anotado por qué se descarta.

    El criterio C − C_ext > 0 de Hyun se evalúa sobre la serie filtrada con
    mediana (`c_ref`) y con tolerancia: sólo descarta un tramo que queda
    claramente por debajo del exterior. Una serie que parte del nivel exterior
    cae por debajo de C_ext la mitad de las veces sólo por el ruido, y el C_ext
    supuesto tiene su propia incertidumbre; el criterio literal descartaría
    toda subida que arranca desde la línea de base. En la tesis se lo pensó para
    cuando Cs es el aire de impulsión, que sí puede superar al del recinto.
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
    """Exponencial general (C0, C_ss ≥ C_ext, λ) sobre los datos sin outliers."""
    t = ctx.series.t_h
    fits = []
    for seg in segmentation.transients:
        sl = slice(seg.i0, seg.i1)
        f = fit_exponential(t[sl], ctx.c_clean[sl], c_ext=ctx.c_ext, css_min=ctx.c_ext, i0=seg.i0, model="general", phase=seg.phase)
        fits.append(apply_filters(f, ctx.c_med[sl], ctx.c_ext, flt, ctx.sigma))
    return fits


def fit_hyun_family(ctx: Context, segmentation: Segmentation, flt: Filters, decay_model: str = "generacion", fit_on: str = "filtered", c0: str = "first") -> list:
    """Ajuste de Hyun: (E, λ) con C0 fijo en la primera muestra, sobre la serie filtrada.

    Las bajadas se ajustan con `decay_model`: "puro" (E = 0), "generacion"
    (E ≥ 0 libre, el mismo modelo que las subidas) o "aicc" (el de menor AICc).
    """
    t = ctx.series.t_h
    data = segmentation.info["c_filt"] if fit_on == "filtered" else ctx.c_clean
    fits = []
    for seg in segmentation.transients:
        sl = slice(seg.i0, seg.i1)
        y = data[sl]
        if c0 == "free":
            f = fit_exponential(t[sl], y, c_ext=ctx.c_ext, css_min=ctx.c_ext, i0=seg.i0, model="general", phase=seg.phase)
        elif seg.phase == RISE:
            f = fit_hyun(t[sl], y, c_ext=ctx.c_ext, c0=y[0], i0=seg.i0, model="Hyun (E, λ)", phase=RISE)
        else:
            pure = fit_exponential(t[sl], y, c_ext=ctx.c_ext, c0=y[0], css=ctx.c_ext, i0=seg.i0, model="decaimiento puro", phase=DECAY)
            gen = fit_hyun(t[sl], y, c_ext=ctx.c_ext, c0=y[0], i0=seg.i0, model="decaimiento con generación", phase=DECAY)
            if decay_model == "puro":
                f = pure
            elif decay_model == "generacion":
                f = gen
            else:
                f = pure if (pure.ok and (not gen.ok or pure.aicc <= gen.aicc)) else gen
        fits.append(apply_filters(f, y, ctx.c_ext, flt, ctx.sigma))
    return fits


def regime_tramos(ctx: Context, seg: Segmentation, tramo: str, by_regime: bool = False, rises_only: bool = False) -> Segmentation:
    """Los tramos que se ajustan: el régimen completo ("TM", transitorio + meseta) o sólo el transitorio ("T").

    Con "TM" los fragmentos del mismo signo que siguen una misma exponencial se
    unen, los inicios que no cambian nada se descartan y cada tramo se extiende
    hasta el inicio del transitorio siguiente (`co2ach.regimes`). Con "T" los
    métodos que ya entregan regímenes (`by_regime`: PELT, oráculo) recortan cada
    uno al 95 % de su escalón; el resto ajusta lo que detecta.
    """
    if tramo == "TM":
        joiner = Joiner(ctx.series.t_h, ctx.c_clean, ctx.sigma)
        return as_regimes(seg, joiner.regimes(seg, by_regime=by_regime, rises_only=rises_only))
    if by_regime:
        return as_regimes(seg, transient_part(with_transient_end(seg.transients, seg.labels)))
    if rises_only:
        return as_regimes(seg, [s for s in seg.transients if s.phase == RISE])
    return seg


def _hyun(key: str, label: str, cfg: HyunConfig, decay_model: str = "generacion", tramo: str = "TM"):
    def run(ctx: Context) -> MethodResult:
        flt = Filters(cfg.min_duration_h, cfg.min_r2)
        rises_only = not cfg.decays  # Hyun fiel: las bajadas de K-means sólo delimitan sus subidas
        seg = segment_hyun(ctx.series.t_h, ctx.series.c_obs, replace(cfg, decays=True) if rises_only else cfg)
        seg = regime_tramos(ctx, seg, tramo, rises_only=rises_only)
        return MethodResult(key, label, seg, fit_hyun_family(ctx, seg, flt, decay_model), flt)

    return run


def _general(key: str, label: str, segmenter, tramo: str = "TM", by_regime: bool = False):
    def run(ctx: Context) -> MethodResult:
        flt = Filters()
        seg = regime_tramos(ctx, segmenter(ctx), tramo, by_regime=by_regime)
        return MethodResult(key, label, seg, fit_general(ctx, seg, flt), flt)

    return run


def build_methods(sigma_min: float = 0.0, tramo: str = "TM") -> dict:
    """Los métodos del estudio.

    `sigma_min` es el piso de ruido de los segmentadores que escalan sus
    umbrales con él. `tramo` es lo que se ajusta: el régimen completo ("TM",
    por defecto) o sólo el transitorio ("T").
    """
    n, k = sigma_min, tramo
    return {
        "M0": ("Oráculo", _general("M0", "Oráculo", lambda ctx: segment_oracle(ctx.series), k, by_regime=True)),
        "M0t": ("Oráculo (sólo el transitorio)", _general("M0t", "Oráculo (sólo el transitorio)", lambda ctx: segment_oracle(ctx.series, transient_only=True), "T")),
        "M1": ("Hyun fiel (configuración de escuela)", _hyun("M1", "Hyun fiel (configuración de escuela)", PRESETS["school"], tramo=k)),
        "M1th": ("Hyun fiel (configuración de casa de ensayo)", _hyun("M1th", "Hyun fiel (configuración de casa de ensayo)", PRESETS["test_house"], tramo=k)),
        "M2": ("Hyun + decaimiento con generación", _hyun("M2", "Hyun + decaimiento con generación", variant(decays=True), "generacion", tramo=k)),
        "M3a": ("Hyun + C″ como variable de K-means", _hyun("M3a", "Hyun + C″ como variable de K-means", variant(decays=True, d2_feature="valor", sigma_min=n), tramo=k)),
        "M3b": ("Hyun + corte en quiebres de C″", _hyun("M3b", "Hyun + corte en quiebres de C″", variant(decays=True, kink_split=True, sigma_min=n), tramo=k)),
        "M4": ("Derivada: umbral en C′ + quiebres", _general("M4", "Derivada: umbral en C′ + quiebres", lambda ctx: segment_derivative(ctx.series.t_h, ctx.series.c_obs, DerivConfig(sigma_min=n)), k)),
        "M5": ("PELT con costo exponencial", _general("M5", "PELT con costo exponencial", lambda ctx: segment_pelt(ctx.series.t_h, ctx.series.c_obs, PeltConfig(sigma_min=n)), k, by_regime=True)),
        "M6": ("HMM de 3 estados", _general("M6", "HMM de 3 estados", lambda ctx: segment_hmm(ctx.series.t_h, ctx.series.c_obs, HmmConfig(sigma_min=n)), k)),
    }


METHODS = build_methods()

# filas de los paneles de segmentación
FAMILY_HYUN = ["M1", "M2", "M3a", "M3b"]
FAMILY_ALT = ["M0", "M4", "M5", "M6"]


def run_method(key: str, series, ctx: Context | None = None) -> MethodResult:
    ctx = ctx or Context.build(series)
    return METHODS[key][1](ctx)


def run_all(series, keys=None, sigma_min: float = 0.0, tramo: str = "TM") -> dict:
    """Corre los métodos pedidos (todos por defecto) sobre una serie.

    `tramo`: "TM" ajusta el régimen completo (transitorio + meseta); "T", sólo
    el transitorio.
    """
    methods = METHODS if (sigma_min <= 0 and tramo == "TM") else build_methods(sigma_min, tramo)
    ctx = Context.build(series, sigma_min)
    return {key: methods[key][1](ctx) for key in (keys or methods)}
