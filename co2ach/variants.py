"""Variantes de ajuste del cap. 5: se segmenta una vez por método y se cruzan

- el tramo: sólo el transitorio ("T") o con la meseta ("TM");
- la generación: libre, conocida (con un error relativo opcional) o acotada.

La familia de Hyun ajusta la serie filtrada con C0 fijo; el resto, la serie sin
picos con C0 libre.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluate import fits_table, onset_metrics, regime_estimates
from .fitting import fit_bounded_source, fit_exponential, fit_hyun, fit_known_source
from .pipeline import Context, Filters, MethodResult, apply_filters
from .regimes import Joiner, drop_false_onsets, transient_part, with_transient_end
from .segmenters.base import RISE, Segmentation
from .segmenters.derivative import DerivConfig, segment_derivative
from .segmenters.hmm import HmmConfig, segment_hmm
from .segmenters.hyun import segment_hyun, variant
from .segmenters.oracle import segment_oracle
from .segmenters.pelt_exp import PeltConfig, segment_pelt

METHOD_KEYS = ("M1", "M2", "M3a", "M3b", "M4", "M5", "M6", "M0")
HYUN_FAMILY = ("M1", "M2", "M3a", "M3b")
TRAMOS = ("T", "TM")
GENS = ("libre", "conocida")
# segmentación de la que sale cada método (M1 y M2 comparten la de Hyun)
SOURCE = {"M1": "hyun", "M2": "hyun", "M3a": "hyun_d2", "M3b": "hyun_kink", "M4": "deriv", "M5": "pelt", "M6": "hmm", "M0": "oracle"}
METHOD_LABELS = {
    "M1": "Hyun fiel",
    "M2": "Hyun + decaimientos",
    "M3a": "Hyun + C″ en K-means",
    "M3b": "Hyun + quiebres",
    "M4": "Derivada",
    "M5": "PELT",
    "M6": "HMM",
    "M0": "Oráculo",
}


@dataclass
class Bases:
    """Las segmentaciones de una serie, calculadas una vez."""

    series: object
    ctx: Context
    seg: dict
    joiner: Joiner = None

    def __post_init__(self):
        if self.joiner is None:
            self.joiner = Joiner(self.series.t_h, self.ctx.c_clean, self.ctx.sigma)


def build_bases(series, methods=METHOD_KEYS) -> Bases:
    t, c = series.t_h, series.c_obs
    builders = {
        "hyun": lambda: segment_hyun(t, c, variant(decays=True)),
        "hyun_d2": lambda: segment_hyun(t, c, variant(decays=True, d2_feature=True)),
        "hyun_kink": lambda: segment_hyun(t, c, variant(decays=True, kink_split=True)),
        "deriv": lambda: segment_derivative(t, c, DerivConfig()),
        "pelt": lambda: segment_pelt(t, c, PeltConfig()),
        "hmm": lambda: segment_hmm(t, c, HmmConfig()),
        "oracle": lambda: segment_oracle(series),
    }
    seg = {name: builders[name]() for name in dict.fromkeys(SOURCE[k] for k in methods)}
    return Bases(series=series, ctx=Context.build(series), seg=seg)


def segments_for(bases: Bases, key: str, tramo: str) -> list:
    """Tramos que ajusta el método `key` con el tramo "T" o "TM"."""
    src = bases.seg[SOURCE[key]]
    if key in ("M5", "M0"):  # ya dan regímenes: para "T" se recortan
        labels = src.labels if key == "M5" else bases.series.phase
        regimes = with_transient_end(src.transients, labels)
        return regimes if tramo == "TM" else transient_part(regimes)
    segs = list(src.transients)
    if tramo == "TM":
        joiner = bases.joiner
        blocked = np.r_[np.asarray(src.info.get("kinks", []), dtype=int), np.asarray(src.info.get("breaks", []), dtype=int)]
        segs = drop_false_onsets(segs, bases.series.n, joiner.is_flat, same_regime=joiner.same_regime, no_merge_at=blocked)
    if key == "M1":  # Hyun fiel ajusta sólo subidas
        segs = [s for s in segs if s.phase == RISE]
    return segs


def known_source(series, i0: int, i1: int) -> float:
    """Fuente nominal [ppm/h] del régimen dominante del tramo."""
    dom = int(np.bincount(series.regime_id[i0:i1]).argmax())
    return float(series.regimes["fuente_ppm_h"].to_numpy()[dom])


def fit_variant(bases: Bases, key: str, segs, gen: str = "libre", g_error: float = 0.0, flt: Filters | None = None, g_bounds: tuple | None = None) -> list:
    """Ajusta los tramos con la generación "libre", "conocida" (con error `g_error`) o "acotada".

    Acotada: S entre g_bounds[0] y g_bounds[1] veces la fuente nominal. Con el
    aula vacía (fuente cero) queda como la conocida.
    """
    ctx, series = bases.ctx, bases.series
    t = series.t_h
    hyun = key in HYUN_FAMILY
    data = bases.seg[SOURCE[key]].info["c_filt"] if hyun else ctx.c_clean
    flt = flt or Filters()
    fits = []
    for seg in segs:
        sl = slice(seg.i0, seg.i1)
        y = data[sl]
        if gen == "libre":
            if hyun:
                f = fit_hyun(t[sl], y, c_ext=ctx.c_ext, c0=y[0], i0=seg.i0, model="Hyun (E, λ)", phase=seg.phase)
            else:
                f = fit_exponential(t[sl], y, c_ext=ctx.c_ext, css_min=ctx.c_ext, i0=seg.i0, model="general", phase=seg.phase)
        elif gen == "acotada" and known_source(series, seg.i0, seg.i1) > 0:
            s_nom = known_source(series, seg.i0, seg.i1)
            f = fit_bounded_source(t[sl], y, c_ext=ctx.c_ext, s_min=s_nom * g_bounds[0], s_max=s_nom * g_bounds[1], c0=(y[0] if hyun else None), i0=seg.i0, phase=seg.phase)
        else:
            s_known = known_source(series, seg.i0, seg.i1) * (1.0 + g_error)
            f = fit_known_source(t[sl], y, c_ext=ctx.c_ext, s=s_known, c0=(y[0] if hyun else None), i0=seg.i0, phase=seg.phase)
        fits.append(apply_filters(f, y if hyun else ctx.c_med[sl], ctx.c_ext, flt, ctx.sigma))
    return fits


def evaluate_variant(series, key: str, tramo: str, gen: str, g_error: float, segs, fits):
    """(tabla por régimen, tabla de ajustes, fila resumen) de una variante."""
    result = MethodResult(key, METHOD_LABELS[key], Segmentation(list(segs), np.zeros(series.n, dtype=int)), list(fits))
    tb = fits_table(series, result)
    reg = regime_estimates(series, tb)
    info = series.regimes[["regimen", "tipo", "n_tau", "personas", "escalon_ppm"]]
    reg = reg.merge(info, on="regimen", how="left")
    meta = {"metodo": key, "tramo": tramo, "gen": gen, "g_error": g_error}
    for name, value in meta.items():
        reg[name] = value
        tb[name] = value
    err = reg["error_rel"].to_numpy(dtype=float)
    ok = np.isfinite(err)
    onsets = onset_metrics(series, result.segmentation)
    run = dict(
        meta,
        tramos=len(segs),
        ajustes_aceptados=int(tb["aceptado"].sum()),
        regimenes=len(reg),
        cubiertos=int(ok.sum()),
        # éxito: régimen con ACH a menos de 20 %; sin estimación cuenta como fallo
        exito20=float(np.mean(ok & (np.abs(np.nan_to_num(err, nan=np.inf)) <= 0.20))) if len(err) else np.nan,
        err_med=float(np.median(np.abs(err[ok]))) if ok.any() else np.nan,
        sesgo_med=float(np.median(err[ok])) if ok.any() else np.nan,
        f1=onsets["f1"],
        precision=onsets["precision"],
        recall=onsets["recall"],
        error_ubicacion_min=onsets["error_ubicacion_min"],
    )
    return reg, tb, run


def run_series_variants(series, g_errors=(0.0,), methods=METHOD_KEYS, bases: Bases | None = None) -> dict:
    """Todas las variantes sobre una serie: tablas `regimenes`, `ajustes` y `corridas`.

    `g_errors`: errores relativos de la generación supuesta que se prueban con "conocida".
    """
    bases = bases or build_bases(series, methods=methods)
    regs, tabs, runs = [], [], []
    for key in methods:
        for tramo in TRAMOS:
            segs = segments_for(bases, key, tramo)
            for gen in GENS:
                for g_error in (0.0,) if gen == "libre" else tuple(g_errors):
                    fits = fit_variant(bases, key, segs, gen, g_error)
                    reg, tb, run = evaluate_variant(series, key, tramo, gen, g_error, segs, fits)
                    regs.append(reg)
                    runs.append(run)
                    if g_error == 0.0:
                        tabs.append(tb)
    return {
        "regimenes": pd.concat(regs, ignore_index=True),
        "ajustes": pd.concat(tabs, ignore_index=True),
        "corridas": pd.DataFrame(runs),
        "bases": bases,
    }
