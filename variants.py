"""Variantes de ajuste para los experimentos.

Para cada método la segmentación se calcula una sola vez; lo que cambia es

- el **tramo** que se ajusta: sólo el transitorio ("T") o el transitorio con
  su meseta ("TM"), es decir, la meseta tomada como asíntota del transitorio.
  Para armar esos regímenes se unen los fragmentos que siguen una misma
  exponencial y se descartan los inicios falsos (ver `co2ach.regimes`);
- la **generación**: libre (C∞ y λ libres) o conocida (C∞ = C_ext + S/λ, sólo
  queda libre λ), con un posible error relativo en el valor supuesto de S.

En la familia Hyun el ajuste mantiene C0 fijo en la primera muestra filtrada y
usa la serie filtrada; en el resto C0 es libre y se usan los datos sin outliers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """Segmentaciones de una serie, calculadas una vez."""

    series: object
    ctx: Context
    seg: dict
    joiner: Joiner = None  # une fragmentos y descarta inicios falsos (`co2ach.regimes`)

    def __post_init__(self):
        if self.joiner is None:
            self.joiner = Joiner(self.series.t_h, self.ctx.c_clean, self.ctx.sigma)

    def fit(self, i0: int, i1: int):
        return self.joiner.fit(i0, i1)

    def same_regime(self, a0: int, b0: int, b1: int) -> bool:
        return self.joiner.same_regime(a0, b0, b1)

    def is_flat(self, i0: int, i1: int) -> bool:
        return self.joiner.is_flat(i0, i1)


def build_bases(series, methods=METHOD_KEYS) -> Bases:
    t, c = series.t_h, series.c_obs
    builders = {
        "hyun": lambda: segment_hyun(t, c, variant(decays=True)),
        "hyun_d2": lambda: segment_hyun(t, c, variant(decays=True, d2_feature="valor")),
        "hyun_kink": lambda: segment_hyun(t, c, variant(decays=True, kink_split=True)),
        "deriv": lambda: segment_derivative(t, c, DerivConfig()),
        "pelt": lambda: segment_pelt(t, c, PeltConfig()),
        "hmm": lambda: segment_hmm(t, c, HmmConfig()),
        "oracle": lambda: segment_oracle(series),
    }
    seg = {name: builders[name]() for name in dict.fromkeys(SOURCE[k] for k in methods)}
    return Bases(series=series, ctx=Context.build(series), seg=seg)


def segments_for(bases: Bases, key: str, tramo: str) -> list:
    """Tramos que ajusta el método `key`: transitorios ("T") o regímenes con meseta ("TM")."""
    src = bases.seg[SOURCE[key]]
    if key in ("M5", "M0"):  # ya segmentan por régimen: el transitorio se recorta
        labels = src.labels if key == "M5" else bases.series.phase
        regimes = with_transient_end(src.transients, labels)
        return regimes if tramo == "TM" else transient_part(regimes)
    segs = list(src.transients)
    if tramo == "TM":
        blocked = np.r_[np.asarray(src.info.get("kinks", []), dtype=int), np.asarray(src.info.get("breaks", []), dtype=int)]
        segs = drop_false_onsets(segs, bases.series.n, bases.is_flat, same_regime=bases.same_regime, no_merge_at=blocked)
    if key == "M1":  # el método fiel sólo ajusta subidas; las bajadas sólo delimitan
        segs = [s for s in segs if s.phase == RISE]
    return segs


def known_source(series, i0: int, i1: int) -> float:
    """Fuente nominal [ppm/h] del régimen dominante del tramo: "conozco cuánta gente hay"."""
    dom = int(np.bincount(series.regime_id[i0:i1]).argmax())
    return float(series.regimes["fuente_ppm_h"].to_numpy()[dom])


def fit_variant(bases: Bases, key: str, segs, gen: str = "libre", g_error: float = 0.0, flt: Filters | None = None, g_bounds: tuple | None = None) -> list:
    """Ajusta los tramos con la generación libre, conocida (con error relativo `g_error`) o acotada.

    Con `gen="acotada"`, S queda libre entre `g_bounds[0]` y `g_bounds[1]` veces
    la fuente nominal del tramo (`fitting.fit_bounded_source`). Si la fuente
    nominal es cero (sala vacía), las dos cotas coinciden y el ajuste es el de
    generación conocida.
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
    """Devuelve (tabla por régimen, tabla de ajustes, fila resumen) de una variante."""
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
    kind = series.regimes["tipo"].to_numpy()[tb["regimen"].to_numpy(dtype=int)] if len(tb) else np.array([], dtype=int)
    accepted = tb["aceptado"].to_numpy(dtype=bool)
    run = dict(
        meta,
        tramos=len(segs),
        ajustes_aceptados=int(accepted.sum()),
        # aceptados donde no hay evento (régimen sin transitorio) o que mezclan dos regímenes
        aceptados_espurios=int(np.sum(accepted & (kind == 0))),
        aceptados_mezclados=int(np.sum(accepted & (kind != 0) & (tb["pureza"].to_numpy(dtype=float) < 0.8))),
        regimenes=len(reg),
        cubiertos=int(ok.sum()),
        exito20=float(np.mean(ok & (np.abs(np.nan_to_num(err, nan=np.inf)) <= 0.20))) if len(err) else np.nan,
        exito10=float(np.mean(ok & (np.abs(np.nan_to_num(err, nan=np.inf)) <= 0.10))) if len(err) else np.nan,
        err_med=float(np.median(np.abs(err[ok]))) if ok.any() else np.nan,
        sesgo_med=float(np.median(err[ok])) if ok.any() else np.nan,
        f1=onsets["f1"],
        precision=onsets["precision"],
        recall=onsets["recall"],
        error_ubicacion_min=onsets["error_ubicacion_min"],
    )
    return reg, tb, run


def run_series_variants(series, g_errors=(0.0,), methods=METHOD_KEYS, bases: Bases | None = None) -> dict:
    """Todas las variantes sobre una serie. Devuelve tablas `regimenes`, `ajustes` y `corridas`.

    `g_errors` son los errores relativos en la generación supuesta que se
    prueban cuando la generación es conocida (0 = valor exacto).
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
