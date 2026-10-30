"""Variantes de tramo de la figura 4.7: sólo el transitorio ("T") o con la meseta ("TM").

Se segmenta una vez por método. La familia de Hyun ajusta la serie filtrada con
C0 fijo; el resto, la serie sin picos con C0 libre.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fitting import fit_exponential, fit_hyun
from .pipeline import Context, Filters, apply_filters
from .regimes import Joiner, drop_false_onsets, transient_part, with_transient_end
from .segmenters.base import RISE
from .segmenters.derivative import DerivConfig, segment_derivative
from .segmenters.hmm import HmmConfig, segment_hmm
from .segmenters.hyun import segment_hyun, variant
from .segmenters.oracle import segment_oracle
from .segmenters.pelt_exp import PeltConfig, segment_pelt

METHOD_KEYS = ("M1", "M2", "M3a", "M3b", "M4", "M5", "M6", "M0")
HYUN_FAMILY = ("M1", "M2", "M3a", "M3b")
# segmentación de la que sale cada método (M1 y M2 comparten la de Hyun)
SOURCE = {"M1": "hyun", "M2": "hyun", "M3a": "hyun_d2", "M3b": "hyun_kink", "M4": "deriv", "M5": "pelt", "M6": "hmm", "M0": "oracle"}


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


def fit_variant(bases: Bases, key: str, segs, flt: Filters | None = None) -> list:
    """Ajusta los tramos con la generación libre."""
    ctx, series = bases.ctx, bases.series
    t = series.t_h
    hyun = key in HYUN_FAMILY
    data = bases.seg[SOURCE[key]].info["c_filt"] if hyun else ctx.c_clean
    flt = flt or Filters()
    fits = []
    for seg in segs:
        sl = slice(seg.i0, seg.i1)
        y = data[sl]
        if hyun:
            f = fit_hyun(t[sl], y, c_ext=ctx.c_ext, c0=y[0], i0=seg.i0, model="Hyun (E, λ)", phase=seg.phase)
        else:
            f = fit_exponential(t[sl], y, c_ext=ctx.c_ext, css_min=ctx.c_ext, i0=seg.i0, model="general", phase=seg.phase)
        fits.append(apply_filters(f, y if hyun else ctx.c_med[sl], ctx.c_ext, flt, ctx.sigma))
    return fits
