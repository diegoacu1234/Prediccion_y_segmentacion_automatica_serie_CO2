"""De fases a regímenes: el tramo va de un inicio de transitorio al siguiente, con la meseta (sección 4.3.4).

Antes de extender los tramos se arreglan dos cosas de la segmentación por fases:

- fragmentos del mismo signo que sigue una sola exponencial: se unen, con la
  misma penalización que PELT;
- inicios falsos: un régimen que casi no se mueve (recorrido < 4σ̂) se
  descarta y el anterior sigue de largo.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .fitting import fit_exponential
from .segmenters.base import RISE, Segment, Segmentation


@dataclass
class Joiner:
    """Los dos criterios de arriba sobre una serie. Guarda los ajustes para no repetirlos."""

    t_h: np.ndarray
    c_clean: np.ndarray
    sigma: float
    pen_factor: float = 4.0  # igual que PELT
    flat_z: float = 4.0  # recorrido < flat_z·σ̂ -> meseta
    min_fit_samples: int = 6
    _fits: dict = field(default_factory=dict)

    def fit(self, i0: int, i1: int):
        """Ajuste general de [i0, i1)."""
        key = (i0, i1)
        if key not in self._fits:
            self._fits[key] = fit_exponential(self.t_h[i0:i1], self.c_clean[i0:i1], c_ext=0.0, i0=i0) if i1 - i0 >= self.min_fit_samples else None
        return self._fits[key]

    def sse(self, i0: int, i1: int) -> float:
        f = self.fit(i0, i1)
        return f.sse if f is not None and f.ok else 0.0

    def same_regime(self, a0: int, b0: int, b1: int) -> bool:
        """¿[b0, b1) sigue la exponencial del régimen que empezó en a0?"""
        pen = self.pen_factor * self.sigma**2 * np.log(len(self.t_h))
        return self.sse(a0, b1) - self.sse(a0, b0) - self.sse(b0, b1) < pen

    def is_flat(self, i0: int, i1: int) -> bool:
        f = self.fit(i0, i1)
        if f is None or not f.ok:
            return True
        curve = f.predict(self.t_h[[i0, i1 - 1]])
        return abs(curve[1] - curve[0]) < self.flat_z * self.sigma

    def regimes(self, seg: Segmentation, by_regime: bool = False, rises_only: bool = False) -> list:
        """Regímenes a partir de la segmentación de un método.

        by_regime: el método ya da regímenes (PELT, oráculo). rises_only: las bajadas sólo delimitan (Hyun fiel).
        """
        if by_regime:
            return with_transient_end(seg.transients, seg.labels)
        blocked = np.r_[np.asarray(seg.info.get("kinks", []), dtype=int), np.asarray(seg.info.get("breaks", []), dtype=int)]
        out = drop_false_onsets(list(seg.transients), len(self.t_h), self.is_flat, same_regime=self.same_regime, no_merge_at=blocked)
        return [s for s in out if s.phase == RISE] if rises_only else out


def as_regimes(seg: Segmentation, regimes: list) -> Segmentation:
    """La misma segmentación con `regimes` como tramos."""
    labels = np.zeros_like(np.asarray(seg.labels))
    for s in regimes:
        labels[s.i0 : s.i1] = s.phase
    return Segmentation(segments=list(regimes), labels=labels, info=dict(seg.info, labels_fase=seg.labels))


def to_regimes(transients, n: int, same_regime=None, merge_gap: int | None = None, no_merge_at=()) -> list:
    """Transitorios -> regímenes.

    1. une fragmentos del mismo signo separados por un hueco, si `same_regime`
       lo dice (o si el hueco es de hasta `merge_gap`). No une tramos pegados
       (son cortes en un quiebre) ni a través de `no_merge_at`;
    2. estira cada tramo hasta el inicio del siguiente. `i_trans` guarda dónde
       terminaba el transitorio.
    """
    segs = sorted((s for s in transients if s.phase != 0), key=lambda s: s.i0)
    blocked = np.asarray(sorted(no_merge_at), dtype=int)
    merged = []
    for s in segs:
        if merged:
            prev = merged[-1]
            gap = s.i0 - prev.i1
            crosses = bool(np.any((blocked >= prev.i1) & (blocked <= s.i0))) if len(blocked) else False
            if s.phase == prev.phase and gap > 0 and not crosses:
                join = same_regime(prev.i0, s.i0, s.i1) if same_regime is not None else (merge_gap is not None and gap <= merge_gap)
                if join:
                    merged[-1] = replace(prev, i1=s.i1)
                    continue
        merged.append(replace(s))
    out = []
    for k, s in enumerate(merged):
        end = merged[k + 1].i0 if k + 1 < len(merged) else n
        out.append(Segment(s.i0, end, s.phase, note="regimen", i_trans=s.i1))
    return out


def drop_false_onsets(transients, n: int, is_flat, same_regime=None, merge_gap: int | None = None, no_merge_at=()) -> list:
    """Como `to_regimes`, sacando los inicios cuyo régimen es plano (`is_flat`)."""
    kw = dict(same_regime=same_regime, merge_gap=merge_gap, no_merge_at=no_merge_at)
    regimes = to_regimes(transients, n, **kw)
    flat = [r for r in regimes if is_flat(r.i0, r.i1)]
    if not flat:
        return regimes
    kept = [s for s in transients if s.phase != 0 and not any(r.i0 <= s.i0 < r.i_trans for r in flat)]
    return [r for r in to_regimes(kept, n, **kw) if not is_flat(r.i0, r.i1)]


def with_transient_end(segments, labels) -> list:
    """Marca en cada régimen dónde termina el transitorio, según las etiquetas."""
    labels = np.asarray(labels)
    out = []
    for s in segments:
        if s.phase == 0:
            continue
        moving = labels[s.i0 : s.i1] != 0
        stop = int(np.argmin(moving)) if not moving.all() else len(moving)  # primera muestra de meseta
        out.append(replace(s, i_trans=min(s.i0 + max(stop, 2), s.i1)))
    return out


def transient_part(segments) -> list:
    """Recorta cada régimen a su transitorio (el contraste "T" del cap. 5)."""
    return [Segment(s.i0, min(s.i_trans or s.i1, s.i1), s.phase, note="transitorio") for s in segments if s.phase != 0]
