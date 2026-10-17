"""Tipos comunes a todos los segmentadores."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..signal_utils import runs
from ..synth import DECAY, RISE, STEADY  # noqa: F401  (se reexportan)


@dataclass
class Segment:
    """Tramo [i0, i1) con una fase: +1 subida, 0 meseta, −1 bajada.

    `i_trans` marca dónde termina la parte transitoria cuando el tramo incluye
    también su meseta (ver `co2ach.regimes`).
    """

    i0: int
    i1: int
    phase: int
    note: str = ""
    i_trans: int | None = None

    @property
    def n(self) -> int:
        return self.i1 - self.i0


@dataclass
class Segmentation:
    segments: list
    labels: np.ndarray  # fase final por muestra
    info: dict = field(default_factory=dict)

    @property
    def transients(self) -> list:
        return [s for s in self.segments if s.phase != 0]


def segments_from_labels(labels, breaks=()) -> list:
    """Convierte etiquetas por muestra en segmentos, cortando además en `breaks`."""
    cuts = sorted(set(int(b) for b in breaks))
    out = []
    for i0, i1, v in runs(labels):
        inner = [i0] + [b for b in cuts if i0 < b < i1] + [i1]
        for a, b in zip(inner[:-1], inner[1:]):
            out.append(Segment(a, b, int(v)))
    return out


def adaptive_minutes(base_min: float, sigma: float, sigma_ref: float = 5.0, lo: float = 0.5, hi: float = 3.0) -> float:
    """Escala una ventana con el ruido para mantener la misma sensibilidad.

    El desvío de una pendiente estimada en una ventana W va como σ·W^(−3/2), así
    que W ∝ σ^(2/3) conserva la pendiente mínima detectable.
    """
    return base_min * float(np.clip((sigma / sigma_ref) ** (2.0 / 3.0), lo, hi))
