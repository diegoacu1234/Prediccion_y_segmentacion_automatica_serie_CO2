"""Oráculo: los regímenes verdaderos de la serie sintética. Lo que un método pierde frente a él es error de segmentación.

Si la ocupación cambia de a poco (`ramp_min` > 0), el tramo arranca al terminar la rampa.
"""

from __future__ import annotations

import numpy as np

from .base import STEADY, Segment, Segmentation


def segment_oracle(series) -> Segmentation:
    segments = []
    ramp = int(round(series.scenario.ramp_min / 60.0 / series.dt_h))
    people_prev = None
    for row in series.regimes.itertuples():
        i0, i1, kind = int(row.i0), int(row.i1), int(row.tipo)
        skip = ramp if (people_prev is not None and row.personas != people_prev) else 0
        people_prev = row.personas
        if kind == STEADY:
            segments.append(Segment(i0, i1, STEADY))
            continue
        if skip and i1 - i0 > skip + 4:
            i0 += skip  # la rampa queda afuera
        segments.append(Segment(i0, i1, kind))
    return Segmentation(segments=segments, labels=np.asarray(series.phase).copy(), info={})
