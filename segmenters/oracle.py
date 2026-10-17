"""Oráculo: segmentos tomados de la verdad de la serie sintética.

Sirve de cota. Con los cortes perfectos, el error que queda es el límite de
identificabilidad del ajuste; lo que un método pierde frente al oráculo es
error de segmentación.

Si la ocupación cambia gradualmente (`ramp_min` > 0), el tramo empieza cuando
termina la rampa: antes de eso los parámetros todavía no son constantes y un
solo exponencial no describe la curva.
"""

from __future__ import annotations

import numpy as np

from .base import STEADY, Segment, Segmentation


def segment_oracle(series, transient_only: bool = False) -> Segmentation:
    """Regímenes verdaderos completos, o sólo su parte transitoria."""
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
            i0 += skip  # la rampa queda sin segmentar
        if transient_only:
            end = min(max(int(row.i_fin_transitorio), i0 + 2), i1)
            segments.append(Segment(i0, end, kind))
            if end < i1:
                segments.append(Segment(end, i1, STEADY))
        else:
            segments.append(Segment(i0, i1, kind))
    return Segmentation(segments=segments, labels=np.asarray(series.phase).copy(), info={})
