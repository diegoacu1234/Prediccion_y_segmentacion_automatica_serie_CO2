"""Método de Hyun (2025): mediana móvil + K-means + DBSCAN (sección 4.4.1, apéndice B.5).

1. mediana móvil;
2. K-means (k = 3) sobre el signo de la pendiente y la posición relativa
   RHL = (C − min) / (max − min) en [t−w, t];
3. DBSCAN sobre la suma acumulada de "no es subida": cada racha de al menos
   `min_samples` muestras es un evento.

Lo que la descripción publicada no especifica: gradiente con `np.gradient`,
RHL = 0,5 si la ventana es plana, el grupo con signo medio mayor es subida (el
menor, bajada) y DBSCAN con eps = 0,5 sobre la suma acumulada.

Variantes de la tesis: `decays` (separa también el grupo de bajada),
`d2_feature` (C″ como tercera variable) y `kink_split` (corta en los quiebres).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from ..signal_utils import despike, find_kinks, median_filter, n_samples, noise_sd, sg_derivative, sg_gain, trailing_minmax
from .base import DECAY, RISE, STEADY, Segment, Segmentation, adaptive_minutes


@dataclass(frozen=True)
class HyunConfig:
    # configuración que Hyun publica para escuelas (tabla B.1), en muestras de 30 s
    median_window: int = 25
    rhl_window: int = 5
    min_samples: int = 10
    eps: float = 0.5
    n_clusters: int = 3
    min_duration_h: float = 0.25
    min_r2: float = 0.1
    seed: int = 0
    # variantes
    decays: bool = False  # separa en eventos también el grupo de bajada
    d2_feature: bool = False  # C″ como tercera variable de K-means
    d2_window_min: float = 20.0
    kink_split: bool = False  # corta los eventos en los quiebres
    kink_half_min: float = 10.0
    kink_z: float = 5.0
    sigma_min: float = 0.0  # piso del ruido [ppm]; sólo lo usan las variantes con C″


SCHOOL = HyunConfig()


def hyun_features(c_filt, rhl_window: int):
    """Signo del gradiente y RHL de la serie filtrada."""
    sign = np.sign(np.gradient(c_filt))
    lo, hi = trailing_minmax(c_filt, rhl_window)
    span = hi - lo
    rhl = np.where(span > 0, (c_filt - lo) / np.where(span > 0, span, 1.0), 0.5)
    return sign, rhl


def cluster_roles(cluster, sign, rhl=None) -> dict:
    """Fase de cada grupo según su signo medio; si dos empatan, desempata el RHL medio."""
    ids = np.unique(cluster)
    mean_sign = {int(c): float(sign[cluster == c].mean()) for c in ids}
    mean_rhl = {int(c): float(rhl[cluster == c].mean()) if rhl is not None else 0.0 for c in ids}
    order = sorted(mean_sign, key=lambda c: (mean_sign[c], mean_rhl[c]))
    roles = {c: STEADY for c in order}
    roles[order[-1]] = RISE
    roles[order[0]] = DECAY
    return roles


def subclusters(phase_km, target: int, min_samples: int, eps: float = 0.5) -> list:
    """Eventos de la fase `target`: DBSCAN sobre la suma acumulada. Devuelve [(i0, i1), ...]."""
    phase_km = np.asarray(phase_km)
    idx = np.flatnonzero(phase_km == target)
    if len(idx) == 0:
        return []
    feature = np.cumsum(phase_km != target)[idx].astype(float).reshape(-1, 1)
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit(feature).labels_
    out = []
    for c in np.unique(lab[lab >= 0]):
        members = idx[lab == c]
        out.append((int(members.min()), int(members.max()) + 1))
    return sorted(out)


def segment_hyun(t_h, c_obs, cfg: HyunConfig | None = None) -> Segmentation:
    cfg = cfg or SCHOOL
    t_h = np.asarray(t_h, dtype=float)
    c_obs = np.asarray(c_obs, dtype=float)
    dt_h = float(t_h[1] - t_h[0])

    c_filt = median_filter(c_obs, cfg.median_window)
    sign, rhl = hyun_features(c_filt, cfg.rhl_window)
    cols = [sign, rhl]

    sigma = max(noise_sd(c_obs), cfg.sigma_min)
    c_clean = None
    if cfg.d2_feature or cfg.kink_split:
        c_clean, _ = despike(c_obs, sigma)
    if cfg.d2_feature:
        win = n_samples(adaptive_minutes(cfg.d2_window_min, sigma), dt_h, odd=True, minimum=7)
        d2 = sg_derivative(c_clean, dt_h, win, 3, 2)
        d2z = d2 / (sigma * sg_gain(win, 3, 2, dt_h))
        cols.append(np.clip(d2z, -10.0, 10.0))

    x = StandardScaler().fit_transform(np.column_stack(cols))
    cluster = KMeans(n_clusters=cfg.n_clusters, n_init=10, random_state=cfg.seed).fit(x).labels_
    roles = cluster_roles(cluster, sign, rhl)
    phase_km = np.vectorize(roles.get)(cluster)

    targets = [RISE] + ([DECAY] if cfg.decays else [])
    pieces = []
    for target in targets:
        for i0, i1 in subclusters(phase_km, target, cfg.min_samples, cfg.eps):
            pieces.append((i0, i1, target))

    kinks = np.array([], dtype=int)
    if cfg.kink_split:
        half = n_samples(adaptive_minutes(cfg.kink_half_min, sigma), dt_h)
        kinks, _ = find_kinks(c_clean, dt_h, sigma, half, cfg.kink_z)
        cut = []
        for i0, i1, target in pieces:
            inner = [i0] + [int(k) for k in kinks if i0 < k < i1] + [i1]
            cut += [(a, b, target) for a, b in zip(inner[:-1], inner[1:]) if b - a >= cfg.min_samples]
        pieces = cut

    pieces.sort()
    labels = np.zeros(len(t_h), dtype=int)
    segments = []
    for i0, i1, target in pieces:
        labels[i0:i1] = target
        segments.append(Segment(i0, i1, target))

    info = {"c_filt": c_filt, "phase_km": phase_km, "kinks": kinks, "sigma": sigma, "config": cfg}
    return Segmentation(segments=segments, labels=labels, info=info)


def variant(**changes) -> HyunConfig:
    """La configuración de escuela con cambios, p. ej. `variant(decays=True)`."""
    return replace(SCHOOL, **changes)
