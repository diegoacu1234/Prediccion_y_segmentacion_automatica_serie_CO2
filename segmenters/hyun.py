"""Segmentación de Hyun (2025): mediana + K-means + DBSCAN.

Pipeline de la tesis (Fig. 2.1):

1. filtro de mediana;
2. K-means con k = 3 sobre dos features estandarizados, el signo del gradiente y
   la posición relativa alto-bajo  RHL(t) = (C(t) − min) / (max − min)  en la
   ventana [t−w, t]  (ec. 1);
3. DBSCAN sobre la suma acumulada de la bandera "no pertenece al cluster de
   subida": los puntos con la misma suma forman un sub-cluster, y `min_samples`
   descarta los cortos.

Supuestos donde la tesis no alcanza (todos son parámetros):

- el gradiente es `np.gradient` de la serie filtrada;
- si max = min en la ventana, RHL vale 0,5;
- el rol de cada cluster sale de su signo medio: el mayor es subida, el menor
  bajada y el restante meseta; si hay empate, desempata el RHL medio;
- DBSCAN usa sólo la suma acumulada como feature, con `eps = 0.5`. Así es una
  codificación por rachas con largo mínimo (ver tests);
- las ventanas están en muestras, para datos cada 30 s como en la tesis.

Variantes propias: usar también el cluster de bajada (`decays`), agregar la
derivada segunda como feature (`d2_feature`) y cortar los sub-clusters en los
quiebres detectados con la derivada segunda (`kink_split`).
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
    preset: str = "school"
    median_window: int = 25
    rhl_window: int = 5
    min_samples: int = 10
    eps: float = 0.5
    n_clusters: int = 3
    min_duration_h: float = 0.25  # filtros de la Tabla 2.5
    min_r2: float = 0.1
    seed: int = 0
    decays: bool = False  # sub-clusters también para el cluster de bajada
    d2_feature: str = "no"  # "no" | "valor" | "signo"
    d2_window_min: float = 20.0
    kink_split: bool = False
    kink_half_min: float = 10.0
    kink_z: float = 5.0
    sigma_min: float = 0.0  # piso del ruido [ppm]: con datos reales, la fluctuación de la señal que el método debe ignorar (sólo afecta a las variantes con C″)


PRESETS = {
    "school": HyunConfig(preset="school", median_window=25, rhl_window=5, min_samples=10, min_duration_h=0.25, min_r2=0.1),
    "test_house": HyunConfig(preset="test_house", median_window=25, rhl_window=3, min_samples=10, min_duration_h=1.25, min_r2=0.5),
    "multizone": HyunConfig(preset="multizone", median_window=5, rhl_window=3, min_samples=30, min_duration_h=1.0, min_r2=0.85),
}


def hyun_features(c_filt, rhl_window: int):
    """Signo del gradiente y posición relativa alto-bajo de la serie filtrada."""
    sign = np.sign(np.gradient(c_filt))
    lo, hi = trailing_minmax(c_filt, rhl_window)
    span = hi - lo
    rhl = np.where(span > 0, (c_filt - lo) / np.where(span > 0, span, 1.0), 0.5)
    return sign, rhl


def cluster_roles(cluster, sign, rhl=None) -> dict:
    """Asigna a cada cluster una fase según el signo medio del gradiente.

    Si dos clusters empatan en signo medio desempata el RHL medio: en una subida
    vale cerca de 1 y en una bajada cerca de 0. El empate aparece en series sin
    mesetas, donde K-means reparte una misma fase en dos clusters puros.
    """
    ids = np.unique(cluster)
    mean_sign = {int(c): float(sign[cluster == c].mean()) for c in ids}
    mean_rhl = {int(c): float(rhl[cluster == c].mean()) if rhl is not None else 0.0 for c in ids}
    order = sorted(mean_sign, key=lambda c: (mean_sign[c], mean_rhl[c]))
    roles = {c: STEADY for c in order}
    roles[order[-1]] = RISE
    roles[order[0]] = DECAY
    return roles


def subclusters(phase_km, target: int, min_samples: int, eps: float = 0.5) -> list:
    """Sub-clusters de Hyun para la fase `target`: DBSCAN sobre la suma acumulada.

    Devuelve una lista de (i0, i1) ordenada.
    """
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


def subclusters_rle(phase_km, target: int, min_samples: int) -> list:
    """Lo mismo por rachas, para comprobar la equivalencia con DBSCAN (eps < 1)."""
    from ..signal_utils import runs

    return [(i0, i1) for i0, i1, v in runs(np.asarray(phase_km)) if v == target and i1 - i0 >= min_samples]


def segment_hyun(t_h, c_obs, cfg: HyunConfig | None = None) -> Segmentation:
    cfg = cfg or PRESETS["school"]
    t_h = np.asarray(t_h, dtype=float)
    c_obs = np.asarray(c_obs, dtype=float)
    dt_h = float(t_h[1] - t_h[0])

    c_filt = median_filter(c_obs, cfg.median_window)
    sign, rhl = hyun_features(c_filt, cfg.rhl_window)
    cols = [sign, rhl]

    sigma = max(noise_sd(c_obs), cfg.sigma_min)
    c_clean = None
    d2z = None
    if cfg.d2_feature != "no" or cfg.kink_split:
        c_clean, _ = despike(c_obs, sigma)
    if cfg.d2_feature != "no":
        win = n_samples(adaptive_minutes(cfg.d2_window_min, sigma), dt_h, odd=True, minimum=7)
        d2 = sg_derivative(c_clean, dt_h, win, 3, 2)
        d2z = d2 / (sigma * sg_gain(win, 3, 2, dt_h))
        cols.append(np.sign(d2z) * (np.abs(d2z) > 3.0) if cfg.d2_feature == "signo" else np.clip(d2z, -10.0, 10.0))

    x = StandardScaler().fit_transform(np.column_stack(cols))
    km = KMeans(n_clusters=cfg.n_clusters, n_init=10, random_state=cfg.seed).fit(x)
    cluster = km.labels_
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

    nz = sign != 0
    info = {
        "c_filt": c_filt,
        "sign": sign,
        "rhl": rhl,
        "d2z": d2z,
        "cluster": cluster,
        "roles": roles,
        "phase_km": phase_km,
        "cumsum_rise": np.cumsum(phase_km != RISE),
        "kinks": kinks,
        "sigma": sigma,
        "agreement_sign": float(np.mean(phase_km == sign)),
        "agreement_sign_nonzero": float(np.mean(phase_km[nz] == sign[nz])) if nz.any() else np.nan,
        "config": cfg,
    }
    return Segmentation(segments=segments, labels=labels, info=info)


def variant(base: str = "school", **changes) -> HyunConfig:
    """Preset publicado con cambios, por ejemplo `variant(decays=True)`."""
    return replace(PRESETS[base], **changes)
