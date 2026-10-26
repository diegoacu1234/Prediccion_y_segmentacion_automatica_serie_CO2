"""Segmentación por la derivada (sección 4.4.3, apéndice B.6).

1. z = C′/σ_C′ con Savitzky–Golay;
2. umbral con histéresis: empieza con |z| > 4 y termina con |z| < 2;
3. limpieza de mesetas y transitorios cortos;
4. cada borde va al quiebre más cercano y un quiebre adentro de un transitorio lo parte.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..signal_utils import (
    close_gaps,
    despike,
    drop_short,
    find_kinks,
    n_samples,
    noise_sd,
    runs,
    sg_derivative,
    sg_gain,
    slope_jump,
    snap_boundaries,
    split_short_gaps,
)
from .base import Segmentation, adaptive_minutes, segments_from_labels


@dataclass(frozen=True)
class DerivConfig:
    # duraciones para σ = 5 ppm; con `adaptive` se escalan por (σ/5)^(2/3)
    win_min: float = 10.0  # ventana de C′
    poly: int = 2
    z_enter: float = 4.0
    z_exit: float = 2.0
    min_dur_min: float = 5.0
    gap_min: float = 4.0
    use_kinks: bool = True
    kink_half_min: float = 10.0
    kink_z: float = 5.0
    snap_min: float = 6.0  # más que media ventana: el umbral ve el cambio hasta media ventana antes
    adaptive: bool = True
    sigma_min: float = 0.0  # piso del ruido [ppm]


def hysteresis_labels(z, z_enter: float, z_exit: float) -> np.ndarray:
    """±1 donde |z| supera `z_exit` en una racha que llega al menos una vez a `z_enter`."""
    z = np.asarray(z, dtype=float)
    lab = np.zeros(len(z), dtype=int)
    for sign in (1, -1):
        weak = sign * z > z_exit
        strong = sign * z > z_enter
        for i0, i1, v in runs(weak):
            if v and strong[i0:i1].any():
                lab[i0:i1] = sign
    return lab


def derivative_diagnostics(c_clean, dt_h: float, sigma: float, cfg: DerivConfig) -> dict:
    """C′, C″ y salto de pendiente, con el desvío que le deja el ruido a cada uno."""
    scale = (lambda m: adaptive_minutes(m, sigma)) if cfg.adaptive else (lambda m: m)
    w1 = n_samples(scale(cfg.win_min), dt_h, odd=True, minimum=5)
    w2 = n_samples(scale(2.0 * cfg.win_min), dt_h, odd=True, minimum=7)
    half = n_samples(scale(cfg.kink_half_min), dt_h)
    d1 = sg_derivative(c_clean, dt_h, w1, cfg.poly, 1)
    d2 = sg_derivative(c_clean, dt_h, w2, 3, 2)
    jump, gain_jump = slope_jump(c_clean, dt_h, half)
    return {
        "d1": d1,
        "d1_sd": sigma * sg_gain(w1, cfg.poly, 1, dt_h),
        "d2": d2,
        "d2_sd": sigma * sg_gain(w2, 3, 2, dt_h),
        "jump": jump,
        "jump_sd": sigma * gain_jump,
        "w1": w1,
        "w2": w2,
        "half": half,
    }


def segment_derivative(t_h, c_obs, cfg: DerivConfig | None = None) -> Segmentation:
    cfg = cfg or DerivConfig()
    t_h = np.asarray(t_h, dtype=float)
    dt_h = float(t_h[1] - t_h[0])
    sigma = max(noise_sd(c_obs), cfg.sigma_min)
    c_clean, _ = despike(c_obs, sigma)
    dg = derivative_diagnostics(c_clean, dt_h, sigma, cfg)

    scale = (lambda m: adaptive_minutes(m, sigma)) if cfg.adaptive else (lambda m: m)
    z1 = dg["d1"] / dg["d1_sd"]
    lab = hysteresis_labels(z1, cfg.z_enter, cfg.z_exit)
    gap = n_samples(scale(cfg.gap_min), dt_h, minimum=1)
    lab = split_short_gaps(lab, gap)
    lab = close_gaps(lab, gap)
    lab = drop_short(lab, n_samples(scale(cfg.min_dur_min), dt_h, minimum=2))

    kinks = np.array([], dtype=int)
    kink_z = np.array([])
    breaks = []
    if cfg.use_kinks:
        kinks, kink_z = find_kinks(c_clean, dt_h, sigma, dg["half"], cfg.kink_z)
        lab = snap_boundaries(lab, kinks, n_samples(scale(cfg.snap_min), dt_h, minimum=1))
        breaks = [int(k) for k in kinks if 0 < k < len(lab) and lab[k] != 0 and lab[k - 1] == lab[k]]

    info = dict(dg, z1=z1, kinks=kinks, kink_z=kink_z, breaks=np.array(breaks, dtype=int), sigma=sigma, c_clean=c_clean, config=cfg)
    return Segmentation(segments=segments_from_labels(lab, breaks), labels=lab, info=info)
