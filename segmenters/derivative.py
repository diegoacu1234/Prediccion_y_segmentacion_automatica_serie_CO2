"""Segmentación por derivadas.

1. C′ suavizada (Savitzky–Golay) y estandarizada por su ruido teórico.
2. Umbral con histéresis: un transitorio empieza cuando |z| supera `z_enter` y
   sigue mientras |z| supere `z_exit`; lo demás es meseta.
3. Limpieza: se unen cortes breves y se descartan transitorios muy cortos.
4. Quiebres (derivada segunda concentrada, ver `signal_utils.slope_jump`): cada
   borde se lleva al quiebre más cercano y los quiebres que caen dentro de un
   transitorio lo parten, porque ahí cambió el régimen aunque la curva siga
   subiendo o bajando.
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
    # Las duraciones valen para σ = 5 ppm. Con `adaptive` se escalan todas por el
    # mismo factor (σ/5)^(2/3), que mantiene la pendiente mínima detectable.
    win_min: float = 10.0  # ventana de C′
    poly: int = 2
    z_enter: float = 4.0
    z_exit: float = 2.0
    min_dur_min: float = 5.0
    gap_min: float = 4.0
    use_kinks: bool = True
    kink_half_min: float = 10.0
    kink_z: float = 5.0
    snap_min: float = 6.0  # mayor que media ventana: el umbral ve el cambio hasta media ventana antes
    adaptive: bool = True
    sigma_min: float = 0.0  # piso del ruido [ppm]: con datos reales, la fluctuación de la señal que el método debe ignorar


def hysteresis_labels(z, z_enter: float, z_exit: float) -> np.ndarray:
    """+1 / −1 donde z supera el umbral débil y toca al menos una vez el fuerte."""
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
    """C′, C″ y salto de pendiente, con el desvío de ruido de cada uno."""
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
