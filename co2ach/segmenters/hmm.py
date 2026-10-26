"""Segmentación con un modelo oculto de Markov de tres estados (sección 4.4.5, apéndice B.8).

Estados: bajada, meseta y subida. Observación: la misma z = C′/σ_C′ del método
de la derivada, así que los dos difieren sólo en la regla de decisión.
Emisiones normales, meseta fija en N(0, 1) y subida y bajada simétricas.
Baum–Welch para los parámetros y Viterbi para la secuencia; después la misma
limpieza y el mismo ajuste de bordes a los quiebres que la derivada.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GaussianHMM

from ..signal_utils import close_gaps, despike, drop_short, find_kinks, n_samples, noise_sd, sg_derivative, sg_gain, snap_boundaries, split_short_gaps
from .base import Segmentation, adaptive_minutes, segments_from_labels

logging.getLogger("hmmlearn").setLevel(logging.ERROR)


@dataclass(frozen=True)
class HmmConfig:
    # duraciones para σ = 5 ppm, escaladas con el ruido salvo `dwell_min`
    win_min: float = 10.0  # ventana de C′
    poly: int = 2
    dwell_min: float = 45.0  # permanencia esperada en un estado
    n_iter: int = 25
    prior_weight: float = 20.0
    min_dur_min: float = 5.0
    gap_min: float = 4.0
    snap: bool = True
    kink_half_min: float = 10.0
    kink_z: float = 5.0
    snap_min: float = 6.0
    adaptive: bool = True
    seed: int = 0
    sigma_min: float = 0.0  # piso del ruido [ppm]


class _AnchoredHMM(GaussianHMM):
    """HMM con la meseta fija en N(0, 1) y bajada/subida simétricas, N(−m, v) y N(+m, v).

    Si la meseta pudiera ensancharse se comería las colas de los transitorios, y
    con anchos distintos una subida lenta podría terminar etiquetada como bajada.
    """

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        n_dn, n_up = float(stats["post"][0]), float(stats["post"][2])
        m_dn, m_up = float(self.means_[0, 0]), float(self.means_[2, 0])
        v_dn, v_up = float(self._covars_[0, 0]), float(self._covars_[2, 0])
        tot = max(n_dn + n_up, 1e-9)
        m = (n_dn * (-m_dn) + n_up * m_up) / tot
        v = (n_dn * (v_dn + (-m_dn - m) ** 2) + n_up * (v_up + (m_up - m) ** 2)) / tot
        self.means_[:, 0] = (-m, 0.0, m)
        self._covars_[:, 0] = (v, 1.0, v)


def _initial_model(z, cfg: HmmConfig, dwell: float) -> GaussianHMM:
    big = np.abs(z[np.abs(z) > 4.0])
    m = float(np.median(big)) if len(big) else 10.0
    v = float(max(np.var(big), 4.0)) if len(big) > 1 else 25.0
    means = np.array([[-m], [0.0], [m]])
    covars = np.array([[v], [1.0], [v]])
    stay = 1.0 - 1.0 / max(dwell, 2.0)
    leave = 1.0 - stay
    trans = np.array(
        [
            [stay, 0.7 * leave, 0.3 * leave],
            [0.5 * leave, stay, 0.5 * leave],
            [0.3 * leave, 0.7 * leave, stay],
        ]
    )
    w = cfg.prior_weight
    model = _AnchoredHMM(
        n_components=3,
        covariance_type="diag",
        n_iter=cfg.n_iter,
        tol=1e-3,
        init_params="",
        params="stmc",
        transmat_prior=1.0 + 0.25 * len(z) * trans,
        means_prior=means,
        means_weight=np.array([[1.0], [1e6], [1.0]]),
        covars_prior=w * covars,
        covars_weight=w + 1.0,
        random_state=cfg.seed,
    )
    model.startprob_ = np.array([0.1, 0.8, 0.1])
    model.transmat_ = trans
    model.means_ = means
    model.covars_ = covars
    return model


def segment_hmm(t_h, c_obs, cfg: HmmConfig | None = None) -> Segmentation:
    cfg = cfg or HmmConfig()
    t_h = np.asarray(t_h, dtype=float)
    dt_h = float(t_h[1] - t_h[0])
    sigma = max(noise_sd(c_obs), cfg.sigma_min)
    c_clean, _ = despike(c_obs, sigma)
    scale = (lambda m: adaptive_minutes(m, sigma)) if cfg.adaptive else (lambda m: m)

    win = n_samples(scale(cfg.win_min), dt_h, odd=True, minimum=5)
    z = sg_derivative(c_clean, dt_h, win, cfg.poly, 1) / (sigma * sg_gain(win, cfg.poly, 1, dt_h))
    x = z.reshape(-1, 1)

    dwell = cfg.dwell_min / 60.0 / dt_h
    model = _initial_model(z, cfg, dwell)
    fitted = True
    try:
        model.fit(x)
        m = model.means_.ravel()
        if not (m[0] < m[1] < m[2]) or not np.all(np.isfinite(model.transmat_)):
            raise ValueError("EM desordenó los estados")
    except ValueError:
        model = _initial_model(z, cfg, dwell)  # se decodifica con los parámetros iniciales
        fitted = False
    states = model.predict(x)
    posterior = model.predict_proba(x)

    lab = states - 1  # 0, 1, 2 -> −1, 0, +1
    gap = n_samples(scale(cfg.gap_min), dt_h, minimum=1)
    lab = split_short_gaps(lab, gap)
    lab = close_gaps(lab, gap)
    lab = drop_short(lab, n_samples(scale(cfg.min_dur_min), dt_h, minimum=2))

    kinks = np.array([], dtype=int)
    if cfg.snap:
        half = n_samples(scale(cfg.kink_half_min), dt_h)
        kinks, _ = find_kinks(c_clean, dt_h, sigma, half, cfg.kink_z)
        lab = snap_boundaries(lab, kinks, n_samples(scale(cfg.snap_min), dt_h, minimum=1))

    info = {
        "z": z,
        "posterior": posterior,
        "means": model.means_.ravel().copy(),
        "sds": np.sqrt(model.covars_.reshape(3, -1)[:, 0]),
        "transmat": model.transmat_.copy(),
        "fitted": fitted,
        "kinks": kinks,
        "sigma": sigma,
        "config": cfg,
    }
    return Segmentation(segments=segments_from_labels(lab), labels=lab, info=info)
