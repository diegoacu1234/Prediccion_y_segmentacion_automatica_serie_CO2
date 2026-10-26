"""PELT con costo exponencial (sección 4.4.4, apéndice B.7).

Minimiza Σ costo(tramo) + β·(cambios), con costo = SSE de la mejor exponencial
C_ss + A·exp(−λτ) del tramo. Así cada tramo es un régimen entero (transitorio y
meseta). Para cada λ de la grilla el ajuste es lineal y sale de sumas
acumuladas; el mínimo en λ se refina con una parábola.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ruptures as rpt
from ruptures.base import BaseCost
from ruptures.exceptions import NotEnoughPoints

from ..fitting import fit_exponential
from ..signal_utils import despike, n_samples, noise_sd
from .base import STEADY, Segment, Segmentation


class CostExp(BaseCost):
    """SSE de la mejor exponencial en [start, end)."""

    model = "exp"

    def __init__(self, dt_h: float = 1.0 / 120.0, lam_min: float = 0.02, lam_max: float = 20.0, n_lam: int = 240, min_size: int = 20, jump: int = 4):
        self.dt_h = dt_h
        self.lam = np.geomspace(lam_min, lam_max, n_lam)
        self.min_size = min_size
        self.jump = jump
        self.signal = None
        self._rows = {}

    def fit(self, signal) -> "CostExp":
        y = np.asarray(signal, dtype=float).reshape(-1)
        self.signal = y.reshape(-1, 1)
        self._y = y - y.mean()
        self._n = len(y)
        w = np.exp(-np.outer(self.lam, np.arange(self._n) * self.dt_h))
        self._w = w
        self._cw = np.cumsum(w, axis=1)
        self._cww = np.cumsum(w * w, axis=1)
        self._cy = np.concatenate([[0.0], np.cumsum(self._y)])
        self._cyy = np.concatenate([[0.0], np.cumsum(self._y**2)])
        self._rows = {}
        return self

    def _best(self, length, sy, syy, sw, sww, swy) -> np.ndarray:
        """Mínimo en λ del SSE a partir de las sumas del tramo (una columna por largo)."""
        syy_c = syy - sy * sy / length
        sxx = sww - sw * sw / length
        sxy = swy - sw * (sy / length)
        good = sxx > 1e-10 * np.maximum(sww, 1e-300)
        rss = syy_c - np.where(good, sxy * sxy / np.where(good, sxx, 1.0), 0.0)
        j = np.argmin(rss, axis=0)
        cols = np.arange(rss.shape[1])
        best = rss[j, cols]
        last = len(self.lam) - 1
        r0 = rss[np.clip(j - 1, 0, last), cols]
        r2 = rss[np.clip(j + 1, 0, last), cols]
        curv = r0 - 2.0 * best + r2
        ok = (j > 0) & (j < last) & (curv > 0)
        best = np.where(ok, best - (r0 - r2) ** 2 / (8.0 * np.where(ok, curv, 1.0)), best)  # parábola
        return np.maximum(best, 0.0)

    def _row(self, start: int):
        """Costo de todos los tramos que arrancan en `start` (fin en la grilla de `jump`)."""
        row = self._rows.get(start)
        if row is None:
            m = self._n - start
            ends = np.arange(self.jump, m + 1, self.jump)
            if m % self.jump:
                ends = np.r_[ends, m]
            prod = self._w[:, :m] * self._y[start:]
            swy = np.add.reduceat(prod, np.arange(0, m, self.jump), axis=1).cumsum(axis=1)
            sy = self._cy[start + ends] - self._cy[start]
            syy = self._cyy[start + ends] - self._cyy[start]
            row = self._best(ends.astype(float), sy, syy, self._cw[:, ends - 1], self._cww[:, ends - 1], swy)
            self._rows[start] = row
        return row

    def error(self, start: int, end: int) -> float:
        length = end - start
        if length < self.min_size:
            raise NotEnoughPoints
        if start % self.jump == 0 and (length % self.jump == 0 or end == self._n):
            row = self._row(start)
            return float(row[length // self.jump - 1] if length % self.jump == 0 else row[-1])
        # fuera de la grilla: cálculo directo
        y = self._y[start:end]
        swy = (self._w[:, :length] @ y).reshape(-1, 1)
        col = self._best(
            np.array([float(length)]),
            np.array([y.sum()]),
            np.array([y @ y]),
            self._cw[:, [length - 1]],
            self._cww[:, [length - 1]],
            swy,
        )
        return float(col[0])


@dataclass(frozen=True)
class PeltConfig:
    min_size_min: float = 10.0
    min_size_samples: int = 6  # con menos muestras no se determina la exponencial
    jump_min: float = 2.0
    pen_factor: float = 4.0  # β = pen_factor · σ̂² · ln(n)
    lam_min: float = 0.02
    lam_max: float = 20.0
    n_lam: int = 240
    flat_z: float = 4.0  # recorrido < flat_z·σ̂ -> meseta
    phase_fraction: float = 0.05
    sigma_min: float = 0.0  # piso del ruido [ppm]


def make_cost(c_clean, dt_h: float, cfg: PeltConfig):
    """El costo y el objeto PELT ya ajustados a la serie."""
    min_size = n_samples(cfg.min_size_min, dt_h, minimum=cfg.min_size_samples)
    jump = n_samples(cfg.jump_min, dt_h, minimum=1)
    cost = CostExp(dt_h=dt_h, lam_min=cfg.lam_min, lam_max=cfg.lam_max, n_lam=cfg.n_lam, min_size=min_size, jump=jump)
    algo = rpt.Pelt(custom_cost=cost, min_size=min_size, jump=jump).fit(np.asarray(c_clean, dtype=float))
    return cost, algo


def penalty(sigma: float, n: int, pen_factor: float) -> float:
    """β por cada cambio (tipo BIC)."""
    return pen_factor * sigma**2 * np.log(n)


def segment_pelt(t_h, c_obs, cfg: PeltConfig | None = None) -> Segmentation:
    cfg = cfg or PeltConfig()
    t_h = np.asarray(t_h, dtype=float)
    dt_h = float(t_h[1] - t_h[0])
    sigma = max(noise_sd(c_obs), cfg.sigma_min)
    c_clean, _ = despike(c_obs, sigma)
    _, algo = make_cost(c_clean, dt_h, cfg)
    bkps = algo.predict(pen=penalty(sigma, len(t_h), cfg.pen_factor))

    # fase de cada tramo según el recorrido de su exponencial
    labels = np.zeros(len(t_h), dtype=int)
    segments = []
    for i0, i1 in zip([0] + bkps[:-1], bkps):
        f = fit_exponential(t_h[i0:i1], c_clean[i0:i1], c_ext=0.0, i0=i0)
        phase = STEADY
        if f.ok:
            curve = f.predict(t_h[i0:i1])
            if abs(curve[-1] - curve[0]) >= cfg.flat_z * sigma:
                phase = 1 if curve[-1] > curve[0] else -1
                moving = np.abs(curve - f.css) > cfg.phase_fraction * abs(f.c0 - f.css)
                labels[i0:i1] = np.where(moving, phase, 0)
        segments.append(Segment(i0, i1, phase))
    info = {"bkps": np.array(bkps[:-1], dtype=int), "sigma": sigma, "c_clean": c_clean, "config": cfg}
    return Segmentation(segments=segments, labels=labels, info=info)


def penalty_sweep(t_h, c_obs, factors, cfg: PeltConfig | None = None) -> list:
    """Regímenes que encuentra PELT para cada factor de penalización (figura B.3)."""
    cfg = cfg or PeltConfig()
    t_h = np.asarray(t_h, dtype=float)
    dt_h = float(t_h[1] - t_h[0])
    sigma = noise_sd(c_obs)
    c_clean, _ = despike(c_obs, sigma)
    _, algo = make_cost(c_clean, dt_h, cfg)  # reutiliza las filas de costo
    return [len(algo.predict(pen=penalty(sigma, len(t_h), f))) for f in factors]
