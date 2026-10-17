"""PELT con un costo exponencial por segmento.

PELT (Killick et al., 2012) minimiza  Σ costo(segmento) + β·(número de cortes).
Los costos estándar (`l2`, `rbf`) suponen un nivel constante por tramo y
partirían cada exponencial en escalones. Acá el costo de un tramo es la suma de
residuos al cuadrado del mejor exponencial  C_ss + A·exp(−λτ), que es la
solución del balance de masa con parámetros constantes. Así los cortes caen
donde cambia el régimen (G o λ), y un régimen incluye su transitorio y su
meseta.

El costo se calcula por proyección de variables: para cada λ de una grilla el
ajuste de (C_ss, A) es lineal y se resuelve con sumas acumuladas, de modo que
toda la fila de costos con un mismo inicio sale de una sola pasada vectorizada.
La grilla es fina (pasos de ~3 %) y el mínimo se refina con una parábola,
porque en tramos largos λ queda determinado con mucha precisión y un costo
inflado por la grilla induciría cortes espurios.
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
    """RSS del mejor exponencial en el tramo [start, end)."""

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
        """Mínimo sobre λ del RSS, dadas las sumas del tramo (una columna por largo)."""
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
        best = np.where(ok, best - (r0 - r2) ** 2 / (8.0 * np.where(ok, curv, 1.0)), best)
        return np.maximum(best, 0.0)

    def _row(self, start: int):
        """Costos de todos los tramos que empiezan en `start` y terminan en la grilla de `jump`."""
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
        # tramo fuera de la grilla: cálculo directo
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
    min_size_samples: int = 6  # piso en muestras: un tramo más corto no determina el exponencial
    jump_min: float = 2.0
    pen_factor: float = 4.0  # β = pen_factor · σ̂² · ln(n)
    lam_min: float = 0.02
    lam_max: float = 20.0
    n_lam: int = 240
    flat_z: float = 4.0  # un régimen es meseta si su recorrido ajustado es < flat_z·σ̂
    phase_fraction: float = 0.05
    sigma_min: float = 0.0  # piso del ruido [ppm]: con datos reales, la fluctuación de la señal que el método debe ignorar


def make_cost(c_clean, dt_h: float, cfg: PeltConfig):
    """Costo ajustado a la serie y el objeto PELT que lo usa."""
    min_size = n_samples(cfg.min_size_min, dt_h, minimum=cfg.min_size_samples)
    jump = n_samples(cfg.jump_min, dt_h, minimum=1)
    cost = CostExp(dt_h=dt_h, lam_min=cfg.lam_min, lam_max=cfg.lam_max, n_lam=cfg.n_lam, min_size=min_size, jump=jump)
    algo = rpt.Pelt(custom_cost=cost, min_size=min_size, jump=jump).fit(np.asarray(c_clean, dtype=float))
    return cost, algo


def penalty(sigma: float, n: int, pen_factor: float) -> float:
    """Penalización tipo BIC por cada régimen extra, en unidades de RSS."""
    return pen_factor * sigma**2 * np.log(n)


def segment_pelt(t_h, c_obs, cfg: PeltConfig | None = None) -> Segmentation:
    cfg = cfg or PeltConfig()
    t_h = np.asarray(t_h, dtype=float)
    dt_h = float(t_h[1] - t_h[0])
    sigma = max(noise_sd(c_obs), cfg.sigma_min)
    c_clean, _ = despike(c_obs, sigma)
    _, algo = make_cost(c_clean, dt_h, cfg)
    bkps = algo.predict(pen=penalty(sigma, len(t_h), cfg.pen_factor))

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
    """Cantidad de regímenes que da PELT para cada factor de penalización."""
    cfg = cfg or PeltConfig()
    t_h = np.asarray(t_h, dtype=float)
    dt_h = float(t_h[1] - t_h[0])
    sigma = noise_sd(c_obs)
    c_clean, _ = despike(c_obs, sigma)
    _, algo = make_cost(c_clean, dt_h, cfg)  # las filas de costo se reutilizan entre penalizaciones
    return [len(algo.predict(pen=penalty(sigma, len(t_h), f))) for f in factors]
