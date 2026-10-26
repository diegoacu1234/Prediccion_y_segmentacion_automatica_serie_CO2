"""Funciones de señal que comparten los segmentadores."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from scipy.signal import find_peaks, savgol_coeffs, savgol_filter


def n_samples(minutes: float, dt_h: float, odd: bool = False, minimum: int = 3) -> int:
    """Muestras que entran en `minutes` (impar si `odd`)."""
    n = max(minimum, int(round(minutes / 60.0 / dt_h)))
    if odd and n % 2 == 0:
        n += 1
    return n


def median_filter(x, size: int) -> np.ndarray:
    """Mediana móvil centrada; en los bordes repite el extremo."""
    return ndimage.median_filter(np.asarray(x, dtype=float), size=size, mode="nearest")


_TRIM_K = 3.0
_TRIM_GAIN = 0.98658  # desvío de una normal recortada en ±3σ


def noise_sd(x, floor: float = 0.5) -> float:
    """Desvío del ruido a partir de las segundas diferencias (var = 6σ²).

    RMS recortado en ±3σ y no la MAD: con lecturas en ppm enteros la MAD
    da valores discretos (1,21 con σ = 1, 1,82 con σ = 2).
    """
    d2 = np.diff(np.asarray(x, dtype=float), n=2)
    if len(d2) < 3:
        return floor
    d2 = d2 - np.median(d2)
    s = 1.4826 * float(np.median(np.abs(d2)))
    if not s > 0:
        s = float(np.sqrt(np.mean(d2**2)))
    for _ in range(8):
        keep = np.abs(d2) <= _TRIM_K * s
        if keep.sum() < 3:
            break
        s = float(np.sqrt(np.mean(d2[keep] ** 2))) / _TRIM_GAIN
    return max(s / np.sqrt(6.0), floor)


def despike(x, sigma: float | None = None, k: float = 4.0, m: int = 5):
    """Saca picos aislados sin tocar quiebres ni máximos. Devuelve (serie, máscara).

    1) pico de una muestra en la segunda diferencia;
    2) muestra que se aparta en el mismo sentido de la recta de las m anteriores
       y de la de las m siguientes (un quiebre real coincide con alguna).
    """
    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    out = np.zeros(n, dtype=bool)
    if n < 2 * m + 1:
        return x, out
    # con menos de 2 ppm los umbrales quedan por debajo de la curvatura real
    sigma = max(noise_sd(x) if sigma is None else sigma, 2.0)

    # pasada 1: r[i] = x[i] − media de las vecinas; un pico a da (−a/2, a, −a/2)
    r = np.zeros(n)
    r[1:-1] = x[1:-1] - 0.5 * (x[:-2] + x[2:])
    score = np.zeros(n)
    score[1:-1] = r[1:-1] - 0.5 * (r[:-2] + r[2:])
    sd_score = sigma * np.sqrt(4.375)
    a = np.abs(score)
    out[2:-2] = (
        (a[2:-2] > (k + 1.0) * sd_score)
        & (a[2:-2] >= a[1:-3])
        & (a[2:-2] >= a[3:-1])
        & (a[2:-2] >= a[:-4])
        & (a[2:-2] >= a[4:])
        & (np.sign(r[1:-3]) == -np.sign(r[2:-2]))
        & (np.sign(r[3:-1]) == -np.sign(r[2:-2]))
        & (np.abs(r[1:-3]) > 0.25 * np.abs(r[2:-2]))  # en un quiebre las vecinas dan ~0
        & (np.abs(r[3:-1]) > 0.25 * np.abs(r[2:-2]))
    )
    hit = np.flatnonzero(out)
    x[hit] = 0.5 * (x[hit - 1] + x[hit + 1])

    # pasada 2
    j = np.arange(m)
    jbar = (m - 1) / 2.0
    w = 1.0 / m + (j - jbar) * (m - jbar) / np.sum((j - jbar) ** 2)  # extrapola a j = m
    sd_pred = sigma * np.sqrt(1.0 + np.sum(w**2))
    win = sliding_window_view(x, m)  # win[i] = x[i : i+m]
    p_left = np.full(n, np.nan)
    p_right = np.full(n, np.nan)
    p_left[m:] = win[: n - m] @ w  # x[i-m : i]
    p_right[: n - m] = win[1:] @ w[::-1]  # x[i+1 : i+m+1]
    d_l, d_r = x - p_left, x - p_right
    with np.errstate(invalid="ignore"):
        second = (np.sign(d_l) == np.sign(d_r)) & (np.minimum(np.abs(d_l), np.abs(d_r)) > k * sd_pred)
    x[second] = 0.5 * (p_left[second] + p_right[second])
    return x, out | second


def sg_derivative(x, dt_h: float, window: int, polyorder: int, deriv: int) -> np.ndarray:
    """Derivada de Savitzky–Golay [ppm/h^deriv]."""
    return savgol_filter(np.asarray(x, dtype=float), window, polyorder, deriv=deriv, delta=dt_h, mode="interp")


def sg_gain(window: int, polyorder: int, deriv: int, dt_h: float) -> float:
    """Cuánto amplifica el filtro el desvío del ruido blanco."""
    return float(np.linalg.norm(savgol_coeffs(window, polyorder, deriv=deriv, delta=dt_h)))


def trailing_minmax(x, w: int):
    """Mínimo y máximo en [t−w, t]."""
    x = np.asarray(x, dtype=float)
    xp = np.concatenate([np.full(w, x[0]), x])
    win = sliding_window_view(xp, w + 1)
    return win.min(axis=1), win.max(axis=1)


def runs(labels) -> list:
    """Rachas de etiquetas iguales: [(i0, i1, valor), ...]."""
    labels = np.asarray(labels)
    n = len(labels)
    if n == 0:
        return []
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, n]
    return [(int(a), int(b), labels[a].item()) for a, b in zip(starts, ends)]


def slope_jump(x, dt_h: float, half_window: int):
    """Salto de pendiente: C′ a la derecha menos C′ a la izquierda. Devuelve (salto [ppm/h], ganancia).

    Cada pendiente sale de una parábola de un solo lado, así que dentro de una
    exponencial el salto es ~0 y en un cambio de régimen vale lo que cambió C′.
    """
    x = np.asarray(x, dtype=float)
    n_side = half_window + 1
    h_left = savgol_coeffs(n_side, 2, deriv=1, delta=dt_h, pos=n_side - 1, use="dot")
    h_right = savgol_coeffs(n_side, 2, deriv=1, delta=dt_h, pos=0, use="dot")
    kernel = np.zeros(2 * n_side - 1)
    kernel[:n_side] -= h_left
    kernel[n_side - 1 :] += h_right
    jump = np.zeros(len(x))
    if len(x) >= len(kernel):
        jump[half_window : len(x) - half_window] = sliding_window_view(x, len(kernel)) @ kernel
    return jump, float(np.linalg.norm(kernel))


def find_kinks(x, dt_h: float, sigma: float, half_window: int, z: float = 5.0):
    """Quiebres: picos de |salto|/σ_S por encima de z. Devuelve (índices, salto estandarizado)."""
    jump, gain = slope_jump(x, dt_h, half_window)
    zscore = jump / (sigma * gain)
    idx, _ = find_peaks(np.abs(zscore), height=z, distance=half_window + 1)
    return idx, zscore[idx]


def close_gaps(labels, max_gap: int) -> np.ndarray:
    """Une dos transitorios del mismo signo separados por una meseta corta."""
    lab = np.asarray(labels).copy()
    rr = runs(lab)
    for k in range(1, len(rr) - 1):
        i0, i1, v = rr[k]
        left, right = rr[k - 1][2], rr[k + 1][2]
        if v == 0 and i1 - i0 <= max_gap and left == right and left != 0:
            lab[i0:i1] = left
    return lab


def split_short_gaps(labels, max_gap: int) -> np.ndarray:
    """Meseta corta entre transitorios de signo opuesto: mitad para cada uno."""
    lab = np.asarray(labels).copy()
    rr = runs(lab)
    for k in range(1, len(rr) - 1):
        i0, i1, v = rr[k]
        left, right = rr[k - 1][2], rr[k + 1][2]
        if v == 0 and i1 - i0 <= max_gap and left != 0 and right != 0 and left != right:
            mid = (i0 + i1) // 2
            lab[i0:mid] = left
            lab[mid:i1] = right
    return lab


def drop_short(labels, min_len: int) -> np.ndarray:
    """Los transitorios de menos de `min_len` muestras pasan a meseta."""
    lab = np.asarray(labels).copy()
    for i0, i1, v in runs(lab):
        if v != 0 and i1 - i0 < min_len:
            lab[i0:i1] = 0
    return lab


def snap_boundaries(labels, kinks, tol: int) -> np.ndarray:
    """Corre cada borde al quiebre más cercano (hasta `tol` muestras).

    La derivada suavizada ve el cambio antes de tiempo; el quiebre da el lugar fino.
    """
    lab = np.asarray(labels).copy()
    kinks = np.asarray(kinks, dtype=int)
    rr = runs(lab)
    if len(kinks) == 0 or len(rr) < 2:
        return lab
    edges = [i0 for i0, _, _ in rr][1:]
    moved = list(edges)
    used = set()
    for j, b in enumerate(edges):
        cand = [int(k) for k in kinks if abs(int(k) - b) <= tol and int(k) not in used]
        if not cand:
            continue
        k = min(cand, key=lambda q: abs(q - b))
        lower = moved[j - 1] if j > 0 else 0
        upper = edges[j + 1] if j + 1 < len(edges) else len(lab)
        if lower < k < upper:  # sin cruzar los bordes vecinos
            moved[j] = k
            used.add(k)
    bounds = [0] + moved + [len(lab)]
    for r, (_, _, v) in enumerate(rr):
        lab[bounds[r] : bounds[r + 1]] = v
    return lab
