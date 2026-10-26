"""Ajuste de la exponencial de un tramo (sección 4.3 y apéndice B.2).

    y(t) ≈ C_ss + (C0 − C_ss)·exp(−λ (t − t0))

Para λ fijo el modelo es lineal en C0 y C_ss: se recorre una grilla de λ y el
mejor punto se refina con `least_squares`. `fit_hyun` es el ajuste (E, λ) de
Hyun con `curve_fit`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit, least_squares

from .physics import exp_curve

LAM_BOUNDS = (0.02, 30.0)  # 1/h
CSS_MAX = 20000.0  # ppm
S_MAX = 5.0e4  # ppm/h


@dataclass
class Fit:
    """Resultado del ajuste del tramo [i0, i1)."""

    i0: int
    i1: int
    t0_h: float
    duration_h: float
    n: int
    model: str
    phase: int = 0
    ok: bool = False
    lam: float = np.nan
    lam_se: float = np.nan
    c0: float = np.nan
    css: float = np.nan
    s: float = np.nan  # fuente [ppm/h] = λ (C_ss − C_ext)
    sse: float = np.nan
    r2: float = np.nan
    rmse: float = np.nan
    at_bound: bool = False
    accepted: bool = False
    reason: str = ""
    s_bound: str = ""  # generación acotada: "inferior" o "superior" si S quedó en la cota

    def predict(self, t_h):
        return exp_curve(np.asarray(t_h, dtype=float) - self.t0_h, self.c0, self.css, self.lam)


def _empty(t_h, i0, i1, model, phase) -> Fit:
    n = len(t_h)
    dur = (t_h[-1] - t_h[0]) + (t_h[1] - t_h[0]) if n > 1 else 0.0
    return Fit(i0=i0, i1=i1, t0_h=float(t_h[0]) if n else np.nan, duration_h=float(dur), n=n, model=model, phase=phase)


def _goodness(fit: Fit, y, resid) -> None:
    sse = float(resid @ resid)
    sst = float(np.sum((y - y.mean()) ** 2))
    fit.sse = sse
    fit.r2 = 1.0 - sse / sst if sst > 0 else np.nan
    fit.rmse = float(np.sqrt(sse / len(y)))


def _profile(tau, y, lam_grid, c0, css, css_min, css_max):
    """Para cada λ de la grilla: mejores (C0, C_ss) y su SSE."""
    w = np.exp(-np.outer(lam_grid, tau))  # (G, n)
    g = len(lam_grid)
    if c0 is None and css is None:
        wm = w.mean(axis=1, keepdims=True)
        ym = y.mean()
        sxx = np.sum((w - wm) ** 2, axis=1)
        sxy = (w - wm) @ (y - ym)
        amp = np.where(sxx > 1e-12, sxy / np.where(sxx > 1e-12, sxx, 1.0), 0.0)
        css_g = ym - amp * wm[:, 0]
        clip = np.clip(css_g, -np.inf if css_min is None else css_min, css_max)
        fixed = clip != css_g
        if np.any(fixed):  # asíntota en la cota: queda sólo la amplitud
            amp_f = (w * (y - clip[:, None])).sum(axis=1) / np.sum(w**2, axis=1)
            amp = np.where(fixed, amp_f, amp)
        css_g = clip
        c0_g = css_g + amp
    elif c0 is not None and css is None:
        u = 1.0 - w
        suu = np.sum(u**2, axis=1)
        b = np.where(suu > 1e-12, (u @ (y - c0)) / np.where(suu > 1e-12, suu, 1.0), 0.0)
        css_g = np.clip(c0 + b, -np.inf if css_min is None else css_min, css_max)
        c0_g = np.full(g, float(c0))
    elif c0 is None and css is not None:
        amp = (w @ (y - css)) / np.sum(w**2, axis=1)
        css_g = np.full(g, float(css))
        c0_g = css_g + amp
    else:
        css_g = np.full(g, float(css))
        c0_g = np.full(g, float(c0))
    pred = css_g[:, None] + (c0_g - css_g)[:, None] * w
    sse = np.sum((pred - y) ** 2, axis=1)
    return c0_g, css_g, sse


def fit_exponential(
    t_h,
    y,
    *,
    c_ext: float,
    c0: float | None = None,
    css: float | None = None,
    css_min: float | None = None,
    css_max: float = CSS_MAX,
    lam_bounds=LAM_BOUNDS,
    n_grid: int = 61,
    i0: int = 0,
    model: str = "general",
    phase: int = 0,
) -> Fit:
    """Exponencial general. Si se pasan, `c0` y `css` quedan fijos; `css_min` acota la asíntota.

    Con css = C_ext es el decaimiento puro.
    """
    t_h = np.asarray(t_h, dtype=float)
    y = np.asarray(y, dtype=float)
    fit = _empty(t_h, i0, i0 + len(t_h), model, phase)
    free = [name for name, val in (("c0", c0), ("css", css)) if val is None] + ["lam"]
    n = len(y)
    if n < len(free) + 2:
        fit.reason = "pocos puntos"
        return fit
    tau = t_h - t_h[0]
    lo, hi = lam_bounds
    grid = np.geomspace(lo, hi, n_grid)
    c0_g, css_g, sse_g = _profile(tau, y, grid, c0, css, css_min, css_max)
    j = int(np.argmin(sse_g))
    p = {"c0": float(c0_g[j]), "css": float(css_g[j]), "lam": float(grid[j])}

    lower = {"c0": -np.inf, "css": -np.inf if css_min is None else css_min, "lam": lo}
    upper = {"c0": np.inf, "css": css_max, "lam": hi}

    def unpack(theta):
        q = dict(p)
        q.update(zip(free, theta))
        return q["c0"], q["css"], q["lam"]

    def resid(theta):
        a, b, lam = unpack(theta)
        return b + (a - b) * np.exp(-lam * tau) - y

    def jac(theta):
        a, b, lam = unpack(theta)
        w = np.exp(-lam * tau)
        cols = {"c0": w, "css": 1.0 - w, "lam": -(a - b) * tau * w}
        return np.column_stack([cols[name] for name in free])

    lb = np.array([lower[name] for name in free])
    ub = np.array([upper[name] for name in free])
    x0 = np.array([p[name] for name in free])
    span = np.where(np.isfinite(ub - lb), ub - lb, 1.0)
    x0 = np.clip(x0, lb + 1e-9 * span, ub - 1e-9 * span)
    try:
        sol = least_squares(resid, x0, jac=jac, bounds=(lb, ub), method="trf", x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    except (ValueError, FloatingPointError):
        fit.reason = "el ajuste falló"
        return fit
    theta = sol.x
    fit.c0, fit.css, fit.lam = unpack(theta)
    _goodness(fit, y, resid(theta))

    # error estándar sólo con los parámetros que no quedaron en una cota (B.3.3)
    tol = 1e-6
    active = np.array([(v - a) > tol * max(1.0, abs(v)) and (b - v) > tol * max(1.0, abs(v)) for v, a, b in zip(theta, lb, ub)])
    k_lam = free.index("lam")
    fit.at_bound = not active[k_lam]
    dof = n - int(active.sum())
    if active[k_lam] and dof > 0:
        ja = jac(theta)[:, active]
        try:
            cov = np.linalg.inv(ja.T @ ja) * fit.sse / dof
            fit.lam_se = float(np.sqrt(cov[-1, -1]))  # λ es la última columna activa
        except np.linalg.LinAlgError:
            fit.lam_se = np.inf
    fit.s = fit.lam * (fit.css - c_ext)
    fit.ok = bool(np.isfinite(fit.lam))
    return fit


def fit_hyun(
    t_h,
    y,
    *,
    c_ext: float,
    c0: float,
    lam_bounds=LAM_BOUNDS,
    s_max: float = S_MAX,
    i0: int = 0,
    model: str = "hyun (E, λ)",
    phase: int = 0,
) -> Fit:
    """Ajuste de Hyun: S = E/V y λ a la vez, con C0 y C_ext fijos. Arranca de varios λ y se queda con el mejor."""
    t_h = np.asarray(t_h, dtype=float)
    y = np.asarray(y, dtype=float)
    fit = _empty(t_h, i0, i0 + len(t_h), model, phase)
    n = len(y)
    if n < 4:
        fit.reason = "pocos puntos"
        return fit
    tau = t_h - t_h[0]
    lo, hi = lam_bounds

    def curve(tt, s, lam):
        css = c_ext + s / lam
        return css + (c0 - css) * np.exp(-lam * tt)

    k = max(2, min(n, 6))
    slope0 = np.polyfit(tau[:k], y[:k], 1)[0]
    best = None
    for lam0 in (0.2, 1.0, 3.0, 8.0):
        s0 = float(np.clip(slope0 + lam0 * (c0 - c_ext), 1.0, s_max * 0.99))
        try:
            popt, pcov = curve_fit(curve, tau, y, p0=[s0, lam0], bounds=([0.0, lo], [s_max, hi]), method="trf", x_scale=[1000.0, 1.0], maxfev=2000)
        except (RuntimeError, ValueError):
            continue
        sse = float(np.sum((curve(tau, *popt) - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, popt, pcov)
    if best is None:
        fit.reason = "el ajuste falló"
        return fit
    _, (s_hat, lam_hat), pcov = best
    fit.lam, fit.s = float(lam_hat), float(s_hat)
    fit.c0 = float(c0)
    fit.css = c_ext + fit.s / fit.lam
    _goodness(fit, y, curve(tau, s_hat, lam_hat) - y)
    fit.at_bound = (lam_hat - lo) < 1e-6 * max(1.0, lo) or (hi - lam_hat) < 1e-6 * hi
    if np.all(np.isfinite(pcov)):
        fit.lam_se = float(np.sqrt(pcov[1, 1]))
    fit.ok = True
    return fit


def fit_known_source(
    t_h,
    y,
    *,
    c_ext: float,
    s: float,
    c0: float | None = None,
    lam_bounds=LAM_BOUNDS,
    n_grid: int = 61,
    i0: int = 0,
    model: str = "G conocida",
    phase: int = 0,
) -> Fit:
    """Generación conocida: C_ss(λ) = C_ext + S/λ. Libre λ, y C0 si no se pasa. Con S = 0 es el decaimiento puro."""
    t_h = np.asarray(t_h, dtype=float)
    y = np.asarray(y, dtype=float)
    fit = _empty(t_h, i0, i0 + len(t_h), model, phase)
    c0_free = c0 is None
    n_par = 2 if c0_free else 1
    n = len(y)
    if n < n_par + 2:
        fit.reason = "pocos puntos"
        return fit
    tau = t_h - t_h[0]
    lo, hi = lam_bounds

    def parts(lam):
        w = np.exp(-lam * tau)
        u = -np.expm1(-lam * tau)  # 1 − w sin perder precisión con λτ chico
        return w, u

    grid = np.geomspace(lo, hi, n_grid)
    w_g = np.exp(-np.outer(grid, tau))
    base_g = (c_ext + s / grid)[:, None] * -np.expm1(-np.outer(grid, tau))
    if c0_free:
        c0_g = np.sum(w_g * (y - base_g), axis=1) / np.sum(w_g * w_g, axis=1)
    else:
        c0_g = np.full(n_grid, float(c0))
    sse_g = np.sum((base_g + c0_g[:, None] * w_g - y) ** 2, axis=1)
    j = int(np.argmin(sse_g))

    def unpack(theta):
        return (theta[0] if c0_free else float(c0)), theta[-1]

    def resid(theta):
        a, lam = unpack(theta)
        w, u = parts(lam)
        return (c_ext + s / lam) * u + a * w - y

    def jac(theta):  # ec. (B.14)
        a, lam = unpack(theta)
        w, u = parts(lam)
        css = c_ext + s / lam
        d_lam = -(s / lam**2) * u + (css - a) * tau * w
        return np.column_stack([w, d_lam]) if c0_free else d_lam.reshape(-1, 1)

    x0 = np.array([c0_g[j], grid[j]]) if c0_free else np.array([grid[j]])
    lb = np.array([-np.inf, lo]) if c0_free else np.array([lo])
    ub = np.array([np.inf, hi]) if c0_free else np.array([hi])
    x0[-1] = np.clip(x0[-1], lo * (1 + 1e-9), hi * (1 - 1e-9))
    try:
        sol = least_squares(resid, x0, jac=jac, bounds=(lb, ub), method="trf", x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    except (ValueError, FloatingPointError):
        fit.reason = "el ajuste falló"
        return fit
    fit.c0, fit.lam = (float(v) for v in unpack(sol.x))
    fit.s = float(s)
    fit.css = c_ext + s / fit.lam
    _goodness(fit, y, resid(sol.x))
    fit.at_bound = (fit.lam - lo) < 1e-6 * max(1.0, lo) or (hi - fit.lam) < 1e-6 * hi
    if not fit.at_bound and n > n_par:
        jm = jac(sol.x)
        try:
            cov = np.linalg.inv(jm.T @ jm) * fit.sse / (n - n_par)
            fit.lam_se = float(np.sqrt(cov[-1, -1]))
        except np.linalg.LinAlgError:
            pass
    fit.ok = bool(np.isfinite(fit.lam))
    return fit


def fit_bounded_source(
    t_h,
    y,
    *,
    c_ext: float,
    s_min: float,
    s_max: float,
    c0: float | None = None,
    lam_bounds=LAM_BOUNDS,
    n_grid: int = 61,
    i0: int = 0,
    model: str = "G acotada",
    phase: int = 0,
) -> Fit:
    """Generación acotada (4.3): C_ss = C_ext + S/λ con s_min ≤ S ≤ s_max. Libres S, λ y C0 si no se pasa.

    Si S queda adentro es el ajuste de asíntota libre; si no, el de generación
    conocida en la cota (`s_bound` dice en cuál).
    """
    t_h = np.asarray(t_h, dtype=float)
    y = np.asarray(y, dtype=float)
    fit = _empty(t_h, i0, i0 + len(t_h), model, phase)
    c0_free = c0 is None
    free = (["c0"] if c0_free else []) + ["s", "lam"]
    n = len(y)
    if n < len(free) + 2:
        fit.reason = "pocos puntos"
        return fit
    tau = t_h - t_h[0]
    lo, hi = lam_bounds
    grid = np.geomspace(lo, hi, n_grid)
    w = np.exp(-np.outer(grid, tau))
    u = -np.expm1(-np.outer(grid, tau))
    # para cada λ el óptimo con C_ss en un intervalo es exacto: si se sale, va al borde
    css_lo, css_hi = c_ext + s_min / grid, c_ext + s_max / grid
    if c0_free:
        wm = w.mean(axis=1, keepdims=True)
        sxx = np.sum((w - wm) ** 2, axis=1)
        amp = np.where(sxx > 1e-12, ((w - wm) @ (y - y.mean())) / np.where(sxx > 1e-12, sxx, 1.0), 0.0)
        css_g = np.clip(y.mean() - amp * wm[:, 0], css_lo, css_hi)
        c0_g = css_g + (w * (y - css_g[:, None])).sum(axis=1) / np.sum(w * w, axis=1)
    else:
        suu = np.sum(u * u, axis=1)
        b = np.where(suu > 1e-12, (u @ (y - c0)) / np.where(suu > 1e-12, suu, 1.0), 0.0)
        css_g = np.clip(c0 + b, css_lo, css_hi)
        c0_g = np.full(n_grid, float(c0))
    sse_g = np.sum((css_g[:, None] * u + c0_g[:, None] * w - y) ** 2, axis=1)
    j = int(np.argmin(sse_g))
    start = {"c0": c0_g[j], "s": float(np.clip((css_g[j] - c_ext) * grid[j], s_min, s_max)), "lam": grid[j]}

    def unpack(theta):
        q = dict(zip(free, theta))
        return q.get("c0", c0), q["s"], q["lam"]

    def resid(theta):
        a, s, lam = unpack(theta)
        uu = -np.expm1(-lam * tau)
        return (c_ext + s / lam) * uu + a * (1.0 - uu) - y

    def jac(theta):
        a, s, lam = unpack(theta)
        ww = np.exp(-lam * tau)
        uu = 1.0 - ww
        css = c_ext + s / lam
        cols = {"c0": ww, "s": uu / lam, "lam": -(s / lam**2) * uu + (css - a) * tau * ww}
        return np.column_stack([cols[k] for k in free])

    lb = np.array([{"c0": -np.inf, "s": s_min, "lam": lo}[k] for k in free])
    ub = np.array([{"c0": np.inf, "s": s_max, "lam": hi}[k] for k in free])
    x0 = np.array([start[k] for k in free], dtype=float)
    span = np.where(np.isfinite(ub - lb), ub - lb, 1.0)
    x0 = np.clip(x0, lb + 1e-9 * span, ub - 1e-9 * span)
    try:
        sol = least_squares(resid, x0, jac=jac, bounds=(lb, ub), method="trf", x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    except (ValueError, FloatingPointError):
        fit.reason = "el ajuste falló"
        return fit
    a, s, lam = unpack(sol.x)
    fit.c0, fit.s, fit.lam = float(a), float(s), float(lam)
    fit.css = c_ext + fit.s / fit.lam
    _goodness(fit, y, resid(sol.x))
    tol = 1e-6
    active = np.array([(v - p) > tol * max(1.0, abs(v)) and (q - v) > tol * max(1.0, abs(v)) for v, p, q in zip(sol.x, lb, ub)])
    k_lam, k_s = free.index("lam"), free.index("s")
    fit.at_bound = not active[k_lam]
    if not active[k_s]:
        fit.s_bound = "inferior" if fit.s - s_min < s_max - fit.s else "superior"
    dof = n - int(active.sum())
    if active[k_lam] and dof > 0:
        ja = jac(sol.x)[:, active]
        try:
            cov = np.linalg.inv(ja.T @ ja) * fit.sse / dof
            fit.lam_se = float(np.sqrt(cov[-1, -1]))
        except np.linalg.LinAlgError:
            pass
    fit.ok = bool(np.isfinite(fit.lam))
    return fit


def lam_se_theory(tau_h, c0: float, css: float, lam: float, sigma: float, free=("c0", "css", "lam")) -> float:
    """Cota de Cramér–Rao para el desvío de λ̂ (asíntota libre o fija), ec. (B.9)."""
    tau = np.asarray(tau_h, dtype=float)
    w = np.exp(-lam * tau)
    cols = {"c0": w, "css": 1.0 - w, "lam": -(c0 - css) * tau * w}
    j = np.column_stack([cols[name] for name in free])
    try:
        cov = np.linalg.inv(j.T @ j) * sigma**2
    except np.linalg.LinAlgError:
        return np.inf
    return float(np.sqrt(cov[-1, -1]))


def theory_known(tau_h, c0: float, s: float, c_ext: float, lam: float, sigma: float) -> float:
    """Lo mismo con la generación conocida, θ = (C0, λ) y C∞ = C_ext + S/λ (ec. B.14)."""
    tau = np.asarray(tau_h, dtype=float)
    w = np.exp(-lam * tau)
    css = c_ext + s / lam
    j = np.column_stack([w, -(s / lam**2) * (1.0 - w) + (css - c0) * tau * w])
    try:
        cov = np.linalg.inv(j.T @ j) * sigma**2
    except np.linalg.LinAlgError:
        return np.inf
    return float(np.sqrt(cov[1, 1]))
