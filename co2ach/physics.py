"""Balance de masa de CO2 con mezcla perfecta (cap. 2).

    dC/dt = λ (C_ext − C) + S,   S = 1000·G/V [ppm/h]
    C(t) = C_ss + (C0 − C_ss)·exp(−λ (t − t0)),   C_ss = C_ext + S/λ

C en ppm, λ (ACH) en 1/h, G en L/h y V en m³.
"""

from __future__ import annotations

import numpy as np


def source_from_generation(g_lh, volume_m3):
    """G [L/h] y V [m³] -> S [ppm/h]."""
    return 1000.0 * np.asarray(g_lh, dtype=float) / volume_m3


def generation_from_source(s_ppm_h, volume_m3):
    """S [ppm/h] y V [m³] -> G [L/h]."""
    return np.asarray(s_ppm_h, dtype=float) * volume_m3 / 1000.0


# Schofield (1985) según Persily y de Jonge (2017): BMR = a·m + b [MJ/día], m en kg.
# edad mínima del tramo -> ((a, b) varones, (a, b) mujeres)
SCHOFIELD = {
    0: ((0.249, -0.127), (0.244, -0.130)),
    3: ((0.095, 2.110), (0.085, 2.033)),
    10: ((0.074, 2.754), (0.056, 2.898)),
    18: ((0.063, 2.896), (0.062, 2.036)),
    30: ((0.048, 3.653), (0.034, 3.538)),
    60: ((0.049, 2.459), (0.038, 2.755)),
}


def bmr_schofield(mass_kg: float, age: float, sex: str | None = None) -> float:
    """Metabolismo basal [MJ/día]. sex: "m", "f" o None (promedio)."""
    male, female = SCHOFIELD[max(a for a in SCHOFIELD if a <= age)]
    values = {"m": male[0] * mass_kg + male[1], "f": female[0] * mass_kg + female[1]}
    return values[sex] if sex else 0.5 * (values["m"] + values["f"])


def co2_generation(bmr_mj_day: float, met: float, rq: float = 0.85, temp_c: float = 20.0, p_kpa: float = 101.0) -> float:
    """CO2 que genera una persona [L/h] (Persily y de Jonge, 2017).

    V = RQ·BMR·M·0,000569 [L/s a 273 K y 101 kPa], pasado a la temperatura y presión del aula.
    """
    return rq * bmr_mj_day * met * 0.000569 * 3600.0 * (273.15 + temp_c) / 273.15 * 101.0 / p_kpa


def exp_curve(tau_h, c0, css, lam):
    """Solución con parámetros constantes; tau_h es el tiempo desde el inicio del tramo."""
    return css + (c0 - css) * np.exp(-lam * np.asarray(tau_h, dtype=float))
