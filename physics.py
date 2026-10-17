"""Balance de masa de CO2 en una zona única bien mezclada.

    dC/dt = λ (C_ext − C) + S,        S = 1000·G/V   [ppm/h]

con C en ppm, λ (ACH) en 1/h, G en L/h de CO2 y V en m³. Si λ y S son
constantes la solución es una exponencial hacia la asíntota

    C(t) = C_ss + (C0 − C_ss)·exp(−λ (t − t0)),      C_ss = C_ext + S/λ.

Es la ecuación (2) de Hyun (2025) con E/V = S.
"""

from __future__ import annotations

import numpy as np


def source_from_generation(g_lh, volume_m3):
    """Fuente S [ppm/h] a partir de la generación G [L/h] y el volumen V [m³]."""
    return 1000.0 * np.asarray(g_lh, dtype=float) / volume_m3


def generation_from_source(s_ppm_h, volume_m3):
    """Generación G [L/h] a partir de la fuente S [ppm/h] y el volumen V [m³]."""
    return np.asarray(s_ppm_h, dtype=float) * volume_m3 / 1000.0


# Schofield (1985), como lo tabulan Persily y de Jonge (2017): BMR = a·m + b [MJ/día], con m la masa corporal [kg].
# Clave: edad mínima del tramo; valor: (a, b) de varones y (a, b) de mujeres.
SCHOFIELD = {
    0: ((0.249, -0.127), (0.244, -0.130)),
    3: ((0.095, 2.110), (0.085, 2.033)),
    10: ((0.074, 2.754), (0.056, 2.898)),
    18: ((0.063, 2.896), (0.062, 2.036)),
    30: ((0.048, 3.653), (0.034, 3.538)),
    60: ((0.049, 2.459), (0.038, 2.755)),
}


def bmr_schofield(mass_kg: float, age: float, sex: str | None = None) -> float:
    """Metabolismo basal [MJ/día] por masa corporal, edad y sexo ("m" o "f"); sin sexo, el promedio de los dos."""
    male, female = SCHOFIELD[max(a for a in SCHOFIELD if a <= age)]
    values = {"m": male[0] * mass_kg + male[1], "f": female[0] * mass_kg + female[1]}
    return values[sex] if sex else 0.5 * (values["m"] + values["f"])


def co2_generation(bmr_mj_day: float, met: float, rq: float = 0.85, temp_c: float = 20.0, p_kpa: float = 101.0) -> float:
    """Generación de CO2 de una persona [L/h] según Persily y de Jonge (2017).

        V = RQ · BMR · M · 0,000569   [L/s, a 273 K y 101 kPa]

    con BMR el metabolismo basal [MJ/día], M el nivel de actividad [met] y RQ el
    cociente respiratorio. Se lleva a la temperatura y la presión del aire del
    recinto, que es el volumen que entra en el balance.
    """
    return rq * bmr_mj_day * met * 0.000569 * 3600.0 * (273.15 + temp_c) / 273.15 * 101.0 / p_kpa


def steady_state(c_ext, s, lam):
    """Concentración de equilibrio C_ss = C_ext + S/λ."""
    return c_ext + s / lam


def exp_curve(tau_h, c0, css, lam):
    """Respuesta con parámetros constantes; tau_h es el tiempo desde el inicio."""
    return css + (c0 - css) * np.exp(-lam * np.asarray(tau_h, dtype=float))


def advance(c, lam, s, c_ext, h):
    """Avanza la concentración un paso h [h] con λ, S y C_ext constantes (exacto)."""
    css = c_ext + s / lam
    return css + (c - css) * np.exp(-lam * h)
