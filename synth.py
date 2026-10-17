"""Series sintéticas de CO2 con verdad conocida.

Un escenario es una lista de regímenes: tramos con ocupación y ACH constantes.
El simulador integra el balance de masa con la solución exacta por sub-paso y
después aplica un modelo de sensor (ruido, cuantización, outliers).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .physics import source_from_generation

RISE, STEADY, DECAY = 1, 0, -1
PHASE_NAMES = {RISE: "subida", STEADY: "meseta", DECAY: "bajada"}


@dataclass(frozen=True)
class Regime:
    """Tramo con ocupación y ventilación constantes."""

    t_start_h: float
    people: float
    ach: float
    label: str = ""


@dataclass(frozen=True)
class Room:
    volume_m3: float = 150.0
    c_ext_ppm: float = 420.0
    g_person_lh: float = 18.0  # L/h de CO2 por persona (adulto sedentario)

    @property
    def s_person(self) -> float:
        """Fuente por persona [ppm/h]."""
        return float(source_from_generation(self.g_person_lh, self.volume_m3))


@dataclass(frozen=True)
class Sensor:
    dt_s: float = 30.0
    noise_sd: float = 5.0  # ppm, ruido blanco gaussiano
    quantization: float = 1.0  # ppm, 0 = sin cuantizar
    outlier_rate: float = 0.002  # fracción de muestras con un pico espurio
    outlier_sd: float = 60.0  # ppm
    seed: int = 0


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    duration_h: float
    regimes: tuple
    room: Room = field(default_factory=Room)
    sensor: Sensor = field(default_factory=Sensor)
    c0_ppm: float | None = None  # None: equilibrio del primer régimen
    ramp_min: float = 0.0  # la gente entra o sale gradualmente en este tiempo
    c_ext_drift_ppm: float = 0.0  # amplitud de la deriva diaria del CO2 exterior
    phase_fraction: float = 0.05  # transitorio hasta quedar a 5 % del escalón (3 τ)
    event_min_ppm: float = 30.0  # escalón mínimo para que un régimen cuente como evento


@dataclass
class Series:
    """Serie simulada y su verdad, muestra a muestra."""

    scenario: Scenario
    sensor: Sensor
    t_h: np.ndarray
    c_obs: np.ndarray
    c_true: np.ndarray
    ach: np.ndarray
    s: np.ndarray  # fuente [ppm/h]
    c_ext: np.ndarray
    css: np.ndarray  # asíntota instantánea
    people: np.ndarray
    regime_id: np.ndarray
    phase: np.ndarray  # +1 subida, 0 meseta, −1 bajada
    regimes: pd.DataFrame  # una fila por régimen
    event_min_ppm: float  # escalón mínimo para contar como evento detectable

    @property
    def dt_h(self) -> float:
        return self.sensor.dt_s / 3600.0

    @property
    def n(self) -> int:
        return len(self.t_h)

    @property
    def c_ext_nominal(self) -> float:
        return self.scenario.room.c_ext_ppm

    @property
    def change_idx(self) -> np.ndarray:
        """Índices donde empieza cada régimen (sin contar el primero)."""
        return self.regimes["i0"].to_numpy()[1:]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "t_h": self.t_h,
                "co2_obs_ppm": self.c_obs,
                "co2_verdadero_ppm": self.c_true,
                "ach_verdadero": self.ach,
                "fuente_ppm_h": self.s,
                "personas": self.people,
                "co2_exterior_ppm": self.c_ext,
                "regimen": self.regime_id,
                "fase": self.phase,
            }
        )


def simulate(scn: Scenario, sensor: Sensor | None = None, substep_s: float = 5.0) -> Series:
    """Simula un escenario. `sensor` reemplaza al del escenario si se pasa."""
    sensor = sensor or scn.sensor
    room = scn.room
    m = int(round(sensor.dt_s / substep_s))
    if m < 1 or abs(m * substep_s - sensor.dt_s) > 1e-9:
        raise ValueError("dt_s debe ser múltiplo de substep_s")
    n = int(round(scn.duration_h * 3600.0 / sensor.dt_s))
    h = substep_s / 3600.0
    k = np.arange(n * m)
    t_f = k * h

    starts = np.array([r.t_start_h for r in scn.regimes], dtype=float)
    if starts[0] != 0.0 or np.any(np.diff(starts) <= 0):
        raise ValueError("los regímenes deben empezar en 0 y estar ordenados")
    ach_r = np.array([r.ach for r in scn.regimes], dtype=float)
    ppl_r = np.array([r.people for r in scn.regimes], dtype=float)
    prev_r = np.r_[ppl_r[0], ppl_r[:-1]]

    # el pequeño margen evita que un borde de régimen caiga un sub-paso tarde por redondeo
    rid_f = np.searchsorted(starts, t_f + 1e-9, side="right") - 1
    lam_f = ach_r[rid_f]
    if scn.ramp_min > 0:
        frac = np.clip((t_f - starts[rid_f]) / (scn.ramp_min / 60.0), 0.0, 1.0)
    else:
        frac = np.ones_like(t_f)
    ppl_f = prev_r[rid_f] + (ppl_r[rid_f] - prev_r[rid_f]) * frac
    s_f = ppl_f * room.s_person
    c_ext_f = room.c_ext_ppm + scn.c_ext_drift_ppm * np.sin(2 * np.pi * (t_f - 9.0) / 24.0)
    css_f = c_ext_f + s_f / lam_f
    a_f = np.exp(-lam_f * h)

    c_f = np.empty(n * m)
    c_f[0] = css_f[0] if scn.c0_ppm is None else scn.c0_ppm
    c = c_f[0]
    for j in range(n * m - 1):
        c = css_f[j] + (c - css_f[j]) * a_f[j]
        c_f[j + 1] = c

    sl = slice(None, None, m)
    t_h, c_true = t_f[sl], c_f[sl]
    ach, s, c_ext, css = lam_f[sl], s_f[sl], c_ext_f[sl], css_f[sl]
    people, regime_id = ppl_f[sl], rid_f[sl]

    event_min = scn.event_min_ppm  # fijo: los regímenes evaluados son los mismos con cualquier ruido
    rows = []
    phase = np.zeros(n, dtype=int)
    for r, reg in enumerate(scn.regimes):
        idx = np.flatnonzero(regime_id == r)
        i0, i1 = int(idx[0]), int(idx[-1]) + 1
        s_nom = reg.people * room.s_person
        css_nom = room.c_ext_ppm + s_nom / reg.ach
        amp = css_nom - c_true[i0]
        if abs(amp) < event_min:
            kind = STEADY
        else:
            kind = RISE if amp > 0 else DECAY
            dist = css[i0:i1] - c_true[i0:i1]
            moving = np.abs(dist) > scn.phase_fraction * abs(amp)
            phase[i0:i1] = np.where(moving, np.sign(dist), 0).astype(int)
        t0, t1 = t_h[i0], t_h[i1 - 1] + sensor.dt_s / 3600.0
        n_trans = int(np.count_nonzero(phase[i0:i1]))
        rows.append(
            {
                "regimen": r,
                "etiqueta": reg.label,
                "i0": i0,
                "i1": i1,
                "t0_h": t0,
                "t1_h": t1,
                "personas": reg.people,
                "ach": reg.ach,
                "fuente_ppm_h": s_nom,
                "css_ppm": css_nom,
                "c_inicio_ppm": c_true[i0],
                "c_fin_ppm": c_true[i1 - 1],
                "escalon_ppm": amp,
                "tipo": kind,
                "n_tau": reg.ach * (t1 - t0),
                "i_fin_transitorio": i0 + n_trans,
            }
        )
    regimes = pd.DataFrame(rows)

    rng = np.random.default_rng(sensor.seed)
    noise = rng.normal(0.0, sensor.noise_sd, n) if sensor.noise_sd > 0 else np.zeros(n)
    if sensor.outlier_rate > 0:
        hit = rng.random(n) < sensor.outlier_rate
        noise[hit] += rng.normal(0.0, sensor.outlier_sd, int(hit.sum()))
    c_obs = c_true + noise
    if sensor.quantization > 0:
        c_obs = np.round(c_obs / sensor.quantization) * sensor.quantization

    return Series(
        scenario=scn,
        sensor=sensor,
        t_h=t_h,
        c_obs=c_obs,
        c_true=c_true,
        ach=ach,
        s=s,
        c_ext=c_ext,
        css=css,
        people=people,
        regime_id=regime_id,
        phase=phase,
        regimes=regimes,
        event_min_ppm=event_min,
    )


def scenario_a() -> Scenario:
    """Día de aula: eventos bien separados, con y sin meseta."""
    return Scenario(
        key="A",
        title="Escenario A: día de aula",
        duration_h=24.0,
        regimes=(
            Regime(0.0, 0, 0.5, "vacío, cerrado"),
            Regime(7.0, 20, 2.0, "clase"),
            Regime(9.5, 4, 5.0, "recreo, ventanas abiertas"),
            Regime(10.0, 28, 3.0, "clase"),
            Regime(12.0, 0, 3.0, "almuerzo"),
            Regime(13.5, 15, 1.0, "clase, mal ventilada"),
            Regime(15.0, 15, 4.0, "se abren ventanas"),
            Regime(16.5, 0, 0.5, "vacío, cerrado"),
        ),
    )


def scenario_b() -> Scenario:
    """Quiebres dentro de una misma fase y escalones meseta a meseta."""
    return Scenario(
        key="B",
        title="Escenario B: quiebres dentro de una misma fase",
        duration_h=16.0,
        regimes=(
            Regime(0.0, 0, 1.0, "vacío"),
            Regime(1.0, 8, 2.0, "llegan 8"),
            Regime(1.5, 20, 2.0, "llegan 12 más"),
            Regime(3.5, 20, 3.5, "se abre la puerta"),
            Regime(5.0, 0, 4.0, "vacío, ventanas abiertas"),
            Regime(5.25, 0, 0.8, "se cierran ventanas"),
            Regime(8.0, 12, 1.5, "entran 12"),
            Regime(10.0, 6, 1.5, "se va la mitad"),
            Regime(12.0, 18, 1.5, "entran 12 más"),
            Regime(14.0, 0, 1.5, "vacío"),
        ),
    )


def scenario_c() -> Scenario:
    """Caso difícil: poca ventilación, eventos cortos, más ruido, cambios graduales."""
    return Scenario(
        key="C",
        title="Escenario C: ventilación baja, eventos cortos y más ruido",
        duration_h=24.0,
        ramp_min=5.0,
        c_ext_drift_ppm=15.0,
        sensor=Sensor(noise_sd=15.0, seed=2),
        regimes=(
            Regime(0.0, 0, 0.3, "vacío"),
            Regime(8.0, 6, 0.5, "reunión"),
            Regime(8.75, 0, 0.5, "vacío"),
            Regime(9.5, 8, 0.8, "reunión"),
            Regime(10.0, 3, 0.8, "quedan 3"),
            Regime(10.0 + 1.0 / 3.0, 5, 0.4, "trabajo, puerta cerrada"),
            Regime(12.0, 0, 0.8, "almuerzo"),
            Regime(13.0, 4, 0.6, "reunión chica"),
            Regime(13.5, 7, 0.6, "se suman 3"),
            Regime(14.25, 2, 0.6, "quedan 2"),
            Regime(15.0, 6, 0.3, "reunión larga, cerrado"),
            Regime(17.0, 0, 0.5, "vacío"),
        ),
    )


SCENARIOS = {"A": scenario_a, "B": scenario_b, "C": scenario_c}


def with_sensor(scn: Scenario, **changes) -> Scenario:
    """Copia del escenario con otro sensor (por ejemplo `noise_sd=0`)."""
    return replace(scn, sensor=replace(scn.sensor, **changes))
