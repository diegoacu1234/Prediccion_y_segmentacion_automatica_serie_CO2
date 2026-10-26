"""Simulación CFD (sección 5.2): sensores virtuales, promedios y ajuste por estado del esquema.

No hay verdad muestra a muestra: se conoce el esquema de ventilación (cuándo
se abre y se cierra) y la ACH de cada estado. Los métodos son los mismos del
estudio sintético con escalas para eventos de 5 min y una muestra por minuto.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .fitting import Fit, fit_bounded_source, fit_exponential, fit_hyun, fit_known_source
from .pipeline import Filters, apply_filters
from .regimes import Joiner, as_regimes
from .segmenters.base import DECAY, RISE, Segment, Segmentation
from .segmenters.derivative import DerivConfig, segment_derivative
from .segmenters.hmm import HmmConfig, segment_hmm
from .segmenters.hyun import segment_hyun, variant
from .segmenters.pelt_exp import PeltConfig, segment_pelt
from .signal_utils import despike, noise_sd

METHODS = ("M1", "M2", "M3a", "M3b", "M4", "M5", "M6", "M0")
HYUN_FAMILY = ("M1", "M2", "M3a", "M3b")
OPEN, CLOSED = "abierto", "cerrado"


# ------------------------------------------------------------------ datos


@dataclass
class CfdCase:
    """Las series de todos los sensores de una simulación."""

    name: str
    t_s: np.ndarray  # desde la primera muestra [s]
    t0_sim_s: float  # tiempo de simulación de la primera muestra; lo anterior es puesta en régimen
    data: np.ndarray  # (muestras, sensores) [ppm]
    sensors: pd.DataFrame  # sensor (desde 1), x, y, z, pared
    c_ext: float = 400.0

    @property
    def t_h(self) -> np.ndarray:
        return self.t_s / 3600.0

    @property
    def n(self) -> int:
        return len(self.t_s)

    @property
    def dt_h(self) -> float:
        return float(np.median(np.diff(self.t_s))) / 3600.0


@dataclass
class ObservedSeries:
    """Una serie medida (o simulada) con lo mínimo que necesitan los métodos."""

    name: str
    t_h: np.ndarray
    c_obs: np.ndarray
    c_ext_nominal: float = 400.0

    @property
    def n(self) -> int:
        return len(self.t_h)

    @property
    def dt_h(self) -> float:
        return float(np.median(np.diff(self.t_h)))


def _wall(x: float, y: float, size: tuple, margin: float = 0.35) -> str:
    """Pared más cercana si el punto está pegado a una; si no, "centro"."""
    dist = {"oeste": x, "este": size[0] - x, "sur": y, "norte": size[1] - y}
    name = min(dist, key=dist.get)
    return name if dist[name] <= margin else "centro"


def load_case(path, c_ext: float = 400.0, room_xy: tuple = (8.0, 6.0)) -> CfdCase:
    """Lee el .mat (v7.3, HDF5) que escribe `BuildTimeSeries_sensores.m`."""
    import h5py

    with h5py.File(path, "r") as f:
        r = f["results"]
        data = np.array(r["data"], dtype=float)  # HDF5 lo guarda traspuesto
        t_abs = np.array(r["time"], dtype=float).ravel()
        x, y, z = (np.array(r[k], dtype=float).ravel() for k in "xyz")
        name = "".join(chr(int(c)) for c in np.array(r["case/name"]).ravel())
    if data.shape[0] != len(t_abs):
        data = data.T
    sensors = pd.DataFrame({"sensor": np.arange(1, len(x) + 1), "x": x, "y": y, "z": z})
    sensors["pared"] = [_wall(a, b, room_xy) for a, b in zip(x, y)]
    return CfdCase(name=name, t_s=t_abs - t_abs[0], t0_sim_s=float(t_abs[0]), data=data, sensors=sensors, c_ext=c_ext)


def groups(case: CfdCase) -> dict:
    """Sensores que se promedian: todos, por altura y por pared."""
    s = case.sensors
    out = {"todos": np.arange(len(s))}
    for height in sorted(s["z"].unique()):
        out[f"z = {height:.1f} m".replace(".", ",")] = np.flatnonzero(np.isclose(s["z"], height))
    for wall in ("oeste", "este", "norte", "sur", "centro"):
        idx = np.flatnonzero(s["pared"] == wall)
        if len(idx):
            out[wall if wall == "centro" else f"pared {wall}"] = idx
    return out


def mean_series(case: CfdCase, columns, name: str) -> ObservedSeries:
    return ObservedSeries(name=name, t_h=case.t_h, c_obs=case.data[:, np.asarray(columns)].mean(axis=1), c_ext_nominal=case.c_ext)


def sensor_series(case: CfdCase, i: int) -> ObservedSeries:
    return ObservedSeries(name=f"sensor {i + 1}", t_h=case.t_h, c_obs=case.data[:, i].copy(), c_ext_nominal=case.c_ext)


# ------------------------------------------------------------------ esquema de ventilación


@dataclass(frozen=True)
class Schedule:
    """Cortes del esquema [min] y estado de cada tramo.

    `mixing_min`: después de cada cierre los sensores siguen bajando unos
    minutos; ese lapso no se ajusta (como la rampa del oráculo sintético).
    """

    edges_min: tuple
    states: tuple
    mixing_min: float = 0.0

    def regimes(self, t_h, mixing: bool = True) -> list:
        """(i0, i1, fase, estado, ciclo) de cada tramo; la muestra del corte va en los dos."""
        t_min = np.asarray(t_h) * 60.0
        half = 0.5 * float(np.median(np.diff(t_min)))
        out, cycle = [], {}
        for k, (a, b, state) in enumerate(zip(self.edges_min[:-1], self.edges_min[1:], self.states)):
            skip = self.mixing_min if (mixing and state == CLOSED and k > 0 and b - a > 3.0 * self.mixing_min) else 0.0
            idx = np.flatnonzero((t_min >= a + skip - half) & (t_min <= b + half))
            cycle[state] = cycle.get(state, 0) + 1
            out.append((int(idx[0]), int(idx[-1]) + 1, DECAY if state == OPEN else RISE, state, cycle[state]))
        return out

    def onsets(self) -> list:
        """(minuto, fase) de cada cambio interior."""
        return [(float(t), DECAY if state == OPEN else RISE) for t, state in zip(self.edges_min[1:-1], self.states[1:])]

    def state_at(self, t_min: float) -> int:
        """Índice del tramo que contiene el instante."""
        return int(np.clip(np.searchsorted(self.edges_min, t_min, side="right") - 1, 0, len(self.states) - 1))


def periodic_schedule(duration_min: float, period_min: float = 30.0, open_min: float = 5.0, mixing_min: float = 0.0) -> Schedule:
    """Abierto `open_min` al principio de cada período."""
    edges, states = [0.0], []
    t = 0.0
    while t < duration_min - 1e-9:
        for length, state in ((open_min, OPEN), (period_min - open_min, CLOSED)):
            t_next = min(t + length, duration_min)
            if t_next > t + 1e-9:
                edges.append(t_next)
                states.append(state)
            t = t_next
    return Schedule(tuple(edges), tuple(states), mixing_min)


def schedule_segmentation(series: ObservedSeries, schedule: Schedule, mixing: bool = True) -> Segmentation:
    """Los cortes del esquema como segmentación (hace de oráculo). Con mixing=False los cerrados incluyen la mezcla."""
    segments = [Segment(i0, i1, phase, note=f"{state} {cycle}") for i0, i1, phase, state, cycle in schedule.regimes(series.t_h, mixing)]
    labels = np.zeros(series.n, dtype=int)
    for s in segments:
        labels[s.i0 : s.i1] = s.phase
    return Segmentation(segments=segments, labels=labels, info={})


# ------------------------------------------------------------------ escalas de tiempo


@dataclass(frozen=True)
class Setup:
    """Escalas de tiempo de todos los métodos."""

    key: str
    label: str
    hyun: dict  # ventanas de Hyun, en muestras de esta serie
    hyun_d2_window_min: float
    hyun_kink_half_min: float
    deriv: DerivConfig
    pelt: PeltConfig
    hmm: HmmConfig
    min_duration_h: float
    min_r2: float = 0.1
    onset_tol_min: float = 2.0
    onset_tol_loose_min: float = 4.0  # admite el retardo de mezcla de los sensores altos

    @property
    def filters(self) -> Filters:
        return Filters(min_duration_h=self.min_duration_h, min_r2=self.min_r2)


def short_setup(dt_h: float) -> Setup:
    """Escalas para eventos de ~5 min con una muestra por minuto.

    Ventanas de 5 muestras, tramos de 3 a 5 muestras y sin escalar con el ruido
    (escalando, las ventanas quedarían más largas que el evento).
    """
    step = dt_h * 60.0
    return Setup(
        key="corta",
        label="escalas para eventos de 5 min",
        hyun={"median_window": 3, "rhl_window": 3, "min_samples": 3},
        hyun_d2_window_min=7.0 * step,
        hyun_kink_half_min=4.0 * step,
        deriv=DerivConfig(win_min=5.0 * step, min_dur_min=3.0 * step, gap_min=2.0 * step, kink_half_min=4.0 * step, snap_min=3.0 * step, adaptive=False),
        pelt=PeltConfig(min_size_min=4.0 * step, min_size_samples=5, jump_min=1.0 * step),
        hmm=HmmConfig(win_min=5.0 * step, dwell_min=12.0 * step, min_dur_min=3.0 * step, gap_min=2.0 * step, kink_half_min=4.0 * step, snap_min=3.0 * step, adaptive=False),
        min_duration_h=4.0 * step / 60.0,
    )


# ------------------------------------------------------------------ segmentación y ajuste


def segment(series: ObservedSeries, key: str, setup: Setup, schedule: Schedule | None = None, tramo: str = "TM") -> Segmentation:
    """Segmentación del método `key`. M0 son los cortes del esquema. Con "TM", régimen completo."""
    if key == "M0":
        return schedule_segmentation(series, schedule)
    seg = _segment_raw(series, key, setup)
    if tramo != "TM":
        return as_regimes(seg, [s for s in seg.transients if s.phase == RISE]) if key == "M1" else seg
    sigma = noise_sd(series.c_obs)
    c_clean, _ = despike(series.c_obs, sigma)
    joiner = Joiner(series.t_h, c_clean, sigma)
    return as_regimes(seg, joiner.regimes(seg, by_regime=key == "M5", rises_only=key == "M1"))


def _segment_raw(series: ObservedSeries, key: str, setup: Setup) -> Segmentation:
    t, c = series.t_h, series.c_obs
    if key in ("M1", "M2"):  # M1 ajusta sólo las subidas; las bajadas las delimitan
        return segment_hyun(t, c, variant(decays=True, **setup.hyun))
    if key == "M3a":
        return segment_hyun(t, c, variant(decays=True, d2_feature=True, d2_window_min=setup.hyun_d2_window_min, **setup.hyun))
    if key == "M3b":
        return segment_hyun(t, c, variant(decays=True, kink_split=True, kink_half_min=setup.hyun_kink_half_min, **setup.hyun))
    if key == "M4":
        return segment_derivative(t, c, setup.deriv)
    if key == "M5":
        return segment_pelt(t, c, setup.pelt)
    if key == "M6":
        return segment_hmm(t, c, setup.hmm)
    raise KeyError(key)


def fit_segments(series: ObservedSeries, key: str, seg: Segmentation, setup: Setup, source: float | None = None, bounds: tuple | None = None) -> list:
    """Ajusta los transitorios de `seg`: generación libre, conocida (`source` [ppm/h]) o acotada (`bounds` = (S mín, S máx)).

    Igual que en el estudio sintético: Hyun sobre la serie filtrada con C0 fijo, el resto sobre la serie sin picos.
    """
    t, c_ext = series.t_h, series.c_ext_nominal
    sigma = noise_sd(series.c_obs)
    c_clean, _ = despike(series.c_obs, sigma)
    hyun = key in HYUN_FAMILY
    data = seg.info["c_filt"] if hyun else c_clean
    fits = []
    for s in seg.transients:
        sl = slice(s.i0, s.i1)
        y = data[sl]
        if bounds is not None:
            f = fit_bounded_source(t[sl], y, c_ext=c_ext, s_min=float(bounds[0]), s_max=float(bounds[1]), c0=(y[0] if hyun else None), i0=s.i0, phase=s.phase)
        elif source is None:
            if hyun:
                f = fit_hyun(t[sl], y, c_ext=c_ext, c0=y[0], i0=s.i0, phase=s.phase)
            else:
                f = fit_exponential(t[sl], y, c_ext=c_ext, css_min=c_ext, i0=s.i0, phase=s.phase)
        else:
            f = fit_known_source(t[sl], y, c_ext=c_ext, s=source, c0=(y[0] if hyun else None), i0=s.i0, phase=s.phase)
        fits.append(apply_filters(f, y, c_ext, setup.filters, sigma))
    return fits


# ------------------------------------------------------------------ contra el esquema


def onset_metrics(series: ObservedSeries, seg: Segmentation, schedule: Schedule, tol_min: float = 2.0) -> dict:
    """Inicios detectados frente a los cambios del esquema (±tol_min, misma fase). Se compara en muestras."""
    t_min = series.t_h * 60.0
    step = series.dt_h * 60.0
    tol = int(round(tol_min / step))
    truth = [(int(np.argmin(np.abs(t_min - t))), phase) for t, phase in schedule.onsets()]
    pred = [(s.i0, s.phase) for s in seg.transients if s.i0 > 0]
    used, hits, errors = set(), [], []
    for i_true, phase in truth:
        cand = [(abs(i - i_true), j) for j, (i, ph) in enumerate(pred) if j not in used and ph == phase and abs(i - i_true) <= tol]
        if cand:
            d, j = min(cand)
            used.add(j)
            hits.append(True)
            errors.append((pred[j][0] - i_true) * step)
        else:
            hits.append(False)
            errors.append(np.nan)
    n_hit = int(np.sum(hits))
    precision = n_hit / len(pred) if pred else np.nan
    recall = n_hit / len(truth) if truth else np.nan
    return {
        "inicios_esquema": len(truth),
        "inicios_detectados": len(pred),
        "inicios_acertados": n_hit,
        "falsos": len(pred) - n_hit,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if n_hit else 0.0,
        "acierto": hits,
        "desfase_min": errors,
    }


def match_fits(series: ObservedSeries, fits: list, schedule: Schedule) -> dict:
    """{tramo del esquema: ajuste que le corresponde}: el aceptado de su misma fase que más se solapa.

    Bajada <-> abierto y subida <-> cerrado. No se pide que el ajuste caiga
    mayormente adentro: los sensores responden 2 a 4 min tarde. Los aceptados
    que sobran son estimaciones de más.
    """
    t_min = series.t_h * 60.0
    step = series.dt_h * 60.0
    edges = schedule.edges_min
    phases = [DECAY if state == OPEN else RISE for state in schedule.states]
    best = {}
    for j, f in enumerate(fits):
        if not (f.accepted and f.ok and np.isfinite(f.lam)):
            continue
        f0, f1 = float(t_min[f.i0]), float(t_min[f.i1 - 1])
        overlap = [max(0.0, min(f1, b) - max(f0, a)) if ph == f.phase else 0.0 for a, b, ph in zip(edges[:-1], edges[1:], phases)]
        k = int(np.argmax(overlap))
        if overlap[k] >= 0.9 * step and (k not in best or overlap[k] > best[k][1]):
            best[k] = (j, overlap[k])
    return {k: j for k, (j, _) in best.items()}


def regime_estimates(series: ObservedSeries, fits: list, schedule: Schedule) -> pd.DataFrame:
    """ACH de cada tramo del esquema, la del ajuste que le asigna `match_fits` (NaN si no hay)."""
    matched = match_fits(series, fits, schedule)
    rows, cycle = [], {}
    for k, state in enumerate(schedule.states):
        cycle[state] = cycle.get(state, 0) + 1
        f = fits[matched[k]] if k in matched else None
        rows.append(
            {
                "tramo": k,
                "estado": state,
                "ciclo": cycle[state],
                "t0_min": schedule.edges_min[k],
                "t1_min": schedule.edges_min[k + 1],
                "ach": f.lam if f is not None else np.nan,
                "ach_se": f.lam_se if f is not None else np.nan,
                "n_ajustes": int(f is not None),
            }
        )
    return pd.DataFrame(rows)


def fit_reference(series: ObservedSeries, fit: Fit, schedule: Schedule, truth: dict | None = None, tramo: int | None = None) -> dict:
    """Con qué estado se compara un ajuste: el de su tramo si está asignado; si no, el que más ocupa."""
    t_min = series.t_h * 60.0
    f0, f1 = float(t_min[fit.i0]), float(t_min[fit.i1 - 1])
    edges = schedule.edges_min
    overlap = [max(0.0, min(f1, b) - max(f0, a)) for a, b in zip(edges[:-1], edges[1:])]
    k = int(np.argmax(overlap)) if tramo is None else int(tramo)
    state = schedule.states[k]
    real = truth.get(state) if truth else None
    ok = fit.ok and np.isfinite(fit.lam)
    return {
        "tramo": k,
        "estado": state,
        "asignado": tramo is not None,
        "pureza": overlap[k] / (f1 - f0) if f1 > f0 else 1.0,
        "ach_real": real,
        "error": (fit.lam / real - 1.0) if (ok and real) else np.nan,
    }


def fits_table(res: "CfdResults", truth: dict | None = None) -> pd.DataFrame:
    """Una fila por ajuste, con el estado que le toca y, si se conoce, el error."""
    rows = []
    for (name, sk, key, gen), fits in res.fits.items():
        ser = res.series[name]
        t_min = ser.t_h * 60.0
        owner = {j: k for k, j in match_fits(ser, fits, res.schedule).items()}
        for j, f in enumerate(fits):
            ref = fit_reference(ser, f, res.schedule, truth, owner.get(j))
            rows.append(
                {
                    "serie": name,
                    "tipo": res.kind[name],
                    "escalas": sk,
                    "metodo": key,
                    "gen": gen,
                    "t0_min": t_min[f.i0],
                    "t1_min": t_min[f.i1 - 1],
                    "fase": "subida" if f.phase == RISE else "bajada",
                    "aceptado": bool(f.accepted),
                    "motivo": f.reason,
                    "asignado": ref["asignado"],
                    "ach": f.lam if f.ok else np.nan,
                    "ach_se": f.lam_se if f.ok else np.nan,
                    "c_inf": f.css if f.ok else np.nan,
                    "rmse": f.rmse if f.ok else np.nan,
                    "estado": ref["estado"],
                    "pureza": ref["pureza"],
                    "ach_real": ref["ach_real"] if ref["ach_real"] is not None else np.nan,
                    "error": ref["error"],
                }
            )
    return pd.DataFrame(rows)


def state_means(table: pd.DataFrame) -> dict:
    """ACH de cada estado, promedio de sus ciclos."""
    return {state: float(g["ach"].mean()) if g["ach"].notna().any() else np.nan for state, g in table.groupby("estado")}


# ------------------------------------------------------------------ generación


def source_scan(series: ObservedSeries, schedule: Schedule, sources, setup: Setup, mixing: bool = True) -> pd.DataFrame:
    """ACH de cada estado y RMSE para cada fuente supuesta, con los cortes del esquema."""
    seg = schedule_segmentation(series, schedule, mixing)
    rows = []
    for s in sources:
        fits = fit_segments(series, "M0", seg, setup, source=float(s))
        est = regime_estimates(series, fits, schedule)
        sse = sum(f.sse for f in fits if f.ok)
        n = sum(f.n for f in fits if f.ok)
        row = {"fuente_ppm_h": float(s), "rmse": float(np.sqrt(sse / n)) if n else np.nan}
        row.update({f"ach_{state}": value for state, value in state_means(est).items()})
        rows.append(row)
    out = pd.DataFrame(rows)
    out["delta_ach"] = out[f"ach_{OPEN}"] - out[f"ach_{CLOSED}"]
    return out


def calibrated_source(series: ObservedSeries, schedule: Schedule, setup: Setup, closed_ach: float, bracket=(300.0, 30000.0)) -> float:
    """Fuente con la que el estado cerrado da `closed_ach` (la "generación calibrada"). NaN si no hay solución."""

    def gap(src: float) -> float:
        lam = float(source_scan(series, schedule, [src], setup)[f"ach_{CLOSED}"].iloc[0])
        return (lam if np.isfinite(lam) else 0.0) - closed_ach

    lo, hi = bracket
    if gap(lo) > 0 or gap(hi) < 0:
        return np.nan
    return float(brentq(gap, lo, hi, xtol=1.0))


def assumption_table(series: ObservedSeries, schedule: Schedule, setup: Setup, sources: dict) -> pd.DataFrame:
    """ACH de cada estado con la generación libre y con cada fuente de `sources` = {clave: (nombre, S)} (tabla 5.3)."""
    seg = schedule_segmentation(series, schedule)
    means = state_means(regime_estimates(series, fit_segments(series, "M0", seg, setup), schedule))
    rows = [{"clave": "libre", "generacion": "libre", "fuente_ppm_h": np.nan, **{f"ach_{k}": v for k, v in means.items()}}]
    for key, (label, src) in sources.items():
        if src is None or not np.isfinite(src):
            continue
        means = state_means(regime_estimates(series, fit_segments(series, "M0", seg, setup, float(src)), schedule))
        rows.append({"clave": key, "generacion": label, "fuente_ppm_h": float(src), **{f"ach_{k}": v for k, v in means.items()}})
    out = pd.DataFrame(rows)
    for state in (OPEN, CLOSED):
        if f"ach_{state}" not in out:
            out[f"ach_{state}"] = np.nan
    out["delta_ach"] = out[f"ach_{OPEN}"] - out[f"ach_{CLOSED}"]
    out.insert(0, "serie", series.name)
    return out


def add_errors(table: pd.DataFrame, truth: dict | None) -> pd.DataFrame:
    """Agrega el error relativo frente a la ACH de la simulación."""
    out = table.copy()
    if not truth:
        return out
    ach_open, ach_closed = truth.get(OPEN), truth.get(CLOSED)
    delta = ach_open - ach_closed if (ach_open is not None and ach_closed is not None) else None
    for col, name, value in ((f"ach_{CLOSED}", "error_cerrado", ach_closed), (f"ach_{OPEN}", "error_abierto", ach_open), ("delta_ach", "error_delta", delta)):
        if value is not None and col in out:
            out[name] = out[col] / value - 1.0
    return out


# ------------------------------------------------------------------ corrida completa


@dataclass
class CfdResults:
    case: CfdCase
    schedule: Schedule
    setups: dict
    source: float
    series: dict  # nombre -> ObservedSeries
    kind: dict  # nombre -> "sensor" | "promedio"
    segmentations: dict = field(default_factory=dict)  # (serie, escalas, método) -> Segmentation
    fits: dict = field(default_factory=dict)  # (serie, escalas, método, gen) -> [Fit]
    onsets: pd.DataFrame | None = None
    regimes: pd.DataFrame | None = None
    segments: pd.DataFrame | None = None
    states: pd.DataFrame | None = None  # ACH de cada estado y la diferencia


def run_case(case: CfdCase, schedule: Schedule, source: float, setups=None, methods=METHODS, bounds: tuple | None = None) -> CfdResults:
    """Todos los métodos sobre cada sensor y cada promedio, con la generación libre, conocida (`source`) y acotada (`bounds`)."""
    setups = setups or {"corta": short_setup(case.dt_h)}
    series, kind = {}, {}
    for name, idx in groups(case).items():
        series[name] = mean_series(case, idx, name)
        kind[name] = "promedio"
    for i in range(case.data.shape[1]):
        s = sensor_series(case, i)
        series[s.name] = s
        kind[s.name] = "sensor"

    res = CfdResults(case=case, schedule=schedule, setups=setups, source=source, series=series, kind=kind)
    onset_rows, regime_rows, segment_rows = [], [], []
    t_min = case.t_h * 60.0
    for name, ser in series.items():
        sigma = noise_sd(ser.c_obs)
        for sk, setup in setups.items():
            for key in methods:
                seg = segment(ser, key, setup, schedule)
                res.segmentations[(name, sk, key)] = seg
                for s in seg.transients:
                    segment_rows.append({"serie": name, "escalas": sk, "metodo": key, "t0_min": t_min[s.i0], "t1_min": t_min[s.i1 - 1], "fase": s.phase})
                marks = schedule_segmentation(ser, schedule, mixing=False) if key == "M0" else seg  # para M0 los cortes son la referencia
                om = onset_metrics(ser, marks, schedule, setup.onset_tol_min)
                loose = onset_metrics(ser, marks, schedule, setup.onset_tol_loose_min)
                row = {"serie": name, "tipo": kind[name], "escalas": sk, "metodo": key, "sigma": sigma}
                row.update({k: v for k, v in om.items() if k not in ("acierto", "desfase_min")})
                row.update({"f1_amplio": loose["f1"], "recall_amplio": loose["recall"], "precision_amplio": loose["precision"]})
                for j, (t_true, phase) in enumerate(schedule.onsets()):
                    row[f"acierto_{t_true:g}"] = om["acierto"][j]
                    row[f"acierto_amplio_{t_true:g}"] = loose["acierto"][j]
                    row[f"desfase_{t_true:g}"] = loose["desfase_min"][j]
                onset_rows.append(row)
                gens = [("libre", None, None), ("conocida", source, None)] + ([("acotada", None, bounds)] if bounds is not None else [])
                for gen, src, lims in gens:
                    fits = fit_segments(ser, key, seg, setup, src, lims)
                    res.fits[(name, sk, key, gen)] = fits
                    est = regime_estimates(ser, fits, schedule)
                    est.insert(0, "gen", gen)
                    est.insert(0, "metodo", key)
                    est.insert(0, "escalas", sk)
                    est.insert(0, "tipo", kind[name])
                    est.insert(0, "serie", name)
                    regime_rows.append(est)
    res.onsets = pd.DataFrame(onset_rows)
    res.regimes = pd.concat(regime_rows, ignore_index=True)
    res.segments = pd.DataFrame(segment_rows)
    res.states = state_table(res.regimes)
    return res


def state_table(regimes: pd.DataFrame) -> pd.DataFrame:
    """ACH de cada estado (media de los ciclos) y su diferencia, por serie, escalas, método y generación."""
    keys = ["serie", "tipo", "escalas", "metodo", "gen"]
    wide = regimes.groupby(keys + ["estado"], sort=False)["ach"].mean().unstack("estado").reset_index()
    wide = wide.rename(columns={OPEN: f"ach_{OPEN}", CLOSED: f"ach_{CLOSED}"})
    wide["delta_ach"] = wide[f"ach_{OPEN}"] - wide[f"ach_{CLOSED}"]
    return wide


def opening_dilution(case: CfdCase, schedule: Schedule, minutes: float = 3.0, horizon_min: float = 12.0) -> pd.DataFrame:
    """Cómo le llega a cada sensor el aire de las aperturas, frente a la media de la sala (figura 5.11).

    - dilucion_rel: fracción del exceso que pierde en los primeros `minutes`,
      sobre la que pierde la media (> 1: le llega antes que al resto);
    - t_minimo_min: minutos hasta su mínimo, dentro de `horizon_min`;
    - caida_rel: caída hasta ese mínimo sobre la de la media;
    - exceso_rel: CO2 medio del sensor menos el de la sala [ppm].
    """
    t = case.t_s / 60.0
    room = case.data.mean(axis=1)
    opens = [t0 for t0, state in zip(schedule.edges_min[:-1], schedule.states) if state == OPEN]
    rows = []
    for i in range(case.data.shape[1]):
        c = case.data[:, i]
        dil, tmin, drop = [], [], []
        for t0 in opens:
            k0 = int(np.argmin(np.abs(t - t0)))
            k1 = int(np.argmin(np.abs(t - (t0 + minutes))))
            win = (t >= t0) & (t <= t0 + horizon_min)
            frac = (c[k0] - c[k1]) / (c[k0] - case.c_ext)
            frac_room = (room[k0] - room[k1]) / (room[k0] - case.c_ext)
            dil.append(frac / frac_room)
            j = int(np.argmin(c[win]))
            tmin.append(float(t[win][j] - t0))
            drop.append((c[k0] - c[win][j]) / (room[k0] - room[win].min()))
        rows.append({"sensor": int(case.sensors["sensor"].iloc[i]), "dilucion_rel": float(np.mean(dil)), "t_minimo_min": float(np.mean(tmin)), "caida_rel": float(np.mean(drop)), "exceso_rel": float(np.mean(c - room))})
    return pd.DataFrame(rows)
