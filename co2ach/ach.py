"""Segmentación y ACH: la interfaz pública (junta `pipeline`, `mediciones` y `ventilacion`).

- `read_room`: lecturas de un salón desde los CSV mensuales.
- `segment`: una serie cualquiera.
- `segment_day`: un día de un salón, como en el cap. 6.
- `analyze_room`: todos los días de un salón, con horario y generación si se conocen.
- `fit_event`: el ajuste de un intervalo elegido a mano.
- `daily_ach`: ACH de cada día, ponderada por duración.
- `plot_day`: la figura de un día.

Entran tablas con una columna de tiempo y una de CO2 [ppm]; salen tablas con una
fila por ajuste. Todos los métodos ajustan el régimen completo y usan el
criterio de reporte de la tesis (`Selection`). Detalle en `docs/referencia.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import mediciones as med
from .cfd import ObservedSeries
from .fitting import fit_bounded_source, fit_exponential, fit_known_source
from .mediciones import Group, Occupancy, Selection
from .pipeline import run_all
from .salones import CLASE, HORARIOS
from .signal_utils import despike, noise_sd

__all__ = [
    "METHODS",
    "DEFAULT_METHODS",
    "THESIS_CLASS",
    "THESIS_SCHEDULE",
    "Group",
    "Occupancy",
    "Selection",
    "SegmentationResult",
    "RoomAnalysis",
    "read_room",
    "prepare_readings",
    "daily_table",
    "segment",
    "segment_day",
    "analyze_room",
    "fit_event",
    "daily_ach",
    "class_source",
    "plot_day",
]

# nombre público -> clave interna
METHODS = {
    "hyun": "M1",  # Hyun fiel: sólo subidas
    "hyun_decaimientos": "M2",
    "hyun_c2": "M3a",
    "hyun_quiebres": "M3b",
    "derivada": "M4",
    "pelt": "M5",
    "hmm": "M6",
}
DEFAULT_METHODS = ("hmm",)
_NAME = {v: k for k, v in METHODS.items()}

# casos de generación y horario del salón A02 (cap. 6)
THESIS_CLASS = dict(CLASE)  # "bajo", "medio", "alto"
THESIS_SCHEDULE = Occupancy.parse(HORARIOS[("Escuelas32y83", 2)])

EVENT_COLUMNS = [
    "metodo", "fecha", "inicio", "fin", "t0_h", "t1_h", "dur_min", "fase", "aceptado", "motivo",
    "ach", "ach_se", "c0", "c_inf", "c_ext", "generacion_ppm_s", "excursion_ppm", "r2", "rmse",
    "rmse_rel", "lam_t", "se_rel", "evento", "reportado",
]  # fmt: skip


# ------------------------------------------------------------------ entradas


def read_room(folder, ip: str, sensor_id: int, cache_dir=None) -> pd.DataFrame:
    """Lecturas del salón (`ip`, `sensor_id`): `tiempo`, `co2`, `temp`, `hum`, ordenadas.

    Con `cache_dir` guarda un parquet y lo reutiliza (necesita pyarrow).
    """
    return med.load_room(folder, ip, int(sensor_id), cache_dir)


def prepare_readings(readings, time_col: str = "tiempo", co2_col: str = "co2") -> pd.DataFrame:
    """Lleva las lecturas al formato de la librería: `tiempo` (datetime) y `co2` [ppm], ordenadas.

    Acepta una tabla (con otros nombres de columna vía `time_col`, `co2_col`) o
    una serie con índice de tiempo. Conserva `temp` y `hum`; saca lecturas sin
    CO2 y tiempos repetidos.
    """
    if isinstance(readings, pd.Series):
        df = pd.DataFrame({"tiempo": readings.index, "co2": readings.to_numpy()})
    else:
        df = readings.rename(columns={time_col: "tiempo", co2_col: "co2"})
        if "tiempo" not in df and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(names="tiempo")
    missing = {"tiempo", "co2"} - set(df.columns)
    if missing:
        raise ValueError(f"faltan columnas: {sorted(missing)}; se esperan una de tiempo y una de CO2")
    keep = [c for c in ("tiempo", "co2", "temp", "hum") if c in df]
    df = df[keep].copy()
    df["tiempo"] = pd.to_datetime(df["tiempo"])
    df["co2"] = pd.to_numeric(df["co2"], errors="coerce")
    df = df.dropna(subset=["tiempo", "co2"]).sort_values("tiempo").drop_duplicates("tiempo")
    return df.reset_index(drop=True)


def daily_table(readings, max_gap_min: float = 10.0, min_range_ppm: float = 150.0) -> pd.DataFrame:
    """Una fila por día: lecturas, mayor hueco, rango, CO2 exterior (`c_ext`), `completo` y `ocupado`."""
    return med.daily_table(prepare_readings(readings), max_gap_min=max_gap_min, min_range_ppm=min_range_ppm)


def _codes(methods) -> list:
    if isinstance(methods, str):
        methods = tuple(METHODS) if methods == "todos" else (methods,)
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"métodos desconocidos: {unknown}; las opciones son {list(METHODS)} o 'todos'")
    return [METHODS[m] for m in methods]


def _floor(series: ObservedSeries, noise_floor) -> float:
    """Piso del ruido: el efectivo, un número o ninguno."""
    if noise_floor is None:
        return 0.0
    if isinstance(noise_floor, str):
        if noise_floor != "efectivo":
            raise ValueError("noise_floor tiene que ser 'efectivo', un número en ppm o None")
        value = med.effective_noise(series)
        return float(value) if np.isfinite(value) else 0.0
    return float(noise_floor)


# ------------------------------------------------------------------ una serie


@dataclass
class SegmentationResult:
    """Lo que devuelven `segment` y `segment_day`.

    events: un ajuste por fila; segments: los tramos detectados; series: la
    serie en la grilla regular; start: el instante de t_h = 0 (None si no había
    fechas); noise_sensor y noise_floor [ppm].
    """

    events: pd.DataFrame
    segments: pd.DataFrame
    series: ObservedSeries
    start: pd.Timestamp | None
    noise_sensor: float
    noise_floor: float
    selection: Selection = field(default_factory=Selection)
    _fits: pd.DataFrame | None = field(default=None, repr=False)  # con las claves internas
    _segs: pd.DataFrame | None = field(default=None, repr=False)

    def reported(self) -> pd.DataFrame:
        """Los eventos que cumplen el criterio de reporte."""
        return self.events[self.events["reportado"]].reset_index(drop=True)

    def plot(self, **kwargs):
        return plot_day(self, **kwargs)


def _time_of(start, t_h):
    if start is None:
        return np.full(len(t_h), pd.NaT)
    return pd.to_datetime(start) + pd.to_timedelta(np.asarray(t_h, dtype=float) * 3600.0, unit="s").round("s")


def _public_events(fits: pd.DataFrame, start, c_ext, sel: Selection) -> pd.DataFrame:
    """Tabla interna de ajustes -> nombres públicos y medidas de calidad."""
    if fits.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    ev = fits.copy()
    ev["metodo"] = ev["metodo"].map(_NAME)
    ev["inicio"] = _time_of(start, ev["t0_h"])
    ev["fin"] = _time_of(start, ev["t1_h"])
    if "c_ext" not in ev:
        ev["c_ext"] = float(c_ext)
    ev["generacion_ppm_s"] = ev["ach"] * (ev["c_inf"] - ev["c_ext"]) / 3600.0
    ev["rmse_rel"] = ev["rmse"] / ev["excursion_ppm"]
    ev["lam_t"] = ev["ach"] * ev["dur_min"] / 60.0
    ev["se_rel"] = ev["ach_se"] / ev["ach"]
    ev["evento"] = ev["aceptado"] & (ev["excursion_ppm"] >= sel.min_excursion_ppm)
    ev["reportado"] = ev["evento"] & (ev["rmse_rel"] <= sel.max_rmse_rel) & (ev["lam_t"] >= sel.min_lam_t)
    extra = [c for c in ev.columns if c not in EVENT_COLUMNS and c not in ("i0", "i1", "config", "asintota_en_c_ext")]
    return ev[EVENT_COLUMNS + extra].reset_index(drop=True)


def _public_segments(segs: pd.DataFrame, start) -> pd.DataFrame:
    if segs.empty:
        return pd.DataFrame(columns=["metodo", "fecha", "inicio", "fin", "t0_h", "t1_h", "fase"])
    out = segs.copy()
    out["metodo"] = out["metodo"].map(_NAME)
    out["inicio"] = _time_of(start, out["t0_h"])
    out["fin"] = _time_of(start, out["t1_h"])
    return out[["metodo", "fecha", "inicio", "fin", "t0_h", "t1_h", "fase"]].reset_index(drop=True)


def _run(series: ObservedSeries, codes, noise_floor, tramo: str):
    """Corre los métodos: (ajustes, tramos, ruido del sensor, piso)."""
    np.seterr(all="ignore")
    floor = _floor(series, noise_floor)
    results = run_all(series, keys=codes, sigma_min=floor, tramo=tramo)
    fits = pd.DataFrame(med.fit_rows(series, results))
    segs = pd.DataFrame(med.segment_rows(series, results))
    return fits, segs, noise_sd(series.c_obs), floor


def segment(
    times,
    co2,
    c_ext: float | None = None,
    methods=DEFAULT_METHODS,
    noise_floor="efectivo",
    dt_s: float = 30.0,
    selection: Selection = Selection(),
    name: str = "serie",
) -> SegmentationResult:
    """Segmenta una serie y ajusta la exponencial en cada tramo.

    `times`: fechas o horas desde el comienzo; `co2` [ppm]. Se interpola a una
    grilla de `dt_s` s. Sin `c_ext` se usa el percentil 1 de la serie (tiene
    sentido sólo si incluye ratos con el aula vacía). `noise_floor`:
    "efectivo" (como en la tesis), un valor en ppm o None.
    """
    t = pd.Series(np.asarray(times))
    c = np.asarray(co2, dtype=float)
    if len(t) != len(c):
        raise ValueError("times y co2 tienen que tener el mismo largo")
    ok = np.isfinite(c) & t.notna().to_numpy()
    if np.issubdtype(t.dtype, np.datetime64) or isinstance(t.iloc[0], (pd.Timestamp, np.datetime64)):
        ts = pd.to_datetime(t[ok]).to_numpy()
        start = pd.Timestamp(ts.min())
        hours = (ts - ts.min()) / np.timedelta64(1, "s") / 3600.0
    else:
        start = None
        hours = t[ok].to_numpy(dtype=float)
    c = c[ok]
    order = np.argsort(hours)
    hours, c = hours[order], c[order]
    if len(c) < 10:
        raise ValueError("la serie tiene menos de 10 lecturas")
    grid = np.arange(0.0, hours[-1] - hours[0] + 1e-9, dt_s / 3600.0)
    values = np.interp(grid + hours[0], hours, c)
    if start is not None and hours[0] != 0.0:
        start = start + pd.to_timedelta(hours[0] * 3600.0, unit="s")
    ce = float(np.percentile(values, 1)) if c_ext is None else float(c_ext)
    series = ObservedSeries(name=name, t_h=grid, c_obs=values, c_ext_nominal=ce)
    fits, segs, sensor, floor = _run(series, _codes(methods), noise_floor, "TM")
    return SegmentationResult(
        events=_public_events(fits, start, ce, selection),
        segments=_public_segments(segs, start),
        series=series,
        start=start,
        noise_sensor=sensor,
        noise_floor=floor,
        selection=selection,
        _fits=fits,
        _segs=segs,
    )


def segment_day(readings, day, c_ext: float | None = None, methods=DEFAULT_METHODS, noise_floor="efectivo", selection: Selection = Selection()) -> SegmentationResult:
    """Segmenta un día (0 a 24 h) de un salón, en una grilla de 30 s.

    Conviene pasar también los días de alrededor: C_ext es el mínimo semanal del
    percentil 1 diario (`daily_table`). Con `c_ext` se usa ese valor.
    """
    df = prepare_readings(readings)
    day = pd.Timestamp(day).normalize()
    if c_ext is None:
        days = med.daily_table(df)
        if day not in days.index or not np.isfinite(days.loc[day, "c_ext"]):
            raise ValueError(f"no hay lecturas del {day:%Y-%m-%d}")
        c_ext = float(days.loc[day, "c_ext"])
    series = med.day_series(df, day, c_ext)
    fits, segs, sensor, floor = _run(series, _codes(methods), noise_floor, "TM")
    return SegmentationResult(
        events=_public_events(fits, day, c_ext, selection),
        segments=_public_segments(segs, day),
        series=series,
        start=day,
        noise_sensor=sensor,
        noise_floor=floor,
        selection=selection,
        _fits=fits,
        _segs=segs,
    )


# ------------------------------------------------------------------ un salón entero


@dataclass
class RoomAnalysis:
    """Lo que devuelve `analyze_room`.

    events y segments: como en `SegmentationResult`, más las columnas de los
    reajustes si se dio horario o generación; days: la tabla diaria con el
    ruido; daily: `daily_ach`; blocks: ACH por día y bloque del horario (si hay
    horario y generación); cases: S [ppm/h] de cada caso.
    """

    events: pd.DataFrame
    segments: pd.DataFrame
    days: pd.DataFrame
    daily: pd.DataFrame
    blocks: pd.DataFrame | None
    cases: dict
    occupancy: Occupancy | None
    readings: pd.DataFrame = field(repr=False)
    selection: Selection = field(default_factory=Selection)
    _fits: pd.DataFrame | None = field(default=None, repr=False)
    _segs: pd.DataFrame | None = field(default=None, repr=False)

    def plot_day(self, day, **kwargs):
        return plot_day(self, day=day, **kwargs)


def _analyze_day(name, t_h, c_obs, c_ext, codes, noise_floor):
    series = ObservedSeries(name=name, t_h=np.asarray(t_h), c_obs=np.asarray(c_obs), c_ext_nominal=float(c_ext))
    fits, segs, sensor, floor = _run(series, codes, noise_floor, "TM")
    return fits, segs, {"fecha": name, "ruido_sensor": sensor, "piso_ruido": floor}


def analyze_room(
    readings,
    methods=DEFAULT_METHODS,
    days=None,
    occupancy: Occupancy | None = None,
    groups: dict | None = None,
    volume_m3: float | None = None,
    noise_floor="efectivo",
    selection: Selection = Selection(),
    n_jobs: int = 1,
) -> RoomAnalysis:
    """Segmenta todos los días de un salón y estima la ACH de cada evento y de cada día.

    Sin `days`: los días con actividad y sin huecos de más de 10 min.
    Con `occupancy` (horario vacío), las bajadas con ≥ 80 % en horario vacío se
    reajustan con la asíntota en C_ext (`*_puro`). Con `groups` y `volume_m3`
    además, los tramos con la clase adentro se reajustan con cada caso
    (`*_<caso>`) y con la generación acotada entre el menor y el mayor
    (`*_acotada`). `n_jobs` reparte los días entre procesos.
    """
    df = prepare_readings(readings)
    codes = _codes(methods)
    table = med.daily_table(df)
    if days is None:
        chosen = table[table["completo"] & table["ocupado"]].index
    else:
        chosen = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in days])
        missing = [d for d in chosen if d not in table.index]
        if missing:
            raise ValueError(f"no hay lecturas de {[d.strftime('%Y-%m-%d') for d in missing]}")
    tasks = []
    for day in chosen:
        s = med.day_series(df, day, table.loc[day, "c_ext"])
        tasks.append((s.name, s.t_h, s.c_obs, s.c_ext_nominal, codes, noise_floor))
    if n_jobs == 1:
        out = [_analyze_day(*task) for task in tasks]
    else:
        from joblib import Parallel, delayed

        out = Parallel(n_jobs=n_jobs)(delayed(_analyze_day)(*task) for task in tasks)
    fits = pd.concat([o[0] for o in out], ignore_index=True) if out else pd.DataFrame()
    segs = pd.concat([o[1] for o in out], ignore_index=True) if out else pd.DataFrame()
    info = pd.DataFrame([o[2] for o in out]).set_index("fecha") if out else pd.DataFrame(columns=["ruido_sensor", "piso_ruido"])
    noise = info["piso_ruido"].to_dict()

    cases = {}
    if groups and not occupancy:
        raise ValueError("los casos de generación necesitan también el horario (occupancy)")
    if occupancy is not None and not fits.empty:
        fits = med.refit_empty(df, table, fits, noise, occupancy)
        if groups:
            if volume_m3 is None:
                raise ValueError("con los casos de generación hace falta volume_m3")
            cases = {name: g.source(volume_m3) for name, g in groups.items()}
            cases = dict(sorted(cases.items(), key=lambda kv: kv[1]))
            fits = med.refit_occupied(df, table, fits, noise, occupancy, cases)
    if not fits.empty:
        fits["c_ext"] = pd.to_datetime(fits["fecha"]).map(table["c_ext"]).to_numpy()

    events = _public_events(fits, None, np.nan, selection)
    if not events.empty:
        base = pd.to_datetime(events["fecha"])
        events["inicio"] = base + pd.to_timedelta(events["t0_h"] * 3600.0, unit="s").round("s")
        events["fin"] = base + pd.to_timedelta(events["t1_h"] * 3600.0, unit="s").round("s")
    segments = _public_segments(segs, None)
    if not segments.empty:
        base = pd.to_datetime(segments["fecha"])
        segments["inicio"] = base + pd.to_timedelta(segments["t0_h"] * 3600.0, unit="s").round("s")
        segments["fin"] = base + pd.to_timedelta(segments["t1_h"] * 3600.0, unit="s").round("s")

    days_out = table.loc[chosen].join(info.set_index(pd.to_datetime(info.index)))
    blocks = None
    if occupancy is not None and len(cases) >= 2 and not fits.empty:
        from .ventilacion import consensus_ach, day_blocks

        ev_int = med.events(fits, selection)
        blocks = consensus_ach(ev_int, day_blocks(occupancy), list(cases), by="bloque")
    return RoomAnalysis(
        events=events,
        segments=segments,
        days=days_out,
        daily=daily_ach(events),
        blocks=blocks,
        cases=cases,
        occupancy=occupancy,
        readings=df,
        selection=selection,
        _fits=fits,
        _segs=segs,
    )


# ------------------------------------------------------------------ ACH del día


def daily_ach(events: pd.DataFrame, only_reported: bool = True, column: str = "ach") -> pd.DataFrame:
    """ACH de cada día y fase: promedio de los eventos ponderado por duración.

    Una fila por fecha, fase y método, y otra con metodo = "todos" (todos los
    métodos juntos). `column`: "ach" o la de un reajuste ("ach_medio",
    "ach_puro", ...). Columnas: fecha, fase, metodo, ach, eventos, minutos.
    """
    cols = ["fecha", "fase", "metodo", "ach", "eventos", "minutos"]
    if events.empty:
        return pd.DataFrame(columns=cols)
    ev = events[events["evento"]]
    if only_reported:
        ev = ev[ev["reportado"]]
    ev = ev[np.isfinite(ev[column].to_numpy(dtype=float))]
    if ev.empty:
        return pd.DataFrame(columns=cols)
    ev = ev.assign(_w=ev["dur_min"].astype(float), _wx=ev["dur_min"].astype(float) * ev[column].astype(float))
    parts = [ev, ev.assign(metodo="todos")]
    g = pd.concat(parts).groupby(["fecha", "fase", "metodo"], sort=True)
    out = pd.DataFrame({"ach": g["_wx"].sum() / g["_w"].sum(), "eventos": g.size(), "minutos": g["_w"].sum()}).reset_index()
    return out[cols]


# ------------------------------------------------------------------ un intervalo elegido a mano


def class_source(group: Group, volume_m3: float, temp_c: float = 20.0) -> float:
    """S = G/V de una clase [ppm/h] (Persily y de Jonge)."""
    return group.source(volume_m3, temp_c)


def fit_event(readings, start, end, c_ext: float, model: str = "libre", source_ppm_h: float | None = None, bounds_ppm_h: tuple | None = None, dt_s: float = 30.0) -> dict:
    """Ajusta la exponencial entre `start` y `end`.

    model: "libre" (C0, C∞ ≥ C_ext y ACH libres), "vacio" (asíntota en c_ext),
    "conocida" (con `source_ppm_h`) o "acotada" (S entre los dos valores de
    `bounds_ppm_h`). Devuelve ach [1/h], ach_se, c0, c_inf, fuente_ppm_h, r2,
    rmse, rmse_rel, lam_t, dur_min y en_cota ("inferior"/"superior" si S quedó
    en una cota, "lambda" si la ACH quedó en la del optimizador).
    """
    df = prepare_readings(readings)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    sub = df[(df["tiempo"] >= t0) & (df["tiempo"] <= t1)]
    if len(sub) < 5:
        raise ValueError("el intervalo tiene menos de 5 lecturas")
    sec = (sub["tiempo"] - t0).dt.total_seconds().to_numpy()
    grid = np.arange(0.0, sec[-1] + 1e-9, dt_s)
    y = np.interp(grid, sec, sub["co2"].to_numpy(dtype=float))
    y, _ = despike(y, noise_sd(y))
    t_h = grid / 3600.0
    if model == "libre":
        f = fit_exponential(t_h, y, c_ext=c_ext, css_min=c_ext)
    elif model == "vacio":
        f = fit_exponential(t_h, y, c_ext=c_ext, css=c_ext)
    elif model == "conocida":
        if source_ppm_h is None:
            raise ValueError("model='conocida' necesita source_ppm_h")
        f = fit_known_source(t_h, y, c_ext=c_ext, s=float(source_ppm_h))
    elif model == "acotada":
        if bounds_ppm_h is None or len(bounds_ppm_h) != 2:
            raise ValueError("model='acotada' necesita bounds_ppm_h = (mínimo, máximo)")
        f = fit_bounded_source(t_h, y, c_ext=c_ext, s_min=float(min(bounds_ppm_h)), s_max=float(max(bounds_ppm_h)))
    else:
        raise ValueError("model tiene que ser 'libre', 'vacio', 'conocida' o 'acotada'")
    ends = f.predict(t_h[[0, -1]]) if f.ok else np.array([np.nan, np.nan])
    excursion = abs(float(ends[1] - ends[0]))
    dur_min = 60.0 * float(t_h[-1])
    return {
        "ach": f.lam if f.ok else np.nan,
        "ach_se": f.lam_se,
        "c0": f.c0,
        "c_inf": f.css,
        "fuente_ppm_h": f.lam * (f.css - c_ext) if f.ok else np.nan,
        "r2": f.r2,
        "rmse": f.rmse,
        "rmse_rel": f.rmse / excursion if excursion > 0 else np.nan,
        "lam_t": f.lam * dur_min / 60.0 if f.ok else np.nan,
        "dur_min": dur_min,
        "en_cota": "lambda" if f.at_bound else f.s_bound,
    }


# ------------------------------------------------------------------ figura


def plot_day(result, day=None, path=None, hours=(7.0, 21.0)):
    """Figura de un día: la serie, una fila por método con sus tramos y curvas, y todas las ACH juntas.

    `result`: un `SegmentationResult` o un `RoomAnalysis` (con `day`). Con
    `path` la guarda (PNG o PDF). Devuelve la figura.
    """
    import matplotlib.pyplot as plt

    from . import plots_exp as px
    from .plots_med import fig_day_compact

    px.apply_pub_style()
    if isinstance(result, RoomAnalysis):
        if day is None:
            raise ValueError("con un RoomAnalysis hay que decir qué día (day=...)")
        name = pd.Timestamp(day).strftime("%Y-%m-%d")
        fits = result._fits[result._fits["fecha"] == name]
        segs = result._segs[result._segs["fecha"] == name]
        if fits.empty and segs.empty:
            raise ValueError(f"el {name} no está entre los días analizados")
        series = med.day_series(result.readings, name, result.days.loc[pd.Timestamp(name), "c_ext"])
        noise = float(result.days.loc[pd.Timestamp(name), "piso_ruido"])
        occ, cases = result.occupancy, tuple(result.cases)
    else:
        fits, segs, noise, occ, cases = result._fits, result._segs, result.noise_floor, None, ()
        s = result.series
        try:
            pd.Timestamp(s.name)
            series = s
        except ValueError:  # no es un día: se le pone una fecha para la figura
            start = result.start or pd.Timestamp("2000-01-03")
            off = (start - start.normalize()) / pd.Timedelta(hours=1)
            series = ObservedSeries(name=start.strftime("%Y-%m-%d"), t_h=s.t_h + off, c_obs=s.c_obs, c_ext_nominal=s.c_ext_nominal)
            fits = fits.assign(t0_h=fits["t0_h"] + off, t1_h=fits["t1_h"] + off) if not fits.empty else fits
            segs = segs.assign(t0_h=segs["t0_h"] + off, t1_h=segs["t1_h"] + off) if not segs.empty else segs
            hours = (float(series.t_h[0]), float(series.t_h[-1]))
    methods = [m for m in med.METHODS if m in set(fits["metodo"]) | set(segs["metodo"])]
    fig = fig_day_compact(series, fits, segs, noise, result.selection, px.Lang(), hours=hours, methods=methods, occupancy=occ, cases=cases)
    if path is not None:
        fig.savefig(path)
    plt.close(fig)
    return fig
