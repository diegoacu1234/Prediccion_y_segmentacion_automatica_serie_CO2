"""Predicción del CO2 de los salones a 1–60 minutos y del momento en que supera 1000 ppm.

Una serie por salón, en una grilla de 1 minuto. Desde cada origen t, un minuto
del horario de clase, se pronostica C(t+h) para los horizontes de `HORIZONS`.
Las variables son causales: sólo usan lo medido hasta t. El objetivo es el
cambio C(t+h) − C(t).

Partición por días de cada mes, la misma para los 32 salones y para todos los
meses (la Escuela B empieza a medir el 6 de junio):

- entrenamiento: del 1 al 15;
- detención temprana e hiperparámetros: del 16 al 19;
- calibración de los intervalos: del 20 al 23;
- prueba: del 24 a fin de mes.

Sólo entran los días hábiles con clase (`mediciones.daily_table`: el CO2
recorre al menos 150 ppm). Los orígenes de entrenamiento van de 7:00 a 18:00,
cada 2 minutos; los de evaluación, de 7:30 a 17:30, todos los minutos.

Las variables se agrupan para poder agregarlas de a una (`GROUPS`):

- G0, CO2 reciente: valor, cambios, pendientes, curvatura y extremos de las
  últimas horas;
- G1, calendario: minuto del día y día de la semana;
- G2, temperatura y humedad, y sus cambios;
- G3, historia del salón: su perfil de los últimos días con clase, el máximo
  del día y el exceso sobre la línea de base;
- G4, el salón y su escuela.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .mediciones import daily_table

HORIZONS = (1, 5, 10, 15, 20, 25, 30, 45, 60)
THRESHOLD = 1000.0  # ppm
REARM = 950.0  # después de un cruce, el CO2 tiene que bajar de acá para que cuente otro
SCHOOLS = {"Escuelas32y83": "A", "Escuelas3y121": "B"}
CO2_RANGE = (350.0, 5000.0)
TEMP_RANGE = (0.0, 45.0)
HUM_RANGE = (0.0, 100.0)
MAX_GAP_MIN = 5  # los huecos de hasta 5 min se interpolan; los más largos quedan sin datos

# partición por días de cada mes, para que todas las estaciones estén en el entrenamiento y en la prueba:
# del 1 al 15 se entrena, del 16 al 19 se detiene el entrenamiento y se eligen hiperparámetros,
# del 20 al 23 se calibran los intervalos y del 24 a fin de mes es la prueba
SPLIT_LAST_DAY = {"train": 15, "es": 19, "cal": 23}
SPLITS = ("train", "es", "cal", "test")
# el día de prueba que se muestra en detalle en el capítulo, los cuatro orígenes de sus abanicos y la semana de prueba que se dibuja entera
EXAMPLE_DAY = ("A02", "2024-06-24")  # un lunes con ingreso, dos recreos, mediodía y cuatro cruces de 1000 ppm
EXAMPLE_ORIGINS = ("7:50", "9:25", "12:45", "15:00")  # antes del ingreso, al empezar el recreo, al final del mediodía, antes del recreo de la tarde
EXAMPLE_WEEK = ("A02", "2024-08-26")  # la única semana de lunes a viernes entera en la prueba de A02
TRAIN_WINDOW = (7 * 60, 18 * 60 - 1)  # minutos del día, ambos incluidos
EVAL_WINDOW = (7 * 60 + 30, 17 * 60 + 30)
TRAIN_STEP = 2

LAGS = (1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240)
SLOPES = (5, 10, 30, 60, 120, 240)
ROLLING = (15, 30, 60, 120, 240)
HISTORY = (15, 30, 60, 120, 240)  # longitudes de historia del estudio de información
FULL_HISTORY = 60
PROFILE_DAYS = 5

# ------------------------------------------------------------------ salones


def room_name(ip: str, sensor_id: int) -> str:
    """El nombre del salón en la tesis: letra de la escuela y número local (A02)."""
    return f"{SCHOOLS[ip]}{int(sensor_id):02d}"


def room_list(path) -> list:
    """Los salones (ip, id) de `outputs/mediciones/salones_id_ip.csv`, ordenados por nombre."""
    t = pd.read_csv(path)
    rooms = [(str(r.ip), int(r.id)) for r in t.itertuples() if str(r.ip) in SCHOOLS]
    return sorted(rooms, key=lambda r: room_name(*r))


def _cache_path(cache_dir, ip: str, sensor_id: int) -> Path:
    return Path(cache_dir) / f"{ip}_id{int(sensor_id):02d}.parquet"  # el mismo nombre que `mediciones.load_room`


def load_rooms(folder, rooms, cache_dir) -> None:
    """Deja en `cache_dir` un parquet por salón, leyendo los CSV una sola vez.

    `mediciones.load_room` hace lo mismo para un salón, pero recorre todos los
    archivos cada vez; para 32 salones serían 32 pasadas por 1,3 GB.
    """
    missing = [r for r in rooms if not _cache_path(cache_dir, *r).exists()]
    if not missing:
        return
    want = {(ip, sid) for ip, sid in missing}
    parts = {r: [] for r in missing}
    for path in sorted(Path(folder).glob("*.csv")):
        df = pd.read_csv(path, engine="pyarrow")
        df["ip"] = df["ip"].astype(str)
        for (ip, sid), g in df.groupby(["ip", "id"], sort=False):
            if (ip, int(sid)) in want:
                parts[(ip, int(sid))].append(g[["tiempo", "co2", "temp", "hum"]])
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    for r, p in parts.items():
        if not p:
            continue
        df = pd.concat(p, ignore_index=True)
        df["tiempo"] = pd.to_datetime(df["tiempo"])
        df = df.sort_values("tiempo").drop_duplicates("tiempo").reset_index(drop=True)
        df.to_parquet(_cache_path(cache_dir, *r), index=False)


def read_room(cache_dir, ip: str, sensor_id: int) -> pd.DataFrame:
    return pd.read_parquet(_cache_path(cache_dir, ip, sensor_id))


# ------------------------------------------------------------------ grilla de 1 minuto


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Valores fuera del rango físico → NaN (Capítulo 3)."""
    out = df.copy()
    for col, (lo, hi) in (("co2", CO2_RANGE), ("temp", TEMP_RANGE), ("hum", HUM_RANGE)):
        v = out[col].astype(float)
        out[col] = v.where((v >= lo) & (v <= hi))
    return out


def fill_short_gaps(x, max_gap: int = MAX_GAP_MIN) -> np.ndarray:
    """Interpola linealmente las rachas de NaN de hasta `max_gap` muestras con datos a los dos lados."""
    x = np.asarray(x, dtype=float).copy()
    nan = np.isnan(x)
    if not nan.any() or nan.all():
        return x
    edges = np.diff(np.r_[0, nan.astype(np.int8), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    short = (ends - starts <= max_gap) & (starts > 0) & (ends < len(x))
    if short.any():
        idx = np.arange(len(x))
        filled = np.interp(idx, idx[~nan], x[~nan])
        for s, e in zip(starts[short], ends[short]):
            x[s:e] = filled[s:e]
    return x


def minute_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Media de las lecturas de cada minuto, de las 0:00 del primer día a las 23:59 del último, con los huecos cortos interpolados."""
    d = clean(df).set_index("tiempo")[["co2", "temp", "hum"]]
    m = d.resample("1min").mean()
    full = pd.date_range(m.index[0].normalize(), m.index[-1].normalize() + pd.Timedelta(minutes=1439), freq="1min")
    m = m.reindex(full)
    for col in m.columns:
        m[col] = fill_short_gaps(m[col].to_numpy())
    m.index.name = "tiempo"
    return m.astype("float32")


def day_table(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por día: si es día hábil con clase y la línea de base causal (sólo días anteriores)."""
    days = daily_table(clean(df).dropna(subset=["co2"]))
    days["clase"] = days["ocupado"] & (days["dia_semana"] < 5)
    # `daily_table` toma el mínimo en una ventana centrada, que mira días futuros
    days["c_ext_causal"] = days["p1"].rolling(7, min_periods=1).min().shift(1)
    return days


# ------------------------------------------------------------------ cruces del umbral


def crossings(c, threshold: float = THRESHOLD, rearm: float = REARM, persist: int = 2):
    """Cruces hacia arriba del umbral y estado de espera en cada minuto.

    Un cruce es el primer minuto de una racha de al menos `persist` minutos en o
    por encima del umbral, estando habilitado. Después de un cruce no cuenta otro
    hasta que el CO2 baje de `rearm`. Devuelve los índices de los cruces y una
    máscara `armed`: True donde, en ese minuto, el próximo cruce todavía puede
    ocurrir (habilitado y por debajo del umbral).
    """
    c = np.asarray(c, dtype=float)
    n = len(c)
    above = c >= threshold
    run_ok = above.copy()
    for k in range(1, persist):
        run_ok[: n - k] &= above[k:]
        run_ok[n - k :] = False
    armed = np.zeros(n, dtype=bool)
    out = []
    state = True
    values, ok = c.tolist(), run_ok.tolist()
    for i in range(n):
        ci = values[i]
        if ci != ci:  # NaN: no cambia el estado
            continue
        if state:
            if ok[i]:
                out.append(i)
                state = False
            elif ci < threshold:
                armed[i] = True
        elif ci < rearm:
            state = True
            armed[i] = True
    return np.asarray(out, dtype=int), armed


def minutes_to_next(n: int, events) -> np.ndarray:
    """Para cada minuto i, minutos hasta el próximo evento estrictamente posterior (inf si no hay)."""
    events = np.asarray(events, dtype=int)
    idx = np.arange(n)
    j = np.searchsorted(events, idx, side="right")
    out = np.full(n, np.inf)
    has = j < len(events)
    out[has] = events[j[has]] - idx[has]
    return out


# ------------------------------------------------------------------ variables


def _slope(c, w: int) -> np.ndarray:
    """Pendiente (ppm/min) de la recta de mínimos cuadrados sobre las últimas `w` muestras, terminando en cada una."""
    j = np.arange(w, dtype=float)
    k = (j - j.mean()) / ((j - j.mean()) ** 2).sum()
    out = np.full(len(c), np.nan)
    out[w - 1 :] = np.convolve(c, k[::-1], mode="valid")
    return out


def _shift(x, k: int) -> np.ndarray:
    """x[i − k] (k > 0, hacia atrás) o x[i + |k|] (k < 0, hacia adelante), con NaN en el borde."""
    out = np.full(len(x), np.nan)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


def daily_profile(grid: pd.DataFrame, days: pd.DataFrame, n_days: int = PROFILE_DAYS, min_days: int = 3) -> np.ndarray:
    """Para cada minuto, la mediana del CO2 a esa hora en los `n_days` días con clase anteriores del salón."""
    c = grid["co2"].to_numpy(dtype=float)
    n_d = len(c) // 1440
    mat = c[: n_d * 1440].reshape(n_d, 1440)
    dates = grid.index[::1440][:n_d].normalize()
    is_class = days["clase"].reindex(dates, fill_value=False).to_numpy(dtype=bool)
    prof = np.full((n_d, 1440), np.nan)
    past = []
    for d in range(n_d):
        if len(past) >= min_days:
            sub = mat[past[-n_days:]]
            ok = np.isfinite(sub).sum(axis=0) >= min_days
            if ok.any():
                prof[d, ok] = np.nanmedian(sub[:, ok], axis=0)
        if is_class[d]:
            past.append(d)
    out = np.full(len(c), np.nan)
    out[: n_d * 1440] = prof.ravel()
    return out


def g0_columns(history: int) -> list:
    """Las variables de CO2 reciente que miran a lo sumo `history` minutos hacia atrás."""
    cols = ["co2"]
    cols += [f"d{k}" for k in LAGS if k <= history]
    cols += [f"pend{w}" for w in SLOPES if w <= history]
    if history >= 10:
        cols.append("curv")
    for w in ROLLING:
        if w <= history:
            cols += [f"max{w}", f"min{w}", f"sd{w}"]
    return cols


G1 = ["minuto", "hora_sen", "hora_cos", "dia_semana"]
G2 = ["temp", "hum", "dtemp5", "dtemp15", "dtemp30", "dhum5", "dhum15", "dhum30"]
G3 = ["perfil", "perfil_dif", *[f"perfil_h{h}" for h in HORIZONS], "max_dia", "exceso"]
G4 = ["salon", "escuela"]
GROUPS = {"G0": g0_columns(FULL_HISTORY), "G1": G1, "G2": G2, "G3": G3, "G4": G4}
GROUP_NAMES = {"G0": "CO₂ reciente", "G1": "hora y día", "G2": "temperatura y humedad", "G3": "historia del salón", "G4": "salón"}
CATEGORICAL = ["salon"]
TARGETS = [f"y{h}" for h in HORIZONS]


def feature_set(groups, history: int = FULL_HISTORY) -> list:
    """Columnas de un conjunto de grupos, con la historia de CO2 recortada a `history` minutos."""
    cols = []
    for g in groups:
        cols += g0_columns(history) if g == "G0" else GROUPS[g]
    return cols


FULL = feature_set(("G0", "G1", "G2", "G3", "G4"))


def room_features(grid: pd.DataFrame, days: pd.DataFrame, room_code: int, school: int) -> pd.DataFrame:
    """Variables, objetivos y estado del umbral de un salón, en cada minuto de su grilla."""
    c = grid["co2"].to_numpy(dtype=float)
    n = len(c)
    s = pd.Series(c)
    f = {"co2": c}
    for k in LAGS:
        f[f"d{k}"] = c - _shift(c, k)
    slope5 = _slope(c, 5)
    for w in SLOPES:
        f[f"pend{w}"] = slope5 if w == 5 else _slope(c, w)
    f["curv"] = slope5 - _shift(slope5, 5)
    for w in ROLLING:
        r = s.rolling(w, min_periods=w)
        f[f"max{w}"] = r.max().to_numpy() - c
        f[f"min{w}"] = c - r.min().to_numpy()
        f[f"sd{w}"] = r.std().to_numpy()

    minute = (grid.index.hour * 60 + grid.index.minute).to_numpy()
    f["minuto"] = minute.astype(float)
    f["hora_sen"] = np.sin(2 * np.pi * minute / 1440)
    f["hora_cos"] = np.cos(2 * np.pi * minute / 1440)
    f["dia_semana"] = grid.index.dayofweek.to_numpy().astype(float)

    for col, name in (("temp", "temp"), ("hum", "hum")):
        x = grid[col].to_numpy(dtype=float)
        f[name] = x
        for k in (5, 15, 30):
            f[f"d{name}{k}"] = x - _shift(x, k)

    prof = daily_profile(grid, days)
    f["perfil"] = prof
    f["perfil_dif"] = c - prof
    for h in HORIZONS:
        f[f"perfil_h{h}"] = _shift(prof, -h) - prof
    day_idx = np.arange(n) // 1440
    f["max_dia"] = pd.Series(c).groupby(day_idx).cummax().to_numpy() - c
    dates = grid.index.normalize()
    c_ext = days["c_ext_causal"].reindex(dates).to_numpy(dtype=float)
    f["exceso"] = c - c_ext
    f["c_ext"] = c_ext

    f["salon"] = np.full(n, room_code, dtype=float)
    f["escuela"] = np.full(n, school, dtype=float)
    for h in HORIZONS:
        f[f"y{h}"] = _shift(c, -h) - c

    cross, armed = crossings(c)
    f["habilitado"] = armed
    f["min_al_cruce"] = minutes_to_next(n, cross)
    out = pd.DataFrame(f, index=grid.index)
    out["clase"] = days["clase"].reindex(dates, fill_value=False).to_numpy(dtype=bool)
    return out


def split_of(ts: pd.DatetimeIndex) -> np.ndarray:
    """La partición de cada instante según su día del mes (`SPLIT_LAST_DAY`)."""
    day = np.asarray(pd.DatetimeIndex(ts).day)
    out = np.full(len(day), "test", dtype=object)
    for name in ("cal", "es", "train"):
        out[day <= SPLIT_LAST_DAY[name]] = name
    return out


def origins(feat: pd.DataFrame) -> pd.DataFrame:
    """Los orígenes de un salón: días con clase, ventana horaria de cada partición y todo lo necesario definido."""
    minute = feat["minuto"].to_numpy()
    split = split_of(feat.index)
    in_train = (split == "train") & (minute >= TRAIN_WINDOW[0]) & (minute <= TRAIN_WINDOW[1]) & (minute % TRAIN_STEP == 0)
    in_eval = (split != "train") & (minute >= EVAL_WINDOW[0]) & (minute <= EVAL_WINDOW[1])
    need = feature_set(("G0", "G1", "G2", "G3"), history=max(HISTORY)) + TARGETS
    ok = feat["clase"].to_numpy() & (in_train | in_eval) & np.isfinite(feat[need].to_numpy()).all(axis=1)
    out = feat.loc[ok].copy()
    out.insert(0, "split", split[ok])
    return out


@dataclass
class Data:
    """Los orígenes de los 32 salones, con sus variables y objetivos, y las grillas de 1 minuto."""

    rows: pd.DataFrame  # una fila por origen: tiempo, salón, partición, variables, objetivos y estado del umbral
    grids: dict  # nombre del salón → grilla de 1 minuto (co2, temp, hum)
    days: dict  # nombre del salón → tabla diaria
    rooms: list  # nombres de los salones, en el orden de su código


def build(cache_dir, rooms) -> Data:
    """Grillas, variables y orígenes de todos los salones."""
    names = [room_name(*r) for r in rooms]
    parts, grids, tables = [], {}, {}
    for code, ((ip, sid), name) in enumerate(zip(rooms, names)):
        df = read_room(cache_dir, ip, sid)
        grid = minute_grid(df)
        days = day_table(df)
        feat = room_features(grid, days, code, 0 if SCHOOLS[ip] == "A" else 1)
        o = origins(feat)
        o.insert(0, "sala", name)
        parts.append(o)
        grids[name] = grid
        tables[name] = days
    rows = pd.concat(parts)
    rows.index.name = "tiempo"
    rows = rows.reset_index()
    for col in rows.columns:
        if rows[col].dtype == np.float64:
            rows[col] = rows[col].astype("float32")
    return Data(rows=rows, grids=grids, days=tables, rooms=names)


def save_data(data: Data, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data.rows.to_parquet(out / "origenes.parquet", index=False)
    g = pd.concat({k: v for k, v in data.grids.items()}, names=["sala", "tiempo"]).reset_index()
    g.to_parquet(out / "grillas.parquet", index=False)
    d = pd.concat({k: v for k, v in data.days.items()}, names=["sala", "fecha"]).reset_index()
    d.to_parquet(out / "dias.parquet", index=False)
    pd.Series(data.rooms, name="sala").to_csv(out / "salones.csv", index=False)


def load_data(out_dir) -> Data:
    out = Path(out_dir)
    rows = pd.read_parquet(out / "origenes.parquet")
    g = pd.read_parquet(out / "grillas.parquet")
    grids = {k: v.drop(columns="sala").set_index("tiempo") for k, v in g.groupby("sala", sort=False)}
    d = pd.read_parquet(out / "dias.parquet")
    days = {k: v.drop(columns="sala").set_index("fecha") for k, v in d.groupby("sala", sort=False)}
    rooms = pd.read_csv(out / "salones.csv")["sala"].tolist()
    return Data(rows=rows, grids=grids, days=days, rooms=rooms)


def subset(rows: pd.DataFrame, split: str) -> pd.DataFrame:
    return rows[rows["split"] == split].reset_index(drop=True)


def actual(rows: pd.DataFrame) -> np.ndarray:
    """C(t+h) medido, (n, horizontes)."""
    return rows["co2"].to_numpy(float)[:, None] + rows[TARGETS].to_numpy(float)


def grid_positions(data: Data, rows: pd.DataFrame) -> dict:
    """Para cada salón: qué filas de `rows` son suyas y en qué posición de su grilla de 1 minuto cae cada origen."""
    out = {}
    for sala, g in rows.groupby("sala", sort=False):
        start = data.grids[sala].index[0].to_datetime64()
        pos = ((g["tiempo"].to_numpy() - start) // np.timedelta64(1, "m")).astype(np.int64)
        out[sala] = (g.index.to_numpy(), pos)
    return out


# ------------------------------------------------------------------ pronósticos


@dataclass
class Forecast:
    """Pronóstico de un modelo sobre un conjunto de orígenes; cada arreglo es (n, horizontes)."""

    mid: np.ndarray  # C(t+h) pronosticado
    lo: np.ndarray | None = None  # extremos del intervalo del 80 %
    hi: np.ndarray | None = None
    p: np.ndarray | None = None  # P(C(t+h) ≥ 1000)
    pw: np.ndarray | None = None  # P(cruce en (t, t+h])

    def save(self, path) -> None:
        arrays = {k: v.astype(np.float32) for k, v in vars(self).items() if v is not None}
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path) -> "Forecast":
        z = np.load(path)
        return cls(**{k: z[k].astype(float) for k in z.files})


QUANTILE_GRID = np.linspace(0.0, 1.0, 1001)
COVER = 0.8  # intervalo del 80 %: cuantiles 0,1 y 0,9


class Residuals:
    """Distribución empírica de los residuos C − Ĉ de la calibración, por horizonte (conformal partido)."""

    def __init__(self, y, mid):
        r = np.asarray(y, float) - np.asarray(mid, float)
        self.q = np.nanquantile(r, QUANTILE_GRID, axis=0)  # (1001, horizontes)

    def interval(self, mid, cover: float = COVER):
        last = len(QUANTILE_GRID) - 1
        i_lo, i_hi = int(round((0.5 - cover / 2) * last)), int(round((0.5 + cover / 2) * last))
        return mid + self.q[i_lo], mid + self.q[i_hi]

    def prob_above(self, mid, thr: float = THRESHOLD) -> np.ndarray:
        """P(C ≥ thr) = P(r ≥ thr − Ĉ)."""
        out = np.empty_like(mid, dtype=float)
        for j in range(mid.shape[1]):
            out[:, j] = 1.0 - np.interp(thr - mid[:, j], self.q[:, j], QUANTILE_GRID, left=0.0, right=1.0)
        return out


def cqr_offset(y, lo, hi, cover: float = COVER) -> np.ndarray:
    """Corrección conformal de un intervalo por cuantiles (CQR): cuánto ensancharlo para cubrir `cover` en la calibración."""
    e = np.maximum(lo - y, y - hi)
    n = np.isfinite(e).sum(axis=0)
    level = np.minimum(1.0, cover * (1 + 1 / np.maximum(n, 1)))
    return np.array([np.nanquantile(e[:, j], level[j]) for j in range(e.shape[1])])


def within(p) -> np.ndarray:
    """P(cruce en (t, t+h]) aproximada por el máximo de P(C(t+h′) ≥ 1000) en los horizontes h′ ≤ h."""
    return np.maximum.accumulate(p, axis=1)


def quantile_prob_above(qs, levels, thr: float = THRESHOLD) -> np.ndarray:
    """P(C ≥ thr) interpolando la función de distribución entre cuantiles (n, horizontes, niveles)."""
    lv = np.asarray(levels, float)
    n, nh, _ = qs.shape
    out = np.empty((n, nh))
    for j in range(nh):
        q = np.sort(qs[:, j, :], axis=1)
        below = (q <= thr).sum(axis=1)  # cuántos cuantiles quedan por debajo del umbral
        k = np.clip(below - 1, 0, len(lv) - 2)
        q0, q1 = q[np.arange(n), k], q[np.arange(n), k + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.clip(np.where(q1 > q0, (thr - q0) / (q1 - q0), 0.5), 0.0, 1.0)
        f = lv[k] + frac * (lv[k + 1] - lv[k])
        f = np.where(below == 0, lv[0] / 2, f)  # por debajo del primer cuantil
        f = np.where(below == len(lv), 1 - (1 - lv[-1]) / 2, f)  # por encima del último
        out[:, j] = 1.0 - f
    return out


# ------------------------------------------------------------------ modelos

MODEL_KEYS = ("persistencia", "perfil", "fisico", "ridge", "mlp", "lightgbm", "chronos")
MODEL_NAMES = {
    "persistencia": "Persistencia",
    "perfil": "Perfil diario",
    "fisico": "Balance de masa",
    "ridge": "Ridge",
    "mlp": "MLP",
    "lightgbm": "LightGBM",
    "chronos": "Chronos",
    "lightgbm_clasif": "LightGBM (clasificador)",
}
REFERENCES = ("persistencia", "perfil")


class Persistence:
    key = "persistencia"

    def fit(self, data, train, es):
        return self

    def predict(self, data, rows) -> np.ndarray:
        return np.repeat(rows["co2"].to_numpy(float)[:, None], len(HORIZONS), axis=1)


class Profile:
    """La mediana del salón a la hora t+h en sus últimos días con clase."""

    key = "perfil"

    def fit(self, data, train, es):
        return self

    def predict(self, data, rows) -> np.ndarray:
        return rows["perfil"].to_numpy(float)[:, None] + rows[[f"perfil_h{h}" for h in HORIZONS]].to_numpy(float)


LAM_MAX = 20.0 / 60.0  # 1/min: 20 renovaciones por hora
PHYS_WINDOWS = (15, 30, 60)


def _window_sum(v, w: int) -> np.ndarray:
    out = np.full(len(v), np.nan)
    out[w - 1 :] = np.convolve(v, np.ones(w), mode="valid")
    return out


BETA_MAX = 2 * np.tanh(LAM_MAX / 2)


def physical_params(c, w: int):
    """Recta ΔC = a − β·C̄ sobre los últimos `w` minutos, terminando en cada uno: (a [ppm/min], β [1/min]).

    ΔC es el cambio de un minuto y C̄ el promedio de sus dos extremos. Con el
    balance de masa del Capítulo 2 y la generación y la ventilación constantes
    en la ventana, C(t) = C∞ + (C0 − C∞)·e^(−λt) cumple esa recta exactamente,
    con β = 2·tanh(λ/2) y a = β·C∞. β se acota a [0, 20 1/h]; con β = 0 queda
    una recta con la pendiente media.
    """
    c = np.asarray(c, float)
    prev = _shift(c, 1)
    y = c - prev
    x = 0.5 * (c + prev) - 1000.0  # centrado para que las sumas no pierdan precisión
    sx, sy, sxx, sxy = (_window_sum(v, w) for v in (x, y, x * x, x * y))
    den = w * sxx - sx * sx
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(den > 1e-9, (w * sxy - sx * sy) / den, 0.0)
    beta = np.clip(-b, 0.0, BETA_MAX)
    a = sy / w + beta * (sx / w + 1000.0)  # la recta, con β ya acotado, pasa por las medias
    return a, beta


def physical_forecast(c0, a, beta, c_ext) -> np.ndarray:
    """C(t+h) = C∞ + (C(t) − C∞)·e^(−λh), con λ = 2·artanh(β/2) y C∞ = a/β, al menos el CO2 exterior; recta si β ≈ 0.

    El resultado se acota al rango físico: entre el CO2 exterior y el máximo de
    `CO2_RANGE`. Sin esa cota, la recta de una bajada sin curvatura sigue de
    largo por debajo del exterior.
    """
    c0, a, beta, c_ext = (np.asarray(v, float) for v in (c0, a, beta, c_ext))
    h = np.asarray(HORIZONS, float)[None, :]
    lin = beta < 1e-4
    lam = 2 * np.arctanh(beta / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        cinf = np.where(lin, c0, a / np.where(lin, 1.0, beta))
    cinf = np.maximum(cinf, c_ext)
    expo = cinf[:, None] + (c0 - cinf)[:, None] * np.exp(-lam[:, None] * h)
    linear = c0[:, None] + (a - beta * c0)[:, None] * h
    return np.clip(np.where(lin[:, None], linear, expo), c_ext[:, None], CO2_RANGE[1])


class Physical:
    """Balance de masa en línea: extrapola la exponencial ajustada a los últimos minutos."""

    key = "fisico"

    def __init__(self, window: int = 30):
        self.window = window
        self.scores = {}

    def params(self, data, rows, w=None):
        w = w or self.window
        a = np.full(len(rows), np.nan)
        beta = np.full(len(rows), np.nan)
        for sala, (idx, pos) in grid_positions(data, rows).items():
            aa, bb = physical_params(data.grids[sala]["co2"].to_numpy(float), w)
            a[idx], beta[idx] = aa[pos], bb[pos]
        return a, beta

    def fit(self, data, train, es):
        y = actual(es)
        for w in PHYS_WINDOWS:
            self.window = w
            self.scores[w] = float(np.nanmean(np.abs(self.predict(data, es) - y)))
        self.window = min(self.scores, key=self.scores.get)
        return self

    def predict(self, data, rows) -> np.ndarray:
        a, beta = self.params(data, rows)
        return physical_forecast(rows["co2"], a, beta, rows["c_ext"])


RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


class Linear:
    """Ridge sobre las variables estandarizadas (el salón, en one-hot); una salida por horizonte."""

    key = "ridge"

    def __init__(self, cols, n_rooms: int):
        self.cols = list(cols)
        self.n_rooms = n_rooms
        self.num = [c for c in self.cols if c != "salon"]
        self.scores = {}

    def _X(self, rows) -> np.ndarray:
        X = (rows[self.num].to_numpy(np.float64) - self.mu) / self.sd
        if "salon" in self.cols:
            X = np.hstack([X, np.eye(self.n_rooms)[rows["salon"].to_numpy().astype(int)]])
        return X

    def _standardize(self, train) -> None:
        v = train[self.num].to_numpy(np.float64)
        self.mu, self.sd = v.mean(axis=0), v.std(axis=0)
        self.sd[self.sd == 0] = 1.0

    def fit(self, data, train, es):
        from sklearn.linear_model import Ridge

        self._standardize(train)
        X, Y = self._X(train), train[TARGETS].to_numpy(np.float64)
        Xe, Ye = self._X(es), es[TARGETS].to_numpy(np.float64)
        best = None
        for alpha in RIDGE_ALPHAS:
            m = Ridge(alpha=alpha).fit(X, Y)
            self.scores[alpha] = float(np.mean(np.abs(m.predict(Xe) - Ye)))
            if best is None or self.scores[alpha] < self.scores[best[0]]:
                best = (alpha, m)
        self.alpha, self.model = best
        return self

    def predict(self, data, rows) -> np.ndarray:
        return rows["co2"].to_numpy(float)[:, None] + self.model.predict(self._X(rows))


class MLP(Linear):
    """Perceptrón de dos capas (64 y 32 neuronas), con las mismas entradas que ridge; se detiene con los días de detención temprana."""

    key = "mlp"

    def __init__(self, cols, n_rooms: int, hidden=(64, 32), max_epochs: int = 60, patience: int = 6, seed: int = 0):
        super().__init__(cols, n_rooms)
        self.hidden, self.max_epochs, self.patience, self.seed = hidden, max_epochs, patience, seed

    def fit(self, data, train, es):
        import copy

        from sklearn.neural_network import MLPRegressor

        self._standardize(train)
        X, Y = self._X(train), train[TARGETS].to_numpy(np.float64)
        Xe, Ye = self._X(es), es[TARGETS].to_numpy(np.float64)
        self.ysd = Y.std(axis=0)
        net = MLPRegressor(hidden_layer_sizes=self.hidden, batch_size=1024, learning_rate_init=1e-3, alpha=1e-4, random_state=self.seed)
        best, wait, self.history = np.inf, 0, []
        for _ in range(self.max_epochs):
            net.partial_fit(X, Y / self.ysd)  # una pasada por los datos, en orden aleatorio
            score = float(np.mean(np.abs(net.predict(Xe) * self.ysd - Ye)))
            self.history.append(score)
            if score < best - 0.01:
                best, wait, self.model = score, 0, copy.deepcopy(net)
            else:
                wait += 1
                if wait >= self.patience:
                    break
        return self

    def predict(self, data, rows) -> np.ndarray:
        return rows["co2"].to_numpy(float)[:, None] + self.model.predict(self._X(rows)) * self.ysd


LGB_PARAMS = {
    "learning_rate": 0.1,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 0,
    "force_col_wise": True,
}
LGB_ROUNDS = 1500
LGB_STOP = 30


def _lgb_X(rows, cols) -> pd.DataFrame:
    X = rows[cols].astype("float32")
    if "salon" in cols:
        X["salon"] = rows["salon"].astype(int)
    return X


class GBM:
    """LightGBM, un modelo por horizonte para el cambio C(t+h) − C(t); con cuantiles 0,1 y 0,9 y un clasificador del cruce, si se piden."""

    key = "lightgbm"

    def __init__(self, cols, quantiles: bool = False, classifier: bool = False, params=None, threads: int = 4, verbose: bool = False):
        self.cols = list(cols)
        self.verbose = verbose
        self.quantiles, self.classifier = quantiles, classifier
        self.params = {**LGB_PARAMS, **(params or {}), "num_threads": threads}
        self.boosters = {}  # (tipo, h) → Booster; tipo: "mid", "q10", "q90", "clasif"

    def _train(self, ds, dv, objective: dict):
        import lightgbm as lgb

        return lgb.train({**self.params, **objective}, ds, num_boost_round=LGB_ROUNDS, valid_sets=[dv], callbacks=[lgb.early_stopping(LGB_STOP, verbose=False)])

    def _datasets(self, train, es, labels_tr, labels_es):
        import lightgbm as lgb

        cat = ["salon"] if "salon" in self.cols else []
        ds = lgb.Dataset(_lgb_X(train, self.cols), label=labels_tr, categorical_feature=cat, free_raw_data=False, params={"verbose": -1, "max_bin": 127})
        dv = lgb.Dataset(_lgb_X(es, self.cols), label=labels_es, reference=ds, categorical_feature=cat, free_raw_data=False)
        ds.construct()
        dv.construct()
        return ds, dv

    def fit(self, data, train, es):
        Y, Ye = train[TARGETS].to_numpy(float), es[TARGETS].to_numpy(float)
        ds, dv = self._datasets(train, es, Y[:, 0], Ye[:, 0])
        kinds = [("mid", {"objective": "regression", "metric": "l1"})]
        if self.quantiles:
            kinds += [("q10", {"objective": "quantile", "alpha": 0.1, "metric": "quantile"}), ("q90", {"objective": "quantile", "alpha": 0.9, "metric": "quantile"})]
        for j, h in enumerate(HORIZONS):
            ds.set_label(Y[:, j])
            dv.set_label(Ye[:, j])
            for kind, obj in kinds:
                t0 = time.time()
                self.boosters[(kind, h)] = self._train(ds, dv, obj)
                if self.verbose:
                    print(f"    LightGBM {kind} a {h} min: {self.boosters[(kind, h)].best_iteration} iteraciones, {time.time() - t0:.0f} s", flush=True)
        if self.classifier:
            tr, ev = train[train["habilitado"]].reset_index(drop=True), es[es["habilitado"]].reset_index(drop=True)
            ds, dv = self._datasets(tr, ev, np.zeros(len(tr)), np.zeros(len(ev)))
            for h in HORIZONS:
                ds.set_label((tr["min_al_cruce"].to_numpy() <= h).astype(float))
                dv.set_label((ev["min_al_cruce"].to_numpy() <= h).astype(float))
                t0 = time.time()
                self.boosters[("clasif", h)] = self._train(ds, dv, {"objective": "binary", "metric": "binary_logloss"})
                if self.verbose:
                    print(f"    LightGBM clasificador a {h} min: {self.boosters[('clasif', h)].best_iteration} iteraciones, {time.time() - t0:.0f} s", flush=True)
        return self

    def _raw(self, kind, rows) -> np.ndarray:
        X = _lgb_X(rows, self.cols)
        return np.column_stack([self.boosters[(kind, h)].predict(X, num_iteration=self.boosters[(kind, h)].best_iteration) for h in HORIZONS])

    def predict(self, data, rows) -> np.ndarray:
        return rows["co2"].to_numpy(float)[:, None] + self._raw("mid", rows)

    def predict_quantiles(self, rows):
        c0 = rows["co2"].to_numpy(float)[:, None]
        return c0 + self._raw("q10", rows), c0 + self._raw("q90", rows)

    def predict_within(self, rows) -> np.ndarray:
        return np.maximum.accumulate(self._raw("clasif", rows), axis=1)

    def rounds(self) -> pd.DataFrame:
        return pd.DataFrame([{"tipo": k, "horizonte": h, "iteraciones": b.best_iteration} for (k, h), b in self.boosters.items()])

    def save(self, folder) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for (kind, h), b in self.boosters.items():
            b.save_model(str(folder / f"{kind}_h{h:02d}.txt"), num_iteration=b.best_iteration)


# ------------------------------------------------------------------ Chronos

CHRONOS_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CHRONOS_MODEL = "amazon/chronos-bolt-tiny"


def chronos_pipeline(name: str = CHRONOS_MODEL):
    import torch
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(name, device_map="cpu", dtype=torch.float32)


def chronos_quantiles(pipe, c, pos, context: int = 512, batch: int = 1024) -> np.ndarray:
    """Cuantiles 0,1…0,9 de Chronos en los horizontes de `HORIZONS`, para los orígenes `pos` de una grilla: (n, horizontes, niveles).

    El contexto son los `context` minutos que terminan en el origen, con los
    huecos largos como NaN, que Chronos acepta.
    """
    import torch

    c = np.asarray(c, dtype=np.float32)
    padded = np.r_[np.full(context, np.nan, dtype=np.float32), c]
    windows = np.lib.stride_tricks.sliding_window_view(padded, context)  # windows[i + 1] termina en c[i]
    hz = np.asarray(HORIZONS) - 1
    out = np.empty((len(pos), len(HORIZONS), len(CHRONOS_LEVELS)), dtype=np.float32)
    for s in range(0, len(pos), batch):
        p = np.asarray(pos[s : s + batch])
        ctx = torch.from_numpy(np.ascontiguousarray(windows[p + 1]))
        q, _ = pipe.predict_quantiles(ctx, prediction_length=max(HORIZONS), quantile_levels=list(CHRONOS_LEVELS))
        out[s : s + len(p)] = q.numpy()[:, hz, :]
    return out


# ------------------------------------------------------------------ métricas del pronóstico


def point_metrics(rows, mid, by=None) -> pd.DataFrame:
    """MAE, RMSE y sesgo por horizonte (y por las columnas `by` de `rows`)."""
    e = np.asarray(mid, float) - actual(rows)
    frames = []
    for j, h in enumerate(HORIZONS):
        df = pd.DataFrame({"e": e[:, j]})
        if by:
            for b in by:
                df[b] = rows[b].to_numpy()
            g = df.groupby(list(by))["e"]
            out = pd.DataFrame({"mae": g.apply(lambda v: np.nanmean(np.abs(v))), "rmse": g.apply(lambda v: np.sqrt(np.nanmean(v**2))), "sesgo": g.mean(), "n": g.count()}).reset_index()
        else:
            v = df["e"].to_numpy()
            out = pd.DataFrame([{"mae": np.nanmean(np.abs(v)), "rmse": np.sqrt(np.nanmean(v**2)), "sesgo": np.nanmean(v), "n": int(np.isfinite(v).sum())}])
        out.insert(0, "horizonte", h)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def coverage(rows, lo, hi, by=None) -> pd.DataFrame:
    """Fracción de los valores medidos dentro del intervalo y su ancho medio, por horizonte."""
    y = actual(rows)
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    frames = []
    for j, h in enumerate(HORIZONS):
        df = pd.DataFrame({"dentro": inside[:, j].astype(float), "ancho": width[:, j]})
        keys = list(by) if by else []
        for b in keys:
            df[b] = rows[b].to_numpy()
        out = df.groupby(keys).mean().reset_index() if keys else df.mean().to_frame().T
        out.insert(0, "horizonte", h)
        frames.append(out)
    return pd.concat(frames, ignore_index=True).rename(columns={"dentro": "cobertura"})


# ------------------------------------------------------------------ cruce del umbral


def time_to_threshold(c0, traj, thr: float = THRESHOLD) -> np.ndarray:
    """Minutos hasta que la trayectoria (C(t) y los valores en `HORIZONS`, unidos por rectas) llega al umbral; inf si no llega."""
    knots = np.r_[0.0, np.asarray(HORIZONS, float)]
    V = np.column_stack([np.asarray(c0, float), np.asarray(traj, float)])
    out = np.full(len(V), np.inf)
    out[V[:, 0] >= thr] = 0.0
    for k in range(len(knots) - 1):
        v0, v1 = V[:, k], V[:, k + 1]
        hit = np.isinf(out) & (v0 < thr) & (v1 >= thr)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = (thr - v0) / (v1 - v0)
        out[hit] = knots[k] + frac[hit] * (knots[k + 1] - knots[k])
    return out


def real_crossings(data: Data, rows: pd.DataFrame) -> pd.DataFrame:
    """Los cruces medidos que caen en la ventana de evaluación de los días de `rows`: sala, fecha y minuto del día."""
    out = []
    for sala, g in rows.groupby("sala", sort=False):
        grid = data.grids[sala]
        idx, _ = crossings(grid["co2"].to_numpy(float))
        ts = grid.index[idx]
        dates = pd.DatetimeIndex(g["tiempo"].dt.normalize().unique())
        minute = ts.hour * 60 + ts.minute
        keep = ts.normalize().isin(dates) & (minute > EVAL_WINDOW[0]) & (minute <= EVAL_WINDOW[1] + 1)
        out.append(pd.DataFrame({"sala": sala, "fecha": ts.normalize()[keep], "minuto": minute[keep]}))
    return pd.concat(out, ignore_index=True)


def _day_arrays(rows: pd.DataFrame, alarm) -> dict:
    """Por (sala, fecha): arreglos de 1440 minutos con presencia, habilitación, alarma, CO2 y minutos al cruce."""
    df = pd.DataFrame(
        {
            "sala": rows["sala"].to_numpy(),
            "fecha": rows["tiempo"].dt.normalize().to_numpy(),
            "minuto": rows["minuto"].to_numpy().astype(int),
            "hab": rows["habilitado"].to_numpy(bool),
            "alarma": np.asarray(alarm, bool),
            "co2": rows["co2"].to_numpy(float),
            "cruce": rows["min_al_cruce"].to_numpy(float),
        }
    )
    out = {}
    for key, g in df.groupby(["sala", "fecha"], sort=False):
        m = g["minuto"].to_numpy()
        arr = {"pres": np.zeros(1440, bool), "hab": np.zeros(1440, bool), "alarma": np.zeros(1440, bool), "co2": np.full(1440, np.nan), "cruce": np.full(1440, np.inf)}
        for k in ("hab", "alarma", "co2", "cruce"):
            arr[k][m] = g[k].to_numpy()
        arr["pres"][m] = True
        out[key] = arr
    return out


def alarm_evaluation(rows: pd.DataFrame, alarm, crosses: pd.DataFrame, max_lead: int = 60, merge_gap: int = 2):
    """Anticipación de cada cruce medido y falsas alarmas, para una regla de alarma.

    - Anticipación: minutos seguidos con alarma (o ya por encima del umbral sin
      confirmar) que terminan justo antes del cruce, hasta `max_lead`. Se corta
      en un minuto habilitado sin alarma, en un minuto sin origen y en el período
      posterior a un cruce anterior.
    - Rachas de alarma: minutos habilitados con alarma, unidos si los separan a
      lo sumo `merge_gap` minutos. Una racha es falsa si no hay un cruce medido en
      los 60 min que siguen a su inicio.

    Devuelve (cruces con su anticipación, rachas).
    """
    days = _day_arrays(rows, alarm)
    leads = []
    for c in crosses.itertuples(index=False):
        arr = days.get((c.sala, c.fecha))
        lead = 0
        if arr is not None:
            for t in range(c.minuto - 1, max(c.minuto - 1 - max_lead, -1), -1):
                if not arr["pres"][t]:
                    break
                if arr["hab"][t]:
                    if not arr["alarma"][t]:
                        break
                elif arr["co2"][t] < THRESHOLD:  # después de un cruce anterior, sin volver a habilitarse
                    break
                lead += 1
        leads.append(lead)
    crosses = crosses.assign(anticipacion=leads)

    runs = []
    for (sala, fecha), arr in days.items():
        on = arr["hab"] & arr["alarma"]
        idx = np.flatnonzero(on)
        if not len(idx):
            continue
        breaks = np.flatnonzero(np.diff(idx) > merge_gap + 1)
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        for s, e in zip(starts, ends):
            runs.append({"sala": sala, "fecha": fecha, "inicio": int(s), "fin": int(e), "verdadera": bool(arr["cruce"][s] <= max_lead)})
    return crosses, pd.DataFrame(runs, columns=["sala", "fecha", "inicio", "fin", "verdadera"])


def alarm_summary(crosses: pd.DataFrame, runs: pd.DataFrame, n_days: int, leads=(5, 10, 15, 30)) -> dict:
    out = {"cruces": len(crosses)}
    for L in leads:
        out[f"anticipados_{L}"] = float((crosses["anticipacion"] >= L).mean()) if len(crosses) else np.nan
    out["anticipacion_mediana"] = float(crosses["anticipacion"].median()) if len(crosses) else np.nan
    out["rachas"] = len(runs)
    out["falsas"] = int((~runs["verdadera"]).sum()) if len(runs) else 0
    out["falsas_por_dia"] = out["falsas"] / max(n_days, 1)
    out["precision"] = float(runs["verdadera"].mean()) if len(runs) else np.nan
    return out


def alarm_minutes(rows: pd.DataFrame, alarm, n_days: int, max_lead: int = 60) -> dict:
    """Minutos por día con alarma y, de ellos, cuántos sin un cruce medido en los `max_lead` minutos siguientes.

    Complementa el conteo de rachas de `alarm_summary`, que premia a una alarma
    que no se apaga: encendida todo el día, es una sola racha.
    """
    on = rows["habilitado"].to_numpy(bool) & np.asarray(alarm, bool)
    false = on & (rows["min_al_cruce"].to_numpy(float) > max_lead)
    return {"minutos_alarma_por_dia": on.sum() / max(n_days, 1), "minutos_falsos_por_dia": false.sum() / max(n_days, 1)}


def timing_errors(rows: pd.DataFrame, t_hat, early=None, late=None, max_lead: int = 60) -> pd.DataFrame:
    """Para cada origen habilitado con alarma y un cruce medido en los próximos 60 min: cuánto faltaba y el error del instante pronosticado."""
    lead = rows["min_al_cruce"].to_numpy(float)
    ok = rows["habilitado"].to_numpy(bool) & np.isfinite(t_hat) & (lead <= max_lead)
    df = pd.DataFrame({"faltaba": lead[ok], "error": np.asarray(t_hat)[ok] - lead[ok]})
    if early is not None and late is not None:
        df["en_banda"] = (lead[ok] >= np.asarray(early)[ok]) & (lead[ok] <= np.asarray(late)[ok])
    return df


def probability_metrics(rows: pd.DataFrame, pw, bins: int = 10):
    """Brier, AUC y confiabilidad de P(cruce en los próximos h min), sobre los orígenes habilitados."""
    from sklearn.metrics import roc_auc_score

    ok = rows["habilitado"].to_numpy(bool)
    lead = rows["min_al_cruce"].to_numpy(float)[ok]
    p = np.clip(np.asarray(pw, float)[ok], 0.0, 1.0)
    summary, rel = [], []
    edges = np.linspace(0, 1, bins + 1)
    for j, h in enumerate(HORIZONS):
        y = (lead <= h).astype(float)
        pj = p[:, j]
        auc = roc_auc_score(y, pj) if 0 < y.sum() < len(y) else np.nan
        summary.append({"horizonte": h, "brier": float(np.mean((pj - y) ** 2)), "brier_ref": float(np.mean((y.mean() - y) ** 2)), "auc": auc, "positivos": float(y.mean()), "n": len(y)})
        k = np.clip(np.digitize(pj, edges) - 1, 0, bins - 1)
        for b in range(bins):
            sel = k == b
            if sel.any():
                rel.append({"horizonte": h, "bin": b, "p_media": float(pj[sel].mean()), "frecuencia": float(y[sel].mean()), "n": int(sel.sum())})
    return pd.DataFrame(summary), pd.DataFrame(rel)


# ------------------------------------------------------------------ importancia y estudio de información

IMPORTANCE_GROUPS = {
    "CO₂ y sus cambios": ["co2", *[f"d{k}" for k in LAGS if k <= FULL_HISTORY]],
    "pendientes y extremos": [c for c in g0_columns(FULL_HISTORY) if c.startswith(("pend", "curv", "max", "min", "sd"))],
    "hora y día": G1,
    "temperatura y humedad": G2,
    "historia del salón": G3,
    "salón": G4,
}

INFO_SETS = {
    "I0": (("G0",), FULL_HISTORY),
    "I1": (("G0", "G1"), FULL_HISTORY),
    "I2": (("G0", "G1", "G2"), FULL_HISTORY),
    "I3": (("G0", "G1", "G2", "G3"), FULL_HISTORY),
    "I4": (("G0", "G1", "G2", "G3", "G4"), FULL_HISTORY),
    "sin_G1": (("G0", "G2", "G3", "G4"), FULL_HISTORY),
    "sin_G2": (("G0", "G1", "G3", "G4"), FULL_HISTORY),
    "sin_G3": (("G0", "G1", "G2", "G4"), FULL_HISTORY),
    **{f"H{H}": (("G0", "G1", "G2", "G3", "G4"), H) for H in HISTORY if H != FULL_HISTORY},
}
INFO_NAMES = {
    "I0": "sólo CO₂",
    "I1": "+ hora y día",
    "I2": "+ temperatura y humedad",
    "I3": "+ historia del salón",
    "I4": "+ salón (completo)",
    "sin_G1": "completo sin hora y día",
    "sin_G2": "completo sin temperatura y humedad",
    "sin_G3": "completo sin historia del salón",
    "sin_G4": "completo sin salón",
}


def info_columns(key: str) -> list:
    groups, history = INFO_SETS[key]
    return feature_set(groups, history)


def permutation_importance(model: GBM, rows: pd.DataFrame, n: int = 200_000, seed: int = 0) -> pd.DataFrame:
    """Cuánto sube el MAE de cada horizonte al desordenar juntas las variables de un grupo."""
    rng = np.random.default_rng(seed)
    sub = rows.iloc[rng.choice(len(rows), min(n, len(rows)), replace=False)].reset_index(drop=True)
    dy = sub[TARGETS].to_numpy(float)  # el cambio medido; el C(t) que se le suma no se desordena
    base = np.mean(np.abs(model._raw("mid", sub) - dy), axis=0)
    out = []
    for name, cols in IMPORTANCE_GROUPS.items():
        cols = [c for c in cols if c in model.cols]
        if not cols:
            continue
        shuffled = sub.copy()
        perm = rng.permutation(len(sub))
        shuffled[cols] = sub[cols].to_numpy()[perm]
        mae = np.mean(np.abs(model._raw("mid", shuffled) - dy), axis=0)
        for j, h in enumerate(HORIZONS):
            out.append({"grupo": name, "horizonte": h, "mae_base": base[j], "aumento_mae": mae[j] - base[j]})
    return pd.DataFrame(out)
