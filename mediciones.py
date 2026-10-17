"""Mediciones reales de CO2 en salones: carga, series diarias y los métodos del estudio sobre cada día.

Los archivos de `mediciones/` traen una fila por lectura (tiempo, id, co2, temp,
hum, ip); un salón es una pareja (ip, id). Cada sensor reporta cada ~32 s, con
huecos. Acá no hay verdad de ningún tipo: ni cortes ni ACH. Lo que se puede
mirar es cómo segmenta cada método, qué tan bien ajusta, qué incertidumbre
informa y cuánto coinciden los métodos entre sí.

La unidad de análisis es el día (0 a 24 h, hora local de los datos), llevado a
una grilla regular de 30 s como las series sintéticas. Los métodos y sus
parámetros son los de la comparación base (`pipeline.METHODS`): sólo
transitorio y generación libre, porque no se conoce la ocupación.

Se corren de dos maneras. "Tal cual" usa el ruido que cada método estima de la
serie, que acá es el del sensor (menos de 1 ppm). "Ruido efectivo" le pone a
ese ruido un piso igual a la fluctuación real de la señal en la escala de las
ventanas de los métodos (`effective_noise`): con gente en el salón el CO2 se
mueve decenas de ppm alrededor de cualquier exponencial, y un método que cree
que el ruido es de 1 ppm toma cada una de esas oscilaciones por un evento.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cfd import ObservedSeries
from .fitting import fit_bounded_source, fit_exponential, fit_known_source
from .physics import bmr_schofield, co2_generation, source_from_generation
from .pipeline import run_all
from .segmenters.base import DECAY, RISE
from .signal_utils import despike, median_filter, noise_sd

METHODS = ("M1", "M2", "M3a", "M3b", "M4", "M5", "M6")
DT_S = 30.0
CONFIGS = ("tal_cual", "ruido_efectivo")


# ------------------------------------------------------------------ carga


def load_room(folder, ip: str, sensor_id: int, cache_dir=None) -> pd.DataFrame:
    """Lecturas de un salón (ip, id), ordenadas por tiempo. Guarda una copia en `cache_dir` para no releer todo."""
    cache = Path(cache_dir) / f"{ip}_id{int(sensor_id):02d}.parquet" if cache_dir else None
    if cache is not None and cache.exists():
        return pd.read_parquet(cache)
    parts = []
    for path in sorted(Path(folder).glob("*.csv")):
        for chunk in pd.read_csv(path, dtype={"ip": "string"}, chunksize=2_000_000):
            parts.append(chunk[(chunk["ip"] == ip) & (chunk["id"] == sensor_id)])
    if not parts or not sum(len(p) for p in parts):
        raise ValueError(f"no hay lecturas de ip = {ip!r}, id = {sensor_id}")
    df = pd.concat(parts, ignore_index=True)[["tiempo", "co2", "temp", "hum"]]
    df["tiempo"] = pd.to_datetime(df["tiempo"])
    df = df.sort_values("tiempo").drop_duplicates("tiempo").reset_index(drop=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


# ------------------------------------------------------------------ línea de base y días


def daily_table(df: pd.DataFrame, max_gap_min: float = 10.0, window_days: int = 7, min_range_ppm: float = 150.0) -> pd.DataFrame:
    """Una fila por día: cuántas lecturas hay, el mayor hueco, el rango de CO2 y el CO2 exterior supuesto.

    - `c_ext`: el mínimo, en una ventana de `window_days` días centrada en el
      día, del percentil 1 diario. En un día de clase el CO2 no llega a bajar al
      nivel exterior antes de la mañana siguiente; el fin de semana sí. Tomarlo
      de la propia serie absorbe además la deriva del cero del sensor.
    - `completo`: ningún hueco mayor que `max_gap_min` dentro del día.
    - `ocupado`: el CO2 recorre al menos `min_range_ppm` (percentil 99 menos
      percentil 1). Los demás días no tienen eventos que segmentar.
    """
    t = df["tiempo"]
    day = t.dt.normalize()
    g = df.groupby(day)["co2"]
    out = pd.DataFrame({"lecturas": g.size(), "p1": g.quantile(0.01), "p99": g.quantile(0.99), "maximo": g.max()})
    days = pd.date_range(day.min(), day.max(), freq="D")
    out = out.reindex(days)
    out.index.name = "fecha"
    out["lecturas"] = out["lecturas"].fillna(0).astype(int)

    # mayor hueco dentro de cada día, contando el borde con los días vecinos
    stamps = t.to_numpy()
    gap = np.diff(stamps) / np.timedelta64(1, "m")
    worst = pd.Series(0.0, index=days)
    for i in np.flatnonzero(gap > max_gap_min):
        for d in pd.date_range(pd.Timestamp(stamps[i]).normalize(), pd.Timestamp(stamps[i + 1]).normalize(), freq="D"):
            if d in worst.index:
                worst[d] = max(worst[d], float(gap[i]))
    out["hueco_max_min"] = worst
    out["completo"] = (out["lecturas"] > 0) & (out["hueco_max_min"] <= max_gap_min)
    out["c_ext"] = out["p1"].rolling(window_days, center=True, min_periods=1).min()
    out["rango"] = out["p99"] - out["p1"]
    out["ocupado"] = out["rango"] >= min_range_ppm
    out["dia_semana"] = out.index.dayofweek
    return out


def day_series(df: pd.DataFrame, day, c_ext: float, dt_s: float = DT_S) -> ObservedSeries:
    """La serie de un día en una grilla regular de `dt_s` segundos, interpolando linealmente entre lecturas."""
    day = pd.Timestamp(day).normalize()
    lo, hi = day - pd.Timedelta(minutes=15), day + pd.Timedelta(days=1, minutes=15)
    t = df["tiempo"].to_numpy()
    i0, i1 = np.searchsorted(t, [np.datetime64(lo), np.datetime64(hi)])
    sec = (t[i0:i1] - np.datetime64(day)) / np.timedelta64(1, "s")
    grid = np.arange(int(round(86400.0 / dt_s))) * dt_s
    c = np.interp(grid, sec, df["co2"].to_numpy()[i0:i1])
    return ObservedSeries(name=day.strftime("%Y-%m-%d"), t_h=grid / 3600.0, c_obs=c, c_ext_nominal=float(c_ext))


# ------------------------------------------------------------------ ruido


def lag_fluctuation(x, k: int) -> np.ndarray:
    """x[i] − (x[i−k] + x[i+k])/2, en unidades de desvío de ruido blanco (dividido por √1,5). NaN en los bordes."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) > 2 * k:
        out[k:-k] = (x[k:-k] - 0.5 * (x[: -2 * k] + x[2 * k :])) / np.sqrt(1.5)
    return out


def robust_sd(values) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def occupied_mask(series: ObservedSeries, min_excess_ppm: float = 150.0, trend_min: float = 15.0) -> np.ndarray:
    """Muestras del período con gente: desde la primera hasta la última vez que el CO2 sube estando por encima del exterior.

    "Sube" es que la serie filtrada con mediana esté más alta que `trend_min`
    minutos antes; "por encima del exterior", que lo supere en `min_excess_ppm`.
    Lo que queda afuera es la noche y la bajada larga después de la última
    salida, donde no hay nada que fluctúe.
    """
    c = smooth(series.c_obs)
    k = max(1, int(round(trend_min * 60.0 / (series.dt_h * 3600.0))))
    rising = np.zeros(len(c), dtype=bool)
    rising[k:] = c[k:] > c[:-k]
    idx = np.flatnonzero(rising & (c - series.c_ext_nominal >= min_excess_ppm))
    mask = np.zeros(len(c), dtype=bool)
    if len(idx):
        mask[max(idx[0] - k, 0) : idx[-1] + 1] = True
    return mask


def effective_noise(series: ObservedSeries, lag_min: float = 5.0) -> float:
    """Fluctuación de la señal que los métodos tienen que tratar como ruido [ppm].

    Es el desvío robusto de la diferencia entre cada muestra y el promedio de
    las que están `lag_min` minutos antes y después, sobre el período con gente
    (`occupied_mask`). Con ruido blanco coincide con su desvío. La escala de 5
    minutos es la mitad de las ventanas de 10 minutos de los métodos: lo que
    cambia más rápido que eso no lo pueden seguir. Incluye algo de la curvatura
    propia de las exponenciales, así que sobrestima un poco.
    """
    k = max(1, int(round(lag_min * 60.0 / (series.dt_h * 3600.0))))
    d = lag_fluctuation(series.c_obs, k)[occupied_mask(series)]
    return robust_sd(d) if np.isfinite(d).sum() >= 60 else np.nan


def noise_spectrum(series: ObservedSeries, lags_min=(0.5, 1.0, 2.0, 5.0, 10.0)) -> dict:
    """Fluctuación según la escala de tiempo, en el período con gente ("ocupado") y en el resto del día."""
    high = occupied_mask(series)
    out = {"sensor": noise_sd(series.c_obs), "horas_con_gente": float(high.sum() * series.dt_h)}
    for lag in lags_min:
        k = max(1, int(round(lag * 60.0 / (series.dt_h * 3600.0))))
        d = lag_fluctuation(series.c_obs, k)
        out[f"ocupado_{lag:g}"] = robust_sd(d[high])
        out[f"resto_{lag:g}"] = robust_sd(d[~high])
    return out


# ------------------------------------------------------------------ métodos sobre un día


def fit_rows(series: ObservedSeries, results: dict) -> list:
    """Una fila por ajuste de cada método."""
    t = series.t_h
    rows = []
    for key, res in results.items():
        for f in res.fits:
            ok = bool(f.ok and np.isfinite(f.lam))
            curve = f.predict(t[[f.i0, f.i1 - 1]]) if ok else np.array([np.nan, np.nan])
            rows.append(
                {
                    "fecha": series.name,
                    "metodo": key,
                    "t0_h": t[f.i0],
                    "t1_h": t[f.i1 - 1],
                    "dur_min": 60.0 * (t[f.i1 - 1] - t[f.i0]),
                    "fase": "subida" if f.phase == RISE else "bajada",
                    "aceptado": bool(f.accepted),
                    "motivo": f.reason,
                    "ach": f.lam if ok else np.nan,
                    "ach_se": f.lam_se if ok else np.nan,
                    "c0": f.c0 if ok else np.nan,
                    "c_inf": f.css if ok else np.nan,
                    "excursion_ppm": abs(float(curve[1] - curve[0])),
                    "asintota_en_c_ext": bool(ok and f.css <= series.c_ext_nominal + 1.0),
                    "r2": f.r2,
                    "rmse": f.rmse,
                    "i0": f.i0,
                    "i1": f.i1,
                }
            )
    return rows


def segment_rows(series: ObservedSeries, results: dict) -> list:
    """Una fila por tramo que detecta cada método (se ajuste o no)."""
    t = series.t_h
    return [
        {"fecha": series.name, "metodo": key, "t0_h": t[s.i0], "t1_h": t[s.i1 - 1], "fase": "subida" if s.phase == RISE else "bajada"}
        for key, res in results.items()
        for s in res.segmentation.transients
    ]


def analyze_day(name: str, t_h, c_obs, c_ext: float, methods=METHODS, keep: bool = False) -> dict:
    """Corre los métodos sobre un día, tal cual y con el ruido efectivo.

    Devuelve las filas de ajustes y de tramos (con la columna `config`), el
    ruido del sensor y el efectivo, y los resultados completos si `keep`.
    """
    series = ObservedSeries(name=name, t_h=np.asarray(t_h), c_obs=np.asarray(c_obs), c_ext_nominal=float(c_ext))
    np.seterr(all="ignore")
    sigma_ef = effective_noise(series)
    fits, segs, kept = [], [], {}
    for config in CONFIGS:
        floor = 0.0 if config == "tal_cual" or not np.isfinite(sigma_ef) else sigma_ef
        results = run_all(series, keys=methods, sigma_min=floor)
        fits += [dict(r, config=config) for r in fit_rows(series, results)]
        segs += [dict(r, config=config) for r in segment_rows(series, results)]
        if keep:
            kept[config] = results
    info = {"fecha": name, "ruido_sensor": noise_sd(series.c_obs), "ruido_efectivo": sigma_ef, **noise_spectrum(series)}
    return {"fits": fits, "segs": segs, "info": info, "kept": kept if keep else None}


def run_days(df: pd.DataFrame, days: pd.DataFrame, methods=METHODS, keep=(), n_jobs: int = 1):
    """Los métodos sobre cada día de `days` (filas de `daily_table`).

    Devuelve (ajustes, tramos, tabla por día con los ruidos, {fecha: {config: resultados}} de las fechas en `keep`).
    """
    from joblib import Parallel, delayed

    keep = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in keep}
    tasks = []
    for day, row in days.iterrows():
        s = day_series(df, day, row["c_ext"])
        tasks.append(delayed(analyze_day)(s.name, s.t_h, s.c_obs, s.c_ext_nominal, methods, s.name in keep))
    out = Parallel(n_jobs=n_jobs, verbose=0)(tasks)
    fits = pd.DataFrame([r for o in out for r in o["fits"]])
    segs = pd.DataFrame([r for o in out for r in o["segs"]])
    info = pd.DataFrame([o["info"] for o in out]).set_index("fecha")
    kept = {o["info"]["fecha"]: o["kept"] for o in out if o["kept"] is not None}
    return fits, segs, info, kept


# ------------------------------------------------------------------ resúmenes


@dataclass(frozen=True)
class Selection:
    """Qué ajustes cuentan como evento y cuáles se reportan."""

    min_excursion_ppm: float = 100.0  # recorrido mínimo de la curva ajustada para contar como evento
    max_rmse_rel: float = 0.05  # el ajuste se reporta si su RMSE no pasa de esta fracción del recorrido...
    min_lam_t: float = 1.0  # ...y el tramo dura al menos estas constantes de tiempo (ACH × duración)


def events(fits: pd.DataFrame, sel: Selection = Selection()) -> pd.DataFrame:
    """Ajustes aceptados que recorren al menos `min_excursion_ppm`, con sus medidas de calidad.

    - `rmse_rel`: RMSE del ajuste sobre el recorrido de la curva. Mide si la
      curva se parece a una exponencial.
    - `lam_t`: ACH por duración del tramo, es decir, cuántas constantes de
      tiempo se observaron. Mide si el tramo alcanza para fijar el ACH: con
      menos de una, la curva es casi una recta y cualquier ACH la ajusta.
    - `se_rel`: error estándar relativo del ACH que informa el ajuste.
    - `reportado`: cumple los dos umbrales de `sel`.
    """
    ev = fits[fits["aceptado"] & (fits["excursion_ppm"] >= sel.min_excursion_ppm)].copy()
    ev["se_rel"] = ev["ach_se"] / ev["ach"]
    ev["rmse_rel"] = ev["rmse"] / ev["excursion_ppm"]
    ev["lam_t"] = ev["ach"] * ev["dur_min"] / 60.0
    ev["reportado"] = (ev["rmse_rel"] <= sel.max_rmse_rel) & (ev["lam_t"] >= sel.min_lam_t)
    return ev


def method_summary(fits: pd.DataFrame, n_days: int, sel: Selection = Selection()) -> pd.DataFrame:
    """Por método y fase: cuántos ajustes hay, cuántos quedan en cada filtro y cómo se reparte el ACH."""
    ev = events(fits, sel)
    rows = []
    for (key, phase), g in fits.groupby(["metodo", "fase"], sort=False):
        acc = g[g["aceptado"]]
        e = ev[(ev["metodo"] == key) & (ev["fase"] == phase)]
        good = e[e["reportado"]]
        row = {
            "metodo": key,
            "fase": phase,
            "tramos_por_dia": len(g) / n_days,
            "aceptados_por_dia": len(acc) / n_days,
            "eventos_por_dia": len(e) / n_days,
            "reportados_por_dia": len(good) / n_days,
            "dias_con_evento": e["fecha"].nunique(),
            "duracion_mediana_min": float(e["dur_min"].median()) if len(e) else np.nan,
            "rmse_rel_mediano": float(e["rmse_rel"].median()) if len(e) else np.nan,
            "lam_t_mediano": float(e["lam_t"].median()) if len(e) else np.nan,
            "se_rel_mediano": float(e["se_rel"].median()) if len(e) else np.nan,
        }
        for name, sub in (("eventos", e), ("reportados", good)):
            q = sub["ach"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]) if len(sub) else pd.Series([np.nan] * 5, index=[0.1, 0.25, 0.5, 0.75, 0.9])
            row.update({f"ach_p10_{name}": q[0.1], f"ach_p25_{name}": q[0.25], f"ach_mediana_{name}": q[0.5], f"ach_p75_{name}": q[0.75], f"ach_p90_{name}": q[0.9]})
        rows.append(row)
    return pd.DataFrame(rows)


def matched_pairs(ev: pd.DataFrame, min_overlap: float = 0.5) -> pd.DataFrame:
    """Emparejamientos entre eventos de métodos distintos que ajustan lo mismo.

    Dos eventos se emparejan si son del mismo día y la misma fase y se solapan
    al menos `min_overlap` de la duración del más corto; cada evento de un
    método va con el que más se solapa de cada uno de los otros. Devuelve una
    fila por emparejamiento con las posiciones `i`, `j` en `ev` (índice
    reiniciado), los métodos y `dlog` = |ln(ACH_i / ACH_j)|.
    """
    ev = ev.reset_index(drop=True)
    rows = []
    for (_, phase), g in ev.groupby(["fecha", "fase"]):
        idx = g.index.to_numpy()
        t0, t1, m, lam = g["t0_h"].to_numpy(), g["t1_h"].to_numpy(), g["metodo"].to_numpy(), g["ach"].to_numpy()
        for i in range(len(g)):
            ov = np.minimum(t1[i], t1) - np.maximum(t0[i], t0)
            ok = (m != m[i]) & (ov > 0) & (ov >= min_overlap * np.minimum(t1[i] - t0[i], t1 - t0))
            for other in np.unique(m[ok]):
                cand = np.flatnonzero(ok & (m == other))
                j = cand[np.argmax(ov[cand])]
                rows.append((phase, idx[i], idx[j], m[i], m[j], abs(np.log(lam[i] / lam[j]))))
    return pd.DataFrame(rows, columns=["fase", "i", "j", "a", "b", "dlog"])


def _pct(dlog) -> float:
    """Mediana de |ln cociente| como diferencia relativa en porcentaje."""
    return 100.0 * (np.exp(np.median(dlog)) - 1.0) if len(dlog) else np.nan


def agreement(ev: pd.DataFrame, pairs: pd.DataFrame, methods=METHODS, only_reported: bool = False) -> pd.DataFrame:
    """Diferencia mediana de ACH entre cada pareja de métodos sobre los eventos que comparten, por fase.

    `ev` y `pairs` son los de `events` (con el índice reiniciado) y
    `matched_pairs`. Con `only_reported` sólo cuentan los emparejamientos en que
    los dos eventos se reportan. Sin verdad, es la única medida de consistencia
    disponible.
    """
    if only_reported:
        rep = ev["reportado"].to_numpy()
        pairs = pairs[rep[pairs["i"].to_numpy()] & rep[pairs["j"].to_numpy()]]
    rows = []
    for phase in ("subida", "bajada"):
        sub = pairs[pairs["fase"] == phase]
        for k, a in enumerate(methods):
            for b in methods[:k]:
                d = sub[((sub["a"] == a) & (sub["b"] == b)) | ((sub["a"] == b) & (sub["b"] == a))]["dlog"]
                rows.append({"fase": phase, "a": a, "b": b, "emparejamientos": len(d), "desacuerdo_mediano_pct": _pct(d)})
    return pd.DataFrame(rows)


THRESHOLDS = {
    "r2": ("ge", (0.1, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99)),
    "rmse_rel": ("le", (1.0, 0.2, 0.1, 0.05, 0.03, 0.02)),
    "se_rel": ("le", (10.0, 0.5, 0.25, 0.1, 0.05, 0.02)),
    "lam_t": ("ge", (0.0, 0.3, 0.5, 1.0, 2.0, 3.0)),
}


def threshold_study(ev: pd.DataFrame, pairs: pd.DataFrame, exclude=("M1",), thresholds=THRESHOLDS) -> pd.DataFrame:
    """Qué pasa al exigir más calidad a los ajustes: cuántos eventos quedan y cuánto coinciden los métodos.

    Para cada criterio (R², RMSE relativo, error estándar relativo y duración en
    constantes de tiempo) y cada umbral devuelve, por fase, el porcentaje de
    eventos que quedan y la diferencia de ACH entre métodos (mediana y cuartil
    superior) sobre los emparejamientos en que los dos eventos pasan el umbral.
    El primer umbral de cada criterio no descarta casi nada y sirve de
    referencia. `exclude` saca métodos del recuento: las subidas de Hyun fiel
    son las mismas que las de Hyun + decaimientos.
    """
    ev = ev.reset_index(drop=True)
    use = ~ev["metodo"].isin(exclude).to_numpy()
    pairs = pairs[use[pairs["i"].to_numpy()] & use[pairs["j"].to_numpy()]]
    rows = []
    for name, (op, values) in thresholds.items():
        col = ev[name].to_numpy(dtype=float)
        for th in values:
            keep = ((col >= th) if op == "ge" else (col <= th)) & use
            for phase in ("subida", "bajada"):
                of_phase = (ev["fase"] == phase).to_numpy() & use
                p = pairs[(pairs["fase"] == phase) & keep[pairs["i"].to_numpy()] & keep[pairs["j"].to_numpy()]]
                rows.append(
                    {
                        "criterio": name,
                        "umbral": th,
                        "fase": phase,
                        "quedan_pct": 100.0 * (keep & of_phase).sum() / max(of_phase.sum(), 1),
                        "emparejamientos": len(p),
                        "diferencia_mediana_pct": _pct(p["dlog"]),
                        "diferencia_p75_pct": 100.0 * (np.exp(p["dlog"].quantile(0.75)) - 1.0) if len(p) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def rule_effect(ev: pd.DataFrame, pairs: pd.DataFrame, exclude=("M1",)) -> pd.DataFrame:
    """Lo mismo que `threshold_study` para la regla de `reportado`: antes y después, por fase."""
    ev = ev.reset_index(drop=True)
    use = ~ev["metodo"].isin(exclude).to_numpy()
    pairs = pairs[use[pairs["i"].to_numpy()] & use[pairs["j"].to_numpy()]]
    rep = ev["reportado"].to_numpy() & use
    rows = []
    for phase in ("subida", "bajada"):
        of_phase = (ev["fase"] == phase).to_numpy() & use
        sub = pairs[pairs["fase"] == phase]
        after = sub[rep[sub["i"].to_numpy()] & rep[sub["j"].to_numpy()]]
        rows.append(
            {
                "fase": phase,
                "eventos": int(of_phase.sum()),
                "reportados_pct": 100.0 * (rep & of_phase).sum() / max(of_phase.sum(), 1),
                "diferencia_mediana_antes_pct": _pct(sub["dlog"]),
                "diferencia_mediana_despues_pct": _pct(after["dlog"]),
                "diferencia_p75_antes_pct": 100.0 * (np.exp(sub["dlog"].quantile(0.75)) - 1.0) if len(sub) else np.nan,
                "diferencia_p75_despues_pct": 100.0 * (np.exp(after["dlog"].quantile(0.75)) - 1.0) if len(after) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ ocupación conocida

HYUN_FAMILY = ("M1", "M2", "M3a", "M3b")  # ajustan sobre la mediana móvil y con C0 fijo en la primera muestra


def clock(h: float) -> str:
    m = int(round(h * 60.0))
    return f"{m // 60:02d}:{m % 60:02d}"


@dataclass(frozen=True)
class Occupancy:
    """Horario del salón: intervalos [desde, hasta) en horas del día en los que está vacío, de lunes a viernes."""

    empty: tuple

    @classmethod
    def parse(cls, items) -> "Occupancy":
        """Desde textos "HH:MM-HH:MM"."""

        def hours(text):
            h, _, m = text.partition(":")
            return int(h) + (int(m) if m else 0) / 60.0

        return cls(tuple(tuple(hours(p) for p in item.split("-")) for item in items))

    @property
    def labels(self) -> list:
        return [f"{clock(a)}–{clock(b)}" for a, b in self.empty]

    def is_empty(self, t_h) -> np.ndarray:
        t_h = np.asarray(t_h, dtype=float)
        return np.any([(t_h >= a) & (t_h < b) for a, b in self.empty], axis=0)

    def cover(self, t0_h: float, t1_h: float):
        """Fracción del tramo [t0, t1] que cae en horario vacío y ventana que más aporta."""
        parts = [max(0.0, min(t1_h, b) - max(t0_h, a)) for a, b in self.empty]
        j = int(np.argmax(parts))
        return sum(parts) / max(t1_h - t0_h, 1e-9), (self.labels[j] if parts[j] > 0 else "")


def schedule_check(df: pd.DataFrame, days: pd.DataFrame, occ: Occupancy, slope_ppm_h: float = 200.0, bin_min: float = 5.0, lag_min: float = 5.0) -> pd.DataFrame:
    """¿El horario coincide con los datos? Por franja del día, porcentaje de días hábiles en que el CO2 sube y en que baja.

    "Sube" o "baja" es que la mediana móvil cambie más de `slope_ppm_h` entre
    `lag_min` minutos antes y `lag_min` después. Con el salón vacío el CO2 no
    puede subir: lo que suba en horario vacío mide cuánto se aparta el horario
    real del nominal.
    """
    week = days[days["dia_semana"] < 5]
    rise, fall, excess = [], [], []
    for day, row in week.iterrows():
        s = day_series(df, day, row["c_ext"])
        c = smooth(s.c_obs)
        k = max(1, int(round(lag_min / 60.0 / s.dt_h)))
        d = np.full(len(c), np.nan)
        d[k:-k] = (c[2 * k :] - c[: -2 * k]) / (2 * k * s.dt_h)
        rise.append(d > slope_ppm_h)
        fall.append(d < -slope_ppm_h)
        excess.append(c - row["c_ext"])
    rise, fall, excess = np.array(rise), np.array(fall), np.array(excess)
    per = max(1, int(round(bin_min / 60.0 / s.dt_h)))
    n = (rise.shape[1] // per) * per
    hour = s.t_h[:n:per]
    return pd.DataFrame(
        {
            "hora": hour,
            "sube_pct": 100.0 * rise[:, :n].reshape(len(week), -1, per).mean(axis=(0, 2)),
            "baja_pct": 100.0 * fall[:, :n].reshape(len(week), -1, per).mean(axis=(0, 2)),
            "exceso_mediano_ppm": np.median(excess[:, :n].reshape(len(week), -1, per), axis=(0, 2)),
            "vacio": occ.is_empty(hour + 0.5 * per * s.dt_h),
        }
    )


def _day_data(df: pd.DataFrame, days: pd.DataFrame, date: str, noise):
    """La serie de un día y las dos versiones que ajustan los métodos: sin outliers y con mediana móvil."""
    day = pd.Timestamp(date)
    series = day_series(df, day, days.loc[day, "c_ext"])
    floor = float(noise.get(date, 0.0))
    clean, _ = despike(series.c_obs, max(noise_sd(series.c_obs), floor if np.isfinite(floor) else 0.0))
    return series, clean, smooth(series.c_obs)


PURE_COLUMNS = ("ach_puro", "ach_se_puro", "c0_puro", "c_inf_puro", "rmse_puro", "r2_puro", "excursion_puro")


def refit_empty(df: pd.DataFrame, days: pd.DataFrame, fits: pd.DataFrame, noise, occ: Occupancy, min_inside: float = 0.8) -> pd.DataFrame:
    """Repite, con la generación en cero, el ajuste de las bajadas que caen en horario vacío.

    Con el salón vacío el balance no tiene fuente y la asíntota de la bajada es
    el CO2 exterior: queda un parámetro menos y el ACH sale de la pendiente de
    ln(C − C_ext). Se reajustan las bajadas aceptadas de días hábiles con al
    menos `min_inside` de su duración en horario vacío, sobre los mismos datos
    que usó el método: la familia de Hyun, la mediana móvil con C0 fijo en la
    primera muestra; el resto, la serie sin outliers con C0 libre. `noise` es el
    piso de ruido de cada día (el de la corrida que produjo `fits`).

    Devuelve `fits` con las columnas `vacio` (fracción del tramo en horario
    vacío), `ventana` y, para las reajustadas, las de `PURE_COLUMNS`. Un
    reajuste con λ en la cota del optimizador queda sin valor.
    """
    out = fits.copy()
    cover = [occ.cover(a, b) for a, b in zip(out["t0_h"], out["t1_h"])]
    weekday = (pd.to_datetime(out["fecha"]).dt.dayofweek < 5).to_numpy()
    out["vacio"] = np.where(weekday, [c[0] for c in cover], 0.0)
    out["ventana"] = np.where(weekday, [c[1] for c in cover], "")
    values = np.full((len(out), len(PURE_COLUMNS)), np.nan)
    todo = np.flatnonzero((out["aceptado"] & (out["fase"] == "bajada") & (out["vacio"] >= min_inside)).to_numpy())
    dates, i0s, i1s, keys = (out[c].to_numpy() for c in ("fecha", "i0", "i1", "metodo"))
    for date in pd.unique(dates[todo]):
        series, clean, filt = _day_data(df, days, date, noise)
        for r in todo[dates[todo] == date]:
            sl = slice(int(i0s[r]), int(i1s[r]))
            hyun = keys[r] in HYUN_FAMILY
            y = (filt if hyun else clean)[sl]
            f = fit_exponential(series.t_h[sl], y, c_ext=series.c_ext_nominal, c0=float(y[0]) if hyun else None, css=series.c_ext_nominal, i0=sl.start, model="decaimiento puro", phase=DECAY)
            if f.ok and not f.at_bound:
                ends = f.predict(series.t_h[[sl.start, sl.stop - 1]])
                values[r] = (f.lam, f.lam_se, f.c0, f.css, f.rmse, f.r2, abs(float(ends[1] - ends[0])))
    for k, name in enumerate(PURE_COLUMNS):
        out[name] = values[:, k]
    return out


@dataclass(frozen=True)
class Group:
    """Una clase: niños de una edad y una masa corporal dadas, con su nivel de actividad, y una maestra."""

    children: int
    age: float
    mass_kg: float
    met: float
    teacher_met: float
    teacher_bmr: float = 6.1  # MJ/día: mujer de 30 a 50 años (Persily y de Jonge, 2017, tabla 2)
    bmr: float | None = None  # MJ/día de cada niño; si no se da, el de Schofield para su masa y edad (promedio de varones y mujeres)

    def generation(self, temp_c: float = 20.0) -> dict:
        """Generación de CO2 [L/h] de un niño, de la maestra y de toda la clase."""
        child = co2_generation(self.bmr if self.bmr is not None else bmr_schofield(self.mass_kg, self.age), self.met, temp_c=temp_c)
        teacher = co2_generation(self.teacher_bmr, self.teacher_met, temp_c=temp_c)
        return {"nino": child, "maestra": teacher, "total": self.children * child + teacher}

    def source(self, volume_m3: float, temp_c: float = 20.0) -> float:
        """Fuente S [ppm/h] de la clase en un salón de `volume_m3`."""
        return float(source_from_generation(self.generation(temp_c)["total"], volume_m3))


def generation_table(groups: dict, volume_m3: float, temp_c: float = 20.0) -> pd.DataFrame:
    """Los casos de generación, con lo que supone cada uno y la fuente que resulta."""
    rows = []
    for name, g in groups.items():
        gen = g.generation(temp_c)
        rows.append(
            {
                "caso": name,
                "ninos": g.children,
                "edad": g.age,
                "masa_kg": g.mass_kg,
                "met_ninos": g.met,
                "met_maestra": g.teacher_met,
                "l_h_por_nino": gen["nino"],
                "l_h_maestra": gen["maestra"],
                "l_h_total": gen["total"],
                "volumen_m3": volume_m3,
                "fuente_ppm_h": g.source(volume_m3, temp_c),
            }
        )
    return pd.DataFrame(rows)


CASE_COLUMNS = ("ach", "ach_se", "c0", "c_inf", "rmse", "r2", "excursion")
BOUNDED = "acotada"  # sufijo de las columnas del ajuste con la generación entre el caso más bajo y el más alto


def refit_occupied(df: pd.DataFrame, days: pd.DataFrame, fits: pd.DataFrame, noise, occ: Occupancy, sources: dict, max_empty: float = 0.2, min_excursion_ppm: float = 100.0) -> pd.DataFrame:
    """Repite, con la generación conocida, el ajuste de los tramos que caen en horario con gente.

    `sources` da la fuente S [ppm/h] de cada caso (bajo, medio, alto). Con S
    conocida la asíntota queda atada al ACH, C_ss = C_ext + S/λ, y sólo se
    ajusta λ (y C0, salvo en la familia de Hyun), sobre los mismos datos que
    usó el método. Se reajustan los ajustes aceptados de días hábiles, subidas
    y bajadas, que recorren al menos `min_excursion_ppm` y tienen a lo sumo
    `max_empty` de su duración en horario vacío. Supone que en todo el horario
    con gente está la clase completa.

    Agrega `con_gente`, `c_ext` y, por caso, `ach_<caso>`, `ach_se_<caso>`,
    `c0_<caso>`, `c_inf_<caso>`, `rmse_<caso>`, `r2_<caso>`, `excursion_<caso>`
    y `cota_<caso>`: "baja" si λ queda en la cota inferior (esa generación no
    alcanza para explicar la curva) o "alta" si queda en la superior; en esos
    casos no hay ACH.
    """
    out = fits.copy()
    weekday = (pd.to_datetime(out["fecha"]).dt.dayofweek < 5).to_numpy()
    if "vacio" not in out:
        out["vacio"] = np.where(weekday, [occ.cover(a, b)[0] for a, b in zip(out["t0_h"], out["t1_h"])], 0.0)
    out["c_ext"] = pd.to_datetime(out["fecha"]).map(days["c_ext"]).to_numpy()
    busy = weekday & (out["vacio"] <= max_empty).to_numpy() & out["aceptado"].to_numpy() & (out["excursion_ppm"] >= min_excursion_ppm).to_numpy()
    out["con_gente"] = busy
    values = {case: np.full((len(out), len(CASE_COLUMNS)), np.nan) for case in sources}
    bound = {case: np.full(len(out), "", dtype=object) for case in sources}
    todo = np.flatnonzero(busy)
    dates, i0s, i1s, keys, phases = (out[c].to_numpy() for c in ("fecha", "i0", "i1", "metodo", "fase"))
    for date in pd.unique(dates[todo]):
        series, clean, filt = _day_data(df, days, date, noise)
        for r in todo[dates[todo] == date]:
            sl = slice(int(i0s[r]), int(i1s[r]))
            hyun = keys[r] in HYUN_FAMILY
            y = (filt if hyun else clean)[sl]
            for case, source in sources.items():
                f = fit_known_source(series.t_h[sl], y, c_ext=series.c_ext_nominal, s=float(source), c0=float(y[0]) if hyun else None, i0=sl.start, phase=RISE if phases[r] == "subida" else DECAY)
                if not f.ok:
                    continue
                if f.at_bound:
                    bound[case][r] = "baja" if f.lam < 1.0 else "alta"
                    continue
                ends = f.predict(series.t_h[[sl.start, sl.stop - 1]])
                values[case][r] = (f.lam, f.lam_se, f.c0, f.css, f.rmse, f.r2, abs(float(ends[1] - ends[0])))
    for case in sources:
        for k, name in enumerate(CASE_COLUMNS):
            out[f"{name}_{case}"] = values[case][:, k]
        out[f"cota_{case}"] = bound[case]
    if len(sources) >= 2:
        out = refit_bounded(df, days, out, noise, (min(sources.values()), max(sources.values())))
    return out


def refit_bounded(df: pd.DataFrame, days: pd.DataFrame, fits: pd.DataFrame, noise, limits: tuple, suffix: str = BOUNDED) -> pd.DataFrame:
    """Reajusta los tramos con gente (`con_gente`) con la generación libre entre dos cotas.

    `limits` son las fuentes S [ppm/h] mínima y máxima, por ejemplo las de los
    casos bajo y alto. Queda libre S dentro de ese intervalo, además de λ y,
    salvo en la familia de Hyun, C0 (`fitting.fit_bounded_source`). Agrega las
    columnas de `CASE_COLUMNS` con el sufijo `suffix`, `fuente_<suffix>` (la S
    que resulta) y `cota_<suffix>`: "inferior" o "superior" si S quedó en esa
    cota, "lambda" si λ quedó en la cota del optimizador (sin ACH) y vacío si S
    quedó adentro.
    """
    out = fits.copy()
    s_min, s_max = (float(v) for v in limits)
    values = np.full((len(out), len(CASE_COLUMNS) + 1), np.nan)
    side = np.full(len(out), "", dtype=object)
    todo = np.flatnonzero(out["con_gente"].to_numpy(dtype=bool))
    dates, i0s, i1s, keys, phases = (out[c].to_numpy() for c in ("fecha", "i0", "i1", "metodo", "fase"))
    for date in pd.unique(dates[todo]):
        series, clean, filt = _day_data(df, days, date, noise)
        for r in todo[dates[todo] == date]:
            sl = slice(int(i0s[r]), int(i1s[r]))
            hyun = keys[r] in HYUN_FAMILY
            y = (filt if hyun else clean)[sl]
            f = fit_bounded_source(series.t_h[sl], y, c_ext=series.c_ext_nominal, s_min=s_min, s_max=s_max, c0=float(y[0]) if hyun else None, i0=sl.start, phase=RISE if phases[r] == "subida" else DECAY)
            if not f.ok:
                continue
            if f.at_bound:
                side[r] = "lambda"
                continue
            side[r] = f.s_bound
            ends = f.predict(series.t_h[[sl.start, sl.stop - 1]])
            values[r] = (f.lam, f.lam_se, f.c0, f.css, f.rmse, f.r2, abs(float(ends[1] - ends[0])), f.s)
    for k, name in enumerate(CASE_COLUMNS):
        out[f"{name}_{suffix}"] = values[:, k]
    out[f"fuente_{suffix}"] = values[:, -1]
    out[f"cota_{suffix}"] = side
    return out


def known_occupancy(ev: pd.DataFrame, sel: Selection = Selection(), min_inside: float = 0.8, case: str | None = None) -> pd.DataFrame:
    """Los eventos tal como quedan al usar lo que se sabe de la ocupación.

    Las bajadas en horario vacío pasan a su ajuste de decaimiento puro (ACH,
    curva y medidas de calidad). Con `case`, además, los tramos con gente pasan
    a su ajuste con la generación de ese caso (`refit_occupied`). Los que no
    tienen reajuste válido dejan de reportarse y quedan marcados en
    `sin_reajuste` (conservan el ACH de asíntota libre); el resto no cambia.
    `en_vacio` marca las bajadas en horario vacío.
    """
    out = ev.copy()

    def swap(mask, suffix):
        for name in ("ach", "ach_se", "c0", "c_inf", "rmse", "r2"):
            out.loc[mask, name] = out.loc[mask, f"{name}_{suffix}"]
        out.loc[mask, "excursion_ppm"] = out.loc[mask, f"excursion_{suffix}"]

    inside = ((out["fase"] == "bajada") & (out["vacio"] >= min_inside)).to_numpy()
    done = inside & out["ach_puro"].notna().to_numpy()
    swap(done, "puro")
    missing = inside & ~done
    if case is not None:
        busy = out["con_gente"].to_numpy(dtype=bool)
        ok = busy & out[f"ach_{case}"].notna().to_numpy()
        swap(ok, case)
        missing |= busy & ~ok
    out["se_rel"] = out["ach_se"] / out["ach"]
    out["rmse_rel"] = out["rmse"] / out["excursion_ppm"]
    out["lam_t"] = out["ach"] * out["dur_min"] / 60.0
    out["reportado"] = (out["rmse_rel"] <= sel.max_rmse_rel) & (out["lam_t"] >= sel.min_lam_t) & ~missing
    out["en_vacio"] = inside
    out["sin_reajuste"] = missing
    return out


def _quartiles(prefix: str, values) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    q = np.quantile(v, [0.25, 0.5, 0.75]) if len(v) else [np.nan] * 3
    return {f"{prefix}_p25": q[0], f"{prefix}_mediana": q[1], f"{prefix}_p75": q[2]}


def generation_effect(ev: pd.DataFrame, known: pd.DataFrame, pairs: pd.DataFrame, sources: dict, mid: str = "medio", exclude=("M1",)) -> pd.DataFrame:
    """Qué cambia en los tramos con gente al conocer la generación, por fase y método y en total.

    `ev` son los eventos con el ajuste de cada método (generación libre),
    `known` los de `known_occupancy(..., case=mid)` y `pairs` los
    emparejamientos de `matched_pairs` (los tres con el mismo índice).
    `sources` va del caso de menor al de mayor generación. Por fila:

    - `sin_solucion_<caso>_pct`: tramos en que esa generación no explica la curva (λ en una cota);
    - cuartiles del ACH libre y del de cada caso, sobre los tramos con los tres casos;
    - `rango`: ACH del caso más alto sobre el del más bajo, tramo a tramo; `amplificacion` lo compara, en
      logaritmos, con el cociente de las generaciones (1: el ACH es proporcional a la generación);
    - `libre_dentro_del_rango_pct`: tramos cuyo ACH libre cae entre los de los casos extremos;
    - `fuente_libre`: generación [ppm/h] que implica el ajuste libre, λ (C∞ − C_ext), y cuántas caen entre las de los casos extremos;
    - calidad del ajuste, cuántos se reportan y diferencia entre métodos, con la generación libre y con la del caso `mid`.
    """
    cases = list(sources)
    lo, hi = cases[0], cases[-1]
    busy = ev["con_gente"].to_numpy(dtype=bool)
    all_ok = busy & np.all([ev[f"ach_{c}"].notna().to_numpy() for c in cases], axis=0)
    mid_ok = busy & ev[f"ach_{mid}"].notna().to_numpy()
    free = ev["ach"].to_numpy(dtype=float)
    implied = free * (ev["c_inf"].to_numpy(dtype=float) - ev["c_ext"].to_numpy(dtype=float))
    ach_mid = known["ach"].to_numpy(dtype=float)
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    method, phase_of = ev["metodo"].to_numpy(), ev["fase"].to_numpy()
    rows = []
    for phase in ("subida", "bajada"):
        keys = [k for k in METHODS if np.any(busy & (phase_of == phase) & (method == k))]
        for key in [*keys, "todos"]:
            who = ~np.isin(method, exclude) if key == "todos" else method == key
            m = busy & (phase_of == phase) & who
            ok, okm = m & all_ok, m & mid_ok
            both = okm[i] & okm[j]
            row = {"fase": phase, "metodo": key, "eventos": int(m.sum()), "con_los_tres_casos": int(ok.sum())}
            row.update({f"sin_solucion_{c}_pct": 100.0 * float(np.mean(ev.loc[m, f"cota_{c}"] != "")) for c in cases})
            row.update(_quartiles("ach_libre", free[ok]))
            for c in cases:
                row.update(_quartiles(f"ach_{c}", ev.loc[ok, f"ach_{c}"]))
            a_lo, a_hi = ev.loc[ok, f"ach_{lo}"].to_numpy(), ev.loc[ok, f"ach_{hi}"].to_numpy()
            row.update(_quartiles("rango", a_hi / a_lo))
            row["amplificacion_mediana"] = float(np.median(np.log(a_hi / a_lo))) / np.log(sources[hi] / sources[lo]) if ok.any() else np.nan
            a_mid = ev.loc[ok, f"ach_{mid}"].to_numpy()
            row["menor_sobre_medio_mediana"] = float(np.median(a_lo / a_mid)) if ok.any() else np.nan
            row["mayor_sobre_medio_mediana"] = float(np.median(a_hi / a_mid)) if ok.any() else np.nan
            row["libre_dentro_del_rango_pct"] = 100.0 * float(np.mean((free[ok] >= np.minimum(a_lo, a_hi)) & (free[ok] <= np.maximum(a_lo, a_hi)))) if ok.any() else np.nan
            row.update(_quartiles("libre_sobre_medio", free[okm] / ach_mid[okm]))
            row.update(_quartiles("fuente_libre", implied[m]))
            row["fuente_libre_dentro_del_rango_pct"] = 100.0 * float(np.mean((implied[m] >= sources[lo]) & (implied[m] <= sources[hi]))) if m.any() else np.nan
            row["rmse_rel_libre_mediano"] = ev.loc[okm, "rmse_rel"].median()
            row["rmse_rel_medio_mediano"] = known.loc[okm, "rmse_rel"].median()
            row["reportados_libre_pct"] = 100.0 * ev.loc[m, "reportado"].mean()
            row["reportados_medio_pct"] = 100.0 * known.loc[m, "reportado"].mean()
            row["emparejamientos"] = int(both.sum()) if key == "todos" else 0
            row["diferencia_entre_metodos_libre_pct"] = _pct(np.abs(np.log(free[i[both]] / free[j[both]]))) if key == "todos" else np.nan
            row["diferencia_entre_metodos_medio_pct"] = _pct(np.abs(np.log(ach_mid[i[both]] / ach_mid[j[both]]))) if key == "todos" else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def range_by_duration(ev: pd.DataFrame, known: pd.DataFrame, sources: dict, edges=(0.0, 0.5, 1.0, 2.0, np.inf), exclude=("M1",)) -> pd.DataFrame:
    """Cuánto se abre el ACH entre el caso de menor y el de mayor generación según lo que dura el tramo.

    La duración va en constantes de tiempo, ACH × duración, con el ACH de
    `known` (el del caso medio). En el equilibrio el ACH es proporcional a la
    generación y el cociente es el de las fuentes; lejos del equilibrio, en una
    subida es mayor y en una bajada, menor.
    """
    cases = list(sources)
    lo, hi = cases[0], cases[-1]
    ok = ev["con_gente"].to_numpy(dtype=bool) & np.all([ev[f"ach_{c}"].notna().to_numpy() for c in cases], axis=0) & ~ev["metodo"].isin(exclude).to_numpy()
    ratio = (ev[f"ach_{hi}"] / ev[f"ach_{lo}"]).to_numpy(dtype=float)
    lam_t = known["lam_t"].to_numpy(dtype=float)
    rows = []
    for phase in ("subida", "bajada"):
        for a, b in zip(edges[:-1], edges[1:]):
            m = ok & (ev["fase"] == phase).to_numpy() & (lam_t >= a) & (lam_t < b)
            rows.append({"fase": phase, "desde": a, "hasta": b, "eventos": int(m.sum()), **_quartiles("rango", ratio[m])})
    return pd.DataFrame(rows)


def with_ach(pairs: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Los mismos emparejamientos, con la diferencia recalculada con los ACH de `ev`."""
    ach = ev["ach"].to_numpy(dtype=float)
    return pairs.assign(dlog=np.abs(np.log(ach[pairs["i"].to_numpy()] / ach[pairs["j"].to_numpy()])))


def occupancy_effect(ev: pd.DataFrame, known: pd.DataFrame, pairs: pd.DataFrame, by: str = "ventana", exclude=("M1",)) -> pd.DataFrame:
    """Qué cambia en las bajadas en horario vacío al fijar la asíntota en el exterior.

    `ev` son los eventos con el ajuste de cada método (asíntota libre), `known`
    los de `known_occupancy` y `pairs` los emparejamientos de `matched_pairs`
    (los tres con el mismo índice). Una fila por valor de `by` ("ventana" o
    "metodo") y una con el total. El cociente es ACH libre / ACH con el salón
    vacío; el exceso, cuánto queda la asíntota libre por encima del exterior; la
    diferencia entre métodos, la mediana sobre los emparejamientos entre bajadas
    en horario vacío.
    """
    use = known["en_vacio"].to_numpy() & known["ach_puro"].notna().to_numpy() & ~ev["metodo"].isin(exclude).to_numpy()
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    free, pure = ev["ach"].to_numpy(dtype=float), known["ach"].to_numpy(dtype=float)
    group = ev[by].to_numpy()
    rows = []
    for name in [*sorted(pd.unique(group[use])), "todas"]:
        m = use if name == "todas" else use & (group == name)
        a, b = ev[m], known[m]
        ratio = a["ach"].to_numpy() / b["ach"].to_numpy()
        excess = a["c_inf"].to_numpy() - b["c_inf"].to_numpy()
        both = m[i] & m[j]
        rows.append(
            {
                by: name,
                "bajadas": int(m.sum()),
                "exceso_asintota_mediano_ppm": float(np.median(excess)),
                "exceso_asintota_p75_ppm": float(np.quantile(excess, 0.75)),
                "asintota_100_ppm_arriba_pct": 100.0 * float(np.mean(excess > 100.0)),
                "ach_libre_p25": a["ach"].quantile(0.25),
                "ach_libre_mediana": a["ach"].median(),
                "ach_libre_p75": a["ach"].quantile(0.75),
                "ach_vacio_p25": b["ach"].quantile(0.25),
                "ach_vacio_mediana": b["ach"].median(),
                "ach_vacio_p75": b["ach"].quantile(0.75),
                "cociente_p25": float(np.quantile(ratio, 0.25)),
                "cociente_mediana": float(np.median(ratio)),
                "cociente_p75": float(np.quantile(ratio, 0.75)),
                "cambia_mas_de_20_pct": 100.0 * float(np.mean(np.abs(np.log(ratio)) > np.log(1.2))),
                "rmse_rel_libre_mediano": a["rmse_rel"].median(),
                "rmse_rel_vacio_mediano": b["rmse_rel"].median(),
                "reportados_libre_pct": 100.0 * a["reportado"].mean(),
                "reportados_vacio_pct": 100.0 * b["reportado"].mean(),
                "emparejamientos": int(both.sum()),
                "diferencia_entre_metodos_libre_pct": _pct(np.abs(np.log(free[i[both]] / free[j[both]]))),
                "diferencia_entre_metodos_vacio_pct": _pct(np.abs(np.log(pure[i[both]] / pure[j[both]]))),
            }
        )
    return pd.DataFrame(rows)


def hourly_profile(df: pd.DataFrame, days) -> pd.DataFrame:
    """Mediana y cuartiles del CO2 por hora del día, sobre los días dados."""
    sel = df[df["tiempo"].dt.normalize().isin(pd.DatetimeIndex(days))]
    g = sel.groupby(sel["tiempo"].dt.hour + (sel["tiempo"].dt.minute // 15) / 4.0)["co2"]
    return pd.DataFrame({"q25": g.quantile(0.25), "mediana": g.median(), "q75": g.quantile(0.75)})


def smooth(c_obs, window: int = 25) -> np.ndarray:
    return median_filter(c_obs, window)
