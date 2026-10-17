"""Eventos marcados a mano sobre las mediciones: etiquetas, ajuste de referencia y comparación con los métodos.

Un evento manual es un intervalo [t0, t1] de un día que una persona reconoce,
mirando la serie, como una subida o una bajada. La fase sale de comparar la
concentración en los dos extremos. Sobre cada evento se ajusta la misma
exponencial general que usan los métodos (C0, C∞ ≥ C_ext y λ libres, sobre la
serie sin outliers): ese ACH es la referencia contra la que se comparan.

Como los métodos, la referencia ajusta por defecto el régimen completo: desde
el inicio marcado hasta el inicio del evento marcado siguiente de ese día (o
hasta el final del día), es decir, el transitorio con la meseta que lo sigue
(`regime_ends`). Las marcas no cambian; sólo cambia hasta dónde se ajusta.

La comparación mira dos cosas por separado. La segmentación: qué parte de cada
evento manual cubren los tramos del método, en cuántos fragmentos y con qué
desfase en el inicio. Y el ACH: para cada evento manual, el ajuste aceptado del
método, de la misma fase, que más se solapa con él.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .fitting import fit_exponential
from .mediciones import DT_S, _day_data, _pct, smooth
from .segmenters.base import DECAY, RISE
from .signal_utils import median_filter

KEY = "manual"
LABEL_COLUMNS = ["fecha", "t0_h", "t1_h", "fase"]
MIN_SAMPLES = 6  # un evento más corto que esto (3 min cada 30 s) no se puede ajustar


def labels_path(folder) -> Path:
    return Path(folder) / "eventos_manuales.csv"


def reviewed_path(folder) -> Path:
    return Path(folder) / "dias_revisados.csv"


def load_labels(folder):
    """Devuelve (eventos, días revisados). Si no hay archivos, una tabla vacía y un conjunto vacío."""
    path, seen = labels_path(folder), reviewed_path(folder)
    labels = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=LABEL_COLUMNS)
    reviewed = set(pd.read_csv(seen)["fecha"]) if seen.exists() else set()
    return labels[LABEL_COLUMNS], reviewed


def save_labels(folder, labels: pd.DataFrame, reviewed) -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)
    labels.sort_values(["fecha", "t0_h"])[LABEL_COLUMNS].to_csv(labels_path(folder), index=False, float_format="%.5f")
    pd.DataFrame({"fecha": sorted(reviewed)}).to_csv(reviewed_path(folder), index=False)


def sample_range(t0_h: float, t1_h: float, n: int, dt_s: float = DT_S):
    """Índices [i0, i1) de la grilla del día que cubren el intervalo."""
    per_hour = 3600.0 / dt_s
    i0 = int(np.clip(round(min(t0_h, t1_h) * per_hour), 0, n - 1))
    i1 = int(np.clip(round(max(t0_h, t1_h) * per_hour) + 1, 1, n))
    return i0, i1


def regime_ends(spans, n: int) -> list:
    """Fin de cada régimen: el inicio del evento siguiente, o `n` para el último.

    `spans` son los (i0, i1) marcados de un día, ordenados por inicio. Si dos
    marcas se pisan, el régimen no termina antes que su propia marca.
    """
    starts = [a for a, _ in spans]
    return [max(b, starts[k + 1]) if k + 1 < len(spans) else n for k, (_, b) in enumerate(spans)]


def _day_spans(g: pd.DataFrame, n: int, regimen: bool) -> list:
    """(i0, i1 marcado, i1 que se ajusta) de los eventos de un día con al menos `MIN_SAMPLES` muestras."""
    spans = [sample_range(ev.t0_h, ev.t1_h, n) for ev in g.sort_values("t0_h").itertuples()]
    spans = [(a, b) for a, b in spans if b - a >= MIN_SAMPLES]
    ends = regime_ends(spans, n) if regimen else [b for _, b in spans]
    return [(a, b, e) for (a, b), e in zip(spans, ends)]


def phase_of(filtered, i0: int, i1: int) -> str:
    """Subida o bajada, según la serie suavizada en los extremos del intervalo."""
    return "subida" if filtered[i1 - 1] > filtered[i0] else "bajada"


def fit_event(series, clean, i0: int, i1: int, phase: str):
    """Exponencial general sobre el intervalo, como el ajuste de los métodos de ajuste general."""
    sl = slice(i0, i1)
    return fit_exponential(series.t_h[sl], clean[sl], c_ext=series.c_ext_nominal, css_min=series.c_ext_nominal, i0=i0, model="general", phase=RISE if phase == "subida" else DECAY)


def fit_row(series, fit, phase: str) -> dict:
    """Una fila con las columnas de `mediciones.fit_rows` para el ajuste de un evento manual."""
    t = series.t_h
    good = bool(fit.ok and np.isfinite(fit.lam))
    ends = fit.predict(t[[fit.i0, fit.i1 - 1]]) if good else np.array([np.nan, np.nan])
    accepted = good and not fit.at_bound
    return {
        "fecha": series.name,
        "metodo": KEY,
        "t0_h": t[fit.i0],
        "t1_h": t[fit.i1 - 1],
        "dur_min": 60.0 * (t[fit.i1 - 1] - t[fit.i0]),
        "fase": phase,
        "aceptado": accepted,
        "motivo": "" if accepted else (fit.reason or ("λ en la cota" if good else "el ajuste falló")),
        "ach": fit.lam if good else np.nan,
        "ach_se": fit.lam_se if good else np.nan,
        "c0": fit.c0 if good else np.nan,
        "c_inf": fit.css if good else np.nan,
        "excursion_ppm": abs(float(ends[1] - ends[0])),
        "asintota_en_c_ext": bool(good and fit.css <= series.c_ext_nominal + 1.0),
        "r2": fit.r2,
        "rmse": fit.rmse,
        "i0": fit.i0,
        "i1": fit.i1,
        "config": KEY,
    }


def manual_fits(df: pd.DataFrame, days: pd.DataFrame, labels: pd.DataFrame, noise, regimen: bool = True) -> pd.DataFrame:
    """El ajuste de referencia de cada evento manual, con las mismas columnas que los ajustes de los métodos.

    `noise` es el piso de ruido de cada día (el ruido efectivo), que fija qué
    cuenta como outlier al limpiar la serie, igual que en la corrida de los
    métodos. Los eventos de menos de `MIN_SAMPLES` muestras se saltean. La fase
    sale del intervalo marcado; con `regimen` el ajuste va hasta el inicio del
    evento siguiente (`regime_ends`), y si no, hasta el fin marcado. `t1_marca_h`
    guarda el fin marcado.
    """
    rows = []
    for date, g in labels.groupby("fecha", sort=True):
        series, clean, filtered = _day_data(df, days, date, noise)
        for i0, i1, end in _day_spans(g, len(series.t_h), regimen):
            phase = phase_of(filtered, i0, i1)
            rows.append(dict(fit_row(series, fit_event(series, clean, i0, end, phase), phase), t1_marca_h=series.t_h[i1 - 1]))
    return pd.DataFrame(rows)


def trim_event(clean, i0: int, i1: int, phase: str, dt_s: float = DT_S, max_edge_min: float = 3.0, max_fraction: float = 0.2, tol_ppm: float = 2.0):
    """Corrige la puntería de los dos clics de un evento marcado a mano, mirando sólo la serie. Devuelve los nuevos (i0, i1).

    Un clic sobre la gráfica de un día entero tiene una precisión de minutos.
    Se corrigen los dos errores de puntería que la serie permite reconocer sin
    ambigüedad, y sólo dentro de `max_edge_min` minutos y de `max_fraction` de
    la duración del evento:

    - un inicio marcado antes de que la serie arranque: se corre al último
      punto en que todavía está, a menos de `tol_ppm`, en su extremo (el mínimo
      antes de una subida, el máximo antes de una bajada);
    - un fin marcado después del punto de giro: si la serie ya cambió de
      sentido en más de `tol_ppm`, se corre al extremo (máximo de una subida,
      mínimo de una bajada).

    Sólo recorta, nunca extiende, y no mira qué detecta ni qué ACH da ningún
    método. No toca la transición de entrada (la serie que arranca despacio):
    dónde empieza a valer la exponencial es una decisión de modelado, no un
    error de puntería, y cambia mucho el ACH (`start_sensitivity`).
    """
    sign = 1.0 if phase == "subida" else -1.0
    c = median_filter(sign * np.asarray(clean, dtype=float), 5)
    n = i1 - i0
    reach = min(int(round(max_edge_min * 60.0 / dt_s)), int(max_fraction * n))
    start, end = i0, i1
    if reach > 0:
        head = c[i0 : i0 + reach + 1]
        still = np.flatnonzero(head <= head.min() + tol_ppm)  # todavía no arrancó
        start = i0 + int(still[-1])
        tail = c[i1 - 1 - reach : i1]
        j = int(np.argmax(tail))
        if tail[j] - tail[-1] > tol_ppm:
            end = i1 - 1 - reach + j + 1
    if end - start < MIN_SAMPLES:
        return i0, i1
    return start, end


def start_sensitivity(df: pd.DataFrame, days: pd.DataFrame, labels: pd.DataFrame, noise, shifts_min=(-3, -2, -1, 0, 1, 2, 3, 4, 5), regimen: bool = True) -> pd.DataFrame:
    """Cuánto cambia el ACH de referencia de cada evento si su inicio se corre unos minutos, con el fin en su lugar.

    Una fila por evento y corrimiento (positivo: empieza más tarde). `cociente`
    es el ACH con el inicio corrido sobre el ACH con el inicio marcado. No se
    corre más de un tercio de la duración del evento marcado. Con `regimen`, el
    fin es el del régimen (`regime_ends`), como en `manual_fits`.
    """
    rows = []
    for date, g in labels.groupby("fecha", sort=True):
        series, clean, filtered = _day_data(df, days, date, noise)
        per_min = 60.0 / DT_S
        for i0, mark_end, i1 in _day_spans(g, len(series.t_h), regimen):
            phase = phase_of(filtered, i0, mark_end)
            base = fit_event(series, clean, i0, i1, phase)
            if not (base.ok and np.isfinite(base.lam) and base.lam > 0):
                continue
            for shift in shifts_min:
                j0 = i0 + int(round(shift * per_min))
                if j0 < 0 or abs(j0 - i0) > (mark_end - i0) / 3.0:
                    continue
                f = base if shift == 0 else fit_event(series, clean, j0, i1, phase)
                good = bool(f.ok and np.isfinite(f.lam) and f.lam > 0)
                rows.append({"fecha": date, "t0_h": series.t_h[i0], "t1_h": series.t_h[i1 - 1], "fase": phase, "corrimiento_min": shift, "ach": f.lam if good else np.nan, "cociente": f.lam / base.lam if good else np.nan, "lam_t_marcado": base.lam * (series.t_h[i1 - 1] - series.t_h[i0])})
    return pd.DataFrame(rows)


def sensitivity_summary(table: pd.DataFrame, limit: float = 0.2) -> pd.DataFrame:
    """Por fase y corrimiento: cociente mediano y cuartiles, cambio absoluto mediano y porcentaje de eventos que cambian más de `limit`."""
    rows = []
    for (phase, shift), g in table.groupby(["fase", "corrimiento_min"], sort=True):
        d = np.log(g["cociente"].dropna())
        if not len(d):
            continue
        q = np.exp(d.quantile([0.25, 0.5, 0.75]))
        rows.append({"fase": phase, "corrimiento_min": shift, "eventos": len(d), "cociente_p25": q.iloc[0], "cociente_mediano": q.iloc[1], "cociente_p75": q.iloc[2], "cambio_mediano_pct": _pct(d.abs()), f"cambian_mas_de_{round(100 * limit)}_pct": 100.0 * float((d.abs() > np.log1p(limit)).mean())})
    return pd.DataFrame(rows)


def refine_labels(df: pd.DataFrame, days: pd.DataFrame, labels: pd.DataFrame, noise, regimen: bool = True, **kwargs):
    """Las marcas con la puntería corregida por `trim_event`, y una tabla con lo que cambió en cada una.

    La tabla tiene, por evento, los minutos recortados en cada borde y el
    ajuste de referencia antes y después (ACH, constantes de tiempo que abarca
    y RMSE sobre el recorrido), más cuatro avisos que no se corrigen solos,
    para volver a mirar la marca: `va_en_contra` (más de un cuarto del tramo
    va en sentido contrario a su fase), `mal_ajuste` (RMSE de más de 10 % del
    recorrido aun corregido), `se_pisa` (se superpone más de un minuto con el
    evento anterior) y `poco_determinado` (abarca menos de una constante de
    tiempo: su ACH de referencia no es firme, se marque como se marque).
    Con `regimen`, el ajuste de antes y de después va hasta el inicio del evento
    siguiente, con las marcas originales y con las corregidas respectivamente.
    """
    kept, rows = [], []
    for date, g in labels.groupby("fecha", sort=True):
        series, clean, filtered = _day_data(df, days, date, noise)
        t, last_end = series.t_h, None
        spans = [(i0, i1) for i0, i1, _ in _day_spans(g, len(t), False)]
        phases = [phase_of(filtered, i0, i1) for i0, i1 in spans]
        trimmed = [trim_event(clean, i0, i1, ph, **kwargs) for (i0, i1), ph in zip(spans, phases)]
        ends_before = regime_ends(spans, len(t)) if regimen else [b for _, b in spans]
        ends_after = regime_ends(trimmed, len(t)) if regimen else [b for _, b in trimmed]
        for (i0, i1), phase, (j0, j1), e0, e1 in zip(spans, phases, trimmed, ends_before, ends_after):
            row = {"fecha": date, "fase": phase, "t0_h": t[i0], "t1_h": t[i1 - 1], "t0_nuevo_h": t[j0], "t1_nuevo_h": t[j1 - 1], "recorte_inicio_min": 60.0 * (t[j0] - t[i0]), "recorte_fin_min": 60.0 * (t[i1 - 1] - t[j1 - 1])}
            for tag, (a, b) in (("antes", (i0, e0)), ("despues", (j0, e1))):
                f = fit_row(series, fit_event(series, clean, a, b, phase), phase)
                row[f"ach_{tag}"] = f["ach"]
                row[f"lam_t_{tag}"] = f["ach"] * f["dur_min"] / 60.0
                row[f"rmse_rel_{tag}"] = f["rmse"] / f["excursion_ppm"] if f["excursion_ppm"] > 0 else np.nan
            step = np.diff(median_filter(clean, 5)[j0:j1])
            row["va_en_contra"] = bool(np.mean(step < 0 if phase == "subida" else step > 0) > 0.25)
            row["mal_ajuste"] = bool(row["rmse_rel_despues"] > 0.10)
            row["se_pisa"] = bool(last_end is not None and t[i0] < last_end - 1.0 / 60.0)
            row["poco_determinado"] = bool(row["lam_t_despues"] < 1.0)
            last_end = t[i1 - 1]
            rows.append(row)
            kept.append({"fecha": date, "t0_h": t[j0], "t1_h": t[j1 - 1], "fase": phase})
    return pd.DataFrame(kept, columns=LABEL_COLUMNS), pd.DataFrame(rows)


def manual_segments(fits: pd.DataFrame) -> pd.DataFrame:
    """Los eventos manuales como tramos, con las columnas de la tabla de tramos de los métodos."""
    return fits[["fecha", "metodo", "t0_h", "t1_h", "fase", "config"]].copy()


# ------------------------------------------------------------------ comparación


def _overlap(a0, a1, b0, b1):
    return np.maximum(0.0, np.minimum(a1, b1) - np.maximum(a0, b0))


def compare(manual: pd.DataFrame, fits: pd.DataFrame, segs: pd.DataFrame, methods, value: str = "ach", min_cover: float = 0.5) -> pd.DataFrame:
    """Una fila por evento manual y método.

    `manual` son los ajustes de referencia; `fits`, los ajustes de los métodos
    y `segs`, sus tramos (todos los que detectan, se ajusten o no). `value` es
    la columna de ACH que se compara. Por fila:

    - `cobertura`: fracción del evento manual cubierta por tramos del método de la misma fase;
    - `fragmentos`: cuántos tramos de esa fase lo tocan;
    - `detectado`: la cobertura llega a `min_cover`;
    - `desfase_inicio_min`: inicio del primer tramo que lo toca menos el inicio manual;
    - `ach_metodo`: ACH del ajuste aceptado de la misma fase que más se solapa con el evento
      (tiene que solaparse al menos `min_cover` del más corto de los dos), y `dlog` = ln(ACH método / ACH manual);
    - `lam_t_manual` y `rmse_rel_manual`: cuántas constantes de tiempo abarca el evento manual
      según su propio ajuste y su RMSE sobre el recorrido, que dicen qué tan firme es la referencia.
    """
    rows = []
    for ev in manual.itertuples():
        dur = max(ev.t1_h - ev.t0_h, 1e-9)
        ref = getattr(ev, value)
        lam_t = ref * dur if np.isfinite(ref) else np.nan
        rmse_rel = ev.rmse / ev.excursion_ppm if np.isfinite(ref) and ev.excursion_ppm > 0 else np.nan
        for key in methods:
            s = segs[(segs["fecha"] == ev.fecha) & (segs["metodo"] == key) & (segs["fase"] == ev.fase)]
            ov = _overlap(ev.t0_h, ev.t1_h, s["t0_h"].to_numpy(), s["t1_h"].to_numpy())
            touching = s[ov > 0]
            f = fits[(fits["fecha"] == ev.fecha) & (fits["metodo"] == key) & (fits["fase"] == ev.fase) & fits["aceptado"] & fits[value].notna()]
            ach, share = np.nan, np.nan
            if len(f):
                fo = _overlap(ev.t0_h, ev.t1_h, f["t0_h"].to_numpy(), f["t1_h"].to_numpy())
                shorter = np.minimum(dur, (f["t1_h"] - f["t0_h"]).to_numpy())
                k = int(np.argmax(fo))
                if fo[k] >= min_cover * shorter[k] and fo[k] > 0:
                    ach, share = float(f[value].iloc[k]), float(fo[k] / dur)
            cover = float(min(ov.sum() / dur, 1.0))
            rows.append(
                {
                    "fecha": ev.fecha,
                    "t0_h": ev.t0_h,
                    "t1_h": ev.t1_h,
                    "fase": ev.fase,
                    "dur_min": 60.0 * dur,
                    "ach_manual": ref,
                    "metodo": key,
                    "cobertura": cover,
                    "fragmentos": int(len(touching)),
                    "detectado": cover >= min_cover,
                    "desfase_inicio_min": 60.0 * float(touching["t0_h"].min() - ev.t0_h) if len(touching) else np.nan,
                    "ach_metodo": ach,
                    "solape_del_ajuste": share,
                    "dlog": float(np.log(ach / ref)) if np.isfinite(ach) and np.isfinite(ref) and ref > 0 else np.nan,
                    "lam_t_manual": lam_t,
                    "rmse_rel_manual": rmse_rel,
                }
            )
    return pd.DataFrame(rows)


def extra_events(manual: pd.DataFrame, events: pd.DataFrame, methods, min_cover: float = 0.5) -> pd.DataFrame:
    """Eventos de cada método que no corresponden a ningún evento manual de su misma fase (menos de `min_cover` adentro)."""
    rows = []
    for key in methods:
        e = events[events["metodo"] == key]
        for ev in e.itertuples():
            m = manual[(manual["fecha"] == ev.fecha) & (manual["fase"] == ev.fase)]
            inside = _overlap(ev.t0_h, ev.t1_h, m["t0_h"].to_numpy(), m["t1_h"].to_numpy()).sum() / max(ev.t1_h - ev.t0_h, 1e-9)
            rows.append({"fecha": ev.fecha, "metodo": key, "fase": ev.fase, "t0_h": ev.t0_h, "t1_h": ev.t1_h, "dentro_de_manual": float(min(inside, 1.0)), "de_mas": inside < min_cover})
    return pd.DataFrame(rows)


def summary(comp: pd.DataFrame, extras: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Por método y fase: cuánto de lo manual detecta, cómo lo fragmenta y cuánto se aparta su ACH.

    `dentro_de_20_pct` es sobre los eventos en que el método da un ACH;
    `aciertos_20_pct`, sobre todos los eventos manuales con referencia, de modo
    que no estimar cuenta como fallo.
    """
    rows = []
    for (key, phase), g in comp.groupby(["metodo", "fase"], sort=False):
        d = g["dlog"].dropna()
        with_ref = int(g["ach_manual"].notna().sum())
        x = extras[(extras["metodo"] == key) & (extras["fase"] == phase)] if len(extras) else extras
        rows.append(
            {
                "metodo": key,
                "fase": phase,
                "eventos_manuales": len(g),
                "detectados_pct": 100.0 * g["detectado"].mean(),
                "cobertura_media_pct": 100.0 * g["cobertura"].mean(),
                "fragmentos_medios": g.loc[g["fragmentos"] > 0, "fragmentos"].mean(),
                "desfase_inicio_mediano_min": g["desfase_inicio_min"].median(),
                "desfase_inicio_abs_mediano_min": g["desfase_inicio_min"].abs().median(),
                "con_ach": len(d),
                "con_ach_pct": 100.0 * len(d) / max(len(g), 1),
                "diferencia_mediana_pct": _pct(d.abs()),
                "sesgo_mediano_pct": 100.0 * (np.exp(d.median()) - 1.0) if len(d) else np.nan,
                "dentro_de_20_pct": 100.0 * float((d.abs() <= np.log(1.2)).mean()) if len(d) else np.nan,
                "aciertos_20_pct": 100.0 * float((d.abs() <= np.log(1.2)).sum()) / with_ref if with_ref else np.nan,
                "eventos_del_metodo": len(x),
                "eventos_de_mas_por_dia": float(x["de_mas"].sum()) / max(n_days, 1) if len(x) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def hit_table(comp: pd.DataFrame, limits=(0.1, 0.2, 0.5, 1.0)) -> pd.DataFrame:
    """Por método y fase: porcentaje de los eventos manuales con referencia en que el ACH del método queda a menos de cada límite.

    Un límite de 0,2 es una diferencia de hasta 20 % en cualquiera de los dos
    sentidos (cociente entre 1/1,2 y 1,2). Los eventos en que el método no da
    un ACH cuentan como fallo.
    """
    rows = []
    for (key, phase), g in comp[comp["ach_manual"].notna()].groupby(["metodo", "fase"], sort=False):
        d = g["dlog"].abs()
        row = {"metodo": key, "fase": phase, "eventos_manuales": len(g), "con_ach_pct": 100.0 * float(d.notna().mean())}
        row.update({f"a_menos_de_{round(100 * x)}_pct": 100.0 * float((d <= np.log1p(x)).mean()) for x in limits})
        rows.append(row)
    return pd.DataFrame(rows)


def by_reference_duration(comp: pd.DataFrame, edges=(0.0, 0.5, 1.0, 2.0, np.inf)) -> pd.DataFrame:
    """Diferencia con la referencia según cuántas constantes de tiempo abarca el evento manual (`lam_t_manual`).

    Un evento que abarca menos de una constante de tiempo es casi una recta, y
    su ACH de referencia está tan mal determinado como el de cualquier método:
    ahí la diferencia mide la indeterminación de los dos, no el error del
    método. Una fila por fase, tramo de duración y método.
    """
    rows = []
    c = comp[comp["lam_t_manual"].notna()]
    for phase in ("subida", "bajada"):
        for lo, hi in zip(edges[:-1], edges[1:]):
            sub = c[(c["fase"] == phase) & (c["lam_t_manual"] >= lo) & (c["lam_t_manual"] < hi)]
            n_events = len(sub.drop_duplicates(["fecha", "t0_h", "t1_h"]))
            for key, g in sub.groupby("metodo", sort=False):
                d = g["dlog"].dropna()
                rows.append(
                    {
                        "fase": phase,
                        "desde": lo,
                        "hasta": hi,
                        "eventos_manuales": n_events,
                        "metodo": key,
                        "con_ach": len(d),
                        "diferencia_mediana_pct": _pct(d.abs()),
                        "sesgo_mediano_pct": 100.0 * (np.exp(d.median()) - 1.0) if len(d) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def filtered_series(series):
    """La serie suavizada con la que se decide la fase de un evento."""
    return smooth(series.c_obs)
