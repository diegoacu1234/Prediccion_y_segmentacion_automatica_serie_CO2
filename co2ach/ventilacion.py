"""Indicadores de ventilación del aula, día por día (sección 6.7).

Con las lecturas y los eventos ya reajustados (`mediciones.refit_empty` y
`refit_occupied`), para cada día hábil con clase:

- exposición: CO2 en horario de clase y tiempo por encima de 1000 y 1500 ppm;
- ACH por bloque del horario: con la clase adentro, de las subidas con la
  generación conocida; con el aula vacía, de las bajadas con la asíntota en C_ext;
- aire exterior por persona.

La ACH de un bloque es el promedio de los ajustes de todos los métodos
ponderado por su duración; el mínimo y el máximo entre métodos dan la
dispersión. Sólo entran ajustes con RMSE ≤ 5 % del recorrido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .mediciones import Occupancy, clock

THRESHOLDS = (1000.0, 1500.0)  # ppm
GROUPS = ("clase", "pausa", "salida", "noche")
GROUP_NAMES = {"clase": "con la clase adentro", "pausa": "en recreos y mediodía", "salida": "al irse la clase", "noche": "de noche, cerrado"}


def day_blocks(occ: Occupancy, night_from: float = 18.5) -> pd.DataFrame:
    """Los bloques del día según el horario.

    tipo: "clase" o "vacio". grupo: "pausa" (entre clases), "salida" (hasta
    `night_from`) o "noche".
    """
    empty = sorted(occ.empty)
    rows = []
    for k, (a, b) in enumerate(empty):
        if k == 0 and a <= 0.0:
            rows.append((a, b, "vacio", "noche"))
        elif k == len(empty) - 1 and b >= 24.0:
            cut = min(max(night_from, a), b)
            rows.append((a, cut, "vacio", "salida"))
            if cut < b:
                rows.append((cut, b, "vacio", "noche"))
        else:
            rows.append((a, b, "vacio", "pausa"))
        if k < len(empty) - 1 and empty[k + 1][0] > b:
            rows.append((b, empty[k + 1][0], "clase", "clase"))
    out = pd.DataFrame(rows, columns=["desde", "hasta", "tipo", "grupo"])
    out.insert(0, "bloque", [f"{clock(a)}–{clock(b)}" for a, b in zip(out["desde"], out["hasta"])])
    return out


def class_hours(blocks: pd.DataFrame) -> float:
    cls = blocks[blocks["tipo"] == "clase"]
    return float((cls["hasta"] - cls["desde"]).sum())


def exposure(df: pd.DataFrame, daily: pd.DataFrame, blocks: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    """CO2 en horario de clase de cada día hábil con clase.

    Media, p95 y máximo, exceso sobre el exterior, % y horas por encima de cada
    umbral, temperatura y humedad medias, y `cobertura` (qué parte del horario
    tiene lecturas).
    """
    days = daily.index[daily["ocupado"] & (daily["dia_semana"] < 5)]
    t = df["tiempo"]
    hour = (t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0).to_numpy()
    cls = blocks[blocks["tipo"] == "clase"]
    inside = np.any([(hour >= a) & (hour < b) for a, b in zip(cls["desde"], cls["hasta"])], axis=0)
    date = t.dt.normalize()
    sub = df[inside & date.isin(days).to_numpy()].assign(fecha=date)
    sub = sub.assign(exceso=sub["co2"] - sub["fecha"].map(daily["c_ext"]))
    g = sub.groupby("fecha")
    step_h = float(np.median(np.diff(t.to_numpy()[:20000]) / np.timedelta64(1, "h")))
    hours = class_hours(blocks)
    out = pd.DataFrame(
        {
            "cobertura": (g.size() * step_h / hours).clip(upper=1.0),
            "co2_medio_clase": g["co2"].mean(),
            "co2_p95_clase": g["co2"].quantile(0.95),
            "co2_max_clase": g["co2"].max(),
            "exceso_medio_clase": g["exceso"].mean(),
        }
    )
    for th in thresholds:
        share = g["co2"].apply(lambda x, th=th: float(np.mean(x > th)))
        out[f"pct_sobre_{th:g}"] = 100.0 * share
        out[f"horas_sobre_{th:g}"] = share * hours
    for col, name in (("temp", "temp_clase"), ("hum", "hum_clase")):
        if col in sub:
            out[name] = g[col].mean()
    out["c_ext"] = daily["c_ext"].reindex(out.index)
    return out.reindex(days)


def _weighted(values, weights) -> float:
    ok = np.isfinite(values)
    return float(np.sum(values[ok] * weights[ok]) / np.sum(weights[ok])) if ok.any() else np.nan


def ach_by_method(ev: pd.DataFrame, blocks: pd.DataFrame, cases, by: str = "bloque", mid: str | None = None, max_rmse_rel: float = 0.05, min_inside: float = 0.8, exclude=("M1",)) -> pd.DataFrame:
    """ACH de cada día por bloque (o grupo) y método, promedio ponderado por duración.

    Clase: subidas con gente con el caso `mid` (el del medio si no se da).
    Vacío: bajadas con al menos `min_inside` en horario vacío, con decaimiento
    puro. En los dos, RMSE ≤ `max_rmse_rel` del recorrido. Cada evento va al
    bloque de su punto medio. `menor` y `mayor` usan los casos extremos y
    `falta` son los minutos sin solución con el caso menor.
    """
    cases = list(cases)
    mid = mid or cases[len(cases) // 2]
    lo, hi = cases[0], cases[-1]
    e = ev[(pd.to_datetime(ev["fecha"]).dt.dayofweek < 5) & ~ev["metodo"].isin(exclude)].copy()
    center = 0.5 * (e["t0_h"] + e["t1_h"]).to_numpy()
    idx = np.clip(np.searchsorted(blocks["hasta"].to_numpy(), center, side="right"), 0, len(blocks) - 1)
    e["bloque"] = blocks["bloque"].to_numpy()[idx]
    e["grupo"] = blocks["grupo"].to_numpy()[idx]
    kind = blocks["tipo"].to_numpy()[idx]
    busy = e["con_gente"].to_numpy(dtype=bool) & (e["fase"] == "subida").to_numpy() & (kind == "clase") & (e[f"rmse_{mid}"] / e[f"excursion_{mid}"] <= max_rmse_rel).to_numpy()
    empty = (e["fase"] == "bajada").to_numpy() & (e["vacio"] >= min_inside).to_numpy() & (kind == "vacio") & (e["rmse_puro"] / e["excursion_puro"] <= max_rmse_rel).to_numpy()
    e["valor"] = np.where(busy, e[f"ach_{mid}"], np.where(empty, e["ach_puro"], np.nan))
    e["menor"] = np.where(busy, e[f"ach_{lo}"], np.nan)
    e["mayor"] = np.where(busy, e[f"ach_{hi}"], np.nan)
    e["falta"] = np.where(busy & e[f"ach_{lo}"].isna().to_numpy(), e["dur_min"], 0.0)
    e = e[np.isfinite(e["valor"])]
    rows = []
    for (date, key, method), g in e.groupby(["fecha", by, "metodo"], sort=False):
        w = g["dur_min"].to_numpy(dtype=float)
        rows.append((date, key, method, _weighted(g["valor"].to_numpy(dtype=float), w), _weighted(g["menor"].to_numpy(dtype=float), w), _weighted(g["mayor"].to_numpy(dtype=float), w), len(g), float(w.sum()), float(g["falta"].sum())))
    return pd.DataFrame(rows, columns=["fecha", by, "metodo", "ach", "menor", "mayor", "eventos", "minutos", "falta"])


def consensus_ach(ev: pd.DataFrame, blocks: pd.DataFrame, cases, by: str = "bloque", mid: str | None = None, max_rmse_rel: float = 0.05, min_inside: float = 0.8, exclude=("M1",)) -> pd.DataFrame:
    """ACH de cada día por bloque (o grupo) juntando todos los métodos (los mismos argumentos que `ach_by_method`).

    ach: promedio de los eventos de todos los métodos ponderado por duración;
    ach_min/ach_max: entre métodos; en clase, ach_<caso> de los casos extremos y
    sin_solucion_pct (tiempo sin solución con el caso menor).
    """
    cases = list(cases)
    lo, hi = cases[0], cases[-1]
    per = ach_by_method(ev, blocks, cases, by=by, mid=mid, max_rmse_rel=max_rmse_rel, min_inside=min_inside, exclude=exclude)
    if per.empty:
        return pd.DataFrame(columns=["fecha", by, "ach", "ach_min", "ach_max", "metodos", "eventos", "minutos", f"ach_{lo}", f"ach_{hi}", "sin_solucion_pct"])
    g = per.groupby(["fecha", by], sort=True)

    def pooled(col):
        return g.apply(lambda d: _weighted(d[col].to_numpy(dtype=float), d["minutos"].to_numpy(dtype=float)), include_groups=False)

    out = pd.DataFrame(
        {
            "ach": pooled("ach"),
            "ach_min": g["ach"].min(),
            "ach_max": g["ach"].max(),
            "metodos": g["ach"].size(),
            "eventos": g["eventos"].sum(),
            "minutos": g["minutos"].median(),
            f"ach_{lo}": pooled("menor"),
            f"ach_{hi}": pooled("mayor"),
            "sin_solucion_pct": 100.0 * g["falta"].sum() / g["minutos"].sum(),
        }
    ).reset_index()
    return out


def daily_report(df: pd.DataFrame, daily: pd.DataFrame, ev: pd.DataFrame, occ: Occupancy, cases_table: pd.DataFrame, max_rmse_rel: float = 0.05, night_from: float = 18.5):
    """Una fila por día hábil con clase. Devuelve (informe, ACH por día y bloque, bloques).

    `cases_table` es la de `mediciones.generation_table`, de menor a mayor.
    ach_clase usa el caso del medio y ach_clase_<caso> los extremos;
    caudal_<caso> = ACH·V/personas [L/s]; ach_necesario_<umbral> = S/(umbral − C_ext),
    la ACH de equilibrio para no pasar el umbral.
    """
    cases = list(cases_table["caso"])
    mid = cases[len(cases) // 2]
    blocks = day_blocks(occ, night_from)
    report = exposure(df, daily, blocks)
    report.index = report.index.strftime("%Y-%m-%d")
    report.index.name = "fecha"
    report.insert(0, "dia_semana", pd.to_datetime(report.index).dayofweek)
    report.insert(1, "mes", pd.to_datetime(report.index).month)
    report.insert(2, "analizado", report.index.isin(set(ev["fecha"])))
    by_group = consensus_ach(ev, blocks, cases, by="grupo", mid=mid, max_rmse_rel=max_rmse_rel)
    for group in GROUPS:
        sub = by_group[by_group["grupo"] == group].set_index("fecha")
        cols = {"ach": f"ach_{group}", "ach_min": f"ach_{group}_min", "ach_max": f"ach_{group}_max", "metodos": f"metodos_{group}", "minutos": f"minutos_{group}"}
        if group == "clase":
            cols.update({f"ach_{cases[0]}": f"ach_clase_{cases[0]}", f"ach_{cases[-1]}": f"ach_clase_{cases[-1]}", "sin_solucion_pct": f"sin_solucion_{cases[0]}_pct"})
        for src, dst in cols.items():
            report[dst] = sub[src].reindex(report.index)
    info = cases_table.set_index("caso")
    volume = float(info["volumen_m3"].iloc[0])
    for case in cases:
        col = "ach_clase" if case == mid else f"ach_clase_{case}"
        report[f"caudal_{case}"] = report[col] * volume / (info.loc[case, "ninos"] + 1) / 3.6
    for th in THRESHOLDS:
        report[f"ach_necesario_{th:g}"] = info.loc[mid, "fuente_ppm_h"] / (th - report["c_ext"])
    by_block = consensus_ach(ev, blocks, cases, by="bloque", mid=mid, max_rmse_rel=max_rmse_rel)
    return report, by_block, blocks


MONTHLY = ("co2_medio_clase", "co2_p95_clase", "pct_sobre_1000", "pct_sobre_1500", "temp_clase", "ach_clase", "ach_pausa", "ach_salida", "ach_noche", "caudal_medio")


def monthly_summary(report: pd.DataFrame, columns=MONTHLY) -> pd.DataFrame:
    """Por mes: días y cuartiles de cada indicador."""
    rows = []
    for month, g in report.groupby("mes"):
        row = {"mes": int(month), "dias": len(g), "dias_con_ach": int(g["ach_clase"].notna().sum())}
        for col in columns:
            if col in g:
                q = g[col].quantile([0.25, 0.5, 0.75])
                row.update({f"{col}_p25": q[0.25], f"{col}_mediana": q[0.5], f"{col}_p75": q[0.75]})
        rows.append(row)
    return pd.DataFrame(rows)
