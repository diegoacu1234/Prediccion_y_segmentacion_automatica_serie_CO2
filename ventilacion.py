"""Indicadores de ventilación de un salón, día por día.

Parte de las lecturas de CO2 y de los eventos que ya ajustaron los métodos
(`mediciones`), con el horario de ocupación y la generación de la clase, y arma
tres cosas para cada día hábil con clase:

- exposición: nivel de CO2 en horario de clase y tiempo por encima de umbrales;
- ACH por bloque del día: con la clase adentro, de las subidas ajustadas con la
  generación conocida; con el salón vacío, de las bajadas ajustadas con la
  asíntota en el exterior;
- caudal de aire exterior por persona.

El ACH de un bloque es un consenso entre métodos: el promedio de los ajustes de
todos los métodos ponderado por su duración (el mínimo y el máximo entre los
promedios de cada método dan la dispersión). Sólo entran los ajustes cuyo RMSE no
pasa de una fracción del recorrido de la curva. Con la clase adentro el ACH sale
sólo de los tramos en que el CO2 sube: no cubre los ratos en que está estable o
baja con gente, así que tiende a quedar por debajo del promedio de la clase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .mediciones import Occupancy, clock

# EN 16798-1:2019, anexo B (valores por defecto): exceso de CO2 sobre el exterior [ppm] que separa las
# categorías I, II y III, y caudal de aire exterior por persona [L/s] de cada una
EN_EXCESS = (550.0, 800.0, 1350.0)
EN_FLOW = (10.0, 7.0, 4.0)
CATEGORIES = ("I", "II", "III", "IV")
THRESHOLDS = (1000.0, 1500.0, 2000.0)  # ppm absolutos
GROUPS = ("clase", "pausa", "salida", "noche")


def day_blocks(occ: Occupancy, night_from: float = 18.5) -> pd.DataFrame:
    """Los bloques del día que deja el horario, en orden.

    `tipo` es "clase" o "vacio". `grupo` separa los bloques vacíos en "pausa"
    (entre dos bloques de clase: recreos y mediodía), "salida" (desde que se va
    la clase hasta `night_from`) y "noche" (el resto, con el salón cerrado).
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


def exposure(df: pd.DataFrame, daily: pd.DataFrame, blocks: pd.DataFrame, thresholds=THRESHOLDS, excess=EN_EXCESS) -> pd.DataFrame:
    """Nivel de CO2 en horario de clase de cada día hábil con clase.

    Por día: media, percentil 95 y máximo del CO2 en horario de clase, exceso
    medio sobre el exterior, porcentaje y horas del horario de clase por encima
    de cada umbral, porcentaje en cada categoría de EN 16798-1 (por el exceso
    sobre el exterior), temperatura y humedad medias, y `cobertura`: qué parte
    del horario de clase tiene lecturas.
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
    edges = (-np.inf, *excess, np.inf)
    for name, lo, hi in zip(CATEGORIES, edges[:-1], edges[1:]):
        out[f"pct_cat_{name}"] = 100.0 * g["exceso"].apply(lambda x, lo=lo, hi=hi: float(np.mean((x > lo) & (x <= hi))))
    for col, name in (("temp", "temp_clase"), ("hum", "hum_clase")):
        if col in sub:
            out[name] = g[col].mean()
    out["c_ext"] = daily["c_ext"].reindex(out.index)
    return out.reindex(days)


def _weighted(values, weights) -> float:
    ok = np.isfinite(values)
    return float(np.sum(values[ok] * weights[ok]) / np.sum(weights[ok])) if ok.any() else np.nan


def ach_by_method(ev: pd.DataFrame, blocks: pd.DataFrame, cases, by: str = "bloque", mid: str | None = None, max_rmse_rel: float = 0.05, min_inside: float = 0.8, exclude=("M1",)) -> pd.DataFrame:
    """ACH de cada día por bloque (o por grupo de bloques) y por método.

    `ev` son los eventos de `mediciones.events` con las columnas de
    `refit_empty` y `refit_occupied`; `cases`, los casos de generación de menor
    a mayor, y `mid`, el que se toma como valor central (por defecto, el del
    medio). Entran:

    - en los bloques de clase, las subidas con gente cuyo ajuste con el caso
      central tiene solución y un RMSE de a lo sumo `max_rmse_rel` del recorrido;
    - en los bloques vacíos, las bajadas en horario vacío con el mismo criterio
      para su ajuste de decaimiento puro.

    Cada evento va al bloque que contiene su punto medio. Una fila por fecha,
    bloque y método: `ach` (promedio de sus ajustes ponderado por duración),
    `menor` y `mayor` (lo mismo con los casos extremos de generación),
    `eventos`, `minutos` y `falta` (minutos sin solución con el caso de menor
    generación).
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
    """ACH de cada día por bloque (o por grupo de bloques), como consenso entre métodos.

    Parte de `ach_by_method`, con los mismos argumentos. Devuelve, por fecha y
    bloque: `ach` (promedio de los eventos de todos los métodos ponderado por
    su duración, o sea, de los promedios de cada método ponderados por los
    minutos que cubre), `ach_mediana` (la mediana entre métodos, que era el
    consenso antes), `ach_min` y `ach_max` entre métodos, `metodos`, `eventos`,
    `minutos` (mediana entre métodos de lo que cubren sus ajustes) y, en los
    bloques de clase, `ach_<caso>` de los casos extremos sobre los mismos
    tramos, ponderados igual, y `sin_solucion_pct`: qué parte de ese tiempo no
    tiene solución con el caso de menor generación.
    """
    cases = list(cases)
    lo, hi = cases[0], cases[-1]
    per = ach_by_method(ev, blocks, cases, by=by, mid=mid, max_rmse_rel=max_rmse_rel, min_inside=min_inside, exclude=exclude)
    if per.empty:
        return pd.DataFrame(columns=["fecha", by, "ach", "ach_mediana", "ach_min", "ach_max", "metodos", "eventos", "minutos", f"ach_{lo}", f"ach_{hi}", "sin_solucion_pct"])
    g = per.groupby(["fecha", by], sort=True)

    def pooled(col):
        return g.apply(lambda d: _weighted(d[col].to_numpy(dtype=float), d["minutos"].to_numpy(dtype=float)), include_groups=False)

    out = pd.DataFrame(
        {
            "ach": pooled("ach"),
            "ach_mediana": g["ach"].median(),
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
    """Una fila por día hábil con clase: exposición, ACH por grupo de bloques y caudal por persona.

    `cases_table` es la de `mediciones.generation_table` (casos de menor a
    mayor). Devuelve (informe diario, ACH por día y bloque, bloques). En el
    informe, `ach_clase` es el del caso central y `ach_clase_<caso>` los de los
    extremos; `caudal_<caso>` es el aire exterior por persona [L/s] que le
    corresponde a cada uno, ACH × volumen / personas; `ach_necesario_<umbral>`
    es el ACH de equilibrio que mantendría el CO2 en ese umbral con el caso
    central, S / (umbral − C_ext). Los días sin eventos (con huecos en las
    lecturas, o sin subidas) quedan con el ACH vacío y `analizado` en falso.
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
    for th in THRESHOLDS[:2]:
        report[f"ach_necesario_{th:g}"] = info.loc[mid, "fuente_ppm_h"] / (th - report["c_ext"])
    by_block = consensus_ach(ev, blocks, cases, by="bloque", mid=mid, max_rmse_rel=max_rmse_rel)
    return report, by_block, blocks


MONTHLY = ("co2_medio_clase", "co2_p95_clase", "pct_sobre_1000", "pct_sobre_1500", "pct_cat_I", "pct_cat_II", "pct_cat_III", "pct_cat_IV", "temp_clase", "ach_clase", "ach_pausa", "ach_salida", "ach_noche", "caudal_medio")


def monthly_summary(report: pd.DataFrame, columns=MONTHLY) -> pd.DataFrame:
    """Por mes: cuántos días hay y cuartiles de cada indicador."""
    rows = []
    for month, g in report.groupby("mes"):
        row = {"mes": int(month), "dias": len(g), "dias_con_ach": int(g["ach_clase"].notna().sum())}
        for col in columns:
            if col in g:
                q = g[col].quantile([0.25, 0.5, 0.75])
                row.update({f"{col}_p25": q[0.25], f"{col}_mediana": q[0.5], f"{col}_p75": q[0.75]})
        rows.append(row)
    return pd.DataFrame(rows)


WEEKDAYS_SHORT = ("lun", "mar", "mié", "jue", "vie", "sáb", "dom")
WEEKDAYS_LONG = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
MONTHS_SHORT = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")
MONTHS_LONG = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre")
GROUP_NAMES = {"clase": "con la clase adentro", "pausa": "en recreos y mediodía", "salida": "al irse la clase", "noche": "de noche, cerrado"}


def _es(value, digits: int = 1, unit: str = "") -> str:
    """Número con coma decimal, o una raya si no hay valor."""
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}".replace(".", ",") + unit


def _ach(value) -> str:
    return _es(value, 2 if np.isfinite(value) and value < 1 else 1)


def day_series_table(df: pd.DataFrame, report: pd.DataFrame, hours=(7.0, 19.0), bin_min: float = 5.0) -> pd.DataFrame:
    """CO2 de cada día del informe en tramos de `bin_min` minutos (mediana), entre las horas dadas; sin fila donde no hay lecturas."""
    t = df["tiempo"]
    hour = (t.dt.hour + t.dt.minute / 60.0).to_numpy()
    keep = t.dt.normalize().isin(pd.to_datetime(report.index)).to_numpy() & (hour >= hours[0]) & (hour < hours[1])
    sub = df[keep]
    slot = np.floor((sub["tiempo"].dt.hour * 60 + sub["tiempo"].dt.minute) / bin_min) * bin_min / 60.0
    out = sub.groupby([sub["tiempo"].dt.strftime("%Y-%m-%d").rename("fecha"), slot.rename("hora")])["co2"].median().round().astype(int).reset_index()
    out["hora"] = out["hora"].round(4)
    return out


def dashboard_tables(df: pd.DataFrame, report: pd.DataFrame, by_block: pd.DataFrame, blocks: pd.DataFrame, cases_table: pd.DataFrame, room: str, max_rmse_rel: float = 0.05) -> dict:
    """Las tablas que lee el tablero interactivo: números para los gráficos y, con prefijo `t_`, los textos que muestra.

    Devuelve `dias`, `resumen` (una fila con las medianas del año), `bloques`
    (ACH por día y bloque), `por_bloque` (el horario con los cuartiles de cada
    bloque), `meses`, `casos` y `series` (CO2 cada 5 minutos).
    """
    cases = list(cases_table["caso"])
    lo, mid, hi = cases[0], cases[len(cases) // 2], cases[-1]
    dates = pd.to_datetime(report.index)
    days = report.reset_index()
    days["t_dia"] = [f"{WEEKDAYS_SHORT[d.dayofweek]} {d.day} {MONTHS_SHORT[d.month - 1]}" for d in dates]
    days["t_fecha"] = [f"{WEEKDAYS_LONG[d.dayofweek]} {d.day} de {MONTHS_LONG[d.month - 1]} de {d.year}" for d in dates]
    days["t_co2"] = [_es(v, 0) for v in report["co2_medio_clase"]]
    days["t_p95"] = [_es(v, 0) for v in report["co2_p95_clase"]]
    days["t_pct1000"] = [_es(v, 0, " %") for v in report["pct_sobre_1000"]]
    days["t_ach_clase"] = [_ach(v) for v in report["ach_clase"]]
    low = [_ach(a) if np.isfinite(a) else "sin solución" for a in report[f"ach_clase_{lo}"]]
    days["t_rango"] = [f"{a} a {_ach(b)}" if np.isfinite(m) else "—" for a, b, m in zip(low, report[f"ach_clase_{hi}"], report["ach_clase"])]
    days["t_ach_pausa"] = [_ach(v) for v in report["ach_pausa"]]
    days["t_ach_salida"] = [_ach(v) for v in report["ach_salida"]]
    days["t_ach_noche"] = [_ach(v) for v in report["ach_noche"]]
    days["t_caudal"] = [_es(v, 1) for v in report[f"caudal_{mid}"]]
    days["t_temp"] = [_es(v, 1) for v in report["temp_clase"]]

    med = report.median(numeric_only=True)
    with_ach = report["ach_clase"].notna()
    flow = report.loc[with_ach, f"caudal_{mid}"]
    summary = pd.DataFrame(
        [
            {
                "salon": room,
                "desde": f"{dates.min().day} de {MONTHS_LONG[dates.min().month - 1]}",
                "hasta": f"{dates.max().day} de {MONTHS_LONG[dates.max().month - 1]} de {dates.max().year}",
                "dias": str(len(report)),
                "dias_con_ach": str(int(with_ach.sum())),
                "co2_medio_clase": _es(med["co2_medio_clase"], 0),
                "co2_p95_clase": _es(med["co2_p95_clase"], 0),
                "pct_sobre_1000": _es(med["pct_sobre_1000"], 0),
                "pct_sobre_1500": _es(med["pct_sobre_1500"], 0),
                "horas_sobre_1000": _es(med["horas_sobre_1000"], 1),
                "ach_clase": _ach(med["ach_clase"]),
                "ach_clase_bajo": _ach(med[f"ach_clase_{lo}"]),
                "ach_clase_alto": _ach(med[f"ach_clase_{hi}"]),
                "ach_pausa": _ach(med["ach_pausa"]),
                "ach_salida": _ach(med["ach_salida"]),
                "ach_noche": _ach(med["ach_noche"]),
                "caudal_medio": _es(med[f"caudal_{mid}"], 1),
                "caudal_bajo": _es(med[f"caudal_{lo}"], 1),
                "caudal_alto": _es(med[f"caudal_{hi}"], 1),
                "ach_necesario_1000": _es(med["ach_necesario_1000"], 1),
                "ach_necesario_1500": _es(med["ach_necesario_1500"], 1),
                "dias_bajo_cat_III": _es(100.0 * float(np.mean(flow < EN_FLOW[2])), 0),
                "pct_cat_IV": _es(report["pct_cat_IV"].mean(), 0),
                "rmse_maximo": _es(100.0 * max_rmse_rel, 0),
                "n_ach_necesario_1000": float(med["ach_necesario_1000"]),
                "n_ach_necesario_1500": float(med["ach_necesario_1500"]),
                "n_c_ext": float(med["c_ext"]),
            }
        ]
    )

    per_block = blocks.copy()
    per_block["nombre"] = per_block["grupo"].map(GROUP_NAMES)
    stats = by_block.groupby("bloque")["ach"].agg(dias="count", p25=lambda x: x.quantile(0.25), mediana="median", p75=lambda x: x.quantile(0.75))
    per_block = per_block.join(stats, on="bloque")
    per_block["dias"] = per_block["dias"].fillna(0).astype(int)

    rows = []
    for month, g in report.groupby("mes"):
        row = {"mes": int(month), "nombre": MONTHS_SHORT[int(month) - 1], "dias": len(g), "dias_con_ach": int(g["ach_clase"].notna().sum())}
        for col, name in (("co2_medio_clase", "co2"), ("pct_sobre_1000", "pct"), ("ach_clase", "ach"), ("ach_pausa", "pausa"), ("ach_noche", "noche")):
            q = g[col].quantile([0.25, 0.5, 0.75])
            row.update({f"{name}_p25": q[0.25], f"{name}_med": q[0.5], f"{name}_p75": q[0.75]})
        row.update({f"cat_{c}": g[f"pct_cat_{c}"].mean() for c in CATEGORIES})
        rows.append(row)
    months = pd.DataFrame(rows)

    case_rows = cases_table.assign(
        t_ninos=[str(int(v)) for v in cases_table["ninos"]],
        t_edad=[f"{v:g} años ({_es(m, 1) if m % 1 else f'{m:g}'} kg)" for v, m in zip(cases_table["edad"], cases_table["masa_kg"])],
        t_met=[f"{_es(a, 1)} / {_es(b, 1)} met" for a, b in zip(cases_table["met_ninos"], cases_table["met_maestra"])],
        t_lh_nino=[_es(v, 1) for v in cases_table["l_h_por_nino"]],
        t_lh_total=[_es(v, 0) for v in cases_table["l_h_total"]],
        t_fuente=[_es(v, 0) for v in cases_table["fuente_ppm_h"]],
    )
    return {
        "dias": days,
        "resumen": summary,
        "bloques": by_block[["fecha", "bloque", "ach", "ach_min", "ach_max", "metodos", f"ach_{lo}", f"ach_{hi}"]],
        "por_bloque": per_block,
        "meses": months,
        "casos": case_rows,
        "series": day_series_table(df, report),
    }


def filter_sensitivity(ev: pd.DataFrame, blocks: pd.DataFrame, cases, limits=(0.03, 0.05, 0.1, 1.0)) -> pd.DataFrame:
    """Cuánto cambia el ACH diario según qué tan bueno se le exige al ajuste (RMSE sobre recorrido)."""
    rows = []
    for limit in limits:
        g = consensus_ach(ev, blocks, cases, by="grupo", max_rmse_rel=limit)
        for group in GROUPS:
            sub = g[g["grupo"] == group]
            q = sub["ach"].quantile([0.25, 0.5, 0.75]) if len(sub) else pd.Series([np.nan] * 3, index=[0.25, 0.5, 0.75])
            spread = (sub["ach_max"] / sub["ach_min"])[sub["metodos"] >= 3]
            rows.append({"rmse_rel_maximo": limit, "grupo": group, "dias": len(sub), "ach_p25": q[0.25], "ach_mediana": q[0.5], "ach_p75": q[0.75], "max_sobre_min_entre_metodos": spread.median() if len(spread) else np.nan})
    return pd.DataFrame(rows)
