"""Comparación con la verdad de las series sintéticas (sección 4.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .physics import generation_from_source
from .segmenters.base import DECAY, RISE, STEADY
from .synth import PHASE_NAMES


def truth_for_span(series, i0: int, i1: int) -> dict:
    """Verdad de [i0, i1): el ACH del régimen dominante y qué fracción del tramo le toca (`purity`)."""
    rid = series.regime_id[i0:i1]
    counts = np.bincount(rid)
    dom = int(np.argmax(counts))
    inside = rid == dom
    return {
        "ach_true": float(np.mean(series.ach[i0:i1][inside])),
        "s_true": float(np.mean(series.s[i0:i1][inside])),
        "regime": dom,
        "purity": float(counts[dom] / len(rid)),
    }


def fits_table(series, result) -> pd.DataFrame:
    """Una fila por ajuste, con su verdad y su error."""
    vol = series.scenario.room.volume_m3
    rows = []
    for f in result.fits:
        tr = truth_for_span(series, f.i0, f.i1)
        rows.append(
            {
                "metodo": result.key,
                "t0_h": series.t_h[f.i0],
                "t1_h": series.t_h[f.i1 - 1] + series.dt_h,
                "dur_min": 60.0 * f.duration_h,
                "fase": PHASE_NAMES[f.phase],
                "modelo": f.model,
                "aceptado": f.accepted,
                "motivo": f.reason,
                "ach_est": f.lam,
                "ach_se": f.lam_se,
                "ach_real": tr["ach_true"],
                "error_rel": (f.lam - tr["ach_true"]) / tr["ach_true"] if f.ok else np.nan,
                "regimen": tr["regime"],
                "pureza": tr["purity"],
                "lam_T": tr["ach_true"] * f.duration_h,
                "r2": f.r2,
                "rmse": f.rmse,
                "css_est": f.css,
                "g_est_lh": float(generation_from_source(f.s, vol)) if f.ok else np.nan,
                "g_real_lh": float(generation_from_source(tr["s_true"], vol)),
                "i0": f.i0,
                "i1": f.i1,
            }
        )
    cols = [
        "metodo", "t0_h", "t1_h", "dur_min", "fase", "modelo", "aceptado", "motivo", "ach_est", "ach_se", "ach_real", "error_rel",
        "regimen", "pureza", "lam_T", "r2", "rmse", "css_est", "g_est_lh", "g_real_lh", "i0", "i1",
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=cols)


def true_onsets(series) -> pd.DataFrame:
    """Cambios de régimen que arrancan un transitorio (sin contar t = 0)."""
    reg = series.regimes
    return reg[(reg["tipo"] != STEADY) & (reg["i0"] > 0)][["regimen", "i0", "tipo"]].reset_index(drop=True)


def onset_metrics(series, segmentation, tol_min: float = 5.0) -> dict:
    """Precisión, recall y F1 de los inicios, con ±tol_min y la misma fase."""
    tol = int(round(tol_min / 60.0 / series.dt_h))
    truth = true_onsets(series)
    pred = [(s.i0, s.phase) for s in segmentation.transients if s.i0 > 0]
    used = set()
    hits, loc_err = 0, []
    hit_kind = {RISE: 0, DECAY: 0}
    for row in truth.itertuples():
        cand = [(abs(p - row.i0), j) for j, (p, ph) in enumerate(pred) if j not in used and ph == row.tipo and abs(p - row.i0) <= tol]
        if cand:
            d, j = min(cand)
            used.add(j)
            hits += 1
            hit_kind[int(row.tipo)] += 1
            loc_err.append(d * series.dt_h * 60.0)
    n_true, n_pred = len(truth), len(pred)
    precision = hits / n_pred if n_pred else np.nan
    recall = hits / n_true if n_true else np.nan
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    n_rise = int((truth["tipo"] == RISE).sum())
    n_decay = int((truth["tipo"] == DECAY).sum())
    return {
        "inicios_reales": n_true,
        "inicios_detectados": n_pred,
        "inicios_acertados": hits,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "recall_subidas": hit_kind[RISE] / n_rise if n_rise else np.nan,
        "recall_bajadas": hit_kind[DECAY] / n_decay if n_decay else np.nan,
        "error_ubicacion_min": float(np.mean(loc_err)) if loc_err else np.nan,
    }


def regime_estimates(series, table: pd.DataFrame, min_purity: float = 0.8) -> pd.DataFrame:
    """ACH de cada régimen verdadero: media de sus ajustes aceptados ponderada por duración."""
    rows = []
    for reg in series.regimes[series.regimes["tipo"] != STEADY].itertuples():
        sel = table[(table["aceptado"]) & (table["regimen"] == reg.regimen) & (table["pureza"] >= min_purity)]
        if len(sel):
            est = float(np.average(sel["ach_est"], weights=sel["dur_min"]))
            rows.append({"regimen": reg.regimen, "ach_real": reg.ach, "ach_est": est, "error_rel": (est - reg.ach) / reg.ach, "n_ajustes": len(sel)})
        else:
            rows.append({"regimen": reg.regimen, "ach_real": reg.ach, "ach_est": np.nan, "error_rel": np.nan, "n_ajustes": 0})
    return pd.DataFrame(rows)


def summarize(series, result, table: pd.DataFrame | None = None) -> dict:
    """Resumen de un método sobre una serie."""
    table = fits_table(series, result) if table is None else table
    acc = table[table["aceptado"]]
    err = acc["error_rel"].to_numpy(dtype=float)
    reg = regime_estimates(series, table)
    reg_err = reg["error_rel"].dropna().to_numpy(dtype=float)
    out = {
        "metodo": result.key,
        "nombre": result.label,
        "transitorios": len(result.segmentation.transients),
        "ajustes_aceptados": len(acc),
        "error_abs_mediano": float(np.median(np.abs(err))) if len(err) else np.nan,
        "sesgo_mediano": float(np.median(err)) if len(err) else np.nan,
        "dentro_20pct": float(np.mean(np.abs(err) <= 0.2)) if len(err) else np.nan,
        "regimenes_con_evento": len(reg),
        "regimenes_cubiertos": int(reg["ach_est"].notna().sum()),
        "error_abs_mediano_por_regimen": float(np.median(np.abs(reg_err))) if len(reg_err) else np.nan,
    }
    out.update(onset_metrics(series, result.segmentation))
    return out
