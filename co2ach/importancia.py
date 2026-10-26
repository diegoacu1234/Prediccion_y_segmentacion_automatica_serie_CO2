"""Importancia de las variables de LightGBM (sección 8.3), con cuatro métodos:

- ganancia: lo que redujo la pérdida cada variable en los árboles;
- permutación: cuánto sube el error de prueba al desordenar la variable;
- SHAP: contribución de cada variable a cada pronóstico (TreeSHAP de LightGBM,
  `pred_contrib=True`); sumadas al valor base dan el pronóstico;
- quitar y reentrenar: cuánto sube el error al entrenar sin la variable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .prediccion import FULL, HORIZONS, LGB_PARAMS, LGB_ROUNDS, LGB_STOP, TARGETS, GBM, _lgb_X


def variable_name(col: str) -> str:
    """Nombre de la variable para las figuras."""
    fixed = {
        "co2": "CO₂ en t",
        "curv": "curvatura",
        "minuto": "minuto del día",
        "hora_sen": "hora (seno)",
        "hora_cos": "hora (coseno)",
        "dia_semana": "día de la semana",
        "temp": "temperatura",
        "hum": "humedad",
        "perfil": "perfil en t",
        "perfil_dif": "CO₂ − perfil",
        "max_dia": "máximo del día − CO₂",
        "exceso": "exceso sobre el exterior",
        "salon": "salón",
        "escuela": "escuela",
    }
    if col in fixed:
        return fixed[col]
    for prefix, text in (("dtemp", "cambio de temperatura en {} min"), ("dhum", "cambio de humedad en {} min"), ("perfil_h", "perfil: cambio a t+{}"), ("pend", "pendiente, {} min"), ("max", "máximo − CO₂, {} min"), ("min", "CO₂ − mínimo, {} min"), ("sd", "desvío, {} min"), ("d", "cambio en {} min")):
        if col.startswith(prefix) and col[len(prefix) :].isdigit():
            return text.format(col[len(prefix) :])
    return col


def variable_group(col: str) -> str:
    if col in ("minuto", "hora_sen", "hora_cos", "dia_semana"):
        return "hora y día"
    if col in ("temp", "hum") or col.startswith(("dtemp", "dhum")):
        return "temperatura y humedad"
    if col.startswith("perfil") or col in ("max_dia", "exceso"):
        return "historia del salón"
    if col in ("salon", "escuela"):
        return "salón"
    return "CO₂ reciente"


def booster(model: GBM, kind: str, h: int):
    return model.boosters[(kind, h)]


def gain_importance(model: GBM, kind: str = "mid") -> pd.DataFrame:
    """Ganancia de cada variable, como fracción del total de cada horizonte."""
    out = []
    for h in HORIZONS:
        if (kind, h) not in model.boosters:
            continue
        b = booster(model, kind, h)
        g = b.feature_importance(importance_type="gain", iteration=b.best_iteration)
        g = g / g.sum()
        out += [{"tipo": kind, "horizonte": h, "variable": c, "ganancia": v} for c, v in zip(b.feature_name(), g)]
    return pd.DataFrame(out)


def permutation_per_variable(model: GBM, rows: pd.DataFrame, horizons=HORIZONS, seed: int = 0, repeats: int = 1) -> pd.DataFrame:
    """Aumento del MAE del cambio pronosticado al desordenar cada variable, por horizonte."""
    rng = np.random.default_rng(seed)
    X = _lgb_X(rows, model.cols)
    dy = rows[TARGETS].to_numpy(float)
    out = []
    for h in horizons:
        j = HORIZONS.index(h)
        b = booster(model, "mid", h)
        base = np.mean(np.abs(b.predict(X, num_iteration=b.best_iteration) - dy[:, j]))
        for col in model.cols:
            inc = []
            for _ in range(repeats):
                Xp = X.copy()
                Xp[col] = X[col].to_numpy()[rng.permutation(len(X))]
                inc.append(np.mean(np.abs(b.predict(Xp, num_iteration=b.best_iteration) - dy[:, j])) - base)
            out.append({"horizonte": h, "variable": col, "mae_base": base, "aumento_mae": float(np.mean(inc))})
    return pd.DataFrame(out)


def shap_values(model: GBM, rows: pd.DataFrame, kind: str, h: int):
    """(contribuciones (n, variables), valor base)."""
    b = booster(model, kind, h)
    c = b.predict(_lgb_X(rows, model.cols), num_iteration=b.best_iteration, pred_contrib=True)
    return c[:, :-1], float(c[0, -1])


def shap_summary(model: GBM, rows: pd.DataFrame, kind: str = "mid", horizons=HORIZONS) -> pd.DataFrame:
    """Media de |SHAP| de cada variable, por horizonte."""
    out = []
    for h in horizons:
        if (kind, h) not in model.boosters:
            continue
        s, _ = shap_values(model, rows, kind, h)
        mean_abs = np.abs(s).mean(axis=0)
        out += [{"tipo": kind, "horizonte": h, "variable": c, "shap_medio": v, "shap_frac": v / mean_abs.sum()} for c, v in zip(model.cols, mean_abs)]
    return pd.DataFrame(out)


def fit_one(train: pd.DataFrame, es: pd.DataFrame, cols, h: int, threads: int = 4, seed: int = 0):
    """Un LightGBM de un horizonte, con los parámetros del modelo principal."""
    import lightgbm as lgb

    j = HORIZONS.index(h)
    cat = ["salon"] if "salon" in cols else []
    ds = lgb.Dataset(_lgb_X(train, cols), label=train[TARGETS].to_numpy(float)[:, j], categorical_feature=cat, params={"verbose": -1, "max_bin": 127})
    dv = lgb.Dataset(_lgb_X(es, cols), label=es[TARGETS].to_numpy(float)[:, j], reference=ds, categorical_feature=cat)
    params = {**LGB_PARAMS, "objective": "regression", "metric": "l1", "num_threads": threads, "seed": seed}
    return lgb.train(params, ds, num_boost_round=LGB_ROUNDS, valid_sets=[dv], callbacks=[lgb.early_stopping(LGB_STOP, verbose=False)])


def drop_column(train, es, test, variables, horizons=(15, 60), threads: int = 4, log=print) -> pd.DataFrame:
    """MAE de prueba al entrenar sin cada variable de `variables[h]`, contra el modelo con todas (figura 8.11)."""
    out = []
    for h in horizons:
        j = HORIZONS.index(h)
        dy = test[TARGETS].to_numpy(float)[:, j]
        for var in [None, *variables[h]]:
            cols = [c for c in FULL if c != var]
            b = fit_one(train, es, cols, h, threads)
            mae = float(np.mean(np.abs(b.predict(_lgb_X(test, cols), num_iteration=b.best_iteration) - dy)))
            out.append({"horizonte": h, "variable": var or "(ninguna)", "mae": mae, "arboles": b.best_iteration})
            log(f"quitar {var or '(ninguna)'} a {h} min: MAE {mae:.1f} ppm")
    df = pd.DataFrame(out)
    base = df[df["variable"] == "(ninguna)"].set_index("horizonte")["mae"]
    df["aumento_mae"] = df["mae"] - df["horizonte"].map(base)
    return df[df["variable"] != "(ninguna)"].assign(mae_base=lambda d: d["horizonte"].map(base))


def seed_variation(train, es, test, horizons=(15, 60), seeds=(1, 2, 3), threads: int = 4, log=print) -> pd.DataFrame:
    """MAE del modelo completo reentrenado con otras semillas: la escala del ruido de reentrenar."""
    out = []
    for h in horizons:
        j = HORIZONS.index(h)
        dy = test[TARGETS].to_numpy(float)[:, j]
        for seed in seeds:
            b = fit_one(train, es, FULL, h, threads, seed=seed)
            mae = float(np.mean(np.abs(b.predict(_lgb_X(test, FULL), num_iteration=b.best_iteration) - dy)))
            out.append({"horizonte": h, "semilla": seed, "mae": mae})
            log(f"semilla {seed} a {h} min: MAE {mae:.2f} ppm")
    return pd.DataFrame(out)
