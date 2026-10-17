"""Importancia de las variables de LightGBM, variable por variable y con varios métodos.

- Ganancia: cuánto redujo la pérdida cada variable en las divisiones de los
  árboles, sumado sobre el entrenamiento. Es interna al modelo y no mira la
  prueba.
- Permutación: cuánto sube el error de prueba al desordenar una variable entre
  los orígenes. Mide lo que el modelo usa; si dos variables están
  correlacionadas, desordenar una sola rompe combinaciones que nunca ocurren.
- SHAP: la contribución de cada variable a cada pronóstico (TreeSHAP, que
  LightGBM calcula con `pred_contrib=True`). Las contribuciones de un origen
  suman, junto con el valor base, el cambio pronosticado.
- Quitar y reentrenar: cuánto sube el error al entrenar sin la variable. Mide
  lo que el modelo necesita: si otra variable dice lo mismo, el error casi no
  cambia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .prediccion import FULL, HORIZONS, LGB_PARAMS, LGB_ROUNDS, LGB_STOP, TARGETS, GBM, _lgb_X

LAG_NAMES = {1: "1", 2: "2", 3: "3", 5: "5", 10: "10", 15: "15", 20: "20", 30: "30", 45: "45", 60: "60"}


def variable_name(col: str) -> str:
    """Nombre de una variable para las figuras."""
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
    """Ganancia de cada variable, en fracción del total de cada horizonte."""
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
    """Aumento del error absoluto medio del cambio pronosticado al desordenar cada variable, por horizonte."""
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
    """Contribuciones SHAP de cada variable a cada origen (n, variables) y el valor base."""
    b = booster(model, kind, h)
    c = b.predict(_lgb_X(rows, model.cols), num_iteration=b.best_iteration, pred_contrib=True)
    return c[:, :-1], float(c[0, -1])


def shap_summary(model: GBM, rows: pd.DataFrame, kind: str = "mid", horizons=HORIZONS) -> pd.DataFrame:
    """Media del valor absoluto de SHAP de cada variable, por horizonte."""
    out = []
    for h in horizons:
        if (kind, h) not in model.boosters:
            continue
        s, _ = shap_values(model, rows, kind, h)
        mean_abs = np.abs(s).mean(axis=0)
        out += [{"tipo": kind, "horizonte": h, "variable": c, "shap_medio": v, "shap_frac": v / mean_abs.sum()} for c, v in zip(model.cols, mean_abs)]
    return pd.DataFrame(out)


def fit_one(train: pd.DataFrame, es: pd.DataFrame, cols, h: int, threads: int = 4, seed: int = 0):
    """Un LightGBM para un horizonte, con los mismos parámetros que el modelo principal."""
    import lightgbm as lgb

    j = HORIZONS.index(h)
    cat = ["salon"] if "salon" in cols else []
    ds = lgb.Dataset(_lgb_X(train, cols), label=train[TARGETS].to_numpy(float)[:, j], categorical_feature=cat, params={"verbose": -1, "max_bin": 127})
    dv = lgb.Dataset(_lgb_X(es, cols), label=es[TARGETS].to_numpy(float)[:, j], reference=ds, categorical_feature=cat)
    params = {**LGB_PARAMS, "objective": "regression", "metric": "l1", "num_threads": threads, "seed": seed}
    return lgb.train(params, ds, num_boost_round=LGB_ROUNDS, valid_sets=[dv], callbacks=[lgb.early_stopping(LGB_STOP, verbose=False)])


def drop_column(train, es, test, variables, horizons=(15, 60), threads: int = 4, log=print) -> pd.DataFrame:
    """Error de prueba al entrenar sin cada variable, frente al modelo con todas (mismos parámetros y semilla)."""
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
    """Error de prueba del modelo completo reentrenado con otras semillas: la escala del ruido de reentrenar."""
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
