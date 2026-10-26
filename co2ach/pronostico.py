"""Pronóstico del CO2 de un salón a 1–60 min y del cruce de un umbral: la interfaz pública (usa `prediccion`).

- `Forecaster`: el LightGBM global de los caps. 7 y 8. Se entrena con uno o más
  salones, pronostica en `HORIZONS` con intervalo del 80 % y probabilidad de
  pasar el umbral, se evalúa en días que no vio y se guarda/carga de una carpeta.
- `persistence_forecast` y `mass_balance_forecast`: las dos referencias sin entrenamiento.
- `minute_grid` y `threshold_crossings`: la grilla de 1 minuto y los cruces.

Las lecturas son tablas con `tiempo` y `co2` [ppm] (y `temp`, `hum` si se usan).
Los pronósticos salen en formato largo: una fila por origen y horizonte.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from . import prediccion as pr
from .ach import prepare_readings

__all__ = [
    "HORIZONS",
    "VARIABLES",
    "DEFAULT_VARIABLES",
    "Forecaster",
    "minute_grid",
    "threshold_crossings",
    "persistence_forecast",
    "mass_balance_forecast",
]

HORIZONS = pr.HORIZONS  # minutos
# grupos de variables (tabla 7.1); el CO2 reciente es obligatorio
VARIABLES = {
    "co2": pr.g0_columns(pr.FULL_HISTORY),  # valor, cambios, pendientes, curvatura y extremos de la última hora
    "calendario": pr.G1,  # minuto del día y día de la semana
    "clima": pr.G2,  # temperatura y humedad, y sus cambios
    "historia": pr.G3,  # perfil de los días anteriores, máximo del día y exceso sobre el exterior
    "salon": ["salon"],  # categórica
}
DEFAULT_VARIABLES = ("co2", "calendario", "clima", "historia", "salon")
FORECAST_COLUMNS = ["origen", "horizonte_min", "tiempo", "co2_origen", "pronostico", "li_80", "ls_80", "prob_sobre_umbral", "prob_cruce", "medido"]


# ------------------------------------------------------------------ datos de un salón


def _readings(readings) -> pd.DataFrame:
    df = prepare_readings(readings)
    for col in ("temp", "hum"):
        if col not in df:
            df[col] = np.nan
    return df


def minute_grid(readings) -> pd.DataFrame:
    """Lecturas en una grilla de 1 minuto (media por minuto, sin valores fuera de rango, huecos ≤ 5 min interpolados)."""
    return pr.minute_grid(_readings(readings))


def threshold_crossings(readings, threshold: float = 1000.0, rearm: float | None = None, persist: int = 2) -> pd.DataFrame:
    """Cruces del umbral hacia arriba en la grilla de 1 minuto: columnas `tiempo` y `co2`.

    Un cruce son `persist` minutos seguidos en o sobre `threshold`; no cuenta
    otro hasta bajar de `rearm` (por defecto, threshold − 50).
    """
    grid = minute_grid(readings)
    idx, _ = pr.crossings(grid["co2"].to_numpy(float), threshold, threshold - 50.0 if rearm is None else rearm, persist)
    return pd.DataFrame({"tiempo": grid.index[idx], "co2": grid["co2"].to_numpy(float)[idx]})


def _room_features(readings, code: int, threshold: float, rearm: float):
    """(grilla, tabla diaria, variables) de un salón."""
    df = _readings(readings)
    grid = pr.minute_grid(df)
    days = pr.day_table(df)
    feat = pr.room_features(grid, days, code, 0)
    if threshold != pr.THRESHOLD or rearm != pr.REARM:
        c = grid["co2"].to_numpy(float)
        cross, armed = pr.crossings(c, threshold, rearm)
        feat["habilitado"] = armed
        feat["min_al_cruce"] = pr.minutes_to_next(len(c), cross)
    return grid, days, feat


def _minutes(text: str) -> int:
    h, _, m = str(text).partition(":")
    return int(h) * 60 + (int(m) if m else 0)


def _select_origins(feat: pd.DataFrame, cols, split_fn, train_hours, eval_hours, step_min: int) -> pd.DataFrame:
    """Orígenes de un salón: días hábiles con clase, ventana horaria de cada partición y todo definido."""
    minute = feat["minuto"].to_numpy()
    split = np.asarray(split_fn(feat.index), dtype=object)
    t0, t1 = (_minutes(x) for x in train_hours)
    e0, e1 = (_minutes(x) for x in eval_hours)
    in_train = (split == "train") & (minute >= t0) & (minute < t1) & (minute % step_min == 0)
    in_eval = (split != "train") & (minute >= e0) & (minute <= e1)
    need = [c for c in cols if c != "salon"] + pr.TARGETS
    ok = feat["clase"].to_numpy() & (in_train | in_eval) & np.isfinite(feat[need].to_numpy(dtype=float)).all(axis=1)
    out = feat.loc[ok].copy()
    out.insert(0, "split", split[ok])
    return out


def _columns(variables) -> list:
    variables = tuple(variables)
    unknown = [v for v in variables if v not in VARIABLES]
    if unknown:
        raise ValueError(f"variables desconocidas: {unknown}; las opciones son {list(VARIABLES)}")
    if "co2" not in variables:
        raise ValueError("el grupo 'co2' (CO2 reciente) es obligatorio")
    return [c for v in variables for c in VARIABLES[v]]


# ------------------------------------------------------------------ el modelo


@dataclass
class Forecaster:
    """El LightGBM de la tesis (`Forecaster.train` o `Forecaster.load`).

    rooms: salón de entrenamiento -> código; metrics: error en los días de
    prueba (una fila por horizonte), si hubo.
    """

    cols: list
    variables: tuple
    rooms: dict
    threshold: float
    rearm: float
    boosters: dict = field(repr=False)  # (tipo, horizonte) -> Booster; tipo: "mid", "q10", "q90", "clasif"
    residual_q: np.ndarray = field(repr=False)  # cuantiles de los residuos de calibración, (1001, horizontes)
    cqr: np.ndarray | None = field(default=None, repr=False)  # ensanche CQR por horizonte
    metrics: pd.DataFrame | None = None
    info: dict = field(default_factory=dict)

    # -------------------------------------------------------------- entrenamiento

    @classmethod
    def train(
        cls,
        rooms: dict,
        variables=DEFAULT_VARIABLES,
        threshold: float = 1000.0,
        rearm: float | None = None,
        quantiles: bool = True,
        classifier: bool = True,
        split=None,
        train_hours=("07:00", "18:00"),
        eval_hours=("07:30", "17:30"),
        step_min: int = 2,
        threads: int = 4,
        params: dict | None = None,
        verbose: bool = False,
    ) -> "Forecaster":
        """Entrena con {nombre del salón: lecturas}.

        `split(DatetimeIndex)` da "train", "es" (detención temprana), "cal"
        (calibración) o "test" para cada instante; por defecto, la partición de
        la tesis por días del mes (1–15, 16–19, 20–23, 24–fin). Sólo entran días
        hábiles con clase. Sin temperatura y humedad, sacar "clima" de
        `variables`. `quantiles`: intervalo por cuantiles; `classifier`:
        probabilidad de cruzar el umbral.
        """
        rearm = threshold - 50.0 if rearm is None else float(rearm)
        cols = _columns(variables)
        split_fn = split or pr.split_of
        codes = {name: k for k, name in enumerate(rooms)}
        parts = []
        for name, readings in rooms.items():
            _, _, feat = _room_features(readings, codes[name], threshold, rearm)
            o = _select_origins(feat, cols, split_fn, train_hours, eval_hours, step_min)
            o.insert(0, "sala", name)
            parts.append(o)
            if verbose:
                print(f"  {name}: {len(o):,} orígenes", flush=True)
        rows = pd.concat(parts)
        rows.index.name = "tiempo"
        rows = rows.reset_index()
        part = {k: rows[rows["split"] == k].reset_index(drop=True) for k in pr.SPLITS}
        if part["train"].empty or part["es"].empty:
            raise ValueError("hacen falta orígenes de entrenamiento ('train') y de detención temprana ('es'); revisar la partición y los días con clase")
        if verbose:
            print("  orígenes por partición: " + ", ".join(f"{k} {len(v):,}" for k, v in part.items()), flush=True)
        with warnings.catch_warnings():  # aviso esperado de LightGBM
            warnings.filterwarnings("ignore", message="Overriding the parameters from Reference Dataset")
            g = pr.GBM(cols, quantiles=quantiles, classifier=classifier, params=params, threads=threads, verbose=verbose).fit(None, part["train"], part["es"])
        cal = part["cal"] if not part["cal"].empty else part["es"]
        y_cal, mid_cal = pr.actual(cal), g.predict(None, cal)
        cqr = None
        if quantiles:
            lo_c, hi_c = g.predict_quantiles(cal)
            cqr = pr.cqr_offset(y_cal, lo_c, hi_c)
        model = cls(
            cols=cols,
            variables=tuple(variables),
            rooms=codes,
            threshold=float(threshold),
            rearm=rearm,
            boosters=dict(g.boosters),
            residual_q=pr.Residuals(y_cal, mid_cal).q,
            cqr=cqr,
            info={
                "version": __version__,
                "origenes": {k: int(len(v)) for k, v in part.items()},
                "calibracion": "cal" if not part["cal"].empty else "es",
                "train_hours": list(train_hours),
                "eval_hours": list(eval_hours),
                "step_min": step_min,
            },
        )
        if not part["test"].empty:
            model.metrics = model._metrics(part["test"])
        return model

    # -------------------------------------------------------------- pronóstico

    def _gbm(self) -> pr.GBM:
        g = pr.GBM(self.cols, quantiles=self.cqr is not None, classifier=any(k == "clasif" for k, _ in self.boosters))
        g.boosters = self.boosters
        return g

    def _residuals(self) -> pr.Residuals:
        res = pr.Residuals.__new__(pr.Residuals)
        res.q = self.residual_q
        return res

    def _forecast(self, rows: pd.DataFrame, threshold: float | None = None) -> dict:
        """Pronóstico, intervalo y probabilidades, cada uno (n, horizontes)."""
        g = self._gbm()
        thr = self.threshold if threshold is None else float(threshold)
        mid = g.predict(None, rows)
        if self.cqr is not None:
            lo, hi = g.predict_quantiles(rows)
            lo, hi = lo - self.cqr, hi + self.cqr
        else:
            lo, hi = self._residuals().interval(mid)
        p = self._residuals().prob_above(mid, thr)
        pw = g.predict_within(rows) if (("clasif", HORIZONS[0]) in self.boosters and thr == self.threshold) else pr.within(p)
        return {"pronostico": mid, "li_80": lo, "ls_80": hi, "prob_sobre_umbral": p, "prob_cruce": pw}

    def predict(self, readings, room: str | None = None, at=None, threshold: float | None = None) -> pd.DataFrame:
        """Pronostica desde uno o más orígenes.

        Conviene pasar al menos la última semana de lecturas (las variables de
        historia usan los días anteriores). `room`: salón de entrenamiento, o
        None para uno nuevo. `at`: None (último minuto con dato), "todos" o una
        lista de instantes. Con otro `threshold`, prob_cruce sale de la
        probabilidad y no del clasificador. Columnas en `FORECAST_COLUMNS`.
        """
        code = self.rooms.get(room, -1) if room is not None else -1
        grid, _, feat = _room_features(readings, code, self.threshold, self.rearm)
        rows = _origin_rows(feat, at)
        out = self._forecast(rows, threshold)
        return _long(rows, grid, out)

    # -------------------------------------------------------------- evaluación

    def _metrics(self, rows: pd.DataFrame) -> pd.DataFrame:
        fc = self._forecast(rows)
        y = pr.actual(rows)
        m = pr.point_metrics(rows, fc["pronostico"])
        inside = (y >= fc["li_80"]) & (y <= fc["ls_80"])
        m["cobertura_80"] = np.nanmean(np.where(np.isfinite(y), inside, np.nan), axis=0)
        m["ancho_80"] = np.nanmean(fc["ls_80"] - fc["li_80"], axis=0)
        return m

    def evaluate(self, rooms: dict, split=None, which: str = "test") -> pd.DataFrame:
        """Error en los días de `rooms` que caen en la partición `which`: mae, rmse, sesgo, n, cobertura_80 y ancho_80, por horizonte."""
        split_fn = split or pr.split_of
        parts = []
        for name, readings in rooms.items():
            _, _, feat = _room_features(readings, self.rooms.get(name, -1), self.threshold, self.rearm)
            o = _select_origins(feat, self.cols, split_fn, self.info.get("train_hours", ("07:00", "18:00")), self.info.get("eval_hours", ("07:30", "17:30")), self.info.get("step_min", 2))
            parts.append(o[o["split"] == which])
        rows = pd.concat(parts).reset_index(drop=True)
        if rows.empty:
            raise ValueError(f"no hay orígenes en la partición {which!r}")
        return self._metrics(rows)

    # -------------------------------------------------------------- guardar y cargar

    def save(self, folder) -> None:
        """Un .txt por modelo de LightGBM, los residuos y `modelo.json`."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for (kind, h), b in self.boosters.items():
            b.save_model(str(folder / f"{kind}_h{h:02d}.txt"), num_iteration=b.best_iteration if b.best_iteration > 0 else None)
        np.save(folder / "residuos.npy", self.residual_q)
        meta = {
            "cols": self.cols,
            "variables": list(self.variables),
            "rooms": self.rooms,
            "threshold": self.threshold,
            "rearm": self.rearm,
            "horizons": list(HORIZONS),
            "cqr": None if self.cqr is None else [float(v) for v in self.cqr],
            "boosters": [[k, int(h)] for k, h in self.boosters],
            "info": self.info,
        }
        (folder / "modelo.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.metrics is not None:
            self.metrics.to_csv(folder / "error_en_prueba.csv", index=False)

    @classmethod
    def load(cls, folder) -> "Forecaster":
        import lightgbm as lgb

        folder = Path(folder)
        meta = json.loads((folder / "modelo.json").read_text(encoding="utf-8"))
        if tuple(meta["horizons"]) != tuple(HORIZONS):
            raise ValueError("el modelo se entrenó con otros horizontes")
        boosters = {(k, h): lgb.Booster(model_file=str(folder / f"{k}_h{h:02d}.txt")) for k, h in meta["boosters"]}
        metrics_path = folder / "error_en_prueba.csv"
        return cls(
            cols=meta["cols"],
            variables=tuple(meta["variables"]),
            rooms={k: int(v) for k, v in meta["rooms"].items()},
            threshold=float(meta["threshold"]),
            rearm=float(meta["rearm"]),
            boosters=boosters,
            residual_q=np.load(folder / "residuos.npy"),
            cqr=None if meta["cqr"] is None else np.asarray(meta["cqr"], float),
            metrics=pd.read_csv(metrics_path) if metrics_path.exists() else None,
            info=meta.get("info", {}),
        )


# ------------------------------------------------------------------ orígenes y salida


def _origin_rows(feat: pd.DataFrame, at) -> pd.DataFrame:
    """Las filas de variables de los orígenes pedidos."""
    has = np.isfinite(feat["co2"].to_numpy(float))
    if at is None:
        if not has.any():
            raise ValueError("las lecturas no tienen ningún minuto con CO2")
        rows = feat.iloc[[int(np.flatnonzero(has)[-1])]]
    elif isinstance(at, str) and at == "todos":
        rows = feat.loc[has]
    else:
        stamps = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(at))).floor("min")
        missing = stamps.difference(feat.index[has])
        if len(missing):
            raise ValueError(f"no hay CO2 en {len(missing)} de los orígenes pedidos, p. ej. {missing[0]}")
        rows = feat.loc[stamps]
    rows = rows.copy()
    rows.index.name = "tiempo"
    return rows.reset_index()


def _long(rows: pd.DataFrame, grid: pd.DataFrame, values: dict) -> pd.DataFrame:
    """(n, horizontes) -> una fila por origen y horizonte."""
    n, nh = len(rows), len(HORIZONS)
    origin = np.repeat(rows["tiempo"].to_numpy(), nh)
    h = np.tile(np.asarray(HORIZONS), n)
    target = origin + h.astype("timedelta64[m]")
    measured = grid["co2"].reindex(target).to_numpy(float)
    out = {"origen": origin, "horizonte_min": h, "tiempo": target, "co2_origen": np.repeat(rows["co2"].to_numpy(float), nh)}
    for col in FORECAST_COLUMNS[4:-1]:
        v = values.get(col)
        out[col] = np.asarray(v, float).ravel() if v is not None else np.full(n * nh, np.nan)
    out["medido"] = measured
    return pd.DataFrame(out)[FORECAST_COLUMNS]


# ------------------------------------------------------------------ referencias sin entrenamiento


def persistence_forecast(readings, at=None) -> pd.DataFrame:
    """Persistencia: C(t+h) = C(t). Mismo formato que `Forecaster.predict`."""
    grid, _, feat = _room_features(readings, -1, pr.THRESHOLD, pr.REARM)
    rows = _origin_rows(feat, at)
    c0 = rows["co2"].to_numpy(float)
    return _long(rows, grid, {"pronostico": np.repeat(c0[:, None], len(HORIZONS), axis=1)})


def mass_balance_forecast(readings, at=None, window: int = 15) -> pd.DataFrame:
    """Balance de masa en línea: recta ΔC = a − β·C̄ en los últimos `window` min, extrapolada como exponencial.

    ACH ≤ 20 1/h y pronóstico entre C_ext y 5000 ppm. Agrega `ach_estimada` [1/h].
    """
    grid, _, feat = _room_features(readings, -1, pr.THRESHOLD, pr.REARM)
    rows = _origin_rows(feat, at)
    a, beta = pr.physical_params(grid["co2"].to_numpy(float), int(window))
    pos = ((rows["tiempo"].to_numpy() - grid.index[0].to_datetime64()) // np.timedelta64(1, "m")).astype(int)
    c_ext = rows["c_ext"].to_numpy(float)
    c_ext = np.where(np.isfinite(c_ext), c_ext, pr.CO2_RANGE[0])
    mid = pr.physical_forecast(rows["co2"], a[pos], beta[pos], c_ext)
    out = _long(rows, grid, {"pronostico": mid})
    out["ach_estimada"] = np.repeat(60.0 * 2.0 * np.arctanh(np.clip(beta[pos], 0.0, pr.BETA_MAX) / 2.0), len(HORIZONS))
    return out
