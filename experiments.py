"""Monte Carlo de los tres experimentos: meseta como asíntota, ruido y generación conocida.

Una corrida es una serie sintética (escenario, nivel de ruido, semilla). Sobre
cada una se segmenta una vez por método y se ajustan todas las variantes de
`co2ach.variants`. Las corridas son independientes y se reparten entre procesos.

Cada corrida se guarda en una carpeta temporal apenas termina, así una
ejecución interrumpida se puede reanudar sin repetir lo ya hecho.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .synth import SCENARIOS, simulate, with_sensor
from .variants import METHOD_KEYS, run_series_variants

FIT_COLUMNS = [
    "metodo", "tramo", "gen", "t0_h", "t1_h", "dur_min", "fase", "aceptado", "motivo",
    "ach_est", "ach_se", "ach_real", "error_rel", "regimen", "pureza", "lam_T", "r2",
]  # fmt: skip


@dataclass(frozen=True)
class Grid:
    """Diseño del Monte Carlo."""

    scenarios: tuple = ("A", "B", "C")
    sigmas: tuple = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0)  # ppm
    n_seeds: int = 30
    g_errors: tuple = (-0.4, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
    g_sigmas: tuple = (5.0, 20.0)  # niveles de ruido donde se barre el error en G
    methods: tuple = METHOD_KEYS
    version: int = 5

    def tasks(self) -> list:
        return [(k, s, i) for k in self.scenarios for s in self.sigmas for i in range(self.n_seeds)]

    def tag(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


def make_series(key: str, sigma: float, seed: int):
    """Serie del escenario `key` con ruido `sigma` y semilla `seed`."""
    return simulate(with_sensor(SCENARIOS[key](), noise_sd=float(sigma), seed=int(seed)))


def run_task(key: str, sigma: float, seed: int, grid: Grid, parts: str | None = None) -> dict:
    """Una corrida completa. Si `parts` es una carpeta, guarda ahí el resultado y lo reutiliza."""
    path = Path(parts) / f"{key}_{sigma:g}_{seed}.pkl" if parts else None
    if path is not None and path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    np.seterr(all="ignore")
    series = make_series(key, sigma, seed)
    g_errors = grid.g_errors if sigma in grid.g_sigmas else (0.0,)
    out = run_series_variants(series, g_errors=g_errors, methods=grid.methods)
    tables = {"regimenes": out["regimenes"], "ajustes": out["ajustes"][FIT_COLUMNS], "corridas": out["corridas"]}
    for tb in tables.values():
        tb.insert(0, "semilla", seed)
        tb.insert(0, "sigma", float(sigma))
        tb.insert(0, "escenario", key)
    tables["segundos"] = time.time() - t0
    if path is not None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(tables, fh)
        tmp.replace(path)
    return tables


def parts_dir(grid: Grid) -> Path:
    path = Path(tempfile.gettempdir()) / f"co2ach_mc_{grid.tag()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_grid(grid: Grid, out_dir, n_jobs: int = 4, verbose: int = 5) -> dict:
    """Corre todo el diseño en paralelo y guarda las tablas en `out_dir`."""
    from joblib import Parallel, delayed

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = str(parts_dir(grid))
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(run_task)(k, s, i, grid, parts) for k, s, i in grid.tasks())
    tables = {name: pd.concat([r[name] for r in results], ignore_index=True) for name in ("regimenes", "ajustes", "corridas")}
    for name, tb in tables.items():
        tb.to_csv(out_dir / f"mc_{name}.csv.gz", index=False, float_format="%.6g")
    info = dict(asdict(grid), corridas=len(results), segundos_cpu=float(sum(r["segundos"] for r in results)), segundos_reloj=time.time() - t0)
    (out_dir / "mc_config.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return tables


def load_results(out_dir) -> dict:
    """Lee las tablas guardadas por `run_grid`."""
    out_dir = Path(out_dir)
    tables = {name: pd.read_csv(out_dir / f"mc_{name}.csv.gz") for name in ("regimenes", "ajustes", "corridas")}
    tables["config"] = json.loads((out_dir / "mc_config.json").read_text(encoding="utf-8"))
    return tables
