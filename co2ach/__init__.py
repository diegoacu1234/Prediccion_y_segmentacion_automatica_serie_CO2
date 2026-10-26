"""co2ach: segmentación de series de CO2 en aulas, estimación de la ACH y pronóstico.

La interfaz pública está en `co2ach.ach` y `co2ach.pronostico`, y lo principal
también se importa desde acá:

    import co2ach
    dia = co2ach.segment_day(lecturas, "2024-10-08", methods=("hmm", "pelt"))
    modelo = co2ach.Forecaster.train({"A02": lecturas})

El resto de los módulos es el código de la tesis. Referencia de cada función en `docs/referencia.md`.
"""

__version__ = "0.3.0"

_ACH = (
    "METHODS", "DEFAULT_METHODS", "THESIS_CLASS", "THESIS_SCHEDULE", "Group", "Occupancy", "Selection",
    "SegmentationResult", "RoomAnalysis", "read_room", "prepare_readings", "daily_table", "segment",
    "segment_day", "analyze_room", "fit_event", "daily_ach", "class_source", "plot_day",
)  # fmt: skip
_PRONOSTICO = ("HORIZONS", "VARIABLES", "DEFAULT_VARIABLES", "Forecaster", "minute_grid", "threshold_crossings", "persistence_forecast", "mass_balance_forecast")

__all__ = ["__version__", *_ACH, *_PRONOSTICO]


def __getattr__(name):
    # importa ach / pronostico recién cuando se usan, así `import co2ach` es liviano
    if name in _ACH:
        from . import ach

        return getattr(ach, name)
    if name in _PRONOSTICO:
        from . import pronostico

        return getattr(pronostico, name)
    raise AttributeError(f"module 'co2ach' has no attribute {name!r}")
