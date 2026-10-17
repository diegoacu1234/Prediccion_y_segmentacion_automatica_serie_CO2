"""co2ach: segmentación de series de CO2 interior, estimación de la tasa de renovación de aire (ACH) y pronóstico.

La interfaz pública está en dos módulos:

- `co2ach.ach`: segmentar una serie, un día o un salón entero y estimar la ACH
  de cada evento y de cada día.
- `co2ach.pronostico`: entrenar, usar, evaluar y guardar el pronóstico del CO2
  a 1–60 minutos y de la probabilidad de superar un umbral.

Las funciones principales también se pueden importar desde `co2ach`:

    import co2ach
    resultado = co2ach.segment_day(lecturas, "2024-10-08", methods=("hmm", "pelt"))
    modelo = co2ach.Forecaster.train({"A02": lecturas})

El resto de los módulos (`pipeline`, `fitting`, `segmenters`, `mediciones`,
`prediccion`, ...) es el código de la tesis sobre el que se apoya esta
interfaz. La referencia de cada función está en `docs/referencia.md`.
"""

__version__ = "0.2.0"

_ACH = (
    "METHODS", "DEFAULT_METHODS", "THESIS_CLASS", "THESIS_SCHEDULE", "Group", "Occupancy", "Selection",
    "SegmentationResult", "RoomAnalysis", "read_room", "prepare_readings", "daily_table", "segment",
    "segment_day", "analyze_room", "fit_event", "daily_ach", "class_source", "plot_day",
)  # fmt: skip
_PRONOSTICO = ("HORIZONS", "VARIABLES", "DEFAULT_VARIABLES", "Forecaster", "minute_grid", "threshold_crossings", "persistence_forecast", "mass_balance_forecast")

__all__ = ["__version__", *_ACH, *_PRONOSTICO]


def __getattr__(name):
    """Importa los módulos públicos recién cuando se usan, para que `import co2ach` sea liviano."""
    if name in _ACH:
        from . import ach

        return getattr(ach, name)
    if name in _PRONOSTICO:
        from . import pronostico

        return getattr(pronostico, name)
    raise AttributeError(f"module 'co2ach' has no attribute {name!r}")
