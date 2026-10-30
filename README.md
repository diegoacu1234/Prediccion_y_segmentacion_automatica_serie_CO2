# Segmentación de series de CO₂ y estimación de ACH

Implementa el método de Hyun (2025) para estimar la tasa de renovación de aire (ACH) a partir de una serie de CO₂ interior, dos variantes propias y tres alternativas, y los compara sobre series sintéticas con ACH conocido.

## Librería
El paquete `co2ach` se instala con pip (`pip install -e ".[completo]"` desde esta carpeta, o el `.whl` de `dist/`) y tiene una interfaz pública para usar la segmentación, la estimación de la ACH y el pronóstico sobre otros datos: `co2ach.ach` y `co2ach.pronostico`. La guía está en [`LIBRERIA.md`](LIBRERIA.md), la especificación de cada función en [`docs/referencia.md`](docs/referencia.md) y dos ejemplos ejecutables en `ejemplos/`. Para volver a armar el `.whl`: `py -3.12 -m pip wheel . --no-deps --no-build-isolation -w dist`.

## Cómo correr

Usar Python 3.12 (`py -3.12`) o el Python de Anaconda. El `python` del PATH de esta máquina tiene scikit-learn y pandas rotos.

```
py -3.12 scripts/run_comparison.py                 # los tres escenarios, ~2 min
py -3.12 scripts/run_comparison.py --escenarios A  # sólo uno
py -3.12 -m unittest discover -s tests -v          # tests, ~2 min
```

El script deja las figuras en `outputs/figs`, las tablas en `outputs/tables` y las series generadas en `outputs/data`.

`notebooks/comparacion.ipynb` hace lo mismo para un escenario y permite cambiar el ruido y los hiperparámetros. En VS Code, elegir como kernel Python 3.12 o Anaconda.

## Métodos

| Clave | Segmentación | Ajuste |
|---|---|---|
| M0 | Oráculo: regímenes verdaderos | exponencial general |
| M0t | Oráculo: sólo el transitorio de cada régimen | exponencial general |
| M1 | Hyun fiel, preset School: mediana + K-means + DBSCAN | sólo subidas, (E, λ) |
| M1th | Hyun fiel, preset Test house | sólo subidas, (E, λ) |
| M2 | Hyun, usando también el cluster de bajada | bajadas con generación (E ≥ 0 libre) |
| M3a | Hyun con C″ como tercer feature de K-means | como M2 |
| M3b | Hyun cortando los sub-clusters en los quiebres de C″ | como M2 |
| M4 | Derivada: umbral con histéresis sobre C′ y quiebres de C″ | exponencial general |
| M5 | PELT con costo exponencial | exponencial general, régimen completo |
| M6 | HMM gaussiano de 3 estados sobre C′ | exponencial general |

La "exponencial general" ajusta `C(t) = C_ss + (C0 − C_ss)·exp(−λ(t − t0))` con C0, C_ss ≥ C_ext y λ libres, sobre los datos sin outliers. El ajuste de Hyun fija C0 en la primera muestra filtrada y estima E y λ.

Para mediciones reales usaría el HMM, que fue el que más se pareció a la segmentación hecha a mano (cap. 6). En series que cumplen bien el modelo, PELT.

## Escenarios

- **A, día de aula**: eventos bien separados, con y sin meseta.
- **B, quiebres dentro de una misma fase**: sube y vuelve a subir más rápido, baja y sigue bajando más lento.
- **C, difícil**: ventilación baja, eventos de 20 a 45 minutos, más ruido, cambios graduales de ocupación y deriva del CO₂ exterior.

Están definidos en `co2ach/synth.py`; ahí se cambian el ACH, la ocupación, el volumen y el sensor.

## Qué hay en cada archivo

```
co2ach/
  physics.py         balance de masa y generación de CO₂ de las personas (cap. 2, apéndice A)
  signal_utils.py    ruido, picos, derivadas de Savitzky–Golay y quiebres (apéndice B.4)
  fitting.py         ajuste de la exponencial: libre, generación conocida y acotada (4.3, B.2, B.3)
  segmenters/        Hyun y variantes, derivada, PELT, HMM y oráculo (4.4)
  regimes.py         de fases a regímenes: el tramo con su meseta (4.3.4)
  pipeline.py        cada método = segmentador + ajuste + filtros
  synth.py           series sintéticas y escenarios A, B y C (5.1)
  evaluate.py        comparación con la verdad
  variants.py        ajuste sólo del transitorio o con la meseta (figura 4.7)
  cfd.py             la simulación CFD (5.2)
  mediciones.py      mediciones del aula, ruido efectivo y reajustes con la ocupación (cap. 6)
  manual.py          la referencia marcada a mano (6.8)
  ventilacion.py     indicadores de ventilación por día (6.7)
  salones.py         horario, volumen y clase del salón A02
  prediccion.py      variables, modelos, intervalos y cruces de 1000 ppm (caps. 7 y 8)
  importancia.py     importancia de las variables (8.3)
  ach.py             interfaz pública de segmentación y ACH
  pronostico.py      interfaz pública del pronóstico
  plots*.py          las figuras de la tesis
```

## Datos

Las mediciones de las escuelas no están en el repo porque no son públicas, y la simulación CFD tampoco (es de los tutores). El código las lee de una carpeta `mediciones/` con los CSV mensuales y del `.mat` que genera el script de MATLAB de la simulación.
