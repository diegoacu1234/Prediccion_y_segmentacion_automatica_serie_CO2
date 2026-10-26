# Predicción y segmentación automática de series de CO₂

Código del proyecto: *Análisis de la dinámica de ventilación en aulas escolares mediante modelos predictivos y segmentación automática de series temporales*

## Instalación

Con Python 3.12:

```
git clone https://github.com/diegoacu1234/Prediccion_y_segmentacion_automatica_serie_CO2
cd Prediccion_y_segmentacion_automatica_serie_CO2
pip install -e .
```

Chronos es opcional porque necesita PyTorch: `pip install -e ".[chronos]"`.

## Uso

```python
import co2ach

lecturas = co2ach.read_room("mediciones", "EscuelaA", 2)  # o cualquier tabla con columnas tiempo y co2

dia = co2ach.segment_day(lecturas, "2024-10-08", methods=("hmm", "pelt"))
dia.reported()            # los eventos con una ACH confiable
dia.plot(path="dia.png")

modelo = co2ach.Forecaster.train({"A02": lecturas})
modelo.predict(lecturas, room="A02")   # CO₂ a 1–60 min, intervalo del 80 % y probabilidad de pasar 1000 ppm
```

Si tenés el horario del salón y sabés cuántos chiquilines hay, `co2ach.analyze_room` lo usa y te da la ACH de cada día. Hay más ejemplos en [LIBRERIA.md](LIBRERIA.md) y el detalle de cada función en [docs/referencia.md](docs/referencia.md).

## Los métodos

| Clave | En la tesis | Nombre en `co2ach` |
|---|---|---|
| M1 | Hyun fiel (mediana, K-means y DBSCAN, sólo subidas) | `"hyun"` |
| M2 | Hyun + decaimientos | `"hyun_decaimientos"` |
| M3a | Hyun + C″ en K-means | `"hyun_c2"` |
| M3b | Hyun + quiebres | `"hyun_quiebres"` |
| M4 | Derivada con histéresis | `"derivada"` |
| M5 | PELT con costo exponencial | `"pelt"` |
| M6 | HMM de tres estados | `"hmm"` |
| M0 | Oráculo (sólo en las series sintéticas) | — |

Para mediciones reales usaría el HMM, que fue el que más se pareció a la segmentación hecha a mano (cap. 6). En series que cumplen bien el modelo, PELT.

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
  variants.py        variantes de tramo y generación del Monte Carlo
  experiments.py     el Monte Carlo
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
