# co2ach

Librería de Python para analizar series de CO₂ medidas en aulas:

- **Segmentación y ACH** (`co2ach.ach`). Encuentra en cada jornada los tramos de subida y de bajada, les ajusta la solución del balance de masa y estima la tasa de renovación de aire (ACH) de cada evento y de cada día. Si se conoce el horario del aula y quiénes la ocupan, lo usa.
- **Pronóstico** (`co2ach.pronostico`). Entrena un modelo que pronostica el CO₂ de 1 a 60 minutos hacia adelante, con un intervalo del 80 % y la probabilidad de superar 1000 ppm, y lo aplica a cualquier salón.

Es el código de los capítulos 4 a 8 de la tesis sobre segmentación de series de CO₂ y estimación de la ACH, con una interfaz para usarlo sobre otros datos. La especificación de cada función está en [`docs/referencia.md`](docs/referencia.md).

## Instalación

Requiere Python 3.10 o posterior. Abrir una terminal en la carpeta donde está el archivo `.whl` y correr:

```bash
pip install co2ach-0.2.0-py3-none-any.whl
```

O, si se descargó el código del repositorio, en esa carpeta:

```bash
pip install -e .
```

La `-e` (modo editable) hace que los cambios en el código se vean sin reinstalar.

pip instala también lo que la librería necesita: numpy, scipy, pandas, scikit-learn, ruptures, hmmlearn, joblib, lightgbm, matplotlib, pyarrow y h5py.

En Windows, si `pip install` falla porque no puede escribir en la carpeta de Python, agregar `--user`.

## Los datos

Todas las funciones reciben las lecturas de un salón como una tabla de pandas con una fila por lectura:

| tiempo | co2 | temp | hum |
|---|---|---|---|
| 2024-10-08 08:00:12 | 612 | 21.3 | 48 |
| 2024-10-08 08:00:44 | 615 | 21.3 | 48 |

`tiempo` y `co2` [ppm] son obligatorias; `temp` [°C] y `hum` [%] sólo las usa el pronóstico. Si las columnas tienen otros nombres, `co2ach.prepare_readings(tabla, time_col="fecha", co2_col="CO2")` las adapta. Para los CSV mensuales del proyecto está `co2ach.read_room(carpeta, ip, id)`.

## Segmentación y ACH

### Un día

```python
import co2ach

lecturas = co2ach.read_room("mediciones", "Escuelas32y83", 2)
dia = co2ach.segment_day(lecturas, "2024-10-08", methods=("hmm", "pelt"))

dia.reported()       # los eventos con una ACH confiable
dia.events           # todos los ajustes, con su calidad
dia.plot(path="dia.png")
```

Cada fila de `dia.events` es un tramo ajustado por un método: cuándo empieza y termina, si es subida o bajada, la ACH [1/h] con su error estándar, la asíntota, la calidad del ajuste y si se reporta. Un evento se reporta si la curva se parece a una exponencial (RMSE de hasta el 5 % de su recorrido) y el tramo dura al menos una constante de tiempo, que es lo que hace falta para que la ACH quede determinada.

![Un día segmentado con el HMM y PELT](docs/dia_ejemplo.png)

### Un salón entero, con lo que se sabe de él

```python
analisis = co2ach.analyze_room(
    lecturas,
    methods=("hmm", "pelt"),
    occupancy=co2ach.Occupancy.parse(["00:00-07:45", "10:00-10:30", "12:00-13:00", "15:00-15:30", "17:15-24:00"]),
    groups={
        "bajo": co2ach.Group(children=20, age=9, mass_kg=31.9, met=1.0, teacher_met=1.6),
        "medio": co2ach.Group(children=25, age=9, mass_kg=31.9, met=1.2, teacher_met=1.6),
        "alto": co2ach.Group(children=30, age=9, mass_kg=31.9, met=1.4, teacher_met=1.6),
    },
    volume_m3=168,
    n_jobs=4,
)
analisis.events   # los ajustes de todos los días, con los reajustes
analisis.daily    # la ACH de cada día, ponderada por la duración de los eventos
analisis.blocks   # la ACH de cada día y bloque del horario
analisis.plot_day("2024-10-08")
```

- Sin `occupancy` ni `groups`, la asíntota de cada curva queda libre: no hace falta saber nada del aula, pero en las subidas la ACH queda mal determinada (los métodos difieren mucho entre sí).
- Con el **horario** (`occupancy`), las bajadas con el aula vacía se reajustan con la asíntota en el CO₂ exterior (columna `ach_puro`).
- Con los **casos de generación** (`groups`) y el **volumen**, los tramos con la clase adentro se reajustan con la generación de cada caso (`ach_bajo`, `ach_medio`, `ach_alto`) y con la generación libre entre el caso menor y el mayor (`ach_acotada`).

`co2ach.THESIS_SCHEDULE` y `co2ach.THESIS_CLASS` son el horario y los tres casos que usa la tesis para el salón A02.

### Una serie cualquiera o un intervalo elegido a mano

```python
r = co2ach.segment(tabla["tiempo"], tabla["co2"], c_ext=420, methods="pelt")   # un ensayo, una simulación
f = co2ach.fit_event(lecturas, "2024-10-08 12:20", "2024-10-08 13:00", c_ext=394, model="vacio")
f["ach"], f["ach_se"]
```

`fit_event` admite cuatro modelos: `"libre"`, `"vacio"` (asíntota en el CO₂ exterior), `"conocida"` (con la fuente S = G/V) y `"acotada"` (la fuente entre dos valores).

### Qué método usar

| Nombre | Cuándo |
|---|---|
| `"hmm"` (por defecto) | Mediciones reales: es el que mejor reprodujo una segmentación hecha a mano y la ACH diaria |
| `"pelt"` | Series que siguen bien el modelo (laboratorio, simulaciones): reproduce la segmentación ideal |
| `"derivada"` | Una alternativa sencilla: en series sintéticas casi iguala a PELT; en mediciones fragmenta más que el HMM |
| `"hyun"`, `"hyun_decaimientos"`, `"hyun_c2"`, `"hyun_quiebres"` | El método de Hyun (2025) y sus variantes, para comparar |

Correr dos o tres métodos y mirar si coinciden es una buena forma de detectar los eventos dudosos.

## Pronóstico

```python
import co2ach

salones = {
    "A02": co2ach.read_room("mediciones", "Escuelas32y83", 2),
    "A06": co2ach.read_room("mediciones", "Escuelas32y83", 6),
}
modelo = co2ach.Forecaster.train(salones)        # LightGBM, un modelo por horizonte
modelo.metrics                                   # el error en los días de prueba
modelo.save("modelos/escuela_a")

modelo = co2ach.Forecaster.load("modelos/escuela_a")
p = modelo.predict(lecturas_hasta_ahora, room="A02")   # desde el último minuto con dato
```

`p` tiene una fila por horizonte (1, 5, 10, 15, 20, 25, 30, 45 y 60 min) con el CO₂ pronosticado, el intervalo del 80 %, la probabilidad de que el CO₂ esté por encima de 1000 ppm y la de que cruce ese umbral antes de ese momento.

- **Partición.** Por defecto, cada mes se usa así: días 1 a 15 para entrenar, 16 a 19 para detener el entrenamiento, 20 a 23 para calibrar el intervalo y del 24 a fin de mes para medir el error (`modelo.metrics`). Con `split=` se puede usar otra.
- **Variables.** El modelo usa el CO₂ reciente, la hora y el día, la temperatura y la humedad, la historia del salón en los días anteriores y el salón. Sin datos de temperatura y humedad: `variables=("co2", "calendario", "historia", "salon")`.
- **Salones nuevos.** Un modelo entrenado con varios salones se puede aplicar a uno que no vio (sin `room=`); conviene pasarle al menos la última semana de lecturas.
- **Referencias.** `co2ach.persistence_forecast` y `co2ach.mass_balance_forecast` no necesitan entrenamiento y sirven para comparar.

En la tesis, con 32 salones, el error absoluto medio fue de 106 ppm a 15 min y de 298 ppm a 60 min.

## Buenas prácticas y límites

- **El CO₂ exterior** se estima de la propia serie (el mínimo semanal del percentil 1 diario). Para que funcione, las lecturas tienen que incluir noches o fines de semana con el aula vacía.
- **Sin información sobre la ocupación, las subidas no determinan bien la ACH.** Usar sólo los eventos reportados y, si se puede, el horario y la generación de la clase.
- **La generación domina la incertidumbre.** Entre 20 y 30 escolares la ACH con la clase adentro varía en un factor de tres; registrar la asistencia la reduce.
- **Un solo sensor mide su entorno.** Con la ventilación abierta, la ACH depende de dónde esté el sensor respecto de la entrada de aire.
- **Los parámetros de los métodos** son los de la tesis, pensados para lecturas cada 30 s y eventos de decenas de minutos.

## Pruebas

```bash
python -m unittest tests.test_api -v      # la interfaz pública, unos 5 minutos
python -m unittest discover -s tests      # todo el código de la tesis
```

## Versiones

| Versión | Cambios |
|---|---|
| 0.2.0 | Primera versión para el grupo: interfaz pública `co2ach.ach` y `co2ach.pronostico`, régimen completo, generación acotada, ACH diaria ponderada por duración |
