# Referencia de `co2ach`

Versión 0.2.0. La interfaz pública está en dos módulos:

- [`co2ach.ach`](#segmentación-y-ach-co2achach): segmentación de series de CO₂ y estimación de la tasa de renovación de aire (ACH).
- [`co2ach.pronostico`](#pronóstico-co2achpronostico): pronóstico del CO₂ a 1–60 minutos y de la probabilidad de superar un umbral.

Todo lo que se lista acá también se puede importar directamente desde `co2ach` (por ejemplo, `co2ach.segment_day`). La guía de instalación y uso está en [`LIBRERIA.md`](../LIBRERIA.md).

## Convenciones

| | |
|---|---|
| **Lecturas** | Una tabla de pandas con una fila por lectura: `tiempo` (fecha y hora) y `co2` [ppm]. Opcionales: `temp` [°C] y `hum` [%], que usa el pronóstico. Puede venir desordenada y con huecos; ver [`prepare_readings`](#prepare_readings). |
| **Unidades** | CO₂ en ppm; ACH en 1/h (renovaciones por hora); duraciones en minutos (`*_min`) u horas (`*_h`); la fuente de CO₂ por unidad de volumen, S = G/V, en ppm/h (`*_ppm_h`) o ppm/s (`*_ppm_s`). |
| **CO₂ exterior** (`c_ext`) | El nivel al que tiende el CO₂ con el salón vacío. Si no se da, se estima de la serie: el mínimo, en la semana centrada en el día, del percentil 1 diario. |
| **Fases** | `"subida"` o `"bajada"`. |
| **Régimen completo** | Todos los métodos ajustan cada tramo desde el inicio de un transitorio hasta el inicio del siguiente, con la meseta incluida (Capítulo 4 de la tesis, «El tramo que se ajusta»). |
| **Hora local** | Los horarios (`Occupancy`, horas de los orígenes) se interpretan en la hora de las lecturas. |

---

## Segmentación y ACH (`co2ach.ach`)

### Métodos

Los métodos se eligen por nombre, en el argumento `methods` (un nombre, una tupla de nombres o `"todos"`).

| Nombre | Método de la tesis | Qué hace |
|---|---|---|
| `"hyun"` | Hyun fiel | Mediana móvil, K-means y DBSCAN; sólo ajusta subidas, con C₀ fijo |
| `"hyun_decaimientos"` | Hyun + decaimientos | Igual, ajustando también las bajadas |
| `"hyun_c2"` | Hyun + C″ en K-means | Agrega la derivada segunda a K-means |
| `"hyun_quiebres"` | Hyun + quiebres | Corta los eventos en los quiebres de pendiente |
| `"derivada"` | Derivada | Umbral con histéresis sobre la derivada suavizada |
| `"pelt"` | PELT | Detección de cambios con costo exponencial |
| `"hmm"` | HMM | Modelo oculto de Markov de tres estados sobre la derivada |

Por defecto se usa `("hmm",)`, el que mejor reprodujo la referencia manual y la ACH diaria en las mediciones (Capítulo 6). En series que cumplen bien el modelo, `"pelt"` reproduce la segmentación ideal (Capítulo 5).

### Tabla de eventos

[`segment`](#segment), [`segment_day`](#segment_day) y [`analyze_room`](#analyze_room) devuelven una tabla `events` con una fila por ajuste de cada método:

| Columna | Tipo | Significado |
|---|---|---|
| `metodo` | str | Nombre del método |
| `fecha` | str | Día (AAAA-MM-DD) o nombre de la serie |
| `inicio`, `fin` | datetime | Comienzo y fin del tramo ajustado (NaT si la serie no tenía fechas) |
| `t0_h`, `t1_h` | float | Lo mismo, en horas desde el comienzo de la serie (o desde las 0 h del día) |
| `dur_min` | float | Duración del tramo [min] |
| `fase` | str | `"subida"` o `"bajada"` |
| `aceptado` | bool | Pasó los filtros de aceptación de Hyun |
| `motivo` | str | Si no se aceptó, por qué: `"el ajuste falló"`, `"λ en la cota"`, `"duración"`, `"R²"`, `"C ≤ C_ext"` |
| `ach` | float | ACH estimada [1/h], con la asíntota libre |
| `ach_se` | float | Su error estándar [1/h] |
| `c0`, `c_inf` | float | CO₂ al comienzo del tramo y asíntota de la curva ajustada [ppm] |
| `c_ext` | float | CO₂ exterior supuesto [ppm] |
| `generacion_ppm_s` | float | Generación por unidad de volumen que implica el ajuste, ACH·(C∞ − C_ext) [ppm/s] |
| `excursion_ppm` | float | Recorrido de la curva ajustada dentro del tramo [ppm] |
| `r2`, `rmse` | float | Calidad del ajuste (`rmse` en ppm) |
| `rmse_rel` | float | RMSE sobre el recorrido |
| `lam_t` | float | Duración en constantes de tiempo, ACH × duración |
| `se_rel` | float | Error estándar relativo de la ACH |
| `evento` | bool | Aceptado y con un recorrido de al menos `Selection.min_excursion_ppm` |
| `reportado` | bool | Evento que cumple el criterio de reporte: `rmse_rel ≤ 0,05` y `lam_t ≥ 1` |

Para usar la ACH, filtrar por `reportado` (o usar `.reported()`). Los ajustes que no se reportan quedan en la tabla para poder revisarlos.

### `prepare_readings`

```python
prepare_readings(readings, time_col="tiempo", co2_col="co2") -> pd.DataFrame
```

Lleva las lecturas al formato de la librería. La usan internamente todas las funciones que reciben lecturas.

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `readings` | DataFrame o Series | — | Tabla con una columna de tiempo y una de CO₂, o serie con índice de tiempo |
| `time_col`, `co2_col` | str | `"tiempo"`, `"co2"` | Nombres de las columnas, si son otros |

**Devuelve** una tabla con `tiempo` (datetime), `co2` (float) y, si estaban, `temp` y `hum`, ordenada por tiempo, sin tiempos repetidos ni lecturas sin CO₂.

**Errores:** `ValueError` si falta la columna de tiempo o la de CO₂.

### `read_room`

```python
read_room(folder, ip, sensor_id, cache_dir=None) -> pd.DataFrame
```

Lee las lecturas de un salón desde la carpeta de CSV mensuales del proyecto (`mediciones-co2-MM-AAAA.csv`, con columnas `tiempo`, `id`, `co2`, `temp`, `hum`, `ip`).

| Parámetro | Tipo | Descripción |
|---|---|---|
| `folder` | ruta | Carpeta con los CSV |
| `ip` | str | Red de la escuela, p. ej. `"Escuelas32y83"` |
| `sensor_id` | int | Número de sensor dentro de esa red |
| `cache_dir` | ruta o None | Si se da, guarda una copia en Parquet y la reutiliza (requiere `pyarrow`) |

**Devuelve** `tiempo`, `co2`, `temp`, `hum`. **Errores:** `ValueError` si no hay lecturas de ese salón.

### `daily_table`

```python
daily_table(readings, max_gap_min=10.0, min_range_ppm=150.0) -> pd.DataFrame
```

Una fila por día (índice: fecha) con `lecturas`, `p1` y `p99` (percentiles 1 y 99 del CO₂), `maximo`, `hueco_max_min` (mayor hueco entre lecturas), `completo` (sin huecos de más de `max_gap_min`), `c_ext` (CO₂ exterior supuesto), `rango` (`p99 − p1`), `ocupado` (rango de al menos `min_range_ppm`) y `dia_semana` (0 = lunes).

### `segment`

```python
segment(times, co2, c_ext=None, methods=("hmm",), noise_floor="efectivo", dt_s=30.0,
        selection=Selection(), name="serie") -> SegmentationResult
```

Segmenta una serie cualquiera (una tarde, un ensayo, una simulación) y ajusta la exponencial del balance de masa en cada tramo.

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `times` | fechas o números | — | Fechas y horas, o tiempos en **horas** desde el comienzo |
| `co2` | array | — | Lecturas [ppm], del mismo largo |
| `c_ext` | float o None | None | CO₂ exterior [ppm]. Sin él se usa el percentil 1 de la serie, que sólo es razonable si la serie incluye ratos con el salón vacío y ventilado |
| `methods` | str o tupla | `("hmm",)` | Ver [Métodos](#métodos) |
| `noise_floor` | `"efectivo"`, float o None | `"efectivo"` | Piso del ruido con que los métodos escalan sus umbrales. `"efectivo"`: la fluctuación de la señal a 5 min mientras hay gente (Capítulo 6, «Sobresegmentación y ruido efectivo»); un número en ppm; o None para usar sólo el ruido del sensor (fragmenta mucho en mediciones reales) |
| `dt_s` | float | 30 | Paso de la grilla regular a la que se interpolan las lecturas [s] |
| `selection` | `Selection` | — | Criterio de evento y de reporte |
| `name` | str | `"serie"` | Nombre de la serie (columna `fecha`) |

**Devuelve** un [`SegmentationResult`](#segmentationresult). **Errores:** `ValueError` si `times` y `co2` no tienen el mismo largo, si hay menos de 10 lecturas o si un método no existe.

Los huecos se interpolan linealmente: conviene cortar la serie en los huecos largos antes de segmentarla.

### `segment_day`

```python
segment_day(readings, day, c_ext=None, methods=("hmm",), noise_floor="efectivo",
            selection=Selection()) -> SegmentationResult
```

Segmenta un día de 0 a 24 h, en una grilla de 30 s, como en el Capítulo 6.

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `readings` | DataFrame | — | Lecturas del salón; conviene que incluyan los tres días anteriores y posteriores, para estimar `c_ext` |
| `day` | fecha | — | El día, p. ej. `"2024-10-08"` |
| `c_ext` | float o None | None | CO₂ exterior [ppm]; sin él, el de [`daily_table`](#daily_table) |
| demás | | | Como en [`segment`](#segment) |

**Devuelve** un [`SegmentationResult`](#segmentationresult). **Errores:** `ValueError` si no hay lecturas de ese día.

### `SegmentationResult`

| Atributo o método | Descripción |
|---|---|
| `events` | [Tabla de eventos](#tabla-de-eventos) |
| `segments` | Una fila por tramo que detecta cada método, se ajuste o no: `metodo`, `fecha`, `inicio`, `fin`, `t0_h`, `t1_h`, `fase` |
| `series` | La serie en la grilla regular: `series.t_h` [h], `series.c_obs` [ppm], `series.c_ext_nominal` |
| `start` | Instante que corresponde a `t_h = 0` (None si la serie no tenía fechas) |
| `noise_sensor` | Ruido que se estima de la serie [ppm] |
| `noise_floor` | Piso de ruido usado [ppm] |
| `reported()` | Los eventos que se reportan |
| `plot(path=None, hours=(7, 21))` | La figura de la serie: ver [`plot_day`](#plot_day) |

### `analyze_room`

```python
analyze_room(readings, methods=("hmm",), days=None, occupancy=None, groups=None,
             volume_m3=None, noise_floor="efectivo", selection=Selection(), n_jobs=1) -> RoomAnalysis
```

Segmenta todos los días de un salón y estima la ACH de cada evento y de cada día. Lo que se sabe del salón cambia el modelo que se ajusta (Capítulo 4 de la tesis, «Aceptación y criterio de reporte»).

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `readings` | DataFrame | — | Lecturas del salón |
| `methods` | str o tupla | `("hmm",)` | Ver [Métodos](#métodos) |
| `days` | lista de fechas o None | None | Días a analizar. Sin ella, los días con actividad (`ocupado`) y sin huecos de más de 10 min (`completo`) |
| `occupancy` | [`Occupancy`](#occupancy) o None | None | Horario en que el salón está vacío, de lunes a viernes. Las bajadas con al menos el 80 % de su duración en ese horario se reajustan con la asíntota en `c_ext` |
| `groups` | dict o None | None | Casos de generación de la clase, {nombre: [`Group`](#group)}, p. ej. bajo, medio y alto. Los tramos con a lo sumo el 20 % de su duración en horario vacío se reajustan con la generación de cada caso y con la generación libre entre el menor y el mayor. Requiere `occupancy` y `volume_m3`. Con tres casos, el del medio es el central |
| `volume_m3` | float o None | None | Volumen del salón [m³] |
| `noise_floor` | | `"efectivo"` | Como en [`segment`](#segment) |
| `selection` | `Selection` | — | Criterio de evento y de reporte |
| `n_jobs` | int | 1 | Procesos en paralelo (un día por tarea; −1 usa todos los núcleos) |

**Devuelve** un [`RoomAnalysis`](#roomanalysis). **Errores:** `ValueError` si se piden días sin lecturas, o casos de generación sin horario o sin volumen.

Un año de un salón (unos 160 días con clase) tarda del orden de 5 a 10 s por día y por método con un proceso; `n_jobs` reparte los días entre núcleos.

### `RoomAnalysis`

| Atributo o método | Descripción |
|---|---|
| `events` | [Tabla de eventos](#tabla-de-eventos), con las columnas de los reajustes si se dio el horario o la generación (abajo) |
| `segments` | Los tramos de cada método y día |
| `days` | [`daily_table`](#daily_table) de los días analizados, con `ruido_sensor` y `piso_ruido` [ppm] |
| `daily` | La ACH de cada día, según [`daily_ach`](#daily_ach) |
| `blocks` | Con el horario y los casos de generación, la ACH de cada día y bloque del horario (abajo); si no, None |
| `cases` | Fuente de cada caso de generación [ppm/h], de menor a mayor |
| `occupancy` | El horario dado |
| `readings` | Las lecturas, preparadas |
| `plot_day(day, path=None)` | La figura de un día analizado: ver [`plot_day`](#plot_day) |

**Columnas de los reajustes en `events`:**

| Columnas | Cuándo | Significado |
|---|---|---|
| `vacio`, `ventana` | con `occupancy` | Fracción del tramo en horario vacío (días hábiles) y ventana del horario que más aporta |
| `ach_puro`, `ach_se_puro`, `c0_puro`, `c_inf_puro`, `rmse_puro`, `r2_puro`, `excursion_puro` | con `occupancy` | Bajadas en horario vacío, reajustadas con la asíntota en `c_ext` |
| `con_gente` | con `groups` | El tramo se reajustó con la generación de la clase |
| `ach_<caso>`, `ach_se_<caso>`, `c0_<caso>`, `c_inf_<caso>`, `rmse_<caso>`, `r2_<caso>`, `excursion_<caso>` | con `groups` | Reajuste con la generación de cada caso |
| `cota_<caso>` | con `groups` | `"baja"` si esa generación no alcanza para explicar la curva, `"alta"` si sobra; en los dos casos no hay ACH |
| `ach_acotada`, …, `fuente_acotada` | con `groups` | Reajuste con la generación libre entre el caso menor y el mayor; `fuente_acotada` [ppm/h] es la que resulta |
| `cota_acotada` | con `groups` | `"inferior"` o `"superior"` si la fuente quedó en esa cota, `"lambda"` si la ACH quedó en la cota del optimizador, vacío si quedó adentro |

**Columnas de `blocks`** (Capítulo 6, «La ventilación del aula»): `fecha`, `bloque` (p. ej. `"07:45–10:00"`), `ach` (promedio de los ajustes de todos los métodos, ponderado por duración), `ach_mediana`, `ach_min` y `ach_max` (entre métodos), `metodos`, `eventos`, `minutos`, `ach_<caso menor>` y `ach_<caso mayor>`, y `sin_solucion_pct`. En los bloques de clase entran las subidas reajustadas con el caso central; en los vacíos, las bajadas con la asíntota en `c_ext`; sólo los ajustes con RMSE de hasta el 5 % del recorrido.

### `daily_ach`

```python
daily_ach(events, only_reported=True, column="ach") -> pd.DataFrame
```

La ACH de cada día y fase como promedio de los eventos ponderado por su duración: un evento de una hora pesa cuatro veces más que uno de 15 min.

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `events` | DataFrame | — | Una [tabla de eventos](#tabla-de-eventos) |
| `only_reported` | bool | True | Sólo los eventos que se reportan |
| `column` | str | `"ach"` | Qué ACH se promedia: `"ach"` (asíntota libre) o la de un reajuste, p. ej. `"ach_medio"` o `"ach_puro"` |

**Devuelve** `fecha`, `fase`, `metodo`, `ach`, `eventos`, `minutos`: una fila por fecha, fase y método, y una más con `metodo = "todos"`, que pondera juntos los eventos de todos los métodos.

### `fit_event`

```python
fit_event(readings, start, end, c_ext, model="libre", source_ppm_h=None,
          bounds_ppm_h=None, dt_s=30.0) -> dict
```

Ajusta la exponencial del balance de masa en un intervalo elegido a mano, por ejemplo un evento que se marcó mirando la serie.

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `readings` | DataFrame | — | Lecturas del salón |
| `start`, `end` | fecha y hora | — | Límites del intervalo |
| `c_ext` | float | — | CO₂ exterior [ppm] |
| `model` | str | `"libre"` | `"libre"`: C₀, C∞ (≥ `c_ext`) y ACH libres. `"vacio"`: salón vacío, asíntota en `c_ext`. `"conocida"`: la fuente `source_ppm_h` ata la asíntota a la ACH, C∞ = C_ext + S/ACH. `"acotada"`: la fuente libre entre los valores de `bounds_ppm_h` |
| `source_ppm_h` | float | None | Fuente S = G/V [ppm/h], para `"conocida"`; ver [`class_source`](#class_source) |
| `bounds_ppm_h` | (float, float) | None | Fuentes mínima y máxima [ppm/h], para `"acotada"` |
| `dt_s` | float | 30 | Paso de la grilla [s] |

**Devuelve** un diccionario con `ach` y `ach_se` [1/h], `c0` y `c_inf` [ppm], `fuente_ppm_h`, `r2`, `rmse` [ppm], `rmse_rel`, `lam_t`, `dur_min` y `en_cota` (`"inferior"`/`"superior"` si la fuente acotada quedó en esa cota, `"lambda"` si la ACH quedó en la cota del optimizador). **Errores:** `ValueError` si el intervalo tiene menos de 5 lecturas o faltan los parámetros del modelo.

### `Occupancy`

```python
Occupancy.parse(["00:00-07:45", "10:00-10:30", "12:00-13:00", "15:00-15:30", "17:15-24:00"])
```

Horario en que el salón está vacío, de lunes a viernes, como intervalos `"HH:MM-HH:MM"` [desde, hasta). Los fines de semana no se consideran. `THESIS_SCHEDULE` es el del salón A02 de la tesis.

### `Group`

```python
Group(children, age, mass_kg, met, teacher_met, teacher_bmr=6.1, bmr=None)
```

Una clase: `children` escolares de `age` años y `mass_kg` kg a `met` met, y un docente a `teacher_met` met. La generación de CO₂ sale del método de Persily y de Jonge (2017): metabolismo basal (`bmr`, MJ/día; si no se da, el de Schofield para esa masa y edad), nivel de actividad y cociente respiratorio 0,85. `teacher_bmr` es el metabolismo basal del docente (6,1 MJ/día: mujer de 30 a 50 años). `Group.generation(temp_c=20)` devuelve la generación [L/h] de un escolar, del docente y del total; `Group.source(volume_m3)`, la fuente [ppm/h].

`THESIS_CLASS` tiene los tres casos del Capítulo 6: `"bajo"` (20 escolares a 1,0 met), `"medio"` (25 a 1,2) y `"alto"` (30 a 1,4), con un docente a 1,6 met y metabolismos de 5,14 y 8,24 MJ/día.

### `class_source`

```python
class_source(group, volume_m3, temp_c=20.0) -> float
```

Fuente S = G/V [ppm/h] de una clase en un salón de `volume_m3` m³, con la generación corregida a `temp_c` °C.

### `Selection`

```python
Selection(min_excursion_ppm=100.0, max_rmse_rel=0.05, min_lam_t=1.0)
```

Qué ajustes cuentan como evento (recorrido de al menos `min_excursion_ppm`) y cuáles se reportan (`rmse_rel ≤ max_rmse_rel` y `lam_t ≥ min_lam_t`). Los valores por defecto son los de la tesis (Capítulo 4, «Aceptación y criterio de reporte»).

### `plot_day`

```python
plot_day(result, day=None, path=None, hours=(7.0, 21.0)) -> matplotlib.figure.Figure
```

La figura de un día: la serie arriba, una fila por método con sus tramos y sus curvas (gruesa y con su ACH si se reporta), y la ACH de todos los métodos superpuesta. Con el horario y la generación, también los reajustes.

| Parámetro | Descripción |
|---|---|
| `result` | Un `SegmentationResult` o un `RoomAnalysis` |
| `day` | El día, si `result` es un `RoomAnalysis` |
| `path` | Si se da, guarda la figura (PNG o PDF según la extensión) |
| `hours` | Ventana horaria que se dibuja; en una serie que no es un día, toda la serie |

Usa matplotlib, que se instala con la librería.

### Constantes

| Nombre | Contenido |
|---|---|
| `METHODS` | {nombre: clave interna de la tesis} |
| `DEFAULT_METHODS` | `("hmm",)` |
| `THESIS_CLASS` | Los tres casos de generación del Capítulo 6 |
| `THESIS_SCHEDULE` | El horario del salón A02 |

---

## Pronóstico (`co2ach.pronostico`)

### Tabla de pronósticos

[`Forecaster.predict`](#forecasterpredict), [`persistence_forecast`](#persistence_forecast) y [`mass_balance_forecast`](#mass_balance_forecast) devuelven una fila por origen y horizonte:

| Columna | Tipo | Significado |
|---|---|---|
| `origen` | datetime | Instante desde el que se pronostica (minuto) |
| `horizonte_min` | int | Horizonte: 1, 5, 10, 15, 20, 25, 30, 45 o 60 min (`HORIZONS`) |
| `tiempo` | datetime | `origen + horizonte` |
| `co2_origen` | float | CO₂ en el origen [ppm] |
| `pronostico` | float | CO₂ pronosticado para `tiempo` [ppm] |
| `li_80`, `ls_80` | float | Intervalo del 80 % [ppm] (sólo `Forecaster`) |
| `prob_sobre_umbral` | float | P(CO₂ en `tiempo` ≥ umbral) (sólo `Forecaster`) |
| `prob_cruce` | float | P(el CO₂ cruce el umbral entre el origen y `tiempo`) (sólo `Forecaster`) |
| `medido` | float | CO₂ medido en `tiempo`, si está en las lecturas (para comparar) |

### `Forecaster.train`

```python
Forecaster.train(rooms, variables=("co2", "calendario", "clima", "historia", "salon"),
                 threshold=1000.0, rearm=None, quantiles=True, classifier=True, split=None,
                 train_hours=("07:00", "18:00"), eval_hours=("07:30", "17:30"), step_min=2,
                 threads=4, params=None, verbose=False) -> Forecaster
```

Entrena el modelo de los Capítulos 7 y 8: un LightGBM global por horizonte para el cambio C(t+h) − C(t), con los cuantiles 0,1 y 0,9 corregidos por calibración conformal (CQR) para el intervalo del 80 %, y un clasificador de la probabilidad de cruzar el umbral.

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `rooms` | dict | — | {nombre del salón: lecturas}. Un modelo global con varios salones rinde más que uno por salón (Capítulo 7) |
| `variables` | tupla | las cinco | Grupos de variables (`VARIABLES`): `"co2"` (obligatorio), `"calendario"`, `"clima"` (temperatura y humedad; quitarlo si no hay esos datos), `"historia"` (perfil de los días con clase anteriores), `"salon"` |
| `threshold` | float | 1000 | Umbral de CO₂ [ppm] |
| `rearm` | float o None | `threshold − 50` | Después de un cruce, el CO₂ tiene que bajar de acá para que cuente otro |
| `quantiles` | bool | True | Entrena los cuantiles para el intervalo; sin ellos, el intervalo sale de los residuos de calibración |
| `classifier` | bool | True | Entrena el clasificador del cruce; sin él, `prob_cruce` sale de `prob_sobre_umbral` |
| `split` | función o None | partición de la tesis | Recibe un `DatetimeIndex` y devuelve, para cada instante, `"train"`, `"es"` (detención temprana), `"cal"` (calibración) o `"test"` (prueba). Por defecto, por días de cada mes: 1–15, 16–19, 20–23 y del 24 a fin de mes |
| `train_hours` | (str, str) | 07:00–18:00 | Ventana de los orígenes de entrenamiento |
| `eval_hours` | (str, str) | 07:30–17:30 | Ventana de los orígenes de detención, calibración y prueba |
| `step_min` | int | 2 | Un origen de entrenamiento cada tantos minutos |
| `threads` | int | 4 | Hilos de LightGBM |
| `params` | dict o None | None | Parámetros de LightGBM que reemplazan a los de la tesis |
| `verbose` | bool | False | Muestra el avance |

Sólo entran los días hábiles con clase (el CO₂ recorre al menos 150 ppm). Hacen falta orígenes de entrenamiento y de detención temprana; sin días de calibración, se calibra con los de detención. **Devuelve** un `Forecaster` con `metrics` (el error en los días de prueba, si los hay). **Errores:** `ValueError` si faltan particiones o el grupo `"co2"`.

Con los 32 salones de la tesis tarda del orden de 20 a 30 minutos; con dos salones y tres meses, unos 2 minutos.

### `Forecaster.predict`

```python
modelo.predict(readings, room=None, at=None, threshold=None) -> pd.DataFrame
```

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `readings` | DataFrame | — | Lecturas del salón hasta el origen. Conviene incluir al menos la última semana: las variables de historia usan los días con clase anteriores, y sin ellas el pronóstico empeora |
| `room` | str o None | None | Nombre del salón, si fue uno de los de entrenamiento; si no, se lo trata como un salón nuevo |
| `at` | None, `"todos"` o fechas | None | Orígenes: el último minuto con dato, cada minuto con dato, o una lista de instantes (se redondean al minuto) |
| `threshold` | float o None | None | Otro umbral para `prob_sobre_umbral`; en ese caso `prob_cruce` sale de esa probabilidad y no del clasificador |

**Devuelve** la [tabla de pronósticos](#tabla-de-pronósticos). **Errores:** `ValueError` si algún origen pedido no tiene CO₂.

### `Forecaster.evaluate`

```python
modelo.evaluate(rooms, split=None, which="test") -> pd.DataFrame
```

Error del pronóstico en los días de `rooms` ({nombre: lecturas}) que caen en la partición `which`. **Devuelve** una fila por horizonte: `horizonte`, `mae`, `rmse` y `sesgo` [ppm], `n`, `cobertura_80` (fracción de los valores medidos dentro del intervalo) y `ancho_80` [ppm].

### `Forecaster.save` y `Forecaster.load`

```python
modelo.save("modelos/escuela_a")
modelo = Forecaster.load("modelos/escuela_a")
```

Guarda en una carpeta un archivo de texto por modelo de LightGBM (`mid_h30.txt`, `q10_h30.txt`, `clasif_h30.txt`, …), los residuos de calibración (`residuos.npy`), la configuración (`modelo.json`) y el error de prueba (`error_en_prueba.csv`). Un modelo guardado se puede compartir y cargar sin volver a entrenar.

**Atributos de `Forecaster`:** `rooms` (salones de entrenamiento → código), `variables`, `cols` (las variables), `threshold`, `rearm`, `metrics`, `info` (versión, orígenes por partición y ventanas).

### `persistence_forecast`

```python
persistence_forecast(readings, at=None) -> pd.DataFrame
```

Referencia: el CO₂ de cada horizonte es el del origen. `at` como en `predict`.

### `mass_balance_forecast`

```python
mass_balance_forecast(readings, at=None, window=15) -> pd.DataFrame
```

Referencia física, sin entrenamiento: ajusta la recta ΔC = a − β·C sobre los últimos `window` minutos y la prolonga como la solución del balance de masa (Capítulo 7), con la ACH acotada a 20 1/h y el pronóstico entre el CO₂ exterior y 5000 ppm. Agrega `ach_estimada` [1/h]. Sirve a horizontes cortos; a 60 min su error supera al de la persistencia.

### `minute_grid`

```python
minute_grid(readings) -> pd.DataFrame
```

Las lecturas en una grilla de 1 minuto (índice: tiempo; columnas `co2`, `temp`, `hum`): la media de cada minuto, sin los valores fuera del rango físico (CO₂ de 350 a 5000 ppm), con los huecos de hasta 5 min interpolados.

### `threshold_crossings`

```python
threshold_crossings(readings, threshold=1000.0, rearm=None, persist=2) -> pd.DataFrame
```

Los cruces del umbral hacia arriba (`tiempo`, `co2`): el primer minuto de una racha de al menos `persist` minutos en o por encima de `threshold`. Después de un cruce no cuenta otro hasta que el CO₂ baje de `rearm` (por defecto, `threshold − 50`).

### Constantes

| Nombre | Contenido |
|---|---|
| `HORIZONS` | `(1, 5, 10, 15, 20, 25, 30, 45, 60)` minutos |
| `VARIABLES` | {grupo: columnas de variables} |
| `DEFAULT_VARIABLES` | Los cinco grupos |
