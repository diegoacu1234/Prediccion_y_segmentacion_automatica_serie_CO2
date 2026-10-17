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

## Escenarios

- **A, día de aula**: eventos bien separados, con y sin meseta.
- **B, quiebres dentro de una misma fase**: sube y vuelve a subir más rápido, baja y sigue bajando más lento.
- **C, difícil**: ventilación baja, eventos de 20 a 45 minutos, más ruido, cambios graduales de ocupación y deriva del CO₂ exterior.

Están definidos en `co2ach/synth.py`; ahí se cambian el ACH, la ocupación, el volumen y el sensor.

## Experimentos

Tres preguntas, respondidas con un Monte Carlo sobre los tres escenarios (6 niveles de ruido × 30 realizaciones):

1. **Meseta como asíntota.** ¿Qué pasa si la meseta se ajusta junto con el transitorio que la precede, en vez de tratarla como una fase aparte?
2. **Ruido.** ¿Cómo se degrada cada método al subir el ruido del sensor, de 1 a 40 ppm?
3. **Generación conocida.** ¿Qué se gana si se conoce G (queda libre sólo el ACH) frente a dejar libres C∞ y el ACH? ¿Y con qué error en G deja de convenir?

```
py -3.12 scripts/run_experimentos.py                  # Monte Carlo, ~45 min con 6 procesos; se puede reanudar
py -3.12 scripts/plot_experimentos.py                 # figuras y tablas, ~2 min
py -3.12 scripts/plot_experimentos.py --idioma es en  # además las mismas figuras en inglés (figs_en/)
py -3.12 scripts/diagnostico_sesgo_hyun.py            # por qué Hyun subestima el ACH, ~3 min
```

Todo queda en `outputs/experimentos`: las tablas del Monte Carlo (`mc_*.csv.gz`), las figuras en PNG a 300 ppp y PDF (`figs/`, con `pies_de_figura.md`) y las tablas resumen en CSV y LaTeX (`tablas/`). `notebooks/experimentos.ipynb` corre las variantes sobre una serie y muestra las figuras principales.

Cada método se corre en cuatro variantes:

| | Generación libre | Generación conocida |
|---|---|---|
| **Sólo transitorio (T)** | lo que detecta el método | C∞ = C_ext + S/λ, queda libre λ |
| **Transitorio + meseta (T+M)** | el tramo va de un inicio de transitorio al siguiente | ídem, con C∞ atada a λ |

Desde el 2026-10-16, a pedido de los tutores, todo el estudio (comparación base, CFD, mediciones y referencia manual) ajusta T+M por defecto (`pipeline.run_all(..., tramo="TM")`). "Sólo transitorio" queda como contraste; las figuras del capítulo 5 que no comparan tramos usan T+M (`plots_exp.BASE_TRAMO`). Las salidas anteriores, con sólo el transitorio, están en `outputs/solo_transitorio/`.

- **Cómo se arman los tramos T+M** (`co2ach/regimes.py`): los fragmentos del mismo signo se unen si un solo exponencial los explica (misma penalización que PELT), y un inicio cuyo régimen casi no se mueve (recorrido menor que 4σ̂) se descarta. PELT y el oráculo ya segmentan por régimen; ahí "sólo transitorio" recorta el tramo al 95 % del escalón.
- **Generación conocida**: se usa la fuente nominal del régimen dominante del tramo. En la familia Hyun C0 sigue fijo, así que queda libre sólo λ; en el resto quedan libres C0 y λ.
- **Error en G**: con el ruido en 5 y 20 ppm se repite el ajuste suponiendo G con un error de −40 a +40 %.
- **Éxito ±20 %**: fracción de los regímenes verdaderos con transitorio para los que el método da un ACH aceptado a menos de 20 % del real. Un régimen sin estimación cuenta como fallo.
- **Estimaciones espurias**: ajustes aceptados cuyo tramo cae en un régimen sin evento o mezcla dos regímenes.

Los presets de Hyun se usan tal como están publicados, sin reajustarlos a cada nivel de ruido; las ventanas de derivada y HMM y la penalización de PELT se escalan con el ruido estimado.

El diagnóstico repite el ajuste de Hyun sobre sus propios tramos cambiando sólo dónde empiezan. Las subidas arrancan unos minutos antes del quiebre verdadero: el signo del gradiente y el RHL no tienen escala, así que alcanza con que la mediana móvil de 25 muestras empiece a moverse. Con C0 fijo en esa primera muestra, el ACH sale subestimado. Con `--mediana 5` el adelanto desaparece y el sesgo baja. Las tablas quedan en `tablas/sesgo_hyun_inicio_del_tramo*.csv`.

**Generación acotada** (`scripts/run_acotada.py`, ~20 min con 4 procesos; `--solo-figuras` rehace figuras y tablas). S queda libre entre dos múltiplos de la fuente nominal del régimen (`fitting.fit_bounded_source`, `variants.fit_variant(..., gen="acotada", g_bounds=...)`). Si la solución libre cae dentro de las cotas, el resultado es el de la generación libre; si no, el de la generación conocida en la cota. Se prueban siete intervalos (el valor real adentro, en una cota o afuera) contra la generación libre y la conocida con distintos errores, con 5 y 20 ppm, sobre los regímenes con gente. Deja `outputs/experimentos/acotada/` (tablas `resumen.csv`, `donde_queda_s.csv`, `cotas_segun_duracion.csv`) y las figuras E4_1 y E4_2.

El ruido se estima de la propia serie (`signal_utils.noise_sd`) con el valor cuadrático medio recortado de las segundas diferencias. No se usa la MAD porque con un sensor que reporta ppm enteros sólo toma valores discretos: da 1,21 ppm con σ = 1 y 1,82 con σ = 2, y esos errores de +16 % y −10 % se trasladan a todos los umbrales.

## Simulación CFD

Los mismos métodos sobre los sensores virtuales de una simulación (`cfd_simulation/*.mat`, el archivo que escribe `BuildTimeSeries_sensores.m`): cada sensor por separado y los promedios de todos, por altura y por pared.

```
py -3.12 scripts/run_cfd.py                               # ~4 min
py -3.12 scripts/run_cfd.py --paneles-sensores            # además un panel de detalle por sensor, ~5 min más
py -3.12 scripts/run_cfd.py --personas N --volumen V     # con la ocupación y el volumen de la simulación
py -3.12 scripts/run_cfd.py --idioma es en                # figuras también en inglés
```

Deja figuras y tablas en `outputs/cfd`. Lo que se sabe de cada simulación está en `CASES`, en el script. Para `s1r32a`: ventilación abierta 5 min cada 30, ACH 6 con la ventilación abierta y 1,5 cerrada. Lo anterior a la primera muestra del archivo es calibración de la simulación y no se usa. La planta es de 8 × 6 m; la altura y la cantidad de personas no están en la carpeta.

Lo que cambia respecto del estudio sintético:

- **No hay verdad muestra a muestra.** La referencia para la segmentación es el esquema de ventilación (`--periodo`, `--abierto`). Los inicios se cuentan con ±2 min y con ±4 min, porque los sensores responden con retardo. El ACH estimado se compara con el de cada estado de la simulación (`--ach-abierto`, `--ach-cerrado`).
- **Minutos de mezcla** (`--mezcla`, 4 por defecto). Después de cada cierre los sensores siguen bajando unos minutos aunque la ventilación ya cambió. En ese lapso el modelo de una zona no vale, y los cortes del esquema lo saltean al ajustar, como hace el oráculo sintético con las rampas de ocupación.
- **Dos juegos de escalas de tiempo.** "Estudio" usa las duraciones del estudio sintético pasadas a la cadencia de la serie (los presets de Hyun, que están en muestras de 30 s, se convierten a minutos). "Corta" está pensado para eventos de 5 min con una muestra por minuto: ventanas de 5 muestras, mediana de 3 y duración mínima de 4 min, sin escalado por ruido.
- **La generación.** Sale, en este orden, de `--fuente`, de `--personas` y `--volumen`, o de calibrarla para que el promedio de la sala dé el ACH cerrado de la simulación. En ese último caso el ACH cerrado del promedio acierta por construcción, y la prueba independiente es el ACH abierto y la diferencia entre los dos (ΔACH), que casi no depende de la fuente. Si no se conoce nada, se usa la fuente que mejor ajusta la propia serie, que la serie fija mal.
- **Comparación con la simulación.** `tablas/comparacion_con_la_simulacion.csv` y la figura C9 muestran el ACH de cada estado según lo que se suponga de la generación (libre, la que mejor ajusta la serie, calibrada), para el promedio de la sala, por altura y por pared.
- **Qué ajuste cuenta para cada estado** (`cfd.match_fits`). A cada tramo del esquema le corresponde un ajuste: el aceptado de su misma fase (bajada para abierto, subida para cerrado) que más se solapa con él. Los demás ajustes aceptados son estimaciones de más y no cuentan. No se pide que el ajuste caiga mayormente dentro del tramo, como en el estudio sintético: los sensores responden 2 a 4 min tarde y la bajada de una apertura de 5 min puede quedar con más de la mitad fuera de ella.
- **Paneles de detalle** (`figs/paneles/`). Una figura por serie y por supuesto sobre la generación (`todos_G_conocida`, `todos_G_libre`, `z08_…`, `z20_…`, `pared_este_…`; con `--paneles-sensores`, también `sensores/sensor_08_…`). Cada fila es un método: a la izquierda, la serie, los tramos que detecta y sus ajustes; a la derecha, el ACH de la simulación y el que el método estima en cada tramo, con su error (barra oscura: el ajuste que cuenta para ese estado; barra clara: estimación de más). La tabla `tablas/ajustes.csv` tiene lo mismo, una fila por ajuste. La figura C10 resume el ACH que da cada método en cada sensor frente al de la simulación, con la generación conocida, acotada y libre.
- **Generación acotada** (`--cotas-fuente`, por defecto 0,7 y 1,4): S libre entre esos múltiplos de la fuente supuesta, el mismo ancho que va del caso bajo al alto en el salón A02.
- **Sensor por sensor** (figuras C11 a C13, columnas nuevas de `tablas/sensores.csv`). C11: ACH de cada estado en cada sensor, en planta y por pared, con los cortes del esquema y la generación conocida. C12: por qué unos sensores estiman mejor el ACH abierto; `dilucion_rel` (`cfd.opening_dilution`) es la fracción del exceso sobre el exterior que el sensor pierde en los 3 primeros minutos de cada apertura, dividida por la que pierde la media de la sala. C13: un panel por sensor, por altura.

## Mediciones reales

Los mismos métodos sobre las mediciones de un salón (`mediciones/*.csv`; un salón es una pareja `ip`, `id`).

```
py -3.12 scripts/run_mediciones.py                         # salón 2 de Escuelas32y83, ~40 min con 6 procesos
py -3.12 scripts/run_mediciones.py --ip EscuelaA --id 7
py -3.12 scripts/run_mediciones.py --dias 2024-08-13       # qué días se dibujan en detalle
py -3.12 scripts/run_mediciones.py --solo-figuras          # rehace las figuras sin recalcular
py -3.12 scripts/run_mediciones.py --meses 9 10 --solo-figuras          # una figura compacta por día de esos meses
py -3.12 scripts/run_mediciones.py --rmse-rel 0.03 --lam-t 2 --solo-figuras   # otra regla para reportar
py -3.12 scripts/run_mediciones.py --vacio 0:00-8:00 12:00-13:00 17:00-24:00 --solo-figuras   # otro horario de ocupación
py -3.12 scripts/run_mediciones.py --ninos 24 26 28 --solo-figuras      # otra cantidad de niños en los casos bajo, medio y alto
```

Deja figuras y tablas en `outputs/mediciones/<ip>_id<NN>/`. Cómo se trabaja:

- **Un día por vez**, de 0 a 24 h, interpolado a una grilla de 30 s. Sólo los días con clase (el CO₂ recorre al menos 150 ppm) y sin huecos de más de 10 minutos.
- **CO₂ exterior**: el mínimo, en la semana centrada en el día, del percentil 1 diario. Sale de la propia serie y absorbe la deriva del sensor.
- **Sin verdad.** No se conocen ni los cortes ni el ACH ni la ocupación, así que la generación queda libre. Las figuras muestran cómo segmenta cada método y qué ACH ajusta; los resúmenes, cuánto coinciden los métodos entre sí.
- **Dos corridas.** "Tal cual" usa el ruido que cada método estima, que es el del sensor (menos de 1 ppm). "Ruido efectivo" le pone como piso la fluctuación real de la señal mientras hay gente, medida en la escala de 5 minutos (`mediciones.effective_noise`). Sin ese piso, derivada, PELT y Hyun + quiebres toman cada oscilación por un evento.
- **Evento**: ajuste aceptado cuya curva recorre al menos 100 ppm.
- **Qué se reporta**: un evento se reporta si el RMSE del ajuste no pasa de 5 % del recorrido de la curva (el ajuste es bueno) y el tramo dura al menos una constante de tiempo, ACH × duración ≥ 1 (el tramo alcanza para fijar el ACH). La figura R6 y `tablas/umbrales.csv` muestran cómo cambian la cantidad de eventos y la coincidencia entre métodos con otros umbrales y con otros criterios (R², error estándar relativo).
- **Figuras por día** (`figs/dias/`, octubre por defecto): la serie arriba, una fila por método con sus tramos, sus curvas ajustadas sobre el CO₂ medido y su ACH, y abajo todos los métodos superpuestos para ver si coinciden.
- **Ocupación conocida.** Si se sabe a qué horas el salón está vacío (`HORARIOS` en el script, o `--vacio`), las bajadas que caen ahí (al menos 80 % de su duración, de lunes a viernes) se reajustan además con la generación en cero, es decir, con la asíntota fija en el CO₂ exterior (`mediciones.refit_empty`). La figura R7 muestra el horario contra los datos y cuánto cambian el ACH y la coincidencia entre métodos; los números están en `tablas/ocupacion_por_ventana.csv`, `ocupacion_por_metodo.csv` y `horario_contra_datos.csv`.
- **Generación conocida, con tres casos.** Si además se cargan el volumen del salón y quiénes lo ocupan (`VOLUMENES` y `CLASES` en el script; `--volumen`, `--ninos`), se arma un caso bajo, uno medio y uno alto de generación de CO₂ con el método de Persily y de Jonge (2017): metabolismo basal por masa corporal y edad (Schofield), nivel de actividad en met y cociente respiratorio 0,85 (`physics.co2_generation`, `mediciones.Group`). Los tramos con la clase adentro (a lo sumo 20 % de su duración en horario vacío) se reajustan con cada caso, lo que ata la asíntota al ACH: C∞ = C_ext + S/λ (`mediciones.refit_occupied`). La figura R8 y `tablas/generacion_*.csv` muestran entre qué valores queda el ACH, cuánto coinciden los métodos y qué generación implican los ajustes libres. Supone que la clase completa está en el salón durante todo el horario con gente.
- **Generación acotada** (`scripts/acotada_mediciones.py`, segundos, después de `run_mediciones.py`). Los mismos tramos con gente se reajustan además con la generación libre entre el caso bajo y el alto (`mediciones.refit_bounded`, columnas `*_acotada` de `eventos.csv.gz`). El script compara en las subidas con gente la generación libre, la acotada y la del caso medio: ACH por método, dónde queda la generación según la duración del tramo y diferencia entre métodos (figura R9, `tablas/acotada_*.csv`).
- **El régimen completo.** Desde el 2026-10-16 cada tramo va de un inicio de transitorio al siguiente (T+M), y la referencia manual también (`manual.regime_ends`). `scripts/cobertura_de_las_curvas.py` mide qué parte de cada evento marcado cubre la curva que da su ACH, con esta corrida y con la anterior (`outputs/solo_transitorio/`): figura M9 y `manual/cobertura_de_las_curvas.csv`.

## Referencia manual

Para tener contra qué comparar en las mediciones, donde no hay verdad, los eventos se pueden marcar a mano y usar como referencia.

```
py -3.12 scripts/etiquetar_eventos.py            # abre octubre de 2024 del salón 2, un día por vez
py -3.12 scripts/etiquetar_eventos.py --mes 9    # otro mes
py -3.12 scripts/comparar_con_manual.py          # compara los métodos con lo marcado
py -3.12 scripts/refinar_marcas.py               # revisa las marcas, corrige la puntería de los clics y vuelve a comparar
```

- **Marcar** (`etiquetar_eventos.py`): dos clics marcan el inicio y el fin de un evento; la fase (subida o bajada) sale de los extremos. Al cerrarlo se le ajusta la misma exponencial que usan los métodos y se muestra su ACH. Clic derecho borra, `z` deshace, las flechas cambian de día y `q` guarda y cierra. Se guarda solo en `outputs/mediciones/<ip>_id<NN>/manual/eventos_manuales.csv`, y al volver a abrir sigue desde el primer día sin revisar. No muestra lo que detecta ningún método.
- **Comparar** (`comparar_con_manual.py`): por método, qué porcentaje de los eventos manuales detecta, en cuántos tramos los parte, con qué desfase en el inicio y cuántos eventos agrega; y cuánto se aparta su ACH del manual, con la asíntota libre y usando la ocupación conocida en los dos lados. Deja tablas, nueve figuras y una figura por día con la fila manual arriba, en `manual/`.
- **Figuras del ACH contra la referencia**: M1 y M3, ACH del método contra el manual (asíntota libre y ocupación conocida); M4, todos los eventos del mes uno por columna, con el ACH de referencia arriba y, por método, cuánto más o menos da; M5, reparto del cociente método / referencia; M6, porcentaje de eventos manuales a menos de cierta diferencia (no estimar cuenta como fallo); M7, diferencia según cuántas constantes de tiempo abarca el evento manual; M8, ACH de cada día por grupo de bloques del horario, referencia y métodos calculados igual. Las tablas correspondientes son `aciertos_por_metodo.csv`, `diferencia_segun_la_referencia.csv`, `ach_por_dia_y_grupo.csv` y `ach_del_dia_contra_manual.csv`.
- **Revisar y corregir la puntería** (`refinar_marcas.py`): un clic sobre la gráfica de un día tiene una precisión de minutos. Mirando sólo la serie, corre un inicio marcado antes de que la serie arranque al último punto en que todavía está en su extremo, y un fin marcado después del punto de giro al extremo; a lo sumo 3 minutos y un quinto de la duración, y sólo recortando (`manual.trim_event`). No toca las marcas originales: deja la copia en `manual_refinado/`, con lo que cambió en cada evento (`cambios_por_evento.csv`), cuatro avisos para volver a mirar a mano (va en contra de su fase, ajusta mal, se pisa con el anterior, abarca menos de una constante de tiempo), una figura por día con los clics originales y los corregidos (`figs/marcas/`) y la comparación con los métodos repetida (`A3_antes_y_despues`). No mueve ninguna marca mirando lo que dan los métodos: dejaría de ser una referencia.
- **La referencia depende de dónde empieza el evento**: `A2_sensibilidad_al_inicio` y `sensibilidad_al_inicio.csv` miden cuánto cambia el ACH de referencia si el inicio se corre unos minutos con el fin en su lugar. Por eso la corrección no decide dónde "empieza de verdad" la exponencial dentro de la transición de entrada: es una decisión de modelado de quien marca, no un error de puntería.
- **Qué tan firme es la referencia**: el ACH manual sale de un ajuste con la asíntota libre, igual que el de los métodos. En un evento que abarca menos de una constante de tiempo está tan mal determinado como el de cualquier método, y la diferencia mide la indeterminación de los dos. M4 lo marca evento por evento y M7 separa por ese criterio. Con la ocupación conocida, los tramos con la clase adentro que no tienen solución con la generación del caso medio quedan afuera de la comparación.

## Informe de ventilación de un salón

Resume, para cada día hábil con clase, lo que ya ajustaron los métodos. Hay que correr antes `run_mediciones.py` para ese salón, con su horario, su volumen y su clase cargados en `co2ach/salones.py`.

```
py -3.12 scripts/run_ventilacion.py                  # salón 2 de A, menos de 1 min
py -3.12 scripts/run_ventilacion.py --rmse-rel 0.1   # qué tan bueno se le exige al ajuste para entrar al ACH del día
```

- **Exposición**: CO₂ medio y percentil 95 en horario de clase, porcentaje y horas por encima de 1000, 1500 y 2000 ppm, y reparto del horario entre las categorías de EN 16798-1 (exceso sobre el exterior de 550, 800 y 1350 ppm).
- **ACH por bloque del horario**, como consenso entre métodos: el promedio de los ajustes de todos los métodos ponderado por su duración (pedido de los tutores, 2026-10-16; antes era la mediana entre métodos, que queda en `ach_mediana`). Para cada método, el promedio de sus ajustes ponderado por duración; el mínimo y el máximo entre métodos dan la dispersión. Con la clase adentro salen de las subidas ajustadas con la generación conocida (caso medio, con el bajo y el alto como rango); con el salón vacío, de las bajadas ajustadas hacia el exterior. Sólo entran los ajustes con RMSE de hasta 5 % del recorrido; `tablas/ventilacion_sensibilidad_al_filtro.csv` muestra cuánto cambia el resultado con otros límites.
- **Aire exterior por persona**: ACH × volumen / personas, contra los 4, 7 y 10 L/s de las categorías III, II y I.
- **Límite del ACH con la clase adentro**: sale sólo de los ratos en que el CO₂ sube, así que no cubre los ratos en que está estable o baja con gente y tiende a quedar por debajo del promedio de la clase.

Deja `tablas/ventilacion_por_dia.csv` (una fila por día), `ventilacion_por_dia_y_bloque.csv` y `ventilacion_por_mes.csv`; diez figuras en `figs/ventilacion/` (calendario del año, ACH por día, por mes, categorías, día tipo, mes por bloque, CO₂ contra ACH, caudal por persona, temperatura y días extremos); un tablero por mes en `figs/ventilacion/tableros/`, con un panel por día; y en `tablero/` las tablas que lee el tablero interactivo (`tablero/index.html`).

## Predicción

Pronostica el CO₂ de los 32 salones a 1, 5, 10, 15, 20, 25, 30, 45 y 60 min, y cuándo va a superar 1000 ppm.

```
py -3.12 scripts/run_prediccion.py --etapa datos          # orígenes y grillas de 1 minuto, ~4 min
py -3.12 scripts/run_prediccion.py --etapa entrenar       # modelos principales
py -3.12 scripts/run_prediccion.py --etapa chronos        # Chronos sin reentrenar, ~2 h; se puede reanudar
py -3.12 scripts/run_prediccion.py --etapa evaluar        # pronósticos de prueba, métricas y umbral
py -3.12 scripts/run_prediccion.py --etapa informacion    # estudio de información y modelos por salón
py -3.12 scripts/run_prediccion.py --etapa figuras        # figuras con lo ya calculado
py -3.12 -m unittest tests.test_prediccion -v
```

Chronos se instaló con `py -3.12 -m pip install --user chronos-forecasting "idna==3.6"`. Sin `--user`, pip no puede escribir en `C:\Python312\Scripts`, y sin fijar `idna` intenta reemplazarlo y falla. Los pesos (`amazon/chronos-bolt-tiny` y `amazon/chronos-bolt-small`) se bajan de Hugging Face la primera vez.

Deja todo en `outputs/prediccion/`: `datos/`, `modelos/`, `pronosticos/`, `tablas/` y `figs/` (con `pies_de_figura.md` y los días en `figs/dias/`). `datos/` ocupa unos 430 MB.

**Datos**

- Una grilla de 1 minuto por salón: media de las lecturas de cada minuto, sin los valores fuera del rango físico, y con los huecos de hasta 5 min interpolados. Los más largos quedan sin datos.
- Sólo cuentan los días hábiles con clase: el CO₂ recorre al menos 150 ppm.
- La línea de base es causal: el mínimo del percentil 1 diario de los 7 días anteriores. No sirve el de `daily_table`, porque usa una ventana centrada.

**Partición por días de cada mes** (`SPLIT_LAST_DAY` en `co2ach/prediccion.py`). Así, todas las estaciones quedan en el entrenamiento y en la prueba. A cambio, el modelo se entrena con días posteriores a algunos de los de prueba.

| Partición | Días de cada mes | Orígenes |
|---|---|---|
| Entrenamiento | 1 al 15 | cada 2 min, de 7:00 a 18:00 |
| Detención temprana | 16 al 19 | todos los minutos, de 7:30 a 17:30 |
| Calibración de los intervalos | 20 al 23 | todos los minutos, de 7:30 a 17:30 |
| Prueba | 24 a fin de mes | todos los minutos, de 7:30 a 17:30 |

Marzo y diciembre no tienen días de prueba con clase: del 24 a fin de mes caen la Semana de Turismo y el fin de los cursos.

Hasta el 2026-10-16 la partición era cronológica: se entrenaba con marzo a julio, se ajustaba con agosto y se probaba con septiembre a diciembre. Los resultados de esa versión están en `outputs/prediccion_particion_anterior/`.

El día de ejemplo, sus cuatro orígenes y la semana que se dibuja entera están en `EXAMPLE_DAY`, `EXAMPLE_ORIGINS` y `EXAMPLE_WEEK`. Son el 24 de junio y la semana del 26 de agosto en A02, y los usan todos los scripts.

Para correr todo de nuevo hay que lanzar `outputs/prediccion/correr_todo.sh`. Corre las etapas de a una y deja marcas, así que se puede relanzar si se corta. Para que no muera con la sesión, conviene lanzarlo desde afuera de la sesión, con WMI.

**Variables**

Todas son causales: sólo miran hasta el origen. Un test lo verifica alterando los datos posteriores. Se agrupan así:

- **G0, CO₂ reciente**: el valor, sus cambios de 1 a 60 min, pendientes, curvatura y extremos de la última hora;
- **G1, calendario**: el minuto del día y el día de la semana;
- **G2, temperatura y humedad**: sus valores y sus cambios en 5, 15 y 30 min;
- **G3, historia del salón**: la mediana a esa hora en sus 5 días con clase anteriores, en t y en t+h; el máximo del día; el exceso sobre la línea de base;
- **G4, el salón**: su identificación y su escuela.

El objetivo es C(t+h) − C(t).

**Modelos**

| Clave | Modelo |
|---|---|
| `persistencia` | C(t+h) = C(t) |
| `perfil` | la mediana del salón a la hora t+h en sus últimos 5 días con clase |
| `fisico` | balance de masa en línea (ver abajo) |
| `ridge` | regresión ridge sobre las variables estandarizadas, con el salón en one-hot |
| `mlp` | perceptrón de 64 y 32 neuronas, con detención temprana |
| `lightgbm` | un modelo por horizonte, con el salón como categoría |
| `chronos` | Chronos-Bolt-tiny sin reentrenar, con 512 min de contexto |

- **Balance de masa en línea.** Ajusta ΔC = a − β·C̄ sobre los últimos 15, 30 o 60 min; la ventana se elige con los días de detención. ΔC es el cambio de un minuto y C̄ el promedio de sus extremos. Con generación y ventilación constantes, la exponencial del Capítulo 2 cumple esa recta exactamente, con β = 2·tanh(λ/2) y a = β·C∞. Después extrapola esa exponencial; si no hay curvatura, extrapola una recta.
- **Chronos.** El contexto de 2048 min daba más error en los días de detención (180 contra 172 ppm) y tardaba 3,5 veces más. Como control se corre Bolt-small, un modelo 5 veces más grande, cada 10 min de la prueba (77 min en esta máquina). Da el mismo error que tiny: 10 % menos a 1 min y 2 % más a 30–60 min (`tablas/chronos_tiny_contra_small.csv`).

**Intervalos del 80 %**

- **Conformal partido**: los cuantiles 0,1 y 0,9 de los residuos de los días de calibración, por horizonte.
- **LightGBM**: sus cuantiles 0,1 y 0,9, corregidos con CQR.
- **Chronos**: sus cuantiles, corregidos igual. La cobertura de los cuantiles sin corregir se reporta aparte.

**Cruce de 1000 ppm**

- **Cruce medido**: el CO₂ llega a 1000 ppm y se mantiene al menos 2 min. No vuelve a contar hasta que baja de 950 ppm.
- **Alarma probable**: la trayectoria pronosticada llega a 1000 ppm en la próxima hora. La trayectoria une C(t) y los nueve horizontes con rectas.
- **Alarma segura**: también llega el extremo inferior del intervalo del 80 %, es decir, con probabilidad de al menos 90 %.
- **Anticipación**: los minutos seguidos con alarma que terminan en el cruce.
- **Falsa alarma**: una racha de alarma sin un cruce en los 60 min siguientes a su inicio. Contar rachas premia a una alarma que no se apaga (encendida todo el día es una sola racha), así que también se mide la duración: `minutos_falsos_por_dia`, los minutos con alarma sin un cruce en la hora siguiente. La curva de compromiso (`tablas/umbral_compromiso.csv`, figura P16) usa esa duración.
- **Probabilidad de cruzar en los próximos h min**: LightGBM tiene además un clasificador entrenado para esa pregunta. Para los otros modelos sale de su distribución pronosticada.

**Estudio de información**

Se hace con LightGBM y con ridge. Compara tres cosas:

- conjuntos anidados de variables: sólo CO₂, + hora y día, + temperatura y humedad, + historia del salón, + salón;
- el modelo completo sin cada grupo;
- la historia de CO₂ recortada a 15, 30, 60, 120 y 240 min.

Además compara el modelo global, con y sin la identificación del salón, contra un modelo por salón.

### Importancia de las variables

Analiza el LightGBM principal variable por variable. Usa los datos y el modelo de `run_prediccion.py`, y deja tablas y figuras en `outputs/prediccion/importancia/`.

```
py -3.12 scripts/run_importancia.py                    # todo, ~35 min
py -3.12 scripts/run_importancia.py --sin-reentrenar   # sin quitar y reentrenar
py -3.12 scripts/run_importancia.py --solo-reentrenar  # sólo quitar y reentrenar, ~15 min
py -3.12 scripts/run_importancia.py --solo-figuras
```

Compara cuatro métodos (`co2ach/importancia.py`):

| Método | Qué mide | Sobre qué |
|---|---|---|
| Ganancia | cuánto redujo la pérdida cada variable en los árboles | el entrenamiento |
| Permutación | cuánto sube el error al desordenar la variable | 100.000 orígenes de prueba |
| SHAP | contribución de cada variable a cada pronóstico | 10.000 de esos orígenes |
| Quitar y reentrenar | cuánto sube el error al entrenar sin la variable | las 10 principales por permutación, a 15 y 60 min |

- **SHAP** es TreeSHAP, el que LightGBM calcula con `pred_contrib=True`. No requiere el paquete `shap`, pero tarda unos 3 ms por origen y por modelo. Las contribuciones, sumadas al valor base, dan el pronóstico (hay un test que lo verifica).
- **El caso concreto** (`I1_caso`) explica el pronóstico a 30 min desde cuatro orígenes del 24 de junio en A02.

Resultados principales:

- Los tres primeros métodos coinciden bastante: Spearman de 0,71 a 0,96 entre métodos, según el horizonte (`acuerdo_entre_metodos.csv`).
- Quitar una de las 10 variables principales casi no cambia el error: menos de 1 % a 15 min y a lo sumo 1,4 % a 60 min. Las variables son redundantes; por ejemplo, el CO₂ actual se reconstruye con el perfil más "CO₂ − perfil".
- La excepción es la temperatura a 60 min: quitarla sube el error 2,7 %, aunque por permutación parezca poco importante. Como escala del ruido, reentrenar el modelo completo con otras semillas cambia el error hasta 0,2 ppm a 15 min y 1,2 ppm a 60 min (`variacion_por_semilla.csv`).
- La medida útil de lo que el modelo necesita es quitar grupos enteros, en el estudio de información de `run_prediccion.py`.
- Todo esto está en el capítulo 7, sección `sec:pred_informacion`, con las figuras I1, I2, I3, I5 e I7.

### Galería de pronósticos

```
py -3.12 scripts/run_galeria.py                  # ~12 min: reentrena LightGBM con los conjuntos I0 a I3
py -3.12 scripts/run_galeria.py --solo-figuras
```

Deja figuras en `outputs/prediccion/galeria/figs/`, con `pies_de_figura.md`.

**Días** (siete en total):

- el 24 de junio en A02;
- en A02, B07 y A06, el día de error mediano de LightGBM a 30 min y el de mayor error.

**Figuras por día:**

- **G1**: todos los modelos a 5, 15, 30 y 60 min.
- **G2**: LightGBM con los cinco niveles de información, a 15, 30 y 60 min.
- **G3**: abanicos desde cuatro orígenes con los cinco niveles de información.

**Figuras de la semana del 7 de octubre en A02** (a 30 min):

- **G4**: persistencia contra LightGBM.
- **G5**: sólo CO₂ contra el modelo completo.

**Figuras por situación** (en A02, B07 y A06, días de prueba):

| Situación | Definición |
|---|---|
| inicio de subida | quieto (±30 ppm en 10 min) y sube al menos 120 ppm en los 10 siguientes |
| subida en curso | subió al menos 100 ppm en 10 min y sube otro tanto en los 10 siguientes |
| cerca de un máximo | subió al menos 80 ppm en 10 min y baja otro tanto en los 10 siguientes |
| inicio de bajada | quieto y baja al menos 120 ppm en los 10 siguientes |
| bajada en curso | bajó al menos 100 ppm en 10 min y baja otro tanto en los 10 siguientes |
| meseta alta | por encima de 900 ppm, cambió menos de 60 ppm en 30 min y cambia menos de 80 en los 30 siguientes |

Las situaciones se definen mirando lo que pasó después del origen. Sirven para comparar los modelos entre sí, no para medir el error de cada uno en abstracto.

- **G6**: una figura por situación, con tres casos de error de LightGBM bajo, típico y alto (percentiles 25, 50 y 75 a 30 min). Cada caso tiene tres paneles: las referencias y el balance de masa; ridge, MLP, LightGBM y Chronos; y LightGBM con cada nivel de información.
- **G7** y **G8**: la trayectoria media de todos los inicios de subida y de bajada, con el origen 10 min antes, justo al inicio y 10 min después. G7 compara los modelos y G8, los niveles de información.
- `error_por_situacion.csv`: error absoluto medio y sesgo de cada modelo y nivel de información, por situación, a 10, 30 y 60 min. Usa todos los orígenes de cada situación.

Los modelos con menos información usan los mismos parámetros que el principal. Todos los niveles llevan intervalo conformal calibrado en los días de calibración, para compararlos con el mismo método.

### Penalizar más quedarse corto

```
py -3.12 scripts/run_asimetrica.py                 # ~55 min: 36 modelos con pérdida cuantil
py -3.12 scripts/run_asimetrica.py --solo-figuras
```

Con la pérdida cuantil de nivel τ, quedarse corto cuesta τ/(1 − τ) veces más que pasarse. Se compara, en la prueba, el modelo actual (media) con:

- LightGBM con pérdida cuantil τ = 0,5 a 0,8, con los mismos parámetros que el principal;
- el cuantil 0,9 del modelo principal;
- el modelo actual corrido por el cuantil τ de sus residuos de calibración.

Tablas y figuras en `outputs/prediccion/asimetrica/`.

| | Modelo actual | τ = 0,7 | τ = 0,8 |
|---|---|---|---|
| Se queda corto | 45 % de las veces | 30 % | 21 % |
| Error absoluto medio a 30 min | 176 ppm | 194 ppm | 221 ppm |
| Cruces anticipados con al menos 15 min | 52 % | 71 % | 80 % |
| Alarma en falso por día | 30 min | 57 min | 80 min |

- Todas las variantes caen sobre la curva de compromiso del clasificador (`A3_umbral`): para la alarma, penalizar más quedarse corto equivale a bajar el nivel p.
- Lo que sí cambia es el pronóstico del instante del cruce: con τ ≈ 0,7–0,8 deja de llegar tarde cuando faltan 10 a 30 min.
- El corrimiento constante logra lo mismo en la alarma, pero con más error absoluto que el modelo cuantil (206 contra 194 ppm a 30 min con τ = 0,7).
- La mediana (τ = 0,5) tiene un error absoluto apenas menor que la media (171 contra 176 ppm a 30 min), porque es lo que minimiza esa métrica.

## Capítulos de la tesis

`latex/capitulo_04.tex` (métodos, con un diagrama de cada uno), `capitulo_05.tex` (series sintéticas y CFD), `capitulo_06.tex` (mediciones del salón y contraste con la referencia marcada a mano) y `capitulo_07.tex` usan las figuras de `latex/figuras/{segmentacion,validacion,aplicacion,prediccion}/`. `capitulo_07.tex` contiene dos capítulos, para no agregar otro `\input` en el documento principal: el 7 (pronóstico de la concentración de CO₂) y el 8 (cruce de 1000 ppm, interpretación del modelo y qué información necesita). Las referencias que agregan están en `latex/referencias_nuevas.bib`.

`latex/apendices.tex` reúne los apéndices y va después de `\appendix`.

- **A, generación de CO₂ por los ocupantes** (capítulo 2): el método de Persily y de Jonge, las ecuaciones de Schofield, la tabla por edad y actividad, y los valores adoptados.
- **B, complementos de la predicción** (capítulo 7):
  - la configuración de los modelos;
  - el error en los nueve horizontes;
  - Chronos tiny contra small;
  - dos jornadas de otros salones (G1 de la galería);
  - los casos de cada situación (G6);
  - complementos del estudio de la información.

Las figuras de situaciones salen de `run_galeria.py`, de su versión sin título en `outputs/prediccion/galeria/figs/tesis/`. Las figuras del capítulo 2 (`latex/figuras/marco/`) no las genera este repo: vienen del borrador de la tesis.

```
py -3.12 scripts/figuras_tesis.py                                # vuelve a dibujar las figuras de la comparación base y los diagramas, y copia todas a latex/figuras/
py -3.12 scripts/figuras_tesis.py --solo-copia --solo-diagramas  # sólo copia y vuelve a dibujar los diagramas
```

Los diagramas de los métodos están en `co2ach/plots_diag.py`: los pasos de Hyun y sus variantes, la derivada con su histéresis, el criterio de PELT sobre un ejemplo calculado, los estados del HMM, qué ajuste corresponde a cada tramo y el esquema de la comparación con la referencia manual. Las figuras del salón se copian de lo que dejaron `run_mediciones.py`, `run_ventilacion.py`, `comparar_con_manual.py` (sobre las marcas originales) y `refinar_marcas.py`, así que hay que correrlos antes si cambió algo. En el texto el salón figura como A02 de la Escuela A, igual que en el capítulo 3.

## Estructura

```
co2ach/physics.py        balance de masa y su solución
co2ach/synth.py          escenarios, simulador y verdad
co2ach/signal_utils.py   mediana, outliers, derivadas, detector de quiebres
co2ach/fitting.py        motor de ajuste (general, Hyun, generación conocida) y cotas teóricas
co2ach/segmenters/       hyun, derivative, pelt_exp, hmm, oracle
co2ach/pipeline.py       método = segmentador + ajuste + filtros
co2ach/evaluate.py       comparación con la verdad y métricas
co2ach/plots.py          figuras de la comparación base
co2ach/regimes.py        de fases a regímenes: la meseta como asíntota
co2ach/variants.py       las cuatro variantes de ajuste de cada método
co2ach/experiments.py    Monte Carlo en paralelo
co2ach/plots_exp.py      figuras de los experimentos, en español o inglés
co2ach/cfd.py            series de una simulación CFD: sensores, promedios, esquema y ajuste por estado
co2ach/plots_cfd.py      figuras del análisis de CFD
co2ach/mediciones.py     mediciones reales: carga de un salón, series diarias, ruido efectivo y resúmenes
co2ach/plots_med.py      figuras del análisis de mediciones
co2ach/plots_diag.py     diagramas de los métodos, para la tesis
co2ach/salones.py        lo que se sabe de cada salón: horario, volumen y quiénes lo ocupan
co2ach/manual.py         referencia manual: eventos marcados a mano, su ajuste y la comparación con los métodos
co2ach/ventilacion.py    informe de ventilación por día: exposición, ACH por bloque y caudal por persona
co2ach/plots_vent.py     figuras y tableros por mes del informe de ventilación
co2ach/prediccion.py     predicción: grilla de 1 minuto, variables, modelos, intervalos, cruces de 1000 ppm y métricas
co2ach/plots_pred.py     figuras de la predicción
tablero/index.html       página del tablero interactivo (lee las tablas de outputs/.../tablero/)
latex/                   capítulos de la tesis, sus figuras y las referencias nuevas
```

## Supuestos donde la tesis de Hyun no alcanza

Están como parámetros y documentados en `co2ach/segmenters/hyun.py` y `co2ach/pipeline.py`.

- C0 fijo en la primera muestra filtrada del segmento.
- El ajuste se hace sobre la serie filtrada con mediana.
- El gradiente es `np.gradient`; si la ventana de RHL es plana, RHL = 0,5.
- DBSCAN usa sólo la suma acumulada, con `eps = 0.5`. Equivale a quedarse con las rachas de al menos `min_samples` puntos.
- El criterio C − C_ext > 0 se evalúa sobre la serie filtrada con mediana y con una tolerancia de 25 ppm (o 3 desvíos de ruido, si es más). Aplicado literalmente descarta toda subida que arranca desde la línea de base, porque ahí el ruido deja a C por debajo de C_ext la mitad de las veces.
- Un λ̂ que queda pegado a una cota del optimizador (0,02 o 30 1/h) no cuenta como estimación.

## Cómo leer las métricas

- **Error por régimen**: para cada régimen verdadero con transitorio, el ACH estimado es la media ponderada por duración de los ajustes aceptados que caen en él (con al menos 80 % del tramo dentro del régimen).
- **ACH de referencia de un ajuste**: el del régimen dominante del tramo. La columna `pureza` dice qué fracción del tramo le pertenece.
- **F1 de inicios**: inicios de transitorio detectados frente a los cambios de régimen verdaderos, con tolerancia de ±5 minutos y el mismo signo (subida o bajada).
- **Filtro por error estándar**: las columnas `se_*` del resumen repiten las métricas quedándose sólo con los ajustes cuyo error estándar relativo de λ̂ es menor que 25 %.
