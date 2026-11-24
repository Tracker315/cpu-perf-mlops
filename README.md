# Prediccion de rendimiento multi-core de CPUs + MLOps

Proyecto final - Elias Rodriguez

Predecir el rendimiento multi-core (cpuMark de PassMark) de un procesador a partir de sus especificaciones
(nucleos, TDP, socket, tipo, anio, nombre del modelo) y llevarlo a algo parecido a produccion con MLflow en local.

El recorrido completo con resultados esta en `notebooks/Proyecto_CPU_MLOps.ipynb`.

## Estructura

```
Proj/
  src/
    config.py        rutas, nombres de experimentos, modelo registrado, semilla
    datos.py         descarga de kaggle (version + hash), limpieza y variables
    externos.py      TechPowerUp + pc_build: cache, nm, frecuencias, hilos (union por nombre)
    calidad.py       validacion de datos antes de entrenar
    modelos.py       preprocesamiento y catalogo de modelos (+ Hibrido, ConRuido)
    entrenar.py      GridSearch, kfold + validacion temporal, log en mlflow, registro y alias
    drift.py         data drift (PSI/KS), label drift y concept drift
    blender.py       Blender Open Data: snapshots con fecha, ajuste por version, un puntaje por CPU
    cruzado.py       modelo cruzado PassMark + Blender + traductor (de aqui sale el v7 en produccion)
    pipeline.py      pipeline automatico (lo que corre cron)
    servir.py        API online (FastAPI)
    batch.py         prediccion por lote
    monitoreo.py     log de predicciones: latencia, alertas, version (monitoreo.db)
    probar_api.py    prueba de la API sin levantar servidor
  archivo/           codigo de las PRUEBAS ya terminadas (experimentos E1-E13, busqueda, extras X1-X7,
                     transferencia, simulacion, analisis). Ver archivo/README.md
  despliegue/        lo minimo para publicar el modelo en produccion (pagina Streamlit)
  notebooks/         notebook del proyecto con salidas
  reportes/          csv/png/json que generan los scripts
  datos/             raw/v{version}/ (ignorado en git), externos/
  mlflow.db          METADATOS de mlflow (sqlite)
  mlartifacts/       ARTIFACTS de mlflow: modelos, graficas, csv (ignorado en git, ~1GB)
  monitoreo.db       predicciones servidas
  correr_pipeline.sh para cron
```

## Como correr

```bash
pip install -r requirements.txt
cd Proj

python -m src.pipeline              # datos fresh -> calidad -> entrenar -> drift -> batch (no reentrena si el hash no cambio)
python -m src.pipeline --forzar     # reentrena aunque los datos sean los mismos
python -m src.cruzado               # modelo cruzado PassMark + Blender (registra si le gana a produccion)

# pruebas ya terminadas (codigo archivado, resultados en reportes/ y mlflow)
python -m archivo.experimentos      # E1..E13 (o: python -m archivo.experimentos e1 e9)
python -m archivo.busqueda          # busqueda aleatoria de 11 familias
python -m archivo.extras            # X1..X6
python -m archivo.analisis
python -m archivo.simulacion

mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000

uvicorn src.servir:app --port 8000                    # API online
python -m src.probar_api                              # o probarla sin servidor
python -m src.batch entrada.csv salida.csv            # batch
```

Cron (cada lunes 6am): `0 6 * * 1 /ruta/a/Proj/correr_pipeline.sh`

## MLflow: metadatos vs artifacts

- **Metadatos** en `mlflow.db` (sqlite): experimentos, runs, parametros, metricas, tags, registro de modelos. Se puede consultar
  con `mlflow.search_runs(filter_string=...)` o con SQL directo (ejemplos en el Paso 12 del notebook).
- **Artifacts** en `mlartifacts/<experimento>/`: modelos (cloudpickle + codigo de `src/`), graficas, csv. El sqlite solo guarda la ruta.

Organizacion:

| experimento | que tiene |
|---|---|
| `cpu-perf/multi-core` | run padre `entrenamiento__data-v26__AAAAMMDD-HHMM` con un run hijo por modelo `hibrido__data-v26` |
| `cpu-perf/single-core` | lo mismo para threadMark (pregunta secundaria) |
| `cpu-perf/experimentos` | E1-E11, tags `experimento` y `variante` |
| `cpu-perf/drift` | reporte de drift con sus graficas |
| `cpu-perf/batch` | cada lote: `batch__40cpus__modelo-v3`, hash de la entrada |
| `cpu-perf/simulacion` | un run por anio simulado con la decision (mantener/reentrenar) |
| `cpu-perf/analisis` | Amdahl, nucleos vs frecuencia, single vs multi |

Tags en cada run: `data_version` (version de kaggle), `data_hash` (md5), `objetivo`, `target_transform`, `version_tpu`,
`version_pc_build`, `cobertura_*`, `modelo`, `familia_modelo`, `split`.

Registro: modelo `cpu-perf-multicore`, alias `produccion`. Una version nueva solo se vuelve `produccion` si mejora el
`temporal_mae` de la actual. La API y el batch cargan `models:/cpu-perf-multicore@produccion`.

## Resultados principales

| | kfold MAE | temporal MAE (train <=2019, test >2019) | temporal dentro de +-20% |
|---|---|---|---|
| solo cores (Actividad 3) | 2941 | | |
| random forest | 805 | 4270 | 24% |
| random forest + ruido | 791 | 3852 | 27% |
| stacking | 821 | 3901 | 25% |
| hibrido (ridge + RF en residuos) | 835 | 3175 | 33% |
| **hibrido_reciente (+ pesos por antiguedad, vida media 5)** | 804 | **2784** | **36%** |

- El kfold aleatorio es muy optimista (MAE ~800) y en procesadores nuevos el error se multiplica por ~4. Por eso el modelo
  de produccion se escoge con validacion temporal.
- El hibrido gana en temporal porque ridge sigue la tendencia (que el RF no puede extrapolar) y el RF corrige lo no lineal.
  Darle mas peso a lo reciente (E7, E11) lo mejora otra vez: 3175 -> 2784. Es la version 3 en `produccion`.
- Aumentacion de datos (ruido gaussiano, SMOTER, SMOGN): mejora poquita (~808 -> ~798).
- Datos externos: 38% de cobertura, bajan el MAE de 847 a 828.
- Amdahl: elasticidad cpuMark-cores de 1.27 controlando por anio, con curvatura negativa (rinde cada vez menos agregar nucleos).
- Cinebench R23: multi-core depende de cores (0.90) y frecuencia (0.78), single-core casi solo de frecuencia (0.95).

## Modo de prediccion

- **Batch**: catalogos o tandas de procesadores nuevos. El pipeline lo corre con los procesadores mas recientes.
- **Online**: API FastAPI (`/predict`, `/health`, `/metricas`) para consultas al momento (ej. comparar antes de comprar).
- **Streaming**: no aplica, no hay un flujo continuo de eventos, salen pocos procesadores al dia.
- **Edge**: no aplica, no hay dispositivo que tenga que predecir sin conexion.

## Observabilidad

- **Latencia y salud**: cada prediccion se guarda en `monitoreo.db`; `/metricas` da p50/p95, tasa de error, version servida.
  `/health` dice que version esta cargada y desde cuando. (p50 ~60 ms, p95 ~250 ms)
- **Data quality**: `calidad.py` antes de entrenar (si el objetivo trae nulos se detiene el pipeline) y `monitoreo.revisar_entrada`
  en cada prediccion (rangos, nulos, anios que el modelo no ha visto).
- **Performance**: kfold y temporal en cada entrenamiento; en batch, si llega el cpuMark real se loggea el error real.
- **Business KPI**: % de procesadores con error <= 20% (`kpi_dentro_20pct`).

## Drift

Referencia <= 2019 vs nuevos > 2019:
- **Data drift** (cambia la distribucion de X): PSI/KS. Fuerte en fabricante, es_mobile, cores (media 3.9 -> 8.2), familia y TDP.
- **Label drift** (cambia la distribucion de y): mediana de cpuMark 2003 -> 4396, PSI 0.49.
- **Concept drift** (cambia la relacion X -> y): el modelo entrenado con lo viejo tiene MAE 530 en lo viejo y 3921 en lo
  nuevo y subestima (predice ~0.69 de lo real).

Simulacion de produccion 2014-2022 (modelo inicial con datos hasta 2013):

| estrategia | reentrenamientos | dentro de +-20% promedio | MAE promedio |
|---|---|---|---|
| congelado | 0 | 35% | 4119 |
| por alerta (KPI cae >10 pts o PSI >0.25) | 3 | 46% | 2517 |
| cada anio | 9 | 54% | 1976 |

(con hibrido_reciente; con un RF normal el congelado cae a 18% promedio porque no extrapola)

## Monitoreo con Evidently

Ademas del PSI / KS propio (`src/drift.py`), el paso de drift genera un reporte estandar de **Evidently**
(viejos <= 2019 como referencia vs nuevos):

- `reportes/evidently/evidently_data_drift.html`: drift por columna (Wasserstein, KS, Jensen-Shannon) + resumen de calidad
- `reportes/evidently/evidently_modelo.html`: desempeno del modelo (en log) y drift del objetivo
- `reportes/evidently/evidently_columnas.csv`: veredicto de Evidently por columna junto a nuestro PSI

Los HTML tambien quedan en MLflow (artifacts `drift/evidently`, experimento `cpu-perf/drift`). Evidently 0.7 aun no
soporta pandas 3, asi que corre en su propio entorno (como un servicio de monitoreo aparte); si no existe, el pipeline
sigue sin ese reporte:

```bash
uv venv ~/evidently_env
uv pip install --python ~/evidently_env/bin/python "evidently==0.7.23" "pandas<3" pyarrow
# otra ruta: export EVIDENTLY_PYTHON=/ruta/al/python
```

## En la nube (GitHub Actions + Hugging Face + Streamlit)

```
GitHub (codigo publico)                    Hugging Face (dataset PRIVADO)
  Actions, cada lunes 3 am   -- bajar -->    mlflow.db + mlartifacts  (MLflow: SQLite + artifacts aparte)
  1. src.nube bajar          <-- subir --
  2. src.pipeline
  3. src.nube subir
  4. src.nube pagina --si-cambio  (regenera despliegue/ y hace push)
                                   --push--> Streamlit Community Cloud: pagina publica (se redespliega sola)
```

- `.github/workflows/pipeline.yml`: el pipeline semanal (tambien se corre a mano en la pestana Actions).
- `src/nube.py`: baja / sube el estado de MLflow y corrige las rutas absolutas de los artifacts (`reubicar`).
- MLflow sigue siendo SQLite + artifacts aparte; Hugging Face solo los guarda (privado) entre corridas.
- Un modelo nuevo pasa a produccion solo si mejora al menos 1% (`config.MEJORA_MINIMA`); si no, la pagina no cambia.

Configuracion (una sola vez):

1. Hugging Face: token con permiso de escritura; `hf auth login` y `HF_REPO_MLFLOW=usuario/cpu-perf-mlflow python -m src.nube inicial`
   (sube el estado local de MLflow a un repo privado).
2. GitHub: Settings -> Secrets and variables -> Actions: secret `HF_TOKEN` (opcional `KAGGLE_USERNAME`, `KAGGLE_KEY`)
   y variable `HF_REPO_MLFLOW`.
3. share.streamlit.io -> Create app -> este repo, rama `main`, archivo `despliegue/streamlit_app.py`,
   Advanced settings -> Python 3.12.
4. Pestana Actions -> "pipeline semanal" -> Run workflow (primera corrida a mano).

## Referencias

Ver `referencias.md` (Wang et al. 2019, SPECnet 2018, Torgo 2013 SMOTER, Branco 2017 SMOGN, Caruana 2004, Gama 2014, etc).
