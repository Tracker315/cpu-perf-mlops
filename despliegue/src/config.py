import os
from pathlib import Path

# carpetas del proyecto
RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "datos"
REPORTES = RAIZ / "reportes"
MLFLOW_DB = RAIZ / "mlflow.db"            # metadatos (experimentos, runs, parametros, metricas, tags)
ARTIFACTS = RAIZ / "mlartifacts"          # artifacts (modelos, graficas, reportes)
MONITOREO_DB = RAIZ / "monitoreo.db"      # predicciones servidas: latencia, entradas, salidas
TRACKING_URI = f"sqlite:///{MLFLOW_DB}"

# despliegue: CPU_PERF_SIN_INTERNET=1 -> los datos externos se leen SOLO de las copias locales (no consulta
# Kaggle ni Hugging Face al predecir). Lo usa la carpeta despliegue/.
SIN_INTERNET = os.environ.get("CPU_PERF_SIN_INTERNET") == "1"
# Evidently corre en su propio entorno (aun no soporta pandas 3). Si no existe, el pipeline sigue sin ese reporte.
EVIDENTLY_PYTHON = os.environ.get("EVIDENTLY_PYTHON", str(Path.home() / "evidently_env" / "bin" / "python"))

# datos
DATASET_KAGGLE = "alanjo/cpu-benchmarks"
ARCHIVO = "CPU_benchmark_v4.csv"
OBJETIVO = "cpuMark"              # rendimiento multi-core (PassMark)
OBJETIVO_SINGLE = "threadMark"    # rendimiento single-core, para la pregunta secundaria

# columnas que NO se usan como entrada: se calculan con el objetivo o son otro benchmark (leakage)
COLUMNAS_LEAKAGE = ["cpuValue", "threadValue", "powerPerf", "threadMark", "cpuMark"]

# mlflow
EXP_MULTI = "cpu-perf/multi-core"
EXP_SINGLE = "cpu-perf/single-core"
EXP_DRIFT = "cpu-perf/drift"
EXP_BATCH = "cpu-perf/batch"
MODELO_REGISTRADO = "cpu-perf-multicore"
ALIAS_PRODUCCION = "produccion"

# drift: procesadores viejos (referencia) vs nuevos
ANIO_CORTE = 2019
METRICA_SELECCION = "temporal_multi_mae"   # promedio de 5 cortes de tiempo (E13): un solo corte puede ser suerte
MEJORA_MINIMA = 0.01   # un modelo nuevo pasa a produccion solo si baja el MAE al menos 1%: diferencias menores son ruido
CORTES_TEMPORALES = [2016, 2017, 2018, 2019, 2020]   # train <= corte, test = los 2 anios siguientes
SEMILLA = 42


def version_codigo():
    """Huella del codigo de src/ (md5 de todos los .py). Si cambia una linea del modelo, cambia la huella."""
    import hashlib
    h = hashlib.md5()
    for f in sorted((RAIZ / "src").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:10]
