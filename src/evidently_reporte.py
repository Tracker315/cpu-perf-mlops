"""Reporte de monitoreo con Evidently (data drift + resumen de datos + desempeno de regresion).

Corre en un entorno APARTE (Evidently 0.7 aun no soporta pandas 3, que usa el proyecto): el monitoreo es un servicio
separado del modelo. Por eso este archivo NO importa nada de src/; solo lee dos parquet y escribe el reporte.

  referencia.parquet  procesadores viejos (<= ANIO_CORTE): entradas + cpuMark real + prediccion fuera de muestra
  actual.parquet      procesadores nuevos (>  ANIO_CORTE): entradas + cpuMark real + prediccion del modelo viejo

Lo llama src/drift.py:  <python del entorno evidently> src/evidently_reporte.py <carpeta>
Crear el entorno:       uv venv ~/evidently_env && uv pip install --python ~/evidently_env/bin/python
                        "evidently==0.7.23" "pandas<3" pyarrow
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset

# "anio" no se revisa: la referencia y lo actual se separan justo por anio (siempre saldria drift)
NUMERICAS = ["cores", "TDP", "ghz_nombre", "generacion", "gama", "l3_mb", "proceso_nm", "base_ghz", "turbo_ghz", "hilos"]
CATEGORICAS = ["fabricante", "familia", "sufijo", "socket", "es_desktop", "es_laptop", "es_server", "es_mobile"]


def dataset(df, con_objetivo):
    num = [c for c in NUMERICAS if c in df]
    cat = [c for c in CATEGORICAS if c in df]
    definicion = DataDefinition(numerical_columns=num + (["log_cpuMark", "log_prediccion"] if con_objetivo else []),
                                categorical_columns=cat,
                                regression=[Regression(target="log_cpuMark", prediction="log_prediccion")] if con_objetivo else None)
    return Dataset.from_pandas(df, data_definition=definicion)


def main(carpeta):
    carpeta = Path(carpeta)
    ref = pd.read_parquet(carpeta / "referencia.parquet")
    act = pd.read_parquet(carpeta / "actual.parquet")
    for d in (ref, act):                       # en log: el puntaje crece multiplicando (igual que en todo el proyecto)
        d["log_cpuMark"] = np.log1p(d.pop("cpuMark"))
        d["log_prediccion"] = np.log1p(d.pop("prediccion").clip(lower=0))
        for c in CATEGORICAS:
            if c in d:
                d[c] = d[c].astype(str)

    entradas = [c for c in NUMERICAS + CATEGORICAS if c in ref]
    # 1) drift de las ENTRADAS (data drift) y resumen de calidad de datos
    rep_datos = Report([DataDriftPreset(columns=entradas), DataSummaryPreset(columns=entradas)])
    snap_datos = rep_datos.run(dataset(act[entradas], False), dataset(ref[entradas], False))
    snap_datos.save_html(str(carpeta / "evidently_data_drift.html"))
    # 2) desempeno del modelo en referencia vs nuevos (concept drift visto como caida de desempeno) + drift del objetivo
    rep_modelo = Report([RegressionPreset(), DataDriftPreset(columns=["log_cpuMark"])])
    snap_modelo = rep_modelo.run(dataset(act, True), dataset(ref, True))
    snap_modelo.save_html(str(carpeta / "evidently_modelo.html"))

    resumen = {"datos": snap_datos.dict(), "modelo": snap_modelo.dict()}
    (carpeta / "evidently_completo.json").write_text(json.dumps(resumen, indent=1, default=str))
    print(json.dumps({"ok": True, "html": ["evidently_data_drift.html", "evidently_modelo.html"]}))


if __name__ == "__main__":
    main(sys.argv[1])
