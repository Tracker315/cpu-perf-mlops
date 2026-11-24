"""Prediccion por lote (batch): un CSV de procesadores -> CSV con el rendimiento estimado.

Correr:  python -m src.batch entrada.csv salida.csv
"""
import hashlib
import sys
import time

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from . import config, datos, monitoreo
from .entrenar import preparar_mlflow


def predecir_lote(entrada, salida):
    mlflow.set_tracking_uri(config.TRACKING_URI)
    uri = f"models:/{config.MODELO_REGISTRADO}@{config.ALIAS_PRODUCCION}"
    modelo = mlflow.sklearn.load_model(uri)
    version = MlflowClient().get_model_version_by_alias(config.MODELO_REGISTRADO, config.ALIAS_PRODUCCION).version

    df = pd.read_csv(entrada)
    t = time.perf_counter()
    alertas = monitoreo.revisar_entrada(df)
    X, _ = datos.preparar(df, externos=True)
    df["cpuMark_estimado"] = modelo.predict(X).round()
    seg = time.perf_counter() - t
    df.to_csv(salida, index=False)
    monitoreo.registrar("batch", version, seg * 1000, {"archivo": str(entrada), "filas": len(df)}, df["cpuMark_estimado"], alertas)

    preparar_mlflow(config.EXP_BATCH)
    hash_entrada = hashlib.md5(open(entrada, "rb").read()).hexdigest()[:12]
    with mlflow.start_run(run_name=f"batch__{len(df)}cpus__modelo-v{version}",
                          tags={"modelo_version": str(version), "hash_entrada": hash_entrada, "etapa": "batch"}):
        mlflow.log_metrics({"n_procesadores": len(df), "segundos": seg, "ms_por_procesador": seg * 1000 / len(df),
                            "n_alertas": len(alertas)})
        if "cpuMark" in df:      # si ya hay benchmark real, se mide el desempeno en produccion
            from .entrenar import metricas
            mlflow.log_metrics({f"real_{k}": v for k, v in metricas(df["cpuMark"], df["cpuMark_estimado"]).items()})
        mlflow.log_artifact(str(salida), "predicciones")
    return df, alertas


if __name__ == "__main__":
    out, al = predecir_lote(sys.argv[1], sys.argv[2])
    print(f"{len(out)} procesadores -> {sys.argv[2]} | alertas: {al}")
