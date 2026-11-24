"""Prediccion online: API con FastAPI.

Correr:  uvicorn src.servir:app --port 8000      (desde la carpeta Proj)
  GET  /health     salud: modelo cargado, version, cuando se cargo
  POST /predict    lista de procesadores -> rendimiento multi-core estimado
  GET  /metricas   observabilidad: latencia p50/p95, errores, alertas de calidad
"""
import time
from datetime import datetime

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

from . import config, datos, monitoreo

app = FastAPI(title="Prediccion de rendimiento de CPUs")
ESTADO = {"modelo": None, "version": None, "cargado_en": None}


class Procesador(BaseModel):
    cpuName: str = ""
    cores: int
    TDP: float | None = None
    testDate: int
    socket: str = "unknown"
    category: str = "Unknown"


def cargar_modelo():
    mlflow.set_tracking_uri(config.TRACKING_URI)
    uri = f"models:/{config.MODELO_REGISTRADO}@{config.ALIAS_PRODUCCION}"
    ESTADO["modelo"] = mlflow.sklearn.load_model(uri)
    ESTADO["version"] = MlflowClient().get_model_version_by_alias(config.MODELO_REGISTRADO, config.ALIAS_PRODUCCION).version
    ESTADO["cargado_en"] = datetime.now().isoformat(timespec="seconds")


def predecir_df(df):
    X, _ = datos.preparar(df, externos=True)
    return ESTADO["modelo"].predict(X)


@app.on_event("startup")
def inicio():
    cargar_modelo()


@app.get("/health")
def health():
    ok = ESTADO["modelo"] is not None
    return {"status": "ok" if ok else "sin modelo", "modelo": config.MODELO_REGISTRADO,
            "version": ESTADO["version"], "alias": config.ALIAS_PRODUCCION, "cargado_en": ESTADO["cargado_en"]}


@app.post("/predict")
def predict(procesadores: list[Procesador]):
    t = time.perf_counter()
    df = pd.DataFrame([p.model_dump() for p in procesadores])
    alertas = monitoreo.revisar_entrada(df)
    try:
        pred = predecir_df(df)
    except Exception as e:
        monitoreo.registrar("online", ESTADO["version"], (time.perf_counter() - t) * 1000, df.to_dict("records"), [], alertas, ok=False)
        raise HTTPException(500, str(e))
    lat = (time.perf_counter() - t) * 1000
    monitoreo.registrar("online", ESTADO["version"], lat, df.to_dict("records"), pred, alertas)
    return {"cpuMark_estimado": [round(float(p)) for p in pred], "latencia_ms": round(lat, 2),
            "modelo_version": ESTADO["version"], "alertas": alertas}


@app.get("/metricas")
def metricas():
    return monitoreo.resumen()
