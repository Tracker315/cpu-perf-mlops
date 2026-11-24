"""Observabilidad: guarda cada prediccion servida (online o batch) en SQLite y calcula latencia, salud y calidad de entradas."""
import json
import sqlite3
import time

import numpy as np
import pandas as pd

from . import config

RANGOS_ENTRADA = {"cores": (1, 256), "TDP": (1, 500), "testDate": (2000, 2035)}


def conectar():
    con = sqlite3.connect(config.MONITOREO_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS predicciones (
        ts REAL, modo TEXT, modelo_version TEXT, latencia_ms REAL, n_filas INTEGER,
        entrada TEXT, prediccion TEXT, alertas TEXT, ok INTEGER)""")
    return con


def revisar_entrada(df):
    """Calidad de datos en produccion: nulos, valores fuera de rango, anios fuera de lo que vio el modelo."""
    alertas = []
    for c, (lo, hi) in RANGOS_ENTRADA.items():
        if c in df:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.isna().any():
                alertas.append(f"{c}: {int(s.isna().sum())} nulos")
            fuera = ((s < lo) | (s > hi)).sum()
            if fuera:
                alertas.append(f"{c}: {int(fuera)} fuera de rango [{lo}, {hi}]")
    if "testDate" in df and (pd.to_numeric(df["testDate"], errors="coerce") > 2022).any():
        alertas.append("testDate > 2022: el modelo no ha visto procesadores tan nuevos (posible drift)")
    return alertas


def registrar(modo, version, latencia_ms, entrada, prediccion, alertas, ok=True):
    con = conectar()
    con.execute("INSERT INTO predicciones VALUES (?,?,?,?,?,?,?,?,?)",
                (time.time(), modo, str(version), float(latencia_ms), len(prediccion),
                 json.dumps(entrada, default=str)[:4000], json.dumps([float(p) for p in prediccion])[:4000],
                 json.dumps(alertas), int(ok)))
    con.commit(); con.close()


def resumen(ultimas_horas=None):
    con = conectar()
    df = pd.read_sql("SELECT * FROM predicciones", con)
    con.close()
    if ultimas_horas:
        df = df[df.ts >= time.time() - ultimas_horas * 3600]
    if df.empty:
        return {"n_peticiones": 0}
    con_alertas = df.alertas.map(lambda a: len(json.loads(a)) > 0)
    return {
        "n_peticiones": int(len(df)),
        "n_procesadores": int(df.n_filas.sum()),
        "latencia_p50_ms": float(np.percentile(df.latencia_ms, 50)),
        "latencia_p95_ms": float(np.percentile(df.latencia_ms, 95)),
        "tasa_error": float(1 - df.ok.mean()),
        "pct_peticiones_con_alertas": float(con_alertas.mean()),
        "por_modo": df.modo.value_counts().to_dict(),
        "versiones_modelo": df.modelo_version.value_counts().to_dict(),
    }
