"""Blender Open Data (opendata.blender.org, licencia CC0): benchmark de render en CPU con TODOS los nucleos.
Es una medida de rendimiento multi-core (samples por minuto) con procesadores hasta hoy (2026).

Del archivo oficial (un JSON por envio, ~2 GB) se arma una tabla de un procesador por fila:
  1. solo CPU, esquema v4 (Blender 3.x en adelante), un solo socket, las 3 escenas (monster, junkshop, classroom)
  2. puntaje de cada envio = suma de samples/minuto de las 3 escenas (asi lo calcula Blender)
  3. cada version de Blender rinde distinto: se ajusta con un modelo log(puntaje) = efecto CPU + efecto version
     (medianas alternadas) y se reporta en "unidades de Blender 4.2"
  4. se queda la mediana por procesador (robusta a equipos raros), con al menos 5 envios

Correr:  python -m src.blender            (baja el snapshot si no esta y arma la tabla)
"""
import json
import urllib.request
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

from . import config, externos

URL = "https://opendata.blender.org/snapshots/opendata-latest.zip"
CARPETA = config.DATOS / "externos" / "blender"
ESCENAS = {"monster", "junkshop", "classroom"}
VERSION_REF = "4.2"
MIN_ENVIOS = 5


def descargar(forzar=False, max_dias=7, conservar=3):
    """Datos fresh con checkpoints: cada snapshot se guarda como opendata-AAAA-MM-DD.zip (no se sobrescribe).
    Se baja uno nuevo si el ultimo tiene mas de 'max_dias' (Blender publica uno diario). Se conservan los ultimos 'conservar'."""
    CARPETA.mkdir(parents=True, exist_ok=True)
    viejo = CARPETA / "opendata-latest.zip"                      # formato anterior: se renombra con su fecha
    if viejo.exists():
        with zipfile.ZipFile(viejo) as z:
            fecha = next(n for n in z.namelist() if n.endswith(".jsonl"))[9:19]
        viejo.rename(CARPETA / f"opendata-{fecha}.zip")
    snaps = sorted(CARPETA.glob("opendata-*.zip"))
    reciente = snaps[-1] if snaps else None
    edad = (datetime.now() - datetime.strptime(reciente.stem[9:19], "%Y-%m-%d")).days if reciente else None
    if forzar or reciente is None or edad > max_dias:
        tmp = CARPETA / "descargando.zip"
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (cpu-perf academic project)"})
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
            while bloque := r.read(1 << 20):
                f.write(bloque)
        with zipfile.ZipFile(tmp) as z:
            fecha = next(n for n in z.namelist() if n.endswith(".jsonl"))[9:19]
        reciente = CARPETA / f"opendata-{fecha}.zip"
        tmp.replace(reciente)
        for sobra in sorted(CARPETA.glob("opendata-*.zip"))[:-conservar]:
            sobra.unlink()
    return reciente


def leer_envios(zip_):
    """Una fila por envio (CPU + version + puntaje). Lee el JSONL dentro del zip sin descomprimirlo a disco."""
    filas = []
    with zipfile.ZipFile(zip_) as z:
        nombre = next(n for n in z.namelist() if n.endswith(".jsonl"))
        with z.open(nombre) as f:
            for linea in f:
                d = json.loads(linea)
                if d.get("schema_version") != "v4":
                    continue
                spm, version, cpu, info = {}, None, None, None
                for r in d["data"]:
                    di = r.get("device_info", {})
                    if di.get("device_type") != "CPU" or "samples_per_minute" not in r.get("stats", {}):
                        continue
                    si = r.get("system_info", {})
                    if si.get("num_cpu_sockets", 1) != 1:
                        continue
                    spm[r["scene"]["label"]] = r["stats"]["samples_per_minute"]
                    version = ".".join(r["blender_version"]["version"].split(".")[:2])
                    cpu = di["compute_devices"][0]["name"]
                    info = (si.get("num_cpu_cores"), si.get("num_cpu_threads"), si.get("system"), r.get("timestamp", "")[:10])
                if cpu and ESCENAS <= set(spm):
                    filas.append({"cpu": cpu.strip(), "version": version, "puntaje": sum(spm[e] for e in ESCENAS),
                                  "nucleos": info[0], "hilos": info[1], "sistema": info[2], "fecha": info[3]})
    return pd.DataFrame(filas)


def ajustar_version(env, iteraciones=20):
    """log(puntaje) = a[cpu] + b[version]. Medianas alternadas: robusto y suficiente para quitar el efecto de version."""
    env = env.assign(lp=np.log(env["puntaje"]))
    b = pd.Series(0.0, index=env["version"].unique())
    for _ in range(iteraciones):
        a = (env["lp"] - env["version"].map(b)).groupby(env["cpu"]).median()
        b = (env["lp"] - env["cpu"].map(a)).groupby(env["version"]).median()
        b -= b.get(VERSION_REF, b.median())          # version de referencia = 0
    return env.assign(puntaje_ajustado=np.exp(env["lp"] - env["version"].map(b))), np.exp(b)


def armar():
    zip_ = descargar()
    env = leer_envios(zip_)
    env = env[env["puntaje"] > 0]
    env, factores = ajustar_version(env)
    g = env.groupby("cpu")
    tabla = pd.DataFrame({
        "blender_puntaje": g["puntaje_ajustado"].median(),
        "envios": g.size(),
        "variacion_pct": (g["puntaje_ajustado"].quantile(0.75) - g["puntaje_ajustado"].quantile(0.25)) / g["puntaje_ajustado"].median(),
        "nucleos": g["nucleos"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan),
        "hilos": g["hilos"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan),
        "primer_envio": g["fecha"].min(),
        "anio_primer_envio": g["fecha"].min().str[:4].astype(int),
    }).reset_index()
    tabla = tabla[tabla["envios"] >= MIN_ENVIOS].sort_values("blender_puntaje", ascending=False)
    tabla["clave"] = tabla["cpu"].map(externos.normalizar)
    tabla.to_parquet(CARPETA / "blender_cpu.parquet", index=False)
    tabla.to_excel(CARPETA / "blender_cpu.xlsx", index=False)
    with zipfile.ZipFile(zip_) as z:
        snap = next(n for n in z.namelist() if n.endswith(".jsonl"))
    meta = {"fuente": "https://opendata.blender.org (CC0)", "snapshot": snap, "armado": datetime.now().isoformat(timespec="seconds"),
            "envios_usados": int(len(env)), "procesadores": int(len(tabla)), "version_referencia": VERSION_REF,
            "factor_por_version": {k: round(float(v), 3) for k, v in factores.sort_index().items()}}
    (CARPETA / "blender_version.json").write_text(json.dumps(meta, indent=2))
    return tabla, meta


if __name__ == "__main__":
    t, m = armar()
    print(json.dumps({k: v for k, v in m.items() if k != "factor_por_version"}, indent=2))
    print("factor por version (vs 4.2):", m["factor_por_version"])
    print(t.head(15).to_string(index=False))
