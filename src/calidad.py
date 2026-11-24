"""Chequeos de calidad de datos antes de entrenar (data quality)."""
import pandas as pd

COLUMNAS_ESPERADAS = ["cpuName", "price", "cpuMark", "cpuValue", "threadMark", "threadValue",
                      "TDP", "powerPerf", "cores", "testDate", "socket", "category"]

RANGOS = {  # columna: (minimo, maximo) razonables
    "cpuMark": (1, 500000),
    "threadMark": (1, 20000),
    "cores": (1, 512),
    "TDP": (1, 1000),
    "testDate": (2000, 2035),
}


def validar(df, min_filas=1000, max_nulos_objetivo=0.0):
    """Regresa un reporte {chequeo: {ok, detalle}}. Los chequeos 'criticos' detienen el pipeline si fallan."""
    rep = {}
    faltan = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
    rep["esquema"] = {"ok": not faltan, "critico": True, "detalle": f"faltan columnas: {faltan}" if faltan else "completo"}
    rep["filas"] = {"ok": len(df) >= min_filas, "critico": True, "detalle": f"{len(df)} filas (minimo {min_filas})"}
    if faltan:
        return rep

    nulos_obj = df["cpuMark"].isna().mean()
    rep["objetivo_sin_nulos"] = {"ok": nulos_obj <= max_nulos_objetivo, "critico": True, "detalle": f"{nulos_obj:.1%} nulos en cpuMark"}

    nulos = df.isna().mean().round(3)
    rep["nulos_por_columna"] = {"ok": True, "critico": False, "detalle": nulos[nulos > 0].to_dict()}

    fuera = {}
    for c, (lo, hi) in RANGOS.items():
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        n = int(((s < lo) | (s > hi)).sum())
        if n:
            fuera[c] = n
    rep["rangos"] = {"ok": not fuera, "critico": False, "detalle": fuera or "todo en rango"}

    dup = int(df["cpuName"].duplicated().sum())
    rep["duplicados"] = {"ok": dup == 0, "critico": False, "detalle": f"{dup} nombres repetidos"}
    return rep


def pasa(reporte):
    return all(v["ok"] for v in reporte.values() if v["critico"])


def resumen(reporte):
    return {k: ("OK" if v["ok"] else "FALLA") + " - " + str(v["detalle"]) for k, v in reporte.items()}
