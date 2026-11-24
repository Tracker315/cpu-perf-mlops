import hashlib
import re
import shutil

import numpy as np
import pandas as pd

from . import config


def descargar():
    """Baja la ultima version del dataset de Kaggle y regresa ruta, version y hash del archivo."""
    try:
        import kagglehub
        carpeta = kagglehub.dataset_download(config.DATASET_KAGGLE)
    except Exception as e:      # sin internet o Kaggle caido: la ultima copia versionada en datos/raw/
        copias = sorted((config.DATOS / "raw").glob(f"v*/{config.ARCHIVO}"), key=lambda p: int(p.parent.name[1:]))
        if not copias:
            raise
        ruta = copias[-1]
        print(f"      Kaggle no respondio ({type(e).__name__}), se usa la copia local {ruta.parent.name}")
        with open(ruta, "rb") as f:
            hash_md5 = hashlib.md5(f.read()).hexdigest()
        return {"ruta": str(ruta), "version": ruta.parent.name[1:], "hash": hash_md5[:12]}
    ruta = f"{carpeta}/{config.ARCHIVO}"
    version = carpeta.rstrip("/").split("/")[-1]          # .../versions/26 -> "26"
    with open(ruta, "rb") as f:
        hash_md5 = hashlib.md5(f.read()).hexdigest()
    # copia local versionada, asi cada run sabe con que datos se entreno
    destino = config.DATOS / "raw" / f"v{version}"
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copy(ruta, destino / config.ARCHIVO)
    return {"ruta": str(destino / config.ARCHIVO), "version": version, "hash": hash_md5[:12]}


def cargar(ruta):
    return pd.read_csv(ruta)


FAMILIAS = [  # orden importa: lo mas especifico primero
    ("threadripper", "Threadripper"), ("epyc", "EPYC"), ("ryzen 9", "Ryzen 9"), ("ryzen 7", "Ryzen 7"),
    ("ryzen 5", "Ryzen 5"), ("ryzen 3", "Ryzen 3"), ("ryzen", "Ryzen otro"), ("opteron", "Opteron"),
    ("athlon", "Athlon"), ("phenom", "Phenom"), ("xeon", "Xeon"), ("core i9", "Core i9"),
    ("core i7", "Core i7"), ("core i5", "Core i5"), ("core i3", "Core i3"), ("core2", "Core2"),
    ("core 2", "Core2"), ("pentium", "Pentium"), ("celeron", "Celeron"), ("atom", "Atom"),
    ("apple", "Apple"), ("snapdragon", "Snapdragon"), ("cortex", "ARM Cortex"),
]


def familia(nombre):
    n = nombre.lower()
    for clave, fam in FAMILIAS:
        if clave in n:
            return fam
    return "Otra"


def fabricante(nombre):
    n = nombre.lower()
    for clave, fab in [("intel", "Intel"), ("amd", "AMD"), ("apple", "Apple"), ("qualcomm", "Qualcomm"),
                       ("snapdragon", "Qualcomm"), ("arm", "ARM"), ("via", "VIA"), ("mediatek", "MediaTek"),
                       ("kirin", "HiSilicon"), ("exynos", "Samsung"), ("hygon", "Hygon"), ("unisoc", "Unisoc")]:
        if clave in n:
            return fab
    # chips moviles que solo traen el numero de parte: MT6893 (MediaTek), SM8350 / SDM845 (Qualcomm)
    if re.match(r"^mt\d{4}", n):
        return "MediaTek"
    if re.match(r"^(sm|sdm|msm)\d{3,4}", n):
        return "Qualcomm"
    return "Otro"


def ghz_del_nombre(nombre):
    # muchos nombres traen la frecuencia: "Intel Core2 Duo E8400 @ 3.00GHz" o "1850 MHz"
    m = re.search(r"(\d+(?:\.\d+)?)\s*GHz", nombre, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d{3,4})\s*MHz", nombre, re.I)
    return float(m.group(1)) / 1000 if m else np.nan


def generacion_modelo(nombre):
    """Del numero de modelo saca (generacion, gama, sufijo). i7-8700K -> (8, 0.70, K); i9-12900K -> (12, 0.90, K);
    Ryzen 5 5600X -> (5, 0.60, X); Xeon E5-2680 v4 -> (4, 0.68, ''). Si no se puede, NaN."""
    n = nombre.replace("@", " ")
    m = re.search(r"\bi[3579]-(\d{4,5})([A-Z]{0,2}\d?)", n)                 # intel core
    if m:
        num, suf = m.group(1), m.group(2)
        gen = int(num[:2]) if len(num) == 5 else int(num[0])
        return gen, int(num[-3:]) / 1000, re.sub(r"\d", "", suf)[:1]
    m = re.search(r"Ryzen\s+(?:Threadripper\s+(?:PRO\s+)?|\d\s+(?:PRO\s+)?)(\d{4})([A-Z]{0,2})", n, re.I)   # amd ryzen
    if m:
        num = m.group(1)
        return int(num[0]), int(num[1:]) / 1000, m.group(2)[:1]
    m = re.search(r"Xeon.*?(\d{4})\w*\s+v(\d)", n)                          # xeon E5-2680 v4
    if m:
        return int(m.group(2)), int(m.group(1)[1:]) / 1000, ""
    return np.nan, np.nan, ""


def preparar(df, extra=True, externos=False):
    """Feature engineering. Regresa X (solo especificaciones) y el dataframe con columnas extra para analisis.
    extra=True agrega generacion, gama y sufijo sacados del nombre del modelo."""
    d = df.copy()
    d["fabricante"] = d["cpuName"].map(fabricante)
    d["familia"] = d["cpuName"].map(familia)
    d["ghz_nombre"] = d["cpuName"].map(ghz_del_nombre)
    d["anio"] = d["testDate"]
    for seg in ["Desktop", "Laptop", "Server", "Mobile/Embedded"]:     # category puede traer varios: "Desktop, Laptop"
        d["es_" + seg.split("/")[0].lower()] = d["category"].fillna("").str.contains(seg, regex=False).astype(int)
    d["socket"] = d["socket"].fillna("unknown")
    columnas = ["cores", "TDP", "anio", "ghz_nombre", "fabricante", "familia", "socket",
                "es_desktop", "es_laptop", "es_server", "es_mobile"]
    if extra:
        g = d["cpuName"].map(generacion_modelo)
        d["generacion"] = [t[0] for t in g]
        d["gama"] = [t[1] for t in g]
        d["sufijo"] = [t[2] or "ninguno" for t in g]
        columnas += ["generacion", "gama", "sufijo"]
    if externos:     # especificaciones de TechPowerUp / pc_build unidas por nombre (cache, litografia, frecuencias, hilos)
        from . import externos as ext
        u, _ = ext.unir(df)
        for c in ["l3_mb", "proceso_nm", "base_ghz", "turbo_ghz", "hilos"]:
            d[c] = u[c].values
        columnas += ["l3_mb", "proceso_nm", "base_ghz", "turbo_ghz", "hilos"]
    return d[columnas], d
