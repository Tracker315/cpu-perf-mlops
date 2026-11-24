"""Datos externos para completar especificaciones que PassMark no trae (cache, litografia, frecuencias, hilos).

- TechPowerUp (Kaggle baraazaid/cpu-and-gpu-stats): 2,614 CPUs con proceso (nm), L3, frecuencia, TDP, fecha.
- pc_build (Hugging Face mehax/pc_build): 1,442 CPUs con hilos, base/boost, L2/L3, litografia, microarquitectura.
Se unen a PassMark por nombre normalizado del modelo.
"""
import json
import re
import urllib.request
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config

RUIDO = [r"@.*$", r"\(tm\)", r"\(r\)", r"\bapu with radeon.*$", r"\bwith radeon.*$", r"\bradeon.*graphics$",
         r"\bprocessor\b", r"\bcpu\b", r"\bintel\b", r"\bamd\b", r"\boem/tray\b", r"\bboxed\b", r"\bbox\b",
         r"\btray\b", r"\b\d+(\.\d+)?\s*ghz\b", r"\b\d+-core\b", r"\blga\S*", r"\bam\d\b", r"\bsocket\b",
         r"\b(single|dual|triple|quad|six|eight|hexa|octa)[ -]core\b", r"\bapu\b", r"\bmobile\b"]


# segunda limpieza: lo que hace que el mismo CPU se escriba distinto en cada fuente (sockets al final, "Extreme",
# "A-Series", "4M Cache", palabras repetidas "FX FX FX-8370", "2440V2" pegado). Revisado a mano: 56 parejas nuevas, todas correctas.
SOCKETS = r"\b(fm1|fm2|am1|am2|am3|am4|am5|str4|strx4|swrx8|str5|sp3|sp5|tr4|lga\S*|fclga\S*|bga\S*|s1700|s1200)\b"
RUIDO_2 = [r"\b\d+(\.\d+)?\s*m\s*cache.*$", r"\bup to\b.*$", r"\b\d+\s*nm\b", r"\bx[ -]series\b", r"\bextreme( edition)?\b",
           r"\b[a-z]?[ -]?series\b", r"\bwof\b", r"\bpib\b", r"\bwith\b.*$", r"\bunlocked\b", r"\bdesktop\b", r"\bboxed?\b",
           r"\b\d+(st|nd|rd|th) gen\b", r"\b\d+ cores?\b"]                                  # "13th Gen Intel Core i9-13900K" (Blender)


def normalizar(nombre):
    """Clave unica del procesador para unir fuentes que lo escriben distinto. Ejemplo (el mismo CPU en 3 fuentes):
        "Intel Core i9-12900K"                         (PassMark)  -> "core i9 12900k"
        "12th Gen Intel(R) Core(TM) i9-12900K"         (Blender)   -> "core i9 12900k"
        "Intel Core i9 12900K 3.2 GHz 16-Core LGA1700" (pc_build)  -> "core i9 12900k"
    """
    n = str(nombre).lower()
    for patron in RUIDO:
        n = re.sub(patron, " ", n)
    n = n.replace("-", " ").replace("_", " ")
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    n = re.sub(SOCKETS, " ", n)
    for patron in RUIDO_2:
        n = re.sub(patron, " ", n)
    n = re.sub(r"\bryzen threadripper\b", "threadripper", n)
    n = re.sub(r"\b(a|e)\s+(\d{1,2})\b", r"\1\2", n)           # "a 10 7860k" -> "a10 7860k"
    n = re.sub(r"(\d)(v\d)\b", r"\1 \2", n)                        # "2440v2" -> "2440 v2"
    tokens = []
    for t in n.split():
        if not tokens or t != tokens[-1]:                           # "fx fx fx 8370" -> "fx 8370"
            tokens.append(t)
    return " ".join(tokens)


def numero(texto):
    m = re.search(r"(\d+(?:\.\d+)?)", str(texto))
    return float(m.group(1)) if m else np.nan


@lru_cache(maxsize=1)            # se lee una vez por proceso (la API no la vuelve a leer en cada peticion)
def techpowerup():
    carpeta, version = copia_kaggle("baraazaid/cpu-and-gpu-stats", "techpowerup")
    t = pd.read_csv(carpeta / "tpu_cpus.csv")
    clock = t["Clock"].astype(str).str.findall(r"(\d+(?:\.\d+)?)")
    en_mhz = t["Clock"].astype(str).str.contains("MHz", case=False)          # "2000 MHz" -> 2.0 GHz
    out = pd.DataFrame({
        "clave": t["Name"].map(normalizar),
        "tpu_proceso_nm": t["Process"].map(numero),
        "tpu_l3_mb": t["L3_Cache"].map(numero),
        "tpu_base_ghz": clock.map(lambda v: float(v[0]) if isinstance(v, list) and v else np.nan) / np.where(en_mhz, 1000, 1),
        "tpu_turbo_ghz": clock.map(lambda v: float(v[-1]) if isinstance(v, list) and v else np.nan) / np.where(en_mhz, 1000, 1),
        "tpu_anio_lanzamiento": pd.to_numeric(t["Release"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce"),
        "tpu_codename": t["Codename"].astype(str),
    })
    return out.drop_duplicates("clave"), version


def copia_kaggle(dataset, carpeta_local):
    """Baja (o reusa) un dataset de Kaggle y guarda una copia en datos/externos/<carpeta_local>/v<version>/.
    Asi cada version queda en el proyecto (checkpoint) y no solo en la cache de kagglehub."""
    import shutil
    ficha = config.DATOS / "externos" / carpeta_local / "version.json"
    if config.SIN_INTERNET:           # despliegue: solo la copia local
        version = json.loads(ficha.read_text())["version"]
        return config.DATOS / "externos" / carpeta_local / f"v{version}", version
    try:
        import kagglehub
        origen = kagglehub.dataset_download(dataset)
    except Exception as e:            # sin internet o Kaggle caido: se usa la ultima copia local (como pc_build)
        if not ficha.exists():
            raise
        version = json.loads(ficha.read_text())["version"]
        print(f"      {carpeta_local}: Kaggle no respondio ({type(e).__name__}), se usa la copia local v{version}")
        return config.DATOS / "externos" / carpeta_local / f"v{version}", version
    version = origen.rstrip("/").split("/")[-1]
    destino = config.DATOS / "externos" / carpeta_local / f"v{version}"
    if not destino.exists():
        shutil.copytree(origen, destino)
    ficha.write_text(json.dumps({"dataset": f"kaggle:{dataset}", "version": version,
                                 "copiado": datetime.now().isoformat(timespec="seconds")}, indent=2))
    return destino, version


HF_PC_BUILD = "mehax/pc_build"


def version_remota_pc_build():
    """Pregunta a Hugging Face la version actual del dataset (sha del ultimo commit). None si no hay internet."""
    if config.SIN_INTERNET:
        return None, None
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{HF_PC_BUILD}", timeout=10) as r:
            meta = json.load(r)
        return meta["sha"][:12], meta.get("lastModified")
    except Exception:
        return None, None


CARPETA_PC_BUILD = config.DATOS / "externos" / "pc_build"      # una subcarpeta por version (sha de Hugging Face)


def ruta_pc_build():
    """Parquet de la version actual: datos/externos/pc_build/<sha>/pc_build.parquet (segun version.json)."""
    ficha = CARPETA_PC_BUILD / "version.json"
    sha = json.loads(ficha.read_text()).get("sha") if ficha.exists() else None
    return CARPETA_PC_BUILD / str(sha) / "pc_build.parquet"


@lru_cache(maxsize=1)            # se lee una vez por proceso (la API no la vuelve a leer en cada peticion)
def pc_build():
    """Datos fresh tambien en los externos: si Hugging Face tiene una version nueva, se baja a su propia carpeta
    (las versiones anteriores se quedan en la suya) y se regenera el Excel. Sin internet usa la copia local."""
    CARPETA_PC_BUILD.mkdir(parents=True, exist_ok=True)
    ficha = CARPETA_PC_BUILD / "version.json"
    local = json.loads(ficha.read_text()) if ficha.exists() else {}
    sha, fecha = version_remota_pc_build()
    if sha and (sha != local.get("sha") or not (CARPETA_PC_BUILD / sha / "pc_build.parquet").exists()):
        destino = CARPETA_PC_BUILD / sha / "pc_build.parquet"
        if not destino.exists():
            destino.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/api/datasets/{HF_PC_BUILD}/parquet/default/train/0.parquet"
            pd.read_parquet(url).to_parquet(destino)
        local = {"dataset": f"huggingface:{HF_PC_BUILD}", "sha": sha, "ultima_modificacion_hf": fecha,
                 "descargado": datetime.now().isoformat(timespec="seconds")}
        ficha.write_text(json.dumps(local, indent=2))
        exportar_excel_pc_build(destino)
        print(f"      pc_build: version nueva {sha} descargada")
    ruta = ruta_pc_build()
    if not ruta.exists():
        raise RuntimeError("No hay copia local de pc_build y no se pudo conectar a Hugging Face")
    p = pd.read_parquet(ruta)
    nombre = p["metadata"].map(lambda m: m["name"])
    out = pd.DataFrame({
        "clave": nombre.map(normalizar),
        "pcb_hilos": p["cores"].map(lambda c: c.get("threads")),
        "pcb_base_ghz": p["clocks"].map(lambda c: (c.get("performance") or {}).get("base")),
        "pcb_boost_ghz": p["clocks"].map(lambda c: (c.get("performance") or {}).get("boost")),
        "pcb_l2_mb": p["cache"].map(lambda c: c.get("l2")),
        "pcb_l3_mb": p["cache"].map(lambda c: c.get("l3")),
        "pcb_litografia_nm": p["specifications"].map(lambda s: numero(s.get("lithography"))),
        "pcb_microarq": p["microarchitecture"].astype(str),
    })
    return out.drop_duplicates("clave"), local.get("sha", "desconocida")


def exportar_excel_pc_build(ruta):
    """Copia en Excel para poder ver el parquet (las columnas agrupadas se separan)."""
    p = pd.read_parquet(ruta)

    def g(d, *ks):
        for k in ks:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
        return d

    def lista(v):
        return ", ".join(map(str, v)) if isinstance(v, (list, np.ndarray)) else v

    t = pd.DataFrame({
        "nombre": p.metadata.map(lambda m: g(m, "name")), "fabricante": p.metadata.map(lambda m: g(m, "manufacturer")),
        "serie": p.series, "arquitectura": p.microarchitecture, "familia_nucleo": p.coreFamily, "socket": p.socket,
        "nucleos_total": p.cores.map(lambda c: g(c, "total")), "nucleos_rendimiento": p.cores.map(lambda c: g(c, "performance")),
        "nucleos_eficiencia": p.cores.map(lambda c: g(c, "efficiency")), "hilos": p.cores.map(lambda c: g(c, "threads")),
        "ghz_base": p.clocks.map(lambda c: g(c, "performance", "base")), "ghz_boost": p.clocks.map(lambda c: g(c, "performance", "boost")),
        "cache_l1": p.cache.map(lambda c: g(c, "l1")), "cache_l2_mb": p.cache.map(lambda c: g(c, "l2")),
        "cache_l3_mb": p.cache.map(lambda c: g(c, "l3")), "litografia": p.specifications.map(lambda s: g(s, "lithography")),
        "tdp_w": p.specifications.map(lambda s: g(s, "tdp")),
        "graficos_integrados": p.specifications.map(lambda s: g(s, "integratedGraphics", "model")),
        "memoria_tipos": p.specifications.map(lambda s: lista(g(s, "memory", "types"))),
        "memoria_max_gb": p.specifications.map(lambda s: g(s, "memory", "maxSupport")),
        "incluye_ventilador": p.specifications.map(lambda s: g(s, "includesCooler")),
        "anio_lanzamiento": p.metadata.map(lambda m: g(m, "releaseYear")),
    })
    with pd.ExcelWriter(CARPETA_PC_BUILD / "pc_build.xlsx") as w:   # Excel de la version actual
        t.to_excel(w, sheet_name="pc_build", index=False)
        ws = w.sheets["pc_build"]
        ws.freeze_panes = "B2"
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(45, max(10, max(len(str(c.value or "")) for c in col[:200]) + 2))


def arreglar_unidades(d):
    """Algunas fuentes mezclan unidades: frecuencias en MHz (3700 en vez de 3.7 GHz) y cache en KB (16384 en vez de 16 MB).
    Lo encontramos porque la red neuronal (E12) explotaba con esos valores. Regresa cuantos valores se corrigieron."""
    n = 0
    for c in ["base_ghz", "turbo_ghz"]:
        mhz = d[c] > 50                       # ningun CPU pasa de ~6 GHz
        d.loc[mhz, c] = d.loc[mhz, c] / 1000
        n += int(mhz.sum())
        d.loc[(d[c] <= 0) | (d[c] > 10), c] = np.nan
    kb = d["l3_mb"] > 1024                    # la cache L3 mas grande es de cientos de MB
    d.loc[kb, "l3_mb"] = d.loc[kb, "l3_mb"] / 1024
    n += int(kb.sum())
    d.loc[d["l3_mb"] <= 0, "l3_mb"] = np.nan
    d.loc[(d["hilos"] <= 0) | (d["hilos"] > 512), "hilos"] = np.nan
    return n


def unir(df):
    """Agrega las columnas externas a PassMark. Regresa (df unido, resumen de cobertura)."""
    d = df.copy()
    d["clave"] = d["cpuName"].map(normalizar)
    tpu, v_tpu = techpowerup()
    pcb, v_pcb = pc_build()
    d = d.merge(tpu, on="clave", how="left").merge(pcb, on="clave", how="left")
    # una sola columna por especificacion: primero TechPowerUp, si no hay, pc_build
    d["l3_mb"] = d["tpu_l3_mb"].fillna(d["pcb_l3_mb"])
    d["proceso_nm"] = d["tpu_proceso_nm"].fillna(d["pcb_litografia_nm"])
    d["base_ghz"] = d["tpu_base_ghz"].fillna(d["pcb_base_ghz"]).fillna(d["cpuName"].str.extract(r"(\d+\.\d+)\s*GHz", expand=False).astype(float))
    d["turbo_ghz"] = d["tpu_turbo_ghz"].fillna(d["pcb_boost_ghz"])
    d["hilos"] = d["pcb_hilos"]
    corregidos = arreglar_unidades(d)
    cobertura = {"tpu": float(d["tpu_proceso_nm"].notna().mean()), "pc_build": float(d["pcb_hilos"].notna().mean()),
                 "cualquiera": float((d["tpu_proceso_nm"].notna() | d["pcb_hilos"].notna()).mean()),
                 **{c: float(d[c].notna().mean()) for c in ["l3_mb", "proceso_nm", "base_ghz", "turbo_ghz", "hilos"]},
                 "version_tpu": v_tpu, "version_pc_build": v_pcb, "valores_corregidos": corregidos}
    return d, cobertura
