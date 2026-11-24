"""Estado de MLflow en la nube (Hugging Face Hub) para que el pipeline corra en GitHub Actions.

MLflow sigue igual que en local: METADATOS en SQLite (mlflow.db) y ARTIFACTS aparte (mlartifacts/). Solo que entre
corridas se guardan en un repositorio PRIVADO de Hugging Face (tipo dataset):

    hf://datasets/<HF_REPO_MLFLOW>/mlflow.db
    hf://datasets/<HF_REPO_MLFLOW>/monitoreo.db
    hf://datasets/<HF_REPO_MLFLOW>/mlartifacts/...

Comandos (desde la carpeta Proj; el token va en la variable HF_TOKEN, nunca en el codigo):
    python -m src.nube inicial            sube TODO el estado local por primera vez (crea el repo privado si no existe)
    python -m src.nube bajar              baja mlflow.db + el modelo en produccion (lo que necesita el pipeline)
    python -m src.nube subir --desde TS   sube mlflow.db, monitoreo.db y los artifacts creados despues de TS (epoch)
    python -m src.nube pagina [--si-cambio] regenera despliegue/ con el modelo en produccion (Streamlit)
    python -m src.nube reubicar           corrige las rutas de artifacts en mlflow.db a la carpeta actual

Variables: HF_TOKEN, HF_REPO_MLFLOW (ej. usuario/cpu-perf-mlflow).
"""
import argparse
import json
import os
import re
import sqlite3
import time
from pathlib import Path

from . import config


def repo_mlflow():
    r = os.environ.get("HF_REPO_MLFLOW")
    if not r:
        raise SystemExit("Falta la variable HF_REPO_MLFLOW (ej. usuario/cpu-perf-mlflow)")
    return r


def api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


# ---------------------------------------------------------------- rutas
def reubicar(db=None, destino=None):
    """MLflow guarda rutas ABSOLUTAS de los artifacts (file:///ruta/Proj/mlartifacts/...). Si el proyecto se mueve
    (otra carpeta, otra computadora, el runner de GitHub) hay que reescribirlas o no encuentra los modelos."""
    db = Path(db or config.MLFLOW_DB)
    destino = str(Path(destino or config.ARTIFACTS).parent).rstrip("/")
    con = sqlite3.connect(db)
    patron = re.compile(r"(/[^'\"\s]*?)/mlartifacts/")
    cambios = 0
    for (tabla,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        columnas = [r[1] for r in con.execute(f"PRAGMA table_info('{tabla}')")
                    if not r[2] or "CHAR" in r[2].upper() or "TEXT" in r[2].upper()]
        for col in columnas:
            prefijos = set()
            for (valor,) in con.execute(f'SELECT DISTINCT "{col}" FROM "{tabla}" WHERE "{col}" LIKE ?', ("%/mlartifacts/%",)):
                for m in patron.finditer(str(valor)):
                    prefijos.add("/" + m.group(1).lstrip("/"))          # file:///a/b -> /a/b
            for p in prefijos - {destino}:
                n = con.execute(f'UPDATE "{tabla}" SET "{col}" = REPLACE("{col}", ?, ?) WHERE "{col}" LIKE ?',
                                (p + "/mlartifacts/", destino + "/mlartifacts/", f"%{p}/mlartifacts/%")).rowcount
                cambios += n
    con.commit()
    con.close()
    print(f"reubicar: {cambios} referencias apuntan ahora a {destino}/mlartifacts")
    return cambios


def carpeta_produccion():
    """Ruta (relativa a mlartifacts/) del modelo que tiene el alias 'produccion' en el registro."""
    con = sqlite3.connect(config.MLFLOW_DB)
    fila = con.execute("""SELECT mv.storage_location FROM registered_model_aliases a JOIN model_versions mv
                          ON mv.name = a.name AND mv.version = a.version WHERE a.name = ? AND a.alias = ?""",
                       (config.MODELO_REGISTRADO, config.ALIAS_PRODUCCION)).fetchone()
    con.close()
    return fila[0].split("/mlartifacts/", 1)[1].rstrip("/") if fila else None


# ---------------------------------------------------------------- sincronizar
def bajar():
    from huggingface_hub import hf_hub_download, snapshot_download
    repo, token = repo_mlflow(), os.environ.get("HF_TOKEN")
    for f in ["mlflow.db", "monitoreo.db"]:
        try:
            hf_hub_download(repo, f, repo_type="dataset", local_dir=config.RAIZ, token=token)
            print(f"bajar: {f}")
        except Exception as e:
            if f == "mlflow.db":
                raise
            print(f"bajar: {f} no existe todavia ({type(e).__name__})")
    reubicar()
    prod = carpeta_produccion()
    if prod:          # el pipeline solo necesita el modelo en produccion (batch); lo demas se queda en la nube
        snapshot_download(repo, repo_type="dataset", local_dir=config.RAIZ, token=token,
                          allow_patterns=[f"mlartifacts/{prod}/*", f"mlartifacts/{prod}/**/*"])
        print(f"bajar: modelo en produccion -> mlartifacts/{prod}")


def subir(desde):
    from huggingface_hub import CommitOperationAdd
    archivos = [p for p in [config.MLFLOW_DB, config.MONITOREO_DB] if p.exists()]
    archivos += [p for p in config.ARTIFACTS.rglob("*") if p.is_file() and p.stat().st_mtime >= desde]
    ops = [CommitOperationAdd(path_in_repo=p.relative_to(config.RAIZ).as_posix(), path_or_fileobj=str(p)) for p in archivos]
    api().create_commit(repo_mlflow(), repo_type="dataset", operations=ops,
                        commit_message=f"pipeline {time.strftime('%Y-%m-%d %H:%M')}: {len(ops)} archivos")
    print(f"subir: {len(ops)} archivos (mlflow.db, monitoreo.db y {len(ops) - 2} artifacts nuevos)")


def inicial():
    a = api()
    repo = repo_mlflow()
    a.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    a.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(config.RAIZ),
                    allow_patterns=["mlflow.db", "monitoreo.db", "mlartifacts/**"],
                    commit_message="estado inicial de MLflow (SQLite + artifacts)")
    print(f"inicial: estado de MLflow subido a datasets/{repo} (privado)")


def pagina(si_cambio=False):
    """Regenera despliegue/ con el modelo en produccion (src.desplegar). En GitHub Actions el siguiente paso hace
    commit + push y Streamlit Community Cloud vuelve a desplegar solo."""
    if si_cambio:
        ultimo = config.REPORTES / "ultimo_pipeline.json"
        if not (ultimo.exists() and json.loads(ultimo.read_text()).get("cambio_produccion")):
            print("pagina: produccion no cambio, la pagina se queda igual")
            return False
    from . import desplegar
    desplegar.armar()
    print("pagina: despliegue/ regenerado con el modelo en produccion")
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("accion", choices=["inicial", "bajar", "subir", "pagina", "reubicar"])
    p.add_argument("--desde", type=float, default=0, help="subir: epoch desde el que se suben artifacts nuevos")
    p.add_argument("--si-cambio", action="store_true", help="pagina: solo si el pipeline cambio el modelo en produccion")
    a = p.parse_args()
    {"inicial": inicial, "bajar": bajar, "reubicar": reubicar}.get(a.accion, lambda: None)()
    if a.accion == "subir":
        subir(a.desde)
    elif a.accion == "pagina":
        pagina(a.si_cambio)
