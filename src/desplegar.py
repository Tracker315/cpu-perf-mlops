"""Empaqueta el modelo en produccion para publicarlo (pagina Streamlit en Streamlit Community Cloud).

Arma la carpeta despliegue/ con LO MINIMO para predecir:
  modelo/        el modelo con alias 'produccion' tal como lo guardo MLflow (model.pkl, MLmodel con la firma, ...)
                 + info.json (version, metricas, con que datos se entreno)
  src/           solo los modulos que usa la prediccion (preparar variables, clases del modelo, monitoreo)
  datos/externos solo las copias de TechPowerUp y pc_build (se usan para completar cache, nm, GHz, hilos por nombre)
Los archivos escritos a mano (streamlit_app.py, requirements.txt, README.md) no se tocan.

Correr:  python -m src.desplegar
Probar:  streamlit run despliegue/streamlit_app.py
"""
import json
import shutil
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from . import config

DESTINO = config.RAIZ / "despliegue"
TEMPORAL = config.RAIZ / ".despliegue_nuevo"     # se arma aqui y solo al final reemplaza a despliegue/
# lo que necesita la prediccion: cruzado (clases SistemaCruzado y Traductor del v7) importa entrenar y externos
MODULOS = ["__init__", "config", "datos", "externos", "modelos", "cruzado", "entrenar", "monitoreo"]


def copiar_modelo():
    mlflow.set_tracking_uri(config.TRACKING_URI)
    cliente = MlflowClient()
    mv = cliente.get_model_version_by_alias(config.MODELO_REGISTRADO, config.ALIAS_PRODUCCION)
    origen = Path(mlflow.artifacts.download_artifacts(f"models:/{config.MODELO_REGISTRADO}/{mv.version}"))
    destino = TEMPORAL / "modelo"
    shutil.copytree(origen, destino, ignore=shutil.ignore_patterns("code"))   # el codigo va en despliegue/src
    run = cliente.get_run(mv.run_id) if mv.run_id else None
    info = {
        "modelo": config.MODELO_REGISTRADO, "version": int(mv.version), "alias": config.ALIAS_PRODUCCION,
        "run_id": mv.run_id, "registrado": datetime.fromtimestamp(mv.creation_timestamp / 1000).isoformat(timespec="seconds"),
        "descripcion": mv.description, "tags_version": dict(mv.tags),
        "tags_run": {k: v for k, v in (run.data.tags.items() if run else []) if not k.startswith("mlflow.")},
        "empaquetado": datetime.now().isoformat(timespec="seconds"),
    }
    (destino / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return info


def copiar_codigo():
    (TEMPORAL / "src").mkdir(parents=True)
    for m in MODULOS:
        shutil.copy(config.RAIZ / "src" / f"{m}.py", TEMPORAL / "src" / f"{m}.py")


def copiar_datos():
    ext, dst = config.DATOS / "externos", TEMPORAL / "datos" / "externos"
    tpu = json.loads((ext / "techpowerup" / "version.json").read_text())["version"]
    (dst / "techpowerup" / f"v{tpu}").mkdir(parents=True)
    shutil.copy(ext / "techpowerup" / "version.json", dst / "techpowerup")
    shutil.copy(ext / "techpowerup" / f"v{tpu}" / "tpu_cpus.csv", dst / "techpowerup" / f"v{tpu}")
    sha = json.loads((ext / "pc_build" / "version.json").read_text())["sha"]
    (dst / "pc_build" / sha).mkdir(parents=True)
    shutil.copy(ext / "pc_build" / "version.json", dst / "pc_build")
    shutil.copy(ext / "pc_build" / sha / "pc_build.parquet", dst / "pc_build" / sha)
    return {"techpowerup": tpu, "pc_build": sha}


def armar():
    """Arma todo en una carpeta temporal; si algo falla (ej. no hay modelo en produccion), despliegue/ queda intacta."""
    shutil.rmtree(TEMPORAL, ignore_errors=True)
    try:
        info = copiar_modelo()
        copiar_codigo()
        versiones = copiar_datos()
    except Exception:
        shutil.rmtree(TEMPORAL, ignore_errors=True)
        raise
    for sub in ["modelo", "src", "datos"]:          # se reemplaza solo lo generado (streamlit_app.py, README, requirements no)
        shutil.rmtree(DESTINO / sub, ignore_errors=True)
        shutil.move(str(TEMPORAL / sub), str(DESTINO / sub))
    shutil.rmtree(TEMPORAL, ignore_errors=True)
    tam = sum(f.stat().st_size for f in DESTINO.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
    print(f"despliegue/ listo: modelo {info['modelo']} v{info['version']} | datos externos {versiones} | {tam / 1e6:.1f} MB")


if __name__ == "__main__":
    armar()
