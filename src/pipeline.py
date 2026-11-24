"""Pipeline automatico: datos frescos -> calidad -> entrenamiento -> modelo cruzado -> registro -> drift -> batch.

Que se reentrena depende de que datos cambiaron:
  - catalogo multi-core / single-core: si cambio PassMark (hash) o pc_build (sha)
  - modelo cruzado (el tipo del v7):    si cambio PassMark, pc_build o el snapshot de Blender (se renueva cada 7 dias)
Cualquier modelo nuevo solo pasa a 'produccion' si le gana al actual en el promedio de los 5 cortes de tiempo.

Correr (desde la carpeta Proj):
    python -m src.pipeline                  solo reentrena lo que tenga datos nuevos
    python -m src.pipeline --forzar         reentrena todo aunque los datos sean los mismos
    python -m src.pipeline --sin-cruzado    omite el modelo cruzado (mas rapido, no usa Blender)
"""
import argparse
import json
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

from . import batch, calidad, config, cruzado, datos, drift, entrenar, externos, modelos


def ya_entrenado(hash_datos, version_pc_build=None):
    """Busca en MLflow (SQLite) si ya hay un entrenamiento con estos mismos datos."""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    if mlflow.get_experiment_by_name(config.EXP_MULTI) is None:
        return False
    runs = mlflow.search_runs(experiment_names=[config.EXP_MULTI],
                              filter_string=f"tags.data_hash = '{hash_datos}' and tags.etapa = 'entrenamiento'"
                              + (f" and tags.version_pc_build = '{version_pc_build}'" if version_pc_build else ""))
    return len(runs) > 0


def guardar(resumen):
    """Siempre se escribe (tambien si no hubo cambios): GitHub Actions lo lee para saber si actualizar la pagina."""
    config.REPORTES.mkdir(exist_ok=True)
    resumen.setdefault("cambio_produccion", False)
    (config.REPORTES / "ultimo_pipeline.json").write_text(json.dumps(resumen, indent=2, default=str))
    return resumen


def produccion():
    mv = MlflowClient().get_model_version_by_alias(config.MODELO_REGISTRADO, config.ALIAS_PRODUCCION)
    return int(mv.version), mv.tags.get("modelo", "")


def correr(forzar=False, sin_cruzado=False):
    inicio = datetime.now()
    mlflow.set_tracking_uri(config.TRACKING_URI)
    print(f"[1/7] datos frescos: Kaggle ({config.DATASET_KAGGLE}), pc_build (Hugging Face), Blender Open Data")
    info = datos.descargar()
    print(f"      PassMark version {info['version']}, hash {info['hash']}")
    _, v_pcb = externos.pc_build()              # revisa si hay version nueva de pc_build en Hugging Face
    print(f"      pc_build version {v_pcb}")
    snap = None if sin_cruzado else cruzado.snapshot_blender()
    if snap:
        print(f"      Blender snapshot {snap}")
    nuevo_catalogo = forzar or not ya_entrenado(info["hash"], v_pcb)
    nuevo_cruzado = not sin_cruzado and (forzar or not cruzado.ya_corrido(info["hash"], v_pcb, snap))
    v_antes, _ = produccion()
    if not (nuevo_catalogo or nuevo_cruzado):
        print(f"      sin datos nuevos: no se reentrena (produccion sigue en v{v_antes})")
        return guardar({"estado": "sin_cambios", **info, "blender_snapshot": snap, "produccion": v_antes,
                        "fecha": inicio.isoformat(timespec="seconds")})

    print("[2/7] calidad de datos")
    df = datos.cargar(info["ruta"])
    rep = calidad.validar(df)
    for k, v in calidad.resumen(rep).items():
        print(f"      {k}: {v}")
    config.REPORTES.mkdir(exist_ok=True)
    (config.REPORTES / "calidad_datos.json").write_text(json.dumps(calidad.resumen(rep), indent=2, ensure_ascii=False))
    if not calidad.pasa(rep):
        print(f"      FALLA un chequeo critico, se detiene el pipeline (produccion sigue en v{v_antes})")
        return guardar({"estado": "falla_calidad", **info, "produccion": v_antes, "fecha": inicio.isoformat(timespec="seconds")})

    modelos.usar_externos(True)
    X, d = datos.preparar(df, externos=True)                 # + cache, litografia, frecuencias, hilos (TechPowerUp / pc_build)
    _, cobertura = externos.unir(df)
    info |= {"version_techpowerup": cobertura["version_tpu"], "version_pc_build": cobertura["version_pc_build"],
             "cobertura_externos": f"{cobertura['cualquiera']:.0%}"}
    print(f"      datos externos: {info['cobertura_externos']} de los procesadores emparejados")

    resumen = {"estado": "ok", **info, "blender_snapshot": snap, "produccion_antes": v_antes}
    if nuevo_catalogo:
        print("[3/7] entrenando el catalogo de modelos multi-core")
        tabla, _ = entrenar.entrenar(X, df[config.OBJETIVO], info)
        with mlflow.start_run(run_id=mlflow.search_runs(experiment_names=[config.EXP_MULTI], max_results=1,
                                                         filter_string="tags.etapa = 'entrenamiento'").iloc[0].run_id):
            mlflow.log_artifact(str(config.REPORTES / "calidad_datos.json"), "calidad")
        resumen |= {"catalogo_mejor_modelo": tabla.iloc[0]["modelo"], "catalogo_mejor_mae": float(tabla.iloc[0]["mae"])}
        print("[4/7] experimento single-core (pregunta secundaria)")
        entrenar.entrenar(X, df[config.OBJETIVO_SINGLE], info, objetivo=config.OBJETIVO_SINGLE,
                          experimento=config.EXP_SINGLE, solo=["ridge", "random_forest", "hgb"], registrar=False)
    else:
        print("[3/7] catalogo multi-core: PassMark y pc_build sin cambios, no se reentrena")
        print("[4/7] single-core: sin cambios")

    if nuevo_cruzado:
        print("[5/7] modelo cruzado PassMark + Blender (el tipo del modelo en produccion)")
        resumen["cruzado"] = cruzado.correr(info)
    else:
        print("[5/7] modelo cruzado: " + ("omitido (--sin-cruzado)" if sin_cruzado else "sin datos nuevos de Blender, no se reentrena"))

    print("[6/7] drift: procesadores viejos vs nuevos (+ Evidently)")
    _, res = drift.reporte_drift(X, df[config.OBJETIVO], "hgb", info)
    print(f"      variables con drift fuerte: {res['variables_con_drift_fuerte']}")
    print(f"      concept drift: MAE viejos {res['concept_drift']['mae_viejos']:.0f} -> nuevos {res['concept_drift']['mae_nuevos']:.0f}")

    print("[7/7] prediccion batch de los procesadores mas recientes (con el modelo en produccion)")
    recientes = df[df.testDate == df.testDate.max()]
    entrada = config.DATOS / "batch_entrada.csv"
    recientes.to_csv(entrada, index=False)
    batch.predecir_lote(entrada, config.DATOS / "batch_salida.csv")

    v_despues, desc = produccion()
    resumen |= {"produccion": v_despues, "produccion_modelo": desc, "cambio_produccion": v_despues != v_antes,
                "duracion_seg": (datetime.now() - inicio).total_seconds(), "fecha": inicio.isoformat(timespec="seconds")}
    guardar(resumen)
    print("listo:", json.dumps(resumen, default=str))
    return resumen


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--forzar", action="store_true", help="reentrenar aunque los datos no hayan cambiado")
    p.add_argument("--sin-cruzado", action="store_true", help="omitir el modelo cruzado (no usa Blender)")
    a = p.parse_args()
    correr(a.forzar, a.sin_cruzado)
