"""Simulacion del ciclo MLOps en el tiempo: cada anio "llega" una version nueva del dataset (los procesadores
probados ese anio). El modelo en produccion se monitorea con los datos nuevos y se reentrena si hace falta.

Correr:  python -m archivo.simulacion
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from src import config, datos, modelos
from src.drift import psi, psi_categorico
from src.entrenar import metricas, preparar_mlflow

EXP = "cpu-perf/simulacion"
MODELO = "cpu-perf-simulacion"
ANIO_INICIAL = 2013
UMBRAL_KPI = 0.10            # reentrenar si el % dentro de 20% cae mas de 10 puntos contra lo esperado
UMBRAL_PSI = 0.25            # o si alguna entrada tiene drift fuerte contra la ventana reciente del entrenamiento


def modelo_base():
    return modelos.con_log(modelos.Hibrido(vida_media=5))     # el mismo tipo de modelo que gano para produccion


def ajustar(X, y):
    m = clone(modelo_base())
    with threadpool_limits(2):
        m.fit(X, y)
    return m


def kpi_esperado(X, y):
    """Lo que se espera del modelo en el anio que sigue: entrena sin el ultimo anio y evalua en ese anio."""
    ultimo = X["anio"].max()
    viejo = X["anio"] < ultimo
    if viejo.sum() < 100:
        viejo = X["anio"] < ultimo - 1
    m = ajustar(X[viejo], y[viejo])
    return metricas(y[~viejo], m.predict(X[~viejo]))["kpi_dentro_20pct"]


def entrenar_y_registrar(X, y, anio, cliente, razon):
    """Entrena con los datos acumulados, registra version y la promueve. Regresa (modelo, version, KPI esperado)."""
    m = ajustar(X, y)
    kpi_ref = kpi_esperado(X, y)
    mlflow.log_metric("kpi_esperado", kpi_ref)
    info = mlflow.sklearn.log_model(m, name="modelo", serialization_format="cloudpickle", code_paths=[str(config.RAIZ / "src")])
    v = mlflow.register_model(info.model_uri, MODELO)
    for k, val in {"datos_hasta": anio, "razon": razon, "kpi_esperado": f"{kpi_ref:.3f}"}.items():
        cliente.set_model_version_tag(MODELO, v.version, k, str(val))
    cliente.set_registered_model_alias(MODELO, config.ALIAS_PRODUCCION, v.version)
    return m, v.version, kpi_ref


def correr():
    info = datos.descargar()
    df = datos.cargar(info["ruta"])
    modelos.usar_externos(True)
    X, _ = datos.preparar(df, externos=True)
    y = df[config.OBJETIVO]
    preparar_mlflow(EXP)
    cliente = MlflowClient()
    config.REPORTES.mkdir(exist_ok=True)

    hist = X["anio"] <= ANIO_INICIAL
    filas = []
    with mlflow.start_run(run_name=f"simulacion__{ANIO_INICIAL}-2022__data-v{info['version']}",
                          tags={"data_version": info["version"], "etapa": "simulacion"}):
        with mlflow.start_run(run_name=f"release-{ANIO_INICIAL}", nested=True, tags={"data_version": f"sim-{ANIO_INICIAL}"}):
            prod, version, kpi_ref = entrenar_y_registrar(X[hist], y[hist], ANIO_INICIAL, cliente, "inicial")
        anio_prod = ANIO_INICIAL                                 # hasta que anio vio datos el modelo en produccion
        congelado = prod                                         # estrategia 1: nunca se reentrena
        anual = prod                                             # estrategia 2: se reentrena cada anio, sin ver nada
        for anio in range(ANIO_INICIAL + 1, 2023):
            nuevos = X["anio"] == anio
            if nuevos.sum() == 0:
                continue
            Xn, yn = X[nuevos], y[nuevos]
            with mlflow.start_run(run_name=f"release-{anio}", nested=True,
                                  tags={"data_version": f"sim-{anio}", "modelo_produccion": str(version)}):
                # 1) monitoreo: desempeno real de cada estrategia con los procesadores nuevos (cuando ya hay benchmark)
                m_prod = metricas(yn, prod.predict(Xn))
                m_cong = metricas(yn, congelado.predict(Xn))
                m_anual = metricas(yn, anual.predict(Xn))
                # 2) drift contra la ventana reciente que vio el modelo en produccion (sus 2 ultimos anios)
                ref = hist & (X["anio"] >= anio_prod - 1)
                psi_in = {c: psi(X.loc[ref, c], Xn[c]) for c in ["cores", "TDP"]}
                psi_in["fabricante"] = psi_categorico(X.loc[ref, "fabricante"], Xn["fabricante"])
                psi_label = psi(np.log1p(y[ref]), np.log1p(yn))
                # 3) politica de reentrenamiento por alerta
                caida = kpi_ref - m_prod["kpi_dentro_20pct"]
                razones = []
                if caida > UMBRAL_KPI:
                    razones.append(f"KPI cae {caida:.0%} pts")
                if max(psi_in.values()) > UMBRAL_PSI:
                    razones.append(f"data drift ({max(psi_in, key=psi_in.get)} {max(psi_in.values()):.2f})")
                mlflow.log_metrics({"prod_mae_nuevos": m_prod["mae"], "prod_dentro20_nuevos": m_prod["kpi_dentro_20pct"],
                                    "prod_r2log_nuevos": m_prod["r2_log"], "congelado_mae_nuevos": m_cong["mae"],
                                    "congelado_dentro20_nuevos": m_cong["kpi_dentro_20pct"], "anual_mae_nuevos": m_anual["mae"],
                                    "anual_dentro20_nuevos": m_anual["kpi_dentro_20pct"], "psi_label": psi_label,
                                    "caida_kpi": caida, "kpi_esperado": kpi_ref, "n_nuevos": int(nuevos.sum()),
                                    **{f"psi_{k}": v for k, v in psi_in.items()}})
                hist = hist | nuevos
                anual = ajustar(X[hist], y[hist])
                if razones:
                    mlflow.set_tag("decision", "reentrenar: " + ", ".join(razones))
                    prod, version, kpi_ref = entrenar_y_registrar(X[hist], y[hist], anio, cliente, ", ".join(razones))
                    anio_prod = anio
                else:
                    mlflow.set_tag("decision", "mantener")
                filas.append({"anio": anio, "n_nuevos": int(nuevos.sum()), "mae_alerta": m_prod["mae"],
                              "dentro20_alerta": m_prod["kpi_dentro_20pct"], "kpi_esperado": kpi_ref if not razones else None,
                              "mae_anual": m_anual["mae"], "dentro20_anual": m_anual["kpi_dentro_20pct"],
                              "mae_congelado": m_cong["mae"], "dentro20_congelado": m_cong["kpi_dentro_20pct"],
                              "psi_cores": psi_in["cores"], "psi_TDP": psi_in["TDP"], "psi_fabricante": psi_in["fabricante"],
                              "psi_label": psi_label, "caida_kpi": caida,
                              "decision": "reentrenar" if razones else "mantener", "razon": ", ".join(razones),
                              "version_despues": version})
                print(f"  {anio}: {nuevos.sum():4d} nuevos | dentro20 alerta {m_prod['kpi_dentro_20pct']:.0%} anual {m_anual['kpi_dentro_20pct']:.0%} "
                      f"congelado {m_cong['kpi_dentro_20pct']:.0%} | MAE {m_prod['mae']:5.0f} {m_anual['mae']:5.0f} {m_cong['mae']:5.0f} | "
                      f"{('REENTRENA: ' + ', '.join(razones)) if razones else 'mantiene'}", flush=True)
        tabla = pd.DataFrame(filas)
        tabla.to_csv(config.REPORTES / "simulacion.csv", index=False)
        grafica(tabla)
        mlflow.log_artifact(str(config.REPORTES / "simulacion.csv"))
        mlflow.log_artifact(str(config.REPORTES / "simulacion.png"))
        resumen = {"dentro20_promedio_alerta": tabla.dentro20_alerta.mean(), "dentro20_promedio_anual": tabla.dentro20_anual.mean(),
                   "dentro20_promedio_congelado": tabla.dentro20_congelado.mean(), "mae_promedio_alerta": tabla.mae_alerta.mean(),
                   "mae_promedio_anual": tabla.mae_anual.mean(), "mae_promedio_congelado": tabla.mae_congelado.mean(),
                   "reentrenamientos_alerta": int((tabla.decision == "reentrenar").sum()), "reentrenamientos_anual": len(tabla)}
        mlflow.log_metrics(resumen)
        json.dump(resumen, open(config.REPORTES / "simulacion_resumen.json", "w"), indent=2)
        print(json.dumps(resumen, indent=2))
    return tabla


def grafica(t):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(t.anio, t.dentro20_congelado, "o--", label=f"congelado ({ANIO_INICIAL})")
    ax[0].plot(t.anio, t.dentro20_anual, "s-", alpha=0.6, label="reentrena cada anio")
    ax[0].plot(t.anio, t.dentro20_alerta, "o-", label="reentrena por alerta")
    for _, f in t[t.decision == "reentrenar"].iterrows():
        ax[0].axvline(f.anio, color="gray", alpha=0.2)
    ax[0].set_ylabel("% de procesadores nuevos con error <= 20%"); ax[0].set_title("KPI en produccion"); ax[0].legend()
    ax[1].plot(t.anio, t.psi_cores, "o-", label="PSI cores")
    ax[1].plot(t.anio, t.psi_TDP, "o-", label="PSI TDP")
    ax[1].plot(t.anio, t.psi_fabricante, "o-", label="PSI fabricante")
    ax[1].plot(t.anio, t.psi_label, "o-", label="PSI cpuMark (label)")
    ax[1].axhline(UMBRAL_PSI, color="red", ls="--"); ax[1].set_title("drift contra la ventana reciente"); ax[1].legend()
    plt.tight_layout(); plt.savefig(config.REPORTES / "simulacion.png", dpi=110); plt.close()


if __name__ == "__main__":
    correr()
