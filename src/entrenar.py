"""Entrena y compara modelos, registra todo en MLflow y promueve el mejor."""
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from threadpoolctl import threadpool_limits

from . import config, modelos


def metricas(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    error_rel = np.abs(pred - y) / y
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": mean_squared_error(y, pred) ** 0.5,
        "r2": r2_score(y, pred),
        "r2_log": r2_score(np.log1p(y), np.log1p(np.clip(pred, 0, None))),
        "mape": float(np.mean(error_rel)),
        "kpi_dentro_20pct": float(np.mean(error_rel <= 0.20)),   # KPI de negocio: % de procesadores con error <= 20%
    }


def preparar_mlflow(experimento):
    mlflow.set_tracking_uri(config.TRACKING_URI)
    if mlflow.get_experiment_by_name(experimento) is None:
        mlflow.create_experiment(experimento, artifact_location=config.ARTIFACTS.as_uri() + "/" + experimento.replace("/", "_"))
    mlflow.set_experiment(experimento)


def grafica_pred_real(y, pred, titulo, ruta):
    plt.figure(figsize=(5, 5))
    plt.scatter(y, pred, s=5, alpha=0.4)
    lim = [max(1, min(y.min(), pred.min())), max(y.max(), pred.max())]
    plt.plot(lim, lim, "r--")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("real"); plt.ylabel("predicho"); plt.title(titulo)
    plt.tight_layout(); plt.savefig(ruta, dpi=110); plt.close()


def entrenar(X, y, info_datos, objetivo=config.OBJETIVO, experimento=config.EXP_MULTI, solo=None, registrar=True):
    """Corre todos los modelos del catalogo (o los de 'solo'). Regresa (tabla de resultados, run_id del mejor)."""
    preparar_mlflow(experimento)
    config.REPORTES.mkdir(exist_ok=True)
    kf_busqueda = KFold(5, shuffle=True, random_state=config.SEMILLA)       # revuelto: el csv viene ordenado
    kf_eval = KFold(5, shuffle=True, random_state=config.SEMILLA + 1)
    tags_datos = {"data_version": info_datos["version"], "data_hash": info_datos["hash"], "objetivo": objetivo,
                  "target_transform": "log1p", "autor": "Elias Rodriguez",
                  "version_codigo": config.version_codigo(),
                  **{k: str(v) for k, v in info_datos.items() if k.startswith(("version_", "cobertura"))}}
    filas = []
    nombre_padre = f"entrenamiento__data-v{info_datos['version']}__{datetime.now():%Y%m%d-%H%M}"
    with mlflow.start_run(run_name=nombre_padre, tags={**tags_datos, "etapa": "entrenamiento"}):
        for nombre, (familia, modelo, rejilla) in modelos.catalogo().items():
            if solo and nombre not in solo:
                continue
            with mlflow.start_run(run_name=f"{nombre}__data-v{info_datos['version']}", nested=True,
                                  tags={**tags_datos, "modelo": nombre, "familia_modelo": familia, "split": "kfold5"}) as run:
                t = time.time()
                # dataset con su huella: aparece en la pestana "Datasets" de la UI de MLflow
                mlflow.log_input(mlflow.data.from_pandas(X.assign(cpuMark=np.asarray(y)), name=f"passmark-v{info_datos['version']}",
                                                         targets="cpuMark"), context="entrenamiento")
                with threadpool_limits(2):
                    if rejilla:
                        busqueda = GridSearchCV(modelo, rejilla, cv=kf_busqueda, scoring="neg_mean_absolute_error", n_jobs=1)
                        busqueda.fit(X, y)
                        mejor, params = busqueda.best_estimator_, busqueda.best_params_
                    else:
                        mejor, params = modelo, {}
                    pred = cross_val_predict(mejor, X, y, cv=kf_eval)          # predicciones fuera de muestra
                    viejo = X["anio"] <= config.ANIO_CORTE                        # validacion temporal: pasado -> futuro
                    temporal = clone(mejor).fit(X[viejo], y[viejo])
                    m_temp = metricas(y[~viejo], temporal.predict(X[~viejo]))
                    multi = []
                    for corte in config.CORTES_TEMPORALES:      # validacion temporal con varios cortes
                        tr, te = X["anio"] <= corte, (X["anio"] > corte) & (X["anio"] <= corte + 2)
                        multi.append(metricas(y[te], clone(mejor).fit(X[tr], y[tr]).predict(X[te])))
                    mejor.fit(X, y)
                m = metricas(y, pred) | {f"temporal_{k}": v for k, v in m_temp.items()}
                m["temporal_multi_mae"] = float(np.mean([r["mae"] for r in multi]))
                m["temporal_multi_mae_std"] = float(np.std([r["mae"] for r in multi]))
                m["temporal_multi_kpi_dentro_20pct"] = float(np.mean([r["kpi_dentro_20pct"] for r in multi]))
                m["segundos_entrenamiento"] = time.time() - t
                mlflow.log_params({k.split("__")[-1]: v for k, v in params.items()} | {"n_filas": len(X), "n_columnas": X.shape[1]})
                mlflow.log_metrics(m)
                ruta = config.REPORTES / f"pred_vs_real_{nombre}.png"
                grafica_pred_real(np.asarray(y), pred, f"{nombre} (fuera de muestra)", ruta)
                mlflow.log_artifact(str(ruta), "graficas")
                # cloudpickle + el codigo de src/ empaquetado: el modelo trae las funciones propias que usa
                info_modelo = mlflow.sklearn.log_model(mejor, name="modelo", input_example=X.head(3),
                                                       signature=infer_signature(X.head(50), mejor.predict(X.head(50))),
                                                       serialization_format="cloudpickle",
                                                       code_paths=[str(config.RAIZ / "src")])
                filas.append({"modelo": nombre, "familia": familia, **m, "run_id": run.info.run_id,
                              "model_uri": info_modelo.model_uri, "estimador": mejor})
                print(f"  {nombre:<20} kfold MAE {m['mae']:6.0f} dentro20 {m['kpi_dentro_20pct']:.1%} | temporal MAE {m['temporal_mae']:6.0f} dentro20 {m['temporal_kpi_dentro_20pct']:.1%} | 5 cortes MAE {m['temporal_multi_mae']:6.0f} \u00b1 {m['temporal_multi_mae_std']:4.0f}", flush=True)

        tabla = pd.DataFrame(filas).sort_values(config.METRICA_SELECCION)
        ganador = tabla.iloc[0]
        mlflow.log_metric("mejor_mae", ganador["mae"])
        mlflow.log_metric("mejor_" + config.METRICA_SELECCION, ganador[config.METRICA_SELECCION])
        mlflow.set_tag("metrica_seleccion", config.METRICA_SELECCION)
        mlflow.set_tag("mejor_modelo", ganador["modelo"])
        tabla.drop(columns=["estimador"]).to_csv(config.REPORTES / "comparacion_modelos.csv", index=False)
        mlflow.log_artifact(str(config.REPORTES / "comparacion_modelos.csv"))
        interpretar(tabla, X, y)

    if registrar:
        promover(ganador, tags_datos)
    return tabla.drop(columns=["estimador"]), ganador["run_id"]


def interpretar(tabla, X, y):
    """Que especificaciones pesan mas: importancia por permutacion del mejor modelo y coeficientes de Ridge."""
    mejor = tabla.iloc[0]["estimador"]
    with threadpool_limits(2):
        imp = permutation_importance(mejor, X, y, scoring="neg_mean_absolute_error", n_repeats=5, random_state=config.SEMILLA)
    imp = pd.Series(imp.importances_mean, index=X.columns).sort_values()
    imp.to_csv(config.REPORTES / "importancia_permutacion.csv")
    plt.figure(figsize=(6, 4)); imp.plot.barh(); plt.title(f"importancia por permutacion ({tabla.iloc[0]['modelo']})")
    plt.xlabel("aumento del MAE al revolver la columna"); plt.tight_layout()
    plt.savefig(config.REPORTES / "importancia_permutacion.png", dpi=110); plt.close()
    mlflow.log_artifact(str(config.REPORTES / "importancia_permutacion.png"), "interpretacion")
    mlflow.log_artifact(str(config.REPORTES / "importancia_permutacion.csv"), "interpretacion")
    if "ridge" in set(tabla["modelo"]):
        ridge = tabla.set_index("modelo").loc["ridge", "estimador"].regressor_
        nombres = ridge[0].get_feature_names_out()
        coef = pd.Series(ridge[-1].coef_, index=nombres).sort_values()
        coef.to_csv(config.REPORTES / "coeficientes_ridge.csv")
        mlflow.log_artifact(str(config.REPORTES / "coeficientes_ridge.csv"), "interpretacion")


def promover(ganador, tags_datos):
    """Registra el ganador; le pone el alias 'produccion' solo si mejora el MAE del modelo actual en produccion."""
    cliente = MlflowClient()
    version = mlflow.register_model(ganador["model_uri"], config.MODELO_REGISTRADO)
    met = config.METRICA_SELECCION
    for k, v in {**tags_datos, "modelo": ganador["modelo"], "mae": f"{ganador['mae']:.1f}", met: f"{ganador[met]:.1f}"}.items():
        cliente.set_model_version_tag(config.MODELO_REGISTRADO, version.version, k, str(v))
    try:
        actual = cliente.get_model_version_by_alias(config.MODELO_REGISTRADO, config.ALIAS_PRODUCCION)
        mae_actual = float(actual.tags.get(met, "inf"))      # se compara con la MISMA metrica
    except Exception:
        actual, mae_actual = None, float("inf")
    if ganador[met] < mae_actual * (1 - config.MEJORA_MINIMA):     # margen: diferencias de <1% son ruido
        cliente.set_registered_model_alias(config.MODELO_REGISTRADO, config.ALIAS_PRODUCCION, version.version)
        print(f"  -> version {version.version} ({ganador['modelo']}) ahora es '{config.ALIAS_PRODUCCION}' ({met} {ganador[met]:.0f} vs {mae_actual:.0f})")
    else:
        print(f"  -> version {version.version} registrada, pero no mejora al menos {config.MEJORA_MINIMA:.0%} a produccion "
              f"(v{actual.version}): {ganador[met]:.0f} vs {mae_actual:.0f}")
    return version.version
