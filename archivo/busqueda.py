"""Busqueda de hiperparametros a fondo (random search), optimizando DIRECTO la validacion temporal de 5 cortes.

Fase 1: RandomizedSearchCV por familia de modelos. El "cv" son los 5 cortes de tiempo (train <= corte, test = 2 anios
        siguientes), no un kfold revuelto: asi se premia lo que predice bien procesadores NUEVOS.
Fase 2: con el mejor de cada familia, Ensemble Selection de Caruana (2004) para escoger pesos de una mezcla.
Fase 3: confirmacion de los finalistas: kfold revuelto, corte 2019 y 3 semillas.

Todo queda en MLflow (experimento cpu-perf/busqueda): un run padre por familia y un hijo por configuracion probada.
Correr:  python -m archivo.busqueda            (todas)      python -m archivo.busqueda hgb mlp   (solo algunas)
"""
import json
import sys
import time
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from src import config, datos, modelos
from src.entrenar import metricas, preparar_mlflow

EXP = "cpu-perf/busqueda"
S = config.SEMILLA
VIDAS = [None, 2, 3, 5, 8]          # pesos por antiguedad: None = todos iguales


def envuelto(est, acotar=False):
    """log del objetivo + preprocesamiento (con o sin externos) + pesos por antiguedad (opcional) + modelo."""
    return modelos.con_log(modelos.ConPesos(est, acotar=acotar))


def p(d):
    """Prefijo de los nombres de parametros dentro de con_log(ConPesos(modelo))."""
    return {f"regressor__modelo__{k}": v for k, v in d.items()}


def familias():
    """nombre -> (modelo base, espacio de busqueda, n_iter). Espacios amplios a proposito."""
    v = {"regressor__vida_media": VIDAS, "regressor__externos": [True, False]}
    e = {"regressor__externos": [True, False]}
    return {
        "ridge": (envuelto(Ridge()), {**p({"alpha": loguniform(1e-2, 1e3)}), **v}, 25),
        "elasticnet": (envuelto(ElasticNet(max_iter=20000)),
                       {**p({"alpha": loguniform(1e-4, 1), "l1_ratio": uniform(0.05, 0.95)}), **v}, 25),
        "knn": (envuelto(KNeighborsRegressor()),
                {**p({"n_neighbors": randint(2, 40), "weights": ["uniform", "distance"], "p": [1, 2]}), **e}, 25),
        "svr": (envuelto(SVR()),
                {**p({"C": loguniform(0.1, 300), "epsilon": loguniform(0.005, 0.5), "gamma": loguniform(1e-3, 1)}), **v,
                 "regressor__acotar": [True, False]}, 30),
        "random_forest": (envuelto(RandomForestRegressor(random_state=S, n_jobs=1)),
                          {**p({"n_estimators": [100, 200, 400], "min_samples_leaf": randint(1, 12),
                                                         "max_features": [0.2, 0.33, 0.5, 0.7, 1.0], "max_depth": [None, 8, 15, 25]}), **v}, 30),
        "extra_trees": (envuelto(ExtraTreesRegressor(random_state=S, n_jobs=1)),
                        {**p({"n_estimators": [100, 200, 400], "min_samples_leaf": randint(1, 12),
                                                     "max_features": [0.2, 0.33, 0.5, 0.7, 1.0], "max_depth": [None, 8, 15, 25]}), **v}, 30),
        "hgb": (envuelto(HistGradientBoostingRegressor(random_state=S)),
                {**p({"learning_rate": loguniform(0.01, 0.3), "max_iter": [100, 200, 400, 800],
                                                       "max_leaf_nodes": randint(4, 64), "min_samples_leaf": randint(3, 80),
                                                       "l2_regularization": loguniform(1e-3, 30), "max_features": [0.3, 0.5, 0.8, 1.0]}), **v}, 40),
        "mlp": (envuelto(MLPRegressor(early_stopping=True, n_iter_no_change=20, max_iter=2000, random_state=S)),
                {**p({"hidden_layer_sizes": [(16,), (32,), (64,), (128,), (32, 16), (64, 32), (128, 64), (128, 64, 32)],
                                      "activation": ["relu", "tanh"], "alpha": loguniform(1e-3, 30),
                                      "learning_rate_init": loguniform(3e-4, 1e-2)}), **v, "regressor__acotar": [True, False]}, 40),
        "hibrido_bosque": (modelos.con_log(modelos.Hibrido()),
                           {"regressor__alpha": loguniform(0.1, 300), "regressor__vida_media": VIDAS,
                            "regressor__n_estimators": [100, 200, 400], "regressor__min_samples_leaf": randint(1, 12),
                            "regressor__max_features": [0.33, 0.5, 1.0], **e}, 35),
        "hibrido_hgb": (modelos.con_log(modelos.Hibrido(segundo="hgb")),
                        {"regressor__alpha": loguniform(0.1, 300), "regressor__vida_media": VIDAS,
                         "regressor__n_estimators": [100, 200, 400], "regressor__min_samples_leaf": randint(5, 40), **e}, 30),
        "hibrido_red": (modelos.con_log(modelos.Hibrido(segundo="red")),
                        {"regressor__alpha": loguniform(0.1, 300), "regressor__vida_media": VIDAS,
                         "regressor__capas": [(32,), (64,), (64, 32), (128, 64)], "regressor__alpha_red": loguniform(1e-3, 30), **e}, 30),
    }


def cortes_cv(X):
    """Los 5 cortes de tiempo como 'cv' de sklearn: lista de (indices train, indices test)."""
    anio = X["anio"].values
    return [(np.where(anio <= c)[0], np.where((anio > c) & (anio <= c + 2))[0]) for c in config.CORTES_TEMPORALES]


def limpio(v):
    return str(v) if isinstance(v, (tuple, list)) or v is None else v


def fase1(X, y, info, cuales):
    cv = cortes_cv(X)
    mejores = {}
    for nombre, (est, espacio, n_iter) in familias().items():
        if cuales and nombre not in cuales:
            continue
        t = time.time()
        busq = RandomizedSearchCV(est, espacio, n_iter=n_iter, cv=cv, scoring="neg_mean_absolute_error", n_jobs=6,
                                  random_state=S, error_score=np.nan, refit=False)
        with threadpool_limits(1):
            busq.fit(X, y)
        r = pd.DataFrame(busq.cv_results_)
        r["mae"] = -r["mean_test_score"]
        r["mae_std"] = r["std_test_score"]
        r = r.sort_values("mae")
        seg = time.time() - t
        with mlflow.start_run(run_name=f"busqueda__{nombre}__data-v{info['version']}",
                              tags={"etapa": "busqueda", "familia": nombre, "data_version": info["version"],
                                    "data_hash": info["hash"], "version_codigo": config.version_codigo(),
                                    "metodo": "RandomizedSearchCV", "cv": "5 cortes temporales"}):
            mlflow.log_metrics({"mejor_mae_5cortes": float(r["mae"].iloc[0]), "n_configuraciones": len(r), "segundos": seg})
            for i, fila in r.iterrows():
                params = {k.split("__")[-1]: limpio(v) for k, v in fila["params"].items()}
                with mlflow.start_run(run_name=f"{nombre}__config-{i:02d}", nested=True,
                                      tags={"familia": nombre, "etapa": "busqueda"}):
                    mlflow.log_params(params)
                    if np.isfinite(fila["mae"]):
                        mlflow.log_metrics({"mae_5cortes": float(fila["mae"]), "mae_5cortes_std": float(fila["mae_std"]),
                                            **{f"mae_corte_{c}": float(-fila[f"split{j}_test_score"])
                                               for j, c in enumerate(config.CORTES_TEMPORALES)}})
            r.drop(columns=["params"]).to_csv(config.REPORTES / f"busqueda_{nombre}.csv", index=False)
            mlflow.log_artifact(str(config.REPORTES / f"busqueda_{nombre}.csv"))
        mejores[nombre] = {"params": r["params"].iloc[0], "mae": float(r["mae"].iloc[0]), "mae_std": float(r["mae_std"].iloc[0]),
                           "peor_config": float(r["mae"].max()), "n": len(r), "segundos": seg}
        legible = {k.split("__")[-1]: limpio(v) for k, v in r["params"].iloc[0].items()}
        print(f"  {nombre:<16} {len(r):3d} configs {seg:5.0f}s | mejor MAE 5 cortes {r['mae'].iloc[0]:7.0f} ± {r['mae_std'].iloc[0]:5.0f} "
              f"| mediana {r['mae'].median():8.0f} | {legible}", flush=True)
    return mejores


def armar(nombre, params):
    est = clone(familias()[nombre][0])
    return est.set_params(**params)


def predicciones_cortes(est, X, y):
    """Predicciones de un modelo en los 5 cortes (para la mezcla)."""
    out = []
    for tr, te in cortes_cv(X):
        with threadpool_limits(2):
            out.append(clone(est).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te]))
    return out


def fase2(X, y, mejores, info, pasos=40):
    """Ensemble Selection de Caruana sobre el mejor de cada familia, optimizando el MAE de los 5 cortes (en log)."""
    cv = cortes_cv(X)
    nombres = list(mejores)
    P = {n: [np.log1p(np.clip(p, 0, None)) for p in predicciones_cortes(armar(n, mejores[n]["params"]), X, y)] for n in nombres}
    reales = [y.iloc[te].values for _, te in cv]

    def mae(pesos):
        tot = sum(pesos.values())
        return float(np.mean([np.mean(np.abs(np.expm1(sum(w * P[n][j] for n, w in pesos.items()) / tot) - reales[j]))
                              for j in range(len(cv))]))

    def seleccionar(cortes_uso):
        def m(pesos):
            tot = sum(pesos.values())
            return float(np.mean([np.mean(np.abs(np.expm1(sum(w * P[n][j] for n, w in pesos.items()) / tot) - reales[j]))
                                  for j in cortes_uso]))
        c = {min(nombres, key=lambda n: m({n: 1})): 1}
        for _ in range(pasos):
            k = min(nombres, key=lambda n: m({**c, n: c.get(n, 0) + 1}))
            c[k] = c.get(k, 0) + 1
        return c

    # evaluacion honesta: los pesos se escogen con 4 cortes y se mide en el que quedo fuera (rotando)
    honesto = []
    for j in range(len(cv)):
        c = seleccionar([i for i in range(len(cv)) if i != j])
        tot = sum(c.values())
        honesto.append(float(np.mean(np.abs(np.expm1(sum(w * P[n][j] for n, w in c.items()) / tot) - reales[j]))))
    mae_honesto = float(np.mean(honesto))

    ind = {n: mae({n: 1}) for n in nombres}
    primero = min(ind, key=ind.get)
    cuenta = {primero: 1}
    historia = [(primero, ind[primero])]
    for _ in range(pasos):
        cand = {n: mae({**cuenta, n: cuenta.get(n, 0) + 1}) for n in nombres}
        k = min(cand, key=cand.get)
        cuenta[k] = cuenta.get(k, 0) + 1
        historia.append((k, cand[k]))
    total = sum(cuenta.values())
    pesos = {n: c / total for n, c in sorted(cuenta.items(), key=lambda t: -t[1])}
    with mlflow.start_run(run_name=f"busqueda__mezcla_caruana__data-v{info['version']}",
                          tags={"etapa": "busqueda", "familia": "mezcla", "metodo": "Ensemble Selection (Caruana 2004)",
                                "data_version": info["version"], "version_codigo": config.version_codigo()}):
        mlflow.log_params({f"peso_{n}": round(w, 3) for n, w in pesos.items()})
        mlflow.log_metrics({"mae_5cortes_optimista": historia[-1][1], "mae_5cortes_honesto": mae_honesto,
                            "mejor_individual": ind[primero]})
    print(f"  individual: {', '.join(f'{n} {v:.0f}' for n, v in sorted(ind.items(), key=lambda t: t[1]))}")
    print(f"  mezcla Caruana: MAE 5 cortes {historia[-1][1]:.0f} optimista, {mae_honesto:.0f} honesto (pesos escogidos sin ver el corte) "
          f"| mejor individual {primero} {ind[primero]:.0f} | pesos "
          + ", ".join(f"{n} {w:.2f}" for n, w in pesos.items()), flush=True)
    return pesos, ind, mae_honesto


def fase3(X, y, finalistas, info):
    """Confirmacion: kfold revuelto (procesadores conocidos), corte 2019, 5 cortes y semillas."""
    kf = KFold(5, shuffle=True, random_state=S + 1)
    filas = []
    for nombre, est in finalistas.items():
        t = time.time()
        with threadpool_limits(2):
            k = metricas(y, cross_val_predict(est, X, y, cv=kf))
            viejo = X["anio"] <= config.ANIO_CORTE
            t19 = metricas(y[~viejo], clone(est).fit(X[viejo], y[viejo]).predict(X[~viejo]))
        cortes = []
        for sem in range(3):
            e = clone(est)
            for nom, val in e.get_params().items():          # cambia todas las semillas internas
                if nom.endswith("random_state") or nom.endswith("semilla"):
                    e.set_params(**{nom: S + sem})
            ps = predicciones_cortes(e, X, y)
            cortes.append(np.mean([metricas(y.iloc[te], p_)["mae"] for (_, te), p_ in zip(cortes_cv(X), ps)]))
        fila = {"modelo": nombre, "kfold_mae": k["mae"], "kfold_dentro20": k["kpi_dentro_20pct"], "corte2019_mae": t19["mae"],
                "corte2019_dentro20": t19["kpi_dentro_20pct"], "cortes5_mae": float(np.mean(cortes)),
                "cortes5_variacion_semillas": float(np.std(cortes)), "segundos": time.time() - t}
        filas.append(fila)
        with mlflow.start_run(run_name=f"confirmacion__{nombre}", tags={"etapa": "busqueda_confirmacion", "modelo": nombre,
                                                                        "data_version": info["version"]}):
            mlflow.log_metrics({k_: v for k_, v in fila.items() if k_ != "modelo"})
        print(f"  {nombre:<26} kfold {k['mae']:5.0f} ({k['kpi_dentro_20pct']:.0%}) | 2019 {t19['mae']:5.0f} ({t19['kpi_dentro_20pct']:.0%}) "
              f"| 5 cortes {np.mean(cortes):5.0f} ± {np.std(cortes):3.0f} (semillas)", flush=True)
    return pd.DataFrame(filas)


def correr(cuales=None):
    info = datos.descargar()
    df = datos.cargar(info["ruta"])
    modelos.usar_externos(True)
    X, _ = datos.preparar(df, externos=True)
    y = df[config.OBJETIVO]
    preparar_mlflow(EXP)
    config.REPORTES.mkdir(exist_ok=True)
    print(f"[fase 1] random search por familia, cv = 5 cortes temporales ({datetime.now():%H:%M})", flush=True)
    mejores = fase1(X, y, info, cuales)
    json.dump({n: {**m, "params": {k: limpio(v) for k, v in m["params"].items()}} for n, m in mejores.items()},
              open(config.REPORTES / "busqueda_mejores.json", "w"), indent=2, default=str)
    if cuales:
        return mejores
    print(f"[fase 2] mezcla de Caruana con el mejor de cada familia ({datetime.now():%H:%M})", flush=True)
    pesos, ind, mae_mezcla = fase2(X, y, mejores, info)
    print(f"[fase 3] confirmacion de finalistas ({datetime.now():%H:%M})", flush=True)
    top = sorted(ind, key=ind.get)[:4]
    finalistas = {f"{n} (mejor config)": armar(n, mejores[n]["params"]) for n in top}
    finalistas["mezcla Caruana"] = modelos.con_log(modelos.Mezcla(
        modelos=[armar(n, mejores[n]["params"]).regressor for n in pesos], pesos=list(pesos.values())))
    finalistas["hibrido_combo (produccion v4)"] = modelos.con_log(modelos.Combo())
    conf = fase3(X, y, finalistas, info)
    conf.to_csv(config.REPORTES / "busqueda_confirmacion.csv", index=False)
    json.dump({"pesos_mezcla": pesos, "mae_mezcla_5cortes": mae_mezcla, "individual_5cortes": ind},
              open(config.REPORTES / "busqueda_mezcla.json", "w"), indent=2)
    print(f"listo ({datetime.now():%H:%M})")
    return mejores, pesos, conf


if __name__ == "__main__":
    correr(sys.argv[1:] or None)
