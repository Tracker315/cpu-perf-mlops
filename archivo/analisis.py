"""Responde las preguntas del proyecto y lo deja en MLflow (experimento cpu-perf/analisis).

1. Amdahl: el rendimiento multi-core crece menos que proporcional a los nucleos?
2. Nucleos vs frecuencia: que conviene mas subir (elasticidades con Cinebench R23, que trae frecuencias)
3. Single-core vs multi-core: dependen de las mismas especificaciones?
4. Procesadores que rinden mas / menos de lo que sus especificaciones sugieren
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from src import config, datos, modelos
from src.entrenar import preparar_mlflow

EXP = "cpu-perf/analisis"


def amdahl(X, y):
    """Pendiente de log(cpuMark) contra log(cores) por segmento; < 1 = crece menos que proporcional."""
    filas = []
    for seg, mascara in [("todos", X.index == X.index), ("desktop", X.es_desktop == 1), ("laptop", X.es_laptop == 1),
                         ("server", X.es_server == 1)]:
        lc, ly = np.log(X.loc[mascara, "cores"]), np.log(y[mascara])
        b1 = np.polyfit(lc, ly, 1)[0]
        b2 = np.polyfit(lc, ly, 2)                           # termino cuadratico: < 0 = la curva se aplana
        filas.append({"segmento": seg, "n": int(mascara.sum()), "elasticidad_cores": b1, "curvatura": b2[0]})
    # controlando por anio (procesadores con mas nucleos tambien suelen ser mas nuevos)
    Z = pd.DataFrame({"log_cores": np.log(X.cores), "anio": X.anio}).dropna()
    reg = LinearRegression().fit(Z, np.log(y[Z.index]))
    filas.append({"segmento": "todos (controlando anio)", "n": len(Z), "elasticidad_cores": reg.coef_[0], "curvatura": np.nan})
    t = pd.DataFrame(filas)
    plt.figure(figsize=(6, 4))
    med = y.groupby(X.cores).median()
    med = med[y.groupby(X.cores).size() >= 5]
    plt.loglog(med.index, med.values, "o", label="mediana por # de cores")
    xs = np.array([1, 64])
    a = np.polyfit(np.log(med.index), np.log(med.values), 1)
    plt.loglog(xs, np.exp(a[1]) * xs ** a[0], "--", label=f"ajuste: pendiente {a[0]:.2f}")
    plt.loglog(xs, med.values[0] * xs, ":", color="gray", label="escalamiento perfecto (pendiente 1)")
    plt.xlabel("nucleos"); plt.ylabel("cpuMark mediano"); plt.legend(); plt.title("Amdahl: rendimiento vs nucleos")
    plt.tight_layout(); plt.savefig(config.REPORTES / "amdahl.png", dpi=110); plt.close()
    return t


def nucleos_vs_frecuencia():
    """Con Cinebench R23 (215 CPUs con base/turbo clock): log(multiScore) ~ log(cores) + log(turbo) + log(hilos/cores)."""
    import kagglehub
    r23 = pd.read_csv(kagglehub.dataset_download(config.DATASET_KAGGLE) + "/CPU_r23_v2.csv")
    Z = pd.DataFrame({"log_cores": np.log(r23.cores), "log_turbo": np.log(r23.turboClock),
                      "log_hilos_por_core": np.log(r23.threads / r23.cores)})
    res = {}
    for obj in ["multiScore", "singleScore"]:
        reg = LinearRegression().fit(Z, np.log(r23[obj]))
        res[obj] = dict(zip(Z.columns, reg.coef_.round(3)), r2=round(reg.score(Z, np.log(r23[obj])), 3))
    return res


def single_vs_multi(X, df):
    """Importancia por permutacion de las mismas especificaciones para predecir multi-core y single-core."""
    imp = {}
    for obj in [config.OBJETIVO, config.OBJETIVO_SINGLE]:
        m = modelos.con_log(make_pipeline(modelos.preprocesamiento(), RandomForestRegressor(n_estimators=200, random_state=config.SEMILLA, n_jobs=-1)))
        with threadpool_limits(2):
            m.fit(X, df[obj])
            r = permutation_importance(m, X, df[obj], scoring="r2", n_repeats=3, random_state=config.SEMILLA)
        s = pd.Series(np.clip(r.importances_mean, 0, None), index=X.columns)
        imp[obj] = s / s.sum()
    t = pd.DataFrame(imp).sort_values(config.OBJETIVO, ascending=True)
    t.plot.barh(figsize=(7, 6)); plt.xlabel("importancia relativa (permutacion)"); plt.title("que pesa en multi-core vs single-core")
    plt.tight_layout(); plt.savefig(config.REPORTES / "single_vs_multi.png", dpi=110); plt.close()
    return t, float(np.corrcoef(np.log(df.cpuMark), np.log(df.threadMark))[0, 1])


def sobre_y_bajo(X, y, nombres):
    """Residuales fuera de muestra: quien rinde mas (o menos) de lo que sus especificaciones sugieren."""
    m = modelos.con_log(make_pipeline(modelos.preprocesamiento(), RandomForestRegressor(n_estimators=200, random_state=config.SEMILLA, n_jobs=-1)))
    with threadpool_limits(2):
        pred = cross_val_predict(m, X, y, cv=KFold(5, shuffle=True, random_state=config.SEMILLA))
    r = pd.DataFrame({"cpu": nombres, "real": y.values, "esperado": pred.round(), "ratio": (y.values / pred).round(2)})
    return r.sort_values("ratio", ascending=False)


def correr():
    info = datos.descargar()
    df = datos.cargar(info["ruta"])
    modelos.usar_externos(True)
    X, _ = datos.preparar(df, externos=True)
    y = df[config.OBJETIVO]
    preparar_mlflow(EXP)
    config.REPORTES.mkdir(exist_ok=True)
    with mlflow.start_run(run_name=f"analisis__data-v{info['version']}", tags={"data_version": info["version"], "etapa": "analisis"}):
        t_am = amdahl(X, y); t_am.to_csv(config.REPORTES / "amdahl.csv", index=False)
        print("Amdahl:\n", t_am.round(3).to_string(index=False))
        nf = nucleos_vs_frecuencia()
        (config.REPORTES / "nucleos_vs_frecuencia.json").write_text(json.dumps(nf, indent=2))
        print("Nucleos vs frecuencia (elasticidades, Cinebench R23):", nf)
        t_sm, corr = single_vs_multi(X, df)
        t_sm.to_csv(config.REPORTES / "single_vs_multi.csv")
        print(f"Single vs multi (correlacion en log = {corr:.2f}):\n", t_sm.round(3).sort_values(config.OBJETIVO, ascending=False).head(8).to_string())
        sb = sobre_y_bajo(X, y, df.cpuName)
        sb.to_csv(config.REPORTES / "sobre_y_bajo_rendimiento.csv", index=False)
        print("Rinden MAS de lo esperado:\n", sb.head(8).to_string(index=False))
        print("Rinden MENOS de lo esperado:\n", sb.tail(8).to_string(index=False))
        mlflow.log_metrics({"elasticidad_cores": float(t_am.iloc[0].elasticidad_cores), "corr_single_multi_log": corr,
                            "elasticidad_cores_r23": nf["multiScore"]["log_cores"], "elasticidad_turbo_r23": nf["multiScore"]["log_turbo"]})
        for f in ["amdahl.png", "amdahl.csv", "nucleos_vs_frecuencia.json", "single_vs_multi.png", "single_vs_multi.csv", "sobre_y_bajo_rendimiento.csv"]:
            mlflow.log_artifact(str(config.REPORTES / f), "analisis")


if __name__ == "__main__":
    correr()
