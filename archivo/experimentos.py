"""Experimentos A vs B. Cada experimento es un run padre en MLflow y cada variante un run hijo.

Correr:  python -m archivo.experimentos            (todos)
         python -m archivo.experimentos e5 e7      (solo algunos)
"""
import sys
import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, TargetEncoder
from threadpoolctl import threadpool_limits

from src import config, datos, modelos
from src.entrenar import metricas, preparar_mlflow

EXP = "cpu-perf/experimentos"
KF = KFold(5, shuffle=True, random_state=config.SEMILLA + 1)


# ---------- utilidades ----------
def cv_pred(modelo, X, y, aumentar=None, pesos=None):
    """Prediccion fuera de muestra con KFold revuelto. 'aumentar' se aplica SOLO al train de cada fold."""
    pred = np.zeros(len(y))
    for tr, te in KF.split(X):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        if aumentar is not None:
            Xtr, ytr = aumentar(Xtr, ytr)
        m = clone(modelo)
        with threadpool_limits(2):
            m.fit(Xtr, ytr)
            pred[te] = m.predict(X.iloc[te])
    return pred


def registrar(experimento, variante, m, extra_tags=None, params=None):
    with mlflow.start_run(run_name=f"{experimento}__{variante}", nested=True,
                          tags={"experimento": experimento, "variante": variante, "version_codigo": config.version_codigo(),
                                **(extra_tags or {})}):
        mlflow.log_metrics(m)
        if params:
            mlflow.log_params(params)
    print(f"   {variante:<38} MAE {m['mae']:7.0f} | R2 log {m['r2_log']:.3f} | dentro 20% {m['kpi_dentro_20pct']:.1%}", flush=True)
    return {"experimento": experimento, "variante": variante, **m}


def rf():
    return modelos.con_log(make_pipeline(modelos.preprocesamiento(), RandomForestRegressor(
        n_estimators=200, min_samples_leaf=1, random_state=config.SEMILLA, n_jobs=-1)))


# ---------- aumentacion de datos ----------
def smoter(X, y, percentil=85, k=5, factor=1.0, semilla=0):
    """SMOTE para regresion (Torgo et al. 2013): crea procesadores sinteticos en la zona escasa (gama alta)
    interpolando entre un procesador y uno de sus k vecinos; el objetivo se interpola igual (en log)."""
    rng = np.random.default_rng(semilla)
    corte = np.percentile(y, percentil)
    raro = np.where(y.values >= corte)[0]
    num = ["cores", "TDP", "anio", "ghz_nombre", "generacion", "gama"] + [c for c in modelos.EXTERNAS if c in X]
    Z = X.iloc[raro][num].copy()
    Z = Z.fillna(Z.median())
    Zs = (Z - Z.mean()) / (Z.std() + 1e-9)
    vecinos = NearestNeighbors(n_neighbors=min(k + 1, len(raro))).fit(Zs.values).kneighbors(Zs.values, return_distance=False)[:, 1:]
    nuevos, ys = [], []
    for _ in range(int(len(raro) * factor)):
        i = rng.integers(len(raro))
        j = vecinos[i, rng.integers(vecinos.shape[1])]
        r = rng.random()
        fila = X.iloc[raro[i]].copy()                      # categoricas y banderas: las del procesador base
        for c in num:
            a, b = X.iloc[raro[i]][c], X.iloc[raro[j]][c]
            if pd.notna(a) and pd.notna(b):
                fila[c] = a + r * (b - a)
        fila["cores"] = round(fila["cores"])
        nuevos.append(fila)
        ys.append(np.expm1(np.log1p(y.iloc[raro[i]]) + r * (np.log1p(y.iloc[raro[j]]) - np.log1p(y.iloc[raro[i]]))))
    return pd.concat([X, pd.DataFrame(nuevos)], ignore_index=True), pd.concat([y, pd.Series(ys)], ignore_index=True)


def ruido_gaussiano(X, y, sigma=0.05, copias=1, semilla=0):
    """Copias de train con ruido multiplicativo pequeno en las numericas continuas (no en cores ni anio)."""
    rng = np.random.default_rng(semilla)
    partes_X, partes_y = [X], [y]
    for _ in range(copias):
        Xn = X.copy()
        for c in ["TDP", "ghz_nombre"] + [c for c in modelos.EXTERNAS if c in X and c != "hilos"]:
            Xn[c] = Xn[c] * (1 + rng.normal(0, sigma, len(Xn)))
        partes_X.append(Xn)
        partes_y.append(y * (1 + rng.normal(0, sigma / 2, len(y))))
    return pd.concat(partes_X, ignore_index=True), pd.concat(partes_y, ignore_index=True)


# ---------- experimentos ----------
def e1_objetivo(X, y):
    print("E1 - objetivo original vs log")
    filas = []
    for nombre, reg in [("ridge", Ridge(alpha=1.0)), ("random_forest", RandomForestRegressor(n_estimators=200, random_state=config.SEMILLA, n_jobs=-1))]:
        crudo = make_pipeline(modelos.preprocesamiento(), reg)
        filas.append(registrar("E1_objetivo", f"{nombre} cpuMark original", metricas(y, cv_pred(crudo, X, y))))
        filas.append(registrar("E1_objetivo", f"{nombre} log(1+cpuMark)", metricas(y, cv_pred(modelos.con_log(crudo), X, y))))
    return filas


def e2_variables(df, y):
    print("E2 - que variables usar (random forest)")
    filas = []
    Xe, _ = datos.preparar(df, externos=True)
    base_num, base_cat = list(modelos.NUMERICAS), list(modelos.CATEGORICAS)
    grupos = [
        ("solo cores (Actividad 3)", ["cores"], []),
        ("specs de hoja de datos", ["cores", "TDP", "anio", "ghz_nombre"], ["fabricante", "familia", "socket"]),
        ("+ generacion/gama/sufijo del nombre", ["cores", "TDP", "anio", "ghz_nombre", "generacion", "gama"], ["fabricante", "familia", "sufijo", "socket"]),
        ("+ externos (cache, nm, GHz, hilos)", ["cores", "TDP", "anio", "ghz_nombre", "generacion", "gama"] + modelos.EXTERNAS, ["fabricante", "familia", "sufijo", "socket"]),
    ]
    for etiqueta, num, cat in grupos:
        if etiqueta.startswith("solo"):      # como en la Actividad 3: una sola variable
            m = modelos.con_log(make_pipeline(ColumnTransformer([("cores", FunctionTransformer(np.log1p), ["cores"])]),
                                              RandomForestRegressor(n_estimators=200, random_state=config.SEMILLA, n_jobs=-1)))
        else:
            modelos.NUMERICAS, modelos.CATEGORICAS = num, cat
            m = rf()
        filas.append(registrar("E2_variables", etiqueta, metricas(y, cv_pred(m, Xe, y)), params={"n_numericas": len(num)}))
    modelos.NUMERICAS, modelos.CATEGORICAS = base_num, base_cat
    return filas


def e3_socket(X, y):
    print("E3 - socket: one-hot vs target encoding")
    filas = []
    original = modelos.preprocesamiento
    for etiqueta in ["target encoding", "one-hot", "sin socket"]:
        def prep(externos=None, et=etiqueta):
            ct = original(externos)
            pasos = [(n, t, [c for c in cols if c != "socket"]) if n == "cat" else (n, t, cols) for n, t, cols in ct.transformers]
            if et == "one-hot":
                pasos.append(("socket", OneHotEncoder(handle_unknown="ignore", min_frequency=10), ["socket"]))
            elif et == "target encoding":            # explicito: produccion ya usa one-hot para el socket
                pasos.append(("socket", TargetEncoder(random_state=config.SEMILLA), ["socket"]))
            ct.transformers = pasos
            return ct
        modelos.preprocesamiento = prep
        for nombre in ["ridge", "random_forest"]:
            m = modelos.catalogo()[nombre][1]
            filas.append(registrar("E3_socket", f"{nombre} {etiqueta}", metricas(y, cv_pred(m, X, y))))
    modelos.preprocesamiento = original
    return filas


def e4_modelos(X, y):
    print("E4 - comparar modelos (con externos)")
    return [registrar("E4_modelos", nombre, metricas(y, cv_pred(m, X, y)), {"familia_modelo": fam})
            for nombre, (fam, m, _) in modelos.catalogo().items()]


def e5_aumentacion(X, y):
    print("E5 - aumentacion de datos (solo en train de cada fold)")
    filas = []
    for etiqueta, f in [("sin aumentar", None),
                        ("SMOTER gama alta (p85, x1)", lambda a, b: smoter(a, b, 85, 5, 1.0)),
                        ("SMOTER gama alta (p70, x2)", lambda a, b: smoter(a, b, 70, 5, 2.0)),
                        ("ruido gaussiano 5%", lambda a, b: ruido_gaussiano(a, b, 0.05, 1))]:
        p = cv_pred(rf(), X, y, aumentar=f)
        alto = y.values >= np.percentile(y, 85)
        m = metricas(y, p) | {f"alto_{k}": v for k, v in metricas(y[alto], p[alto]).items() if k in ["mae", "kpi_dentro_20pct"]}
        filas.append(registrar("E5_aumentacion", etiqueta, m))
        print(f"      gama alta: MAE {m['alto_mae']:.0f} | dentro 20% {m['alto_kpi_dentro_20pct']:.1%}")
    return filas


def e6_validacion(X, y):
    print("E6 - validacion aleatoria vs temporal (random forest)")
    m_aleatoria = metricas(y, cv_pred(rf(), X, y))
    viejo = X["anio"] <= config.ANIO_CORTE
    mod = clone(rf())
    with threadpool_limits(2):
        mod.fit(X[viejo], y[viejo])
        m_temporal = metricas(y[~viejo], mod.predict(X[~viejo]))
    return [registrar("E6_validacion", "kfold aleatorio", m_aleatoria),
            registrar("E6_validacion", f"temporal (train <= {config.ANIO_CORTE}, test > {config.ANIO_CORTE})", m_temporal)]


def e7_drift(X, y):
    print("E7 - estrategias contra el concept drift (train <= 2020, test 2021-2022)")
    filas = []
    tr, te = X["anio"] <= 2020, X["anio"] >= 2021
    for etiqueta, mascara, pesos in [
        ("todo el pasado", tr, None),
        ("ventana ultimos 5 anios", tr & (X["anio"] >= 2016), None),
        ("ventana ultimos 3 anios", tr & (X["anio"] >= 2018), None),
        ("pesos por antiguedad (vida media 3 anios)", tr, lambda a: 0.5 ** ((2020 - a) / 3)),
    ]:
        mod = clone(rf())
        kw = {}
        if pesos is not None:
            kw["randomforestregressor__sample_weight"] = pesos(X.loc[mascara, "anio"]).values
        with threadpool_limits(2):
            mod.fit(X[mascara], y[mascara], **kw)
            filas.append(registrar("E7_drift", etiqueta, metricas(y[te], mod.predict(X[te])), params={"n_train": int(mascara.sum())}))
    return filas


def e8_combinador(X, y):
    """Stacking (regresion lineal con pesos positivos) vs Ensemble Selection de Caruana et al. 2004 vs promedio."""
    print("E8 - como combinar los modelos")
    base = {n: modelos.catalogo()[n][1] for n in ["ridge", "knn", "random_forest", "hgb"]}
    oof = {n: cv_pred(m, X, y) for n, m in base.items()}
    filas = [registrar("E8_combinador", f"solo {n}", metricas(y, p)) for n, p in oof.items()]
    P = np.column_stack(list(oof.values()))
    filas.append(registrar("E8_combinador", "promedio simple (en log)", metricas(y, np.expm1(np.log1p(P).mean(axis=1)))))
    # Caruana: agrega con reemplazo el modelo que mas baja el MAE (en log) del promedio. Pesos evaluados con el mismo OOF
    # (optimista); para comparar justo se usa la mitad de los datos para escoger pesos y la otra mitad para medir.
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); a, b = idx[: len(y) // 2], idx[len(y) // 2:]
    L = np.log1p(P); ly = np.log1p(y.values)
    cuenta = np.zeros(L.shape[1]); suma = np.zeros(len(a))
    for _ in range(25):
        k = int(np.argmin([np.mean(np.abs((suma + L[a, j]) / (cuenta.sum() + 1) - ly[a])) for j in range(L.shape[1])]))
        cuenta[k] += 1; suma += L[a, k]
    w = cuenta / cuenta.sum()
    pred_b = np.expm1(L[b] @ w)
    filas.append(registrar("E8_combinador", "Ensemble Selection (Caruana)", metricas(y.values[b], pred_b),
                           params={f"peso_{n}": round(v, 2) for n, v in zip(oof, w)}))
    print(f"      pesos Caruana: {dict(zip(oof, w.round(2)))}")
    return filas


def e9_temporal(X, y):
    print(f"E9 - modelos en validacion temporal (train <= {config.ANIO_CORTE}, test > {config.ANIO_CORTE})")
    filas = []
    viejo = X["anio"] <= config.ANIO_CORTE
    for nombre in ["ridge", "knn", "random_forest", "random_forest_ruido", "hgb", "hibrido", "stacking"]:
        m = clone(modelos.catalogo()[nombre][1])
        with threadpool_limits(2):
            m.fit(X[viejo], y[viejo])
            filas.append(registrar("E9_temporal", nombre, metricas(y[~viejo], m.predict(X[~viejo]))))
    return filas


def e11_hibrido(X, y):
    """Lo que gano en E7 (pesos por antiguedad) y en E5 (ruido) aplicado al hibrido, que gano en E9. Validacion temporal."""
    print(f"E11 - hibrido + pesos por antiguedad / ruido (train <= {config.ANIO_CORTE}, test > {config.ANIO_CORTE})")
    viejo = X["anio"] <= config.ANIO_CORTE
    filas = []
    for etiqueta, m in [("hibrido", modelos.Hibrido()),
                        ("hibrido vida media 5 anios", modelos.Hibrido(vida_media=5)),
                        ("hibrido vida media 3 anios", modelos.Hibrido(vida_media=3)),
                        ("hibrido vida media 2 anios", modelos.Hibrido(vida_media=2)),
                        ("hibrido + ruido", modelos.ConRuido(modelos.Hibrido())),
                        ("hibrido vida media 3 + ruido", modelos.ConRuido(modelos.Hibrido(vida_media=3)))]:
        m = modelos.con_log(m)
        with threadpool_limits(2):
            m.fit(X[viejo], y[viejo])
            filas.append(registrar("E11_hibrido", etiqueta, metricas(y[~viejo], m.predict(X[~viejo]))))
    return filas


def e12_redes(X, y):
    """Redes neuronales (MLP, clase 12): arquitectura, activacion, regularizacion, y la red como corrector del hibrido.
    Cada variante se mide con kfold (procesadores conocidos) y temporal (train <= 2019, test > 2019)."""
    print("E12 - redes neuronales (kfold | temporal)")
    viejo = X["anio"] <= config.ANIO_CORTE
    variantes = [
        ("hibrido_reciente (referencia)", modelos.Hibrido(vida_media=5), {}),
        ("red 1 capa de 32", make_pipeline(modelos.preprocesamiento(), modelos.mlp((32,))), {"capas": "32"}),
        ("red 1 capa de 128", make_pipeline(modelos.preprocesamiento(), modelos.mlp((128,))), {"capas": "128"}),
        ("red 2 capas 64-32", make_pipeline(modelos.preprocesamiento(), modelos.mlp((64, 32))), {"capas": "64-32"}),
        ("red 3 capas 128-64-32", make_pipeline(modelos.preprocesamiento(), modelos.mlp((128, 64, 32))), {"capas": "128-64-32"}),
        ("red 64-32 tanh", make_pipeline(modelos.preprocesamiento(), modelos.mlp((64, 32), "tanh")), {"activacion": "tanh"}),
        ("red 64-32 alpha 0.1", make_pipeline(modelos.preprocesamiento(), modelos.mlp((64, 32), alpha=0.1)), {"alpha": 0.1}),
        ("red 64-32 alpha 1", make_pipeline(modelos.preprocesamiento(), modelos.mlp((64, 32), alpha=1.0)), {"alpha": 1.0}),
        ("hibrido: recta + red", modelos.Hibrido(alpha=10, segundo="red"), {"segundo": "red"}),
        ("hibrido: recta + red, peso reciente", modelos.Hibrido(alpha=10, segundo="red", vida_media=5), {"segundo": "red", "vida_media": 5}),
    ]
    filas = []
    for etiqueta, base, params in variantes:
        m = modelos.con_log(base)
        t = time.time()
        k = metricas(y, cv_pred(m, X, y))
        with threadpool_limits(2):
            temp = metricas(y[~viejo], clone(m).fit(X[viejo], y[viejo]).predict(X[~viejo]))
        todo = {**k, **{f"temporal_{a}": b for a, b in temp.items()}, "segundos": time.time() - t}
        fila = registrar("E12_redes", etiqueta, todo, params=params)
        print(f"   {'':<38} temporal MAE {temp['mae']:7.0f} | dentro 20% {temp['kpi_dentro_20pct']:.1%} | {todo['segundos']:.0f} s", flush=True)
        filas.append(fila)
    return filas


def e13_robustez(X, y):
    """Validacion temporal con 5 cortes (train <= corte, test = los 2 anios siguientes) y, en las redes, 3 semillas.
    Un modelo robusto gana en casi todos los cortes, no solo en uno."""
    print("E13 - robustez: 5 cortes de tiempo (y 3 semillas en las redes)")
    cortes = [2016, 2017, 2018, 2019, 2020]
    candidatos = {
        "ridge": (lambda s: make_pipeline(modelos.preprocesamiento(), Ridge(alpha=10)), 1),
        "random_forest": (lambda s: make_pipeline(modelos.preprocesamiento(), RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, random_state=s, n_jobs=-1)), 1),
        "hgb": (lambda s: make_pipeline(modelos.preprocesamiento(), HistGradientBoostingRegressor(random_state=s)), 1),
        "hibrido (bosque)": (lambda s: modelos.Hibrido(alpha=10), 1),
        "hibrido_reciente (bosque, produccion)": (lambda s: modelos.Hibrido(alpha=10, vida_media=5), 1),
        "red 64-32 alpha 1": (lambda s: make_pipeline(modelos.preprocesamiento(), modelos.mlp((64, 32), alpha=1.0, semilla=s)), 3),
        "red 64-32 alpha 3": (lambda s: make_pipeline(modelos.preprocesamiento(), modelos.mlp((64, 32), alpha=3.0, semilla=s)), 3),
        "hibrido recta + red, reciente": (lambda s: modelos.Hibrido(alpha=10, segundo="red", vida_media=5), 1),
    }
    filas, detalle = [], []
    for nombre, (hacer, n_sem) in candidatos.items():
        maes, kpis = [], []
        for corte in cortes:
            tr, te = X["anio"] <= corte, (X["anio"] > corte) & (X["anio"] <= corte + 2)
            for sem in range(n_sem):
                with threadpool_limits(2):
                    m = modelos.con_log(hacer(config.SEMILLA + sem)).fit(X[tr], y[tr])
                    r = metricas(y[te], m.predict(X[te]))
                maes.append(r["mae"]); kpis.append(r["kpi_dentro_20pct"])
                detalle.append({"modelo": nombre, "corte": corte, "semilla": sem, "mae": r["mae"], "kpi": r["kpi_dentro_20pct"]})
        res = {"mae": float(np.mean(maes)), "mae_std": float(np.std(maes)), "kpi_dentro_20pct": float(np.mean(kpis)),
               "kpi_std": float(np.std(kpis)), "r2_log": float("nan"), "n_pruebas": len(maes)}
        filas.append(registrar("E13_robustez", nombre, res, params={"cortes": "2016-2020", "semillas": n_sem}))
        print(f"   {'':<38} MAE {np.mean(maes):6.0f} \u00b1 {np.std(maes):5.0f} | dentro 20% {np.mean(kpis):.1%} \u00b1 {np.std(kpis):.1%}", flush=True)
    det = pd.DataFrame(detalle)
    # en cuantos cortes gana cada modelo (con el promedio de semillas)
    piv = det.groupby(["corte", "modelo"]).mae.mean().unstack()
    gana = piv.idxmin(axis=1).value_counts()
    print("   cortes ganados:", gana.to_dict())
    print(piv.round(0).to_string())
    det.to_csv(config.REPORTES / "robustez_detalle.csv", index=False)
    mlflow.log_artifact(str(config.REPORTES / "robustez_detalle.csv"))
    return filas


def smogn(X, y, percentil=85, k=5, semilla=0):
    """SMOGN (Branco et al. 2017): como SMOTER, pero si el vecino esta lejos usa ruido gaussiano en vez de interpolar."""
    Xs, ys = smoter(X, y, percentil, k, 1.0, semilla)
    nuevos = Xs.iloc[len(X):].copy()
    rng = np.random.default_rng(semilla + 1)
    lejos = rng.random(len(nuevos)) < 0.3          # aproximacion: 30% de los sinteticos se perturban con ruido
    for c in ["TDP", "ghz_nombre"] + [c for c in modelos.EXTERNAS if c in X and c != "hilos"]:
        nuevos.loc[lejos, c] = nuevos.loc[lejos, c] * (1 + rng.normal(0, 0.05, lejos.sum()))
    return pd.concat([X, nuevos], ignore_index=True), ys


def e10_smogn(X, y):
    print("E10 - SMOGN vs SMOTER vs ruido")
    filas = []
    alto = y.values >= np.percentile(y, 85)
    for etiqueta, f in [("sin aumentar", None), ("SMOTER p85", lambda a, b: smoter(a, b, 85)),
                        ("SMOGN p85", lambda a, b: smogn(a, b, 85)), ("ruido gaussiano 5%", lambda a, b: ruido_gaussiano(a, b, 0.05))]:
        p = cv_pred(rf(), X, y, aumentar=f)
        m = metricas(y, p) | {"alto_mae": metricas(y[alto], p[alto])["mae"]}
        filas.append(registrar("E10_smogn", etiqueta, m))
    return filas


TODOS = {"e1": e1_objetivo, "e2": e2_variables, "e3": e3_socket, "e4": e4_modelos, "e5": e5_aumentacion,
         "e6": e6_validacion, "e7": e7_drift, "e8": e8_combinador, "e9": e9_temporal, "e10": e10_smogn, "e11": e11_hibrido, "e12": e12_redes, "e13": e13_robustez}


def correr(cuales=None):
    info = datos.descargar()
    df = datos.cargar(info["ruta"])
    modelos.usar_externos(True)
    X, _ = datos.preparar(df, externos=True)
    y = df[config.OBJETIVO]
    preparar_mlflow(EXP)
    config.REPORTES.mkdir(exist_ok=True)
    filas = []
    for clave in (cuales or TODOS):
        t = time.time()
        with mlflow.start_run(run_name=f"{clave}__data-v{info['version']}",
                              tags={"data_version": info["version"], "data_hash": info["hash"], "etapa": "experimento"}):
            entrada = df if clave == "e2" else X
            filas += TODOS[clave](entrada, y)
        print(f"   ({time.time() - t:.0f} s)\n", flush=True)
    tabla = pd.DataFrame(filas)
    previo = config.REPORTES / "experimentos.csv"
    if previo.exists() and cuales:
        viejo = pd.read_csv(previo)
        tabla = pd.concat([viejo[~viejo.experimento.isin(tabla.experimento.unique())], tabla])
    tabla.to_csv(previo, index=False)
    return tabla


if __name__ == "__main__":
    correr(sys.argv[1:] or None)
