"""Transfer learning entre PassMark y Blender: que lo aprendido con una fuente ayude a entrenar con la otra.

Referencia (ganador actual, v7): A' + traducidos = combo con nucleos reales + CPUs de Blender traducidos a PassMark (C).

  T1  B recibe a A    : el modelo de Blender usa la prediccion de A (PassMark) como columna extra; luego C lo traduce
  T2  A recibe a B    : la referencia + la prediccion del modelo de Blender como columna extra ("pista")
  T3  red neuronal    : pre-entrenada con PassMark y re-afinada con Blender traducido (transfer learning clasico)
      T3b             : la misma red sin pre-entrenar (para ver si el pre-entrenamiento sirve)
  T4  multi-tarea     : un solo modelo con TODAS las filas de las dos fuentes (Blender en escala PassMark) y una
                        columna "es_blender"; se predice con es_blender = 0

Pruebas (las mismas de antes):
  a) 5 cortes de PassMark (2016-2020): MAE en cpuMark. Blender solo aporta CPUs lanzados <= corte y nunca los del test.
  b) futuro: se entrena con Blender lanzado <= 2023 (o 2024) y se prueba el ORDEN contra Blender en los posteriores.

Correr:  python -m archivo.transferencia
"""
import json

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src import config, cruzado as cz, datos, modelos
from src.entrenar import metricas, preparar_mlflow

EXP = "cpu-perf/transferencia"


# ---------------------------------------------------------------- utilidades
def con_extra(columnas, fn):
    """Corre fn() con columnas extra habilitadas en el preprocesamiento."""
    antes = list(modelos.EXTRA_NUMERICAS)
    modelos.EXTRA_NUMERICAS[:] = columnas
    try:
        return fn()
    finally:
        modelos.EXTRA_NUMERICAS[:] = antes


def ajustar(m, X, y):
    with threadpool_limits(2):
        return clone(m).fit(X, y)


def modelo_b():
    return cz.candidatos_b()["hibrido_hgb"]      # el que gano para predecir Blender


def oof(m, X, y, k=3):
    """Predicciones fuera de muestra (para usar la prediccion de un modelo como columna sin que haga trampa)."""
    p = np.zeros(len(y))
    for tr, te in KFold(k, shuffle=True, random_state=config.SEMILLA).split(X):
        p[te] = ajustar(m, X.iloc[tr], y[tr]).predict(X.iloc[te])
    return p


# ---------------------------------------------------------------- los sistemas
# Cada sistema recibe lo que tiene permitido ver y regresa una funcion predecir(X) en escala cpuMark.
# d = {"Xpm","ypm","Xb","yb" (puntaje blender),"solo_b" (bool: CPU solo en Blender),"ov" (bool: en ambos, permitido),
#      "pm_y_b" (cpuMark de las filas de Blender que estan en PassMark)}

def _traductor(d):
    return cz.Traductor().fit(d["Xb"][d["ov"]].reset_index(drop=True), d["yb"][d["ov"]], d["pm_y_b"][d["ov"]])


def referencia(d):
    c = _traductor(d)
    yt = c.predict(d["Xb"][d["solo_b"]].reset_index(drop=True), d["yb"][d["solo_b"]])
    a = ajustar(cz.modelo_base(), pd.concat([d["Xpm"], d["Xb"][d["solo_b"]]], ignore_index=True), np.r_[d["ypm"], yt])
    return a.predict


def t1_b_recibe_a(d):
    a = ajustar(cz.modelo_base(), d["Xpm"], d["ypm"])
    # pista de A para las filas de Blender: fuera de muestra si el CPU estaba en el train de A (si no, B confiaria de mas)
    pa_oof = pd.Series(oof(cz.modelo_base(), d["Xpm"], d["ypm"]), index=d["clave_pm"])
    pa_oof = pa_oof[~pa_oof.index.duplicated()]
    pista_b = d["clave_b"].map(pa_oof).values
    pista_b = np.log(np.clip(np.where(np.isnan(pista_b), a.predict(d["Xb"]), pista_b), 1, None))
    Xb = d["Xb"].assign(pista=pista_b)
    b = con_extra(["pista"], lambda: ajustar(modelo_b(), Xb, d["yb"]))
    c = _traductor(d)

    def predecir(X):
        Xp = X.assign(pista=np.log(np.clip(a.predict(X), 1, None)))
        return c.predict(X.reset_index(drop=True), np.clip(b.predict(Xp), 1e-3, None))
    return predecir


def t2_a_recibe_b(d):
    # B fuera de muestra para sus propias filas; las filas de PassMark que estan en Blender toman esa prediccion
    pb_b = oof(modelo_b(), d["Xb"], d["yb"])
    b = ajustar(modelo_b(), d["Xb"], d["yb"])
    clave_b = pd.Series(np.log(np.clip(pb_b, 1e-3, None)), index=d["clave_b"].values)
    clave_b = clave_b[~clave_b.index.duplicated()]
    pista_pm = pd.Series(d["clave_pm"]).map(clave_b).values
    pista_pm = np.where(np.isnan(pista_pm), np.log(np.clip(b.predict(d["Xpm"]), 1e-3, None)), pista_pm)
    c = _traductor(d)
    yt = c.predict(d["Xb"][d["solo_b"]].reset_index(drop=True), d["yb"][d["solo_b"]])
    Xtr = pd.concat([d["Xpm"].assign(pista=pista_pm),
                     d["Xb"][d["solo_b"]].assign(pista=np.log(np.clip(pb_b[d["solo_b"]], 1e-3, None)))], ignore_index=True)
    a = con_extra(["pista"], lambda: ajustar(cz.modelo_base(), Xtr, np.r_[d["ypm"], yt]))

    def predecir(X):
        return a.predict(X.assign(pista=np.log(np.clip(b.predict(X), 1e-3, None))))
    return predecir


def _red(d, pre_entrenar):
    """Red neuronal: pre-entrena con PassMark y re-afina con Blender en escala PassMark (lo de ambos con su cpuMark real,
    lo de solo-Blender traducido con C). Sin pre-entrenar: la misma red con todo junto desde cero."""
    c = _traductor(d)
    yb_pm = np.where(d["ov"], d["pm_y_b"], np.nan)
    yb_pm = np.where(np.isnan(yb_pm), c.predict(d["Xb"].reset_index(drop=True), d["yb"]), yb_pm)
    prep = modelos.preprocesamiento().fit(pd.concat([d["Xpm"], d["Xb"]], ignore_index=True))
    sy = StandardScaler().fit(np.log1p(d["ypm"]).reshape(-1, 1))
    t = lambda yy: sy.transform(np.log1p(yy).reshape(-1, 1)).ravel()
    Zpm, Zb = prep.transform(d["Xpm"]), prep.transform(d["Xb"])
    red = MLPRegressor(hidden_layer_sizes=(64, 32), activation="tanh", alpha=1.0, learning_rate_init=3e-3,
                       max_iter=300, early_stopping=False, random_state=config.SEMILLA)
    with threadpool_limits(2):
        if pre_entrenar:
            red.fit(Zpm, t(d["ypm"]))                                     # 1) aprende con PassMark (3,825 CPUs)
            red.set_params(warm_start=True, learning_rate_init=3e-4, max_iter=150)
            red.fit(np.r_[Zb, Zpm[::4]], np.r_[t(yb_pm), t(d["ypm"])[::4]])  # 2) re-afina con Blender (+ un poco de PassMark para no olvidar)
        else:
            red.fit(np.r_[Zpm, Zb], np.r_[t(d["ypm"]), t(yb_pm)])
    lim = (np.log1p(d["ypm"]).min() - 0.5, np.log1p(np.r_[d["ypm"], yb_pm]).max() + 1)

    def predecir(X):
        z = sy.inverse_transform(red.predict(prep.transform(X)).reshape(-1, 1)).ravel()
        return np.expm1(np.clip(z, *lim))
    return predecir


def t3_red_preentrenada(d):
    return _red(d, True)


def t3b_red_sin_preentrenar(d):
    return _red(d, False)


def t4_multitarea(d):
    """Todas las filas de Blender (tambien las que ya estan en PassMark) en escala PassMark, con indicador de fuente."""
    c = cz.Traductor(simple=True).fit(d["Xb"][d["ov"]].reset_index(drop=True), d["yb"][d["ov"]], d["pm_y_b"][d["ov"]])
    yb_pm = c.predict(d["Xb"].reset_index(drop=True), d["yb"])
    Xtr = pd.concat([d["Xpm"].assign(es_blender=0.0), d["Xb"].assign(es_blender=1.0)], ignore_index=True)
    a = con_extra(["es_blender"], lambda: ajustar(cz.modelo_base(), Xtr, np.r_[d["ypm"], yb_pm]))
    return lambda X: a.predict(X.assign(es_blender=0.0))


SISTEMAS = {"referencia (A' + traducidos, v7)": referencia, "T1 B recibe a A": t1_b_recibe_a, "T2 A recibe a B": t2_a_recibe_b,
            "T3 red pre-entrenada": t3_red_preentrenada, "T3b red sin pre-entrenar": t3b_red_sin_preentrenar,
            "T4 multi-tarea": t4_multitarea}


# ---------------------------------------------------------------- escenarios
def paquete(pm, b, Xpm, Xb, m_pm, m_b, corte_pm=None):
    ov = m_b & b["pm_cpuMark"].notna().values
    if corte_pm is not None:
        ov &= b["pm_testDate"].values <= corte_pm
    sel = m_b
    return {"Xpm": Xpm[m_pm].reset_index(drop=True), "ypm": pm.loc[m_pm, "cpuMark"].values,
            "clave_pm": pm.loc[m_pm, "clave"].values, "Xb": Xb[sel].reset_index(drop=True), "yb": b.loc[sel, "blender"].values,
            "clave_b": b.loc[sel, "clave"], "solo_b": ~b.loc[sel, "en_passmark"].values, "ov": ov[sel],
            "pm_y_b": b.loc[sel, "pm_cpuMark"].values}


def correr():
    info = datos.descargar()
    df = datos.cargar(info["ruta"])
    modelos.usar_externos(True)
    preparar_mlflow(EXP)
    f, bl = cz.fichas(), cz.blender_por_clave()
    pm, b = cz.construir(df, f, bl)
    b["pm_cpuMark"] = b["clave"].map(pm.drop_duplicates("clave").set_index("clave")["cpuMark"])
    b = b[b["anio_lanz"].notna() | b["en_passmark"]].reset_index(drop=True)
    Xpm, Xb = cz.matriz(pm, "testDate"), cz.matriz(b, "anio_lanz")
    ov_all = b["pm_cpuMark"].notna().values
    c_ref = cz.Traductor().fit(Xb[ov_all].reset_index(drop=True), b.loc[ov_all, "blender"].values, b.loc[ov_all, "pm_cpuMark"].values)

    res = {n: {"cortes_mae": [], "cortes_kpi": [], "futuro_orden": [], "futuro_dentro20": []} for n in SISTEMAS}
    with mlflow.start_run(run_name=f"transferencia__data-v{info['version']}",
                          tags={"etapa": "transferencia", "data_version": info["version"], "version_codigo": config.version_codigo()}):
        print("[a] cortes de PassMark (MAE)", flush=True)
        for corte in config.CORTES_TEMPORALES:
            tr = pm["testDate"].values <= corte
            te = (pm["testDate"].values > corte) & (pm["testDate"].values <= corte + 2)
            m_b = (b["anio_lanz"].values <= corte) & ~b["clave"].isin(set(pm.loc[te, "clave"])).values
            d = paquete(pm, b, Xpm, Xb, tr, m_b, corte_pm=corte)
            Xt, yt = Xpm[te].reset_index(drop=True), pm.loc[te, "cpuMark"].values
            linea = []
            for n, fn in SISTEMAS.items():
                r = metricas(yt, np.clip(fn(d)(Xt), 1, None))
                res[n]["cortes_mae"].append(r["mae"]); res[n]["cortes_kpi"].append(r["kpi_dentro_20pct"])
                linea.append(f"{n.split(' ')[0]} {r['mae']:.0f}")
            print(f"  corte {corte}: " + " | ".join(linea), flush=True)
        print("[b] futuro: orden contra Blender en CPUs posteriores", flush=True)
        todo_pm = np.ones(len(pm), bool)
        for hasta in [2023, 2024]:
            m_b = b["anio_lanz"].values <= hasta
            te = (b["anio_lanz"].values > hasta) & ~b["en_passmark"].values
            d = paquete(pm, b, Xpm, Xb, todo_pm, m_b)
            Xt, real = Xb[te].reset_index(drop=True), b.loc[te, "blender"].values
            ref = c_ref.predict(Xt, real)
            linea = [f"nucleos {spearmanr(Xt['cores'], real)[0]:.3f}"]
            for n, fn in SISTEMAS.items():
                p = np.clip(fn(d)(Xt), 1, None)
                res[n]["futuro_orden"].append(spearmanr(p, real)[0])
                res[n]["futuro_dentro20"].append(metricas(ref, p)["kpi_dentro_20pct"])
                linea.append(f"{n.split(' ')[0]} {res[n]['futuro_orden'][-1]:.3f}")
            print(f"  hasta {hasta} ({te.sum()} CPUs): " + " | ".join(linea), flush=True)

        print("\nresumen:", flush=True)
        filas = []
        for n, r in res.items():
            fila = {"sistema": n, "cortes_mae": np.mean(r["cortes_mae"]), "cortes_dentro20": np.mean(r["cortes_kpi"]),
                    "futuro_orden": np.mean(r["futuro_orden"]), "futuro_dentro20": np.mean(r["futuro_dentro20"])}
            filas.append(fila)
            with mlflow.start_run(run_name=n, nested=True, tags={"etapa": "transferencia", "sistema": n}):
                mlflow.log_metrics({k: float(v) for k, v in fila.items() if k != "sistema"})
            print(f"  {n:<34} cortes MAE {fila['cortes_mae']:6.0f} ({fila['cortes_dentro20']:.1%}) | futuro orden "
                  f"{fila['futuro_orden']:.3f}, dentro 20% {fila['futuro_dentro20']:.1%}", flush=True)
        t = pd.DataFrame(filas)
        t.to_csv(config.REPORTES / "transferencia.csv", index=False)
        mlflow.log_artifact(str(config.REPORTES / "transferencia.csv"))
        json.dump(res, open(config.REPORTES / "transferencia_detalle.json", "w"), indent=2, default=float)
        mlflow.log_artifact(str(config.REPORTES / "transferencia_detalle.json"))
    return t


if __name__ == "__main__":
    correr()
