"""Modelo cruzado PassMark + Blender, con el resultado final en escala PassMark.

  A  : predice cpuMark (PassMark, 2007-2022)
  B  : predice el puntaje de Blender (Blender Open Data CC0, hasta 2026)
  C  : "traductor" Blender -> PassMark, aprendido con los CPUs que estan en las dos fuentes
  final = mezcla (en log) de A y C(B), con el peso escogido por validacion

Pasos:
  1. fichas(): ficha tecnica unificada (oficiales Intel/AMD > TechPowerUp > pc_build): nucleos REALES, hilos,
     anio de lanzamiento, TDP, socket, tipo. Con eso se corrige PassMark (cuenta mal nucleos de Intel 12a+ y AMD FX) y se
     completa Blender (no trae TDP, socket ni tipo).
  2. traductor C, validado con kfold.
  3. B con validacion por cortes de tiempo (anio de lanzamiento).
  4. evaluacion honesta del sistema:
     a) cortes 2016-2020 de PassMark: B y C solo ven CPUs lanzados <= corte y nunca los del test
     b) prueba del futuro: todo entrenado con lanzamientos <= 2022, se prueba el ORDEN contra Blender en 2023-2026
  5. si le gana al modelo en produccion, se registra como nueva version.

Correr:  python -m src.cruzado
"""
import json
import re

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from . import config, datos, externos, modelos
from .entrenar import metricas, preparar_mlflow

EXP = "cpu-perf/cruzado"
CORTES = config.CORTES_TEMPORALES
ANIO_FUTURO = 2022          # prueba del futuro: se entrena con lanzamientos <= 2022 y se prueba con 2023-2026


# ---------------------------------------------------------------- 1. fichas
def _num(v):
    m = re.search(r"(\d+(?:\.\d+)?)", str(v))
    return float(m.group(1)) if m else np.nan


def _anio(v):
    m = re.search(r"(19|20)\d{2}", str(v))
    return float(m.group(0)) if m else np.nan


def tipo_por_nombre(n):
    n = str(n).lower()
    if re.search(r"epyc|xeon|opteron", n):
        return "Server"
    if re.search(r"snapdragon|apple|oryon|cortex|mediatek|exynos|\bmt\d{4}", n):
        return "Mobile/Embedded"
    if re.search(r"\d{4,5}(h|hx|hs|hk|u|p|y|g\d)\b|ultra \d+ \d{3}[hvu]\b|\bm\d\b", n):
        return "Laptop"
    return "Desktop"


def fichas():
    """Una fila por clave normalizada con lo mejor de cada fuente."""
    c, _ = externos.copia_kaggle("alanjo/amd-processor-specifications", "specs_oficiales")
    it = pd.read_csv(c / "INTELpartialspecs_adjusted.csv")
    am = pd.read_csv(c / "AMDfullspecs_adjusted.csv")
    intel = pd.DataFrame({"clave": it["product"].map(externos.normalizar), "f_nucleos": it["cores"], "f_hilos": it["threads"],
                          "f_anio": it["releaseDate"], "f_turbo": it["maxTurboClock"], "f_base": it["baseClock"],
                          "f_nm": it["lith"].map(_num)})
    amd = pd.DataFrame({"clave": ("AMD " + am["model"]).map(externos.normalizar), "f_nucleos": am["numCores"],
                        "f_hilos": am["numThreads"], "f_anio": am["launchDate"], "f_turbo": am["maxboostClock"],
                        "f_base": am["baseClock"], "f_nm": am["processTech"], "f_l3": am["L3Cache"],
                        "f_tdp": am["defaultTDP"].map(_num), "f_socket": am["cpuSocket"], "f_tipo": am["platform"]})
    oficial = pd.concat([intel, amd]).drop_duplicates("clave")
    ct, _ = externos.copia_kaggle("baraazaid/cpu-and-gpu-stats", "techpowerup")
    t = pd.read_csv(ct / "tpu_cpus.csv")
    tpu = pd.DataFrame({"clave": t["Name"].map(externos.normalizar), "f_nucleos": t["Cores"].map(_num),
                        "f_anio": t["Release"].map(_anio), "f_tdp": t["TDP"].map(_num),
                        "f_socket": t["Socket"].astype(str).str.replace("Socket ", "", regex=False),
                        "f_nm": t["Process"].map(_num)}).drop_duplicates("clave")
    raw = pd.read_parquet(externos.ruta_pc_build())
    pcb = pd.DataFrame({"clave": raw["metadata"].map(lambda m: m["name"]).map(externos.normalizar),
                        "f_nucleos": raw["cores"].map(lambda x: x.get("total")), "f_hilos": raw["cores"].map(lambda x: x.get("threads")),
                        "f_tdp": raw["specifications"].map(lambda s: s.get("tdp")), "f_socket": raw["socket"],
                        "f_anio": raw["metadata"].map(lambda m: m.get("releaseYear"))}).drop_duplicates("clave")
    f = oficial.set_index("clave")
    for otra in [tpu, pcb]:
        f = f.combine_first(otra.set_index("clave"))     # primero oficial, luego TechPowerUp, luego pc_build
    f["f_anio"] = f["f_anio"].where(f["f_anio"].between(1990, 2030))
    f["f_nucleos"] = f["f_nucleos"].where(f["f_nucleos"].between(1, 512))
    return f


# ---------------------------------------------------------------- datos de A y B
def blender_por_clave():
    """Blender agrupado por clave normalizada (une variantes del mismo nombre), mediana del puntaje ajustado por version."""
    from . import blender
    env = blender.leer_envios(blender.descargar())
    env = env[env["puntaje"] > 0]
    env, _ = blender.ajustar_version(env)
    env["clave"] = env["cpu"].map(externos.normalizar)
    # en la nube (AWS, Google...) se prueba una maquina virtual con PARTE del procesador: su puntaje no es del CPU
    # completo. Se usan solo las pruebas con todos los nucleos (>= 90% del maximo visto para ese modelo)
    maximo = env.groupby("clave")["nucleos"].transform("max")
    total = len(env)
    env = env[env["nucleos"] >= 0.9 * maximo]
    print(f"  Blender: {total - len(env)} de {total} pruebas descartadas por usar solo parte del procesador", flush=True)
    g = env.groupby("clave")
    t = pd.DataFrame({"blender": g["puntaje_ajustado"].median(), "envios": g.size(),
                      "b_nombre": g["cpu"].agg(lambda s: s.mode().iloc[0]),
                      "b_nucleos": g["nucleos"].max(), "b_hilos": g["hilos"].max(),
                      "b_primer_anio": g["fecha"].min().str[:4].astype(float)})
    return t[t["envios"] >= 5]


def construir(df_pm, f, bl):
    """Regresa (pm, bl_tabla): PassMark corregido y Blender completado, ambos con las columnas que usa datos.preparar."""
    pm = df_pm.copy()
    pm["clave"] = pm["cpuName"].map(externos.normalizar)
    pm = pm.join(f, on="clave").join(bl[["b_nucleos", "b_hilos"]], on="clave")
    pm["cores_pm"] = pm["cores"]
    nuevo = pm["f_nucleos"].fillna(pm["b_nucleos"])
    pm["cores"] = np.where(nuevo > pm["cores"], nuevo, pm["cores"])     # nucleos REALES: PassMark subcuenta, solo se corrige hacia arriba
    pm["hilos_real"] = pm["f_hilos"].fillna(pm["b_hilos"])
    pm["anio_lanz"] = pm["f_anio"].fillna(pm["testDate"])
    pm["TDP"] = pm["TDP"].fillna(pm["f_tdp"])
    pm_cat = pm.drop_duplicates("clave").set_index("clave")[["TDP", "socket", "category", "testDate"]]

    b = bl.copy().join(f, how="left").join(pm_cat.rename(columns=lambda c: "pm_" + c), how="left")
    b["cpuName"] = b["b_nombre"]
    b["cores"] = b["b_nucleos"].fillna(b["f_nucleos"])
    b["hilos_real"] = b["b_hilos"].fillna(b["f_hilos"])
    # anio de lanzamiento: ficha > PassMark > primera prueba en Blender (solo confiable si es >= 2023)
    b["anio_lanz"] = b["f_anio"].fillna(b["pm_testDate"]).fillna(b["b_primer_anio"].where(b["b_primer_anio"] >= 2023))
    b["TDP"] = b["pm_TDP"].fillna(b["f_tdp"])
    b["socket"] = b["pm_socket"].fillna(b["f_socket"]).fillna("unknown")
    b["category"] = b["pm_category"].fillna(b["f_tipo"]).fillna(b["cpuName"].map(tipo_por_nombre))
    b["en_passmark"] = b.index.isin(pm_cat.index)
    b = b.reset_index().rename(columns={"index": "clave"})
    return pm, b


def matriz(tabla, anio_col):
    """Features como en produccion, pero con nucleos reales, anio indicado y los hilos reales cuando se conocen."""
    t = tabla.assign(testDate=tabla[anio_col])
    X, _ = datos.preparar(t, externos=True)
    X = X.reset_index(drop=True)
    X["hilos"] = pd.Series(tabla["hilos_real"].values).fillna(X["hilos"])
    return X


# ---------------------------------------------------------------- 2. traductor C
class Traductor(RegressorMixin, BaseEstimator):
    """log(cpuMark) = Ridge(log blender, log nucleos, log hilos, fabricante, tipo). Entra el puntaje de Blender."""
    def __init__(self, alpha=1.0, simple=False):
        self.alpha = alpha
        self.simple = simple

    def _feat(self, X, blender_):
        F = pd.DataFrame({"lb": np.log(np.asarray(blender_, float))})
        if not self.simple:
            F["lc"] = np.log1p(X["cores"].values)
            F["lh"] = np.log1p(pd.Series(X["hilos"].values).fillna(pd.Series(X["cores"].values * 2)).values)
            for col in ["es_desktop", "es_laptop", "es_server", "es_mobile"]:
                F[col] = X[col].values
            for fab in ["Intel", "AMD", "Apple", "Qualcomm"]:
                F["fab_" + fab] = (X["fabricante"].values == fab).astype(float)
        return F

    def fit(self, X, blender_, y):
        self.m_ = Ridge(alpha=self.alpha).fit(self._feat(X, blender_), np.log1p(np.asarray(y, float)))
        return self

    def predict(self, X, blender_):
        return np.expm1(self.m_.predict(self._feat(X, blender_)))


# ---------------------------------------------------------------- el sistema final (lo que se registra y se sirve)
class SistemaCruzado(RegressorMixin, BaseEstimator):
    """Ya entrenado: A, B, C y el peso. Recibe la misma X que produccion y regresa cpuMark."""
    def __init__(self, a=None, b=None, c=None, peso_a=0.5):
        self.a, self.b, self.c, self.peso_a = a, b, c, peso_a

    def fit(self, X, y=None):
        return self                      # se arma ya entrenado (A y B aprenden de datos distintos)

    def predict(self, X):
        pa = np.clip(self.a.predict(X), 1, None)
        pc = np.clip(self.c.predict(X, np.clip(self.b.predict(X), 1e-3, None)), 1, None)
        return np.exp(self.peso_a * np.log(pa) + (1 - self.peso_a) * np.log(pc))


def modelo_base():
    return modelos.con_log(modelos.Combo())          # el mismo tipo que gano en produccion


def candidatos_b():
    return {"combo": modelos.con_log(modelos.Combo()),
            "hibrido_bosque": modelos.con_log(modelos.Hibrido(alpha=10, vida_media=5)),
            "hibrido_hgb": modelos.con_log(modelos.Hibrido(alpha=235, vida_media=2, segundo="hgb", min_samples_leaf=34)),
            "random_forest": modelos.con_log(modelos.ConPesos(modelos.RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                                                            random_state=config.SEMILLA, n_jobs=-1)))}


# ---------------------------------------------------------------- evaluacion
def ajustar(m, X, y):
    with threadpool_limits(1):       # con varios hilos el HGB de scikit-learn se traba en esta maquina (WSL)
        return clone(m).fit(X, y)


def paso2_traductor(b, Xb):
    ov = b["en_passmark"].values & b["pm_cpuMark"].notna().values
    X, bl, y = Xb[ov].reset_index(drop=True), b.loc[ov, "blender"].values, b.loc[ov, "pm_cpuMark"].values
    res = {}
    for nombre, c in [("simple (solo log blender)", Traductor(simple=True)), ("con nucleos, hilos, tipo, marca", Traductor())]:
        pred = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=config.SEMILLA).split(X):
            pred[te] = clone(c).fit(X.iloc[tr], bl[tr], y[tr]).predict(X.iloc[te], bl[te])
        res[nombre] = metricas(y, pred)
        print(f"  C {nombre:<34} n={len(y)} | MAE {res[nombre]['mae']:6.0f} | dentro 20% {res[nombre]['kpi_dentro_20pct']:.1%} "
              f"| R2 log {res[nombre]['r2_log']:.3f}", flush=True)
    return res


def paso3_b(b, Xb):
    """B con cortes de tiempo sobre el anio de lanzamiento (Blender tiene CPUs hasta 2026)."""
    ok = b["anio_lanz"].notna().values
    X, y, anio = Xb[ok].reset_index(drop=True), b.loc[ok, "blender"].reset_index(drop=True), b.loc[ok, "anio_lanz"].values
    res = {}
    for nombre, m in candidatos_b().items():
        maes, kpis = [], []
        for corte in [2019, 2020, 2021, 2022, 2023]:
            tr, te = anio <= corte, (anio > corte) & (anio <= corte + 2)
            r = metricas(y[te], ajustar(m, X[tr], y[tr]).predict(X[te]))
            maes.append(r["mape"]); kpis.append(r["kpi_dentro_20pct"])
        res[nombre] = {"mape": float(np.mean(maes)), "kpi": float(np.mean(kpis))}
        print(f"  B {nombre:<16} error relativo {np.mean(maes):.1%} | dentro 20% {np.mean(kpis):.1%} (5 cortes 2019-2023)", flush=True)
    mejor = min(res, key=lambda k: res[k]["mape"])
    return mejor, res


def paso4a_cortes(pm_orig, pm, b, Xpm_orig, Xpm, Xb, nombre_b):
    """Cortes de PassMark por anio de prueba. Mismos CPUs de test para todos los sistemas."""
    y = pm["cpuMark"].values
    preds = {k: [] for k in ["A (actual)", "A' (nucleos reales)", "C(B)", "A' + pseudo-etiquetas"]}
    reales, por_corte = [], []
    for corte in CORTES:
        tr = pm["testDate"].values <= corte
        te = (pm["testDate"].values > corte) & (pm["testDate"].values <= corte + 2)
        claves_test = set(pm.loc[te, "clave"])
        pA = ajustar(modelo_base(), Xpm_orig[tr], y[tr]).predict(Xpm_orig[te])
        a2 = ajustar(modelo_base(), Xpm[tr], y[tr])
        pA2 = a2.predict(Xpm[te])
        # B y C: solo CPUs lanzados <= corte y que no esten en el test
        okb = (b["anio_lanz"].values <= corte) & ~b["clave"].isin(claves_test).values
        mb = ajustar(candidatos_b()[nombre_b], Xb[okb], b.loc[okb, "blender"].values)
        ov = okb & b["pm_cpuMark"].notna().values & (b["pm_testDate"].values <= corte)
        c = Traductor().fit(Xb[ov].reset_index(drop=True), b.loc[ov, "blender"].values, b.loc[ov, "pm_cpuMark"].values)
        pCB = c.predict(Xpm[te].reset_index(drop=True), np.clip(mb.predict(Xpm[te]), 1e-3, None))
        # pseudo-etiquetas: CPUs que solo estan en Blender (lanzados <= corte) traducidos a PassMark y sumados al train de A'
        solo_b = okb & ~b["en_passmark"].values
        yb = c.predict(Xb[solo_b].reset_index(drop=True), b.loc[solo_b, "blender"].values)
        Xaug = pd.concat([Xpm[tr], Xb[solo_b]], ignore_index=True)
        pAug = ajustar(modelo_base(), Xaug, np.r_[y[tr], yb]).predict(Xpm[te])
        for k, p in zip(preds, [pA, pA2, pCB, pAug]):
            preds[k].append(np.clip(p, 1, None))
        reales.append(y[te])
        por_corte.append({"corte": corte, "n_test": int(te.sum()), "pseudo_etiquetas": int(solo_b.sum()),
                          **{k: metricas(y[te], preds[k][-1])["mae"] for k in preds}})
        print(f"  corte {corte}: " + " | ".join(f"{k} {por_corte[-1][k]:.0f}" for k in preds), flush=True)

    def mezcla(w, base="A' (nucleos reales)"):
        return [np.exp(w * np.log(a) + (1 - w) * np.log(cb)) for a, cb in zip(preds[base], preds["C(B)"])]

    # peso de la mezcla escogido de forma honesta: con 4 cortes, medido en el quinto (rotando)
    pesos = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    honesto = []
    for j in range(len(CORTES)):
        otros = [i for i in range(len(CORTES)) if i != j]
        w = min(pesos, key=lambda w: np.mean([metricas(reales[i], mezcla(w)[i])["mae"] for i in otros]))
        honesto.append(metricas(reales[j], mezcla(w)[j])["mae"])
    w_final = min(pesos, key=lambda w: np.mean([metricas(reales[i], mezcla(w)[i])["mae"] for i in range(len(CORTES))]))
    resumen = {}
    for k in preds:
        resumen[k] = {"mae": float(np.mean([metricas(r, p)["mae"] for r, p in zip(reales, preds[k])])),
                      "kpi": float(np.mean([metricas(r, p)["kpi_dentro_20pct"] for r, p in zip(reales, preds[k])]))}
    resumen[f"mezcla A' + C(B) (peso A {w_final})"] = {
        "mae": float(np.mean(honesto)),
        "kpi": float(np.mean([metricas(r, p)["kpi_dentro_20pct"] for r, p in zip(reales, mezcla(w_final))]))}
    return resumen, w_final, pd.DataFrame(por_corte)


def paso4b_futuro(pm, b, Xpm, Xb, nombre_b, w):
    """Entrenado con lanzamientos <= 2022. Prueba: CPUs 2023-2026 que solo estan en Blender. Se mide el ORDEN."""
    y = pm["cpuMark"].values
    a2 = ajustar(modelo_base(), Xpm, y)
    okb = b["anio_lanz"].values <= ANIO_FUTURO
    mb = ajustar(candidatos_b()[nombre_b], Xb[okb], b.loc[okb, "blender"].values)
    ov = okb & b["pm_cpuMark"].notna().values
    c = Traductor().fit(Xb[ov].reset_index(drop=True), b.loc[ov, "blender"].values, b.loc[ov, "pm_cpuMark"].values)
    fut = (b["anio_lanz"].values > ANIO_FUTURO) & ~b["en_passmark"].values
    Xf = Xb[fut].reset_index(drop=True)
    real = b.loc[fut, "blender"].values
    pA2 = a2.predict(Xf)
    pCB = c.predict(Xf, np.clip(mb.predict(Xf), 1e-3, None))
    final = np.exp(w * np.log(np.clip(pA2, 1, None)) + (1 - w) * np.log(np.clip(pCB, 1, None)))
    out = {"n": int(fut.sum()),
           "solo contar nucleos": spearmanr(Xf["cores"], real)[0],
           "A' (nucleos reales)": spearmanr(pA2, real)[0],
           "C(B)": spearmanr(pCB, real)[0],
           f"mezcla final (peso A {w})": spearmanr(final, real)[0]}
    return out


def paso4c_traducidos(pm, b, Xpm, Xb):
    """La idea cruzada completa: CPUs que solo estan en Blender, traducidos a PassMark con C, se suman al entrenamiento.
    Se entrena con traducidos lanzados <= 2023 (o 2024) y se prueba con los posteriores."""
    y = pm["cpuMark"].values
    ov = b["pm_cpuMark"].notna().values
    c = Traductor().fit(Xb[ov].reset_index(drop=True), b.loc[ov, "blender"].values, b.loc[ov, "pm_cpuMark"].values)
    solo, anio = ~b["en_passmark"].values, b["anio_lanz"].values
    a = ajustar(modelo_base(), Xpm, y)
    filas = []
    for hasta in [2023, 2024]:
        tr, te = solo & (anio <= hasta), solo & (anio > hasta)
        Xt, real = Xb[te].reset_index(drop=True), b.loc[te, "blender"].values
        yb = c.predict(Xb[tr].reset_index(drop=True), b.loc[tr, "blender"].values)
        aug = ajustar(modelo_base(), pd.concat([Xpm, Xb[tr]], ignore_index=True), np.r_[y, yb])
        pa, pg, ref = a.predict(Xt), aug.predict(Xt), c.predict(Xt, real)
        fila = {"entrena_hasta": hasta, "n_test": int(te.sum()), "traducidos": int(tr.sum()),
                "orden_nucleos": spearmanr(Xt["cores"], real)[0], "orden_A2": spearmanr(pa, real)[0],
                "orden_A2_traducidos": spearmanr(pg, real)[0], "dentro20_A2": metricas(ref, pa)["kpi_dentro_20pct"],
                "dentro20_A2_traducidos": metricas(ref, pg)["kpi_dentro_20pct"]}
        filas.append(fila)
        print(f"  hasta {hasta}: test {fila['n_test']} | orden nucleos {fila['orden_nucleos']:.3f} | A' {fila['orden_A2']:.3f} "
              f"| A'+traducidos {fila['orden_A2_traducidos']:.3f} | dentro 20%: A' {fila['dentro20_A2']:.1%} "
              f"vs A'+traducidos {fila['dentro20_A2_traducidos']:.1%}", flush=True)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- todo junto
def snapshot_blender():
    """Nombre del snapshot de Blender en uso (baja uno nuevo si el ultimo tiene mas de 7 dias)."""
    from . import blender
    return blender.descargar().stem          # opendata-AAAA-MM-DD


def ya_corrido(hash_datos, version_pc_build, snapshot):
    """Hay un run del modelo cruzado con exactamente los mismos datos (PassMark + pc_build + Blender)?"""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    if mlflow.get_experiment_by_name(EXP) is None:
        return False
    runs = mlflow.search_runs(experiment_names=[EXP], filter_string=(
        f"tags.data_hash = '{hash_datos}' and tags.version_pc_build = '{version_pc_build}' "
        f"and tags.blender_snapshot = '{snapshot}' and attributes.status = 'FINISHED'"))
    return len(runs) > 0


def correr(info=None):
    """Modelo cruzado completo. Regresa un resumen; registra una version nueva solo si le gana a produccion.
    'info' viene del pipeline (version y hash de PassMark); si no, se descarga aqui."""
    info = info or datos.descargar()
    df = datos.cargar(info["ruta"])
    snap = snapshot_blender()
    _, v_pcb = externos.pc_build()
    modelos.usar_externos(True)
    preparar_mlflow(EXP)
    print("[1] fichas tecnicas y Blender", flush=True)
    f = fichas()
    bl = blender_por_clave()
    pm, b = construir(df, f, bl)
    pm_y = pm.drop_duplicates("clave").set_index("clave")["cpuMark"]
    b["pm_cpuMark"] = b["clave"].map(pm_y)
    Xpm_orig, _ = datos.preparar(df, externos=True)
    Xpm = matriz(pm, "testDate")
    b_ok = b[b["anio_lanz"].notna() | b["en_passmark"]].reset_index(drop=True)
    Xb = matriz(b_ok, "anio_lanz")
    cobertura = {"passmark": len(pm), "nucleos_corregidos": int((pm["cores"] != pm["cores_pm"]).sum()),
                 "blender_cpus": len(b), "blender_con_anio": int(b["anio_lanz"].notna().sum()),
                 "blender_con_tdp": int(b["TDP"].notna().sum()), "en_ambos": int(b_ok["pm_cpuMark"].notna().sum()),
                 "solo_blender": int((~b_ok["en_passmark"]).sum())}
    print("  ", cobertura, flush=True)
    print("  ejemplos de nucleos corregidos:", pm.loc[pm["cores"] != pm["cores_pm"], ["cpuName", "cores_pm", "cores"]]
          .head(6).values.tolist(), flush=True)

    with mlflow.start_run(run_name=f"cruzado__data-v{info['version']}",
                          tags={"etapa": "cruzado", "data_version": info["version"], "data_hash": info["hash"],
                                "version_codigo": config.version_codigo(), "fuente_blender": "opendata.blender.org CC0",
                                "blender_snapshot": snap, "version_pc_build": v_pcb}):
        mlflow.log_params({k: v for k, v in cobertura.items()})
        print("[2] traductor C (Blender -> PassMark), kfold en los CPUs que estan en ambos", flush=True)
        rc = paso2_traductor(b_ok, Xb)
        mlflow.log_metrics({f"C_{'simple' if 'simple' in k else 'completo'}_{m}": v[m] for k, v in rc.items()
                            for m in ["mae", "kpi_dentro_20pct", "r2_log"]})
        print("[3] modelo B (predice Blender)", flush=True)
        nombre_b, rb = paso3_b(b_ok, Xb)
        mlflow.set_tag("modelo_b", nombre_b)
        print(f"  gana {nombre_b}", flush=True)
        print("[4a] sistema completo en los cortes de PassMark (MAE en cpuMark)", flush=True)
        r4, w, tabla = paso4a_cortes(df, pm, b_ok, Xpm_orig, Xpm, Xb, nombre_b)
        for k, v in r4.items():
            print(f"  {k:<40} MAE {v['mae']:6.0f} | dentro 20% {v['kpi']:.1%}", flush=True)
        mlflow.log_metrics({re.sub(r"[^a-zA-Z0-9_]", "_", f"cortes_{k}")[:60]: v["mae"] for k, v in r4.items()})
        tabla.to_csv(config.REPORTES / "cruzado_cortes.csv", index=False)
        mlflow.log_artifact(str(config.REPORTES / "cruzado_cortes.csv"))
        print("[4b] prueba del futuro: CPUs 2023-2026 que solo estan en Blender (correlacion de orden)", flush=True)
        r5 = paso4b_futuro(pm, b_ok, Xpm, Xb, nombre_b, w)
        for k, v in r5.items():
            print(f"  {k:<40} {v if k == 'n' else round(v, 3)}", flush=True)
        mlflow.log_metrics({re.sub(r"[^a-zA-Z0-9_]", "_", f"futuro_{k}")[:60]: float(v) for k, v in r5.items()})
        print("[4c] CPUs de Blender traducidos y sumados al entrenamiento: prueba con CPUs posteriores", flush=True)
        r6 = paso4c_traducidos(pm, b_ok, Xpm, Xb)
        r6.to_csv(config.REPORTES / "cruzado_traducidos.csv", index=False)
        mlflow.log_artifact(str(config.REPORTES / "cruzado_traducidos.csv"))
        mlflow.log_metrics({f"traducidos_{c}_{int(r.entrena_hasta)}": float(r[c]) for _, r in r6.iterrows()
                            for c in ["orden_A2", "orden_A2_traducidos", "dentro20_A2", "dentro20_A2_traducidos"]})
        json.dump({"cobertura": cobertura, "traductor": rc, "modelo_b": rb, "cortes": r4, "peso_a": w, "futuro": r5,
                   "traducidos": r6.to_dict("records")},
                  open(config.REPORTES / "cruzado_resumen.json", "w"), indent=2, default=float)
        mlflow.log_artifact(str(config.REPORTES / "cruzado_resumen.json"))

        # 5. se escoge el mejor de los candidatos (MAE promedio de 5 cortes) y se registra si le gana a produccion
        candidatos = {k: v["mae"] for k, v in r4.items() if k != "A (actual)" and not k.startswith("C(B)")}
        ganador = min(candidatos, key=candidatos.get)
        mae_sistema = candidatos[ganador]
        cliente = MlflowClient()
        prod = cliente.get_model_version_by_alias(config.MODELO_REGISTRADO, config.ALIAS_PRODUCCION)
        mae_prod = float(prod.tags.get(config.METRICA_SELECCION, "inf"))
        print(f"[5] mejor candidato: {ganador} ({mae_sistema:.0f}) vs produccion v{prod.version} ({mae_prod:.0f})", flush=True)
        mlflow.set_tag("ganador", ganador)
        salida = {"ganador": ganador, "mae": mae_sistema, "produccion_version": int(prod.version), "mae_produccion": mae_prod,
                  "blender_snapshot": snap, "registrado": None}
        if mae_sistema >= mae_prod * (1 - config.MEJORA_MINIMA):     # margen: diferencias de <1% son ruido
            print(f"  -> no se registra: no mejora al menos {config.MEJORA_MINIMA:.0%} al modelo en produccion", flush=True)
            return salida
        y = pm["cpuMark"].values
        ov = b_ok["pm_cpuMark"].notna().values
        c_fin = Traductor().fit(Xb[ov].reset_index(drop=True), b_ok.loc[ov, "blender"].values, b_ok.loc[ov, "pm_cpuMark"].values)
        if ganador.startswith("mezcla"):
            b_fin = ajustar(candidatos_b()[nombre_b], Xb[b_ok["anio_lanz"].notna().values],
                            b_ok.loc[b_ok["anio_lanz"].notna(), "blender"].values)
            final = SistemaCruzado(ajustar(modelo_base(), Xpm, y), b_fin, c_fin, w)
            descripcion = f"mezcla A' + C(B {nombre_b}), peso A {w}"
        elif "pseudo" in ganador:
            solo = ~b_ok["en_passmark"].values & b_ok["anio_lanz"].notna().values     # todos los que solo estan en Blender
            yb = c_fin.predict(Xb[solo].reset_index(drop=True), b_ok.loc[solo, "blender"].values)
            final = ajustar(modelo_base(), pd.concat([Xpm, Xb[solo]], ignore_index=True), np.r_[y, yb])
            descripcion = f"combo con nucleos reales + {int(solo.sum())} CPUs de Blender traducidos a PassMark"
        else:
            final = ajustar(modelo_base(), Xpm, y)
            descripcion = "combo con nucleos reales"
        info_m = mlflow.sklearn.log_model(final, name="modelo", input_example=Xpm.head(3),
                                          signature=infer_signature(Xpm.head(50), final.predict(Xpm.head(50))),
                                          serialization_format="cloudpickle", code_paths=[str(config.RAIZ / "src")])
        v = mlflow.register_model(info_m.model_uri, config.MODELO_REGISTRADO)
        for k, val in {"modelo": "cruzado: " + descripcion, config.METRICA_SELECCION: f"{mae_sistema:.1f}",
                       "data_version": info["version"], "data_hash": info["hash"], "fuente_blender": "opendata.blender.org CC0",
                       "nucleos": "reales (fichas oficiales / Blender)", "version_codigo": config.version_codigo(),
                       "blender_snapshot": snap, "version_pc_build": v_pcb}.items():
            cliente.set_model_version_tag(config.MODELO_REGISTRADO, v.version, k, str(val))
        cliente.set_registered_model_alias(config.MODELO_REGISTRADO, config.ALIAS_PRODUCCION, v.version)
        print(f"  -> version {v.version} ({descripcion}) ahora es 'produccion'", flush=True)
        return salida | {"registrado": int(v.version)}

if __name__ == "__main__":
    correr()
