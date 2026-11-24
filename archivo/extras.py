"""Experimentos extra (X1-X6): lo que no se habia probado, para justificar las decisiones del proyecto.
Todo se mide igual que la busqueda: 5 cortes temporales (train <= corte, test = los 2 anios siguientes).

  X1  boosting de otras librerias (XGBoost, LightGBM, CatBoost) con su propia busqueda aleatoria, solos y como
      segundo paso del hibrido. CatBoost tambien con sus categorias "nativas" (sin one-hot).
  X2  restricciones logicas (monotonia): "mas nucleos / mas nuevo nunca baja el puntaje" + prueba de sentido comun
  X3  intervalos de confianza: conformal temporal y regresion por cuantiles (cobertura real vs prometida)
  X4  optimizacion bayesiana (Optuna) vs la busqueda aleatoria, con el mismo presupuesto
  X5  lineas base (lo minimo que hay que superar) y un modelo que solo lee el nombre del procesador
  X6  ablacion temporal: quitar un grupo de variables / una pieza del hibrido a la vez

Correr:  python -m archivo.extras            (todos)
         python -m archivo.extras x2 x3      (solo algunos)
"""
import re
import sys
import time

import mlflow
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from threadpoolctl import threadpool_limits

from src import config, datos, modelos
from archivo.busqueda import VIDAS, cortes_cv, envuelto, familias, limpio, p
from src.entrenar import metricas, preparar_mlflow

EXP = "cpu-perf/extras"
S = config.SEMILLA
CORTES = config.CORTES_TEMPORALES

# la mejor configuracion de la busqueda aleatoria (hibrido recta + HGB): referencia de todos los extras
REF = dict(alpha=235.7730502596829, vida_media=2, n_estimators=200, min_samples_leaf=34)


def hgb_hibrido(**kw):
    """El mismo HGB que usa modelos.Hibrido(segundo='hgb') como segundo paso."""
    base = dict(learning_rate=0.05, max_iter=REF["n_estimators"], max_leaf_nodes=15, min_samples_leaf=REF["min_samples_leaf"],
                l2_regularization=1.0, random_state=S)
    return HistGradientBoostingRegressor(**(base | kw))


def referencia():
    return modelos.con_log(modelos.Hibrido(segundo="hgb", externos=False, **REF))


# ---------------------------------------------------------------- piezas
GRUPOS = {   # para la ablacion (X6): que columnas se "apagan" en cada grupo
    "nucleos": ["cores", "hilos"],
    "anio": ["anio"],
    "TDP": ["TDP"],
    "frecuencia": ["ghz_nombre", "base_ghz", "turbo_ghz"],
    "generacion/gama/sufijo/familia (del nombre)": ["generacion", "gama", "sufijo", "familia"],
    "fabricante": ["fabricante"],
    "socket": ["socket"],
    "tipo (desktop/laptop/server/movil)": ["es_desktop", "es_laptop", "es_server", "es_mobile"],
    "externos (cache, nm, GHz, hilos)": modelos.EXTERNAS,
}
SIGNOS = {   # X2: direccion logica de cada variable (+1 sube el puntaje, -1 lo baja)
    "suave": {"cores": 1, "hilos": 1, "anio": 1},
    "completo": {"cores": 1, "hilos": 1, "anio": 1, "ghz_nombre": 1, "base_ghz": 1, "turbo_ghz": 1, "l3_mb": 1,
                 "TDP": 1, "generacion": 1, "proceso_nm": -1},
}


def apagar(X, columnas):
    """Quita la informacion de unas columnas sin cambiar el esquema: numericas -> vacio (se imputa la mediana, o sea
    una constante), categoricas -> un solo valor, banderas -> 0."""
    X = X.copy()
    for c in columnas or []:
        if c not in X:
            continue
        if c in modelos.CATEGORICAS:
            X[c] = "oculto"
        elif c in modelos.BANDERAS:
            X[c] = 0
        else:
            X[c] = np.nan
    return X


def restricciones(nombres, signos):
    """Vector monotonic_cst para HGB a partir de los nombres que salen del preprocesamiento (num__cores, ...)."""
    return [signos.get(n.split("__", 1)[1], 0) if n.startswith("num__") else 0 for n in nombres]


class HibridoX(RegressorMixin, BaseEstimator):
    """Como modelos.Hibrido (recta Ridge para la tendencia + un segundo modelo para el residuo), pero el segundo modelo
    puede ser cualquier estimador (XGBoost, LightGBM, CatBoost, HGB con monotonia...), se pueden 'apagar' columnas
    (ablacion) y se puede quitar cualquiera de los dos pasos. Va dentro de con_log, asi que y llega en log."""
    def __init__(self, segundo=None, alpha=REF["alpha"], vida_media=REF["vida_media"], externos=False, ocultar=None,
                 signos=None, sin_recta=False, sin_segundo=False):
        self.segundo = segundo
        self.alpha = alpha
        self.vida_media = vida_media
        self.externos = externos
        self.ocultar = ocultar
        self.signos = signos
        self.sin_recta = sin_recta
        self.sin_segundo = sin_segundo

    def fit(self, X, y):
        y = np.asarray(y, float)
        w = (0.5 ** ((X["anio"].max() - X["anio"]) / self.vida_media)).values if self.vida_media else None
        Xo = apagar(X, self.ocultar)
        self.prep_ = modelos.preprocesamiento(externos=self.externos).fit(Xo)
        Xt = self.prep_.transform(Xo)
        if self.sin_recta:
            self.recta_, residuo = None, y
        else:
            self.recta_ = Ridge(alpha=self.alpha).fit(Xt, y, sample_weight=w)
            residuo = y - self.recta_.predict(Xt)
        self.seg_ = None
        if not self.sin_segundo:
            est = clone(self.segundo if self.segundo is not None else hgb_hibrido())
            if self.signos:
                est.set_params(monotonic_cst=restricciones(self.prep_.get_feature_names_out(), self.signos))
            self.seg_ = est.fit(Xt, residuo, sample_weight=w)
        return self

    def predict(self, X):
        Xt = self.prep_.transform(apagar(X, self.ocultar))
        p_ = np.zeros(len(X)) if self.recta_ is None else self.recta_.predict(Xt)
        return p_ + (0 if self.seg_ is None else np.ravel(self.seg_.predict(Xt)))


class CatNativo(RegressorMixin, BaseEstimator):
    """CatBoost con sus categorias nativas (sin one-hot, sin imputar: CatBoost maneja vacios y texto categorico)."""
    def __init__(self, iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3.0, vida_media=None):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.vida_media = vida_media

    def _tabla(self, X):
        T = X.copy()
        for c in modelos.CATEGORICAS:
            T[c] = T[c].astype(str)
        return T

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        w = (0.5 ** ((X["anio"].max() - X["anio"]) / self.vida_media)).values if self.vida_media else None
        self.m_ = CatBoostRegressor(iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth,
                                    l2_leaf_reg=self.l2_leaf_reg, loss_function="MAE", random_seed=S, verbose=0,
                                    thread_count=1, allow_writing_files=False,
                                    cat_features=modelos.CATEGORICAS)
        self.m_.fit(self._tabla(X), y, sample_weight=w)
        return self

    def predict(self, X):
        return self.m_.predict(self._tabla(X))


class SoloNombre(RegressorMixin, BaseEstimator):
    """Ridge sobre pedacitos de 2-5 letras del nombre (TF-IDF de caracteres). con_specs=True agrega la ficha tecnica."""
    def __init__(self, alpha=1.0, con_specs=False):
        self.alpha = alpha
        self.con_specs = con_specs

    def fit(self, X, y):
        partes = [("txt", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, min_df=2, sublinear_tf=True),
                   "cpuName")]
        if self.con_specs:
            partes.append(("specs", modelos.preprocesamiento(externos=True), [c for c in X.columns if c != "cpuName"]))
        self.m_ = make_pipeline(ColumnTransformer(partes), Ridge(alpha=self.alpha)).fit(X, y)
        return self

    def predict(self, X):
        return self.m_.predict(X)


# ---------------------------------------------------------------- evaluacion en los 5 cortes
def _un_corte(hacer, X, y, corte):
    tr, te = X["anio"] <= corte, (X["anio"] > corte) & (X["anio"] <= corte + 2)
    with threadpool_limits(1):
        m = hacer().fit(X[tr], y[tr])
        return corte, np.where(te)[0], m.predict(X[te])


def en_cortes(hacer, X, y, n_jobs=5):
    """Entrena/predice los 5 cortes en paralelo. Regresa el resumen con el MAE de cada corte."""
    salida = Parallel(n_jobs=n_jobs)(delayed(_un_corte)(hacer, X, y, c) for c in CORTES)
    maes, kpis, det = [], [], {}
    for corte, idx, pred in salida:
        r = metricas(y.iloc[idx], pred)
        maes.append(r["mae"]); kpis.append(r["kpi_dentro_20pct"])
        det[corte] = r["mae"]
    res = {"mae": float(np.mean(maes)), "mae_std": float(np.std(maes)), "kpi_dentro_20pct": float(np.mean(kpis)),
           "peor_corte_mae": float(np.max(maes)), **{f"mae_corte_{c}": v for c, v in det.items()}}
    return res


FILAS = []


def anotar(experimento, variante, m, params=None, tags=None):
    with mlflow.start_run(run_name=f"{experimento}__{variante}"[:250], nested=True,
                          tags={"experimento": experimento, "variante": variante, "version_codigo": config.version_codigo(),
                                "cv": "5 cortes temporales", **(tags or {})}):
        mlflow.log_metrics({re.sub(r"[^a-zA-Z0-9_.: /-]", "_", k): float(v) for k, v in m.items() if isinstance(v, (int, float, np.floating)) and np.isfinite(v)})
        if params:
            mlflow.log_params({k: limpio(v) for k, v in params.items()})
    extra = f" | dentro 20% {m['kpi_dentro_20pct']:.1%}" if "kpi_dentro_20pct" in m else ""
    mae = f"MAE 5 cortes {m['mae']:6.0f} ± {m.get('mae_std', 0):4.0f}" if "mae" in m else ""
    print(f"   {variante:<52} {mae}{extra}", flush=True)
    FILAS.append({"experimento": experimento, "variante": variante, **m})
    return m


# ---------------------------------------------------------------- X1 boosting de otras librerias
def x1_boosting(X, y, df):
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    print("X1 - XGBoost / LightGBM / CatBoost (busqueda aleatoria con los 5 cortes, igual que las 11 familias)")
    v = {"regressor__vida_media": VIDAS, "regressor__externos": [True, False]}
    espacios = {
        "xgboost": (envuelto(xgb.XGBRegressor(n_jobs=1, random_state=S, tree_method="hist")),
                    {**p({"n_estimators": [200, 400, 800], "learning_rate": loguniform(0.01, 0.3), "max_depth": randint(2, 9),
                              "min_child_weight": loguniform(1, 50), "subsample": uniform(0.5, 0.5),
                              "colsample_bytree": uniform(0.4, 0.6), "reg_lambda": loguniform(1e-2, 30)}), **v}, 30),
        "lightgbm": (envuelto(lgb.LGBMRegressor(n_jobs=1, random_state=S, verbose=-1, subsample_freq=1)),
                     {**p({"n_estimators": [200, 400, 800], "learning_rate": loguniform(0.01, 0.3), "num_leaves": randint(4, 64),
                               "min_child_samples": randint(5, 80), "subsample": uniform(0.5, 0.5),
                               "colsample_bytree": uniform(0.4, 0.6), "reg_lambda": loguniform(1e-2, 30)}), **v}, 30),
        "catboost": (envuelto(CatBoostRegressor(thread_count=1, random_seed=S, verbose=0, allow_writing_files=False)),
                     {**p({"iterations": [300, 600, 1000], "learning_rate": loguniform(0.01, 0.3), "depth": randint(3, 9),
                               "l2_leaf_reg": loguniform(1, 30)}), **v}, 20),
    }
    cv = cortes_cv(X)
    for nombre, (est, espacio, n_iter) in espacios.items():
        espacio = {k.replace("modelo____", "modelo__"): val for k, val in espacio.items()}
        t = time.time()
        b = RandomizedSearchCV(est, espacio, n_iter=n_iter, cv=cv, scoring="neg_mean_absolute_error", n_jobs=8,
                               random_state=S, error_score=np.nan, refit=False)
        with threadpool_limits(1):
            b.fit(X, y)
        r = pd.DataFrame(b.cv_results_).sort_values("rank_test_score")
        mejor = r.iloc[0]
        cortes = {f"mae_corte_{c}": float(-mejor[f"split{j}_test_score"]) for j, c in enumerate(CORTES)}
        anotar("X1_boosting", f"{nombre} solo (mejor de {n_iter} configs)",
               {"mae": -mejor["mean_test_score"], "mae_std": mejor["std_test_score"], "peor_corte_mae": max(cortes.values()),
                "mediana_configs": float(-r["mean_test_score"].median()), "segundos": time.time() - t, **cortes},
               params={k.split("__")[-1]: val for k, val in mejor["params"].items()})
        r.drop(columns=["params"]).to_csv(config.REPORTES / f"extras_x1_{nombre}.csv", index=False)

    # como segundo paso del hibrido: misma recta y mismos pesos que la referencia, solo cambia quien corrige el residuo
    segundos = {
        "HGB (referencia, scikit-learn)": hgb_hibrido(),
        "XGBoost": xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, min_child_weight=34, reg_lambda=1.0,
                                    n_jobs=1, random_state=S, tree_method="hist"),
        "LightGBM": lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=34, reg_lambda=1.0,
                                      n_jobs=1, random_state=S, verbose=-1),
        "CatBoost": CatBoostRegressor(iterations=200, learning_rate=0.05, depth=4, l2_leaf_reg=3, thread_count=1,
                                      random_seed=S, verbose=0, allow_writing_files=False),
    }
    for nombre, est in segundos.items():
        anotar("X1_boosting", f"hibrido: recta + {nombre}",
               en_cortes(lambda e=est: modelos.con_log(HibridoX(segundo=e)), X, y))
    anotar("X1_boosting", "CatBoost con categorias nativas (sin one-hot)",
           en_cortes(lambda: modelos.con_log(CatNativo(vida_media=2)), X, y))


# ---------------------------------------------------------------- X2 monotonia
def sentido_comun(modelo, X):
    """% de procesadores donde DUPLICAR los nucleos (e hilos) hace que la prediccion BAJE. Un modelo logico da 0%."""
    X2 = X.copy()
    X2["cores"] = X2["cores"] * 2
    if "hilos" in X2:
        X2["hilos"] = X2["hilos"] * 2
    a, b = modelo.predict(X), modelo.predict(X2)
    X3 = X.copy()
    X3["anio"] = X3["anio"] + 2
    c = modelo.predict(X3)
    return float(np.mean(b < a * 0.999)), float(np.median(b / a)), float(np.mean(c < a * 0.999))


def x2_monotonia(X, y, df):
    print("X2 - restricciones logicas (monotonia) + prueba de sentido comun en el corte 2020")
    variantes = {
        "HGB solo (sin restricciones)": lambda: modelos.con_log(HibridoX(sin_recta=True, externos=True)),
        "HGB solo + monotonia suave (nucleos, hilos, anio)": lambda: modelos.con_log(HibridoX(sin_recta=True, externos=True, signos=SIGNOS["suave"])),
        "HGB solo + monotonia completa": lambda: modelos.con_log(HibridoX(sin_recta=True, externos=True, signos=SIGNOS["completo"])),
        "hibrido referencia (sin restricciones)": referencia,
        "hibrido + monotonia suave en el 2o paso": lambda: modelos.con_log(HibridoX(signos=SIGNOS["suave"])),
        "hibrido + monotonia completa en el 2o paso": lambda: modelos.con_log(HibridoX(signos=SIGNOS["completo"])),
    }
    tr, te = X["anio"] <= 2020, X["anio"] > 2020
    for nombre, hacer in variantes.items():
        m = en_cortes(hacer, X, y)
        with threadpool_limits(1):
            mod = hacer().fit(X[tr], y[tr])
        baja, mediana, baja_anio = sentido_comun(mod, X[te])
        anotar("X2_monotonia", nombre, m | {"pct_baja_al_duplicar_nucleos": baja, "factor_mediano_al_duplicar_nucleos": mediana,
                                            "pct_baja_al_sumar_2_anios": baja_anio})
        print(f"   {'':<52} al duplicar nucleos baja en {baja:.1%} | sube x{mediana:.2f} (mediana) | "
              f"+2 anios baja en {baja_anio:.1%}", flush=True)


# ---------------------------------------------------------------- X3 intervalos
def x3_intervalos(X, y, df, nivel=0.80):
    """Tres formas de dar un rango "entre A y B" en vez de un solo numero, y cuanto cumplen lo prometido."""
    print(f"X3 - intervalos de confianza al {nivel:.0%}: cobertura REAL en los 5 cortes")
    ly = np.log1p(y)

    def resumen(nombre, cubre, ancho, por_corte):
        m = {"cobertura": float(np.mean(cubre)), "ancho_mediano_pct": float(np.median(ancho)), "nivel_prometido": nivel, **por_corte}
        anotar("X3_intervalos", nombre, m)
        print(f"   {'':<52} prometido {nivel:.0%} -> cubre {m['cobertura']:.1%} | rango tipico ±{m['ancho_mediano_pct'] / 2:.0%}",
              flush=True)

    def conformal(residuos_de):
        cubre, ancho, por_corte = [], [], {}
        for corte in CORTES:
            tr, te = X["anio"] <= corte, (X["anio"] > corte) & (X["anio"] <= corte + 2)
            res = residuos_de(corte)
            q = np.quantile(res, min(1.0, nivel * (1 + 1 / len(res))))
            with threadpool_limits(1):
                lp = np.log1p(referencia().fit(X[tr], y[tr]).predict(X[te]))
            lo, hi = np.expm1(lp - q), np.expm1(lp + q)
            c = (y[te].values >= lo) & (y[te].values <= hi)
            cubre += list(c); ancho += list((hi - lo) / np.expm1(lp))
            por_corte[f"cobertura_corte_{corte}"] = float(c.mean())
        return cubre, ancho, por_corte

    def res_kfold(corte):       # (a) la forma "de libro": errores de validacion aleatoria dentro del train
        from sklearn.model_selection import KFold
        tr = np.where(X["anio"] <= corte)[0]
        r = []
        for a, b in KFold(5, shuffle=True, random_state=S).split(tr):
            with threadpool_limits(1):
                m = referencia().fit(X.iloc[tr[a]], y.iloc[tr[a]])
            r += list(np.abs(ly.iloc[tr[b]] - np.log1p(m.predict(X.iloc[tr[b]]))))
        return np.array(r)

    def res_futuro(corte):      # (b) temporal: lo que fallo un modelo entrenado 2 anios antes al predecir 2 anios adelante
        cal_tr, cal = X["anio"] <= corte - 2, (X["anio"] > corte - 2) & (X["anio"] <= corte)
        with threadpool_limits(1):
            m = referencia().fit(X[cal_tr], y[cal_tr])
        return np.abs(ly[cal] - np.log1p(m.predict(X[cal]))).values

    resumen("conformal con errores de k-fold aleatorio (de libro)", *conformal(res_kfold))
    resumen("conformal temporal (errores de predecir 2 anios adelante)", *conformal(res_futuro))

    cubre, ancho, por_corte = [], [], {}
    bajo, alto = (1 - nivel) / 2, 1 - (1 - nivel) / 2
    for corte in CORTES:
        tr, te = X["anio"] <= corte, (X["anio"] > corte) & (X["anio"] <= corte + 2)
        preds = {}
        for qq in [bajo, 0.5, alto]:
            with threadpool_limits(1):
                m = modelos.con_log(HibridoX(segundo=hgb_hibrido(loss="quantile", quantile=qq), sin_recta=True, externos=True))
                preds[qq] = m.fit(X[tr], y[tr]).predict(X[te])
        c = (y[te].values >= preds[bajo]) & (y[te].values <= preds[alto])
        cubre += list(c); ancho += list((preds[alto] - preds[bajo]) / preds[0.5])
        por_corte[f"cobertura_corte_{corte}"] = float(c.mean())
    resumen(f"regresion por cuantiles (HGB p{bajo * 100:.0f} y p{alto * 100:.0f})", cubre, ancho, por_corte)


# ---------------------------------------------------------------- X4 Optuna vs busqueda aleatoria
def x4_optuna(X, y, df, n_trials=30):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"X4 - Optuna (bayesiano, TPE) vs busqueda aleatoria, {n_trials} intentos cada uno, hibrido recta + HGB")

    def evaluar(params):
        return en_cortes(lambda: modelos.con_log(modelos.Hibrido(segundo="hgb", **params)), X, y)["mae"]

    def objetivo(trial):
        return evaluar({"alpha": trial.suggest_float("alpha", 0.1, 300, log=True),
                        "vida_media": trial.suggest_categorical("vida_media", VIDAS),
                        "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 400]),
                        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 40),
                        "externos": trial.suggest_categorical("externos", [True, False])})

    t = time.time()
    estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=S))
    estudio.optimize(objetivo, n_trials=n_trials)
    seg = time.time() - t
    curva_opt = np.minimum.accumulate([tr.value for tr in estudio.trials])

    # la busqueda aleatoria ya corrida (busqueda.py): se reconstruye el ORDEN en que probo las configuraciones
    _, espacio, n_iter = familias()["hibrido_hgb"]
    orden = list(ParameterSampler(espacio, n_iter=n_iter, random_state=S))
    previa = pd.read_csv(config.REPORTES / "busqueda_hibrido_hgb.csv")
    mae_de = {round(float(a), 6): m for a, m in zip(previa["param_regressor__alpha"], previa["mae"])}
    curva_rnd = np.minimum.accumulate([mae_de.get(round(float(o["regressor__alpha"]), 6), np.nan) for o in orden][:n_trials])

    hitos = {f"optuna_mejor_tras_{k}": float(curva_opt[k - 1]) for k in [5, 10, 20, n_trials]}
    hitos |= {f"aleatoria_mejor_tras_{k}": float(curva_rnd[k - 1]) for k in [5, 10, 20, n_trials] if k <= len(curva_rnd)}
    anotar("X4_optuna", "Optuna TPE vs RandomizedSearch", {"mae": float(curva_opt[-1]), "segundos": seg, **hitos},
           params=estudio.best_params)
    print(f"   mejor tras 5/10/20/{n_trials} intentos -> Optuna {[round(curva_opt[k - 1]) for k in [5, 10, 20, n_trials]]} "
          f"| aleatoria {[round(curva_rnd[k - 1]) for k in [5, 10, 20, n_trials] if k <= len(curva_rnd)]}", flush=True)
    pd.DataFrame({"intento": range(1, n_trials + 1), "optuna": curva_opt,
                  "aleatoria": np.r_[curva_rnd, [np.nan] * (n_trials - len(curva_rnd))]}).to_csv(
        config.REPORTES / "extras_x4_curvas.csv", index=False)


# ---------------------------------------------------------------- X5 lineas base y "solo el nombre"
def x5_bases(X, y, df):
    print("X5 - lineas base (lo minimo a superar) y modelos que leen el nombre")
    variantes = {
        "mediana (predice siempre lo mismo)": lambda: modelos.con_log(DummyRegressor(strategy="median")),
        "solo el anio (tendencia tipo ley de Moore)": lambda: modelos.con_log(make_pipeline(
            ColumnTransformer([("a", "passthrough", ["anio"])]), Ridge(alpha=1e-3))),
        "solo nucleos (log)": lambda: modelos.con_log(make_pipeline(
            ColumnTransformer([("c", FunctionTransformer(np.log1p), ["cores"])]), Ridge(alpha=1e-3))),
        "nucleos + anio": lambda: modelos.con_log(make_pipeline(ColumnTransformer([
            ("c", FunctionTransformer(np.log1p), ["cores"]), ("a", "passthrough", ["anio"])]), Ridge(alpha=1e-3))),
        "Ridge con toda la ficha": lambda: modelos.con_log(modelos.ConPesos(Ridge(alpha=10), externos=True)),
        "hibrido referencia (produccion base)": referencia,
    }
    Xn = X.assign(cpuName=df["cpuName"].values)
    for nombre, hacer in variantes.items():
        anotar("X5_bases", nombre, en_cortes(hacer, X, y))
    for alpha in [0.3, 1.0]:
        anotar("X5_bases", f"solo el NOMBRE (letras, TF-IDF + Ridge a={alpha})",
               en_cortes(lambda a=alpha: modelos.con_log(SoloNombre(alpha=a)), Xn, y))
    anotar("X5_bases", "nombre + ficha tecnica (TF-IDF + Ridge)",
           en_cortes(lambda: modelos.con_log(SoloNombre(alpha=1.0, con_specs=True)), Xn, y))


# ---------------------------------------------------------------- X6 ablacion
def x6_ablacion(X, y, df):
    print("X6 - ablacion temporal sobre el hibrido (con externos): quitar UNA cosa a la vez")
    base = lambda **kw: (lambda: modelos.con_log(HibridoX(externos=True, **kw)))
    ref = anotar("X6_ablacion", "COMPLETO (recta + HGB, pesos, log, todas las variables)", en_cortes(base(), X, y))
    for grupo, cols in GRUPOS.items():
        m = en_cortes(base(ocultar=cols), X, y)
        anotar("X6_ablacion", f"sin {grupo}", m | {"cambio_mae": m["mae"] - ref["mae"]})
    piezas = {
        "sin pesos por antiguedad": base(vida_media=None),
        "sin la recta (solo HGB)": base(sin_recta=True),
        "sin el 2o paso (solo recta)": base(sin_segundo=True),
        "sin logaritmo en el objetivo": lambda: HibridoX(externos=True),
    }
    for nombre, hacer in piezas.items():
        m = en_cortes(hacer, X, y)
        anotar("X6_ablacion", nombre, m | {"cambio_mae": m["mae"] - ref["mae"]})


TODOS = {"x1": x1_boosting, "x2": x2_monotonia, "x3": x3_intervalos, "x4": x4_optuna, "x5": x5_bases, "x6": x6_ablacion}


def correr(cuales=None):
    info = datos.descargar()
    df = datos.cargar(info["ruta"])
    modelos.usar_externos(True)
    X, _ = datos.preparar(df, externos=True)
    y = df[config.OBJETIVO]
    preparar_mlflow(EXP)
    config.REPORTES.mkdir(exist_ok=True)
    for clave in (cuales or TODOS):
        t = time.time()
        with mlflow.start_run(run_name=f"{clave}__data-v{info['version']}",
                              tags={"data_version": info["version"], "data_hash": info["hash"], "etapa": "experimento_extra",
                                    "version_codigo": config.version_codigo()}):
            TODOS[clave](X, y, df)
        print(f"   ({time.time() - t:.0f} s)\n", flush=True)
    tabla = pd.DataFrame(FILAS)
    ruta = config.REPORTES / "extras.csv"
    if ruta.exists() and cuales:
        viejo = pd.read_csv(ruta)
        tabla = pd.concat([viejo[~viejo.experimento.isin(tabla.experimento.unique())], tabla])
    tabla.to_csv(ruta, index=False)
    return tabla


if __name__ == "__main__":
    correr(sys.argv[1:] or None)
