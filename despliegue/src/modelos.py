"""Catalogo de modelos. Todos predicen log(1 + objetivo) por dentro, porque el puntaje va de ~100 a ~100,000."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler, TargetEncoder

from .config import SEMILLA

NUMERICAS = ["cores", "TDP", "anio", "ghz_nombre", "generacion", "gama"]
CATEGORICAS = ["fabricante", "familia", "sufijo"]
ALTA_CARDINALIDAD = ["socket"]       # 200+ valores: target encoding en vez de one-hot
BANDERAS = ["es_desktop", "es_laptop", "es_server", "es_mobile"]


def log_cores_tdp(X):
    # cores y TDP crecen multiplicando (2, 4, 8, 16...), en log la relacion con el rendimiento es mas lineal
    X = X.copy()
    X[:, 0] = np.log1p(X[:, 0])
    X[:, 1] = np.log1p(X[:, 1])
    return X


EXTERNAS = ["l3_mb", "proceso_nm", "base_ghz", "turbo_ghz", "hilos"]
EXTRA_NUMERICAS = []     # transfer learning: columnas extra (pista de otro modelo, indicador de fuente); vacio = normal


def usar_externos(si=True):
    """Agrega o quita las columnas de los datos externos en la parte numerica del preprocesamiento."""
    global NUMERICAS
    base = [c for c in NUMERICAS if c not in EXTERNAS]
    NUMERICAS = base + EXTERNAS if si else base


def preprocesamiento(polinomio=0, externos=None):
    """externos=None usa lo que diga usar_externos(); True/False lo fija (necesario en procesos paralelos, donde el
    valor global no se hereda)."""
    numericas = NUMERICAS if externos is None else [c for c in NUMERICAS if c not in EXTERNAS] + (EXTERNAS if externos else [])
    numericas = list(numericas) + [c for c in EXTRA_NUMERICAS if c not in numericas]
    pasos_num = [SimpleImputer(strategy="median", add_indicator=True),   # el indicador avisa si TDP o ghz faltaban
                 FunctionTransformer(log_cores_tdp, feature_names_out="one-to-one")]
    if polinomio:
        pasos_num.append(PolynomialFeatures(degree=polinomio, include_bias=False))
    pasos_num.append(StandardScaler())
    return ColumnTransformer([
        ("num", make_pipeline(*pasos_num), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CATEGORICAS),
        ("socket", OneHotEncoder(handle_unknown="ignore", min_frequency=10), ALTA_CARDINALIDAD),   # E3: gano a target encoding
        ("banderas", "passthrough", BANDERAS),
    ], sparse_threshold=0)        # siempre matriz normal (HGB no acepta dispersa)


def mlp(capas=(64, 32), activacion="relu", alpha=1e-3, semilla=SEMILLA):
    """Red neuronal (perceptron multicapa, clase 12). early_stopping: aparta 10% y para cuando deja de mejorar."""
    return MLPRegressor(hidden_layer_sizes=capas, activation=activacion, alpha=alpha, learning_rate_init=1e-3,
                        early_stopping=True, n_iter_no_change=20, max_iter=2000, random_state=semilla)


class Hibrido(RegressorMixin, BaseEstimator):
    """Ridge para la tendencia (si extrapola: anios y generaciones nuevas) + random forest para lo que la recta no explica.
    Se usa dentro de con_log, asi que y ya viene en log."""
    def __init__(self, alpha=1.0, n_estimators=200, min_samples_leaf=3, vida_media=None, segundo="bosque", capas=(64, 32),
                 alpha_red=1e-3, max_features=1.0, externos=True, semilla=SEMILLA):
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.vida_media = vida_media     # E11: pesos por antiguedad, lo reciente pesa mas (None = todos iguales)
        self.segundo = segundo           # quien corrige a la recta: "bosque", "red" (red neuronal) o "hgb"
        self.capas = capas
        self.alpha_red = alpha_red
        self.max_features = max_features
        self.externos = externos
        self.semilla = semilla

    def fit(self, X, y):
        w = None
        if self.vida_media:
            w = (0.5 ** ((X["anio"].max() - X["anio"]) / self.vida_media)).values
        prep = lambda: preprocesamiento(externos=self.externos)
        self.lineal_ = make_pipeline(prep(), Ridge(alpha=self.alpha)).fit(X, y, ridge__sample_weight=w)
        residuo = y - self.lineal_.predict(X)
        if self.segundo == "red":
            self.arboles_ = make_pipeline(prep(), mlp(self.capas, alpha=self.alpha_red, semilla=self.semilla)).fit(
                X, residuo, mlpregressor__sample_weight=w)
        elif self.segundo == "hgb":
            self.arboles_ = make_pipeline(prep(), HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=self.n_estimators, max_leaf_nodes=15, min_samples_leaf=max(5, self.min_samples_leaf),
                l2_regularization=1.0, random_state=self.semilla)).fit(X, residuo, histgradientboostingregressor__sample_weight=w)
        else:
            self.arboles_ = make_pipeline(prep(), RandomForestRegressor(
                n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                random_state=self.semilla, n_jobs=-1)).fit(X, residuo, randomforestregressor__sample_weight=w)
        return self

    def predict(self, X):
        return self.lineal_.predict(X) + self.arboles_.predict(X)


class ConPesos(RegressorMixin, BaseEstimator):
    """Envuelve cualquier modelo: preprocesamiento (con o sin datos externos) + (1) pesos por antiguedad (lo reciente pesa
    mas) y (2) opcionalmente acota la prediccion (en log) al rango visto en train +1, para que un modelo que extrapola mal
    (redes, SVR) no prediga cosas absurdas."""
    def __init__(self, modelo=None, vida_media=None, acotar=False, externos=True):
        self.modelo = modelo
        self.vida_media = vida_media
        self.acotar = acotar
        self.externos = externos

    def fit(self, X, y):
        kw = {}
        if self.vida_media:
            kw["sample_weight"] = (0.5 ** ((X["anio"].max() - X["anio"]) / self.vida_media)).values
        self.prep_ = preprocesamiento(externos=self.externos).fit(X)
        self.modelo_ = clone(self.modelo).fit(self.prep_.transform(X), y, **kw)
        self.rango_ = (float(np.min(y)) - 0.5, float(np.max(y)) + 1.0)
        return self

    def predict(self, X):
        p = self.modelo_.predict(self.prep_.transform(X))
        return np.clip(p, *self.rango_) if self.acotar else p


class Mezcla(RegressorMixin, BaseEstimator):
    """Promedio ponderado (en log) de varios modelos ya configurados. Los pesos salen de la busqueda (Caruana)."""
    def __init__(self, modelos=None, pesos=None):
        self.modelos = modelos
        self.pesos = pesos

    def fit(self, X, y):
        self.ajustados_ = [clone(m).fit(X, y) for m in self.modelos]
        return self

    def predict(self, X):
        return sum(w * m.predict(X) for w, m in zip(self.pesos, self.ajustados_))


class Combo(RegressorMixin, BaseEstimator):
    """E13: promedio (en log) de dos hibridos que se equivocan distinto: recta+bosque y recta+red neuronal."""
    def __init__(self, peso_bosque=0.7, alpha=10.0, vida_media=5):
        self.peso_bosque = peso_bosque
        self.alpha = alpha
        self.vida_media = vida_media

    def fit(self, X, y):
        self.bosque_ = Hibrido(alpha=self.alpha, vida_media=self.vida_media).fit(X, y)
        self.red_ = Hibrido(alpha=self.alpha, vida_media=self.vida_media, segundo="red").fit(X, y)
        return self

    def predict(self, X):
        return self.peso_bosque * self.bosque_.predict(X) + (1 - self.peso_bosque) * self.red_.predict(X)


class ConRuido(RegressorMixin, BaseEstimator):
    """Aumentacion (E5): en fit agrega una copia de train con ruido gaussiano pequeno en las numericas continuas."""
    def __init__(self, modelo=None, sigma=0.05, semilla=0):
        self.modelo = modelo
        self.sigma = sigma
        self.semilla = semilla

    def fit(self, X, y):
        rng = np.random.default_rng(self.semilla)
        Xn = X.copy()
        for c in ["TDP", "ghz_nombre", "l3_mb", "base_ghz", "turbo_ghz"]:
            if c in Xn:
                Xn[c] = Xn[c] * (1 + rng.normal(0, self.sigma, len(Xn)))
        yn = np.asarray(y) + rng.normal(0, self.sigma / 2, len(y))     # y ya viene en log
        X2 = pd.concat([X, Xn], ignore_index=True)
        self.modelo_ = clone(self.modelo).fit(X2, np.r_[np.asarray(y), yn])
        return self

    def predict(self, X):
        return self.modelo_.predict(X)


def con_log(modelo):
    return TransformedTargetRegressor(regressor=modelo, func=np.log1p, inverse_func=np.expm1)


def catalogo():
    """nombre -> (familia, modelo sin entrenar, rejilla para GridSearch). Los nombres de parametros llevan regressor__."""
    return {
        "lineal": ("lineal", con_log(make_pipeline(preprocesamiento(), LinearRegression())), {}),
        "polinomial": ("lineal", con_log(make_pipeline(preprocesamiento(polinomio=2), Ridge(alpha=1.0))),
                       {"regressor__ridge__alpha": [0.1, 1.0, 10.0]}),
        "ridge": ("lineal", con_log(make_pipeline(preprocesamiento(), Ridge())),
                  {"regressor__ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
        "lasso": ("lineal", con_log(make_pipeline(preprocesamiento(), Lasso(max_iter=5000))),
                  {"regressor__lasso__alpha": [0.0005, 0.001, 0.005, 0.01, 0.05]}),
        "knn": ("vecinos", con_log(make_pipeline(preprocesamiento(), KNeighborsRegressor())),
                {"regressor__kneighborsregressor__n_neighbors": [3, 5, 10, 20], "regressor__kneighborsregressor__weights": ["uniform", "distance"]}),
        "random_forest": ("arboles", con_log(make_pipeline(preprocesamiento(), RandomForestRegressor(n_estimators=300, random_state=SEMILLA, n_jobs=-1))),
                          {"regressor__randomforestregressor__min_samples_leaf": [1, 3, 5]}),
        "hgb": ("arboles", con_log(make_pipeline(preprocesamiento(), HistGradientBoostingRegressor(random_state=SEMILLA))),
                {"regressor__histgradientboostingregressor__learning_rate": [0.05, 0.1],
                 "regressor__histgradientboostingregressor__max_leaf_nodes": [15, 31]}),
        "random_forest_ruido": ("arboles", con_log(ConRuido(make_pipeline(preprocesamiento(), RandomForestRegressor(
            n_estimators=300, random_state=SEMILLA, n_jobs=-1)), sigma=0.05)), {}),
        "hibrido": ("ensamble", con_log(Hibrido()), {"regressor__alpha": [0.1, 1.0, 10.0]}),
        # E11: el mismo hibrido pero lo reciente pesa mas (vida media 5 anios). El GridSearch es con kfold revuelto y ahi no se
        # nota el drift, por eso va como modelo aparte y lo escoge (o no) la validacion temporal
        "hibrido_reciente": ("ensamble", con_log(Hibrido(vida_media=5)), {"regressor__alpha": [0.1, 1.0, 10.0]}),
        # E12: red neuronal (MLP) sola y como corrector de la recta
        "red_neuronal": ("red", con_log(make_pipeline(preprocesamiento(), mlp())),
                         {"regressor__mlpregressor__hidden_layer_sizes": [(32,), (64, 32)],
                          "regressor__mlpregressor__alpha": [1e-3, 1e-1]}),
        "hibrido_red": ("ensamble", con_log(Hibrido(vida_media=5, segundo="red")), {"regressor__alpha": [1.0, 10.0]}),
        "hibrido_combo": ("ensamble", con_log(Combo()), {}),     # E13: 70% recta+bosque + 30% recta+red
        "stacking": ("ensamble", con_log(StackingRegressor(
            estimators=[("ridge", make_pipeline(preprocesamiento(), Ridge(alpha=1.0))),
                        ("rf", make_pipeline(preprocesamiento(), RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEMILLA, n_jobs=-1))),
                        ("hgb", make_pipeline(preprocesamiento(), HistGradientBoostingRegressor(random_state=SEMILLA))),
                        ("knn", make_pipeline(preprocesamiento(), KNeighborsRegressor(n_neighbors=10, weights="distance")))],
            final_estimator=LinearRegression(positive=True),       # pesos positivos: no extrapola en log
            cv=KFold(5, shuffle=True, random_state=SEMILLA))), {}),  # revuelto: el csv viene ordenado por cpuMark
    }
