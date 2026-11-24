"""Drift: procesadores viejos (referencia, <= ANIO_CORTE) contra nuevos (> ANIO_CORTE).

- data drift: cambia la distribucion de las entradas (mas nucleos, otro TDP...)
- label drift: cambia la distribucion del objetivo (cpuMark sube con los anios)
- concept drift: cambia la RELACION entradas -> objetivo (misma especificacion, mas rendimiento por mejor microarquitectura)
"""
import json
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict
from threadpoolctl import threadpool_limits

from . import config, modelos
from .entrenar import metricas, preparar_mlflow

NUMERICAS = ["cores", "TDP", "ghz_nombre"]
CATEGORICAS = ["fabricante", "familia", "es_desktop", "es_laptop", "es_server", "es_mobile"]


def psi(ref, act, bins=10):
    """Population Stability Index. < 0.1 estable, 0.1-0.25 moderado, > 0.25 drift fuerte."""
    ref, act = pd.Series(ref).dropna(), pd.Series(act).dropna()
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    p = np.histogram(ref, cortes)[0] / len(ref) + 1e-4
    q = np.histogram(act, cortes)[0] / len(act) + 1e-4
    return float(np.sum((q - p) * np.log(q / p)))


def psi_categorico(ref, act):
    cats = sorted(set(ref.astype(str)) | set(act.astype(str)))
    p = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0) + 1e-4
    q = act.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0) + 1e-4
    return float(np.sum((q - p) * np.log(q / p)))


def nivel(v):
    return "fuerte" if v > 0.25 else ("moderado" if v > 0.1 else "estable")


def reporte_drift(X, y, nombre_modelo="hgb", info_datos=None):
    preparar_mlflow(config.EXP_DRIFT)
    config.REPORTES.mkdir(exist_ok=True)
    viejo = X["anio"] <= config.ANIO_CORTE
    Xr, Xa, yr, ya = X[viejo], X[~viejo], y[viejo], y[~viejo]

    # 1) data drift
    filas = []
    for c in NUMERICAS:
        ks = ks_2samp(Xr[c].dropna(), Xa[c].dropna())
        v = psi(Xr[c], Xa[c])
        filas.append({"variable": c, "tipo": "numerica", "psi": v, "nivel": nivel(v), "ks": ks.statistic, "p_valor": ks.pvalue,
                      "media_ref": Xr[c].mean(), "media_nuevos": Xa[c].mean()})
    for c in CATEGORICAS:
        v = psi_categorico(Xr[c], Xa[c])
        tabla = pd.crosstab(pd.concat([Xr[c], Xa[c]]).astype(str), np.r_[np.zeros(len(Xr)), np.ones(len(Xa))])
        filas.append({"variable": c, "tipo": "categorica", "psi": v, "nivel": nivel(v), "ks": np.nan,
                      "p_valor": chi2_contingency(tabla)[1], "media_ref": np.nan, "media_nuevos": np.nan})
    data_drift = pd.DataFrame(filas).sort_values("psi", ascending=False)

    # 2) label drift (en log porque el puntaje crece multiplicando)
    ks_y = ks_2samp(np.log1p(yr), np.log1p(ya))
    label = {"psi": psi(np.log1p(yr), np.log1p(ya)), "ks": ks_y.statistic, "p_valor": ks_y.pvalue,
             "mediana_ref": float(yr.median()), "mediana_nuevos": float(ya.median())}

    # 3) concept drift: entrenar con viejos, medir en viejos (fuera de muestra) y en nuevos
    modelo = clone(modelos.catalogo()[nombre_modelo][1])
    with threadpool_limits(2):
        pred_ref = cross_val_predict(modelo, Xr, yr, cv=KFold(5, shuffle=True, random_state=config.SEMILLA))
        modelo.fit(Xr, yr)
        pred_nuevos = modelo.predict(Xa)
    m_ref, m_nuevos = metricas(yr, pred_ref), metricas(ya, pred_nuevos)
    residuo_log = pd.Series(np.log1p(ya.values) - np.log1p(np.clip(pred_nuevos, 0, None)), index=Xa.index)
    sesgo_anio = residuo_log.groupby(Xa["anio"]).mean()      # > 0: el modelo viejo SUBESTIMA a los nuevos
    concepto = {"mae_viejos": m_ref["mae"], "mae_nuevos": m_nuevos["mae"], "r2_log_viejos": m_ref["r2_log"],
                "r2_log_nuevos": m_nuevos["r2_log"], "dentro20_viejos": m_ref["kpi_dentro_20pct"],
                "dentro20_nuevos": m_nuevos["kpi_dentro_20pct"],
                "sesgo_log_medio_nuevos": float(residuo_log.mean()),
                "factor_subestimacion": float(np.exp(residuo_log.mean()))}

    graficas(X, y, data_drift, sesgo_anio, viejo)
    ev = evidently(Xr, Xa, yr, ya, pred_ref, pred_nuevos, data_drift)
    resumen = {"corte": f"<= {config.ANIO_CORTE} vs > {config.ANIO_CORTE}", "n_viejos": int(viejo.sum()),
               "n_nuevos": int((~viejo).sum()), "label_drift": label, "concept_drift": concepto,
               "variables_con_drift_fuerte": data_drift[data_drift.nivel == "fuerte"].variable.tolist(),
               "evidently": ev}
    data_drift.to_csv(config.REPORTES / "data_drift.csv", index=False)
    (config.REPORTES / "drift_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))

    tags = {"modelo": nombre_modelo, "corte": str(config.ANIO_CORTE), "etapa": "monitoreo"}
    if info_datos:
        tags |= {"data_version": info_datos["version"], "data_hash": info_datos["hash"]}
    with mlflow.start_run(run_name=f"drift__corte-{config.ANIO_CORTE}__data-v{tags.get('data_version', 'x')}", tags=tags):
        mlflow.log_metrics({"psi_max_entradas": data_drift.psi.max(), "n_variables_drift_fuerte": len(resumen["variables_con_drift_fuerte"]),
                            "psi_label": label["psi"], "ks_label": label["ks"], **{k: v for k, v in concepto.items()}})
        for f in ["data_drift.csv", "drift_resumen.json", "drift_entradas.png", "drift_label.png", "drift_concepto.png"]:
            mlflow.log_artifact(str(config.REPORTES / f), "drift")
        if ev.get("ok"):
            mlflow.log_metrics({"evidently_columnas_con_drift": ev["columnas_con_drift"],
                                "evidently_pct_columnas_con_drift": ev["pct_columnas_con_drift"],
                                "evidently_coincide_con_psi_pct": ev["coincide_con_psi_pct"]})
            for f in ["evidently_data_drift.html", "evidently_modelo.html", "evidently_columnas.csv"]:
                mlflow.log_artifact(str(CARPETA_EVIDENTLY / f), "drift/evidently")
        mlflow.set_tag("evidently", "ok" if ev.get("ok") else ev.get("motivo", "no"))
    return data_drift, resumen


CARPETA_EVIDENTLY = config.REPORTES / "evidently"


def leer_evidently(ruta_json):
    """Del JSON de Evidently saca, por columna: metodo, valor, umbral y si detecto drift."""
    filas = []
    for m in json.loads(Path(ruta_json).read_text())["datos"]["metrics"]:
        g = re.match(r"ValueDrift\(column=([^,]+),method=([^,]+),threshold=([0-9.eE-]+)\)", m.get("metric_name", ""))
        if g:
            col, metodo, umbral, valor = g.group(1), g.group(2), float(g.group(3)), float(m["value"])
            drift = valor < umbral if "p_value" in metodo else valor >= umbral
            filas.append({"variable": col, "metodo_evidently": metodo, "valor_evidently": valor, "umbral": umbral,
                          "drift_evidently": bool(drift)})
    return pd.DataFrame(filas)


def evidently(Xr, Xa, yr, ya, pred_ref, pred_nuevos, data_drift):
    """Reporte estandar de Evidently (HTML interactivo) + comparacion con nuestro PSI. Evidently corre en su propio
    entorno (config.EVIDENTLY_PYTHON); si no esta, se omite sin detener el pipeline."""
    if not Path(config.EVIDENTLY_PYTHON).exists():
        print(f"      evidently: se omite (no existe {config.EVIDENTLY_PYTHON})")
        return {"ok": False, "motivo": "sin entorno evidently"}
    CARPETA_EVIDENTLY.mkdir(parents=True, exist_ok=True)
    Xr.assign(cpuMark=yr.values, prediccion=pred_ref).to_parquet(CARPETA_EVIDENTLY / "referencia.parquet")
    Xa.assign(cpuMark=ya.values, prediccion=pred_nuevos).to_parquet(CARPETA_EVIDENTLY / "actual.parquet")
    r = subprocess.run([config.EVIDENTLY_PYTHON, "-W", "ignore", str(Path(__file__).with_name("evidently_reporte.py")),
                        str(CARPETA_EVIDENTLY)], capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print("      evidently: fallo ->", r.stderr.strip().splitlines()[-1:] or r.stdout[-300:])
        return {"ok": False, "motivo": "fallo evidently"}
    col = leer_evidently(CARPETA_EVIDENTLY / "evidently_completo.json")
    # comparacion con nuestro PSI en las variables que revisan los dos (PSI > 0.1 = al menos drift moderado)
    nuestro = data_drift.set_index("variable")[["psi", "nivel"]]
    col = col.join(nuestro, on="variable")
    col["drift_psi"] = col["psi"].gt(0.1).where(col["psi"].notna())
    ambos = col.dropna(subset=["psi"])
    coincide = float((ambos["drift_psi"].astype(bool) == ambos["drift_evidently"]).mean()) if len(ambos) else float("nan")
    col.to_csv(CARPETA_EVIDENTLY / "evidently_columnas.csv", index=False)
    res = {"ok": True, "columnas_revisadas": int(len(col)), "columnas_con_drift": int(col["drift_evidently"].sum()),
           "pct_columnas_con_drift": float(col["drift_evidently"].mean()),
           "con_drift": col.loc[col["drift_evidently"], "variable"].tolist(),
           "coincide_con_psi_pct": coincide, "variables_comparadas_con_psi": int(len(ambos)),
           "html": ["reportes/evidently/evidently_data_drift.html", "reportes/evidently/evidently_modelo.html"]}
    print(f"      evidently: {res['columnas_con_drift']}/{res['columnas_revisadas']} columnas con drift | "
          f"coincide con nuestro PSI en {coincide:.0%} de {len(ambos)} variables")
    return res


def graficas(X, y, data_drift, sesgo_anio, viejo):
    plt.figure(figsize=(6, 4))
    colores = data_drift.nivel.map({"fuerte": "tab:red", "moderado": "tab:orange", "estable": "tab:green"})
    plt.barh(data_drift.variable, data_drift.psi, color=colores)
    plt.axvline(0.1, ls="--", c="gray"); plt.axvline(0.25, ls="--", c="red")
    plt.xlabel("PSI (viejos vs nuevos)"); plt.title("data drift por variable"); plt.tight_layout()
    plt.savefig(config.REPORTES / "drift_entradas.png", dpi=110); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(np.log10(y[viejo]), bins=40, alpha=0.6, density=True, label=f"<= {config.ANIO_CORTE}")
    ax[0].hist(np.log10(y[~viejo]), bins=40, alpha=0.6, density=True, label=f"> {config.ANIO_CORTE}")
    ax[0].set_xlabel("log10(cpuMark)"); ax[0].legend(); ax[0].set_title("label drift")
    y.groupby(X["anio"]).median().plot(ax=ax[1], marker="o")
    ax[1].set_yscale("log"); ax[1].set_ylabel("cpuMark mediano"); ax[1].set_title("cpuMark por anio de prueba")
    plt.tight_layout(); plt.savefig(config.REPORTES / "drift_label.png", dpi=110); plt.close()

    plt.figure(figsize=(6, 4))
    sesgo_anio.plot(marker="o")
    plt.axhline(0, c="gray", ls="--")
    plt.ylabel("residuo medio en log (real - predicho)")
    plt.title(f"concept drift: modelo entrenado con <= {config.ANIO_CORTE}")
    plt.tight_layout(); plt.savefig(config.REPORTES / "drift_concepto.png", dpi=110); plt.close()
