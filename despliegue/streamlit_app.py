"""Estimador de rendimiento multi-core de CPUs (cpuMark de PassMark) - pagina publica del modelo en produccion.

Usa exactamente el modelo que MLflow tiene con alias 'produccion' (empaquetado con `python -m src.desplegar`).
Pestanas: un procesador (online) | lote CSV (batch) | monitoreo (observabilidad) | acerca del modelo.

Streamlit Community Cloud: archivo principal `despliegue/streamlit_app.py` (las dependencias salen de
despliegue/requirements.txt). Se vuelve a desplegar solo cada vez que cambia el repo de GitHub.
Local:  pip install -r requirements.txt && streamlit run streamlit_app.py
"""
import json
import os
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))                        # usar despliegue/src (no el src/ completo del proyecto)
os.environ.setdefault("CPU_PERF_SIN_INTERNET", "1")  # datos externos solo de las copias incluidas
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import cloudpickle
import pandas as pd
import streamlit as st

from src import datos, monitoreo

st.set_page_config(page_title="Estimador de rendimiento de CPUs", page_icon="🖥️", layout="wide")

INFO = json.loads((AQUI / "modelo" / "info.json").read_text())
VERSION = INFO["version"]
TAGS = INFO.get("tags_version", {})
COLUMNAS = ["cpuName", "cores", "TDP", "testDate", "socket", "category"]
CATEGORIAS = ["Desktop", "Laptop", "Server", "Mobile/Embedded"]
EJEMPLOS = {
    "(escribir mis datos)": ["", 8, None, 2021, "", "Desktop"],
    "AMD Ryzen 7 5800X": ["AMD Ryzen 7 5800X", 8, 105.0, 2021, "AM4", "Desktop"],
    "Intel Core i7-12700K": ["Intel Core i7-12700K", 12, 125.0, 2022, "FCLGA1700", "Desktop"],
    "Intel Core i5-1135G7 (laptop)": ["Intel Core i5-1135G7", 4, 28.0, 2021, "FCBGA1449", "Laptop"],
    "AMD EPYC 7763 (servidor)": ["AMD EPYC 7763", 64, 280.0, 2021, "SP3", "Server"],
    "Intel Core i7-4770 (2013)": ["Intel Core i7-4770", 4, 84.0, 2013, "LGA1150", "Desktop"],
}


@st.cache_resource
def modelo():
    with open(AQUI / "modelo" / "model.pkl", "rb") as f:
        return cloudpickle.load(f)


def predecir_df(df, modo):
    """Mismo camino que la API de produccion (src/servir.py): revisar entrada -> preparar variables -> predecir -> registrar."""
    t = time.perf_counter()
    df = df.copy()
    for c, defecto in [("cpuName", ""), ("socket", "unknown"), ("category", "Desktop")]:
        df[c] = (df[c] if c in df else defecto)
        df[c] = df[c].fillna(defecto).astype(str).replace("", defecto)
    alertas = monitoreo.revisar_entrada(df)
    try:
        X, _ = datos.preparar(df, externos=True)
        pred = modelo().predict(X)
    except Exception:
        monitoreo.registrar(modo, VERSION, (time.perf_counter() - t) * 1000, df.to_dict("records"), [], alertas, ok=False)
        raise
    lat = (time.perf_counter() - t) * 1000
    monitoreo.registrar(modo, VERSION, lat, df.to_dict("records") if modo == "online" else {"filas": len(df)}, pred, alertas)
    return pred, alertas, lat


st.title("🖥️ Estimador de rendimiento de CPUs")
st.caption(f"Estima el **rendimiento multi-nucleo (cpuMark de PassMark)** a partir de la ficha tecnica. "
           f"Modelo en produccion: **{INFO['modelo']} v{VERSION}**. Proyecto academico (aprendizaje automatico + MLOps).")

uno, lote, moni, acerca = st.tabs(["Un procesador", "Lote (CSV)", "Monitoreo", "Acerca del modelo"])

with uno:
    ej = st.selectbox("Cargar un ejemplo", list(EJEMPLOS))
    n, c, t, a, s, cat = EJEMPLOS[ej]
    with st.form("uno"):
        nombre = st.text_input("Nombre del procesador", n, key=f"n_{ej}", placeholder="ej. Intel Core i7-12700K",
                               help="Del nombre se sacan fabricante, familia, generacion, gama y sufijo")
        col1, col2, col3 = st.columns(3)
        nucleos = col1.number_input("Nucleos (reales, incluyendo eficientes)", 1, 256, c, key=f"c_{ej}")
        tdp = col2.number_input("TDP (W) - opcional", 1.0, 500.0, t, key=f"t_{ej}")
        anio = col3.number_input("Anio", 2000, 2035, a, key=f"a_{ej}")
        col4, col5 = st.columns(2)
        socket = col4.text_input("Socket - opcional", s, key=f"s_{ej}", placeholder="ej. AM4, FCLGA1700")
        categoria = col5.selectbox("Tipo", CATEGORIAS, CATEGORIAS.index(cat), key=f"k_{ej}")
        enviar = st.form_submit_button("Estimar", type="primary")
    if enviar:
        df = pd.DataFrame([{"cpuName": nombre, "cores": int(nucleos), "TDP": tdp, "testDate": int(anio),
                            "socket": socket or "unknown", "category": categoria}])
        pred, alertas, lat = predecir_df(df, "online")
        p = float(pred[0])
        st.metric("cpuMark estimado", f"{p:,.0f}")
        st.write(f"Margen tipico en procesadores nuevos: la mitad de las veces el real queda a menos de ±20% "
                 f"(aprox. {p * 0.8:,.0f} - {p * 1.2:,.0f}). Modelo v{VERSION} · {lat:.0f} ms")
        for al in alertas:
            st.warning(f"Alerta de calidad de la entrada: {al}")

with lote:
    st.write("Sube un CSV con columnas `cpuName, cores, TDP, testDate, socket, category` "
             "(obligatorias: `cores` y `testDate`). Si trae `cpuMark` real, tambien se mide el acierto.")
    archivo = st.file_uploader("CSV de entrada", type="csv")
    if archivo is not None:
        df = pd.read_csv(archivo)
        faltan = [c for c in ["cores", "testDate"] if c not in df]
        if faltan:
            st.error(f"Al CSV le faltan columnas obligatorias: {faltan}. Esperadas: {COLUMNAS}")
        else:
            for c in COLUMNAS:
                if c not in df:
                    df[c] = None
            pred, alertas, lat = predecir_df(df, "batch")
            df["cpuMark_estimado"] = pd.Series(pred).round().astype(int).values
            st.success(f"{len(df)} procesadores en {lat:.0f} ms · modelo v{VERSION}")
            if "cpuMark" in df and df["cpuMark"].notna().any():
                ok = df["cpuMark"].notna()
                err = (df.loc[ok, "cpuMark_estimado"] - df.loc[ok, "cpuMark"]).abs() / df.loc[ok, "cpuMark"]
                st.info(f"El CSV trae cpuMark real: {(err <= 0.2).mean():.0%} de las estimaciones quedan dentro de ±20%")
            for al in alertas:
                st.warning(f"Alerta de calidad: {al}")
            st.dataframe(df, use_container_width=True)
            st.download_button("Descargar resultados", df.to_csv(index=False), "predicciones.csv", "text/csv")

with moni:
    st.write("Observabilidad del servicio: peticiones, latencia p50/p95, tasa de error y % con alertas de calidad "
             "(se reinicia cuando la pagina se reinicia).")
    if st.button("Actualizar"):
        st.rerun()
    st.json(monitoreo.resumen())

with acerca:
    st.markdown(f"""
**Que es:** {TAGS.get('modelo', 'modelo cruzado PassMark + Blender')}.

**Como se midio (validacion temporal, 5 cortes: entrena con el pasado y predice los 2 anios siguientes):**
error absoluto medio **{float(TAGS.get('temporal_multi_mae', 'nan')):,.0f} puntos de cpuMark** y ~49% de los procesadores
nuevos dentro de ±20%. En procesadores 2023-2026 (sin PassMark, comparados con Blender) ordena bien de mas rapido a mas
lento (correlacion de orden ~0.83-0.85).

**Limitaciones:** los datos de PassMark llegan a 2022; procesadores muy distintos a los vistos (arquitecturas nuevas)
pueden tener mas error. Usar como estimacion, no como benchmark.

**Datos:** PassMark via Kaggle (`alanjo/cpu-benchmarks` v{TAGS.get('data_version', '?')}), TechPowerUp (Kaggle),
pc_build (Hugging Face), fichas oficiales AMD/Intel (Kaggle) y Blender Open Data (CC0).

**Trazabilidad:** run de MLflow `{INFO.get('run_id')}`, huella de datos `{TAGS.get('data_hash')}`,
huella de codigo `{TAGS.get('version_codigo')}`, empaquetado {INFO.get('empaquetado')}.
Se actualiza solo: un pipeline semanal en GitHub Actions reentrena cuando hay datos nuevos y, si el modelo mejora al
menos 1%, publica la version nueva aqui.
""")
