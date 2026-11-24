"""Genera el reporte final en PDF (reporte/Reporte_Final_CPU_MLOps.pdf) a partir de los resultados del proyecto.

Las figuras se dibujan desde los archivos de reportes/ (csv / json que escriben el pipeline y los experimentos), asi que
el reporte se puede regenerar cuando cambien los resultados:

    python reporte/generar_reporte.py           (desde la raiz del repo)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

RAIZ = Path(__file__).resolve().parents[1]
REP = RAIZ / "reportes"
FIG = RAIZ / "reporte" / "figuras"
SALIDA = RAIZ / "reporte" / "Reporte_Final_CPU_MLOps.pdf"
FIG.mkdir(parents=True, exist_ok=True)

AZUL, AZUL_CLARO, NARANJA, OSCURO, GRIS = "#2F6690", "#A9C6DE", "#C9622A", "#14213D", "#6B7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11})


# ------------------------------------------------------------------ figuras
def guardar(fig, nombre):
    ruta = FIG / nombre
    fig.tight_layout()
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    return ruta


def fig_validacion():
    fig, ejes = plt.subplots(1, 2, figsize=(8, 2.8))
    etiquetas = ["k-fold aleatorio", "temporal\n(≤2019 → 2020–22)"]
    for ax, vals, titulo, fmt in [(ejes[0], [828, 5157], "MAE (puntos de cpuMark)", "{:,.0f}"),
                                  (ejes[1], [72.7, 21.1], "% dentro de ±20%", "{:.0f}%")]:
        b = ax.bar(etiquetas, vals, color=[AZUL_CLARO, NARANJA], width=0.55)
        ax.set_title(titulo)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.2)
    return guardar(fig, "validacion.png")


def fig_cortes():
    fig, ax = plt.subplots(figsize=(8, 2.4))
    for i, c in enumerate([2016, 2017, 2018, 2019, 2020]):
        ax.barh(i, c - 2007 + 1, left=2007, color=AZUL_CLARO, height=0.6)
        ax.barh(i, 2, left=c + 1, color=NARANJA, height=0.6)
        ax.text(2006.6, i, f"corte {c}", ha="right", va="center", fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(2003.5, 2023.5)
    ax.set_xticks(range(2007, 2024, 2))
    ax.invert_yaxis()
    ax.set_title("Entrenar con el pasado (azul), examinar los 2 años siguientes (naranja)")
    return guardar(fig, "cortes.png")


def fig_robustez():
    r = pd.read_csv(REP / "robustez_detalle.csv").groupby(["modelo", "corte"]).mae.mean().unstack()
    nombres = {"ridge": "Ridge", "random_forest": "Random Forest", "hgb": "HGB",
               "hibrido_reciente (bosque, produccion)": "Híbrido recta + bosque (reciente)", "red 64-32 alpha 3": "Red neuronal 64-32"}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    estilos = {"Híbrido recta + bosque (reciente)": dict(color=NARANJA, lw=2.5), "Red neuronal 64-32": dict(color=GRIS, ls="--")}
    for k, n in nombres.items():
        ax.plot(r.columns, r.loc[k], marker="o", label=n, **estilos.get(n, {}))
    ax.set_yscale("log")
    ax.set_xticks(list(r.columns))
    ax.set_xlabel("corte (entrena ≤ año, examina los 2 siguientes)")
    ax.set_ylabel("MAE (escala log)")
    ax.set_title("Un solo corte engaña: MAE de cada modelo en los 5 cortes")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    return guardar(fig, "robustez.png")


def fig_busqueda():
    d = json.load(open(REP / "busqueda_mejores.json"))
    nombres = {"hibrido_hgb": "Híbrido recta + HGB", "hibrido_bosque": "Híbrido recta + bosque", "hibrido_red": "Híbrido recta + red",
               "mlp": "Red neuronal (MLP)", "ridge": "Ridge", "svr": "SVR", "elasticnet": "ElasticNet", "hgb": "HGB",
               "random_forest": "Random Forest", "extra_trees": "Extra Trees", "knn": "KNN"}
    filas = sorted(((nombres[k], v["mae"], v["mae_std"], v["n"]) for k, v in d.items()), key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(filas))
    ax.barh(y, [f[1] for f in filas], xerr=[f[2] for f in filas], color=[AZUL if "Híbrido" in f[0] else AZUL_CLARO for f in filas],
            error_kw=dict(ecolor=GRIS, lw=1))
    ax.set_yticks(y, [f"{f[0]} ({f[3]})" for f in filas])
    ax.invert_yaxis()
    for i, f in enumerate(filas):
        ax.text(150, i, f"{f[1]:,.0f}", va="center", color="white" if "Híbrido" in f[0] else OSCURO, fontsize=9)
    ax.set_xlabel("mejor MAE promedio de 5 cortes (± desviación entre cortes)\nentre paréntesis: configuraciones probadas")
    ax.set_title("Búsqueda aleatoria: 340 configuraciones en 11 familias")
    return guardar(fig, "busqueda.png")


def fig_comparacion():
    filas = [("Predecir la mediana", 9596, 7.8), ("Solo el año (tendencia)", 9243, 15.1), ("Solo el nombre (TF-IDF)", 5680, 20.3),
             ("Ridge con la ficha", 3934, 34.2), ("HGB solo", 3486, 41.4), ("Híbrido recta + HGB", 2647, 47.1),
             ("v7 cruzado (producción)", 2442, 49.4)]
    fig, ejes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    y = np.arange(len(filas))
    col = [NARANJA if "v7" in f[0] else (AZUL if "Híbrido" in f[0] else AZUL_CLARO) for f in filas]
    ejes[0].barh(y, [f[1] for f in filas], color=col)
    ejes[0].set_yticks(y, [f[0] for f in filas])
    ejes[0].invert_yaxis()
    ejes[0].set_title("MAE (menor es mejor)")
    ejes[1].barh(y, [f[2] for f in filas], color=col)
    ejes[1].set_title("% dentro de ±20% (mayor es mejor)")
    for i, f in enumerate(filas):
        ejes[0].text(f[1] + 120, i, f"{f[1]:,.0f}", va="center", fontsize=8)
        ejes[1].text(f[2] + 0.8, i, f"{f[2]:.0f}%", va="center", fontsize=8)
    ejes[0].set_xlim(0, 11500)
    ejes[1].set_xlim(0, 60)
    fig.suptitle("Líneas base contra el modelo en producción (promedio de 5 cortes temporales)", fontweight="bold", fontsize=11)
    return guardar(fig, "comparacion.png")


def fig_ablacion():
    filas = [("Núcleos", 5498), ("2º paso (árboles)", 1191), ("Logaritmo del objetivo", 778), ("Generación/gama/sufijo/familia", 640),
             ("La recta (Ridge)", 482), ("Año", 380), ("TDP", 286), ("Fabricante", 183), ("Tipo (desktop/laptop…)", 56),
             ("Pesos por antigüedad", 46), ("Frecuencia", -58), ("Socket", -63), ("Datos externos", -357)]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(filas))
    ax.barh(y, [f[1] for f in filas], color=[AZUL if f[1] > 0 else GRIS for f in filas])
    ax.set_yticks(y, [f[0] for f in filas])
    ax.invert_yaxis()
    ax.axvline(0, color=OSCURO, lw=0.8)
    for i, f in enumerate(filas):
        ax.text(f[1] + (60 if f[1] >= 0 else -60), i, f"{f[1]:+,}", va="center", ha="left" if f[1] >= 0 else "right", fontsize=8)
    ax.set_xlim(-900, 6400)
    ax.set_xlabel("cambio en MAE al quitar la pieza (base 3,004; positivo = la pieza ayuda)")
    ax.set_title("Ablación temporal: quitar una cosa a la vez")
    return guardar(fig, "ablacion.png")


def fig_drift():
    dd = pd.read_csv(REP / "data_drift.csv")
    c = json.load(open(REP / "drift_resumen.json"))["concept_drift"]
    fig, ejes = plt.subplots(1, 2, figsize=(8, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = ejes[0]
    ax.barh(dd.variable, dd.psi, color=[NARANJA if n == "fuerte" else (AZUL if n == "moderado" else AZUL_CLARO) for n in dd.nivel])
    ax.invert_yaxis()
    ax.axvline(0.1, color=GRIS, ls=":", lw=1)
    ax.axvline(0.25, color=GRIS, ls="--", lw=1)
    ax.set_title("Data drift (PSI): ≤2019 vs 2020–2022")
    ax.set_xlabel("PSI (0.1 moderado, 0.25 fuerte)")
    ax = ejes[1]
    b = ax.bar(["CPUs viejos\n(fuera de muestra)", "CPUs nuevos"], [c["mae_viejos"], c["mae_nuevos"]], color=[AZUL_CLARO, NARANJA], width=0.55)
    for r, v in zip(b, [c["mae_viejos"], c["mae_nuevos"]]):
        ax.text(r.get_x() + r.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom")
    ax.set_title("Concept drift: MAE de un\nmodelo entrenado con ≤2019")
    ax.set_ylim(0, c["mae_nuevos"] * 1.25)
    return guardar(fig, "drift.png")


def fig_pred_real():
    """Prediccion vs real en el FUTURO: el hibrido entrenado con CPUs <= 2019 predice los de 2020-2022."""
    sys.path.insert(0, str(RAIZ))
    from threadpoolctl import threadpool_limits
    from src import config, datos, modelos
    copias = sorted((RAIZ / "datos" / "raw").glob(f"v*/{config.ARCHIVO}"))
    df = pd.read_csv(copias[-1])
    modelos.usar_externos(True)
    X, _ = datos.preparar(df, externos=True)
    y = df[config.OBJETIVO]
    tr = X["anio"] <= 2019
    m = modelos.con_log(modelos.Hibrido(segundo="hgb", externos=False, alpha=235.77, vida_media=2, n_estimators=200, min_samples_leaf=34))
    with threadpool_limits(1):
        p = m.fit(X[tr], y[tr]).predict(X[~tr])
    real = y[~tr].values
    dentro = np.abs(p - real) / real <= 0.2
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(real[~dentro], p[~dentro], s=7, alpha=0.5, color=AZUL_CLARO, label="fuera de ±20%")
    ax.scatter(real[dentro], p[dentro], s=7, alpha=0.6, color=AZUL, label="dentro de ±20%")
    lim = [50, 2e5]
    ax.plot(lim, lim, color=NARANJA, lw=1.2)
    ax.fill_between(lim, [v * 0.8 for v in lim], [v * 1.2 for v in lim], color=NARANJA, alpha=0.12)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("cpuMark real"); ax.set_ylabel("cpuMark predicho")
    ax.set_title(f"Entrena ≤2019, predice 2020–2022\n{dentro.mean():.0%} dentro de ±20% ({len(real)} CPUs)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    return guardar(fig, "pred_vs_real_futuro.png"), float(dentro.mean())


def fig_arquitectura():
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.axis("off")
    cajas = [(0.02, "GitHub Actions\n(lunes 3 am)", "datos frescos → calidad →\nreentrena si hay datos nuevos →\ndrift + Evidently → batch"),
             (0.36, "MLflow (privado)", "SQLite (metadatos) +\nartefactos, guardados en\nHugging Face entre corridas"),
             (0.70, "Página pública", "Streamlit: un CPU, lote CSV,\nmonitoreo; se redespliega\nsola si cambia el modelo")]
    for x, t, s in cajas:
        ax.add_patch(plt.Rectangle((x, 0.12), 0.28, 0.76, facecolor="#F4F1EA", edgecolor=AZUL, lw=1.5))
        ax.text(x + 0.14, 0.72, t, ha="center", va="center", fontweight="bold", fontsize=10, color=OSCURO)
        ax.text(x + 0.14, 0.38, s, ha="center", va="center", fontsize=8.5, color=OSCURO)
    for x in [0.30, 0.64]:
        ax.annotate("", xy=(x + 0.06, 0.5), xytext=(x, 0.5), arrowprops=dict(arrowstyle="->", color=NARANJA, lw=2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return guardar(fig, "arquitectura.png")


# ------------------------------------------------------------------ documento
pdfmetrics.registerFont(TTFont("DV", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVB", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DV", boldItalic="DVB")

E = {
    "titulo": ParagraphStyle("titulo", fontName="DVB", fontSize=22, leading=28, textColor=colors.HexColor(OSCURO), alignment=TA_CENTER),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=12, leading=17, textColor=colors.HexColor(GRIS), alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=15, leading=20, textColor=colors.HexColor(OSCURO), spaceBefore=10, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=11.5, leading=15, textColor=colors.HexColor(AZUL), spaceBefore=8, spaceAfter=3),
    "p": ParagraphStyle("p", fontName="DV", fontSize=9.6, leading=13.6, alignment=TA_JUSTIFY, spaceAfter=5),
    "li": ParagraphStyle("li", fontName="DV", fontSize=9.6, leading=13.4, leftIndent=12, bulletIndent=2, spaceAfter=2),
    "cap": ParagraphStyle("cap", fontName="DV", fontSize=8.3, leading=11, textColor=colors.HexColor(GRIS), alignment=TA_CENTER, spaceAfter=8),
    "concl": ParagraphStyle("concl", fontName="DV", fontSize=9.6, leading=13.6, alignment=TA_JUSTIFY, backColor=colors.HexColor("#F4F1EA"),
                            borderColor=colors.HexColor(NARANJA), borderWidth=0, borderPadding=7, spaceBefore=6, spaceAfter=10, leftIndent=4, rightIndent=4),
    "celda": ParagraphStyle("celda", fontName="DV", fontSize=8.6, leading=11),
    "celdab": ParagraphStyle("celdab", fontName="DVB", fontSize=8.6, leading=11, textColor=colors.white),
}


def P(t, s="p"):
    return Paragraph(t, E[s])


def lista(items):
    return [Paragraph(t, E["li"], bulletText="•") for t in items]


def figura(ruta, ancho_cm, texto):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(str(ruta)).getSize()
    return KeepTogether([Image(str(ruta), width=ancho_cm * cm, height=ancho_cm * cm * h / w), P(texto, "cap")])


def tabla(filas, anchos):
    datos_t = [[Paragraph(str(c), E["celdab"] if i == 0 else E["celda"]) for c in f] for i, f in enumerate(filas)]
    t = Table(datos_t, colWidths=[a * cm for a in anchos], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(AZUL)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F1EA")]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#D6CFBF")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def conclusion(t):
    return P("<b>Conclusión de la sección.</b> " + t, "concl")


def pie(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 8)
    canvas.setFillColor(colors.HexColor(GRIS))
    canvas.drawString(2 * cm, 1.2 * cm, "Predicción de rendimiento multi-núcleo de CPUs · Reporte final")
    canvas.drawRightString(letter[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.restoreState()


def construir():
    f = {n: fn() for n, fn in [("validacion", fig_validacion), ("cortes", fig_cortes), ("robustez", fig_robustez),
                                ("busqueda", fig_busqueda), ("comparacion", fig_comparacion), ("ablacion", fig_ablacion),
                                ("drift", fig_drift), ("arquitectura", fig_arquitectura)]}
    f["pred_real"], dentro_fut = fig_pred_real()
    ult = json.loads((REP / "ultimo_pipeline.json").read_text())
    s = []

    # ---------------- portada
    s += [Spacer(1, 4.5 * cm), P("¿Qué tan rápido es un procesador antes de que alguien lo pruebe?", "titulo"), Spacer(1, 10),
          P("Predicción del rendimiento multi-núcleo de CPUs (cpuMark de PassMark) a partir de su ficha técnica, "
            "con validación temporal y un sistema MLOps automático", "sub"), Spacer(1, 1.6 * cm),
          P("Reporte final del proyecto de Machine Learning", "sub"), P("Elias Rodriguez", "sub"), P("Noviembre 2026", "sub"),
          Spacer(1, 2.5 * cm),
          tabla([["Material", "Enlace"],
                 ["Código fuente, notebook y reportes", "github.com/Tracker315/cpu-perf-mlops"],
                 ["Página pública del modelo (demo)", "cpu-perf-mlops.streamlit.app"],
                 ["Dataset principal", "kaggle.com/datasets/alanjo/cpu-benchmarks (v26; copia en datos/raw/v26/)"],
                 ["Datos de apoyo", "TechPowerUp y fichas AMD/Intel (Kaggle), pc_build (Hugging Face), opendata.blender.org (CC0)"]],
                [5.2, 11.3]),
          PageBreak()]

    # ---------------- resumen
    s += [P("Resumen", "h1"),
          P("El rendimiento real de un procesador se conoce hasta que alguien lo prueba, meses después de su lanzamiento; su ficha "
            "técnica, en cambio, está disponible desde el primer día. Este proyecto construye un modelo que estima el puntaje "
            "multi-núcleo de PassMark (cpuMark) a partir de la ficha: núcleos, TDP, año, socket, tipo y la información escondida en el "
            "nombre (familia, generación, gama y sufijo), complementada con caché, litografía, frecuencias e hilos de fuentes externas y "
            "con un segundo benchmark independiente (Blender Open Data, hasta 2026)."),
          P("La decisión metodológica central fue medir siempre <b>hacia el futuro</b>: con validación aleatoria un Random Forest "
            "parece equivocarse 828 puntos, pero al predecir procesadores posteriores a los de entrenamiento se equivoca 5,157. Con "
            "validación temporal de 5 cortes se compararon 14 familias de modelos (340 configuraciones en la búsqueda aleatoria, más "
            "XGBoost, LightGBM, CatBoost, Optuna, combinaciones y transfer learning). El modelo final (v7) combina una recta que "
            "extrapola la tendencia con árboles que corrigen el residuo, entrenado con núcleos corregidos y 353 procesadores de "
            "Blender traducidos a la escala de PassMark: <b>MAE de 2,442 puntos y 49% de los procesadores nuevos dentro de ±20%</b>, "
            "frente a 9,596 y 8% de una línea base. La mayor mejora no vino de un modelo más complejo sino de corregir los datos "
            "(−303 puntos al corregir núcleos mal contados)."),
          P("El modelo opera como un sistema MLOps: MLflow con metadatos en SQLite y artefactos aparte, registro de modelos con "
            "alias de producción, predicción en línea y por lotes, monitoreo de calidad y drift (PSI propio y Evidently), y un "
            "pipeline semanal en GitHub Actions que reentrena solo cuando hay datos nuevos y publica el modelo en una página pública "
            "si mejora al menos 1%."),
          Spacer(1, 6)]

    # ---------------- 1. introduccion
    s += [P("1. Introducción, justificación y objetivos", "h1"),
          P("1.1 El problema y su relevancia", "h2"),
          P("Elegir, comparar o presupuestar procesadores (compras de equipo, catálogos de tiendas, planeación de capacidad en "
            "servidores) requiere saber cuánto rinden. Los benchmarks públicos llegan tarde y no cubren todos los modelos: hay "
            "variantes OEM, móviles o de servidor que nadie mide. Contar núcleos no basta: el rango de cpuMark va de 77 a 108,822 "
            "(un factor de 1,400) y procesadores con los mismos 8 núcleos rinden muy distinto según su generación y "
            "microarquitectura."),
          P("Desde el punto de vista de aprendizaje automático es una regresión supervisada con <b>cambio en el tiempo</b> "
            "(concept drift): cada generación cambia la relación entre especificaciones y rendimiento, y el uso real del modelo es "
            "predecir procesadores que todavía no existían cuando se entrenó. Eso obliga a una validación distinta a la habitual y a "
            "componentes capaces de extrapolar."),
          P("1.2 Objetivos", "h2"),
          P("<b>Objetivo general.</b> Estimar el cpuMark multi-núcleo de un procesador a partir de su ficha técnica, con un error medido "
            "en procesadores posteriores a los de entrenamiento, y operar el modelo como un sistema reproducible y automático."),
          P("<b>Objetivos específicos.</b>")] + lista([
              "Reunir datos de fuentes con licencia clara, versionados (versión y hash) y con revisión automática de calidad.",
              "Diseñar variables a partir de la ficha y del nombre del procesador, sin fuga de información.",
              "Comparar familias de modelos y ajustar hiperparámetros con validación temporal de 5 cortes.",
              "Evaluar con métricas interpretables: MAE, porcentaje dentro de ±20% (KPI de negocio) y correlación de orden en CPUs 2023–2026.",
              "Registrar experimentos y modelos en MLflow (SQLite + artefactos), servir en línea y por lotes, monitorear drift y reentrenar automáticamente.",
          ]) + [conclusion("El problema es útil (estimar antes de que exista el benchmark) y difícil por una razón concreta: el futuro "
                           "no se parece al pasado. Por eso los objetivos combinan precisión con honestidad en la medición y con la "
                           "capacidad de mantener el modelo actualizado.")]

    # ---------------- 2. datos
    s += [PageBreak(), P("2. Adquisición de datos y preprocesamiento", "h1"),
          P("2.1 Origen de los datos", "h2"),
          tabla([["Fuente", "Contenido", "Tamaño / cobertura", "Uso"],
                 ["PassMark (Kaggle alanjo/cpu-benchmarks v26)", "cpuMark (objetivo), threadMark, núcleos, TDP, socket, tipo, año de prueba, precio",
                  "3,825 CPUs, 2007–2022", "Fuente principal"],
                 ["TechPowerUp (Kaggle baraazaid/cpu-and-gpu-stats v2)", "Caché L3, litografía, frecuencia base y turbo, lanzamiento", "2,614 CPUs; empareja 32.5%", "Variables externas"],
                 ["pc_build (Hugging Face mehax/pc_build)", "Hilos, núcleos P/E, caché, litografía", "1,442 filas (~541 CPUs); empareja 10.8%", "Variables externas"],
                 ["Fichas oficiales (Kaggle alanjo/amd-processor-specifications v15)", "Año de lanzamiento, núcleos, hilos, TDP", "582 AMD + 2,880 Intel", "Modelo cruzado"],
                 ["Blender Open Data (CC0)", "Benchmark de render multi-núcleo de usuarios", "173,961 pruebas → 1,114 CPUs (hasta 2026)", "Segunda opinión, modelo cruzado"]],
                [4.6, 5.3, 3.9, 2.7]),
          Spacer(1, 6),
          P("Las fuentes de apoyo se unen por nombre con una normalización (quitar marcas registradas, frecuencias, “N-Core”, "
            "sockets y palabras de relleno), que subió el emparejamiento de ~9% a 38% de los procesadores con algún dato externo. "
            "No se usó scraping de la página de PassMark por sus términos de uso. Cada fuente queda versionada: versión de Kaggle y "
            "hash md5 del archivo, sha de Hugging Face y snapshot de Blender con fecha; esas versiones se guardan como etiquetas de "
            "cada corrida en MLflow."),
          P("2.2 Limpieza y control de calidad", "h2")] + lista([
              "<b>Fuga de información:</b> se excluyen cpuValue y powerPerf (se calculan con cpuMark), threadMark (otro benchmark del mismo programa), threadValue y el precio (48.6% vacío y variable en el tiempo). También cualquier ID o número de fila, porque el archivo viene ordenado por puntaje. Regla: solo entra lo que trae la ficha antes de probar el procesador.",
              "<b>Revisión automática de calidad</b> antes de entrenar: esquema completo, ≥1,000 filas y 0% de vacíos en el objetivo (críticas: detienen el pipeline y producción sigue con el modelo anterior), rangos lógicos, vacíos y duplicados (avisos). Resultado actual: todo OK; vacíos conocidos en precio (48.6%) y TDP (17.9%).",
              "<b>Unidades:</b> 5 valores externos en unidades equivocadas (MHz por GHz, KB por MB) se corrigen; hacían divergir a la red neuronal.",
              "<b>Núcleos mal contados:</b> al cruzar con Blender se descubrió que PassMark cuenta solo los núcleos de rendimiento en Intel de 12ª generación en adelante (i9-12900K: 8 en lugar de 16) y módulos en AMD FX. Se corrigieron 137 procesadores, siempre hacia arriba.",
              "<b>Blender:</b> solo pruebas de CPU con las 3 escenas y un socket; puntajes ajustados por versión de Blender (modelo log puntaje = efecto CPU + efecto versión, por medianas alternadas, referencia 4.2); se descartan 61,913 pruebas en máquinas virtuales que usan menos del 90% de los núcleos; mediana por procesador con al menos 5 pruebas.",
          ]) + [
          P("2.3 Ingeniería y selección de variables", "h2"),
          P("Del nombre se extraen con reglas de texto el fabricante, la familia (Core i7, Ryzen 5, Xeon…), la frecuencia, la "
            "generación, la gama dentro de la familia y el sufijo (K, U, H, X…). Por ejemplo, “Intel Core i9-12900K” → Intel, Core "
            "i9, generación 12, gama 0.90, sufijo K. La preparación vive dentro de un pipeline de scikit-learn, de modo que medianas y "
            "escalas se aprenden solo con los datos de entrenamiento de cada validación:"),
          ] + lista([
              "Objetivo transformado a log(1 + cpuMark): el error relativo pesa igual en procesadores baratos y caros. En Ridge, el R² pasa de −16.9 sin log a 0.75 con log (E1).",
              "Núcleos y TDP en logaritmo (crecen multiplicando); vacíos imputados con la mediana más una columna indicadora.",
              "Estandarización de numéricas; one-hot de categorías con agrupación de las raras (menos de 10 casos) y tolerancia a categorías nuevas. Para el socket (200+ valores) one-hot ganó a target encoding (MAE 807 vs 827, E3).",
          ]) + [
          tabla([["Variables (E2, validación aleatoria)", "MAE", "Dentro de ±20%"],
                 ["Solo núcleos", "2,941", "19%"], ["+ ficha técnica (TDP, año, GHz, familia, socket, tipo)", "888", "71%"],
                 ["+ generación, gama y sufijo del nombre", "847", "72%"], ["+ externos (caché, nm, frecuencias, hilos)", "828", "73%"]],
                [9.5, 3.2, 3.8]),
          Spacer(1, 4),
          P("Cada grupo de variables aporta en validación aleatoria. Sin embargo, la ablación temporal (sección 4.4) mostró que los "
            "datos externos <b>empeoran</b> la predicción del futuro (cubren sobre todo procesadores viejos), por lo que el mejor "
            "híbrido de la búsqueda los deja fuera."),
          conclusion("La calidad de los datos fue decisiva: excluir variables que delatan la respuesta evita resultados falsamente "
                     "perfectos, la revisión automática protege a producción, y el cruce con una segunda fuente reveló un error "
                     "sistemático (núcleos mal contados) cuya corrección resultó ser la mejora individual más grande del proyecto.")]

    # ---------------- 3. modelos
    s += [PageBreak(), P("3. Modelos predictivos", "h1"),
          P("3.1 Validación: medir hacia el futuro", "h2"),
          P("El experimento E6 comparó el mismo Random Forest con dos formas de examinarlo. Con k-fold aleatorio el modelo ve "
            "procesadores casi idénticos a los del examen (el i7-8700K entrena, el i7-8700 se evalúa) y su error parece seis veces "
            "menor que al predecir procesadores posteriores. Desde ahí todas las decisiones se tomaron con validación temporal."),
          figura(f["validacion"], 14, "Figura 1. El mismo modelo evaluado con k-fold aleatorio y con validación temporal (E6)."),
          P("Un solo corte también engaña, porque depende de qué años caen en el examen. Se usaron 5 cortes (entrenar hasta 2016, "
            "2017, 2018, 2019 y 2020; examinar los 2 años siguientes, entre 376 y 848 procesadores por examen) y la métrica de "
            "selección es el MAE promedio de los 5, junto con su desviación entre cortes. Es el esquema walk-forward que se usa en "
            "series de tiempo; los mismos cortes se pasaron como <i>cv</i> a RandomizedSearchCV."),
          figura(f["cortes"], 14, "Figura 2. Los 5 cortes de la validación temporal."),
          figura(f["robustez"], 14, "Figura 3. MAE por corte (E13, escala log). La red neuronal parecía buena en 2019–2020 y diverge en "
                                    "2017–2018; el Random Forest empeora conforme avanza el tiempo; el híbrido es el más parejo."),
          P("3.2 Modelos probados", "h2"),
          P("Se evaluaron modelos lineales (regresión lineal, polinomial, Ridge, Lasso, ElasticNet), de vecinos (KNN), de kernel "
            "(SVR), de árboles (Random Forest, Extra Trees, HistGradientBoosting, XGBoost, LightGBM, CatBoost), redes neuronales "
            "(MLPRegressor), combinaciones (stacking y Ensemble Selection de Caruana et al.) y aumentación de datos (ruido, SMOTER, "
            "SMOGN). Los modelos de árboles tienen una limitación de fondo en este problema: solo repiten promedios de lo que vieron, "
            "así que no pueden predecir un procesador más rápido que el más rápido del entrenamiento (E9)."),
          P("De ahí surge el <b>modelo híbrido</b>: (1) una recta Ridge sobre las variables preparadas captura la tendencia y sí "
            "extrapola; (2) un segundo modelo (bosque, HGB o red) aprende el residuo que la recta no explica; todo en escala log y "
            "con pesos por antigüedad (vida media de 2 a 5 años) para que lo reciente pese más (E11: con vida media 5 el MAE de un "
            "corte bajó de 3,260 a 2,928)."),
          P("3.3 Entrenamiento y ajuste de hiperparámetros", "h2"),
          P("RandomizedSearchCV con espacios amplios en 11 familias (25 a 40 configuraciones cada una, 340 en total, ~12 minutos), "
            "usando los 5 cortes como validación cruzada. Los tres primeros lugares son híbridos. El mejor (recta + HGB) usa "
            "α = 236, vida media de 2 años, 200 árboles, mínimo 34 procesadores por hoja y sin variables externas."),
          figura(f["busqueda"], 14.5, "Figura 4. Mejor configuración de cada familia en la búsqueda aleatoria."),
          tabla([["Prueba adicional", "Resultado (MAE, 5 cortes)", "Decisión"],
                 ["Mezcla de Caruana del mejor de cada familia", "2,549 optimista → 2,717 honesto (pesos escogidos sin ver el corte)", "No supera al mejor individual (2,647)"],
                 ["Optuna (TPE) con el mismo presupuesto", "2,622 tras 20 intentos", "Solo 1% mejor: la búsqueda aleatoria bastó"],
                 ["XGBoost / LightGBM / CatBoost (búsqueda propia)", "3,212 / 3,164 / 3,466; dentro del híbrido 2,751–3,025", "No superan a HGB de scikit-learn"],
                 ["Restricciones de monotonía (más núcleos o año nunca bajan)", "2,506 en datos originales; 2,436 vs 2,440 en condiciones del v7", "Empate con producción; ordena peor el futuro"],
                 ["Transfer learning PassMark ↔ Blender (4 variantes)", "2,533–5,070", "Ninguna supera al v7 (2,442)"]],
                [5.2, 6.4, 4.9]),
          P("3.4 Modelo final: sistema cruzado PassMark + Blender (v7)", "h2"),
          P("PassMark llega a 2022, mientras Blender tiene procesadores hasta 2026 en otra escala (samples por minuto). En los 724 "
            "procesadores presentes en ambas fuentes los puntajes están fuertemente correlacionados (~0.94). Un traductor Ridge "
            "(log Blender, log núcleos, log hilos, tipo y marca) convierte Blender a cpuMark con MAE de 1,486 y 81% dentro de ±20% "
            "(validación cruzada de 5 partes). Con él se tradujeron 353 procesadores que solo existen en Blender y se sumaron como "
            "pseudo-etiquetas al entrenamiento del modelo principal (híbrido combinado: 70% recta + bosque y 30% recta + red)."),
          tabla([["Variante (mismos 5 cortes de PassMark)", "MAE", "Dentro de ±20%"],
                 ["Modelo anterior en producción (v5)", "2,764", "46.2%"],
                 ["+ núcleos reales (corregidos)", "2,461", "49.9%"],
                 ["Solo Blender traducido a cpuMark", "6,510", "32.0%"],
                 ["+ núcleos reales + 353 CPUs traducidos (v7, producción)", "2,442", "49.4%"],
                 ["Mezcla v7 + Blender traducido (peso 0.9)", "2,606", "50.9%"]],
                [9.5, 3.2, 3.8]),
          conclusion("Probamos 14 familias de modelos y técnicas de búsqueda, combinación y transferencia con el mismo examen "
                     "temporal. Los modelos convergieron a resultados parecidos; lo que distinguió al ganador fue su estructura (un "
                     "componente que extrapola la tendencia) y, sobre todo, mejores datos.")]

    # ---------------- 4. evaluacion
    s += [PageBreak(), P("4. Evaluación y métricas de desempeño", "h1"),
          P("4.1 Métricas", "h2")] + lista([
              "<b>MAE</b> (error absoluto medio, en puntos de cpuMark): interpretable y usada para seleccionar. Su debilidad es que la dominan los procesadores caros.",
              "<b>Porcentaje dentro de ±20%</b> (KPI de negocio): de cada 100 procesadores, en cuántos la estimación sirve. Es relativa, trata igual a baratos y caros, y 20% equivale a “la misma liga de rendimiento”.",
              "<b>Correlación de orden de Spearman</b>: para procesadores 2023–2026 no hay cpuMark real, solo Blender en otra escala; se mide si el modelo los ordena del más lento al más rápido.",
              "RMSE, R², R² en log y MAPE se registran en MLflow pero no deciden: RMSE y R² se inflan por el rango de 77 a 108,822 y MAPE explota en los procesadores más lentos (llegó a 96%).",
          ]) + [
          P("4.2 Comparación de modelos", "h2"),
          figura(f["comparacion"], 15.5, "Figura 5. Líneas base contra el modelo en producción (experimento X5 y modelo cruzado)."),
          P(f"El modelo en producción reduce el error casi cuatro veces respecto a predecir la mediana y pasa de 8% a 49% de "
            f"procesadores dentro de ±20%. La Figura 6 muestra el comportamiento hacia el futuro del mejor híbrido de la búsqueda "
            f"entrenado con procesadores hasta 2019: {dentro_fut:.0%} de los procesadores de 2020–2022 quedan en la banda de ±20%."),
          figura(f["pred_real"], 9, "Figura 6. Predicción contra valor real en procesadores posteriores al entrenamiento (escala log; banda = ±20%)."),
          P("4.3 Prueba con procesadores realmente nuevos (2023–2026)", "h2"),
          P("Con modelos entrenados con lanzamientos hasta 2023 (y hasta 2024) se evaluaron los procesadores posteriores que solo "
            "existen en Blender. El v7 ordena con correlación de 0.85 y 0.83, y 57% y 70% quedan dentro de ±20% del valor de Blender "
            "traducido a cpuMark (sin los procesadores traducidos: 45% y 43%). Contar núcleos ordena con 0.84–0.86: en procesadores "
            "nuevos el valor del modelo no está en el orden sino en dar un puntaje estimado."),
          KeepTogether([P("4.4 ¿Qué pieza importa? (ablación temporal)", "h2"),
                        figura(f["ablacion"], 14.5, "Figura 7. Cambio en el MAE de 5 cortes al quitar una sola pieza del híbrido (X6).")]),
          P("Cada componente del diseño se justifica: los núcleos son imprescindibles; quitar el segundo paso, el logaritmo, las "
            "variables del nombre o la recta empeora el error. Los datos externos, en cambio, estorban hacia el futuro, y frecuencia "
            "y socket no aportan."),
          KeepTogether([P("4.5 Drift e incertidumbre", "h2"),
                        figura(f["drift"], 15, "Figura 8. Cambio en las entradas (PSI) y caída de desempeño de un modelo entrenado con procesadores ≤2019.")]),
          P("Entre procesadores ≤2019 y 2020–2022 hay drift fuerte en fabricante, tipo móvil, núcleos, familia y TDP. Evidently, "
            "ejecutado como servicio aparte, detecta drift en 17 de 18 variables y coincide con nuestro PSI en 8 de 9 (el caso "
            "distinto está en la frontera: PSI 0.092 contra umbral 0.1). Un modelo entrenado con procesadores viejos pasa de MAE 533 "
            "en ellos a 3,727 en los nuevos (R² en log de 0.97 a 0.63) y en promedio sobreestima (el valor real es ~0.71 veces lo "
            "predicho). También se midieron intervalos de confianza al 80%: conformal clásico cubrió 52%, conformal temporal 61% y "
            "regresión por cuantiles 31%; por eso no se publican intervalos."),
          KeepTogether([P("4.6 El modelo en producción", "h2"), figura(f["arquitectura"], 15, "Figura 9. Arquitectura en la nube.")]),
          P("MLflow guarda metadatos en SQLite y artefactos aparte; cada corrida registra versión y hash de los datos, versión del "
            "código y métricas; el registro de modelos usa el alias “produccion”, de modo que cambiar o revertir un modelo es mover "
            "el alias. Se sirve en línea (API FastAPI local y página Streamlit) y por lotes, con monitoreo de latencia, errores y "
            "alertas de calidad de entrada. Un pipeline semanal en GitHub Actions baja el estado de MLflow, reentrena solo lo que "
            "tenga datos nuevos, compara con producción y promueve únicamente si el MAE mejora al menos 1%. "
            f"En su primera corrida en la nube (snapshot de Blender {ult.get('blender_snapshot', '')}) el modelo cruzado reentrenado "
            f"obtuvo {ult.get('cruzado', {}).get('mae', 0):,.0f} contra {ult.get('cruzado', {}).get('mae_produccion', 0):,.0f} de "
            f"producción (0.99% mejor) y, por la regla del 1%, producción se mantuvo en v{ult.get('produccion', 7)}."),
          conclusion("Con un examen honesto, el modelo final se equivoca ~2,400 puntos y acierta dentro de ±20% en la mitad de los "
                     "procesadores nuevos, cuatro veces mejor que las líneas base. La ablación justifica cada pieza, el monitoreo "
                     "confirma con dos métodos que el futuro sí cambia, y las mediciones de incertidumbre mostraron que prometer "
                     "rangos sin calibración temporal sería engañoso.")]

    # ---------------- 5. conclusiones
    s += [PageBreak(), P("5. Conclusiones y trabajo futuro", "h1"),
          P("5.1 Reflexión sobre los resultados", "h2")] + lista([
              "<b>Los datos movieron más que los modelos.</b> Corregir núcleos mal contados bajó el error 303 puntos; 340 configuraciones, Optuna, boosting de otras librerías y transfer learning movieron 1% o menos.",
              "<b>Cómo se mide cambia lo que se escoge.</b> Con k-fold aleatorio ganaba otro modelo y el error parecía seis veces menor; la validación temporal de 5 cortes es la que se parece al uso real, y evitó escoger una red neuronal que diverge en algunos periodos.",
              "<b>Extrapolar requiere estructura.</b> Los árboles no predicen más allá de lo visto; la recta del híbrido y los pesos por antigüedad son lo que permite seguir la tendencia.",
              "<b>Un sistema automático necesita reglas contra el ruido.</b> Sin el margen mínimo de 1%, el pipeline promovía versiones por diferencias de 2 puntos.",
          ]) + [
          P("5.2 Utilidad", "h2"),
          P("El modelo sirve para ubicar un procesador en su “liga” de rendimiento antes de que exista un benchmark, comparar "
            "opciones y estimar catálogos completos por lotes. Está publicado en una página con predicción individual, por archivo "
            "CSV y monitoreo, y se mantiene solo."),
          P("5.3 Limitaciones", "h2"),
          tabla([["Limitación", "Consecuencia"],
                 ["PassMark en Kaggle llega a 2022", "Para 2023–2026 se depende de Blender (otra escala) y de un traductor con 81% de acierto"],
                 ["Año de prueba en lugar de año de lanzamiento", "Algunos procesadores aparecen más nuevos de lo que son (se corrige con fichas cuando existen)"],
                 ["Ordena casi igual que contar núcleos en CPUs nuevos", "Su valor está en estimar el puntaje, no en ordenar"],
                 ["~50% de los procesadores nuevos fuera de ±20%", "Es una estimación, no reemplaza un benchmark"],
                 ["Intervalos de confianza sin calibrar", "Cubren 31–61% cuando prometen 80%; no se publican"],
                 ["Datos externos con 38% de cobertura", "Casi vacíos para procesadores nuevos; hoy estorban"]],
                [6.5, 10]),
          P("5.4 Trabajo futuro", "h2")] + lista([
              "Conseguir datos recientes de PassMark (licencia académica) y el año de lanzamiento real para todos los procesadores: es la palanca con más impacto esperado.",
              "Intervalos de confianza calibrados con el error de años futuros (conformal temporal adaptativo), mostrados con advertencia.",
              "Más cobertura de fichas técnicas para procesadores nuevos (caché, litografía, núcleos P/E) para que las variables externas ayuden en lugar de estorbar.",
              "Monitoreo con verdad de campo: comparar semanalmente contra benchmarks reales conforme lleguen y disparar alertas de drift.",
              "Servir la API en la nube junto a la página y evaluar restricciones de monotonía cuando haya más datos recientes.",
          ]) + [
          conclusion("El proyecto cumple su objetivo con un resultado honesto: un modelo útil para estimar y comparar procesadores, "
                     "medido como se usaría y operado como un sistema que se mantiene solo. Su principal límite es la antigüedad de "
                     "los datos, y por eso el trabajo futuro se concentra en conseguir mejores datos más que en modelos más complejos.")]

    # ---------------- anexo
    s += [P("Anexo: reproducibilidad y entregables", "h1"),
          tabla([["Entregable", "Ubicación"],
                 ["Código fuente", "github.com/Tracker315/cpu-perf-mlops: src/ (producción y pipeline), archivo/ (experimentos E1–E13, búsqueda, X1–X7, transferencia)"],
                 ["Notebook", "notebooks/Proyecto_CPU_MLOps.ipynb"],
                 ["Dataset", "datos/raw/v26/CPU_benchmark_v4.csv (copia de Kaggle v26) y datos/externos/ (fuentes de apoyo con su versión)"],
                 ["Resultados", "reportes/ (csv, json y figuras); MLflow con todos los experimentos (repositorio privado en Hugging Face)"],
                 ["Demo", "cpu-perf-mlops.streamlit.app"],
                 ["Este reporte", "reporte/generar_reporte.py regenera el PDF y sus figuras desde reportes/"]],
                [3.6, 12.9]),
          Spacer(1, 6),
          P("Comandos principales (desde la raíz del repositorio): <i>python -m src.pipeline</i> (pipeline completo), "
            "<i>python -m archivo.experimentos e13</i>, <i>python -m archivo.busqueda</i>, <i>python -m archivo.extras</i>, "
            "<i>mlflow ui --backend-store-uri sqlite:///mlflow.db</i>."),
          P("Referencias", "h2")] + lista([
              "Caruana, R., Niculescu-Mizil, A., Crew, G. y Ksikes, A. (2004). Ensemble selection from libraries of models. ICML.",
              "Torgo, L., Ribeiro, R., Pfahringer, B. y Branco, P. (2013). SMOTE for regression. EPIA.",
              "Branco, P., Torgo, L. y Ribeiro, R. (2017). SMOGN: a pre-processing approach for imbalanced regression. LIDTA.",
              "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.",
              "PassMark Software, CPU benchmarks; Blender Foundation, Blender Open Data (CC0); MLflow; Evidently.",
          ])

    doc = SimpleDocTemplate(str(SALIDA), pagesize=letter, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="Reporte final: rendimiento de CPUs con ML y MLOps", author="Elias Rodriguez")
    doc.build(s, onFirstPage=lambda c, d: None, onLaterPages=pie)
    print(f"listo: {SALIDA}")


if __name__ == "__main__":
    construir()
