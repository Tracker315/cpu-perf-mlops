"""X7: el ganador de X2 (hibrido con monotonia) en las condiciones de produccion v7: nucleos reales + CPUs de Blender
traducidos (pseudo-etiquetas). Misma evaluacion que cruzado.paso4a (5 cortes) y paso4b (orden en CPUs 2023-2026)."""
import json

import mlflow
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from src import config, cruzado, datos, modelos
from src.entrenar import preparar_mlflow
from archivo.extras import SIGNOS, HibridoX, anotar


def ajustar_1hilo(m, X, y):
    with threadpool_limits(1):
        return clone(m).fit(X, y)


BASES = {
    "Combo (base actual de v7)": lambda: modelos.con_log(modelos.Combo()),
    "hibrido recta + HGB": lambda: modelos.con_log(HibridoX()),
    "hibrido recta + HGB + monotonia suave": lambda: modelos.con_log(HibridoX(signos=SIGNOS["suave"])),
    "hibrido recta + HGB + monotonia completa": lambda: modelos.con_log(HibridoX(signos=SIGNOS["completo"])),
}


def correr():
    cruzado.ajustar = ajustar_1hilo
    info = datos.descargar()
    df = datos.cargar(info["ruta"])
    modelos.usar_externos(True)
    f = cruzado.fichas()
    bl = cruzado.blender_por_clave()
    pm, b = cruzado.construir(df, f, bl)
    b["pm_cpuMark"] = b["clave"].map(pm.drop_duplicates("clave").set_index("clave")["cpuMark"])
    Xpm_orig, _ = datos.preparar(df, externos=True)
    Xpm = cruzado.matriz(pm, "testDate")
    b_ok = b[b["anio_lanz"].notna() | b["en_passmark"]].reset_index(drop=True)
    Xb = cruzado.matriz(b_ok, "anio_lanz")
    rb = json.load(open(config.REPORTES / "cruzado_resumen.json"))["modelo_b"]
    nombre_b = min(rb, key=lambda k: rb[k]["mape"])
    print("modelo B:", nombre_b, flush=True)
    preparar_mlflow("cpu-perf/extras")
    salida = {}
    with mlflow.start_run(run_name=f"x7_condiciones_v7__data-v{info['version']}",
                          tags={"etapa": "experimento_extra", "data_version": info["version"], "data_hash": info["hash"],
                                "version_codigo": config.version_codigo()}):
        for nombre, hacer in BASES.items():
            cruzado.modelo_base = hacer
            print(f"== base A: {nombre}", flush=True)
            r4, w, tabla = cruzado.paso4a_cortes(pm, b_ok, Xpm_orig, Xpm, Xb, nombre_b)
            r5 = cruzado.paso4b_futuro(pm, b_ok, Xpm, Xb, nombre_b, w)
            m = {"mae": r4["A' + pseudo-etiquetas"]["mae"], "kpi_dentro_20pct": r4["A' + pseudo-etiquetas"]["kpi"],
                 "mae_sin_pseudo": r4["A' (nucleos reales)"]["mae"],
                 **{f"mae_corte_{int(r.corte)}": float(r["A' + pseudo-etiquetas"]) for _, r in tabla.iterrows()},
                 **{f"futuro_{k}": float(v) for k, v in r5.items()}}
            anotar("X7_condiciones_v7", nombre, m)
            print("   futuro:", {k: round(float(v), 3) for k, v in r5.items()}, flush=True)
            salida[nombre] = m
    json.dump(salida, open(config.REPORTES / "extras_x7_v7.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    correr()
