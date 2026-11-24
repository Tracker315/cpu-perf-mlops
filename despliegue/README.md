# Pagina del modelo (Streamlit)

Pagina publica del modelo en produccion del proyecto (`cpu-perf-multicore`, alias `produccion`).
Estima el rendimiento multi-nucleo (cpuMark de PassMark) a partir de la ficha tecnica del procesador.

Publicada en **Streamlit Community Cloud** (gratis) desde el repo de GitHub: archivo principal
`despliegue/streamlit_app.py`. Cada vez que el pipeline semanal cambia el modelo en produccion, regenera esta carpeta,
hace push y Streamlit vuelve a desplegar solo (despliegue continuo).

| Carpeta / archivo | Que es |
|---|---|
| `streamlit_app.py` | La pagina: un procesador, lote CSV, monitoreo, acerca del modelo |
| `modelo/` | El modelo tal como lo guardo MLflow (`model.pkl`, `MLmodel` con la firma) + `info.json` (version, metricas, trazabilidad) |
| `src/` | Solo los modulos que usa la prediccion |
| `datos/externos/` | Copias de TechPowerUp y pc_build (completan cache, nm, GHz e hilos por nombre) |
| `requirements.txt` | Mismas versiones con las que se guardo el modelo |

`modelo/`, `src/` y `datos/` los genera `python -m src.desplegar`; no editarlos a mano.

Local: `pip install -r requirements.txt && streamlit run streamlit_app.py`
