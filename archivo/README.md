# archivo/ — experimentos ya terminados

Aqui esta el codigo de las PRUEBAS que se hicieron para escoger y justificar el modelo. No se usa en produccion ni en el
pipeline: se guarda para poder demostrar como se llego a cada decision y repetir cualquier prueba.

Los RESULTADOS siguen en `reportes/` (csv, json, png), en MLflow (`mlflow.db` + `mlartifacts/`) y en `logs/`.

| Archivo | Que probo | Resultados | Experimento MLflow |
|---|---|---|---|
| `experimentos.py` | E1-E13: objetivo en log, variables, socket, modelos, aumentacion, validacion aleatoria vs temporal, drift, Caruana, hibrido, redes, robustez (5 cortes) | `reportes/experimentos.csv`, `robustez_detalle.csv` | `cpu-perf/experimentos` |
| `busqueda.py` | Busqueda aleatoria: 11 familias, 340 configuraciones, mezcla de Caruana honesta, confirmacion de finalistas | `reportes/busqueda_*.csv/json`, `logs/busqueda.log` | `cpu-perf/busqueda` |
| `extras.py` | X1-X6: XGBoost/LightGBM/CatBoost, monotonia, intervalos de confianza, Optuna, lineas base, ablacion | `reportes/extras*.csv` | `cpu-perf/extras` |
| `extras_v7.py` | X7: monotonia en las condiciones del v7 | `reportes/extras_x7_v7.json` | `cpu-perf/extras` |
| `transferencia.py` | Transfer learning PassMark <-> Blender (T1-T4) | `reportes/transferencia*.csv` | `cpu-perf/transferencia` |
| `simulacion.py` | Simula que cada anio llega una version nueva del dataset (monitoreo + reentrenamiento) | `reportes/simulacion.*` | `cpu-perf/simulacion` |
| `analisis.py` | Nucleos vs frecuencia, ley de Amdahl, importancia de variables | `reportes/amdahl.*`, `nucleos_vs_frecuencia.json` | `cpu-perf/analisis` |

Correr (desde la carpeta `Proj`):

```
python -m archivo.experimentos e13
python -m archivo.busqueda
python -m archivo.extras x2 x3
```

Usan los modulos de produccion de `src/` (datos, modelos, entrenar, cruzado...), asi que miden exactamente con el mismo
codigo que el modelo en produccion.
