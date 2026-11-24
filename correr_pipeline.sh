#!/bin/bash
# Corre el pipeline completo. Pensado para cron, por ejemplo cada lunes a las 3 am:
#   0 3 * * 1  /ruta/a/Proj/correr_pipeline.sh
# Solo reentrena si Kaggle publico una version nueva del dataset (cambia el hash).
cd "$(dirname "$0")"
mkdir -p logs
PY="${PYTHON:-python}"
export MLFLOW_DISABLE_AGENT_HINT=1
"$PY" -m src.pipeline "$@" >> "logs/pipeline_$(date +%Y%m%d).log" 2>&1
echo "pipeline terminado con codigo $? ($(date))" >> logs/pipeline_$(date +%Y%m%d).log
