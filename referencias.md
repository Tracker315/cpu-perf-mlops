# Referencias

## Prediccion de rendimiento de procesadores
- Joseph, Vaswani y Thazhuthaveetil (2006). Regresion lineal para predecir rendimiento a partir de parametros de microarquitectura (citado en el Avance 1).
- Ein-Dor y Feldmesser (1987). Computer Hardware dataset (UCI), usado en la Actividad 4.
- Wang et al. (2019), ACM TACO. Red neuronal que predice SPEC y Geekbench a partir de las especificaciones de CPUs de Intel (~5% de error en SPEC); reporta que gana a la regresion lineal. https://vlsiarch.eecs.harvard.edu/resource/predicting-new-workload-or-cpu-performance-analyzing-public-datasets
- Silhouette (2022), arXiv 2212.08046. Une SPEC CPU 2017 con Intel ARK para predecir CPUs no vistas; misma idea que unir PassMark con TechPowerUp/pc_build en este proyecto. https://arxiv.org/pdf/2212.08046v1.pdf
- SPECnet (AMD, ICPE 2018 companion). Configuracion de la maquina -> puntaje SPEC con 5-7% de error. https://research.spec.org/icpe_proceedings/2018/companion/p29.pdf

## Aumentacion de datos para regresion
- Torgo, Ribeiro, Pfahringer y Branco (2013). SMOTE for Regression (SMOTER): interpola entre casos raros del objetivo.
- Branco, Torgo y Ribeiro (2017). SMOGN: a Pre-processing Approach for Imbalanced Regression. PMLR 74:36-50. https://proceedings.mlr.press/v74/branco17a.html
- Chawla et al. (2002). SMOTE (visto en clase, diapositivas de clustering).

## Ensambles
- Caruana, Niculescu-Mizil, Crew y Ksikes (2004). Ensemble Selection from Libraries of Models. ICML. Base del combinador de la competencia y del experimento E8.
- Wolpert (1992). Stacked Generalization.

## Drift y MLOps
- Gama, Zliobaite, Bifet, Pechenizkiy y Bouchachia (2014). A survey on concept drift adaptation. ACM Computing Surveys 46(4). https://eprints.bournemouth.ac.uk/22491/
- MLflow: tracking (backend SQLite), artifact store y Model Registry. https://mlflow.org/docs/latest/

## Datos
- PassMark CPU benchmarks (Kaggle alanjo/cpu-benchmarks, CPU_benchmark_v4.csv y CPU_r23_v2.csv).
- TechPowerUp CPU database (Kaggle baraazaid/cpu-and-gpu-stats).
- pc_build (Hugging Face mehax/pc_build).
