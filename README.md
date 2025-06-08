Librería de python utilizada para análisis y ajuste de datos, basado en contenidos vistos en la materia Ciencia de Datos III de la Licenciatura en Ciencia de Datos de la Universidad Nacional del Litoral.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Contiene una serie de clases de alto y bajo nivel.

Clases de alto nivel:
  - subclases de Modelo (RL, Log, etc.)
  - Dataframe

Clases de bajo nivel:
  - ContDist (distribuciones contínuas, subclase de scipy.stats.rv_continuous)
  - Datos (subclase de np.ndarray)
    - QuantDatos (datos cuantitativos)
    - CualDatos (datos cualitativos)
  - Modelo
    - RQmodel (respuesta cuantitativa)
    - RCmodel (respuesta cualitativa)
    - PQmodel (predictor cuantitativo)
    - PCmodel (predictor cualitativo)
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Workflow típico:
  - Modelado:
    - carga de datos en un objeto tipo Dataframe
    - modelo basado en el Dataframe
    - anova, validación, predicciones, parámetros, gráfico, análisis de residuos, etc.
  - Estimación de densidades:
    - (por ahora sólo en bajo nivel; se planea trasladar a Dataframe)
