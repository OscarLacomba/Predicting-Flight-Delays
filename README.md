# Predicting-Flight-Delays
His project will take you through the complete data science for predicting flight delays
# ✈️ Predicción de Retrasos de Vuelos

Proyecto de ciencia de datos de extremo a extremo: predicción de retrasos en vuelos domésticos de EE. UU. usando el dataset [2015 Flight Delays and Cancellations](https://www.kaggle.com/datasets/usdot/flight-delays) (Kaggle, `usdot/flight-delays`, fuente original: U.S. DOT — Bureau of Transportation Statistics).

Incluye un notebook completo (limpieza de datos, EDA, feature engineering, 3 modelos de clasificación, evaluación, interpretación y recomendaciones de negocio, más un bonus de regresión) y un dashboard interactivo en Streamlit con predicción en vivo.

🔗 **Demo en vivo:** `[agrega aquí el link a tu Space de Hugging Face una vez desplegado]`

## Tabla de contenidos

- [Descripción del proyecto](#descripción-del-proyecto)
- [Dataset](#dataset)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Metodología](#metodología)
- [Resultados](#resultados)
- [Dashboard](#dashboard)
- [Cómo usarlo](#cómo-usarlo)
- [Tech stack](#tech-stack)
- [Recomendaciones de negocio](#recomendaciones-de-negocio)
- [Próximos pasos](#próximos-pasos)
- [Licencia y fuente de datos](#licencia-y-fuente-de-datos)

## Descripción del proyecto

¿Se puede predecir si un vuelo va a salir tarde antes de que despegue — usando solo información disponible al momento de reservar (aerolínea, ruta, fecha y hora programada)? Este proyecto aborda el problema como una tarea de **clasificación binaria** (¿se retrasa más de 15 minutos o no?), y de forma adicional como una tarea de **regresión** (¿cuántos minutos exactos se retrasará?).

El trabajo cubre el ciclo completo de un proyecto de ciencia de datos:

1. Comprensión de los datos
2. Limpieza de datos
3. Análisis exploratorio (EDA) con 5+ visualizaciones
4. Ingeniería de variables (con manejo explícito de *data leakage*)
5. Definición formal del problema de predicción
6. Entrenamiento de 3 modelos de clasificación
7. Evaluación de modelos (accuracy, precision, recall, F1, ROC-AUC)
8. Interpretación de modelos (feature importance + permutation importance)
9. Recomendaciones de negocio
10. **Bonus:** regresión para predecir los minutos exactos de retraso

## Dataset

- **Fuente:** [usdot/flight-delays en Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays)
- **Tamaño:** ~5.8 millones de vuelos domésticos de EE. UU. durante 2015
- **Archivos:** `flights.csv` (datos de cada vuelo), `airlines.csv` (códigos IATA → nombre de aerolínea), `airports.csv` (códigos IATA → nombre/ciudad/estado de aeropuerto)
- **Variable objetivo:** `IS_DELAYED` = 1 si `DEPARTURE_DELAY` > 15 minutos (clasificación); `DEPARTURE_DELAY` en minutos (regresión, bonus)

El dataset completo no está incluido en este repositorio por su tamaño; ver instrucciones de descarga en [Cómo usarlo](#cómo-usarlo).

## Estructura del repositorio

```
.
├── app.py                          # Dashboard de Streamlit — página "Overview"
├── pages/
│   ├── 1_📊_EDA.py                  # Visualizaciones exploratorias interactivas
│   ├── 2_🔮_Predictor.py            # Predicción en vivo (clasificación + regresión)
│   └── 3_📈_Model_Performance.py    # Métricas comparativas e interpretación
├── utils.py                        # Funciones compartidas del dashboard
├── artifacts/                      # Modelos entrenados (.joblib), métricas y datos de muestra
├── notebooks/
│   └── flight_delay_prediction.ipynb   # Notebook completo (Tasks 1-9 + bonus)
├── scripts/
│   └── export_artifacts.py         # Regenera artifacts/ desde el notebook ya ejecutado
├── requirements.txt
└── README.md
```

## Metodología

| Task | Contenido | Dónde |
|---|---|---|
| 1. Data Understanding | Shape, tipos de datos, valores faltantes, variables relevantes | Notebook |
| 2. Data Cleaning | Deduplicación, manejo de cancelados/desviados, parsing de fechas/horas, outliers | Notebook |
| 3. EDA | Tasa de retraso por aerolínea, aeropuerto, mes, hora, distribución de minutos | Notebook + Dashboard |
| 4. Feature Engineering | Variables temporales, de ruta, y tasas históricas (sin data leakage) | Notebook |
| 5. Define Target | `IS_DELAYED` (umbral de 15 min), análisis de desbalance de clases | Notebook |
| 6. Build Models | Logistic Regression, Random Forest, XGBoost (pipelines con preprocesamiento) | Notebook |
| 7. Evaluate | Accuracy, precision, recall, F1, ROC-AUC, matriz de confusión, curva ROC | Notebook + Dashboard |
| 8. Interpretation | Feature importance, permutation importance | Notebook + Dashboard |
| 9. Recommendations | Recomendaciones para aerolíneas, aeropuertos y viajeros | Notebook |
| Bonus | Regresión (Linear, Random Forest, XGBoost) para minutos exactos de retraso | Notebook + Dashboard |

Una decisión de diseño importante: las variables que solo se conocen **después** de que el vuelo despega o aterriza (`ARRIVAL_DELAY`, `TAXI_OUT`, `AIR_TIME`, los desgloses de causas de retraso, etc.) se excluyen explícitamente de los modelos para evitar *data leakage*. Las variables históricas (tasa de retraso por aerolínea/aeropuerto) se calculan únicamente con el conjunto de entrenamiento.

## Resultados

> Los valores de la tabla corresponden a la corrida de referencia incluida en `artifacts/`. Reemplázalos con los de tu propia corrida sobre el dataset real (ver `classification_results.csv` y `regression_results.csv` generados por `scripts/export_artifacts.py`, o la página *Model Performance* del dashboard).

**Clasificación — ¿se retrasa el vuelo? (>15 min)**

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.717 | 0.620 | 0.717 | 0.665 | 0.791 |
| Random Forest | 0.741 | 0.651 | 0.724 | 0.686 | 0.811 |
| **XGBoost** | **0.740** | **0.652** | 0.717 | 0.683 | **0.812** |

**Regresión (bonus) — minutos exactos de retraso**

| Modelo | MAE (min) | RMSE (min) | R² |
|---|---|---|---|
| Linear Regression | 10.20 | 12.94 | 0.363 |
| Random Forest Regressor | 9.75 | 12.41 | 0.414 |
| **XGBoost Regressor** | **9.75** | **12.40** | **0.415** |

Las variables más influyentes consistentemente son las tasas históricas de retraso por aerolínea/aeropuerto, la hora de salida programada, y el mes — ver el detalle de feature importance y permutation importance en el notebook o en el dashboard.

## Dashboard

Dashboard de Streamlit con 4 páginas:

- **Overview** — resumen del dataset y del proyecto.
- **EDA** — gráficas interactivas con filtros por aerolínea y mes.
- **Predictor** — ingresa aerolínea, ruta, fecha/hora y distancia, y obtén la probabilidad de retraso y los minutos estimados, en vivo.
- **Model Performance** — tablas comparativas e importancia de variables.

## Cómo usarlo

```bash
# Clonar el repositorio
git clone <url-de-este-repo>
cd <nombre-del-repo>

# Instalar dependencias
pip install -r requirements.txt

# Correr el dashboard
streamlit run app.py
```

Para correr el notebook con el dataset real: descarga `flights.csv`, `airlines.csv` y `airports.csv` desde [Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays) (requiere cuenta gratuita), y sigue las instrucciones de configuración al inicio de `notebooks/flight_delay_prediction.ipynb` (admite carga vía API de Kaggle, subida manual, o un generador de datos sintéticos para pruebas rápidas).

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · Streamlit · Plotly · matplotlib/seaborn · Jupyter / Google Colab

## Recomendaciones de negocio

- **Aerolíneas:** ampliar los tiempos de holgura en vuelos de tarde/noche (donde se acumulan los retrasos de rotaciones previas) y en los meses de mayor demanda.
- **Aeropuertos:** reforzar personal y recursos de pista en los meses pico y en las franjas horarias de mayor tasa de retraso.
- **Viajeros:** preferir vuelos matutinos y directos, y considerar el historial de retraso de la aerolínea/aeropuerto al reservar.

(Detalle completo en la sección Task 9 del notebook.)

## Próximos pasos

- Reentrenar con el dataset completo (~5.8M filas) en vez de una muestra.
- Incorporar datos de clima por aeropuerto/fecha.
- Explorar SHAP para interpretación más granular a nivel de predicción individual.
- Agregar autenticación/monitoreo si el dashboard se usa en producción.

## Licencia y fuente de datos

Datos originales publicados por el U.S. Department of Transportation (dominio público), distribuidos en Kaggle como [usdot/flight-delays](https://www.kaggle.com/datasets/usdot/flight-delays). Código de este repositorio disponible bajo licencia MIT (agrega un archivo `LICENSE` si quieres declararlo formalmente).
