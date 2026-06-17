import streamlit as st

from utils import load_csv, load_json

st.set_page_config(page_title="Flight Delay Dashboard", page_icon="✈️", layout="wide")

st.title("✈️ Predicción de Retrasos de Vuelos")
st.caption(
    "Dataset: 2015 Flight Delays and Cancellations (Kaggle, usdot/flight-delays) · "
    "Dashboard del proyecto de ciencia de datos"
)

model_info = load_json("model_info.json")
if model_info.get("is_demo_data"):
    st.warning(
        "⚠️ Estos resultados usan datos **sintéticos de demostración**. Reemplaza la carpeta "
        "`artifacts/` con los archivos exportados desde tu notebook (ejecutado con el dataset real) "
        "para ver resultados reales. Instrucciones en el README."
    )

flights = load_csv("flights_sample.csv")

st.subheader("Resumen del dataset")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Vuelos en la muestra", f"{len(flights):,}")
col2.metric("Tasa de retraso global", f"{flights['IS_DELAYED'].mean():.1%}")
col3.metric("Aerolíneas", flights["AIRLINE"].nunique())
col4.metric("Aeropuertos", flights["ORIGIN_AIRPORT"].nunique())

st.divider()

st.subheader("Acerca de este proyecto")
st.markdown(
    """
Este dashboard acompaña un proyecto completo de ciencia de datos sobre retrasos de vuelos:
comprensión y limpieza de datos, análisis exploratorio, ingeniería de variables, y 3 modelos
de clasificación (Logistic Regression, Random Forest, XGBoost) para predecir si un vuelo se
retrasará más de 15 minutos — más un modelo de regresión (bonus) para estimar los minutos
exactos de retraso.

Usa el menú de la izquierda para navegar:
- **EDA** — visualizaciones exploratorias interactivas, con filtros por aerolínea y mes.
- **Predictor** — ingresa los datos de un vuelo y obtén una predicción en vivo.
- **Model Performance** — tablas comparativas e interpretación de los modelos.
"""
)

st.info(
    f"Mejor modelo de clasificación: **{model_info['best_classifier']}** · "
    f"Mejor modelo de regresión: **{model_info['best_regressor']}**"
)
