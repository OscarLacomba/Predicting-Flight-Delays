import json
import os

import joblib
import pandas as pd
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts")


@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(ARTIFACT_DIR, filename))


@st.cache_data
def load_csv(filename, **kwargs):
    return pd.read_csv(os.path.join(ARTIFACT_DIR, filename), **kwargs)


@st.cache_data
def load_json(filename):
    with open(os.path.join(ARTIFACT_DIR, filename)) as f:
        return json.load(f)


def month_to_season(month):
    if month in (12, 1, 2):
        return "Invierno"
    elif month in (3, 4, 5):
        return "Primavera"
    elif month in (6, 7, 8):
        return "Verano"
    return "Otono"


def build_feature_row(airline, origin, dest, month, day_of_week, sched_dep_hour, distance, lookups, model_info):
    """Construye una sola fila de variables, con la misma logica de ingenieria de variables
    usada para entrenar los modelos (ver Task 4/6 del notebook). Las tasas historicas usan un
    valor global de respaldo para aerolineas/aeropuertos/rutas no vistas en entrenamiento."""
    global_rate = lookups["global_delay_rate"]
    route = f"{origin}-{dest}"
    row = {
        "AIRLINE": airline,
        "SEASON": month_to_season(month),
        "MONTH": month,
        "DAY_OF_WEEK": day_of_week,
        "SCHED_DEP_HOUR": sched_dep_hour,
        "IS_WEEKEND": int(day_of_week in (6, 7)),
        "DISTANCE": distance,
        "AIRLINE_DELAY_RATE": lookups["airline_delay_rate"].get(airline, global_rate),
        "ORIGIN_DELAY_RATE": lookups["origin_delay_rate"].get(origin, global_rate),
        "DEST_DELAY_RATE": lookups["dest_delay_rate"].get(dest, global_rate),
        "ROUTE_FREQUENCY": lookups["route_frequency"].get(route, 0),
    }
    return pd.DataFrame([row])[model_info["feature_columns"]]


st.set_page_config(page_title="Predictor", page_icon="🔮", layout="wide")
st.title("🔮 Predictor de Retrasos")
st.caption("Ingresa los datos de un vuelo para predecir si se retrasará y cuántos minutos.")

model_info = load_json("model_info.json")
lookups = load_json("lookup_tables.json")
airlines_df = load_csv("airlines.csv")
airports_df = load_csv("airports.csv")
classifier = load_model("best_classifier.joblib")
regressor = load_model("best_regressor.joblib")

airline_options = dict(zip(airlines_df["AIRLINE"], airlines_df["IATA_CODE"]))
airport_options = dict(
    zip(airports_df["AIRPORT"] + " (" + airports_df["IATA_CODE"] + ")", airports_df["IATA_CODE"])
)
airport_labels = sorted(airport_options.keys())

col1, col2, col3 = st.columns(3)
with col1:
    airline_name = st.selectbox("Aerolínea", sorted(airline_options.keys()))
    airline_code = airline_options[airline_name]
with col2:
    origin_label = st.selectbox("Aeropuerto de origen", airport_labels, index=0)
    origin_code = airport_options[origin_label]
with col3:
    dest_label = st.selectbox("Aeropuerto de destino", airport_labels, index=min(1, len(airport_labels) - 1))
    dest_code = airport_options[dest_label]

col4, col5, col6, col7 = st.columns(4)
month_names = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
dow_names = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
with col4:
    month = st.selectbox("Mes", list(range(1, 13)), format_func=lambda m: month_names[m - 1])
with col5:
    day_of_week = st.selectbox("Día de la semana", list(range(1, 8)), format_func=lambda d: dow_names[d - 1])
with col6:
    sched_dep_hour = st.slider("Hora de salida programada", 0, 23, 8)
with col7:
    distance = st.number_input("Distancia (millas)", min_value=50, max_value=5000, value=800, step=50)

if origin_code == dest_code:
    st.error("El aeropuerto de origen y destino no pueden ser el mismo.")
elif st.button("Predecir", type="primary"):
    X = build_feature_row(
        airline_code, origin_code, dest_code, month, day_of_week, sched_dep_hour, distance,
        lookups, model_info,
    )
    prob_delay = classifier.predict_proba(X)[0, 1]
    pred_minutes = regressor.predict(X)[0]

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Probabilidad de retraso (>15 min)", f"{prob_delay:.1%}")
        if prob_delay >= 0.5:
            st.warning("Es probable que este vuelo se retrase.")
        else:
            st.success("Es probable que este vuelo salga a tiempo.")
    with c2:
        st.metric("Minutos de retraso estimados", f"{pred_minutes:.0f} min")
        st.caption("Estimación del modelo de regresión (bonus); puede ser negativa (salida anticipada).")

if model_info.get("is_demo_data"):
    st.info("⚠️ Modelo entrenado con datos sintéticos de demostración — las predicciones no reflejan vuelos reales.")
