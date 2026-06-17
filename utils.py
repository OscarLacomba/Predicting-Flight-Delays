import os
import json

import joblib
import pandas as pd
import streamlit as st

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


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
    """Construye una sola fila de variables, exactamente con las mismas columnas y la misma
    logica de ingenieria de variables usada para entrenar los modelos (ver Task 4/6 del notebook).
    Las tasas historicas usan un valor global de respaldo para aerolineas/aeropuertos/rutas no vistas."""
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
