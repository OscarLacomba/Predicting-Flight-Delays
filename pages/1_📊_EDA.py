import os

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts")


@st.cache_data
def load_csv(filename, **kwargs):
    return pd.read_csv(os.path.join(ARTIFACT_DIR, filename), **kwargs)


st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Análisis Exploratorio de Datos")

flights = load_csv("flights_sample.csv")
airlines_df = load_csv("airlines.csv")
flights = flights.drop(columns=["AIRLINE_NAME"], errors="ignore")
flights = flights.merge(
    airlines_df.rename(columns={"AIRLINE": "AIRLINE_NAME"}),
    left_on="AIRLINE", right_on="IATA_CODE", how="left",
)

with st.sidebar:
    st.header("Filtros")
    selected_airlines = st.multiselect(
        "Aerolíneas", sorted(flights["AIRLINE_NAME"].dropna().unique())
    )
    month_range = st.slider("Mes", 1, 12, (1, 12))

filtered = flights.copy()
if selected_airlines:
    filtered = filtered[filtered["AIRLINE_NAME"].isin(selected_airlines)]
filtered = filtered[(filtered["MONTH"] >= month_range[0]) & (filtered["MONTH"] <= month_range[1])]

st.caption(f"{len(filtered):,} vuelos en la selección actual")

col1, col2 = st.columns(2)

with col1:
    delay_by_airline = (
        filtered.groupby("AIRLINE_NAME")["IS_DELAYED"].mean().sort_values(ascending=False) * 100
    )
    fig = px.bar(
        delay_by_airline, orientation="h",
        labels={"value": "Tasa de retraso (%)", "AIRLINE_NAME": ""},
        title="Tasa de retraso por aerolínea",
    )
    fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")

with col2:
    top_airports = filtered["ORIGIN_AIRPORT"].value_counts().head(15).index
    delay_by_airport = (
        filtered[filtered["ORIGIN_AIRPORT"].isin(top_airports)]
        .groupby("ORIGIN_AIRPORT")["IS_DELAYED"].mean().sort_values(ascending=False) * 100
    )
    fig = px.bar(
        delay_by_airport, orientation="h",
        labels={"value": "Tasa de retraso (%)", "ORIGIN_AIRPORT": ""},
        title="Tasa de retraso por aeropuerto de origen (top 15 por volumen)",
    )
    fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")

col3, col4 = st.columns(2)

with col3:
    delay_by_month = filtered.groupby("MONTH")["IS_DELAYED"].mean() * 100
    fig = px.line(
        delay_by_month, markers=True,
        labels={"value": "Tasa de retraso (%)", "MONTH": "Mes"},
        title="Tasa de retraso por mes",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

with col4:
    delay_by_hour = filtered.groupby("SCHED_DEP_HOUR")["IS_DELAYED"].mean() * 100
    fig = px.line(
        delay_by_hour, markers=True,
        labels={"value": "Tasa de retraso (%)", "SCHED_DEP_HOUR": "Hora de salida"},
        title="Tasa de retraso por hora de salida programada",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

st.subheader("Distribución de los minutos de retraso")
fig = px.histogram(
    filtered, x="DEPARTURE_DELAY", nbins=60, range_x=[-30, 180],
    title="Distribución de DEPARTURE_DELAY (recortado entre -30 y 180 min para visualizar mejor)",
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, width="stretch")
