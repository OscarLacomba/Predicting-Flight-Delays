import plotly.express as px
import streamlit as st

from utils import load_csv, load_json

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("📈 Desempeño e Interpretación de los Modelos")

model_info = load_json("model_info.json")
results_df = load_csv("classification_results.csv", index_col=0)
reg_results_df = load_csv("regression_results.csv", index_col=0)

st.subheader("Clasificación: ¿se retrasa el vuelo? (>15 min)")
st.dataframe(results_df.style.highlight_max(axis=0, color="lightgreen"), width="stretch")

st.subheader("Regresión (bonus): minutos exactos de retraso")
styled_reg = reg_results_df.style.highlight_max(subset=["R2"], color="lightgreen").highlight_min(
    subset=["MAE (min)", "RMSE (min)"], color="lightgreen"
)
st.dataframe(styled_reg, width="stretch")

st.divider()

st.subheader(f"Importancia de variables — {model_info['best_classifier']}")
try:
    fi_df = load_csv("feature_importance.csv").head(10)
    fig = px.bar(
        fi_df, x="importance", y="feature", orientation="h",
        title="Top 10 variables más importantes (feature_importances_)",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
    st.plotly_chart(fig, width="stretch")
except FileNotFoundError:
    st.info("No hay datos de importancia de variables disponibles para este modelo.")

st.caption(f"Mejor clasificador: {model_info['best_classifier']} · Mejor regresor: {model_info['best_regressor']}")
if model_info.get("is_demo_data"):
    st.warning("⚠️ Estas métricas corresponden a datos sintéticos de demostración, no al dataset real.")
