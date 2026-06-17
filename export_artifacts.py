# ============================================================================
# EXPORTAR ARTEFACTOS PARA EL DASHBOARD (GitHub + Hugging Face Space)
# ============================================================================
# IMPORTANTE: este codigo NO se ejecuta como un script independiente.
# Esta pensado para PEGARSE como una celda nueva al FINAL de tu notebook de
# Colab, DESPUES de haber corrido todas las celdas de los Tasks 1-9 y el
# Bonus de regresion (es decir, despues de "corrio todo el notebook").
# Reutiliza variables que ya existen en memoria en esa sesion: flights,
# train_df, clf_models, reg_models, results_df, reg_results_df,
# airline_delay_rate, origin_delay_rate, dest_delay_rate, route_freq,
# global_delay_rate, FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
# DELAY_THRESHOLD, airlines_df, airports_df, RANDOM_STATE, IN_COLAB.
#
# Al correrlo, genera una carpeta artifacts/ con todo lo que necesita el
# dashboard de Streamlit, la comprime en artifacts.zip, y (si estas en Colab)
# la descarga automaticamente a tu computadora. Luego solo reemplaza el
# contenido de la carpeta artifacts/ de este repositorio con esos archivos.
# ============================================================================

import os
import json
import zipfile
import joblib

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# 1) Mejor modelo de cada tipo, segun las metricas ya calculadas en el notebook
best_clf_name = results_df["ROC-AUC"].idxmax()
best_reg_name = reg_results_df["R2"].idxmax()
joblib.dump(clf_models[best_clf_name], os.path.join(ARTIFACT_DIR, "best_classifier.joblib"))
joblib.dump(reg_models[best_reg_name], os.path.join(ARTIFACT_DIR, "best_regressor.joblib"))

with open(os.path.join(ARTIFACT_DIR, "model_info.json"), "w") as f:
    json.dump({
        "best_classifier": best_clf_name,
        "best_regressor": best_reg_name,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "delay_threshold": DELAY_THRESHOLD,
        "is_demo_data": False,  # estos SI son resultados con el dataset real
    }, f, indent=2)

# 2) Tablas de metricas (para la pagina Model Performance)
results_df.to_csv(os.path.join(ARTIFACT_DIR, "classification_results.csv"))
reg_results_df.to_csv(os.path.join(ARTIFACT_DIR, "regression_results.csv"))

# 3) Feature importance del mejor clasificador (si es un modelo de arbol)
try:
    feature_names = clf_models[best_clf_name].named_steps["prep"].get_feature_names_out()
    importances = clf_models[best_clf_name].named_steps["clf"].feature_importances_
    import pandas as pd
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    fi_df.to_csv(os.path.join(ARTIFACT_DIR, "feature_importance.csv"), index=False)
except AttributeError:
    print("El mejor clasificador no es un modelo de arbol; se omite feature_importance.csv")

# 4) Tablas de lookup, necesarias para construir las variables de un vuelo
#    nuevo en la pagina Predictor (mismas tablas calculadas en el Task 6)
lookup_tables = {
    "global_delay_rate": float(global_delay_rate),
    "airline_delay_rate": airline_delay_rate.to_dict(),
    "origin_delay_rate": origin_delay_rate.to_dict(),
    "dest_delay_rate": dest_delay_rate.to_dict(),
    "route_frequency": {k: int(v) for k, v in route_freq.to_dict().items()},
}
with open(os.path.join(ARTIFACT_DIR, "lookup_tables.json"), "w") as f:
    json.dump(lookup_tables, f, indent=2)

# 5) Nombres de aerolineas/aeropuertos, para los menus desplegables del dashboard
airlines_df.to_csv(os.path.join(ARTIFACT_DIR, "airlines.csv"), index=False)
airports_df.to_csv(os.path.join(ARTIFACT_DIR, "airports.csv"), index=False)

# 6) Muestra del dataset limpio, para que la pagina EDA del dashboard tenga
#    datos reales con los que graficar (no se sube el dataset completo)
flights.sample(min(20_000, len(flights)), random_state=RANDOM_STATE).to_csv(
    os.path.join(ARTIFACT_DIR, "flights_sample.csv"), index=False
)

# 7) Comprimir todo en un .zip para descargar facilmente
zip_path = "artifacts.zip"
with zipfile.ZipFile(zip_path, "w") as zf:
    for fname in os.listdir(ARTIFACT_DIR):
        zf.write(os.path.join(ARTIFACT_DIR, fname), fname)

print(f"Artefactos exportados en '{ARTIFACT_DIR}/' y comprimidos en '{zip_path}'")
print("Contenido:", os.listdir(ARTIFACT_DIR))

if IN_COLAB:
    from google.colab import files
    files.download(zip_path)
