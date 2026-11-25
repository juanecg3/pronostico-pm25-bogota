"""Importancia por permutación del modelo congelado sobre test junio."""

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.modelo_pm25 import FEATURES, SEED, cargar_parametros_campeon, cargar_particiones, entrenar_campeon  # noqa: E402

OUT_JSON = PROJECT / "experiments" / "resultados explicabilidad.json"
OUT_PLOT = PROJECT / "experiments" / "importancia variables.png"


def main():
    _, train, test = cargar_particiones()
    params = cargar_parametros_campeon()
    modelo = entrenar_campeon(train, params)
    resultado = permutation_importance(
        modelo,
        test[FEATURES],
        test["target_24h"],
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=SEED,
        n_jobs=-1,
    )
    tabla = pd.DataFrame({
        "variable": FEATURES,
        "caida_media_mae_ug_m3": resultado.importances_mean,
        "desviacion_estandar_ug_m3": resultado.importances_std,
    }).sort_values("caida_media_mae_ug_m3", ascending=False)
    ranking = [
        {
            "ranking": int(i + 1),
            "variable": fila.variable,
            "caida_media_mae_ug_m3": round(float(fila.caida_media_mae_ug_m3), 4),
            "desviacion_estandar_ug_m3": round(float(fila.desviacion_estandar_ug_m3), 4),
        }
        for i, fila in enumerate(tabla.itertuples(index=False))
    ]
    base_mae = float((test["target_24h"] - modelo.predict(test[FEATURES])).abs().mean())
    metadata = {
        "descripcion": "Importancia por permutación en la prueba de junio; ranking descriptivo, no causal",
        "metrica": "MAE; importancia = incremento del MAE al permutar una variable",
        "semilla": SEED,
        "muestras_entrenamiento": int(len(train)),
        "muestras_prueba": int(len(test)),
        "mae_rf_prueba_ug_m3": round(base_mae, 4),
        "hiperparametros_congelados_de_validacion": params,
        "variables": FEATURES,
        "repeticiones_permutacion": 10,
        "ranking_importancia": ranking,
        "nota": "La permutación se hizo en el mismo conjunto de junio ya evaluado. No se usó para ajustar el modelo ni elegir parámetros.",
    }
    OUT_JSON.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    plot = tabla.sort_values("caida_media_mae_ug_m3")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot["variable"], plot["caida_media_mae_ug_m3"], color="#277C78")
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel("Aumento medio del MAE al permutar (µg/m³)")
    ax.set_title("Random Forest: importancia por permutación en junio")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Entrenamiento: {len(train)} filas | Prueba: {len(test)} filas")
    print(f"Parámetros congelados: {params}")
    print(f"MAE RF reproducido en prueba: {base_mae:.4f} µg/m³")
    print(tabla.to_string(index=False, formatters={
        "caida_media_mae_ug_m3": "{:.4f}".format,
        "desviacion_estandar_ug_m3": "{:.4f}".format,
    }))
    print(f"JSON: {OUT_JSON}\nGráfico: {OUT_PLOT}")


if __name__ == "__main__":
    main()
