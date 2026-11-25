"""Evaluación final, de una pasada, sobre objetivos de junio."""
from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "datos limpios.csv"
TUNING = PROJECT / "experiments" / "resultados ajuste modelos.json"
OUT_JSON = PROJECT / "experiments" / "evaluacion final prueba.json"
OUT_CSV = PROJECT / "experiments" / "predicciones detalladas prueba.csv"
SEED = 42


def metricas(real, pred):
    return {
        "mae_ug_m3": round(float(mean_absolute_error(real, pred)), 4),
        "rmse_ug_m3": round(float(mean_squared_error(real, pred) ** 0.5), 4),
        "r2": round(float(r2_score(real, pred)), 4),
    }


def main():
    ajuste = json.loads(TUNING.read_text(encoding="utf-8"))
    parametros = ajuste["resultados_validacion_mayo"]["random_forest"]["mejores_hiperparametros"]
    datos = pd.read_csv(DATA, parse_dates=["fecha_hora"]).sort_values("fecha_hora").reset_index(drop=True)

    # Las ventanas se calculan solo con lecturas originales observadas y hacia atrás.
    observadas = datos.set_index("fecha_hora")["pm25_original"].dropna()
    for n in (3, 6):
        media = observadas.rolling(n, min_periods=n).mean()
        datos[f"pm25_roll{n}"] = media.reindex(pd.DatetimeIndex(datos["fecha_hora"])).ffill().to_numpy()
    datos["fecha_objetivo"] = datos["fecha_hora"] + pd.Timedelta(hours=24)
    datos["target_24h"] = datos["pm25_original"].shift(-24)
    datos["hora"] = datos["fecha_hora"].dt.hour
    datos["dia_semana"] = datos["fecha_hora"].dt.day_of_week
    datos["pm25_lag1"] = datos["pm25_original"].shift(1)
    datos["pm25_lag24"] = datos["pm25_original"].shift(24)
    features = ["pm25_original", "pm25_lag1", "pm25_lag24", "pm25_roll3", "pm25_roll6", "hora", "dia_semana"]
    datos = datos.dropna(subset=features + ["target_24h"])
    train = datos[datos["fecha_objetivo"].between("2026-01-01", "2026-05-31 23:59:59")]
    test = datos[datos["fecha_objetivo"].between("2026-06-01", "2026-06-30 23:59:59")]
    if train.empty or test.empty or train.fecha_objetivo.max() >= test.fecha_objetivo.min():
        raise ValueError("Particiones vacías o con solapamiento temporal.")

    rf = RandomForestRegressor(random_state=SEED, n_jobs=-1, **parametros)
    hgb = HistGradientBoostingRegressor(random_state=SEED, max_iter=200, learning_rate=0.05)
    rf.fit(train[features], train["target_24h"])
    hgb.fit(train[features], train["target_24h"])
    real = test["target_24h"]
    persistencia = test["pm25_original"]
    pred_rf = rf.predict(test[features])
    pred_hgb = hgb.predict(test[features])

    mae_persistencia = mean_absolute_error(real, persistencia)
    mae_rf = mean_absolute_error(real, pred_rf)
    resultado = {
        "experimento": "Evaluación final de una pasada en junio de 2026",
        "estacion": "Kennedy",
        "semilla": SEED,
        "entrenamiento_objetivo": [str(train.fecha_objetivo.min()), str(train.fecha_objetivo.max())],
        "prueba_objetivo": [str(test.fecha_objetivo.min()), str(test.fecha_objetivo.max())],
        "muestras_entrenamiento": int(len(train)),
        "muestras_evaluacion_comunes": int(len(test)),
        "variables_entrada": features,
        "random_forest_hiperparametros_congelados_de_mayo": parametros,
        "metricas_junio": {
            "persistencia": metricas(real, persistencia),
            "random_forest": metricas(real, pred_rf),
            "hist_gradient_boosting": metricas(real, pred_hgb),
        },
        "criterio_mae_rf_menor_igual_5_33": bool(mae_rf <= 5.33),
        "mejora_rf_vs_persistencia_pct": round(float((1 - mae_rf / mae_persistencia) * 100), 2),
        "criterio_mejora_10_pct_vs_persistencia": bool(mae_rf <= 0.9 * mae_persistencia),
        "nota": "Una sola evaluación; misma muestra válida para los tres métodos; parámetros congelados en mayo.",
    }
    detalle = pd.DataFrame({
        "fecha y hora": test["fecha_hora"].to_numpy(),
        "fecha objetivo": test["fecha_objetivo"].to_numpy(),
        "target_24h": real.to_numpy(),
        "pred_persistencia": persistencia.to_numpy(),
        "pred_random_forest": pred_rf,
        "pred_hist_gradient_boosting": pred_hgb,
    })
    for col, pred in (("persistencia", "pred_persistencia"), ("rf", "pred_random_forest"), ("hist_gradient_boosting", "pred_hist_gradient_boosting")):
        detalle[f"error_abs_{col}"] = (detalle["target_24h"] - detalle[pred]).abs()
    OUT_JSON.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    detalle.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Entrenamiento Ene-Mayo: {len(train)}")
    print(f"Junio, muestra común: {len(test)} ({test.fecha_objetivo.min()} – {test.fecha_objetivo.max()})")
    for nombre, m in resultado["metricas_junio"].items():
        print(f"{nombre}: MAE={m['mae_ug_m3']:.4f} | RMSE={m['rmse_ug_m3']:.4f} | R²={m['r2']:.4f}")
    print(f"Cambio relativo RF vs persistencia: {resultado['mejora_rf_vs_persistencia_pct']:.2f}%")
    print(f"{OUT_JSON}\n{OUT_CSV}")


if __name__ == "__main__":
    main()
