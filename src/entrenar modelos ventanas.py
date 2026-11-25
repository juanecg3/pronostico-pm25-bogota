"""Busca hiperparámetros para pronóstico PM2.5 a 24 h; valida en mayo."""

from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "datos limpios.csv"
OUTPUT = PROJECT / "experiments" / "resultados ajuste modelos.json"
SEED = 42


def main():
    # El experimento no usa ni evalúa registros posteriores a mayo.
    datos = pd.read_csv(DATA, parse_dates=["fecha_hora"])
    datos = datos.loc[datos["fecha_hora"] < "2026-06-01"].copy()
    datos = datos.sort_values("fecha_hora").reset_index(drop=True)

    # Ventanas retrospectivas de las últimas N lecturas reales disponibles.
    # Se conserva el último promedio durante horas sin lectura; no hay look-ahead.
    observadas = datos.set_index("fecha_hora")["pm25_original"].dropna()
    for window in (3, 6):
        rolling = observadas.rolling(window=window, min_periods=window).mean()
        datos[f"pm25_roll{window}"] = rolling.reindex(
            pd.DatetimeIndex(datos["fecha_hora"])
        ).ffill().to_numpy()

    datos["target_24h"] = datos["pm25_original"].shift(-24)
    datos["fecha_objetivo"] = datos["fecha_hora"] + pd.Timedelta(hours=24)
    datos["hora"] = datos["fecha_hora"].dt.hour
    datos["dia_semana"] = datos["fecha_hora"].dt.dayofweek
    datos["pm25_lag1"] = datos["pm25_original"].shift(1)
    datos["pm25_lag24"] = datos["pm25_original"].shift(24)
    features = [
        "pm25_original", "pm25_lag1", "pm25_lag24", "pm25_roll3",
        "pm25_roll6", "hora", "dia_semana",
    ]

    # Las particiones se definen por la fecha del objetivo pronosticado.
    valid = datos.dropna(subset=features + ["target_24h"])
    train = valid[valid["fecha_objetivo"].between("2026-01-01", "2026-04-30 23:59:59")]
    may = valid[valid["fecha_objetivo"].between("2026-05-01", "2026-05-31 23:59:59")]
    if train.empty or may.empty:
        raise ValueError("No hay suficientes filas completas para train o validación de mayo.")

    X_train, y_train = train[features], train["target_24h"]
    X_may, y_may = may[features], may["target_24h"]
    params = {
        "n_estimators": [200, 400],
        "max_depth": [10, 15, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 1.0],
    }
    search = GridSearchCV(
        RandomForestRegressor(random_state=SEED, n_jobs=-1), params,
        scoring="neg_mean_absolute_error", cv=TimeSeriesSplit(n_splits=3), n_jobs=-1,
    )
    search.fit(X_train, y_train)
    pred_rf = search.best_estimator_.predict(X_may)
    hgb = HistGradientBoostingRegressor(random_state=SEED, max_iter=200, learning_rate=0.05)
    hgb.fit(X_train, y_train)
    pred_hgb = hgb.predict(X_may)
    pred_persistence = may["pm25_original"]

    def metrics(y, pred):
        return {
            "mae": round(float(mean_absolute_error(y, pred)), 4),
            "rmse": round(float(mean_squared_error(y, pred) ** 0.5), 4),
            "r2": round(float(r2_score(y, pred)), 4),
        }

    results = {
        "experimento": "Ventanas móviles de 3 y 6 lecturas observadas; train enero-abril, validación mayo",
        "semilla": SEED,
        "variables_entrada": features,
        "periodos": {
            "entrenamiento_objetivo": [str(train.fecha_objetivo.min()), str(train.fecha_objetivo.max())],
            "validacion_objetivo": [str(may.fecha_objetivo.min()), str(may.fecha_objetivo.max())],
        },
        "muestras": {"train": len(train), "validation_may": len(may)},
        "resultados_validacion_mayo": {
            "persistencia": metrics(y_may, pred_persistence),
            "random_forest": {"mejores_hiperparametros": search.best_params_, **metrics(y_may, pred_rf)},
            "hist_gradient_boosting": metrics(y_may, pred_hgb),
        },
        "random_forest_mae_bajo_5_26": bool(mean_absolute_error(y_may, pred_rf) < 5.26),
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Train Ene-Abr: {len(train)} | Validación Mayo: {len(may)}")
    print(f"Random Forest MAE Mayo: {results['resultados_validacion_mayo']['random_forest']['mae']:.4f} µg/m³")
    print(f"¿Baja de 5.26?: {'Sí' if results['random_forest_mae_bajo_5_26'] else 'No'}")
    print(f"Mejores hiperparámetros: {search.best_params_}")
    print(f"Resultados: {OUTPUT}")


if __name__ == "__main__":
    main()
