"""Ajusta y compara modelos para pronosticar PM2.5 a 24 horas.

La búsqueda usa enero-abril para entrenar y mayo como única partición de
selección. Este script no carga ni evalúa el conjunto de prueba de junio.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "datos_modelo.csv"
CONFIG_PATH = PROJECT_DIR / "experiments" / "config.json"
RESULTS_PATH = PROJECT_DIR / "experiments" / "resultados ajuste de hiperparámetros.json"
FEATURES = ["pm25_t", "pm25_lag1", "pm25_lag24", "hora", "dia_semana"]
TARGET = "target_24h"


def metricas(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float | int]:
    """Calcula las métricas solo para objetivos observados en datos reales."""
    return {
        "muestras_evaluadas": int(len(y_true)),
        "mae_ug_m3": round(float(mean_absolute_error(y_true, predictions)), 4),
        "rmse_ug_m3": round(float(root_mean_squared_error(y_true, predictions)), 4),
        "r2": round(float(r2_score(y_true, predictions)), 4),
    }


def buscar_modelo(
    nombre: str,
    pipeline: Pipeline,
    param_grid: dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_jobs: int,
) -> tuple[dict, dict]:
    """Elige hiperparámetros con MAE de mayo y evalúa en el mismo bloque."""
    X_search = pd.concat([X_train, X_val], ignore_index=True)
    y_search = pd.concat([y_train, y_val], ignore_index=True)
    # -1 conserva esas filas en el ajuste; 0 las reserva para la puntuación.
    fold = np.r_[np.full(len(X_train), -1), np.zeros(len(X_val), dtype=int)]
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=PredefinedSplit(test_fold=fold),
        refit=False,
        n_jobs=n_jobs,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X_search, y_search)

    # Se reajusta solo con train, dejando la validación fuera del estimador.
    model = pipeline.set_params(**search.best_params_)
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    return (
        {
            "nombre": nombre,
            "mejores_hiperparametros": search.best_params_,
            "mae_cv_validacion_ug_m3": round(float(-search.best_score_), 4),
            "metricas_validacion": metricas(y_val, prediction),
        },
        {"pipeline": model, "predicciones": prediction},
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. Ejecuta primero los scripts de partición y variables."
        )

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))
    data = pd.read_csv(DATA_PATH, parse_dates=["fecha_objetivo"])
    required = set(FEATURES + [TARGET, "particion", "pm25_original"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {DATA_PATH.name}: {sorted(missing)}")

    train = data.loc[data["particion"].eq("train")].dropna(subset=[TARGET]).copy()
    validation = data.loc[data["particion"].eq("validation")].dropna(subset=[TARGET]).copy()
    if train.empty or validation.empty:
        raise ValueError("Se necesitan objetivos observados en train y validation.")
    if not train["fecha_objetivo"].max() < validation["fecha_objetivo"].min():
        raise ValueError("Las particiones train y validation no están ordenadas cronológicamente.")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = validation[FEATURES], validation[TARGET]
    n_jobs = -1

    rf_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(random_state=seed, n_jobs=1)),
        ]
    )
    rf_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", 1.0],
    }

    hgb_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(random_state=seed)),
        ]
    )
    hgb_grid = {
        "model__learning_rate": [0.05, 0.1],
        "model__max_iter": [100, 200],
        "model__max_leaf_nodes": [15, 31],
    }

    print("Ajustando Random Forest; train: enero-abril, validación: mayo...")
    rf_result, _ = buscar_modelo(
        "RandomForestRegressor", rf_pipeline, rf_grid,
        X_train, y_train, X_val, y_val, n_jobs,
    )
    print("Ajustando HistGradientBoosting; train: enero-abril, validación: mayo...")
    hgb_result, _ = buscar_modelo(
        "HistGradientBoostingRegressor", hgb_pipeline, hgb_grid,
        X_train, y_train, X_val, y_val, n_jobs,
    )

    baseline_rows = validation.dropna(subset=["pm25_original", TARGET])
    if baseline_rows.empty:
        raise ValueError("No hay pares observados para evaluar persistencia en validation.")
    baseline_result = {
        "nombre": "Persistencia",
        "descripcion": "La predicción a 24 horas es la medición observada en el origen.",
        "metricas_validacion": metricas(
            baseline_rows[TARGET], baseline_rows["pm25_original"].to_numpy()
        ),
    }

    candidates = [rf_result, hgb_result, baseline_result]
    best_ml_model = min(
        [rf_result, hgb_result],
        key=lambda result: result["metricas_validacion"]["mae_ug_m3"],
    )
    champion = min(candidates, key=lambda result: result["metricas_validacion"]["mae_ug_m3"])
    results = {
        "fase": "CRISP-ML(Q) Fase 3 - Model Engineering",
        "estacion": "Kennedy",
        "objetivo": "target_24h",
        "seed": seed,
        "seleccion": "Menor MAE en la partición validation (mayo de 2026).",
        "particiones": {
            "train": {
                "periodo_objetivo": [str(train["fecha_objetivo"].min()), str(train["fecha_objetivo"].max())],
                "muestras": int(len(train)),
            },
            "validation": {
                "periodo_objetivo": [str(validation["fecha_objetivo"].min()), str(validation["fecha_objetivo"].max())],
                "muestras": int(len(validation)),
            },
            "test": "No cargado ni evaluado durante el ajuste de hiperparámetros.",
        },
        "busqueda": {
            "metodo": "GridSearchCV con PredefinedSplit: train ajusta y validation puntúa.",
            "scoring": "neg_mean_absolute_error",
            "preprocesamiento": "SimpleImputer(strategy='median') dentro del Pipeline.",
            "n_jobs": n_jobs,
        },
        "modelos": [rf_result, hgb_result, baseline_result],
        "mejor_modelo_ml_validation": {
            "nombre": best_ml_model["nombre"],
            "mejores_hiperparametros": best_ml_model["mejores_hiperparametros"],
            "metricas_validacion": best_ml_model["metricas_validacion"],
        },
        "campeon_validation": {
            "nombre": champion["nombre"],
            "mae_ug_m3": champion["metricas_validacion"]["mae_ug_m3"],
            "criterio": "Menor MAE en mayo entre Random Forest, HistGradientBoosting y persistencia.",
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n--- RESULTADOS DE VALIDACIÓN (MAYO) ---")
    for result in candidates:
        metrics = result["metricas_validacion"]
        print(
            f"{result['nombre']}: MAE={metrics['mae_ug_m3']:.4f} µg/m³ | "
            f"RMSE={metrics['rmse_ug_m3']:.4f} µg/m³ | R²={metrics['r2']:.4f}"
        )
    print(f"Campeón de validación: {champion['nombre']}")
    print(f"Resultados guardados en: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
