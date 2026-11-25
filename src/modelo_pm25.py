"""Preparación y modelo congelado compartidos por XAI y la aplicación."""

from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


PROJECT = Path(__file__).resolve().parents[1]
FEATURES = [
    "pm25_original", "pm25_lag1", "pm25_lag24", "pm25_roll3",
    "pm25_roll6", "hora", "dia_semana",
]
SEED = 42


def cargar_particiones():
    datos = pd.read_csv(PROJECT / "data" / "datos limpios.csv", parse_dates=["fecha_hora"])
    datos = datos.sort_values("fecha_hora").reset_index(drop=True)
    observadas = datos.set_index("fecha_hora")["pm25_original"].dropna()
    for ventana in (3, 6):
        medias = observadas.rolling(ventana, min_periods=ventana).mean()
        datos[f"pm25_roll{ventana}"] = medias.reindex(
            pd.DatetimeIndex(datos["fecha_hora"])
        ).ffill().to_numpy()

    datos["fecha_objetivo"] = datos["fecha_hora"] + pd.Timedelta(hours=24)
    datos["target_24h"] = datos["pm25_original"].shift(-24)
    datos["hora"] = datos["fecha_hora"].dt.hour
    datos["dia_semana"] = datos["fecha_hora"].dt.day_of_week
    datos["pm25_lag1"] = datos["pm25_original"].shift(1)
    datos["pm25_lag24"] = datos["pm25_original"].shift(24)
    modelo = datos.dropna(subset=FEATURES + ["target_24h"]).copy()
    train = modelo[modelo["fecha_objetivo"].between("2026-01-01", "2026-05-31 23:59:59")].copy()
    test = modelo[modelo["fecha_objetivo"].between("2026-06-01", "2026-06-30 23:59:59")].copy()
    if train.empty or test.empty:
        raise ValueError("Las particiones de enero-mayo o junio están vacías.")
    if train["fecha_objetivo"].max() >= test["fecha_objetivo"].min():
        raise ValueError("Hay solapamiento entre las fechas objetivo de entrenamiento y prueba.")
    return datos, train, test


def cargar_parametros_campeon():
    ruta = PROJECT / "experiments" / "resultados ajuste modelos.json"
    resultados = json.loads(ruta.read_text(encoding="utf-8"))
    return resultados["resultados_validacion_mayo"]["random_forest"]["mejores_hiperparametros"]


def entrenar_campeon(train, parametros=None):
    parametros = parametros or cargar_parametros_campeon()
    modelo = RandomForestRegressor(random_state=SEED, n_jobs=-1, **parametros)
    modelo.fit(train[FEATURES], train["target_24h"])
    return modelo
