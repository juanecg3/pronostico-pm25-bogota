"""Calcula MAE y RMSE de la regla de persistencia sobre junio."""

# Importamos las librerías necesarias para manejar rutas y datos tabulares.
from pathlib import Path
import json

import pandas as pd


# Definimos las rutas del proyecto para localizar la base de datos y el archivo
# donde se guardarán los resultados de la línea base.
CARPETA_PROYECTO = Path(__file__).resolve().parents[1]
ARCHIVO_ENTRADA = CARPETA_PROYECTO / "data" / "particion_temporal.csv"
ARCHIVO_RESULTADOS = CARPETA_PROYECTO / "experiments" / "resultados_baseline.json"


def main():
    # Cargamos el dataset y convertimos la columna de fecha a tipo datetime.
    # Esto facilita filtrar por partición y usar la fecha como información temporal.
    datos = pd.read_csv(ARCHIVO_ENTRADA, parse_dates=["fecha_objetivo"])

    # Nos quedamos solo con la partición de prueba, que es la que se usa para
    # evaluar el modelo en este script.
    prueba = datos[datos["particion"] == "test"].copy()

    # Regla de persistencia: la predicción para una hora futura es igual a la
    # medición observada en la misma hora 24 horas antes.
    # Para hacer una comparación justa, conservamos solo filas en las que
    # existen valores reales tanto en la observación original como en el target.
    prueba = prueba.dropna(subset=["pm25_original", "target_24h"])
    if prueba.empty:
        raise ValueError("No hay pares observados en la partición de prueba.")

    # Calculamos el error de cada predicción:
    # error = valor real (target_24h) - valor predicho por persistencia (pm25_original)
    errores = prueba["target_24h"] - prueba["pm25_original"]

    # Metricas de evaluación: error medio absoluto (MAE) y raíz del error cuadrático medio (RMSE).
    mae = errores.abs().mean()
    rmse = (errores.pow(2).mean()) ** 0.5

    # Construimos un diccionario con los resultados para guardar un resumen en JSON.
    resultados = {
        "modelo": "Persistencia: predicción igual a la medición 24 horas antes",
        "estacion": "Kennedy",
        "particion": "test (junio de 2026)",
        "muestras_evaluadas": int(len(prueba)),
        "fecha_objetivo_inicio": str(prueba["fecha_objetivo"].min()),
        "fecha_objetivo_fin": str(prueba["fecha_objetivo"].max()),
        "mae_ug_m3": round(float(mae), 2),
        "rmse_ug_m3": round(float(rmse), 2),
        "semilla_configurada": 42,
        "nota_semilla": "La persistencia es determinista; no utiliza números aleatorios.",
    }

    # Guardamos los resultados en un archivo JSON para que puedan ser comparados
    # con otros modelos o consultados luego sin ejecutar el script de nuevo.
    ARCHIVO_RESULTADOS.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Imprimimos una salida legible en consola para revisar rápidamente el desempeño.
    print("--- LÍNEA BASE DE PERSISTENCIA (TEST) ---")
    print(f"Muestras evaluadas: {len(prueba)}")
    print(f"MAE: {mae:.2f} µg/m³")
    print(f"RMSE: {rmse:.2f} µg/m³")
    print(f"Resultados guardados en: {ARCHIVO_RESULTADOS}")


if __name__ == "__main__":
    # Punto de entrada del script: se ejecuta únicamente cuando se corre directamente.
    main()
