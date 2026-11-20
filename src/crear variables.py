"""Prepara las variables de entrada y el objetivo para el pronóstico."""

# Path ayuda a construir rutas de archivos compatibles con distintos sistemas.
from pathlib import Path
# json permite guardar un resumen de los resultados procesados.
import json

# pandas se utiliza para leer y transformar los datos tabulares.
import pandas as pd


# Ubica la carpeta principal del proyecto a partir de la ubicación de este script.
CARPETA_PROYECTO = Path(__file__).resolve().parents[1]
# Archivo de origen con las mediciones y su partición temporal.
ARCHIVO_ENTRADA = CARPETA_PROYECTO / "data" / "particion_temporal.csv"
# Archivo de salida con las variables preparadas para el modelo.
ARCHIVO_SALIDA = CARPETA_PROYECTO / "data" / "datos_modelo.csv"
# Archivo JSON para registrar cuántas filas válidas hay en cada partición.
ARCHIVO_RESUMEN = CARPETA_PROYECTO / "experiments" / "resumen_variables.json"


def main():
    # Carga los datos y convierte estas columnas a fechas para poder extraer sus componentes.
    datos = pd.read_csv(ARCHIVO_ENTRADA, parse_dates=["fecha_hora", "fecha_objetivo"])
    # Ordena las filas por tiempo; así, los rezagos corresponden a mediciones anteriores.
    datos = datos.sort_values("fecha_hora").reset_index(drop=True)

    # Crea variables conocidas en el momento en que se inicia el pronóstico.
    # La hora y el día de la semana pueden representar patrones periódicos.
    datos["hora"] = datos["fecha_hora"].dt.hour
    datos["dia_semana"] = datos["fecha_hora"].dt.day_of_week
    # Usa mediciones observadas, no valores interpolados que podrían depender de datos futuros.
    datos["pm25_t"] = datos["pm25_original"]
    # Guarda como variable la medición de la fila anterior.
    datos["pm25_lag1"] = datos["pm25_original"].shift(1)
    # Guarda la medición de 24 filas atrás (24 horas si cada fila representa una hora).
    datos["pm25_lag24"] = datos["pm25_original"].shift(24)

    # Selecciona únicamente las columnas necesarias para el análisis y el modelo.
    columnas_utiles = [
        "fecha_hora",
        "fecha_objetivo",
        "particion",
        "pm25_original",
        "pm25_limpio",
        "imputado",
        "hora",
        "dia_semana",
        "pm25_t",
        "pm25_lag1",
        "pm25_lag24",
        "target_24h",
    ]
    datos = datos[columnas_utiles]

    # Define las variables y el valor real que el modelo deberá predecir.
    variables = ["pm25_t", "pm25_lag1", "pm25_lag24", "target_24h"]
    # Descarta filas incompletas en esas columnas, por ejemplo, las primeras filas sin rezagos.
    datos = datos.dropna(subset=variables).reset_index(drop=True)
    # Guarda los datos preparados como CSV, sin añadir el índice de pandas.
    datos.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8")

    # Cuenta las filas válidas de cada grupo de datos (partición).
    resumen = datos.groupby("particion").size().to_dict()
    # Escribe el resumen en JSON con formato legible.
    ARCHIVO_RESUMEN.write_text(
        json.dumps({"filas_validas_por_particion": resumen}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Informa en consola sobre las variables, los conteos y el archivo generado.
    print("--- VARIABLES Y OBJETIVO CREADOS ---")
    print("Objetivo: target_24h (medición real 24 horas después)")
    print("Variables: pm25_t, hora, dia_semana, pm25_lag1 y pm25_lag24")
    print(f"Filas válidas: {len(datos)}")
    print(f"Filas por partición: {resumen}")
    print(f"Archivo generado: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    main()
