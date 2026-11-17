"""Crea el objetivo a 24 horas y separa los datos por fecha objetivo."""

from pathlib import Path
import json

import pandas as pd


# Define las carpetas y los archivos que usa el proceso dentro del proyecto.
CARPETA_PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_DATOS = CARPETA_PROYECTO / "data"
ARCHIVO_ENTRADA = CARPETA_DATOS / "datos limpios.csv"
ARCHIVO_SALIDA = CARPETA_DATOS / "particion_temporal.csv"
ARCHIVO_RESUMEN = CARPETA_PROYECTO / "experiments" / "resumen_particion.json"


def main():
    # Carga el archivo limpio, convierte la columna de fecha y ordena las filas
    # cronológicamente para que el desplazamiento del objetivo sea correcto.
    datos = pd.read_csv(ARCHIVO_ENTRADA, parse_dates=["fecha_hora"])
    datos = datos.sort_values("fecha_hora").reset_index(drop=True)

    # El objetivo es el valor original de PM2.5 registrado 24 horas después.
    # Usar el dato original evita evaluar el modelo con valores imputados.
    datos["target_24h"] = datos["pm25_original"].shift(-24)
    # Registra explícitamente la fecha que corresponde al objetivo pronosticado.
    datos["fecha_objetivo"] = datos["fecha_hora"] + pd.Timedelta(hours=24)

    # La partición depende de la fecha pronosticada, no de la fecha de entrada.
    mes_objetivo = datos["fecha_objetivo"].dt.month
    anio_objetivo = datos["fecha_objetivo"].dt.year
    datos["particion"] = None
    # Usa enero-abril para entrenar, mayo para validar y junio para probar.
    datos.loc[(anio_objetivo == 2026) & mes_objetivo.between(1, 4), "particion"] = "train"
    datos.loc[(anio_objetivo == 2026) & (mes_objetivo == 5), "particion"] = "validation"
    datos.loc[(anio_objetivo == 2026) & (mes_objetivo == 6), "particion"] = "test"
    # Elimina las filas cuyas fechas objetivo quedan fuera de esos periodos.
    datos = datos[datos["particion"].notna()].copy()

    # Guarda los datos seleccionados, incluyendo objetivo y partición.
    datos.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8")

    # Resume el rango temporal, el total de filas y los objetivos disponibles
    # (es decir, aquellos que no están vacíos) para cada partición.
    resumen = {}
    for nombre in ["train", "validation", "test"]:
        parte = datos[datos["particion"] == nombre]
        resumen[nombre] = {
            "inicio_objetivo": str(parte["fecha_objetivo"].min()),
            "fin_objetivo": str(parte["fecha_objetivo"].max()),
            "filas": int(len(parte)),
            "objetivos_observados": int(parte["target_24h"].notna().sum()),
        }

    # Escribe el resumen en formato JSON con indentación para facilitar su lectura.
    ARCHIVO_RESUMEN.write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Informa en consola los resultados y la ubicación del archivo generado.
    print("--- PARTICIÓN POR FECHA OBJETIVO ---")
    for nombre, valores in resumen.items():
        print(
            f"{nombre}: {valores['inicio_objetivo']} a {valores['fin_objetivo']} "
            f"({valores['filas']} filas; {valores['objetivos_observados']} objetivos observados)"
        )
    print(f"Archivo generado: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    main()
