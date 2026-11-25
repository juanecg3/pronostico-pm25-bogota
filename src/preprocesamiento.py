"""Carga, limpia y resume la serie horaria de PM2.5 de la estación Kennedy."""

from io import StringIO
from pathlib import Path
import json
import sys

import pandas as pd


# Define las rutas del proyecto, el archivo de entrada y los archivos de salida.
CARPETA_PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_DATOS = CARPETA_PROYECTO / "data"
ARCHIVO_ENTRADA = Path(sys.argv[1]) if len(sys.argv) > 1 else CARPETA_DATOS / "datos.xls"
CARPETA_SALIDA = CARPETA_DATOS
ARCHIVO_SALIDA = CARPETA_SALIDA / "datos limpios.csv"
RESUMEN_SALIDA = CARPETA_SALIDA / "resumen_preprocesamiento.json"


def cargar_datos(archivo):
    """Carga la exportación descargada, ya sea una tabla HTML o un archivo Excel."""
    # Inspecciona el inicio del archivo para distinguir HTML de un libro de Excel.
    with open(archivo, "rb") as entrada:
        inicio = entrada.read(1000).lstrip().lower()

    if inicio.startswith((b"<html", b"<!doctype html", b"<table")):
        # Lee las tablas HTML y selecciona la primera que tenga una columna de fecha.
        contenido = Path(archivo).read_text(encoding="utf-8", errors="replace")
        tablas = pd.read_html(StringIO(contenido))
        datos = None

        for tabla in tablas:
            columnas = [str(col[-1] if isinstance(col, tuple) else col).strip() for col in tabla.columns]
            if any("fecha" in columna.casefold() for columna in columnas):
                tabla.columns = columnas
                datos = tabla
                break

        if datos is None:
            raise ValueError("No se encontró una tabla con la columna de fecha y hora.")
    else:
        # Si no es HTML, intenta abrir el archivo como una hoja de cálculo.
        datos = pd.read_excel(archivo)
        datos.columns = [str(col).strip() for col in datos.columns]

    # Identifica las columnas relevantes y normaliza sus nombres para el resto del proceso.
    columna_fecha = next((col for col in datos.columns if "fecha" in col.casefold()), None)
    columna_pm25 = next((col for col in datos.columns if "concentr" in col.casefold()), None)
    if columna_fecha is None or columna_pm25 is None:
        raise ValueError(
            "No se encontraron las columnas de fecha y concentración. "
            f"Columnas detectadas: {list(datos.columns)}"
        )

    return datos[[columna_fecha, columna_pm25]].rename(
        columns={columna_fecha: "fecha_hora", columna_pm25: "pm25_original"}
    )


def main():
    # Verifica que exista la entrada antes de iniciar el preprocesamiento.
    if not ARCHIVO_ENTRADA.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {ARCHIVO_ENTRADA}")

    # Carga los datos, convierte sus tipos y elimina fechas inválidas.
    datos = cargar_datos(ARCHIVO_ENTRADA)
    datos["fecha_hora"] = pd.to_datetime(datos["fecha_hora"], errors="coerce")
    datos["pm25_original"] = pd.to_numeric(datos["pm25_original"], errors="coerce")
    datos = datos.dropna(subset=["fecha_hora"])
    # Ordena cronológicamente, conserva el último registro duplicado y crea la frecuencia horaria.
    datos = datos.sort_values("fecha_hora").drop_duplicates("fecha_hora", keep="last")
    datos = datos.set_index("fecha_hora").asfreq("h")

    # Cuenta los faltantes y detecta grupos consecutivos para limitar la imputación.
    serie = datos["pm25_original"]
    faltantes_iniciales = int(serie.isna().sum())

    # Solo se interpolan huecos internos de una o dos horas.
    # Los bloques más largos y los extremos quedan como NaN, sin usar bfill/ffill.
    faltantes = serie.isna()
    grupo_hueco = faltantes.ne(faltantes.shift()).cumsum()
    longitud_hueco = faltantes.groupby(grupo_hueco).transform("sum")
    huecos_cortos = faltantes & longitud_hueco.between(1, 2)
    interpolada = serie.interpolate(method="time", limit_area="inside")

    # Mantiene los valores originales y sustituye solo los faltantes de huecos cortos.
    datos["pm25_limpio"] = serie
    datos.loc[huecos_cortos, "pm25_limpio"] = interpolada.loc[huecos_cortos]
    # Marca las filas cuyo valor se completó mediante interpolación.
    datos["imputado"] = huecos_cortos & datos["pm25_limpio"].notna()

    # Calcula ventanas sobre mediciones observadas (pm25_original), excluyendo
    # horas sin lectura. El ffill solo propaga la última ventana ya calculada;
    # nunca incorpora valores de una hora posterior.
    observadas = datos["pm25_original"].dropna()
    for ventana in (3, 6):
        medias = observadas.rolling(window=ventana, min_periods=ventana).mean()
        datos[f"pm25_roll{ventana}"] = medias.reindex(datos.index).ffill()

    # Guarda la serie procesada en CSV con las columnas necesarias.
    CARPETA_SALIDA.mkdir(exist_ok=True)
    resultado = datos.reset_index()[[
        "fecha_hora", "pm25_original", "pm25_limpio", "imputado",
        "pm25_roll3", "pm25_roll6",
    ]]
    resultado.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8")

    # Calcula estadísticas y construye el resumen del proceso en formato JSON.
    valores_validos = datos["pm25_limpio"].dropna()
    estadisticas = valores_validos.describe().round(2).to_dict()
    resumen = {
        "archivo_entrada": str(ARCHIVO_ENTRADA.name),
        "estacion": "Kennedy",
        "periodo_inicio": str(datos.index.min()),
        "periodo_fin": str(datos.index.max()),
        "filas_horarias": int(len(datos)),
        "faltantes_iniciales": faltantes_iniciales,
        "valores_imputados_en_huecos_de_hasta_2_horas": int(datos["imputado"].sum()),
        "faltantes_restantes_en_bloques_largos_o_extremos": int(datos["pm25_limpio"].isna().sum()),
        "estadisticas_pm25_ug_m3": estadisticas,
        "archivo_salida": "data/datos limpios.csv",
    }
    RESUMEN_SALIDA.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    # Muestra en consola los indicadores principales y las rutas de los resultados.
    print("--- PREPROCESAMIENTO PM2.5 ---")
    print(f"Archivo leído: {ARCHIVO_ENTRADA.name}")
    print(f"Estación: Kennedy")
    print(f"Periodo horario: {datos.index.min()} a {datos.index.max()}")
    print(f"Filas horarias: {len(datos)}")
    print(f"Faltantes iniciales: {faltantes_iniciales}")
    print(f"Valores imputados (huecos de hasta 2 horas): {int(datos['imputado'].sum())}")
    print(f"Faltantes conservados (bloques largos/extremos): {int(datos['pm25_limpio'].isna().sum())}")
    print("\nEstadísticas de PM2.5 limpio (µg/m³):")
    print(valores_validos.describe().round(2).to_string())
    print(f"\nCSV generado: {ARCHIVO_SALIDA}")
    print(f"Resumen generado: {RESUMEN_SALIDA}")


# Ejecuta el flujo principal solo cuando el archivo se ejecuta directamente.
if __name__ == "__main__":
    main()
