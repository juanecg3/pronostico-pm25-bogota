"""Analiza de forma descriptiva los errores congelados de la prueba de junio."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "experiments" / "predicciones detalladas prueba.csv"
OUTPUT_JSON = PROJECT / "experiments" / "top10_errores_analisis.json"
OUTPUT_PLOT = PROJECT / "experiments" / "grafico_residuos_top10.png"

# Festivos nacionales de junio de 2026 según el calendario de MinHacienda.
FESTIVOS_JUNIO = {
    "2026-06-08": "Corpus Christi",
    "2026-06-15": "Sagrado Corazón de Jesús",
    "2026-06-29": "San Pedro y San Pablo",
}
FUENTE_FESTIVOS = (
    "https://sedeelectronica.minhacienda.gov.co/SedeElectronica/info/inicio.do"
    "?formAction=btDiasInhabiles&target=print"
)
DIA_SEMANA_ES = {
    "Monday": "lunes", "Tuesday": "martes", "Wednesday": "miércoles",
    "Thursday": "jueves", "Friday": "viernes", "Saturday": "sábado", "Sunday": "domingo",
}


def clasificar(fila):
    real = float(fila["target_24h"])
    residuo = float(fila["residuo_rf"])
    fecha_objetivo = fila["fecha objetivo"]
    fecha_texto = fecha_objetivo.strftime("%Y-%m-%d")
    festivo = FESTIVOS_JUNIO.get(fecha_texto)
    manana = 5 <= fecha_objetivo.hour <= 10
    fin_de_semana = fecha_objetivo.day_of_week >= 5

    if residuo < 0 and (fin_de_semana or (festivo and manana)):
        return (
            "Desajuste de calendario/tráfico (hipótesis)",
            "El modelo sobreestimó durante una mañana de fin de semana o festivo; un cambio del patrón de movilidad podría contribuir. "
            "El CSV no contiene tráfico para comprobarlo.",
        )
    if residuo < 0:
        return (
            "Sobreestimación / posible pérdida de inercia",
            "La predicción quedó por encima de la medición observada, compatible con una reducción rápida de PM2.5 respecto al patrón reciente. "
            "Dispersión por viento o lluvia es una hipótesis, no verificable con las variables disponibles.",
        )
    if real > 30:
        return (
            "Subestimación de pico atípico",
            "La medición objetivo superó 30 µg/m³ y el modelo predijo menos. Un episodio abrupto o acumulación local no anticipable con hora, calendario y rezagos puede explicar la brecha; no hay datos meteorológicos para identificar el mecanismo.",
        )
    return (
        "Subestimación de nivel elevado (sin superar 30)",
        "La concentración observada superó la predicción, aunque no pasó 30 µg/m³. El modelo pudo suavizar un aumento local o transitorio; se requieren variables meteorológicas y de actividad para probar la causa.",
    )


def main():
    datos = pd.read_csv(INPUT, parse_dates=["fecha y hora", "fecha objetivo"])
    requeridas = ["fecha y hora", "fecha objetivo", "target_24h", "pred_random_forest", "pred_persistencia"]
    if datos[requeridas].isna().any().any():
        raise ValueError("El archivo de predicciones contiene campos faltantes; no se descartaron filas.")

    # El residuo es positivo cuando RF subestima; se conserva el conjunto completo.
    datos["residuo_rf"] = datos["target_24h"] - datos["pred_random_forest"]
    datos["error_abs_rf"] = datos["residuo_rf"].abs()
    if len(datos) != 580:
        raise ValueError(f"Se esperaban los 580 casos congelados de junio; se encontraron {len(datos)}.")

    ordenados = datos.sort_values("error_abs_rf", ascending=False, kind="mergesort")
    top = ordenados.head(10).copy().reset_index(drop=True)
    casos = []
    for i, fila in top.iterrows():
        categoria, hipotesis = clasificar(fila)
        fecha_objetivo = fila["fecha objetivo"]
        fecha_objetivo_dia = fecha_objetivo.strftime("%Y-%m-%d")
        casos.append({
            "ranking": i + 1,
            "fecha_hora_origen": fila["fecha y hora"].strftime("%Y-%m-%d %H:%M"),
            "fecha_hora_objetivo": fecha_objetivo.strftime("%Y-%m-%d %H:%M"),
            "dia_semana_objetivo": DIA_SEMANA_ES[fecha_objetivo.day_name()],
            "es_festivo_nacional": fecha_objetivo_dia in FESTIVOS_JUNIO,
            "festivo": FESTIVOS_JUNIO.get(fecha_objetivo_dia),
            "pm25_real_ug_m3": round(float(fila["target_24h"]), 4),
            "pm25_predicho_rf_ug_m3": round(float(fila["pred_random_forest"]), 4),
            "pm25_persistencia_ug_m3": round(float(fila["pred_persistencia"]), 4),
            "residuo_real_menos_prediccion": round(float(fila["residuo_rf"]), 4),
            "error_absoluto_rf_ug_m3": round(float(fila["error_abs_rf"]), 4),
            "categoria": categoria,
            "hipotesis_causal_no_verificada": hipotesis,
        })

    residuos = datos["residuo_rf"]
    stats = {
        "muestras_evaluadas_sin_eliminar_atipicos": int(len(datos)),
        "sesgo_medio_residuo_real_menos_prediccion_ug_m3": round(float(residuos.mean()), 4),
        "mae_rf_ug_m3": round(float(datos["error_abs_rf"].mean()), 4),
        "rmse_rf_ug_m3": round(float((residuos.pow(2).mean()) ** 0.5), 4),
        "error_absoluto_mediano_ug_m3": round(float(datos["error_abs_rf"].median()), 4),
        "error_absoluto_p90_ug_m3": round(float(datos["error_abs_rf"].quantile(0.90)), 4),
        "error_absoluto_maximo_ug_m3": round(float(datos["error_abs_rf"].max()), 4),
        "desviacion_estandar_muestral_residuo_ug_m3": round(float(residuos.std()), 4),
        "residuos_positivos_subestimacion": int((residuos > 0).sum()),
        "residuos_negativos_sobreestimacion": int((residuos < 0).sum()),
        "residuos_cero": int((residuos == 0).sum()),
        "concentraciones_objetivo_mayores_a_30_ug_m3": int((datos["target_24h"] > 30).sum()),
    }
    conteos = {}
    for caso in casos:
        conteos[caso["categoria"]] = conteos.get(caso["categoria"], 0) + 1
    resultado = {
        "descripcion": "Análisis descriptivo de los 10 mayores errores absolutos de Random Forest en junio de 2026",
        "origen": "experiments/predicciones detalladas prueba.csv",
        "metodo": "Se recalculó real menos predicción; las predicciones congeladas no se modificaron ni se reentrenó el modelo.",
        "estadisticas_residuos_globales": stats,
        "conteos_categorias_en_top10": conteos,
        "fuente_calendario_festivos": FUENTE_FESTIVOS,
        "limite_interpretacion": "Las causas son hipótesis. El conjunto no contiene lluvia, viento, tráfico ni inventario de fuentes; las coincidencias temporales no demuestran causalidad.",
        "top_10_errores": casos,
    }
    OUTPUT_JSON.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")

    # Figura diagnóstica en dos paneles: serie temporal y distribución de residuos.
    fig, (ax_serie, ax_hist) = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    tiempo = datos.set_index("fecha objetivo").sort_index()
    horas = pd.date_range(tiempo.index.min(), tiempo.index.max(), freq="h")
    tiempo = tiempo.reindex(horas)  # Los pares no disponibles quedan como huecos, no como segmentos inventados.
    ax_serie.plot(tiempo.index, tiempo["target_24h"], color="#252525", lw=1.2, label="PM2.5 real")
    ax_serie.plot(tiempo.index, tiempo["pred_random_forest"], color="#276FBF", lw=1.1, label="Random Forest")
    ax_serie.scatter(top["fecha objetivo"], top["target_24h"], color="#D62728", marker="o", s=42,
                     edgecolor="white", linewidth=0.5, zorder=5, label="Top 10 errores absolutos")
    ax_serie.set_title("Junio de 2026: real vs. Random Forest y 10 mayores errores")
    ax_serie.set_ylabel("PM2.5 (µg/m³)")
    ax_serie.set_xlim(pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-01"))
    ax_serie.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax_serie.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_serie.legend(ncols=3, frameon=False)
    ax_serie.grid(axis="y", alpha=0.25)
    ax_hist.hist(residuos, bins=24, color="#6C91BF", edgecolor="white")
    ax_hist.axvline(0, color="#252525", lw=1, label="Residuo cero")
    ax_hist.axvline(residuos.mean(), color="#D62728", ls="--", lw=1.4,
                    label=f"Media = {residuos.mean():.2f} µg/m³")
    ax_hist.set_title("Distribución de residuos (real − predicción)")
    ax_hist.set_xlabel("Residuo (µg/m³); positivo = subestimación")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.legend(frameon=False)
    ax_hist.grid(axis="y", alpha=0.25)
    fig.savefig(OUTPUT_PLOT, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Muestras conservadas: {len(datos)}")
    print(f"MAE recalculado: {stats['mae_rf_ug_m3']:.4f} µg/m³")
    print(f"Sesgo medio: {stats['sesgo_medio_residuo_real_menos_prediccion_ug_m3']:.4f} µg/m³")
    print(top[["fecha y hora", "fecha objetivo", "target_24h", "pred_random_forest", "error_abs_rf"]].to_string(index=False))
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Gráfico: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
