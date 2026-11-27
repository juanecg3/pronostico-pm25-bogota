# Pronóstico de PM2.5 en Bogotá

Proyecto académico de la asignatura Inteligencia Artificial (Universidad Tecnológica de Bolívar, ruta C: aprendizaje automático clásico). Pronostica la concentración horaria de PM2.5 con 24 horas de anticipación en la estación Kennedy de la Red de Monitoreo de Calidad del Aire de Bogotá (RMCAB) y compara un Random Forest y un HistGradientBoosting con la regla de persistencia.

**Integrantes:** Juan Eduardo Caicedo Gómez (T00068444) y Cesar David Pinzon Arrieta (T00065994).

> Ejercicio académico: no emite alertas ni sustituye el pronóstico oficial ni el IBOCA de la Secretaría Distrital de Ambiente (SDA).

## Instalación y ejecución (4 comandos)

Requisito: Python 3.11 o posterior. Desde la carpeta principal del repositorio:

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\main.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**Linux / macOS**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/main.py
.venv/bin/python -m streamlit run app.py
```

El tercer comando ejecuta todo el flujo (unos 75 s) y regenera los archivos de `data/` y `experiments/`. Con `src/main.py --rapido` se omiten las dos búsquedas de hiperparámetros y se usan los ya guardados. El cuarto comando abre el prototipo web.

## Flujo (`src/main.py`)

| Paso | Script | Salida principal |
| --- | --- | --- |
| 1. Preprocesamiento | `src/preprocesamiento.py` | `data/datos limpios.csv`, `data/resumen_preprocesamiento.json` |
| 2. Partición temporal | `src/particion temporal.py` | `data/particion_temporal.csv`, `experiments/resumen_particion.json` |
| 3. Variables | `src/crear variables.py` | `data/datos_modelo.csv`, `experiments/resumen_variables.json` |
| 4. Línea base (Hito 2) | `src/linea base de persistencia.py` | `experiments/resultados_baseline.json` |
| 5. Ajuste sin ventanas | `src/entrenar modelos.py` | `experiments/resultados ajuste de hiperparámetros.json` |
| 6. Ajuste con ventanas (campeón) | `src/entrenar modelos ventanas.py` | `experiments/resultados ajuste modelos.json` |
| 7. Evaluación final en junio | `src/evaluar prueba.py` | `experiments/evaluacion final prueba.json`, `experiments/predicciones detalladas prueba.csv` |
| 8. 10 mayores errores | `src/analisis_errores.py` | `experiments/top10_errores_analisis.json`, `experiments/grafico_residuos_top10.png` |
| 9. Explicabilidad | `src/explicabilidad.py` | `experiments/resultados explicabilidad.json`, `experiments/importancia variables.png` |

Todos los modelos aleatorios usan la semilla 42 (`experiments/config.json`). Ejecutar el flujo dos veces produce las mismas métricas.

## Datos y preprocesamiento

- Fuente: series horarias de PM2.5 de la estación Kennedy, descargadas del portal de la RMCAB (SDA). Son mediciones ambientales públicas, sin datos personales; se usan con atribución a la SDA (licencia CC BY 4.0 del catálogo de Datos Abiertos Bogotá).
- `data/datos.xls` es una tabla HTML con extensión `.xls`; el programa detecta ese formato y también lee archivos Excel normales.
- Periodo: 1 de enero a 1 de julio de 2026, 4.345 horas. Había 347 faltantes ("Sin data"): se interpolaron solo 18 huecos internos de una o dos horas y los 329 restantes se conservaron vacíos.
- El objetivo `target_24h` y todas las variables (`pm25_original`, `pm25_lag1`, `pm25_lag24`, `pm25_roll3`, `pm25_roll6`, `hora`, `dia_semana`) se calculan solo con mediciones originales, para que ninguna dependa de interpolaciones que usan lecturas posteriores.
- La partición se hace por fecha objetivo: enero-abril para entrenamiento, mayo para validación y junio para prueba.

## Resultados

Evaluación final de una sola pasada sobre 580 pares válidos comunes de junio, con los hiperparámetros congelados en mayo (Random Forest: 400 árboles, `max_depth=10`, `max_features="sqrt"`, `min_samples_split=5`, `min_samples_leaf=1`) y reentrenamiento con enero-mayo (2.950 muestras):

| Método | MAE (µg/m³) | RMSE (µg/m³) | R² |
| --- | --- | --- | --- |
| Persistencia | 6.1500 | 8.5797 | -0.1517 |
| HistGradientBoosting | 6.1137 | 7.6862 | 0.0757 |
| Random Forest | 6.0465 | 7.3634 | 0.1517 |

El Random Forest mejoró el MAE en 1.68 % frente a la persistencia, por lo que **no alcanzó** el criterio definido en el Hito 1 (mejora ≥ 10 %, MAE ≤ 5.33 µg/m³). Sí redujo el RMSE en 14.2 %. El mayor error fue de 41.01 µg/m³ (pico del 18 de junio a las 07:00). Las explicaciones de los errores son hipótesis: el conjunto no incluye clima ni tráfico.

Nota: el MAE de persistencia del Hito 2 (5.92 µg/m³, `experiments/resultados_baseline.json`) se calculó sobre 636 pares. La evaluación final usa 580 porque exige que también existan las ventanas móviles y el rezago de 24 h.

## Prototipo

`app.py` (Streamlit) permite consultar las predicciones de junio y simular un pronóstico con valores manuales. La banda ambiental que muestra es orientativa: el IBOCA oficial para PM2.5 usa una media NowCast ponderada de 12 h, no una predicción puntual a 24 h ([documentación oficial](https://iboca.ambientebogota.gov.co/publicaciones/175/que-es-el-iboca/)).

## Carpetas

- `data/`: datos de origen, datos limpios, partición y variables.
- `src/`: scripts del flujo (`main.py` los ejecuta en orden) y `modelo_pm25.py`, compartido por la explicabilidad y la app.
- `notebooks/`: análisis exploratorio, correlación y retardos, y verificación de la línea base. Ejecutar el flujo antes de abrirlos.
- `experiments/`: configuración, métricas, predicciones y figuras que respaldan cada tabla del informe.
- `docs/`: bitácora humano-IA (`bitacora_ia.md`) e informe técnico final (`pronostico_pm25_bogota_hito3.pdf`).
