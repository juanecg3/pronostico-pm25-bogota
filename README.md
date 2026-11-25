# Pronóstico de PM2.5 en Bogotá

Proyecto académico para preparar y pronosticar la concentración horaria de PM2.5 de la estación Kennedy.

## Estado actual

El proyecto limpia los datos horarios de la estación Kennedy, crea una partición cronológica y calcula una línea base sencilla para pronosticar PM2.5 a 24 horas.

El archivo de origen `data/datos.xls` es una tabla HTML descargada con extensión `.xls`. El programa detecta ese formato y también puede leer archivos Excel normales.

## Requisitos

- Python 3.11 o posterior.
- Windows y PowerShell para usar los comandos siguientes.

## Instalación y ejecución

Desde la carpeta principal del repositorio, ejecuta estos comandos:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\preprocesamiento.py
.\.venv\Scripts\python.exe "src\particion temporal.py"
.\.venv\Scripts\python.exe "src\crear variables.py"
.\.venv\Scripts\python.exe "src\linea base de persistencia.py"
```

Los últimos tres comandos crean la partición, las variables y la línea base. Los archivos resultantes quedan en `data/` y `experiments/`.

Para repetir el ajuste y la evaluación de modelos:

```powershell
.\.venv\Scripts\python.exe "src\entrenar modelos.py"
.\.venv\Scripts\python.exe "src\evaluar prueba.py"
```

La evaluación usa los hiperparámetros elegidos en mayo, reentrena con objetivos de enero a mayo y mide una sola vez sobre objetivos de junio. No se ajustan parámetros con junio.

El análisis pasivo de errores se ejecuta con `python src/analisis_errores.py`. Lee las 580 predicciones congeladas, guarda los diez mayores errores y estadísticas de residuos en `experiments/top10_errores_analisis.json`, y genera `experiments/grafico_residuos_top10.png` con la serie temporal y el histograma. Las explicaciones son hipótesis: el conjunto no incluye datos de clima o tráfico.

## Explicabilidad y prototipo

Para calcular la importancia por permutación sobre las mismas 580 filas de prueba y generar los artefactos XAI:

```powershell
.\.venv\Scripts\python.exe src\explicabilidad.py
```

Los valores quedan en `experiments/resultados explicabilidad.json` y la figura en `experiments/importancia variables.png`. El script toma los hiperparámetros campeones del JSON de ajuste de mayo; en los resultados guardados son 400 árboles, profundidad 10, `min_samples_split=5`, hoja mínima 1 y `max_features=sqrt`.

Para iniciar el prototipo desde la carpeta del proyecto:

```powershell
streamlit run app.py
```

En Windows también se puede ejecutar con `\.venv\Scripts\python.exe -m streamlit run app.py`. La app ofrece consulta de predicciones de junio y simulación manual. La banda ambiental es orientativa: una predicción puntual a 24 h no basta para calcular el IBOCA oficial, que usa una media NowCast ponderada de 12 h para PM₂.₅ ([documentación oficial](https://iboca.ambientebogota.gov.co/publicaciones/175/que-es-el-iboca/)).

Para explorar los resultados con gráficos, abre los cuadernos de `notebooks/` en orden y selecciona el entorno `.venv` como kernel.

## ¿Qué hace el preprocesamiento?

1. Lee los datos de la estación Kennedy y organiza las horas en orden.
2. Convierte los textos `Sin data` en valores vacíos.
3. Interpola solo huecos internos de una o dos horas.
4. Conserva como vacíos los bloques largos y los extremos sin medición.
5. Calcula `pm25_roll3` y `pm25_roll6` con las últimas tres y seis mediciones originales observadas, respectivamente, y guarda la tabla limpia en `data/datos limpios.csv`.

En esta descarga hay 4.345 horas entre el 1 de enero y el 1 de julio de 2026. Se imputaron 18 valores en huecos cortos; 329 faltantes se conservaron sin completar.

## Variables y evaluación

- `target_24h` es la medición original 24 horas después de la hora de origen.
- Las filas se separan por la fecha que se pronostica: enero-abril para train, mayo para validation y junio para test.
- Se crean la hora, el día de la semana y la concentración observada actual y sus retardos de una y 24 horas. Así las variables no dependen de interpolaciones que usan una lectura posterior.
- La persistencia predice que el valor dentro de 24 horas será igual a la medición actual.
- MAE y RMSE se calculan en test usando pares de mediciones originales disponibles; los valores imputados no se cuentan como mediciones reales.
- El ajuste de modelos y sus métricas de validación de mayo se guardan en `experiments/resultados ajuste modelos.json`.
- La evaluación final de junio se registra en `experiments/evaluacion final prueba.json`; las predicciones de los tres métodos y sus errores por fila quedan en `experiments/predicciones detalladas prueba.csv`.
- Para ejecutar la evaluación final: `python "src/evaluar prueba.py"`. La configuración de Random Forest se congela desde los resultados de mayo.

En la evaluación final registrada, se usaron 2.950 muestras de entrenamiento, con objetivos disponibles del 3 de enero al 29 de mayo a las 12:00, y 580 pares válidos comunes de junio. Random Forest obtuvo MAE 6.0465, RMSE 7.3634 y R² 0.1517; persistencia obtuvo MAE 6.1500 y HistGradientBoosting MAE 6.1137. Random Forest mejoró 1.68% frente a persistencia, por lo que no alcanzó la mejora del 10% ni el umbral de MAE ≤ 5.33 µg/m³. La muestra evaluada cubrió del 2 de junio 11:00 al 30 de junio 23:00 por disponibilidad de pares completos.

La configuración de experimentos guarda `seed=42`. La línea base no usa aleatoriedad, pero la semilla quedará disponible para los modelos posteriores.

## Carpetas

- `data/`: datos de origen, datos limpios y resúmenes.
- `src/`: scripts de preprocesamiento, preparación de variables, entrenamiento y evaluación final.
- `notebooks/`: cuadernos de análisis exploratorio, variables y evaluación visual.
- `experiments/`: configuración y resultados de la partición, las variables y la línea base.
- `docs/`: bitácora de uso de IA.
