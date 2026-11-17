# Pronóstico de PM2.5 en Bogotá

Proyecto académico para preparar y pronosticar la concentración horaria de PM2.5 de la estación Kennedy.

## Estado actual

El proyecto limpia los datos horarios de la estación Kennedy, crea una partición cronológica y calcula una línea base sencilla para pronosticar PM2.5 a 24 horas.

El archivo de origen `data/datos.xls` es una tabla HTML descargada con extensión `.xls`. El programa detecta ese formato y también puede leer archivos Excel normales.

## Requisitos

- Python 3.11 o posterior.
- Windows y PowerShell para usar los comandos siguientes.

## Instalación y ejecución

Desde la carpeta principal del repositorio, ejecuta estos cuatro comandos:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\preprocesamiento.py
.\.venv\Scripts\python.exe src\main.py
```

El último comando crea la partición, las variables y la línea base. Los archivos resultantes quedan en `data/` y `experiments/`.

Para explorar los resultados con gráficos, abre los cuadernos de `notebooks/` en orden y selecciona el entorno `.venv` como kernel.

## ¿Qué hace el preprocesamiento?

1. Lee los datos de la estación Kennedy y organiza las horas en orden.
2. Convierte los textos `Sin data` en valores vacíos.
3. Interpola solo huecos internos de una o dos horas.
4. Conserva como vacíos los bloques largos y los extremos sin medición.
5. Guarda la tabla limpia en `data/datos limpios.csv`.

En esta descarga hay 4.345 horas entre el 1 de enero y el 1 de julio de 2026. Se imputaron 18 valores en huecos cortos; 329 faltantes se conservaron sin completar.

## Variables y evaluación

- `target_24h` es la medición original 24 horas después de la hora de origen.
- Las filas se separan por la fecha que se pronostica: enero-abril para train, mayo para validation y junio para test.
- Se crean la hora, el día de la semana y la concentración observada actual y sus retardos de una y 24 horas. Así las variables no dependen de interpolaciones que usan una lectura posterior.
- La persistencia predice que el valor dentro de 24 horas será igual a la medición actual.
- MAE y RMSE se calculan en test usando pares de mediciones originales disponibles; los valores imputados no se cuentan como mediciones reales.

La configuración de experimentos guarda `seed=42`. La línea base no usa aleatoriedad, pero la semilla quedará disponible para los modelos posteriores.

## Carpetas

- `data/`: datos de origen, datos limpios y resúmenes.
- `src/`: scripts de preprocesamiento, preparación de variables y línea base.
- `notebooks/`: cuadernos de análisis exploratorio, variables y evaluación visual.
- `experiments/`: configuración y resultados de la partición, las variables y la línea base.
- `docs/`: bitácora de uso de IA.
