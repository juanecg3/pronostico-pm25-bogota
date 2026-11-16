# Pronóstico de PM2.5 en Bogotá

Proyecto académico para preparar y pronosticar la concentración horaria de PM2.5 de la estación Kennedy.

## Estado actual

En esta primera etapa, el proyecto limpia los datos horarios de la estación Kennedy. La partición temporal y la línea base de persistencia se agregarán en los siguientes avances.

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
```

Al terminar, los archivos limpios y los resúmenes quedan en `data/`.

## ¿Qué hace el preprocesamiento?

1. Lee los datos de la estación Kennedy y organiza las horas en orden.
2. Convierte los textos `Sin data` en valores vacíos.
3. Interpola solo huecos internos de una o dos horas.
4. Conserva como vacíos los bloques largos y los extremos sin medición.
5. Guarda la tabla limpia en `data/datos limpios.csv`.

En esta descarga hay 4.345 horas entre el 1 de enero y el 1 de julio de 2026. Se imputaron 18 valores en huecos cortos; 329 faltantes se conservaron sin completar.

## Carpetas

- `data/`: datos de origen, datos limpios y resúmenes.
- `src/`: scripts de preprocesamiento y partición.
- `notebooks/`: cuadernos para exploración; no se necesitan para ejecutar los scripts.
- `experiments/`: configuración reproducible y, en el siguiente avance, resultados de la línea base.
- `docs/`: bitácora de uso de IA.
