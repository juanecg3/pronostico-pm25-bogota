"""Ejecuta el flujo completo del proyecto en orden.

Uso:
    python src/main.py              # flujo completo (incluye ajuste de hiperparámetros)
    python src/main.py --rapido     # omite el GridSearch y usa los hiperparámetros ya guardados

Pasos:
    1. Preprocesamiento de datos.xls -> data/datos limpios.csv
    2. Partición temporal por fecha objetivo (ene-abr / may / jun)
    3. Variables de entrada (rezagos y calendario)
    4. Línea base de persistencia (Hito 2)
    5. Ajuste de hiperparámetros sin ventanas móviles (GridSearch, validación en mayo)
    6. Ajuste con ventanas móviles pm25_roll3 / pm25_roll6 (modelo campeón)
    7. Evaluación final de una sola pasada en junio
    8. Análisis de los 10 mayores errores
    9. Explicabilidad por permutación
"""

from pathlib import Path
import runpy
import sys
import time


CARPETA_SRC = Path(__file__).resolve().parent

PASOS = [
    ("Preprocesamiento", "preprocesamiento.py", False),
    ("Partición temporal", "particion temporal.py", False),
    ("Variables de entrada", "crear variables.py", False),
    ("Línea base de persistencia", "linea base de persistencia.py", False),
    ("Ajuste de hiperparámetros (sin ventanas)", "entrenar modelos.py", True),
    ("Ajuste con ventanas móviles", "entrenar modelos ventanas.py", True),
    ("Evaluación final en junio", "evaluar prueba.py", False),
    ("Análisis de los 10 mayores errores", "analisis_errores.py", False),
    ("Explicabilidad por permutación", "explicabilidad.py", False),
]


def main():
    rapido = "--rapido" in sys.argv[1:]
    argv_original = sys.argv[:]
    inicio_total = time.time()
    for numero, (nombre, script, es_ajuste) in enumerate(PASOS, start=1):
        if rapido and es_ajuste:
            print(f"\n[{numero}/{len(PASOS)}] {nombre}: omitido (--rapido)")
            continue
        print(f"\n[{numero}/{len(PASOS)}] {nombre} ({script})")
        inicio = time.time()
        # Cada script se ejecuta como si se llamara directamente, sin argumentos extra.
        sys.argv = [str(CARPETA_SRC / script)]
        runpy.run_path(str(CARPETA_SRC / script), run_name="__main__")
        print(f"    listo en {time.time() - inicio:.1f} s")
    sys.argv = argv_original
    print(f"\nFlujo completo terminado en {time.time() - inicio_total:.1f} s.")
    print("Resultados en data/ y experiments/. Prototipo: streamlit run app.py")


if __name__ == "__main__":
    main()
