"""Ejecuta la partición, las variables y la línea base en orden."""

from pathlib import Path
import runpy


CARPETA_SRC = Path(__file__).resolve().parent


def main():
    for script in (
        "particion temporal.py",
        "crear variables.py",
        "linea base de persistencia.py",
    ):
        runpy.run_path(str(CARPETA_SRC / script), run_name="__main__")


if __name__ == "__main__":
    main()
