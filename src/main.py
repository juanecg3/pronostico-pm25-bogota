"""Ejecuta la partición, las variables y la línea base en orden."""

from baseline_persistencia import main as calcular_baseline
from crear_variables import main as crear_variables
from particion_temporal import main as crear_particion


def main():
    crear_particion()
    crear_variables()
    calcular_baseline()


if __name__ == "__main__":
    main()
