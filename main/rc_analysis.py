from __future__ import annotations

from math import pi
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any

from pyngs.core import NGSpiceInstance


# Chemin du netlist RC
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NETLIST_PATH = PROJECT_ROOT / "spice" / "rc_filter.cir"


def theoretical_cutoff(R: float, C: float) -> float:
    """Fréquence de coupure théorique: fc = 1 / (2*pi*R*C)."""
    return 1.0 / (2.0 * pi * R * C)


def sweep_cutoff(rc_values: Iterable[Tuple[float, float]]) -> list[Dict[str, Any]]:
    """
    Pour chaque couple (R, C), lance ngspice via pyngs, récupère la mesure f_cutoff
    et renvoie une liste de dictionnaires avec théorie + mesure.
    """
    inst = NGSpiceInstance()
    results: list[Dict[str, Any]] = []

    try:
        # Charger le netlist une seule fois
        inst.load(NETLIST_PATH)

        for R, C in rc_values:
            # Met à jour les paramètres du netlist
            inst.set_parameter("Rval", R)
            inst.set_parameter("Cval", C)

            # Lance la simulation AC
            inst.run()

            # Récupère la mesure .meas f_cutoff définie dans le netlist
            f_meas = inst.get_measure("f_cutoff")

            # Calcule la valeur théorique
            f_th = theoretical_cutoff(R, C)

            results.append(
                {
                    "R": R,
                    "C": C,
                    "f_theoretical": f_th,
                    "f_measured": f_meas,
                }
            )

    finally:
        # Très important pour éviter le crash Python en fin de programme
        inst.stop()

    return results
