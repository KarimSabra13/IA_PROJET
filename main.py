from __future__ import annotations

import os
import sys

import pandas as pd

from main.pools import create_pool


def main() -> None:
    # Jeux de valeurs comme dans l'énoncé
    values = pd.DataFrame({
        "R_val": [10, 100, 1000, 10000, 10, 100, 1000, 10000],
        "C_val": [1e-6, 1e-6, 1e-6, 1e-6, 2e-6, 2e-6, 2e-6, 2e-6],
    })

    # Choix du mode de simulation
    # "sequential" pour SequentialPool
    # "parallel" pour ParallelPool
    mode = "sequential"

    # Création du pool adapté
    pool = create_pool(mode, ["spice/rc_filter.cir"], measure_name="fcut")

    # Lancement des simulations
    result = pool.run(values)

    # Affichage des résultats
    print(result)

    # On vide la sortie standard, puis on coupe le process tout de suite
    # pour éviter le crash de pyngs à la fermeture de l'interpréteur.
    sys.stdout.flush()
    os._exit(0)

# test test 
if __name__ == "__main__":
    main()
