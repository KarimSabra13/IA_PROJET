# main/pools.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Dict, Any, Tuple, Iterable

import pandas as pd
from pyngs.core import NGSpiceInstance


# =====================================================================
#  Fonctions utilitaires pour le parallélisme (workers multiprocessing)
# =====================================================================


def _worker_task(args: Tuple[str, str, Dict[str, float]]) -> float:
    """
    Fonction exécutée dans un processus fils (version parallèle).

    Paramètres
    ----------
    args : tuple
        (netlist_path_str, measure_name, params)

        - netlist_path_str : chemin vers le fichier .cir à utiliser
        - measure_name     : nom de la mesure SPICE (ex: 'fcut')
        - params           : dictionnaire {nom_param: valeur}

    Retour
    ------
    float
        Valeur de la mesure demandée (par exemple fréquence de coupure).
    """
    netlist_path_str, measure_name, params = args

    inst = NGSpiceInstance()

    try:
        # Charge la netlist dans ngspice
        inst.load(Path(netlist_path_str))

        # Applique chaque paramètre SPICE (R_val, C_val, etc.)
        for name, value in params.items():
            inst.set_parameter(name, float(value))

        # Lance la simulation SPICE (AC dans notre cas)
        inst.run()

        # Récupère la mesure demandée (ex: .meas ac fcut ...)
        value = inst.get_measure(measure_name)
        return float(value)

    finally:
        # Arrêt propre de l'instance ngspice dans ce processus
        inst.stop()


# =====================================================================
#  Classe de base commune
# =====================================================================


class BasePool:
    """
    Classe de base pour les environnements de simulation.

    Cette classe ne lance pas directement de simulation.
    Elle stocke simplement les informations communes:

    - la liste des chemins de netlists
    - le nom de la mesure SPICE à récupérer (par exemple 'fcut')
    """

    def __init__(
        self,
        netlist_paths: Sequence[str | Path],
        measure_name: str = "fcut",
    ) -> None:
        if not netlist_paths:
            raise ValueError("Au moins une netlist est requise.")

        # On convertit tout en Path pour être plus robustes
        self._netlist_paths: list[Path] = [Path(p) for p in netlist_paths]
        self._measure_name: str = measure_name

    # --- propriétés "propres" pour accéder aux attributs ---

    @property
    def netlist_paths(self) -> list[Path]:
        """Liste des chemins vers les netlists SPICE."""
        return self._netlist_paths

    @property
    def measure_name(self) -> str:
        """Nom de la mesure SPICE à lire (ex: 'fcut')."""
        return self._measure_name

    # --- méthode d'interface que les classes filles doivent implémenter ---

    def run(self, values: pd.DataFrame) -> pd.DataFrame:
        """
        Méthode à implémenter dans les classes dérivées.

        Paramètres
        ----------
        values : pandas.DataFrame
            - Colonnes : noms des paramètres SPICE (par ex. 'R_val', 'C_val').
            - Lignes   : jeux de valeurs à simuler.

        Retour
        ------
        pandas.DataFrame
            Doit contenir au minimum une colonne avec la mesure (par ex. 'fcut').
        """
        raise NotImplementedError("La méthode run() doit être implémentée.")


# =====================================================================
#  Version séquentielle
# =====================================================================


class SequentialPool(BasePool):
    """
    Version séquentielle de l'environnement de simulation.

    - Toutes les simulations sont faites dans le même processus Python.
    - On utilise une ou plusieurs instances de NGSpiceInstance.
    - Les simulations sont exécutées les unes après les autres.
    """

    def __init__(
        self,
        netlist_paths: Sequence[str | Path],
        measure_name: str = "fcut",
    ) -> None:
        super().__init__(netlist_paths, measure_name)

        # Liste d'instances ngspice, une par netlist.
        # Dans l'énoncé, on peut utiliser une seule netlist,
        # mais cette structure permet d'en utiliser plusieurs si besoin.
        self._instances: list[NGSpiceInstance] = []

        for path in self._netlist_paths:
            inst = NGSpiceInstance()
            inst.load(path)
            self._instances.append(inst)

    # -----------------------------------------------------------------
    #  Méthode interne pour une seule simulation
    # -----------------------------------------------------------------

    def _simulate_one(
        self,
        inst: NGSpiceInstance,
        params: Dict[str, float],
    ) -> float:
        """
        Applique un jeu de paramètres à une instance ngspice et renvoie la mesure.

        Paramètres
        ----------
        inst : NGSpiceInstance
            Instance déjà chargée avec la bonne netlist.
        params : dict
            Dictionnaire {nom_param: valeur}, par ex. {"R_val": 1000, "C_val": 1e-6}.

        Retour
        ------
        float
            Valeur de la mesure (par ex. la fréquence de coupure).
        """
        # Mise à jour des paramètres SPICE
        for name, value in params.items():
            inst.set_parameter(name, float(value))

        # Lancement de la simulation (AC, etc.)
        inst.run()

        # Lecture de la mesure .meas définie dans la netlist
        value = inst.get_measure(self._measure_name)
        return float(value)

    # -----------------------------------------------------------------
    #  Méthode run (interface principale)
    # -----------------------------------------------------------------

    def run(self, values: pd.DataFrame) -> pd.DataFrame:
        """
        Exécute les simulations une par une pour chaque ligne de la DataFrame.

        Paramètres
        ----------
        values : pandas.DataFrame
            - Colonnes : noms des paramètres SPICE (R_val, C_val, ...).
            - Lignes   : jeux de valeurs à tester.

        Retour
        ------
        pandas.DataFrame
            Une DataFrame avec une colonne 'fcut' (ou le nom choisi) contenant
            la valeur de la mesure pour chaque ligne d'entrée.
        """
        if values.empty:
            # Rien à faire, on renvoie une DataFrame vide avec la bonne colonne.
            return pd.DataFrame({self._measure_name: []})

        # Résultats de la mesure pour chaque ligne
        results: list[float] = []

        # On va utiliser les instances de ngspice de manière "round robin"
        # (utile si on a plusieurs netlists dans self._instances).
        nb_insts = len(self._instances)

        try:
            for idx, (_, row) in enumerate(values.iterrows()):
                # Dictionnaire {nom_param: valeur} pour cette ligne
                params = {col: row[col] for col in values.columns}

                # Choix de l'instance (si plusieurs netlists)
                inst = self._instances[idx % nb_insts]

                # Simulation et récupération de la mesure
                value = self._simulate_one(inst, params)
                results.append(value)

        finally:
            # On s'assure d'arrêter proprement toutes les instances ngspice
            for inst in self._instances:
                inst.stop()

        # On renvoie les résultats dans une DataFrame,
        # avec le même index que la DataFrame d'entrée.
        return pd.DataFrame(
            {self._measure_name: results},
            index=values.index.copy(),
        )


# =====================================================================
#  Version parallèle
# =====================================================================


class ParallelPool(BasePool):
    """
    Version parallèle de l'environnement de simulation.

    - Utilise multiprocessing.Pool pour exécuter plusieurs simulations en parallèle.
    - Chaque processus crée sa propre instance NGSpiceInstance.
    - Intéressant quand on a beaucoup de jeux de paramètres à simuler.
    """

    def __init__(
        self,
        netlist_paths: Sequence[str | Path],
        measure_name: str = "fcut",
    ) -> None:
        super().__init__(netlist_paths, measure_name)

    def run(self, values: pd.DataFrame) -> pd.DataFrame:
        """
        Exécute les simulations en parallèle.

        Paramètres
        ----------
        values : pandas.DataFrame
            - Colonnes : noms des paramètres SPICE (R_val, C_val, ...).
            - Lignes   : jeux de valeurs à tester.

        Retour
        ------
        pandas.DataFrame
            Une DataFrame avec une colonne 'fcut' (ou autre nom de mesure),
            les lignes étant dans le même ordre que la DataFrame d'entrée.
        """
        import multiprocessing as mp

        if values.empty:
            return pd.DataFrame({self._measure_name: []})

        # On transforme chaque ligne de la DataFrame en "tâche" pour un worker.
        tasks: list[Tuple[str, str, Dict[str, float], int]] = []

        netlists = [str(p) for p in self._netlist_paths]
        nb_netlists = len(netlists)

        # Construction des tâches
        for idx, (_, row) in enumerate(values.iterrows()):
            # Dictionnaire {nom_param: valeur}
            params = {col: row[col] for col in values.columns}

            # Sélection d'une netlist en round robin
            netlist_path = netlists[idx % nb_netlists]

            # On stocke aussi l'indice d'origine pour reconstruire l'ordre
            tasks.append((netlist_path, self._measure_name, params, idx))

        # Fonction interne pour appeler _worker_task et transmettre l'indice
        def _task_with_index(
            args: Tuple[str, str, Dict[str, float], int]
        ) -> Tuple[int, float]:
            netlist_path_str, measure_name, params, index = args
            value = _worker_task((netlist_path_str, measure_name, params))
            return index, value

        # Nombre de processus à utiliser
        # On prend min(nombre de netlists, nombre de CPU)
        n_processes = min(len(netlists), mp.cpu_count() or 1)

        with mp.Pool(processes=n_processes) as pool:
            # pool.map renvoie une liste de (index, valeur)
            results_with_index: list[Tuple[int, float]] = pool.map(
                _task_with_index, tasks
            )

        # On remet les résultats dans l'ordre des indices d'origine
        results_with_index.sort(key=lambda x: x[0])
        measures = [val for _, val in results_with_index]

        return pd.DataFrame(
            {self._measure_name: measures},
            index=values.index.copy(),
        )


# =====================================================================
#  Fonction utilitaire : choisir la version du pool
# =====================================================================


def create_pool(
    mode: str,
    netlist_paths: Sequence[str | Path],
    measure_name: str = "fcut",
) -> BasePool:
    """
    Fabrique un pool en fonction du mode demandé.

    Paramètres
    ----------
    mode : str
        - "sequential" : renvoie un SequentialPool
        - "parallel"   : renvoie un ParallelPool
    netlist_paths : liste de chemins vers les .cir
    measure_name  : nom de la mesure SPICE (par défaut 'fcut')

    Retour
    ------
    BasePool
        Une instance de SequentialPool ou ParallelPool.
    """
    mode = mode.lower()

    if mode in ("seq", "sequential"):
        return SequentialPool(netlist_paths, measure_name)
    elif mode in ("par", "parallel"):
        return ParallelPool(netlist_paths, measure_name)
    else:
        raise ValueError(f"Mode inconnu: {mode!r}. Utiliser 'sequential' ou 'parallel'.")
