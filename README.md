# TP – Filtre RC, ngspice et pyngs

## Structure du projet

Le projet est organisé comme suit:

* `spice/` : netlists ngspice

  * `spice/rc_filter.cir` : filtre RC du premier ordre avec mesure de la fréquence de coupure
* `main/` : code Python réutilisable

  * `main/rc_analysis.py` : fonctions pour lancer ngspice via pyngs et récupérer la fréquence de coupure
  * `main/pools.py` : infrastructure de simulation séquentielle et parallèle autour de pyngs
* `main.py` : script principal qui appelle les fonctions de `main/rc_analysis.py` (et éventuellement les pools)
* `deps/` : fichiers wheel fournis pour pyngs

  * `deps/pyngs-0.0.2-cp312-cp312-linux_x86_64.whl`
* `results/` : répertoire prévu pour logs, CSV, figures
* `pyproject.toml`, `uv.lock` : configuration du projet uv et dépendances

---

## Question 1 – Filtre RC du premier ordre

Objectif
Décrire un filtre RC du premier ordre en SPICE, avec des paramètres pour R et C, puis calculer la fréquence de coupure théorique.

Description du circuit
On modélise un filtre RC passe-bas:

* Source Vin appliquée sur une résistance R
* Un condensateur C entre la sortie et la masse
* La sortie se trouve entre R et C

Le netlist SPICE utilise des paramètres pour R et C:

* `Rval` = valeur de la résistance
* `Cval` = valeur de la capacité

Netlist de base (contenu de `spice/rc_filter.cir`):

```spice
* Filtre RC passe-bas du premier ordre

.param Rval = 1k
.param Cval = 100n

Vin in 0 AC 1
R1 in out {Rval}
C1 out 0 {Cval}

.ac dec 100 10 1Meg

.end
```

Fréquence de coupure théorique

Formule:

```text
fc = 1 / (2 * pi * R * C)
```

Avec:

* R = 1 kΩ = 1e3 ohms
* C = 100 nF = 100e-9 farads

Produit RC:

```text
RC = 1e3 * 100e-9 = 1e-4 secondes
```

Donc:

```text
fc = 1 / (2 * pi * 1e-4)
fc ≈ 1 / (6.283e-4)
fc ≈ 1.59e3 Hz
```

Fréquence de coupure théorique du filtre:

```text
fc ≈ 1.6 kHz
```

Le netlist complet du filtre est dans le fichier `spice/rc_filter.cir`.

---

## Question 2 – Source de tension et simulation AC

Objectif
Ajouter une source de tension sur l’entrée du filtre et configurer une simulation AC pour pouvoir extraire la fréquence de coupure.

Choix de la source
On utilise une source de tension Vin entre le nœud `in` et la masse, avec une amplitude AC de 1 V.
En SPICE, le mot clé `AC` définit l’amplitude pour l’analyse fréquentielle petit signal.

Ligne correspondante dans le netlist:

```spice
Vin in 0 AC 1
```

Configuration de la simulation AC
On réalise un balayage fréquentiel logarithmique de 10 Hz à 1 MHz avec 100 points par décade.
Directive utilisée:

```spice
.ac dec 100 10 1Meg
```

Interprétation

* À basse fréquence, le condensateur se comporte comme un circuit ouvert, le gain en sortie vaut environ 1 (0 dB).
* Quand la fréquence augmente, la tension de sortie diminue.
* La fréquence de coupure est la fréquence pour laquelle le gain vaut `1 / sqrt(2)` de la valeur basse fréquence, soit environ −3 dB.
* La simulation AC permet de visualiser `V(out)` en fonction de la fréquence et de comparer ensuite la fréquence de coupure mesurée à la valeur théorique calculée à la question 1.

---

## Question 3 – Mesure de la fréquence de coupure avec `.meas`

Objectif
Automatiser le calcul de la fréquence de coupure directement dans ngspice.

Principe
Pour une source AC de 1 V, le gain en décibels vaut:

```text
gain_dB = vdb(out)
```

La fréquence de coupure correspond au point où le gain vaut −3 dB.

La directive `.meas` utilisée dans `spice/rc_filter.cir` est:

```spice
.meas ac f_cutoff WHEN vdb(out) = -3
```

Explication des mots clés:

* `.meas ac` : demande une mesure sur l’analyse AC
* `f_cutoff` : nom du résultat qui apparaîtra dans la sortie de ngspice
* `WHEN vdb(out) = -3` : ngspice recherche la fréquence pour laquelle `vdb(out)` atteint −3 dB en interpolant entre les points du sweep AC

Utilisation

Lancer ngspice sur le fichier:

```bash
ngspice spice/rc_filter.cir
```

À la fin de l’analyse, ngspice affiche une ligne du type:

```text
f_cutoff = 1.59e+03
```

Cette fréquence se compare ensuite à la valeur théorique 1.6 kHz obtenue à la question 1.

---

## Question 4 – Exécution de ngspice et validation de la fréquence de coupure

Objectif
Exécuter le fichier `spice/rc_filter.cir` dans ngspice et vérifier que la mesure `.meas` donne une fréquence de coupure proche de la valeur théorique.

Procédure en mode interactif

Depuis le dossier du projet:

```bash
ngspice spice/rc_filter.cir
```

Ngspice charge le netlist et affiche un prompt:

```text
ngspice 1 ->
```

On lance alors la simulation AC avec:

```bash
run
```

À la fin de la simulation, ngspice exécute la directive:

```spice
.meas ac f_cutoff WHEN vdb(out) = -3
```

et affiche une ligne du type:

```text
f_cutoff = 1.590e+03
```

Comparaison avec la théorie

* Fréquence de coupure théorique calculée à la question 1:

```text
fc_th ≈ 1.59e3 Hz
```

* Valeur mesurée par ngspice:

```text
fc_meas ≈ 1.59e3 Hz
```

Les deux valeurs sont presque identiques, ce qui confirme que le netlist, la source de tension, la simulation AC et la mesure `.meas` sont corrects.

---

## Question 5 – Utilisation de pyngs et balayage en Python

Objectif
Récupérer la fréquence de coupure depuis ngspice en Python à l’aide du module `pyngs`, puis mesurer cette fréquence pour une liste de couples (R, C).

Installation de pyngs

Le wheel fourni est placé dans le dossier `deps/` puis installé avec uv:

```bash
uv pip install deps/pyngs-0.0.2-cp312-cp312-linux_x86_64.whl
```

uv met automatiquement à jour `pyproject.toml` et `uv.lock`.

Intégration avec pyngs

On utilise la classe `NGSpiceInstance` du module `pyngs.core`.
La logique est encapsulée dans `main/rc_analysis.py`:

* `spice/rc_filter.cir` contient le filtre RC et la mesure `.meas ac f_cutoff WHEN vdb(out) = -3`.
* La fonction `sweep_cutoff` charge le netlist, met à jour les paramètres `Rval` et `Cval` pour chaque couple `(R, C)`, lance la simulation et lit la mesure `f_cutoff` avec `inst.get_measure("f_cutoff")`.
* La fonction `theoretical_cutoff` calcule la fréquence de coupure théorique:

```text
fc = 1 / (2 * pi * R * C)
```

Exemple de code dans `main/rc_analysis.py` (simplifié):

```python
from math import pi
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any

from pyngs.core import NGSpiceInstance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NETLIST_PATH = PROJECT_ROOT / "spice" / "rc_filter.cir"


def theoretical_cutoff(R: float, C: float) -> float:
    return 1.0 / (2.0 * pi * R * C)


def sweep_cutoff(rc_values: Iterable[Tuple[float, float]]) -> list[Dict[str, Any]]:
    inst = NGSpiceInstance()
    inst.load(NETLIST_PATH)

    results = []

    for R, C in rc_values:
        inst.set_parameter("Rval", R)
        inst.set_parameter("Cval", C)
        inst.run()
        f_meas = inst.get_measure("f_cutoff")
        f_th = theoretical_cutoff(R, C)

        results.append(
            {
                "R": R,
                "C": C,
                "f_theoretical": f_th,
                "f_measured": f_meas,
            }
        )

    inst.stop()
    return results
```

Script principal `main.py`

Le script principal définit une liste de couples `(R, C)` et appelle `sweep_cutoff`:

```python
from main.rc_analysis import sweep_cutoff


def main() -> None:
    rc_values = [
        (1e3, 100e-9),
        (2e3, 100e-9),
        (1e3, 220e-9),
        (4.7e3, 47e-9),
    ]

    results = sweep_cutoff(rc_values)

    print("R (ohm)\tC (F)\t\tf_th (Hz)\t\tf_meas (Hz)")
    for r in results:
        print(
            f"{r['R']:.0f}\t"
            f"{r['C']:.2e}\t"
            f"{r['f_theoretical']:.2f}\t"
            f"{r['f_measured']:.2f}"
        )


if __name__ == "__main__":
    main()
```

Exécution:

```bash
uv run python main.py
```

Le programme affiche pour chaque couple `(R, C)`:

* la fréquence de coupure théorique,
* la fréquence de coupure mesurée par ngspice.

Les valeurs mesurées sont très proches des valeurs théoriques, ce qui valide le filtre RC, la mesure `.meas` et l’intégration Python via pyngs.

---

## Environnement Python, uv et pyngs

Le projet utilise `uv` pour gérer l’environnement Python et les dépendances.

### Version de Python

Au départ, le projet était configuré avec:

```toml
requires-python = ">=3.13"
```

et uv utilisait Python 3.13.9.
Avec cette configuration, l’appel au module `pyngs.core` provoquait une erreur fatale à la fin de l’exécution de Python:

```text
Fatal Python error: _PyThreadState_Attach: non-NULL old thread state
Python runtime state: initialized
Extension modules: ..., pyngs.core
```

Cette erreur ne vient pas du code Python, mais d’un problème de compatibilité entre la version 3.13 de Python et le module natif `pyngs.core`.

Pour stabiliser l’environnement, le projet a été basculé sur Python 3.12, de la façon suivante:

1. Modifier `pyproject.toml` pour autoriser Python 3.12:

   ```toml
   requires-python = ">=3.12,<3.13"
   ```

2. Installer et pinner Python 3.12 avec uv:

   ```bash
   uv python install 3.12
   uv python pin 3.12
   ```

3. Recréer l’environnement virtuel et resynchroniser les dépendances:

   ```bash
   rm -rf .venv
   uv sync
   ```

4. Vérifier la version utilisée par uv:

   ```bash
   uv run python -c "import sys; print(sys.version)"
   # → 3.12.x
   ```

### Installation de pyngs

Pour être cohérent avec cette version de Python, on installe le wheel `cp312` de pyngs:

```bash
uv pip install deps/pyngs-0.0.2-cp312-cp312-linux_x86_64.whl
```

uv met à jour `pyproject.toml` et `uv.lock` et installe `pyngs` dans l’environnement virtuel `.venv`.

Avec cette configuration:

* les appels à `pyngs.core.NGSpiceInstance` fonctionnent correctement,
* la mesure `f_cutoff` est récupérée sans erreur,
* l’erreur fatale `_PyThreadState_Attach` ne se reproduit plus.

Ce choix de rester sur Python 3.12 est cohérent avec le contexte du TP, où `pyngs` est fourni sous forme de wheel précompilé pour cette version.

---

## Question 6 – Environnement de simulation en Python (SequentialPool et ParallelPool)

Objectif
Créer un environnement de simulation générique capable de lancer automatiquement des simulations ngspice pour une liste de paramètres, avec deux modes:

* un mode séquentiel (`SequentialPool`) qui exécute les simulations une par une
* un mode parallèle (`ParallelPool`) qui utilise plusieurs processus

Interface commune
Les deux classes dérivent d’une classe de base `BasePool` définie dans `main/pools.py`.
Cette classe stocke:

* la liste des netlists SPICE à utiliser
* le nom de la mesure SPICE à récupérer (par exemple `f_cutoff`)

L’interface attendue est:

```python
from main.pools import SequentialPool, ParallelPool

pool = SequentialPool(["spice/rc_filter.cir"], measure_name="f_cutoff")
result = pool.run(values)
```

où `values` est une `pandas.DataFrame` dont les colonnes correspondent aux paramètres SPICE (ici `Rval` et `Cval`).

### SequentialPool

`SequentialPool` crée une instance de `NGSpiceInstance` par netlist et charge les netlists une seule fois au constructeur.
La méthode interne `_simulate_one`:

* reçoit un dictionnaire `{"Rval": ..., "Cval": ...}`
* met à jour les paramètres dans ngspice avec `set_parameter`
* lance la simulation avec `run`
* lit la mesure demandée avec `get_measure(measure_name)`

La méthode `run(values)`:

1. Parcourt les lignes de la DataFrame `values`
2. Pour chaque ligne, construit un dictionnaire `{nom_param: valeur}`
3. Choisit une instance ngspice (round robin si plusieurs netlists)
4. Appelle `_simulate_one` et stocke la mesure dans une liste
5. À la fin, appelle `stop()` sur toutes les instances pour arrêter proprement ngspice
6. Renvoie une `DataFrame` avec une colonne `f_cutoff` contenant toutes les mesures

Ce mode est simple et suffisant pour un volume modéré de simulations.

### ParallelPool

`ParallelPool` utilise le module `multiprocessing` pour exécuter plusieurs simulations en parallèle.
Chaque simulation est traitée par un processus fils qui:

* crée sa propre `NGSpiceInstance`
* charge la netlist
* applique les paramètres
* lance la simulation
* renvoie la mesure `f_cutoff`
* puis appelle `inst.stop()`

L’implémentation suit les étapes suivantes:

1. Transformer chaque ligne de `values` en une tâche du type
   `(chemin_netlist, measure_name, params, index)`
2. Associer les tâches aux netlists de manière round robin
3. Utiliser un `multiprocessing.Pool` avec un nombre de processus égal au minimum entre le nombre de netlists et le nombre de cœurs CPU
4. Appliquer la fonction `_worker_task` à toutes les tâches
5. Récupérer les couples `(index, valeur)`, trier par `index` pour retrouver l’ordre d’origine, et construire une DataFrame avec la colonne `f_cutoff`

Ce mode permet de réduire le temps total pour de grandes listes de paramètres, au prix d’une complexité un peu plus élevée.

### Utilisation via `create_pool` et `main.py`

Pour simplifier l’utilisation dans le script principal, une fonction utilitaire `create_pool` est définie dans `main/pools.py`:

```python
from main.pools import create_pool

pool = create_pool("sequential", ["spice/rc_filter.cir"], measure_name="f_cutoff")
# ou bien
# pool = create_pool("parallel", ["spice/rc_filter.cir"], measure_name="f_cutoff")
```

Dans `main.py`:

1. On construit la DataFrame `values` avec les colonnes `Rval` et `Cval`
2. On choisit le mode `"sequential"` ou `"parallel"`
3. On crée le pool avec `create_pool(...)`
4. On appelle `pool.run(values)` pour obtenir une DataFrame des fréquences de coupure `f_cutoff`
5. On affiche les résultats

Les valeurs obtenues sont très proches de la fréquence de coupure théorique `fc = 1 / (2π R C)`, ce qui valide à la fois:

* le filtre RC et la mesure `.meas ac f_cutoff WHEN vdb(out) = -3` dans ngspice
* l’intégration de ngspice via `pyngs`
* et la logique de l’environnement de simulation en Python (séquentiel et parallèle)

---

```bash
git add README.md pyproject.toml uv.lock
git commit -m "Document Python 3.12 environment, pyngs setup and simulation pools"
git push
```
