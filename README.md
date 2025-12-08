````markdown
# TP – Filtre RC, ngspice et pyngs

## Structure du projet

Le projet est organisé comme suit:

- `spice/` : netlists ngspice  
  - `spice/rc_filter.cir` : filtre RC du premier ordre avec mesure de la fréquence de coupure
- `main/` : code Python réutilisable  
  - `main/rc_analysis.py` : fonctions pour lancer ngspice via pyngs et récupérer la fréquence de coupure
- `main.py` : script principal qui appelle les fonctions de `main/rc_analysis.py`
- `deps/` : fichiers wheel fournis pour pyngs  
  - `deps/pyngs-0.0.2-cp313-cp313-linux_x86_64.whl`
- `results/` : répertoire prévu pour logs, CSV, figures
- `pyproject.toml`, `uv.lock` : configuration du projet uv et dépendances

---

## Question 1 – Filtre RC du premier ordre

Objectif  
Décrire un filtre RC du premier ordre en SPICE, avec des paramètres pour R et C, puis calculer la fréquence de coupure théorique.

Description du circuit  
On modélise un filtre RC passe-bas:

- Source Vin appliquée sur une résistance R.
- Un condensateur C entre la sortie et la masse.
- La sortie se trouve entre R et C.

Le netlist SPICE utilise des paramètres pour R et C:

- `Rval` = valeur de la résistance
- `Cval` = valeur de la capacité

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
````

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
* La fréquence de coupure est la fréquence pour laquelle le gain est tombé à `1 / sqrt(2)` de la valeur basse fréquence, soit environ −3 dB.
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

* `.meas ac` : demande une mesure sur l’analyse AC.
* `f_cutoff` : nom du résultat qui apparaîtra dans la sortie de ngspice.
* `WHEN vdb(out) = -3` : ngspice recherche la fréquence pour laquelle `vdb(out)` atteint −3 dB en interpolant entre les points du sweep AC.

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

```text
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
uv pip install deps/pyngs-0.0.2-cp313-cp313-linux_x86_64.whl
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

````

Ensuite:

```bash
git add README.md
git commit -m "Update README with project structure and questions 1–5"
git push
````

