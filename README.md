# Optimisation RL d'une standard cell SKY130

Ce dépôt assemble l'ensemble de la chaîne "PDK Ciel → ngspice → pyngs → PPO" pour optimiser un inverseur CMOS (standard cell) sur SKY130 avec des métriques PPA (delay / static power / area). L'application Streamlit permet de lancer la boucle RL (séquentielle ou parallèle) et de visualiser les résultats.

## Prérequis
- Python 3.12 géré par `uv` (le wheel `pyngs` plante en 3.13).
- `ngspice` 42 installé (WSL ou Linux natif).
- PDK SKY130 installé localement via Ciel, ex. `./pdk_local/ciel/sky130/versions/<hash>/sky130A`.
- Wheel `deps/pyngs-0.0.2-cp312-...whl` installé avec `uv pip install deps/...`.

## Activer le PDK SKY130 avec Ciel
1. Lister les versions : `uvx ciel ls --pdk-family sky130`.
2. Activer la version choisie :
   ```bash
   uvx ciel enable --pdk-family sky130 <hash>
   ```
3. Vérifier la présence des modèles ngspice :
   ```bash
   ls pdk_local/ciel/sky130/versions/<hash>/sky130A/libs.tech/ngspice/tt.lib.spice
   ```
4. La ligne `.lib` dans `spice/inv_char.cir` utilise **le chemin absolu** vers `tt.lib.spice` (pas de `~` ni de variables d'env).

## Structure du dépôt
- `spice/inv_char.cir` : netlist ngspice unique pour l'inverseur avec mesures delay/fuite/puissance/aire.
- `main/inverter_spice.py` : wrapper pyngs pour mesurer PPA sur un jeu de largeurs (wn/wp).
- `main/rl_env.py` : environnement Gymnasium multi-objectif (actions = [wn, wp]).
- `main/optimize_inv.py` : boucle PPO (séquentielle ou `SubprocVecEnv`), suivi des meilleures métriques.
- `streamlit_app.py` : GUI pour piloter l'optimisation et visualiser les courbes.
- `deps/` : wheel pyngs fournie.

## Installation rapide
```bash
uv sync
uv pip install deps/pyngs-0.0.2-cp312-*whl
```
Assurez-vous que `ngspice` est accessible dans le PATH.

## Lancer une mesure PPA seule
```bash
uv run python -m main.inverter_spice
```
Cela charge `spice/inv_char.cir`, fixe les paramètres wn/wp/vdd/lch et imprime tphl/tplh/tpavg + fuite + puissance statique + aire.

## Lancer l'optimisation RL en ligne de commande
```bash
uv run python -m main.optimize_inv
```
Paramètres clés (voir `optimize_inverter`):
- `w_delay`, `w_power`, `w_area` : poids PPA.
- `total_timesteps` : nombre total d'itérations PPO.
- `n_envs` : 1 = séquentiel, >1 = pool de process `SubprocVecEnv`.
- `eval_interval` / `eval_episodes` : fréquence et nombre d'épisodes pour suivre les meilleurs points.

## Lancer la GUI Streamlit
```bash
uv run streamlit run streamlit_app.py
```
Fonctionnalités :
- Sélection des poids P/P/A et des hyperparamètres RL (timesteps, steps/épisode, envs parallèles, fréquence d'éval).
- Lancement du training et exécution ngspice intégrée.
- Affichage des meilleures largeurs, reward, PPA brutes et normalisées.
- Courbes de suivi reward / tpavg sur les évaluations.

## Notes importantes
- Les longueurs/largeurs des MOS sont en **microns** (les modèles SKY130 ngspice incluent `scale=1e-6`).
- Le seuil des mesures de delay est `0.9 V` (VDD/2 pour VDD=1.8 V).
- La puissance statique est calculée comme `-VDD * avg(I(VDD))` sur la fenêtre 30–40 ns de la transitoire.
- La mesure d'aire est un proxy linéaire (`wn + wp`) suffisant pour la comparaison PPA.
- Évitez d'utiliser plusieurs netlists dans un même process pyngs : `inv_char.cir` regroupe toutes les mesures pour la robustesse.
