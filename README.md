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

## Architecture détaillée
1. **PDK et modèles** : Ciel fournit le PDK SKY130A avec les modèles ngspice (tt, ff, ss…). La netlist `inv_char.cir` référence directement le `.lib` du PDK via un chemin absolu pour éviter toute ambiguïté d'environnement.
2. **Simulation SPICE** : la netlist unique génère un stimulus commun, mesure tphl/tplh/tpavg avec `.meas` et calcule la puissance statique via la source VDD. Cela garantit que toutes les mesures sont cohérentes et limite les ré-initialisations libngspice.
3. **Couche pyngs** : `main/inverter_spice.py` charge la netlist en mémoire, force les paramètres (wn/wp/vdd/lch), et exécute une transitoire courte. Un runner robuste isole chaque run dans un répertoire temporaire, surveille les erreurs « circuit not parsed », détecte les forks et peut redémarrer proprement libngspice.
4. **Environnement RL** : `main/rl_env.py` expose une action continue `[wn, wp]` (µm) avec normalisation d'observation. La récompense combine delay, puissance statique et aire via des poids configurables.
5. **Boucle PPO** : `main/optimize_inv.py` configure PPO (stable-baselines3) avec contrôle des threads BLAS/OMP, choix du `start_method` (spawn > fork) et callbacks : snapshots streamlit, suivi « best so far », early stop par plateau ou timeout.
6. **Interface Streamlit** : `streamlit_app.py` permet de régler les poids PPA, le nombre d'environnements parallèles, les pas d'entraînement et d'afficher en direct reward, métriques brutes/normalisées et meilleures largeurs.

## Cheminement des données (résumé)
- L'utilisateur choisit `wn/wp` → l'env RL renvoie une observation normalisée (delays/power/area scalés) et une récompense PPA.
- `rl_env` appelle le runner pyngs → libngspice simule la transitoire et extrait les mesures.
- PPO collecte les transitions, met à jour la politique et enregistre périodiquement des snapshots (métriques + temps écoulé) pour la GUI.
- Streamlit consomme les snapshots pour tracer reward/tpavg et afficher les meilleurs points atteints.

## Détails installation et environnement
- **Isolation Python** : `uv sync` installe toutes les dépendances (sauf pyngs). Le wheel `deps/pyngs-0.0.2-...whl` doit être installé via `uv pip install` car il n'est pas publié sur PyPI.
- **Version Python** : rester en 3.12 (libngspice/pyngs instable en 3.13).
- **ngspice** : vérifier `ngspice --version` (42 recommandé) et la présence des libs partagées nécessaires à pyngs.
- **OMP/BLAS** : l'entraînement force des variables (ex. `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) pour éviter la contention CPU et des ralentissements inattendus.
- **TMPDIR** : le runner SPICE crée un dossier temporaire par episode et peut forcer `TMPDIR` afin d'éviter les collisions de fichiers temporaires libngspice.

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

### Paramètres avancés du runner SPICE
- `workdir_root` : base pour les répertoires temporaires (par défaut dans `/tmp`).
- `max_restarts` : nombre de redémarrages libngspice autorisés avant d'échouer.
- `detect_fork` : refuse l'exécution si un fork est détecté (libngspice n'est pas fork-safe).
- `tmp_policy` : contrôle la gestion de `TMPDIR` (per-episode pour éviter les collisions).
- Les logs détaillés indiquent les erreurs de parsing, les redémarrages et les chemins utilisés pour diagnostiquer les corruptions internes.

## Lancer l'optimisation RL en ligne de commande
```bash
uv run python -m main.optimize_inv
```
Paramètres clés (voir `optimize_inverter`):
- `w_delay`, `w_power`, `w_area` : poids PPA.
- `total_timesteps` : nombre total d'itérations PPO.
- `n_envs` : 1 = séquentiel, >1 = pool de process `SubprocVecEnv`.
- `start_method` : `spawn` recommandé, fallback automatique si indisponible.
- `eval_interval` / `eval_episodes` : fréquence et nombre d'épisodes pour suivre les meilleurs points.
- `max_walltime` : interrompt l'entraînement si la durée totale dépasse ce budget.
- `early_stop_plateau` : tuple `(patience, min_delta, warmup)` pour stopper sur plateau de reward.
- `snapshot_interval` : fréquence (en steps) des snapshots envoyés à Streamlit.

### Conseils pour le multi-processus
- Préférer `spawn` (et non `fork`) pour éviter la corruption libngspice ; `fork` est toléré mais moins robuste.
- Les processus `daemon` (utilisés par défaut dans `SubprocVecEnv`) ne peuvent pas créer d'enfants : évitez de lancer des sous-processus supplémentaires depuis l'env.
- Si des erreurs "circuit not parsed" apparaissent en parallèle, réduire `n_envs` à 1, ou basculer vers une exécution ngspice en CLI par process (non embarquée) pour isoler les états.
- Éviter les tests via `python - <<'PY'` en mode spawn : placer les scripts dans `scripts/test_*.py` et exécuter avec `python -m scripts.test_xxx` pour que le module soit importable par les workers.

## Lancer la GUI Streamlit
```bash
uv run streamlit run streamlit_app.py
```
Fonctionnalités :
- Sélection des poids P/P/A et des hyperparamètres RL (timesteps, steps/épisode, envs parallèles, fréquence d'éval).
- Lancement du training et exécution ngspice intégrée.
- Affichage des meilleures largeurs, reward, PPA brutes et normalisées.
- Courbes de suivi reward / tpavg sur les évaluations.
- Streaming de snapshots (reward, tpavg, pstatic, area, temps écoulé) pour suivre la progression sans attendre la fin.

### Astuces d'usage Streamlit
- Dans un environnement distant, passer `--server.runOnSave=true` pour recharger automatiquement après modification du code.
- Le bouton « Start Training » lance l'optimisation en tâche de fond ; les snapshots s'affichent dès la première évaluation.
- Le panneau « Best so far » est mis à jour par le callback PPO et conserve la meilleure récompense observée même si les dernières itérations régressent.

## Notes importantes
- Les longueurs/largeurs des MOS sont en **microns** (les modèles SKY130 ngspice incluent `scale=1e-6`).
- Le seuil des mesures de delay est `0.9 V` (VDD/2 pour VDD=1.8 V).
- La puissance statique est calculée comme `-VDD * avg(I(VDD))` sur la fenêtre 30–40 ns de la transitoire.
- La mesure d'aire est un proxy linéaire (`wn + wp`) suffisant pour la comparaison PPA.
- Évitez d'utiliser plusieurs netlists dans un même process pyngs : `inv_char.cir` regroupe toutes les mesures pour la robustesse.
- Pour des runs longs en séquentiel, réduire la fenêtre de transitoire (points/time) ou activer un caching des résultats peut accélérer le balayage.
- Conserver les logs `results/` pour déboguer les crashes libngspice : ils contiennent les redémarrages et le temps écoulé par épisode.
