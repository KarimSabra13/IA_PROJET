```markdown
# SKY130 StdCell RL Optimizer (PoC)  
Fast, robust, parallel training with live GUI, and reproducible setup.

This repo is a proof of concept for optimizing SKY130 standard-cell style CMOS gates with reinforcement learning.  
We start with an inverter. Next step will be NOR2, then more cells.

Core goals:
- Optimize propagation delay (TPHL, TPLH, and TPAVG)
- Optimize static power (leakage, measured across input states)
- Optimize area proxy (sum of widths)
- Provide a professional GUI that shows live progress and explains what is happening

This README covers:
- What the project does
- The full setup from scratch
- The full flow end-to-end
- Each file and its logic
- Common failures and fixes
- How to run overnight training safely


## 0) What we built so far

### What works now
- Netlist generation from Python
- ngspice batch runs for each action
- Robust log parsing for measures
- Reward computation from delay, static power, area
- PPO training that can run with multiple parallel workers
- Clean run isolation per training session
- Live GUI that:
  - Starts and stops training
  - Shows live curves and a Pareto-style scatter
  - Shows backend logs
  - Simulates the best solution and plots waveform (vin, vout)

### Key design decisions (based on real issues we hit)
We originally tried embedded ngspice via pyngs. It worked for single runs but broke under stress and parallel training. We saw errors like “circuit not parsed” and even memory corruption crashes.
So we moved to the most robust approach for parallelism:
- Use ngspice CLI (`ngspice -b`) per simulation
- Run workers in separate processes (SubprocVecEnv)
- Use `spawn` start method
- Write netlists and outputs to unique paths per worker and per simulation

This is the fastest path that also stays stable for long runs.


## 1) Repo structure

Typical structure:
```

stdcell_rl_poc/
main/
config.py
netlist_gen.py
spice_runner.py
smoke_test.py
rl_env.py
reward.py
optimize_inv.py
gui_app.py
spice/
templates/              (optional, if you use templates)
generated/              (older single-file approach, kept for reference)
results/
run_YYYYMMDD_HHMMSS/
status.json
best.json
history.csv
checkpoints/
worker_<pid>/
inv_<id>.cir
inv_<id>.log
inv_<id>.raw
pyproject.toml
uv.lock
README.md

````

Notes:
- New robust flow writes into `results/run_.../worker_<pid>/...`
- The old flow used `spice/generated/inv_char.cir`. That is unsafe under parallel runs.


## 2) Requirements

### System
- Linux recommended
- ngspice installed and working:
  ```bash
  ngspice -v
````

### Python

* Python 3.12 (we pinned this because some wheels fail on 3.13 in our setup)
* uv for env management

If uv is installed:

```bash
uv --version
```

## 3) Setup from scratch

Go to your repo directory:

```bash
cd ~/stdcell_rl_poc
```

### 3.1 Create and sync the environment

```bash
uv python pin 3.12
uv sync
```

Important rule:

* Always run project commands like this:

  ```bash
  uv run python -m <module>
  ```

If you run `python -m ...` directly, you might use system Python and miss deps.

Example of what happened before:

* GUI crashed with `ModuleNotFoundError: numpy`
* Fix was running with `uv run`

### 3.2 Install missing Python deps (if needed)

If the GUI or optimizer complains about missing packages, install them:

```bash
uv add numpy pandas pyside6 pyqtgraph stable-baselines3 gymnasium jinja2 spicelib
uv sync
```

## 4) SKY130 PDK setup (ciel)

You currently have:

```bash
uvx ciel ls --pdk-family sky130
No PDKs installed.
```

### 4.1 List available SKY130 builds

Try:

```bash
uvx ciel ls-remote --pdk-family sky130
```

### 4.2 Enable one PDK version

Pick a hash from the output and enable it:

```bash
uvx ciel enable --pdk-family sky130 <HASH>
```

### 4.3 Find `sky130.lib.spice` on disk

After enabling, locate the ngspice library:

```bash
find ~/.ciel -type f -name "sky130.lib.spice" | head -n 20
```

Take the path you want and export it:

```bash
export SKY130_LIB_SPICE="/absolute/path/to/sky130.lib.spice"
```

Optional: persist in your shell config:

```bash
echo 'export SKY130_LIB_SPICE="/absolute/path/to/sky130.lib.spice"' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
echo "$SKY130_LIB_SPICE"
ls -l "$SKY130_LIB_SPICE"
```

If `$SKY130_LIB_SPICE` is wrong, ngspice will fail with “can’t find model …”.

Also verify ngspice works with that lib:

```bash
ngspice -b -o /tmp/ng_check.log - <<'SP'
* tiny check
.end
SP
```

## 5) The full flow end-to-end

### Step A: Generate a netlist for the inverter

```bash
uv run python -m main.netlist_gen --cell inv --wn 0.5 --wp 1.0
```

This creates a characterization netlist with:

* VDD supply
* Input pulse
* Inverter devices with parametrized widths
* Transient run
* `.meas` for:

  * tphl
  * tplh
  * tpavg = (tphl + tplh)/2
  * idd_low, idd_high
  * pstatic_low, pstatic_high, pstatic_avg

Earlier bug we hit:

* We generated `.meas` using braces `{}` and it broke ngspice parsing.
* We fixed the netlist generator so ngspice accepts it.

### Step B: Smoke test one simulation

```bash
uv run python -m main.smoke_test --wn 0.5 --wp 1.0
```

Expected output example:

* tphl, tplh, tpavg
* pstatic_avg
* idd_low, idd_high
* log path

Earlier bug we hit:

* Measures were in the log with extra fields like `targ=` and `trig=`.
* Our parser regex was too strict and missed them.
* We fixed `_MEAS_RE` to capture the value even if text follows it.

### Step C: Produce raw waveform for plotting

```bash
uv run python -m main.smoke_test --wn 0.5 --wp 1.0 --raw
```

This produces a `.raw` file that can be loaded by the GUI waveform viewer.

### Step D: Run training in CLI (parallel workers)

Example:

```bash
uv run python -m main.optimize_inv --n_envs 6 --timesteps 400000 --patience 60000
```

What it does:

* Spawns `n_envs` worker processes
* Each worker repeatedly:

  * Samples action (wn, wp)
  * Generates unique netlist
  * Runs ngspice in batch mode
  * Extracts measures
  * Computes reward
* The callback writes live files:

  * `status.json` with timesteps, fps, errors, best
  * `best.json` updated on improvements
  * `history.csv` snapshots of best over time
  * checkpoints in `checkpoints/`

Stop conditions:

* Plateau stop after `--patience` steps with no best improvement
* Hard stop at `--timesteps`

### Step E: Run the GUI (best user experience)

```bash
uv run python -m main.gui_app
```

GUI features:

* Select weights for delay, static power, area
* Set width bounds
* Set number of parallel envs
* Set timesteps and plateau patience
* Start and stop training
* Live plots:

  * best_reward vs step
  * best_tpavg vs step
  * best_pstatic_avg vs step
  * Pareto-style scatter tpavg vs pstatic
* Backend logs tab
* “Simulate best + waveform” button:

  * runs ngspice for the best wn/wp
  * loads `.raw`
  * plots vin and vout

## 6) Multi-objective logic

We do not truly optimize three objectives at once directly. We convert them into a scalar reward using weights.

Metrics:

* delay metric: `tpavg`
* static metric: `pstatic_avg`
* area metric: `area_um = wn_um + wp_um`

Reward is a weighted sum of normalized metrics:

* Normalize against a reference point (first point of an episode)
* Penalize larger delay, power, area
* Reward improves when these go down relative to reference

Weights:

* `w_delay`
* `w_pstatic`
* `w_area`

You control them in:

* CLI flags
* GUI spinboxes

Practical weight presets:

* Speed first:

  * delay weight high
  * area low
  * power medium
* Leakage first:

  * pstatic weight high
  * delay medium
  * area low
* Compact:

  * area weight higher
  * delay and power moderate

## 7) Static power measurement plan

For inverter:

* Two DC states exist for input, low and high
* We measure supply current in each state and compute pstatic_avg

For NOR2 and above:

* We must test all input combinations
* Between each successive input vector, only one bit changes
* That is a Gray-code style traversal

Example for 2 inputs:

* 00 → 01 → 11 → 10

This reduces switching complexity and helps attribute static conditions cleanly.

This is the next cell extension milestone.

## 8) Why we enforce width minimum around 0.42 µm

We observed real failures when W became too small during training.
Example:

* W dropped to 0.15 µm
* ngspice failed with “could not find a valid modelname”

So we set safe bounds:

* `wn_min >= 0.42`
* `wp_min >= 0.42`

This is a practical stability guard for this setup.

## 9) File-by-file explanation

### `main/config.py`

Purpose:

* Central config for paths and tools
* Reads environment variables, especially:

  * `SKY130_LIB_SPICE`
  * ngspice binary path if needed

Typical fields:

* `project_root`
* `ngspice_bin`
* `sky130_lib_spice`

Common failure:

* `SKY130_LIB_SPICE` not set or wrong path
* Fix: export correct absolute path

### `main/netlist_gen.py`

Purpose:

* Generate characterization netlists from parameters
* Takes `--cell inv` and widths
* Writes a `.cir` netlist

Key details:

* Avoid ngspice syntax that breaks parsing
* Use correct `.meas` statements for tphl and tplh
* Define tpavg as average

In the robust parallel setup, we generate netlists inside worker folders through `spice_runner.py`.

### `main/spice_runner.py`

Purpose:

* Run ngspice in batch mode for one simulation
* Parse measures from ngspice log
* Optionally write `.raw` waveform

Key stability features we added:

* Each simulation writes to a unique path
* Uses `RUN_DIR` environment variable for run isolation
* Worker folder is based on `pid`
* `cwd` is set near `sky130.lib.spice` to handle relative includes

Outputs per sim:

* `inv_<id>.cir`
* `inv_<id>.log`
* `inv_<id>.raw` if enabled

### `main/smoke_test.py`

Purpose:

* One command to validate the full chain:

  * generate netlist
  * run ngspice
  * parse measures
  * print results

This is your first debugging tool if anything breaks.

### `main/reward.py`

Purpose:

* Define the reward model
* Define:

  * `RewardWeights`
  * `NormRef`
  * `compute_reward(...)`

Core idea:

* reward is better when tpavg, pstatic_avg, area go down
* weights control which trade-off the optimizer learns

### `main/rl_env.py`

Purpose:

* Gymnasium environment for the inverter
* Action:

  * `[wn_um, wp_um]`
* One step:

  * run one simulation
  * compute reward
  * return terminal transition (one-step episode)

Important guardrails:

* Clip wn/wp to safe ranges
* Catch ngspice failures and return a strong negative reward instead of crashing
* Track best point seen in that environment instance

### `main/optimize_inv.py`

Purpose:

* Main training script for overnight runs
* Uses:

  * `SubprocVecEnv` for parallel workers
  * `spawn` multiprocessing method
  * PPO from stable-baselines3

Live monitoring outputs:

* `status.json`
* `best.json`
* `history.csv`
* checkpoints

Also fixes SB3 warning:

* batch_size chosen to divide `n_steps * n_envs` when possible

### `main/gui_app.py`

Purpose:

* Professional GUI front end
* Launches backend training as a subprocess
* Monitors run files live and updates plots
* Adds waveform plotting for best solution

Why it is robust:

* GUI never runs training inside the same process
* Backend can run overnight without freezing the UI
* Communication is file-based (status.json, best.json, history.csv)
* This avoids fragile IPC and is easy to debug

## 10) Commands cheat sheet

### Setup

```bash
cd ~/stdcell_rl_poc
uv python pin 3.12
uv sync
```

### Install deps if needed

```bash
uv add numpy pandas pyside6 pyqtgraph stable-baselines3 gymnasium jinja2 spicelib
uv sync
```

### Install SKY130 with ciel

```bash
uvx ciel ls-remote --pdk-family sky130
uvx ciel enable --pdk-family sky130 <HASH>
find ~/.ciel -type f -name "sky130.lib.spice" | head
export SKY130_LIB_SPICE="/abs/path/to/sky130.lib.spice"
```

### Smoke test

```bash
uv run python -m main.smoke_test --wn 0.5 --wp 1.0
uv run python -m main.smoke_test --wn 0.5 --wp 1.0 --raw
```

### Training CLI

```bash
uv run python -m main.optimize_inv --n_envs 6 --timesteps 400000 --patience 60000
```

### GUI

```bash
uv run python -m main.gui_app
```

### Watch progress without GUI

```bash
watch -n 2 "cat results/run_*/status.json | tail -n 40"
```

## 11) Troubleshooting

### “No module named numpy”

Cause:

* Running outside uv environment
  Fix:

```bash
uv run python -m main.gui_app
```

### “Missing measures in log”

Cause:

* parser regex too strict
  Fix:
* `_MEAS_RE` must accept trailing fields after the value
* This is already fixed in current code

### “can’t find model sky130_fd_pr__...”

Cause:

* SKY130 lib not included or wrong path
  Fix:
* ensure `SKY130_LIB_SPICE` points to a real `sky130.lib.spice`
* ensure netlist `.lib` uses an absolute path

### “could not find a valid modelname” after training starts

Cause:

* widths got too small for this setup
  Fix:
* enforce `wn_min >= 0.42` and `wp_min >= 0.42`
* do not let the policy explore below that

### Parallel collisions or weird overwrites

Cause:

* multiple workers writing the same netlist or log path
  Fix:
* use the current run isolation design:

  * RUN_DIR
  * worker_<pid>
  * unique run_id per sim

## 12) What we do next (planned)

* Add NOR2 netlist generation
* Add Gray-code input traversal for pstatic across all input combinations
* Extend GUI cell selection:

  * inv
  * nor2
* Add a clean “cell interface” so new cells plug in without touching the trainer
* Add a true Pareto front extraction from sampled points, not only best snapshots

## 13) Quick “overnight” recommendation

If you want a safe overnight run:

* Use n_envs = number of physical cores you can spare
* Use a large timesteps cap
* Use plateau stop patience to prevent wasting time

Example:

```bash
uv run python -m main.optimize_inv --n_envs 6 --timesteps 3000000 --patience 300000
```

Then inspect:

* best.json
* waveform from GUI button
* history.csv curves

```

If tu veux, je peux aussi te donner une version “README.md prêt à écrire” en une commande `cat > README.md <<'MD' ... MD` adaptée exactement à ton repo actuel, avec les noms de fichiers exacts après un `ls -R` que tu colles ici.
::contentReference[oaicite:0]{index=0}
```
