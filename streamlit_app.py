from __future__ import annotations

import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from main.optimize_inv import TrainingSnapshot, optimize_inverter

st.set_page_config(page_title="Standard Cell Optimizer", layout="wide")

st.title("Standard Cell Optimization – SKY130 / ngspice / RL")

# ---------- UI helpers ----------

def history_to_df(history: List[TrainingSnapshot]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [h.step for h in history],
            "reward": [h.reward for h in history],
            "tpavg_ps": [h.tpavg_s * 1e12 for h in history],
            "pstatic_pW": [h.pstatic_w * 1e12 for h in history],
            "area_um": [h.area_um for h in history],
            "wn_um": [h.wn_um for h in history],
            "wp_um": [h.wp_um for h in history],
            "elapsed_s": [h.elapsed_s for h in history],
        }
    )


def render_live(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if df.empty:
        st.info("En attente du premier snapshot…")
        return None, None, None

    last = df.iloc[-1]
    best_reward = float(df["reward"].max())

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Last step", int(last["step"]))
    colB.metric("Best reward (snapshots)", f"{best_reward:.6f}")
    colC.metric("Last Wn (µm)", f"{last['wn_um']:.3f}")
    colD.metric("Last Wp (µm)", f"{last['wp_um']:.3f}")

    colE, colF = st.columns(2)
    colE.metric("Last elapsed", f"{float(last['elapsed_s']):.1f} s")
    colF.metric("Snapshots", len(df))

    fig_reward = px.line(df, x="step", y="reward", markers=True, title="Best reward (snapshot) vs step")
    fig_reward.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_reward, use_container_width=True)

    fig_delay = px.line(df, x="step", y="tpavg_ps", markers=True, title="tpavg (ps) vs step")
    fig_delay.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_delay, use_container_width=True)

    fig_pstatic = px.line(df, x="step", y="pstatic_pW", markers=True, title="Pstatic (pW) vs step")
    fig_pstatic.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_pstatic, use_container_width=True)

    return best_reward, float(last["step"]), float(last["elapsed_s"])


# ---------- Sidebar ----------

with st.sidebar:
    st.header("PPA weights")
    w_delay = st.slider("Delay weight", 0.0, 1.0, 0.5, 0.05)
    w_power = st.slider("Static power weight", 0.0, 1.0, 0.3, 0.05)
    w_area = st.slider("Area weight", 0.0, 1.0, 0.2, 0.05)

    st.header("RL settings")
    total_timesteps = st.number_input("Total timesteps", 200, 50000, 4000, 200)
    max_steps = st.number_input("Max steps per episode", 5, 200, 40, 5)

    # IMPORTANT: keep n_envs=1 for now (stable with PyngsWorker)
    n_envs = 1
    st.caption("Parallel envs: locked to 1 (PyngsWorker stability).")

    snapshot_interval = st.number_input("Snapshot interval (timesteps)", 50, 5000, 400, 50)

    st.header("Early stopping")
    min_delta = st.number_input("Min improvement (min_delta)", value=1e-3, format="%.4g")
    patience = st.number_input("Patience (snapshots)", 1, 50, 8, 1)
    warmup = st.number_input("Warmup (snapshots)", 0, 20, 3, 1)
    max_walltime_min = st.number_input("Max walltime (minutes)", 0, 240, 30, 5)
    target_reward = st.text_input("Target reward (optional)", value="")

    refresh_s = st.slider("UI refresh (seconds)", 0.2, 2.0, 0.5, 0.1)

run = st.button("Lancer l'optimisation", type="primary")

# ---------- Training (live) ----------

if "training_running" not in st.session_state:
    st.session_state.training_running = False

if run and not st.session_state.training_running:
    st.session_state.training_running = True

    q: "queue.Queue[tuple[TrainingSnapshot, Dict[str, float]]]" = queue.Queue()
    history: List[TrainingSnapshot] = []
    best_live: Optional[Dict[str, float]] = None
    result_holder: Dict[str, object] = {"done": False, "summary": None, "error": None}

    def on_snapshot(snap: TrainingSnapshot, best: Dict[str, float]) -> None:
        q.put((snap, best))

    def train_thread() -> None:
        try:
            tr = optimize_inverter(
                w_delay=float(w_delay),
                w_power=float(w_power),
                w_area=float(w_area),
                total_timesteps=int(total_timesteps),
                max_steps=int(max_steps),
                n_envs=int(n_envs),
                snapshot_interval=int(snapshot_interval),
                min_delta=float(min_delta),
                patience_snapshots=int(patience),
                warmup_snapshots=int(warmup),
                max_walltime_s=(int(max_walltime_min) * 60 if int(max_walltime_min) > 0 else None),
                target_reward=(float(target_reward) if target_reward.strip() else None),
                on_snapshot=on_snapshot,
            )
            result_holder["summary"] = tr
        except Exception as e:
            result_holder["error"] = str(e)
        finally:
            result_holder["done"] = True

    th = threading.Thread(target=train_thread, daemon=True)
    th.start()

    # Live placeholders
    top = st.empty()
    prog = st.progress(0)
    live_area = st.empty()

    t0 = time.time()

    while not result_holder["done"] or not q.empty():
        # drain queue
        drained = 0
        while True:
            try:
                snap, _best = q.get_nowait()
                history.append(snap)
                drained += 1
                if _best and (best_live is None or float(_best.get("reward", -1e30)) > float(best_live.get("reward", -1e30))):
                    best_live = _best
            except queue.Empty:
                break

        elapsed = time.time() - t0
        best_reward_str = f" · best reward: **{best_live['reward']:.4f}**" if best_live else ""
        top.markdown(
            f"### Training running…  \nElapsed: **{elapsed:.1f}s** · snapshots: **{len(history)}**{best_reward_str}"
        )

        if history:
            df = history_to_df(history)
            best_reward, last_step, _last_elapsed = render_live(df)
            if last_step is not None:
                prog.progress(min(1.0, float(last_step) / float(total_timesteps)))

        time.sleep(float(refresh_s))
        live_area.empty()
        with live_area.container():
            if history:
                df = history_to_df(history)
                render_live(df)
            else:
                st.info("Waiting for first snapshot…")

    st.session_state.training_running = False

    if result_holder["error"] is not None:
        st.error(f"Training failed: {result_holder['error']}")
        st.stop()

    summary = result_holder["summary"]
    assert isinstance(summary, dict)

    best = summary["best"]
    ppa: Dict[str, float] = best.get("ppa", {})

    st.success("Training terminé.")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Best widths & reward")
        st.metric("Wn", f"{best['wn_um']:.3f} µm")
        st.metric("Wp", f"{best['wp_um']:.3f} µm")
        st.metric("Reward", f"{best['reward']:.6f}")
        st.caption(f"Temps: {summary['training_time_s']:.1f} s")

    with col2:
        st.subheader("PPA metrics")
        st.metric("tpavg", f"{ppa.get('tpavg', float('nan')) * 1e12:.2f} ps")
        st.metric("Pstatic", f"{ppa.get('pstatic', float('nan')) * 1e12:.6f} pW")
        st.metric("Area*", f"{ppa.get('area_um', float('nan')):.6f} µm (wn + wp)")
