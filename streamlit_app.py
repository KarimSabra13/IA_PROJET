from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from main.optimize_inv import TrainingSnapshot, optimize_inverter


st.set_page_config(page_title="Standard Cell Optimizer", layout="wide")

st.title("Standard Cell Optimization – SKY130 / ngspice / RL")
st.write(
    "Optimisation assistée par RL (PPO) des largeurs Wn/Wp d'un inverseur SKY130, "
    "avec mesures ngspice/pyngs pour les métriques PPA."
)


def _render_history(history: List[TrainingSnapshot]) -> None:
    if not history:
        st.info("Aucune évaluation intermédiaire n'a été collectée (timesteps trop courts ou intervalle d'éval trop grand).")
        return

    df = pd.DataFrame(
        {
            "step": [h.step for h in history],
            "reward": [h.reward for h in history],
            "tpavg_ps": [h.tpavg_s * 1e12 for h in history],
            "pstatic_pW": [h.pstatic_w * 1e12 for h in history],
            "area_um": [h.area_um for h in history],
        }
    )

    fig_reward = px.line(df, x="step", y="reward", markers=True, title="Reward au fil des évaluations")
    fig_reward.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_reward, use_container_width=True)

    fig_delay = px.line(df, x="step", y="tpavg_ps", markers=True, title="tpavg (ps) sur les évaluations")
    fig_delay.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_delay, use_container_width=True)


with st.sidebar:
    st.header("Standard cell")
    cell_type = st.selectbox("Choose cell", options=["Inverter (NOT)"])

    st.header("PPA weights")
    w_delay = st.slider("Weight for delay", 0.0, 1.0, 0.5, 0.05)
    w_power = st.slider("Weight for static power", 0.0, 1.0, 0.3, 0.05)
    w_area = st.slider("Weight for area", 0.0, 1.0, 0.2, 0.05)

    st.header("RL settings")
    timesteps = st.number_input("Total timesteps", 500, 50000, 5000, 500)
    max_steps = st.number_input("Max steps per episode", 5, 200, 40, 5)
    n_envs = st.number_input("Parallel envs", 1, 8, 1, 1)
    eval_interval = st.number_input("Eval interval (timesteps)", 100, 5000, 500, 100)
    eval_episodes = st.number_input("Eval episodes", 1, 10, 3, 1)

st.markdown(
    """
    Cette interface lance en arrière-plan un entraînement RL pour optimiser les largeurs Wn/Wp de la cellule choisie.
    Le PDK SKY130 doit déjà être activé via Ciel et accessible depuis les netlists ngspice.
    """
)

run = st.button("Lancer l'optimisation", type="primary")

if run:
    with st.spinner("Entraînement PPO + simulations ngspice en cours..."):
        if not cell_type.startswith("Inverter"):
            st.error("Seul l'inverseur est implémenté pour l'instant.")
            st.stop()

        summary = optimize_inverter(
            w_delay=float(w_delay),
            w_power=float(w_power),
            w_area=float(w_area),
            total_timesteps=int(timesteps),
            max_steps=int(max_steps),
            n_envs=int(n_envs),
            eval_interval=int(eval_interval),
            eval_episodes=int(eval_episodes),
        )

    best = summary["best"]
    ppa: Dict[str, float] = best.get("ppa", {})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Best widths & reward")
        st.metric("Wn", f"{best['wn_um']:.3f} µm")
        st.metric("Wp", f"{best['wp_um']:.3f} µm")
        st.metric("Reward", f"{best['reward']:.4f}")
        st.caption(f"Temps d'entraînement: {summary['training_time_s']:.1f} s · {summary['n_envs']} env(s) en parallèle")

    with col2:
        st.subheader("PPA metrics")
        st.metric("tpavg", f"{ppa.get('tpavg', float('nan')) * 1e12:.2f} ps")
        st.metric("Pstatic", f"{ppa.get('pstatic', float('nan')) * 1e12:.4f} pW")
        st.metric("Area*", f"{ppa.get('area_um', float('nan')):.4f} µm-eq")

    st.markdown("### PPA normalisés (plus bas est meilleur)")
    if ppa:
        norm_df = pd.DataFrame(
            {
                "Metric": ["Delay", "Static power", "Area"],
                "Normalized": [
                    ppa.get("delay_norm", float("nan")),
                    ppa.get("power_norm", float("nan")),
                    ppa.get("area_norm", float("nan")),
                ],
            }
        )
        fig = px.bar(norm_df, x="Metric", y="Normalized", text_auto=".3f", color="Metric")
        fig.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Impossible d'afficher les PPA normalisées : données manquantes.")

    st.markdown("### Historique d'évaluations")
    _render_history(summary.get("history", []))

    st.caption(
        "Area* est une estimation proportionnelle à Wn + Wp. Les valeurs normalisées sont utilisées en interne par le RL." \
        " L'optimisation parallèle utilise SubprocVecEnv pour distribuer les simulations ngspice."
    )

