from __future__ import annotations

import streamlit as st

from main.optimize_inv import optimize_inverter


st.set_page_config(page_title="Standard Cell Optimizer", layout="wide")

st.title("Standard Cell Optimization – SKY130 / ngspice / RL")

st.sidebar.header("Standard cell")
cell_type = st.sidebar.selectbox(
    "Choose cell",
    options=["Inverter (NOT)"],
    index=0,
)

st.sidebar.header("PPA weights")
w_delay = st.sidebar.slider("Weight for delay", 0.0, 1.0, 0.5, 0.05)
w_power = st.sidebar.slider("Weight for static power", 0.0, 1.0, 0.3, 0.05)
w_area = st.sidebar.slider("Weight for area", 0.0, 1.0, 0.2, 0.05)

st.sidebar.header("RL settings")
timesteps = st.sidebar.number_input("Total timesteps", 500, 20000, 3000, 500)

st.markdown(
    "Cette interface lance en arrière-plan un entraînement RL "
    "pour optimiser les largeurs Wn/Wp de la standard cell choisie."
)

if st.button("Run optimization"):
    with st.spinner("Training RL agent and running ngspice simulations..."):
        if cell_type.startswith("Inverter"):
            best = optimize_inverter(
                w_delay=float(w_delay),
                w_power=float(w_power),
                w_area=float(w_area),
                total_timesteps=int(timesteps),
            )
        else:
            st.error("Only inverter is implemented for now.")
            st.stop()

    Wn = best["Wn_m"]
    Wp = best["Wp_m"]
    ppa = best["ppa"]
    reward = best["reward"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Best widths")
        st.metric("Wn", f"{Wn * 1e6:.3f} µm")
        st.metric("Wp", f"{Wp * 1e6:.3f} µm")
        st.metric("Reward", f"{reward:.4f}")

    with col2:
        st.subheader("PPA metrics")
        st.metric("tpavg", f"{ppa['tpavg'] * 1e12:.2f} ps")
        st.metric("Pstatic", f"{ppa['Pstatic'] * 1e12:.4f} pW")
        st.metric("Area*", f"{ppa['area'] * 1e12:.4f} µm-eq²")

    st.caption(
        "Area* est une estimation proportionnelle à Wn + Wp. "
        "Les valeurs normalisées sont utilisées en interne par le RL."
    )

