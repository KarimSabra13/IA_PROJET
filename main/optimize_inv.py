from __future__ import annotations

from typing import Dict, Any

from stable_baselines3 import PPO

from .rl_env import InverterEnv


def optimize_inverter(
    w_delay: float,
    w_power: float,
    w_area: float,
    total_timesteps: int = 3000,
) -> Dict[str, Any]:
    """Lance une optimisation RL de l'inverseur pour un jeu de poids PPA."""
    env = InverterEnv(
        w_delay=w_delay,
        w_power=w_power,
        w_area=w_area,
        max_steps=40,
    )

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    best = env.get_best_solution()
    if best is None:
        raise RuntimeError("No best solution found during RL training.")

    return best


def main() -> None:
    best = optimize_inverter(w_delay=1.0, w_power=1.0, w_area=1.0, total_timesteps=1000)

    Wn = best["Wn_m"]
    Wp = best["Wp_m"]
    ppa = best["ppa"]
    reward = best["reward"]

    print("=== Best solution (RL) ===")
    print(f"Wn = {Wn * 1e6:.3f} um")
    print(f"Wp = {Wp * 1e6:.3f} um")
    print(f"Reward = {reward:.4f}")
    print(f"tpavg = {ppa['tpavg'] * 1e12:.2f} ps")
    print(f"Pstatic = {ppa['Pstatic'] * 1e12:.4f} pW")
    print(f"Area* = {ppa['area'] * 1e12:.4f} um-equivalent^2")


if __name__ == "__main__":
    main()
