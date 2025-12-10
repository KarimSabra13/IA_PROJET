from __future__ import annotations

import time
from typing import Dict, Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .rl_env import InverterEnv


def make_env() -> InverterEnv:
    return InverterEnv(w_delay=1.0, w_power=1.0, w_area=1.0, max_steps=5)


def evaluate_policy(model: PPO, n_episodes: int = 3) -> Dict[str, Any]:
    env = make_env()
    best: Dict[str, Any] | None = None

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            if best is None or reward > best["reward"]:
                best = {
                    "reward": reward,
                    "wn_um": info["wn_um"],
                    "wp_um": info["wp_um"],
                    "ppa": info["ppa"],
                }

    assert best is not None
    return best


def main() -> None:
    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=2,          # logs détaillés
        n_steps=32,         # petit pour tester
        batch_size=32,
        learning_rate=3e-4,
        device="cpu",
    )

    total_timesteps = 100  # très petit pour test

    print(f"[SEQ] Starting training for {total_timesteps} timesteps...")
    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    t1 = time.perf_counter()
    print(f"[SEQ] Training done in {t1 - t0:.2f} s")

    print("[SEQ] Evaluating policy...")
    best = evaluate_policy(model, n_episodes=3)

    print("\n=== Best design (sequential PPO) ===")
    print(f"Wn = {best['wn_um']:.3f} µm")
    print(f"Wp = {best['wp_um']:.3f} µm")
    ppa = best["ppa"]
    print(f"tpavg   = {ppa['tpavg'] * 1e12:.3f} ps")
    print(f"pstatic = {ppa['pstatic'] * 1e12:.6f} pW")
    print(f"area    = {ppa['area_um']:.3f} µm (wn + wp)")
    print(f"reward  = {best['reward']:.4f}")


if __name__ == "__main__":
    main()
