from __future__ import annotations

import time
from typing import Callable, Dict, Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from .rl_env import InverterEnv


def make_env_fn(rank: int) -> Callable[[], InverterEnv]:
    def _init() -> InverterEnv:
        # on peut varier le seed si on veut
        return InverterEnv(w_delay=1.0, w_power=1.0, w_area=1.0, max_steps=20)
    return _init


def evaluate_policy(model: PPO, n_episodes: int = 10) -> Dict[str, Any]:
    env = InverterEnv(w_delay=1.0, w_power=1.0, w_area=1.0, max_steps=20)
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
    n_envs = 4

    env = SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])

    # n_steps est par env, total n_steps*n_envs par update
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256 // n_envs,
        batch_size=64,
        learning_rate=3e-4,
	device="cpu",
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=2000)
    t1 = time.perf_counter()

    print(f"\n[PAR] Training time with {n_envs} envs: {t1 - t0:.2f} s\n")

    best = evaluate_policy(model, n_episodes=10)

    print(f"=== Best design (parallel PPO, {n_envs} envs) ===")
    print(f"Wn = {best['wn_um']:.3f} µm")
    print(f"Wp = {best['wp_um']:.3f} µm")
    ppa = best["ppa"]
    print(f"tpavg   = {ppa['tpavg'] * 1e12:.3f} ps")
    print(f"pstatic = {ppa['pstatic'] * 1e12:.6f} pW")
    print(f"area    = {ppa['area_um']:.3f} µm (wn + wp)")
    print(f"reward  = {best['reward']:.4f}")


if __name__ == "__main__":
    main()
