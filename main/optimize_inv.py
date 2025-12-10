from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .rl_env import InverterEnv


@dataclass
class TrainingSnapshot:
    step: int
    reward: float
    wn_um: float
    wp_um: float
    tpavg_s: float
    pstatic_w: float
    area_um: float


def _make_env_factory(w_delay: float, w_power: float, w_area: float, max_steps: int) -> Callable[[], InverterEnv]:
    def _init() -> InverterEnv:
        return InverterEnv(w_delay=w_delay, w_power=w_power, w_area=w_area, max_steps=max_steps)

    return _init


class BestEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env: InverterEnv,
        eval_interval: int = 500,
        eval_episodes: int = 3,
    ) -> None:
        super().__init__()
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.history: List[TrainingSnapshot] = []
        self.best: Dict[str, Any] | None = None

    def _evaluate_once(self) -> None:
        env = self.eval_env
        local_best: Dict[str, Any] | None = None

        for _ in range(self.eval_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                if local_best is None or reward > local_best["reward"]:
                    local_best = {
                        "reward": reward,
                        "wn_um": info["wn_um"],
                        "wp_um": info["wp_um"],
                        "ppa": info["ppa"],
                    }

        if local_best is None:
            return

        snap = TrainingSnapshot(
            step=self.num_timesteps,
            reward=float(local_best["reward"]),
            wn_um=float(local_best["wn_um"]),
            wp_um=float(local_best["wp_um"]),
            tpavg_s=float(local_best["ppa"]["tpavg"]),
            pstatic_w=float(local_best["ppa"]["pstatic"]),
            area_um=float(local_best["ppa"]["area_um"]),
        )
        self.history.append(snap)

        if self.best is None or local_best["reward"] > self.best["reward"]:
            self.best = local_best

    def _on_step(self) -> bool:
        if self.eval_interval > 0 and self.num_timesteps % self.eval_interval == 0:
            self._evaluate_once()
        return True

    def _on_training_end(self) -> None:
        # Final eval to capture last policy state
        self._evaluate_once()


def optimize_inverter(
    w_delay: float,
    w_power: float,
    w_area: float,
    total_timesteps: int = 3000,
    max_steps: int = 40,
    n_envs: int = 1,
    eval_interval: int = 500,
    eval_episodes: int = 3,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Lance une optimisation RL de l'inverseur pour un jeu de poids PPA."""

    factory = _make_env_factory(w_delay, w_power, w_area, max_steps)
    if n_envs > 1:
        env = SubprocVecEnv([factory for _ in range(n_envs)])
        n_steps = max(16, 256 // n_envs)
    else:
        env = DummyVecEnv([factory])
        n_steps = 256

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        device="cpu",
        n_steps=n_steps,
        batch_size=64,
        learning_rate=3e-4,
        seed=seed,
    )

    eval_env = InverterEnv(w_delay=w_delay, w_power=w_power, w_area=w_area, max_steps=max_steps)
    callback = BestEvalCallback(eval_env=eval_env, eval_interval=eval_interval, eval_episodes=eval_episodes)

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    t1 = time.perf_counter()

    best = callback.best or eval_env.get_best() or {
        "reward": float("nan"),
        "wn_um": float("nan"),
        "wp_um": float("nan"),
        "ppa": {},
    }

    summary = {
        "best": best,
        "history": callback.history,
        "training_time_s": t1 - t0,
        "n_envs": n_envs,
        "total_timesteps": total_timesteps,
        "n_steps": n_steps,
        "weights": {"delay": w_delay, "power": w_power, "area": w_area},
    }

    env.close()
    return summary


def main() -> None:
    summary = optimize_inverter(
        w_delay=1.0,
        w_power=1.0,
        w_area=1.0,
        total_timesteps=1000,
        n_envs=1,
    )

    best = summary["best"]
    Wn = best["wn_um"]
    Wp = best["wp_um"]
    ppa = best["ppa"]
    reward = best["reward"]

    print("=== Best solution (RL) ===")
    print(f"Wn = {Wn * 1e6:.3f} um")
    print(f"Wp = {Wp * 1e6:.3f} um")
    print(f"Reward = {reward:.4f}")

    if ppa:
        print(f"tpavg = {ppa['tpavg'] * 1e12:.2f} ps")
        print(f"Pstatic = {ppa['pstatic'] * 1e12:.4f} pW")
        print(f"Area* = {ppa['area_um'] * 1e12:.4f} um-equivalent^2")


if __name__ == "__main__":
    main()
