# main/optimize_inv.py
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

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


def _make_env_factory(
    w_delay: float,
    w_power: float,
    w_area: float,
    max_steps: int,
) -> Callable[[], InverterEnv]:
    def _init() -> InverterEnv:
        return InverterEnv(
            w_delay=w_delay,
            w_power=w_power,
            w_area=w_area,
            max_steps=max_steps,
        )

    return _init


def _choose_n_steps(n_envs: int) -> int:
    # total rollout ~128 transitions
    if n_envs <= 1:
        return 128
    return max(32, 128 // n_envs)


def _choose_batch_size(rollout_size: int) -> int:
    # choose a batch size dividing rollout_size (SB3 likes this)
    for bs in (256, 128, 64, 32, 16, 8, 4):
        if bs <= rollout_size and rollout_size % bs == 0:
            return bs
    return min(64, rollout_size)


class BestTrainCallback(BaseCallback):
    """
    Tracks best point seen during TRAINING only (no extra eval env).
    Optional early-stop by plateau/target/walltime.
    """

    def __init__(
        self,
        snapshot_interval: int = 400,
        *,
        min_delta: float = 1e-3,
        patience_snapshots: int = 8,
        warmup_snapshots: int = 3,
        target_reward: float | None = None,
        max_walltime_s: float | None = None,
    ) -> None:
        super().__init__()
        self.snapshot_interval = int(snapshot_interval)

        self.min_delta = float(min_delta)
        self.patience_snapshots = int(patience_snapshots)
        self.warmup_snapshots = int(warmup_snapshots)
        self.target_reward = None if target_reward is None else float(target_reward)
        self.max_walltime_s = None if max_walltime_s is None else float(max_walltime_s)

        self.history: List[TrainingSnapshot] = []
        self.best: Dict[str, Any] | None = None

        self._t0_wall = 0.0
        self._snap_idx = 0
        self._last_best = -1e30
        self._last_improve_snap_idx = 0

    @staticmethod
    def _pick_best_from_envs(vec_env: VecEnv) -> Optional[Dict[str, Any]]:
        try:
            bests = vec_env.env_method("get_best")
        except Exception:
            return None

        best_global: Optional[Dict[str, Any]] = None
        for b in bests:
            if b is None:
                continue
            if best_global is None or float(b["reward"]) > float(best_global["reward"]):
                best_global = b
        return best_global

    def _snapshot(self) -> None:
        b = self._pick_best_from_envs(self.training_env)
        if b is None:
            return

        ppa = b.get("ppa", {})
        snap = TrainingSnapshot(
            step=int(self.num_timesteps),
            reward=float(b["reward"]),
            wn_um=float(b["wn_um"]),
            wp_um=float(b["wp_um"]),
            tpavg_s=float(ppa.get("tpavg", float("nan"))),
            pstatic_w=float(ppa.get("pstatic", float("nan"))),
            area_um=float(ppa.get("area_um", float("nan"))),
        )
        self.history.append(snap)
        self._snap_idx += 1

        if self.best is None or float(b["reward"]) > float(self.best["reward"]):
            self.best = b

        # stop if target reached
        if self.target_reward is not None and snap.reward >= self.target_reward:
            self.model.stop_training = True
            return

        # plateau stop
        improved = (snap.reward - self._last_best) >= self.min_delta
        if improved:
            self._last_best = snap.reward
            self._last_improve_snap_idx = self._snap_idx

        if self._snap_idx >= self.warmup_snapshots:
            no_improve = self._snap_idx - self._last_improve_snap_idx
            if no_improve >= self.patience_snapshots:
                self.model.stop_training = True

    def _on_training_start(self) -> None:
        self._t0_wall = time.time()

    def _on_step(self) -> bool:
        # walltime stop
        if self.max_walltime_s is not None and (time.time() - self._t0_wall) >= self.max_walltime_s:
            self.model.stop_training = True
            return False

        if self.snapshot_interval > 0 and self.num_timesteps > 0 and self.num_timesteps % self.snapshot_interval == 0:
            self._snapshot()
        return True

    def _on_training_end(self) -> None:
        self._snapshot()


def optimize_inverter(
    w_delay: float,
    w_power: float,
    w_area: float,
    *,
    total_timesteps: int = 4000,
    max_steps: int = 40,
    n_envs: int = 6,
    snapshot_interval: int = 400,
    seed: int | None = None,
    start_method: str = "spawn",
    # early stop
    min_delta: float = 1e-3,
    patience_snapshots: int = 8,
    warmup_snapshots: int = 3,
    target_reward: float | None = None,
    max_walltime_s: float | None = None,
) -> Dict[str, Any]:
    factory = _make_env_factory(w_delay, w_power, w_area, max_steps)

    if n_envs > 1:
        env: VecEnv = SubprocVecEnv([factory for _ in range(n_envs)], start_method=start_method)
    else:
        env = DummyVecEnv([factory])

    n_steps = _choose_n_steps(n_envs)
    rollout_size = n_steps * max(1, n_envs)
    batch_size = _choose_batch_size(rollout_size)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=3e-4,
        seed=seed,
    )

    callback = BestTrainCallback(
        snapshot_interval=snapshot_interval,
        min_delta=min_delta,
        patience_snapshots=patience_snapshots,
        warmup_snapshots=warmup_snapshots,
        target_reward=target_reward,
        max_walltime_s=max_walltime_s,
    )

    t0 = time.perf_counter()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    finally:
        env.close()
    t1 = time.perf_counter()

    best = callback.best or {"reward": float("nan"), "wn_um": float("nan"), "wp_um": float("nan"), "ppa": {}}

    return {
        "best": best,
        "history": callback.history,
        "training_time_s": t1 - t0,
        "n_envs": n_envs,
        "total_timesteps": total_timesteps,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "weights": {"delay": w_delay, "power": w_power, "area": w_area},
    }


def main() -> None:
    summary = optimize_inverter(
        w_delay=1.0,
        w_power=1.0,
        w_area=1.0,
        total_timesteps=4000,
        n_envs=6,
        max_steps=40,
        snapshot_interval=400,
        # early stop plateau
        min_delta=1e-3,
        patience_snapshots=8,
        warmup_snapshots=3,
        # stop after 30 minutes max
        max_walltime_s=30 * 60,
        start_method="spawn",
    )

    best = summary["best"]
    ppa = best.get("ppa", {})

    print("\n=== Best solution (TRAIN) ===")
    print(f"Wn = {float(best['wn_um']):.3f} µm")
    print(f"Wp = {float(best['wp_um']):.3f} µm")
    print(f"Reward = {float(best['reward']):.6f}")
    if ppa:
        print(f"tpavg   = {ppa['tpavg'] * 1e12:.3f} ps")
        print(f"pstatic = {ppa['pstatic'] * 1e12:.6f} pW")
        print(f"area*   = {ppa['area_um']:.6f} µm (wn + wp)")

    print("\n=== Run info ===")
    print(f"training_time_s = {summary['training_time_s']:.2f}")
    print(f"n_envs          = {summary['n_envs']}")
    print(f"n_steps         = {summary['n_steps']}")
    print(f"batch_size      = {summary['batch_size']}")
    print(f"total_timesteps = {summary['total_timesteps']}")


if __name__ == "__main__":
    # IMPORTANT: force spawn (libngspice is NOT fork-safe)
    mp.set_start_method("spawn", force=True)
    main()
