from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .inverter_spice import measure_ppa


@dataclass
class PPATargets:
    delay_ref: float
    power_ref: float
    area_ref: float


class InverterEnv(gym.Env):
    """
    Env RL pour optimiser (wn, wp) en microns d'un inverseur SKY130.

    Action: Box[2] -> [wn_um, wp_um]
    Observation: [wn_norm, wp_norm, delay_norm, power_norm, area_norm]
    Reward: - (w_delay * d_norm + w_power * p_norm + w_area * a_norm)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        w_delay: float = 1.0,
        w_power: float = 1.0,
        w_area: float = 1.0,
        max_steps: int = 20,
    ) -> None:
        super().__init__()

        self.WN_MIN = 0.24
        self.WN_MAX = 5.0
        self.WP_MIN = 0.48
        self.WP_MAX = 10.0

        self.action_space = spaces.Box(
            low=np.array([self.WN_MIN, self.WP_MIN], dtype=np.float32),
            high=np.array([self.WN_MAX, self.WP_MAX], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=5.0,
            shape=(5,),
            dtype=np.float32,
        )

        self.w_delay = float(w_delay)
        self.w_power = float(w_power)
        self.w_area = float(w_area)
        self.max_steps = int(max_steps)

        self._step_count = 0
        self._wn = 0.42
        self._wp = 0.84
        self._targets: PPATargets | None = None
        self._rng = np.random.default_rng()
        self._best: Dict[str, Any] | None = None

    # ---------------- Helpers ----------------

    def _clip_widths(self, wn: float, wp: float) -> Tuple[float, float]:
        wn_c = float(np.clip(wn, self.WN_MIN, self.WN_MAX))
        wp_c = float(np.clip(wp, self.WP_MIN, self.WP_MAX))
        return wn_c, wp_c

    def _norm_width(self, wn: float, wp: float) -> Tuple[float, float]:
        wn_norm = (wn - self.WN_MIN) / (self.WN_MAX - self.WN_MIN)
        wp_norm = (wp - self.WP_MIN) / (self.WP_MAX - self.WP_MIN)
        return float(wn_norm), float(wp_norm)

    def _compute_ppa(self, wn: float, wp: float) -> Dict[str, float]:
        data = measure_ppa(wn, wp)

        tpavg = data["tpavg"]
        pstatic = data["pstatic"]
        area_um = data["area_um"]

        if self._targets is None:
            self._targets = PPATargets(
                delay_ref=max(tpavg, 1e-15),
                power_ref=max(pstatic, 1e-15),
                area_ref=max(area_um, 1e-6),
            )

        d_ref = self._targets.delay_ref
        p_ref = self._targets.power_ref
        a_ref = self._targets.area_ref

        return {
            "tphl": data["tphl"],
            "tplh": data["tplh"],
            "tpavg": tpavg,
            "pstatic": pstatic,
            "area_um": area_um,
            "delay_norm": tpavg / d_ref,
            "power_norm": pstatic / p_ref,
            "area_norm": area_um / a_ref,
        }

    def _make_obs(self, wn: float, wp: float, ppa: Dict[str, float]) -> np.ndarray:
        wn_norm, wp_norm = self._norm_width(wn, wp)
        return np.array(
            [
                wn_norm,
                wp_norm,
                ppa["delay_norm"],
                ppa["power_norm"],
                ppa["area_norm"],
            ],
            dtype=np.float32,
        )

    def _compute_reward(self, ppa: Dict[str, float]) -> float:
        d = ppa["delay_norm"]
        p = ppa["power_norm"]
        a = ppa["area_norm"]

        wsum = self.w_delay + self.w_power + self.w_area
        if wsum <= 0:
            wd = wp = wa = 1.0 / 3.0
        else:
            wd = self.w_delay / wsum
            wp = self.w_power / wsum
            wa = self.w_area / wsum

        obj = wd * d + wp * p + wa * a
        return -float(obj)

    def get_best(self) -> Dict[str, Any] | None:
        return self._best

    # ---------------- API Gymnasium ----------------

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._targets = None
        self._best = None

        self._wn = float(self._rng.uniform(0.3, 1.2))
        self._wp = float(self._rng.uniform(0.6, 2.4))
        self._wn, self._wp = self._clip_widths(self._wn, self._wp)

        ppa = self._compute_ppa(self._wn, self._wp)
        obs = self._make_obs(self._wn, self._wp, ppa)
        info: Dict[str, Any] = {"wn_um": self._wn, "wp_um": self._wp, "ppa": ppa}
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        wn, wp = float(action[0]), float(action[1])
        self._wn, self._wp = self._clip_widths(wn, wp)

        ppa = self._compute_ppa(self._wn, self._wp)
        obs = self._make_obs(self._wn, self._wp, ppa)
        reward = self._compute_reward(ppa)

        terminated = bool(self._step_count >= self.max_steps)
        truncated = False

        info: Dict[str, Any] = {
            "wn_um": self._wn,
            "wp_um": self._wp,
            "ppa": ppa,
        }

        if self._best is None or reward > self._best["reward"]:
            self._best = {
                "reward": reward,
                "wn_um": self._wn,
                "wp_um": self._wp,
                "ppa": ppa,
            }

        return obs, reward, terminated, truncated, info
