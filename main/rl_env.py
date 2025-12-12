from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .inverter_spice import InverterSpiceRunner


@dataclass
class PPATargets:
    delay_ref: float
    power_ref: float
    area_ref: float


class InverterEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        w_delay: float = 1.0,
        w_power: float = 1.0,
        w_area: float = 1.0,
        max_steps: int = 40,
        *,
        restart_every: int = 50,
        sim_fail_penalty: float = -1_000.0,
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
        self.observation_space = spaces.Box(low=0.0, high=5.0, shape=(5,), dtype=np.float32)

        self.w_delay = float(w_delay)
        self.w_power = float(w_power)
        self.w_area = float(w_area)
        wsum = self.w_delay + self.w_power + self.w_area
        if wsum <= 0:
            self._wd = self._wpw = self._wa = 1.0 / 3.0
        else:
            self._wd = self.w_delay / wsum
            self._wpw = self.w_power / wsum
            self._wa = self.w_area / wsum

        self.max_steps = int(max_steps)
        self.sim_fail_penalty = float(sim_fail_penalty)

        self._step_count = 0
        self._wn = 0.42
        self._wp = 0.84
        self._targets: PPATargets | None = None
        self._rng = np.random.default_rng()
        self._best: Dict[str, Any] | None = None

        # IMPORTANT: in-proc runner (no child process) -> compatible with SubprocVecEnv
        self._spice = InverterSpiceRunner(restart_every=restart_every, debug=False)

    def _clip_widths(self, wn: float, wp: float) -> Tuple[float, float]:
        return float(np.clip(wn, self.WN_MIN, self.WN_MAX)), float(np.clip(wp, self.WP_MIN, self.WP_MAX))

    def _norm_width(self, wn: float, wp: float) -> Tuple[float, float]:
        wn_norm = (wn - self.WN_MIN) / (self.WN_MAX - self.WN_MIN)
        wp_norm = (wp - self.WP_MIN) / (self.WP_MAX - self.WP_MIN)
        return float(wn_norm), float(wp_norm)

    def _default_ppa(self, wn: float, wp: float) -> Dict[str, float]:
        area_um = float(wn + wp)
        return {
            "tphl": 1.0,
            "tplh": 1.0,
            "tpavg": 1.0,
            "pstatic": 1.0,
            "area_um": area_um,
            "delay_norm": 1.0,
            "power_norm": 1.0,
            "area_norm": 1.0,
            "sim_ok": 0.0,
        }

    def _compute_ppa(self, wn: float, wp: float) -> Dict[str, float]:
        try:
            data = self._spice.measure(wn, wp)
        except Exception:
            return self._default_ppa(wn, wp)

        tpavg = float(data["tpavg"])
        pstatic = float(data["pstatic"])
        area_um = float(data["area_um"])

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
            "tphl": float(data["tphl"]),
            "tplh": float(data["tplh"]),
            "tpavg": tpavg,
            "pstatic": pstatic,
            "area_um": area_um,
            "delay_norm": tpavg / d_ref,
            "power_norm": pstatic / p_ref,
            "area_norm": area_um / a_ref,
            "sim_ok": 1.0,
        }

    def _make_obs(self, wn: float, wp: float, ppa: Dict[str, float]) -> np.ndarray:
        wn_norm, wp_norm = self._norm_width(wn, wp)
        return np.array([wn_norm, wp_norm, ppa["delay_norm"], ppa["power_norm"], ppa["area_norm"]], dtype=np.float32)

    def _compute_reward(self, ppa: Dict[str, float]) -> float:
        if float(ppa.get("sim_ok", 1.0)) < 0.5:
            return float(self.sim_fail_penalty)
        d = float(ppa["delay_norm"])
        p = float(ppa["power_norm"])
        a = float(ppa["area_norm"])
        return -float(self._wd * d + self._wpw * p + self._wa * a)

    def get_best(self) -> Dict[str, Any] | None:
        return self._best

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
        return obs, {"wn_um": self._wn, "wp_um": self._wp, "ppa": ppa}

    def step(self, action: np.ndarray):
        self._step_count += 1
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        self._wn, self._wp = self._clip_widths(float(a[0]), float(a[1]))

        ppa = self._compute_ppa(self._wn, self._wp)
        obs = self._make_obs(self._wn, self._wp, ppa)
        reward = self._compute_reward(ppa)

        terminated = bool(self._step_count >= self.max_steps)
        truncated = bool(float(ppa.get("sim_ok", 1.0)) < 0.5)

        info: Dict[str, Any] = {"wn_um": self._wn, "wp_um": self._wp, "ppa": ppa}

        if not truncated:
            if self._best is None or float(reward) > float(self._best["reward"]):
                self._best = {"reward": float(reward), "wn_um": float(self._wn), "wp_um": float(self._wp), "ppa": ppa}

        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self._spice.close()
        except Exception:
            pass
        return super().close()
