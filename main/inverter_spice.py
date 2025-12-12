from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from pyngs.core import NGSpiceInstance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INV_CHAR_NETLIST = PROJECT_ROOT / "spice" / "inv_char.cir"
MEASURES = ("tphl", "tplh", "tpavg", "ileak", "pstatic")


class InverterSpiceRunner:
    """
    In-process runner, compatible SubprocVecEnv.
    Key: each process gets its own workdir to avoid ngspice temp/raw collisions.
    """

    def __init__(
        self,
        netlist_path: Path = INV_CHAR_NETLIST,
        *,
        restart_every: int = 25,
        debug: bool = False,
    ) -> None:
        self.netlist_path = Path(netlist_path)
        self.restart_every = int(restart_every)
        self.debug = bool(debug)

        self._tmp: Optional[tempfile.TemporaryDirectory] = None
        self._inst: Optional[NGSpiceInstance] = None
        self._jobs = 0

        self._make_workdir()
        self._init()

    def _make_workdir(self) -> None:
        if self._tmp is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="pyngs_inv_")
        # isolate ngspice side-files per worker
        os.chdir(self._tmp.name)
        os.environ["TMPDIR"] = self._tmp.name

    def _init(self) -> None:
        self._make_workdir()
        if self.debug:
            print(f"[DEBUG] CWD={os.getcwd()}")
            print(f"[DEBUG] Loading netlist: {self.netlist_path}")
        self._inst = NGSpiceInstance()
        self._inst.load(self.netlist_path)
        self._jobs = 0

    def _restart(self) -> None:
        try:
            if self._inst is not None:
                self._inst.stop()
        except Exception:
            pass
        self._inst = None
        self._init()

    def measure(
        self,
        wn_um: float,
        wp_um: float,
        *,
        vdd: float = 1.8,
        lch_um: float = 0.15,
        k_area: float = 1.0,
    ) -> Dict[str, Any]:
        if self._inst is None:
            self._init()

        if self.restart_every > 0 and self._jobs >= self.restart_every:
            self._restart()

        def _run_once() -> Dict[str, Any]:
            assert self._inst is not None
            self._inst.set_parameter("wn", float(wn_um))
            self._inst.set_parameter("wp", float(wp_um))
            self._inst.set_parameter("vdd", float(vdd))
            self._inst.set_parameter("lch", float(lch_um))

            self._inst.run()

            out: Dict[str, Any] = {}
            for m in MEASURES:
                out[m] = float(self._inst.get_measure(m))

            out["area_um"] = float(k_area * (float(wn_um) + float(wp_um)))
            out["wn_um"] = float(wn_um)
            out["wp_um"] = float(wp_um)
            return out

        try:
            res = _run_once()
        except Exception:
            # hard reset + retry once
            self._restart()
            res = _run_once()

        self._jobs += 1
        return res

    def close(self) -> None:
        try:
            if self._inst is not None:
                self._inst.stop()
        except Exception:
            pass
        self._inst = None

        try:
            if self._tmp is not None:
                self._tmp.cleanup()
        except Exception:
            pass
        self._tmp = None
