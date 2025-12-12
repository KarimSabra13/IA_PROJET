from __future__ import annotations

import contextlib
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
    Improvements:
    - dedicated temp dir per process (no shared raw/tmp collisions)
    - avoids leaking CWD/TMPDIR outside ngspice calls
    - auto-restarts on corruption or after N jobs
    - detects post-fork reuse and re-initialises cleanly
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
        self._workdir: Optional[Path] = None
        self._inst: Optional[NGSpiceInstance] = None
        self._jobs = 0
        self._pid = os.getpid()

        self._make_workdir()
        self._init()

    def _make_workdir(self) -> None:
        if self._tmp is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="pyngs_inv_")
        self._workdir = Path(self._tmp.name)

    @contextlib.contextmanager
    def _in_workdir(self):
        prev_cwd = os.getcwd()
        prev_tmpdir = os.environ.get("TMPDIR")
        assert self._workdir is not None
        os.chdir(self._workdir)
        os.environ["TMPDIR"] = str(self._workdir)
        try:
            yield
        finally:
            os.chdir(prev_cwd)
            if prev_tmpdir is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = prev_tmpdir

    def _init(self) -> None:
        self._make_workdir()
        if self.debug:
            print(f"[DEBUG] CWD={os.getcwd()}")
            print(f"[DEBUG] workdir={self._workdir}")
            print(f"[DEBUG] Loading netlist: {self.netlist_path}")
        with self._in_workdir():
            self._inst = NGSpiceInstance()
            self._inst.load(self.netlist_path)
        self._jobs = 0
        self._pid = os.getpid()

    def _restart(self) -> None:
        try:
            if self._inst is not None:
                self._inst.stop()
        except Exception:
            pass
        self._inst = None
        self._init()

    def _ensure_proc_safe(self) -> None:
        if os.getpid() != self._pid:
            # We have been forked: never reuse the old libngspice handle
            self._restart()

    def measure(
        self,
        wn_um: float,
        wp_um: float,
        *,
        vdd: float = 1.8,
        lch_um: float = 0.15,
        k_area: float = 1.0,
    ) -> Dict[str, Any]:
        self._ensure_proc_safe()

        if self._inst is None:
            self._init()

        if self.restart_every > 0 and self._jobs >= self.restart_every:
            self._restart()

        def _run_once() -> Dict[str, Any]:
            assert self._inst is not None
            with self._in_workdir():
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
        self._workdir = None
