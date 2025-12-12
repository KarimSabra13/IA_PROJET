from __future__ import annotations

import __main__
import multiprocessing as mp
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from pyngs.core import NGSpiceInstance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INV_CHAR_NETLIST = PROJECT_ROOT / "spice" / "inv_char.cir"
MEASURES = ("tphl", "tplh", "tpavg", "ileak", "pstatic")


def _pick_ctx(start_method: str) -> mp.context.BaseContext:
    # spawn is safer for C libs; fallback to fork only for <stdin>
    if start_method == "auto":
        main_file = getattr(__main__, "__file__", None)
        if main_file is None or str(main_file).endswith("<stdin>"):
            try:
                return mp.get_context("fork")
            except Exception:
                return mp.get_context("spawn")
        return mp.get_context("spawn")

    try:
        return mp.get_context(start_method)
    except Exception:
        return mp.get_context("spawn")


def _install_warning_policy() -> None:
    # Convert ONLY the problematic message into an exception
    warnings.filterwarnings(
        "error",
        message=r".*circuit not parsed.*",
        category=RuntimeWarning,
    )


def _worker_loop(conn, netlist_path: str, restart_every: int) -> None:
    _install_warning_policy()

    inst: Optional[NGSpiceInstance] = None
    jobs = 0

    def _new_instance() -> NGSpiceInstance:
        i = NGSpiceInstance()
        i.load(Path(netlist_path))
        return i

    inst = _new_instance()

    while True:
        msg = conn.recv()
        if msg is None:
            break

        if isinstance(msg, dict) and msg.get("__cmd__") == "restart":
            # hard reset inside the process
            try:
                if inst is not None:
                    inst.stop()
            except Exception:
                pass
            inst = _new_instance()
            jobs = 0
            conn.send({"ok": True})
            continue

        wn = float(msg["wn"])
        wp = float(msg["wp"])
        vdd = float(msg.get("vdd", 1.8))
        lch = float(msg.get("lch", 0.15))
        k_area = float(msg.get("k_area", 1.0))

        try:
            if restart_every > 0 and jobs >= restart_every:
                try:
                    inst.stop()
                except Exception:
                    pass
                inst = _new_instance()
                jobs = 0

            inst.set_parameter("wn", wn)
            inst.set_parameter("wp", wp)
            inst.set_parameter("vdd", vdd)
            inst.set_parameter("lch", lch)

            inst.run()

            out: Dict[str, Any] = {m: float(inst.get_measure(m)) for m in MEASURES}
            out["area_um"] = float(k_area * (wn + wp))
            out["wn_um"] = wn
            out["wp_um"] = wp

            jobs += 1
            conn.send(out)

        except Exception as e:
            # If we get here, the instance is unreliable: force the worker to die.
            conn.send({"__error__": f"{type(e).__name__}: {e}"})
            os._exit(1)

    try:
        if inst is not None:
            inst.stop()
    except Exception:
        pass


class PyngsWorker:
    """
    pyngs/libngspice in a dedicated process.
    If it errors or hangs -> kill + restart.
    """

    def __init__(
        self,
        netlist_path: Path = INV_CHAR_NETLIST,
        *,
        timeout_s: float = 10.0,
        restart_every: int = 25,
        start_method: str = "auto",
    ) -> None:
        self.netlist_path = Path(netlist_path)
        self.timeout_s = float(timeout_s)
        self.restart_every = int(restart_every)

        self._ctx = _pick_ctx(start_method)
        self._parent_conn, self._child_conn = self._ctx.Pipe()
        self._proc: Optional[mp.Process] = None
        self._start()

    def _start(self) -> None:
        self._proc = self._ctx.Process(
            target=_worker_loop,
            args=(self._child_conn, str(self.netlist_path), self.restart_every),
            daemon=True,
        )
        self._proc.start()

    def _kill(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.is_alive():
                self._proc.kill()
        except Exception:
            pass
        try:
            self._proc.join(timeout=0.5)
        except Exception:
            pass
        self._proc = None

    def _restart_proc(self) -> None:
        self.close()
        self._parent_conn, self._child_conn = self._ctx.Pipe()
        self._start()

    def restart(self) -> None:
        if self._proc is None or not self._proc.is_alive():
            self._restart_proc()
            return
        try:
            self._parent_conn.send({"__cmd__": "restart"})
            if not self._parent_conn.poll(self.timeout_s):
                self._kill()
                self._restart_proc()
                return
            _ = self._parent_conn.recv()
        except Exception:
            self._restart_proc()

    def measure(
        self,
        wn_um: float,
        wp_um: float,
        *,
        vdd: float = 1.8,
        lch_um: float = 0.15,
        k_area: float = 1.0,
        _retry: bool = True,
    ) -> Dict[str, Any]:
        if self._proc is None or not self._proc.is_alive():
            self._restart_proc()

        req = {
            "wn": float(wn_um),
            "wp": float(wp_um),
            "vdd": float(vdd),
            "lch": float(lch_um),
            "k_area": float(k_area),
        }

        try:
            self._parent_conn.send(req)
        except Exception:
            self._restart_proc()
            self._parent_conn.send(req)

        if not self._parent_conn.poll(self.timeout_s):
            self._kill()
            self._restart_proc()
            if _retry:
                return self.measure(wn_um, wp_um, vdd=vdd, lch_um=lch_um, k_area=k_area, _retry=False)
            raise RuntimeError("pyngs worker timeout (stuck ngspice)")

        res = self._parent_conn.recv()

        if isinstance(res, dict) and "__error__" in res:
            self._kill()
            self._restart_proc()
            if _retry:
                return self.measure(wn_um, wp_um, vdd=vdd, lch_um=lch_um, k_area=k_area, _retry=False)
            raise RuntimeError(res["__error__"])

        return res

    def close(self) -> None:
        try:
            if self._proc is not None and self._proc.is_alive():
                try:
                    self._parent_conn.send(None)
                except Exception:
                    pass
                try:
                    self._proc.join(timeout=0.5)
                except Exception:
                    pass
                if self._proc.is_alive():
                    self._kill()
        finally:
            self._proc = None
            try:
                self._parent_conn.close()
            except Exception:
                pass
            try:
                self._child_conn.close()
            except Exception:
                pass
