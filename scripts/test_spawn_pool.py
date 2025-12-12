# scripts/test_spawn_pool.py
from __future__ import annotations

import multiprocessing as mp
import os

from main.inverter_spice import InverterSpiceRunner


def one(_):
    r = InverterSpiceRunner(restart_every=25, debug=False)
    out = r.measure(0.5, 1.0)
    r.close()
    return (os.getpid(), out["tpavg"])


def main():
    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(6) as p:
        vals = p.map(one, range(6))
    print("OK:", vals)


if __name__ == "__main__":
    main()
