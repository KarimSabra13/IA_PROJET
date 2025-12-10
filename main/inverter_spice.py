from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from pyngs.core import NGSpiceInstance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INV_CHAR_NETLIST = PROJECT_ROOT / "spice" / "inv_char.cir"


def _run_measures(
    parameters: Dict[str, float],
    measure_names: list[str],
) -> Dict[str, float]:
    """
    Charge inv_char.cir, applique des paramètres, lance ngspice et lit les mesures.
    """
    print(f"[DEBUG] Loading netlist: {INV_CHAR_NETLIST}")
    inst = NGSpiceInstance()
    inst.load(INV_CHAR_NETLIST)

    for name, value in parameters.items():
        print(f"[DEBUG] set_parameter {name} = {value}")
        inst.set_parameter(name, value)

    print("[DEBUG] Running ngspice simulation...")
    inst.run()

    out: Dict[str, float] = {}
    for m in measure_names:
        val = inst.get_measure(m)
        print(f"[DEBUG] get_measure('{m}') = {val}")
        out[m] = float(val)

    inst.stop()
    print("[DEBUG] Stopped NGSpiceInstance")
    return out


def measure_ppa(
    wn_um: float,
    wp_um: float,
    vdd: float = 1.8,
    lch_um: float = 0.15,
    k_area: float = 1.0,
) -> Dict[str, Any]:
    """
    Mesure tphl, tplh, tpavg, ileak, pstatic à partir d'un seul netlist.
    """
    params = {
        "wn": wn_um,
        "wp": wp_um,
        "vdd": vdd,
        "lch": lch_um,
    }

    measures = ["tphl", "tplh", "tpavg", "ileak", "pstatic"]
    res = _run_measures(params, measures)

    # Aire estimate: proportionnelle à wn + wp
    res["area_um"] = k_area * (wn_um + wp_um)
    res["wn_um"] = wn_um
    res["wp_um"] = wp_um
    return res


def main() -> None:
    wn = 0.42
    wp = 0.84

    print("=== Inverter SKY130 – full PPA via pyngs ===")
    print(f"Wn = {wn:.3f} µm, Wp = {wp:.3f} µm\n")

    ppa = measure_ppa(wn, wp)

    print("\n=== Results ===")
    print("Delay (ps):")
    print(f"  tphl  = {ppa['tphl'] * 1e12:.3f} ps")
    print(f"  tplh  = {ppa['tplh'] * 1e12:.3f} ps")
    print(f"  tpavg = {ppa['tpavg'] * 1e12:.3f} ps\n")

    print("Static power / leakage:")
    print(f"  ileak   = {ppa['ileak'] * 1e12:.6f} pA")
    print(f"  pstatic = {ppa['pstatic'] * 1e12:.6f} pW")

    print("\nArea estimate:")
    print(f"  area*   = {ppa['area_um']:.3f} µm (wn + wp)")


if __name__ == "__main__":
    main()

