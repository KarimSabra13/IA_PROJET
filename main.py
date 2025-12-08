from __future__ import annotations

from main.rc_analysis import sweep_cutoff
import sys
import os


def main() -> None:
    rc_values = [
        (1_000, 100e-9),
        (2_000, 100e-9),
        (1_000, 220e-9),
        (4_700, 47e-9),
    ]

    results = sweep_cutoff(rc_values)

    print("R (ohm) C (F)           f_th (Hz)               f_meas (Hz)")
    for row in results:
        print(
            f"{int(row['R']):<7d}"
            f"{row['C']:<12.2e}"
            f"{row['f_theoretical']:>10.2f} "
            f"{row['f_measured']:>10.2f}"
        )

    # On flush la sortie, puis on coupe le process pour éviter
    # le bug de finalisation (_PyThreadState_Attach)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
