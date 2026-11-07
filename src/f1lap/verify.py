"""Verification: does the fitting code recover known truth, with calibrated parameter CIs?

Verification (are we solving the model right?) is distinct from validation (is it the right
model for real races?). Synthetic races with known parameters make verification possible.
"""

from __future__ import annotations

import pandas as pd

from .data import TRUE_PARAMS, synthetic_race
from .models import PhysicsLapModel


def true_values() -> dict[str, float]:
    t = TRUE_PARAMS
    return {
        "offset_SOFT": t["off"]["SOFT"],
        "offset_HARD": t["off"]["HARD"],
        "k_fuel_track": t["k"],
        "g_SOFT": t["g"]["SOFT"],
        "g_MEDIUM": t["g"]["MEDIUM"],
        "g_HARD": t["g"]["HARD"],
        "p_shape": t["p"],
    }


def run_verification(
    n_races: int = 30, seed0: int = 1000, outlier_frac: float = 0.0
) -> pd.DataFrame:
    truth = true_values()
    rows = []
    for i in range(n_races):
        df = synthetic_race(seed=seed0 + i, name=f"verify-{i}", outlier_frac=outlier_frac)
        pp = PhysicsLapModel().fit(df).physical_params()
        for _, r in pp.iterrows():
            tv = truth[r["param"]]
            rows.append(
                {
                    "race": i,
                    "param": r["param"],
                    "true": tv,
                    "estimate": r["estimate"],
                    "ci95_low": r["ci95_low"],
                    "ci95_high": r["ci95_high"],
                    "covered": bool(r["ci95_low"] <= tv <= r["ci95_high"]),
                }
            )
    return pd.DataFrame(rows)
