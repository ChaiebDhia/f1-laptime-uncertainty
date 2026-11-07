"""Data loading, cleaning (with a drop log), validation and synthetic race generation."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
COMP_IDX = {c: i for i, c in enumerate(COMPOUNDS)}
REQUIRED = [
    "race",
    "driver",
    "stint",
    "lap",
    "compound",
    "tyre_age",
    "lap_time",
    "green",
    "accurate",
    "pit",
]


def clean_laps(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep representative racing laps and say exactly what was removed and why.

    Returns (clean laps sorted by stint and lap, drop log). The 107% rule is applied last,
    relative to the fastest lap that survived every other filter.
    """
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"raw laps are missing columns: {missing}")
    steps = [
        (
            "missing lap time or tyre data",
            lambda d: d[["lap_time", "tyre_age", "compound", "stint"]].notna().all(axis=1),
        ),
        ("not a dry compound", lambda d: d["compound"].isin(COMPOUNDS)),
        ("lap 1", lambda d: d["lap"] > 1),
        ("track status not green", lambda d: d["green"].astype(bool)),
        ("flagged inaccurate by FastF1", lambda d: d["accurate"].astype(bool)),
        ("pit in/out lap", lambda d: ~d["pit"].astype(bool)),
    ]
    df = raw.copy()
    log_rows = [{"step": "raw laps", "removed": 0, "remaining": len(df)}]
    for name, keep in steps:
        mask = keep(df)
        log_rows.append({"step": name, "removed": int((~mask).sum()), "remaining": int(mask.sum())})
        df = df[mask]
    if len(df):
        mask = df["lap_time"] <= 1.07 * df["lap_time"].min()
        log_rows.append(
            {
                "step": "outside 107% of fastest lap",
                "removed": int((~mask).sum()),
                "remaining": int(mask.sum()),
            }
        )
        df = df[mask]
    df = df.copy()
    df["stint_id"] = df["race"] + "|" + df["driver"] + "|" + df["stint"].astype(int).astype(str)
    df = df.sort_values(["stint_id", "lap"]).reset_index(drop=True)
    return df, pd.DataFrame(log_rows)


def validate_laps(df: pd.DataFrame) -> list[str]:
    """Return a list of data-quality problems (empty list = clean)."""
    issues: list[str] = []
    if df.empty:
        return ["no laps"]
    if df[["lap_time", "tyre_age", "lap", "compound"]].isna().any().any():
        issues.append("NaN values in lap_time/tyre_age/lap/compound")
    if (df["lap_time"] <= 0).any():
        issues.append("non-positive lap times")
    if (df["tyre_age"] < 1).any():
        issues.append("tyre_age below 1")
    if df.duplicated(["driver", "lap"]).any():
        issues.append("duplicate (driver, lap) rows")
    bad = ~df["compound"].isin(COMPOUNDS)
    if bad.any():
        issues.append(f"unexpected compounds: {sorted(df.loc[bad, 'compound'].unique())}")
    for sid, g in df.sort_values("lap").groupby("stint_id"):
        if (np.diff(g["tyre_age"].to_numpy()) < 0).any():
            issues.append(f"tyre age decreases inside stint {sid}")
            break
    return issues


def load_race(
    year: int, gp: str, cache_dir: str = ".fastf1_cache"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download one race session via FastF1 and return (clean laps, drop log)."""
    import fastf1

    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    session = fastf1.get_session(year, gp, "R")
    session.load(telemetry=False, weather=False, messages=False)
    laps = session.laps
    raw = pd.DataFrame(
        {
            "race": f"{year} {gp}",
            "driver": laps["Driver"].to_numpy(),
            "stint": laps["Stint"].to_numpy(),
            "lap": laps["LapNumber"].to_numpy(),
            "compound": laps["Compound"].to_numpy(),
            "tyre_age": laps["TyreLife"].to_numpy(),
            "lap_time": laps["LapTime"].dt.total_seconds().to_numpy(),
            "green": laps["TrackStatus"].astype(str).eq("1").to_numpy(),
            "accurate": laps["IsAccurate"].eq(True).to_numpy(),
            "pit": (laps["PitInTime"].notna() | laps["PitOutTime"].notna()).to_numpy(),
        }
    )
    return clean_laps(raw)


# ----------------------------------------------------------------- synthetic data
TRUE_PARAMS = {
    "k": -0.045,
    "off": {"SOFT": -0.6, "MEDIUM": 0.0, "HARD": 0.5},
    "g": {"SOFT": 0.08, "MEDIUM": 0.05, "HARD": 0.03},
    "p": 1.3,
    "sigma": 0.25,
}
_PLANS = [
    ("MEDIUM", "HARD", "HARD"),
    ("SOFT", "MEDIUM", "MEDIUM"),
    ("MEDIUM", "HARD", "MEDIUM"),
    ("HARD", "MEDIUM", "SOFT"),
    ("SOFT", "HARD", "MEDIUM"),
]


def synthetic_race(
    n_laps: int = 57,
    n_drivers: int = 20,
    seed: int = 0,
    name: str = "synthetic",
    outlier_frac: float = 0.0,
) -> pd.DataFrame:
    """Generate a race from known parameters (TRUE_PARAMS).

    Used for verification (does the fitting code recover known truth?), tests, and
    offline dry runs. `outlier_frac` adds traffic-like slow laps (+1 to +4 s).
    """
    rng = np.random.default_rng(seed)
    t = TRUE_PARAMS
    rows = []
    for d in range(n_drivers):
        base = 90.0 + rng.normal(0, 0.4)
        p1, p2 = int(rng.integers(14, 24)), int(rng.integers(34, 44))
        plan = _PLANS[int(rng.integers(len(_PLANS)))]
        edges = [1, p1, p2, n_laps + 1]
        for s in range(3):
            c = plan[s]
            for lap in range(edges[s], edges[s + 1]):
                age = lap - edges[s] + 1
                y = (
                    base
                    + t["off"][c]
                    + t["k"] * lap
                    + t["g"][c] * age ** t["p"]
                    + rng.normal(0, t["sigma"])
                )
                if rng.random() < outlier_frac:
                    y += rng.uniform(1.0, 4.0)
                rows.append(
                    {
                        "race": name,
                        "driver": f"D{d:02d}",
                        "stint": s + 1,
                        "lap": lap,
                        "compound": c,
                        "tyre_age": age,
                        "lap_time": y,
                        "green": True,
                        "accurate": True,
                        "pit": False,
                    }
                )
    clean, _ = clean_laps(pd.DataFrame(rows))
    return clean
