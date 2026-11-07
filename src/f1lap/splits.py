"""Train/test splitting schemes that respect the stint structure of race data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def stint_holdout_splits(df: pd.DataFrame, n_splits: int = 5):
    """Hold out whole stints: can a model predict a stint it has never seen?"""
    yield from GroupKFold(n_splits=n_splits).split(df, groups=df["stint_id"])


def late_stint_split(df: pd.DataFrame, frac: float = 0.3, min_len: int = 8):
    """Train on the early part of each long stint, test on its last `frac`.

    This is the strategy-relevant task: forecasting tyre degradation forward in time.
    Requires `df` sorted by (stint_id, lap), which `clean_laps` guarantees.
    """
    pos = df.groupby("stint_id").cumcount()
    n = df.groupby("stint_id")["lap"].transform("size")
    test = ((n >= min_len) & (pos >= np.ceil(n * (1 - frac)))).to_numpy()
    return np.flatnonzero(~test), np.flatnonzero(test)


def random_stint_split(df: pd.DataFrame, frac: float = 0.2, seed: int = 0):
    """Hold out a random subset of whole stints (used for conformal calibration)."""
    ids = df["stint_id"].unique()
    if len(ids) < 2:
        return np.arange(len(df)), np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    n_cal = min(max(1, int(round(frac * len(ids)))), len(ids) - 1)
    cal = set(rng.choice(ids, size=n_cal, replace=False))
    mask = df["stint_id"].isin(cal).to_numpy()
    return np.flatnonzero(~mask), np.flatnonzero(mask)
