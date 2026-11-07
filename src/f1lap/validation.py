"""Out-of-sample evaluation: schemes, metrics (incl. interval score), cluster bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import HybridLapModel, PhysicsLapModel, QuantileGBM
from .splits import late_stint_split, stint_holdout_splits

SCHEMES = ("stint_holdout", "late_stint")
MODEL_LABELS = {
    "physics": "Physics (analytic CI)",
    "hybrid": "Hybrid (physics+GBM, conformal)",
    "gbm": "GBM (quantile)",
}
SCHEME_LABELS = {
    "stint_holdout": "Unseen stints (5-fold, whole stints held out)",
    "late_stint": "Late-stint extrapolation (last 30% of each stint)",
}


def make_model(name: str, alpha: float, scheme: str, seed: int = 0):
    if name == "physics":
        return PhysicsLapModel(alpha)
    if name == "hybrid":
        return HybridLapModel(
            alpha, calib="late_stint" if scheme == "late_stint" else "stint", seed=seed
        )
    if name == "gbm":
        return QuantileGBM(alpha, seed)
    raise KeyError(name)


def evaluate_race(
    df: pd.DataFrame,
    scheme: str,
    alpha: float = 0.10,
    models: tuple[str, ...] = tuple(MODEL_LABELS),
    seed: int = 0,
) -> pd.DataFrame:
    """Out-of-sample predictions of every model under a validation scheme.

    The returned frame holds the held-out laps with columns <model>_mu/_lo/_hi and `valid`
    (True when every model produced a finite forecast for that lap).
    """
    if scheme == "stint_holdout":
        splits = list(stint_holdout_splits(df))
    elif scheme == "late_stint":
        splits = [late_stint_split(df)]
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    out = []
    for tr, te in splits:
        train, test = df.iloc[tr], df.iloc[te].copy()
        for name in models:
            mu, lo, hi = make_model(name, alpha, scheme, seed).fit(train).predict(test)
            test[f"{name}_mu"], test[f"{name}_lo"], test[f"{name}_hi"] = mu, lo, hi
        out.append(test)
    preds = pd.concat(out)
    finite = np.column_stack(
        [np.isfinite(preds[f"{m}_{s}"]) for m in models for s in ("mu", "lo", "hi")]
    ).all(axis=1)
    preds["valid"] = finite
    return preds


def interval_score(y, lo, hi, alpha: float) -> float:
    """Winkler/interval score: width plus a penalty (2/alpha) per unit outside. Lower is better."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((hi - lo) + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))))


def metrics(y, mu, lo, hi, alpha: float = 0.10) -> dict:
    y, mu, lo, hi = (np.asarray(a, float) for a in (y, mu, lo, hi))
    return {
        "n": int(len(y)),
        "MAE_s": float(np.mean(np.abs(y - mu))),
        "RMSE_s": float(np.sqrt(np.mean((y - mu) ** 2))),
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width_s": float(np.mean(hi - lo)),
        "interval_score": interval_score(y, lo, hi, alpha),
    }


def cluster_bootstrap_rmse(y, mu, groups, n_boot: int = 300, seed: int = 0):
    """95% CI for RMSE, resampling whole stints (laps within a stint are not independent)."""
    err2 = (np.asarray(y, float) - np.asarray(mu, float)) ** 2
    g = pd.Series(err2).groupby(np.asarray(groups))
    sse, cnt = g.sum().to_numpy(), g.size().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sse), size=(n_boot, len(sse)))
    rmse = np.sqrt(sse[idx].sum(axis=1) / cnt[idx].sum(axis=1))
    return float(np.percentile(rmse, 2.5)), float(np.percentile(rmse, 97.5))


def summarise(
    preds: pd.DataFrame,
    alpha: float = 0.10,
    models: tuple[str, ...] = tuple(MODEL_LABELS),
    n_boot: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per model: accuracy, calibration, sharpness, interval score, RMSE bootstrap CI."""
    p = preds[preds["valid"]]
    rows = []
    for m in models:
        y = p["lap_time"].to_numpy()
        row = {
            "model": MODEL_LABELS[m],
            **metrics(y, p[f"{m}_mu"], p[f"{m}_lo"], p[f"{m}_hi"], alpha),
        }
        row["RMSE_lo"], row["RMSE_hi"] = cluster_bootstrap_rmse(
            y, p[f"{m}_mu"], p["stint_id"], n_boot, seed
        )
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_by_group(
    preds: pd.DataFrame, models=tuple(MODEL_LABELS), group: str = "compound"
) -> pd.DataFrame:
    """Empirical interval coverage per subgroup: reveals heteroscedasticity and miscalibration."""
    p = preds[preds["valid"]]
    rows = []
    for key, g in p.groupby(group):
        for m in models:
            rows.append(
                {
                    group: key,
                    "model": MODEL_LABELS[m],
                    "n": len(g),
                    "coverage": float(
                        np.mean((g["lap_time"] >= g[f"{m}_lo"]) & (g["lap_time"] <= g[f"{m}_hi"]))
                    ),
                }
            )
    return pd.DataFrame(rows)
