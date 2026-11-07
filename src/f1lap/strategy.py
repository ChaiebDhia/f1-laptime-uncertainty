"""Decision support: compare tyre strategies with parameter uncertainty propagated through.

Given a fitted PhysicsLapModel, race time for a plan (a sequence of (compound, stint length)) is

    sum over laps of [ offset[c] + k * lap + g[c] * tyre_age**p ]  +  pit_loss * n_stops

The driver base pace is identical across plans and cancels, so plans are compared on this
quantity alone. Parameter uncertainty is propagated by sampling from the fitted covariance.

This is a *model-based what-if*, not a race simulator: no traffic, safety cars, track position,
tyre warm-up or engine modes. The pit-lane time loss is an input (circuit specific), not a result.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import COMPOUNDS
from .models import PhysicsLapModel

MIN_STINT = 5


def parameter_draws(model: PhysicsLapModel, n: int = 2000, seed: int = 0) -> np.ndarray:
    """Draws of the 7 physical parameters, shape (n, 7):
    [offset_SOFT, offset_HARD, k, g_SOFT, g_MEDIUM, g_HARD, p]."""
    nd = model.n_drivers
    idx = np.arange(nd, nd + 7)
    rng = np.random.default_rng(seed)
    th = rng.multivariate_normal(
        model.theta[idx], model.cov[np.ix_(idx, idx)], n, check_valid="ignore"
    )
    th[:, 3:6] = np.clip(th[:, 3:6], 0.0, None)
    th[:, 6] = np.clip(th[:, 6], 0.5, 3.0)
    return th


def point_params(model: PhysicsLapModel) -> np.ndarray:
    nd = model.n_drivers
    return model.theta[nd : nd + 7][None, :]


def stint_time(th: np.ndarray, compound: str, first_lap: int, n_laps: int) -> np.ndarray:
    """Time (relative to driver base pace) of one stint, for each parameter draw. Shape (n,)."""
    c = COMPOUNDS.index(compound)
    ages = np.arange(1, n_laps + 1, dtype=float)
    laps = first_lap + ages - 1
    offset = (th[:, 0], np.zeros(len(th)), th[:, 1])[c]
    return (
        offset * n_laps
        + th[:, 2] * laps.sum()
        + th[:, 3 + c] * (ages[None, :] ** th[:, 6][:, None]).sum(axis=1)
    )


def plan_time(th: np.ndarray, plan: tuple, pit_loss: float) -> np.ndarray:
    """Race time of a plan ((compound, n_laps), ...) for each parameter draw."""
    total, lap = np.zeros(len(th)), 1
    for compound, n_laps in plan:
        total += stint_time(th, compound, lap, n_laps)
        lap += n_laps
    return total + pit_loss * (len(plan) - 1)


def canonical(plan: tuple) -> tuple:
    """Order stints by (compound, length). Race time in this model does not depend on stint
    order (the fuel/track-evolution term sums to a constant over a full race), so plans that
    differ only in order are the same plan."""
    return tuple(sorted(plan, key=lambda s: (COMPOUNDS.index(s[0]), s[1])))


def label(plan: tuple) -> str:
    return " + ".join(f"{c[0]}{n}" for c, n in plan)


def _compositions(total: int, parts: int, step: int):
    """All ways to split `total` laps into `parts` stints of at least MIN_STINT laps."""
    if parts == 1:
        if total >= MIN_STINT:
            yield (total,)
        return
    for first in range(MIN_STINT, total - MIN_STINT * (parts - 1) + 1, step):
        for rest in _compositions(total - first, parts - 1, step):
            yield (first, *rest)


def best_plans(
    model: PhysicsLapModel, n_laps: int, pit_loss: float, max_stops: int = 2, step: int = 1
) -> list[tuple]:
    """For every legal compound sequence, the stint split that is fastest at the point estimate.

    Legal here means at least two different dry compounds (the dry-race regulation) and only
    compounds seen when fitting.
    """
    seen = [COMPOUNDS[i] for i in sorted(model.compounds_seen)]
    th = point_params(model)
    plans: dict[tuple, tuple] = {}
    for stops in range(1, max_stops + 1):
        for seq in itertools.product(seen, repeat=stops + 1):
            if len(set(seq)) < 2:
                continue
            best, best_t = None, np.inf
            for split in _compositions(n_laps, stops + 1, step):
                plan = tuple(zip(seq, split, strict=True))
                t = plan_time(th, plan, pit_loss)[0]
                if t < best_t:
                    best, best_t = plan, t
            if best is not None:
                plans.setdefault(canonical(best), canonical(best))
    return list(plans.values())


@dataclass
class StrategyResult:
    table: pd.DataFrame
    pit_curve: pd.DataFrame
    curve_label: str
    n_laps: int
    pit_loss: float


def max_observed_age(df: pd.DataFrame) -> dict[str, int]:
    return {c: int(g["tyre_age"].max()) for c, g in df.groupby("compound")}


def analyse_strategies(
    model: PhysicsLapModel,
    df: pd.DataFrame,
    n_laps: int | None = None,
    pit_loss: float = 22.0,
    max_stops: int = 2,
    n_draws: int = 2000,
    seed: int = 0,
    step: int = 1,
) -> StrategyResult:
    """Rank strategies (order-invariant, see `canonical`) and say how sure the ranking is.

    Columns: mean race time relative to the best plan, 90% interval of that gap over parameter
    uncertainty, probability that the plan is the fastest of all candidates, and whether any
    stint is longer than the oldest tyre the model has actually seen (extrapolation).
    """
    n_laps = int(n_laps or df["lap"].max())
    plans = best_plans(model, n_laps, pit_loss, max_stops, step)
    if not plans:
        raise ValueError("no legal strategy: need at least two dry compounds in the training data")
    th = parameter_draws(model, n_draws, seed)
    times = np.column_stack([plan_time(th, p, pit_loss) for p in plans])  # (draws, plans)
    mean_t = times.mean(axis=0)
    ref = int(np.argmin(mean_t))
    gap = times - times[:, [ref]]
    wins = np.bincount(times.argmin(axis=1), minlength=len(plans)) / len(times)
    seen_age = max_observed_age(df)
    rows = []
    for i, plan in enumerate(plans):
        extrap = any(n > seen_age.get(c, 0) for c, n in plan)
        rows.append(
            {
                "plan": label(plan),
                "stops": len(plan) - 1,
                "mean_gap_s": float(mean_t[i] - mean_t[ref]),
                "gap_p05_s": float(np.percentile(gap[:, i], 5)),
                "gap_p95_s": float(np.percentile(gap[:, i], 95)),
                "P_fastest": float(wins[i]),
                "extrapolates_tyre_life": bool(extrap),
            }
        )
    table = pd.DataFrame(rows).sort_values("mean_gap_s").reset_index(drop=True)

    # pit-window curve for the best one-stop plan's compound order
    one_stop = [i for i, p in enumerate(plans) if len(p) == 2]
    best_stop1 = plans[min(one_stop, key=lambda i: mean_t[i])] if one_stop else None
    curve = pd.DataFrame()
    curve_label = ""
    if best_stop1 is not None:
        (c1, _), (c2, _) = best_stop1
        laps_grid = list(range(MIN_STINT, n_laps - MIN_STINT + 1))
        t = np.column_stack(
            [plan_time(th, ((c1, L), (c2, n_laps - L)), pit_loss) for L in laps_grid]
        )
        rel = t - t.mean(axis=0).min()
        curve = pd.DataFrame(
            {
                "pit_lap": laps_grid,
                "mean_s": rel.mean(axis=0),
                "p05_s": np.percentile(rel, 5, axis=0),
                "p95_s": np.percentile(rel, 95, axis=0),
            }
        )
        curve_label = f"{c1} + {c2}"
    return StrategyResult(table, curve, curve_label, n_laps, pit_loss)
