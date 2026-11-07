import numpy as np
import pytest

import f1lap
from f1lap.data import TRUE_PARAMS
from f1lap.models import PhysicsLapModel
from f1lap.strategy import (
    analyse_strategies,
    best_plans,
    canonical,
    plan_time,
    stint_time,
)


@pytest.fixture(scope="module")
def fitted():
    df = f1lap.synthetic_race(seed=21, name="strat")
    return df, PhysicsLapModel().fit(df)


def _true_theta():
    t = TRUE_PARAMS
    return np.array(
        [
            [
                t["off"]["SOFT"],
                t["off"]["HARD"],
                t["k"],
                t["g"]["SOFT"],
                t["g"]["MEDIUM"],
                t["g"]["HARD"],
                t["p"],
            ]
        ]
    )


def test_stint_time_matches_hand_calculation():
    th = np.array([[-0.5, 0.4, -0.04, 0.1, 0.05, 0.02, 1.0]])
    # SOFT, laps 3..5, ages 1..3, p = 1: offset -0.5*3 + k*(3+4+5) + 0.1*(1+2+3)
    expected = -0.5 * 3 + (-0.04) * 12 + 0.1 * 6
    assert stint_time(th, "SOFT", 3, 3)[0] == pytest.approx(expected)


def test_plan_time_adds_pit_loss_per_stop():
    th = _true_theta()
    one = plan_time(th, (("MEDIUM", 30), ("HARD", 27)), pit_loss=20.0)[0]
    zero = plan_time(th, (("MEDIUM", 30), ("HARD", 27)), pit_loss=0.0)[0]
    assert one - zero == pytest.approx(20.0)


def test_stint_order_does_not_change_race_time():
    th = _true_theta()
    a = plan_time(th, (("SOFT", 15), ("HARD", 42)), 20.0)[0]
    b = plan_time(th, (("HARD", 42), ("SOFT", 15)), 20.0)[0]
    assert a == pytest.approx(b)
    assert canonical((("HARD", 42), ("SOFT", 15))) == canonical((("SOFT", 15), ("HARD", 42)))


def test_candidate_plans_are_legal_and_unique(fitted):
    df, model = fitted
    plans = best_plans(model, 57, pit_loss=22.0, step=3)
    assert len(plans) == len({canonical(p) for p in plans})
    for p in plans:
        assert len({c for c, _ in p}) >= 2  # two different dry compounds
        assert sum(n for _, n in p) == 57
        assert all(n >= 5 for _, n in p)


def test_win_probabilities_sum_to_one_and_gaps_are_consistent(fitted):
    df, model = fitted
    r = analyse_strategies(model, df, pit_loss=22.0, n_draws=500, step=3)
    assert r.table["P_fastest"].sum() == pytest.approx(1.0)
    assert r.table.iloc[0]["mean_gap_s"] == pytest.approx(0.0)
    assert (r.table["mean_gap_s"].diff().dropna() >= -1e-9).all()
    assert (r.table["gap_p05_s"] <= r.table["gap_p95_s"]).all()


def test_higher_pit_loss_never_favours_more_stops(fitted):
    df, model = fitted
    cheap = analyse_strategies(model, df, pit_loss=0.0, n_draws=300, step=3).table.iloc[0]
    costly = analyse_strategies(model, df, pit_loss=60.0, n_draws=300, step=3).table.iloc[0]
    assert costly["stops"] <= cheap["stops"]


def test_extrapolation_beyond_observed_tyre_life_is_flagged(fitted):
    df, model = fitted
    r = analyse_strategies(model, df, n_laps=120, pit_loss=60.0, n_draws=200, step=5)
    assert r.table["extrapolates_tyre_life"].any()


def test_cli_strategy_end_to_end(tmp_path):
    from f1lap.cli import main

    rc = main(["strategy", "--synthetic", "--step", "3", "--draws", "200", "--out", str(tmp_path)])
    assert rc == 0
    for f in ("strategy.csv", "strategy.png", "STRATEGY.md"):
        assert (tmp_path / f).exists()
