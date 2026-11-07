import numpy as np
import pytest

import f1lap
from f1lap.data import TRUE_PARAMS
from f1lap.models import HybridLapModel, PhysicsLapModel, QuantileGBM, conformal_quantile
from f1lap.validation import evaluate_race, interval_score, metrics


def test_physics_point_estimates_close_to_truth(race):
    pp = PhysicsLapModel().fit(race).physical_params().set_index("param")
    assert abs(pp.loc["k_fuel_track", "estimate"] - TRUE_PARAMS["k"]) < 0.006
    assert (pp.loc[["g_SOFT", "g_MEDIUM", "g_HARD"], "estimate"] > 0).all()
    assert not pp["at_bound"].any()


def test_parameter_cis_are_roughly_calibrated():
    """Over many synthetic races the 95% CIs should cover the truth most of the time.
    (A single-seed check would fail ~5% of the time by construction.)"""
    from f1lap.verify import run_verification

    v = run_verification(n_races=12, seed0=500)
    assert v["covered"].mean() >= 0.85


def test_trimming_makes_fit_robust_to_traffic_outliers():
    df = f1lap.synthetic_race(seed=11, outlier_frac=0.05)
    k_true = TRUE_PARAMS["k"]
    trimmed = PhysicsLapModel(trim=3.5).fit(df)
    untrimmed = PhysicsLapModel(trim=None).fit(df)
    err_t = abs(
        trimmed.physical_params().set_index("param").loc["k_fuel_track", "estimate"] - k_true
    )
    err_u = abs(
        untrimmed.physical_params().set_index("param").loc["k_fuel_track", "estimate"] - k_true
    )
    assert trimmed.n_trimmed > 0
    assert err_t <= err_u + 1e-9


def test_unseen_compound_gives_nan_not_a_silent_extrapolation(race):
    train = race[race["compound"] != "SOFT"]
    test = race[race["compound"] == "SOFT"].head(10)
    mu, lo, hi = PhysicsLapModel().fit(train).predict(test)
    assert np.isnan(mu).all() and np.isnan(lo).all() and np.isnan(hi).all()


def test_unseen_driver_gets_wider_interval(race):
    drivers = sorted(race["driver"].unique())
    train = race[race["driver"] != drivers[0]]
    test = race[race["driver"] == drivers[0]].head(20)
    seen = race[race["driver"] == drivers[1]].head(20)
    m = PhysicsLapModel().fit(train)
    _, lo_u, hi_u = m.predict(test)
    _, lo_s, hi_s = m.predict(seen)
    assert np.mean(hi_u - lo_u) > np.mean(hi_s - lo_s)


def test_conformal_quantile_finite_sample_rule():
    r = np.arange(1, 10, dtype=float)  # n = 9, alpha = 0.2 -> ceil(10*0.8)/9 = 8/9
    assert conformal_quantile(r, 0.2) == 9.0
    assert conformal_quantile(np.array([1.0, 2.0, 3.0]), 0.5) == 3.0


def test_metrics_and_interval_score_hand_computed():
    y = np.array([0.0, 2.0])
    lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    # widths 2,2; second point is 1 above the upper bound -> penalty (2/0.1)*1 = 20
    assert interval_score(y, lo, hi, 0.1) == pytest.approx((2 + (2 + 20)) / 2)
    m = metrics(y, np.array([0.0, 1.0]), lo, hi, 0.1)
    assert m["coverage"] == 0.5 and m["MAE_s"] == 0.5


@pytest.mark.parametrize("scheme", ["stint_holdout", "late_stint"])
def test_all_models_give_sane_out_of_sample_forecasts(race, scheme):
    p = evaluate_race(race, scheme, alpha=0.10)
    assert p["valid"].all() and len(p) > 100
    for m in ("physics", "hybrid", "gbm"):
        rmse = np.sqrt(np.mean((p["lap_time"] - p[f"{m}_mu"]) ** 2))
        assert rmse < 1.5, (m, rmse)
        assert (p[f"{m}_hi"] >= p[f"{m}_lo"]).all()


def test_late_stint_intervals_near_nominal_on_synthetic_data(race):
    p = evaluate_race(race, "late_stint", alpha=0.10)
    for m, lo_bound in (("physics", 0.80), ("hybrid", 0.80)):
        cov = np.mean((p["lap_time"] >= p[f"{m}_lo"]) & (p["lap_time"] <= p[f"{m}_hi"]))
        assert lo_bound <= cov <= 0.98, (m, cov)


def test_hybrid_rejects_bad_calibration_mode():
    with pytest.raises(ValueError):
        HybridLapModel(calib="nonsense")


def test_gbm_intervals_are_ordered(race):
    mu, lo, hi = QuantileGBM().fit(race).predict(race)
    assert (lo <= mu).all() and (mu <= hi).all()
