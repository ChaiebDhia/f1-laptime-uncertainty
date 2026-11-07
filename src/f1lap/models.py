"""Lap-time models, all with the interface fit(df) -> self, predict(df) -> (mean, lo, hi).

PhysicsLapModel  reduced-order physical model, analytic (delta-method) prediction intervals.
QuantileGBM      data-driven baseline, quantile-loss gradient boosting.
HybridLapModel   physics backbone + boosted residual correction, split-conformal intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import COMP_IDX
from .splits import late_stint_split, random_stint_split


def _features(df: pd.DataFrame, driver_idx: dict) -> np.ndarray:
    d = df["driver"].map(driver_idx).to_numpy(float)  # NaN for an unseen driver
    c = df["compound"].map(COMP_IDX).to_numpy(float)
    return np.column_stack([d, c, df["lap"].to_numpy(float), df["tyre_age"].to_numpy(float)])


# ------------------------------------------------------------------------ physics
class PhysicsLapModel:
    """lap_time = base[driver] + offset[compound] + k*lap + g[compound] * tyre_age**p + noise

    k  fuel burn plus track evolution (s per lap; expected < 0, the car gets quicker)
    g  tyre degradation rate per compound (expected > 0)
    p  degradation shape (1 = linear, > 1 = accelerating "cliff")

    Fitted by bounded nonlinear least squares with one round of robust trimming (laps beyond
    `trim` robust sigmas, typically traffic or mistakes, are excluded from the fit but never
    from evaluation). Prediction intervals add parameter uncertainty (delta method on the
    least-squares covariance) to residual noise.
    """

    def __init__(self, alpha: float = 0.10, trim: float | None = 3.5):
        self.alpha, self.trim = alpha, trim

    # --- internals
    def _arrays(self, df):
        d = df["driver"].map(self.driver_idx).fillna(-1).astype(int).to_numpy()
        c = df["compound"].map(COMP_IDX).to_numpy()
        age = np.maximum(df["tyre_age"].to_numpy(float), 1.0)
        return d, c, df["lap"].to_numpy(float), age

    def _f(self, theta, arrays):
        d, c, lap, age = arrays
        nd = self.n_drivers
        base_all = theta[:nd]
        base = np.where(d >= 0, base_all[np.clip(d, 0, None)], base_all.mean())
        off = np.array([theta[nd], 0.0, theta[nd + 1]])  # SOFT, MEDIUM (reference), HARD
        k, g, p = theta[nd + 2], theta[nd + 3 : nd + 6], theta[nd + 6]
        return base + off[c] + k * lap + g[c] * age**p

    def _solve(self, arrays, y, theta0):
        return optimize.least_squares(
            lambda th: self._f(th, arrays) - y, theta0, bounds=(self.lo, self.hi)
        )

    # --- API
    def fit(self, df: pd.DataFrame) -> PhysicsLapModel:
        self.drivers = sorted(df["driver"].unique())
        self.driver_idx = {d: i for i, d in enumerate(self.drivers)}
        self.n_drivers = nd = len(self.drivers)
        self.compounds_seen = {COMP_IDX[c] for c in df["compound"].unique()}
        arrays, y = self._arrays(df), df["lap_time"].to_numpy()
        base0 = df.groupby("driver")["lap_time"].median().reindex(self.drivers).to_numpy()
        theta0 = np.r_[base0, 0.0, 0.0, -0.03, [0.05] * 3, 1.2]
        self.lo = np.r_[np.full(nd, -np.inf), -5, -5, -0.5, [0.0] * 3, 0.5]
        self.hi = np.r_[np.full(nd, np.inf), 5, 5, 0.5, [2.0] * 3, 3.0]
        res = self._solve(arrays, y, theta0)
        self.n_trimmed = 0
        if self.trim:
            r = res.fun - np.median(res.fun)
            robust_sigma = 1.4826 * np.median(np.abs(r))
            keep = np.abs(r) <= self.trim * max(robust_sigma, 1e-9)
            if (~keep).any():
                arrays_k = tuple(a[keep] for a in arrays)
                res = self._solve(arrays_k, y[keep], res.x)
                self.n_trimmed = int((~keep).sum())
        n = len(res.fun)
        self.theta = res.x
        self.dof = max(n - len(res.x), 1)
        self.sigma = float(np.sqrt(np.sum(res.fun**2) / self.dof))
        self.cov = self.sigma**2 * np.linalg.pinv(res.jac.T @ res.jac)
        self.base_var = float(np.var(res.x[:nd])) if nd > 1 else 0.0
        return self

    def mean(self, df: pd.DataFrame) -> np.ndarray:
        mu = self._f(self.theta, self._arrays(df))
        unseen = ~df["compound"].map(COMP_IDX).isin(self.compounds_seen).to_numpy()
        mu[unseen] = np.nan  # compound never seen in training: do not extrapolate silently
        return mu

    def predict(self, df: pd.DataFrame):
        arrays = self._arrays(df)
        mu = self.mean(df)
        f0 = self._f(self.theta, arrays)
        jac = np.empty((len(mu), len(self.theta)))
        for j in range(len(self.theta)):
            th = self.theta.copy()
            h = 1e-6 * max(1.0, abs(th[j]))
            th[j] += h
            jac[:, j] = (self._f(th, arrays) - f0) / h
        var = np.einsum("ij,jk,ik->i", jac, self.cov, jac) + self.sigma**2
        var = var + np.where(arrays[0] < 0, self.base_var, 0.0)  # unseen driver: pace unknown
        half = stats.t.ppf(1 - self.alpha / 2, self.dof) * np.sqrt(var)
        return mu, mu - half, mu + half

    def physical_params(self) -> pd.DataFrame:
        nd = self.n_drivers
        names = [
            "offset_SOFT",
            "offset_HARD",
            "k_fuel_track",
            "g_SOFT",
            "g_MEDIUM",
            "g_HARD",
            "p_shape",
        ]
        idx = np.arange(nd, nd + 7)
        est, se = self.theta[idx], np.sqrt(np.diag(self.cov)[idx])
        tcrit = stats.t.ppf(0.975, self.dof)
        lo, hi = self.lo[idx], self.hi[idx]
        at_bound = (est <= lo + 1e-6 * (1 + np.abs(lo))) | (est >= hi - 1e-6 * (1 + np.abs(hi)))
        return pd.DataFrame(
            {
                "param": names,
                "estimate": est,
                "std_err": se,
                "ci95_low": est - tcrit * se,
                "ci95_high": est + tcrit * se,
                "at_bound": at_bound,
            }
        )

    def degradation_draws(self, ages, n: int = 300, seed: int = 0) -> np.ndarray:
        """Sampled degradation curves (n, 3 compounds, len(ages)) from parameter uncertainty."""
        nd = self.n_drivers
        idx = np.arange(nd + 3, nd + 7)
        rng = np.random.default_rng(seed)
        th = rng.multivariate_normal(
            self.theta[idx], self.cov[np.ix_(idx, idx)], n, check_valid="ignore"
        )
        g, p = np.clip(th[:, :3], 0, None), np.clip(th[:, 3], 0.5, 3.0)
        return g[:, :, None] * np.asarray(ages, float)[None, None, :] ** p[:, None, None]


# --------------------------------------------------------------------- ML baseline
class QuantileGBM:
    """Gradient boosting with quantile loss: lower, median and upper bound models."""

    def __init__(self, alpha: float = 0.10, seed: int = 0):
        self.alpha, self.seed = alpha, seed

    def fit(self, df: pd.DataFrame) -> QuantileGBM:
        self.driver_idx = {d: i for i, d in enumerate(sorted(df["driver"].unique()))}
        x, y = _features(df, self.driver_idx), df["lap_time"].to_numpy()
        self.models = {
            q: HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                categorical_features=[0, 1],
                max_iter=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=self.seed,
            ).fit(x, y)
            for q in (self.alpha / 2, 0.5, 1 - self.alpha / 2)
        }
        return self

    def predict(self, df: pd.DataFrame):
        x = _features(df, self.driver_idx)
        q_lo, q_mid, q_hi = sorted(self.models)
        lo, mu, hi = (self.models[q].predict(x) for q in (q_lo, q_mid, q_hi))
        return mu, np.minimum(lo, mu), np.maximum(hi, mu)


# -------------------------------------------------------------------------- hybrid
def conformal_quantile(abs_resid: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile of absolute calibration residuals."""
    n = len(abs_resid)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(abs_resid, level, method="higher"))


class HybridLapModel:
    """Physics backbone + gradient-boosted residual correction + split-conformal intervals.

    The physics model carries the structure (and extrapolates), the booster learns what the
    physics misses (e.g. circuit-specific effects), and conformal calibration on held-back
    data turns the combined point forecast into an interval with a finite-sample coverage
    guarantee. Calibration data is carved out of the *training* data and mimics the
    deployment task: `calib="late_stint"` calibrates on the last part of each training stint
    (forward extrapolation), `calib="stint"` on whole held-back stints.
    """

    def __init__(
        self, alpha: float = 0.10, calib: str = "stint", calib_frac: float = 0.25, seed: int = 0
    ):
        if calib not in {"stint", "late_stint"}:
            raise ValueError("calib must be 'stint' or 'late_stint'")
        self.alpha, self.calib, self.calib_frac, self.seed = alpha, calib, calib_frac, seed

    def _mean(self, df):
        return self.physics.mean(df) + self.booster.predict(_features(df, self.physics.driver_idx))

    def fit(self, df: pd.DataFrame) -> HybridLapModel:
        if self.calib == "late_stint":
            tr_i, ca_i = late_stint_split(df, frac=self.calib_frac)
        else:
            tr_i, ca_i = random_stint_split(df, frac=0.2, seed=self.seed)
        if len(ca_i) < 20:  # too little data to calibrate honestly: fall back to in-sample
            tr_i, ca_i = np.arange(len(df)), np.arange(len(df))
        train, calib = df.iloc[tr_i], df.iloc[ca_i]
        self.physics = PhysicsLapModel(self.alpha).fit(train)
        resid = train["lap_time"].to_numpy() - self.physics.mean(train)
        ok = np.isfinite(resid)
        self.booster = HistGradientBoostingRegressor(
            categorical_features=[0, 1],
            max_iter=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=self.seed,
        ).fit(_features(train, self.physics.driver_idx)[ok], resid[ok])
        cal_resid = np.abs(calib["lap_time"].to_numpy() - self._mean(calib))
        self.q = conformal_quantile(cal_resid[np.isfinite(cal_resid)], self.alpha)
        return self

    def predict(self, df: pd.DataFrame):
        mu = self._mean(df)
        return mu, mu - self.q, mu + self.q
