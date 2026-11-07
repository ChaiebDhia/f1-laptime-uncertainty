# Methods

## 1. Question

Given green-flag race laps, how well can we **forecast lap time and its uncertainty** for stints the model
has not seen, and for the later laps of a stint it has only seen the start of? The second task is the
strategy-relevant one: tyre degradation has to be extrapolated forward in time.

## 2. Data

Public timing data via [FastF1](https://docs.fastf1.dev). One row per lap with driver, stint, race lap, tyre
compound, tyre age and lap time. `clean_laps` keeps representative racing laps and **logs every removal**:

| Step | Why |
|---|---|
| missing lap time or tyre data | cannot be modelled |
| not a dry compound | wet/intermediate running is a different regime |
| lap 1 | standing start, full fuel, traffic |
| track status not green | safety car / yellow flags change the pace regime |
| flagged inaccurate by FastF1 | FastF1's own timing-sync flag |
| pit in/out laps | not racing pace |
| outside 107% of the fastest surviving lap | gross outliers (spins, damage, slow zones) |

`validate_laps` then checks for NaNs, non-positive times, duplicate (driver, lap) rows, unknown compounds and
tyre age that decreases within a stint. Problems are logged as warnings, not silently ignored.

## 3. Models

All three expose `fit(df)` and `predict(df) -> (mean, lower, upper)`.

### 3.1 Physics-structured model (`PhysicsLapModel`)

```
lap_time = base[driver] + offset[compound] + k * lap + g[compound] * tyre_age^p + noise
```

| Term | Meaning | Expectation |
|---|---|---|
| `base[driver]` | driver/car pace in this race | free |
| `offset[compound]` | compound pace offset, medium as reference | soft < 0 < hard |
| `k` | fuel burn **plus** track evolution, s per lap | negative |
| `g[compound]` | tyre degradation rate | positive |
| `p` | degradation shape, 1 = linear, > 1 = accelerating | bounded to [0.5, 3] |

It is a reduced-order model: five physical quantities that an engineer can read and challenge, fitted per race.

**Fitting.** Bounded nonlinear least squares (`scipy.optimize.least_squares`, trust-region reflective), then one
round of robust trimming: laps further than 3.5 robust sigmas (1.4826 x MAD) from the fit are dropped from the
fit and the model is refitted. Trimmed laps are **never** removed from evaluation.

**Uncertainty.** With Jacobian `J` at the solution and residual variance `s^2`, parameter covariance is
`s^2 (J'J)^+`. A prediction at a new lap uses the delta method: `var = j(x)' Cov j(x) + s^2`, with `j(x)` the
numerical gradient of the model with respect to the parameters, and a Student-t critical value. For a driver
unseen in training, the variance of the fitted base paces is added, because that driver's pace is unknown.
A compound never seen in training returns `NaN` instead of extrapolating silently.

### 3.2 Gradient-boosting baseline (`QuantileGBM`)

`HistGradientBoostingRegressor` on (driver, compound, lap, tyre age) with the driver and compound treated as
categorical. Intervals come from separate quantile-loss models at alpha/2, 0.5 and 1 - alpha/2.

### 3.3 Hybrid (`HybridLapModel`)

Physics model as backbone, plus a heavily regularised booster trained on the physics residuals (so the booster
only learns what the physics misses), plus **split-conformal** intervals: the interval half-width is the
finite-sample conformal quantile of absolute residuals on calibration data held back from training.
Calibration data mimics the deployment task: for late-stint forecasting the calibration set is the last part of
each *training* stint, so calibration residuals reflect extrapolation error, not interpolation error. Test data
is never used for calibration.

## 4. Validation

Two out-of-sample schemes, both split by **stint** (laps within a stint are strongly dependent, so a random
lap split would leak):

1. **Unseen stints**: 5-fold `GroupKFold` over whole stints.
2. **Late-stint extrapolation**: train on the first 70% of every stint of at least 8 laps, predict the last 30%.

| Metric | Reading |
|---|---|
| MAE, RMSE | accuracy of the mean forecast |
| Coverage of the nominal 90% interval | whether the uncertainty is honest |
| Mean interval width | sharpness: narrow is only good if coverage holds |
| Interval score (Winkler) | width + (2/alpha) x distance outside the interval; combines both, lower is better |
| RMSE 95% CI | **cluster bootstrap resampling whole stints** |

Coverage is also reported **by compound**, to expose heteroscedasticity (the physics intervals assume constant
noise variance).

## 5. Verification (is the code solving the model correctly?)

Validation asks whether the model suits reality; verification asks whether the code recovers the truth when the
truth is known. `f1lap verify` generates synthetic races from known parameters and checks that the fitted 95%
confidence intervals contain them. Over 60 synthetic races, the 95% intervals covered the true parameters
**91.7%** of the time overall (88% for the shape parameter `p`). The shortfall is expected: the delta method
linearises the model, which is slightly optimistic for the exponent `p`. Reported as is.

The test suite additionally checks: hand-computed interval scores, the finite-sample conformal quantile rule,
no train/test leakage in the stint splits, robustness of the trimmed fit to injected outliers, `NaN` for unseen
compounds, wider intervals for unseen drivers, and the real FastF1 loader against a mock session that uses
FastF1's documented column names and dtypes.

## 6. Sanity band for `k` (heuristic)

Engineers' rules of thumb put the cost of fuel at roughly 0.03 s per kg (about 0.02 to 0.04 depending on
circuit speed), and a modern F1 car burns roughly 1.5 to 2 kg per lap, so the fuel term alone is about 0.03
to 0.08 s per lap. Track evolution adds to it. The report flags a fitted `k` outside **-0.12 to -0.02 s per
lap** as suspicious. This is an informal plausibility check, not a validated threshold. Informal sources:
[Mark Hughes on 2022 fuel and tyre effects](https://michaelbarrett.ghost.is/mark-hughes-the-make-or-break-element-in-the-2022-f1-rules)
(about 0.32 s per 10 kg at Bahrain, 1.875 kg per lap) and
[The Race](https://www.the-race.com/formula-1/the-three-second-estimate-that-scarred-f1-again-in-spanish-gp.md)
(the usual 0.03 s per kg allowance).

## 7. From forecast to decision: tyre strategy under uncertainty (`f1lap strategy`)

Race time of a plan, a sequence of (compound, stint length), is
`sum over laps of [offset[c] + k*lap + g[c]*age^p] + pit_loss * n_stops`; driver base pace is common to all
plans and cancels. Parameter uncertainty is propagated by sampling the 7 physical parameters from the fitted
covariance (2,000 draws). For every legal compound sequence the best stint split is found at the point
estimate, then all candidates are compared over the same draws, giving the mean gap to the best plan, a 90%
interval on that gap, and the **probability that each plan is the fastest**.

Design choices worth knowing:

- *Legal* means at least two different dry compounds and only compounds seen in training.
- Race time in this model does not depend on stint **order** (the fuel and track-evolution term sums to a
  constant over a full race), so order-equivalent plans are merged. Real order effects (track evolution,
  warm-up, traffic) are not modelled.
- Splits are optimised at the point estimate and not re-optimised per draw, which slightly understates the
  value of flexibility.
- Plans with a stint longer than the oldest tyre ever observed are **flagged as extrapolation**.
- Pit-lane time loss is an *input* (`--pit-loss`, default 22 s), circuit specific; it is not estimated.
- This is a model-based what-if, not a race simulator: no safety cars, traffic or track position.

## 8. Limitations

- **Identifiability.** Fuel burn and track evolution are not separable from lap number alone, and tyre age is
  correlated with race lap; identification relies on staggered stints. Parameters can hit their bounds, which the
  report flags.
- **Unmodelled effects.** Traffic, dirty air, engine modes, fuel saving, weather and tyre temperature appear as
  residual noise. Trimming reduces their influence on the fit but not on the test error.
- **Constant noise variance** in the physics intervals; coverage by compound shows where this fails.
- **Per-race fits.** Parameters are not comparable across circuits without a hierarchical model.
- **No simulation or wind-tunnel data.** Public data only has track timing.

## 9. Related work

Bayesian state-space treatments of tyre degradation with compound-specific rates exist, for example
[arXiv:2512.00640](https://arxiv.org/html/2512.00640v1). This project takes a deliberately simpler, fully
inspectable parametric route and puts the weight on validation and calibrated uncertainty.
