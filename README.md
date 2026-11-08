<div align="center">

# f1lap

### Forecasting F1 race lap time and tyre strategy, with uncertainty you can check

[![CI](https://github.com/ChaiebDhia/f1-laptime-uncertainty/actions/workflows/ci.yml/badge.svg)](https://github.com/ChaiebDhia/f1-laptime-uncertainty/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Tests](https://img.shields.io/badge/tests-37-brightgreen)
![Data](https://img.shields.io/badge/data-OpenF1%20(free)-orange)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

> **Question.** Can a small physics-structured model forecast race lap time, with *honest* uncertainty, better
> than a generic ML model, especially when it must extrapolate tyre degradation forward in time?

**At a glance.** Three models, two out-of-sample validation schemes, a calibration check on the uncertainty, a
verification step against known ground truth, and a tyre-strategy ranking that carries the uncertainty through.
Public data, one command, fully reproducible.

| | Physics-structured | Hybrid (physics + boosting) | Gradient boosting |
|---|---|---|---|
| **Late-stint RMSE** (865 held-out laps, 3 races) | **0.53 s** | 0.54 s | 0.68 s |
| **90% interval coverage** (target 90%) | 87% | 88% | 65% |

Physics and hybrid are statistically indistinguishable; both clearly beat boosting, whose intervals are badly
overconfident. Full tables, parameters and caveats: [`results/REPORT.md`](results/REPORT.md).

## Why it matters

Strategy and performance decisions rest on forecasts. A number without an honest error bar is a guess: if the
interval says 90% but holds 65% of the time, every decision built on it is overconfident. This repo treats
uncertainty as something to *measure and score*, not decorate, and shows where the model is weak.

| Model | What it is | Uncertainty |
|---|---|---|
| **Physics** | `base[driver] + offset[compound] + k·lap + g[compound]·age^p`, fitted per race with SciPy | Parameter covariance propagated to each forecast (delta method) |
| **GBM** | Gradient boosting on driver, compound, lap, tyre age | Quantile-loss models |
| **Hybrid** | Physics backbone + boosted residual correction | Split-conformal, calibrated on held-back stints |

## Quickstart

```bash
git clone https://github.com/ChaiebDhia/f1-laptime-uncertainty.git
cd f1-laptime-uncertainty
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest                             # unit tests, no network needed
f1lap run --synthetic              # full pipeline on generated data, no network needed
f1lap source-check --year 2024 --race Bahrain         # is the data source reachable and as expected?
f1lap run --year 2024 --races Bahrain Spain Hungary   # real data (first run downloads and caches)
f1lap verify                       # parameter-recovery check on synthetic races
f1lap strategy --year 2024 --race Bahrain --pit-loss 22   # tyre strategy under uncertainty
```

Outputs land in `results/`: `REPORT.md` (every number computed, none typed), CSV tables and figures.

## Results

Real-data results from `f1lap run` (full tables in [`results/REPORT.md`](results/REPORT.md)):

![Late-stint forecasts](results/late_stint_forecast.png)
![Fitted tyre degradation](results/degradation.png)
![Diagnostics](results/diagnostics.png)

**Key findings**

- **Accuracy.** On late-stint forecasts (865 held-out laps, 3 races), RMSE was physics 0.53 s (95% CI 0.48 to 0.58); hybrid 0.54 s (not distinguishable from physics: intervals overlap); boosting 0.68 s (clearly worse: 95% intervals do not overlap). On unseen stints the lowest RMSE was physics at 0.55 s.
- **Calibration.** Empirical coverage of the nominal 90% intervals on late-stint forecasts: physics 87%, hybrid 88%, boosting 65%. Coverage by compound is in `results/REPORT.md`.
- **Physics.** The fuel and track-evolution slope `k` was fitted between -0.071 and -0.045 s per lap across the races, with narrow confidence intervals.
- **Caveat.** In 2 of 3 races the fitted degradation rates were not ordered soft >= medium >= hard, in 2 the degradation shape `p` was below 1, and in 1 at least one parameter had no defined standard error. Forecasts and `k` are well determined, but the per-compound rates are weakly identified (tyre age and race lap are collinear): read them as fitted terms, not physical constants.
- **Decision.** For 2024 Bahrain at an assumed 22 s pit loss, the highest-probability plan was S17 + H20 + H20 (probability fastest 100% under parameter uncertainty only; model error, traffic and safety cars are not included).

## Data sources

| Source | Needs | Notes |
|---|---|---|
| **OpenF1** (default, `--source openf1`) | nothing; free, no account | Historical data from 2023. Tyre compound, stints, pit stops, race control. Green-flag laps are approximated from race-control messages |
| **FastF1** (`--source fastf1`) | an F1TV Access account | Formula 1 put its data APIs behind a login in October 2026; FastF1 3.9 prompts for it |

Both feed the same cleaning and validation code. Raw data is cached locally (`.openf1_cache/`,
`.fastf1_cache/`) and never committed.

## How the evaluation is built (the part that matters)

- **Split by stint, never by lap.** Laps in a stint are strongly dependent, so a random lap split leaks.
- **Two tasks.** (1) predict stints never seen; (2) train on the first 70% of each stint and forecast the last 30%.
- **Uncertainty is scored, not decorated.** Each model reports the empirical coverage of its nominal 90%
  interval, interval width, and the interval score. An accurate model with 60% coverage is overconfident.
- **Honest error bars.** RMSE confidence intervals come from a cluster bootstrap over whole stints.
- **Calibration data mimics deployment.** For late-stint forecasting, the hybrid's conformal calibration uses
  the last part of each *training* stint, so calibration residuals reflect extrapolation error.
- **Every dropped lap is accounted for.** The cleaning log shows what was removed and why.

## From forecast to decision

`f1lap strategy` turns the fitted physics model into a tyre-strategy comparison with the parameter
uncertainty carried through: for each legal plan it reports the mean gap to the best plan, a 90% interval on
that gap, and the **probability that the plan is the fastest**. Plans that need a tyre to last longer than any
tyre observed are flagged as extrapolation. It is a model-based what-if (no traffic, safety cars or track
position; pit loss is an input), documented in [`docs/METHODS.md`](docs/METHODS.md).

## Verification on known ground truth

Validation asks whether a model suits reality; verification asks whether the code recovers the truth when it is
known. On 60 synthetic races generated from known parameters, the fitted 95% confidence intervals contained the
true values **91.7%** of the time (88% for the shape parameter): slightly optimistic, as expected from
linearising a power-law exponent, and reported as is.

![Verification](docs/img/verification.png)

## Engineering

- Installable package (`src/` layout, `f1lap` CLI with `run`, `verify`, `strategy` and `source-check`), type-annotated, linted with Ruff.
- 37 tests covering cleaning, validation, models, splits, metrics and strategy maths (hand-computed values),
  robustness to outliers, and the CLI end to end. Both data loaders are tested offline: FastF1 against a mock
  session with its documented schema, OpenF1 against mocked HTTP (retries, caching, session matching, tyre age).
- GitHub Actions CI on Python 3.11 and 3.12 (lint, tests, end-to-end smoke runs) plus a Docker build job; Dependabot keeps actions and dependencies current.
- Deterministic: every stochastic step takes a seed.
- Data hygiene: input validation with warnings, a cleaning log, and no raw data committed.

```
src/f1lap/
  data.py        FastF1 loading, cleaning with drop log, validation, synthetic races
  openf1.py      OpenF1 loading (free, no account), caching, retries, green-flag approximation
  models.py      PhysicsLapModel, QuantileGBM, HybridLapModel (+ split-conformal)
  splits.py      stint hold-out, late-stint, calibration splits
  validation.py  schemes, metrics, interval score, cluster bootstrap, coverage by group
  strategy.py    strategy ranking with propagated uncertainty
  verify.py      known-truth parameter recovery
  plots.py       figures        report.py   generated REPORT.md        cli.py   run | verify | strategy | source-check
docs/METHODS.md  equations, fitting, uncertainty, validation design, limitations
```

Method details, assumptions and limitations: [`docs/METHODS.md`](docs/METHODS.md).

## Limitations and next steps

Fuel burn and track evolution are not separable from lap number alone; tyre age and race lap are correlated;
traffic, dirty air, engine modes and weather are not modelled and appear as residual noise; intervals in the
physics model assume constant noise variance; fits are per race. Details in `docs/METHODS.md`.

Next: a hierarchical model across races (to pin down per-compound rates), heteroscedastic noise per compound,
conformal calibration for the physics intervals, and weather and track temperature as covariates.

## Related work

Bayesian state-space models of tyre degradation with compound-specific rates exist
([arXiv:2512.00640](https://arxiv.org/html/2512.00640v1)); this project takes a simpler, fully inspectable
parametric route and puts the weight on validation and calibrated uncertainty.

## Disclaimer

Unofficial project using publicly available timing data through the open-source
[OpenF1](https://openf1.org) API and, optionally, the [FastF1](https://github.com/theOehrly/Fast-F1) library. Not affiliated with, endorsed by or connected to
Formula 1, the FIA or any team. No raw timing data is included in this repository.

## License

MIT, see [LICENSE](LICENSE).
