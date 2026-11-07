"""Command line interface: `f1lap run` (study) and `f1lap verify` (known-truth check)."""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from . import plots
from .data import load_race, synthetic_race, validate_laps
from .models import PhysicsLapModel
from .openf1 import describe, load_race_openf1
from .report import write_report
from .strategy import analyse_strategies
from .validation import SCHEMES, coverage_by_group, evaluate_race, summarise
from .verify import run_verification

log = logging.getLogger("f1lap")
MIN_LAPS = 200


def _load_one(args, year: int, gp: str):
    """Load one race from the selected data source."""
    if args.source == "fastf1":
        return load_race(year, gp, args.cache_dir or ".fastf1_cache")
    return load_race_openf1(year, gp, args.cache_dir or ".openf1_cache")


def _load(args):
    races, droplogs = {}, {}
    if args.synthetic:
        for s in range(args.synthetic_races):
            name = f"synthetic-{s}"
            races[name] = synthetic_race(seed=s, name=name, outlier_frac=0.02)
            droplogs[name] = None
        return races, droplogs
    for gp in args.races:
        name = f"{args.year} {gp}"
        try:
            races[name], droplogs[name] = _load_one(args, args.year, gp)
        except Exception as exc:  # network, unknown event, schema change: keep going
            log.error("could not load %s: %s", name, exc)
    return races, droplogs


def run(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    races, droplogs = _load(args)
    for name in list(races):
        issues = validate_laps(races[name])
        for i in issues:
            log.warning("%s: data check: %s", name, i)
        if len(races[name]) < MIN_LAPS:
            log.warning("%s: only %d clean laps, skipping", name, len(races[name]))
            races.pop(name)
    if not races:
        log.error("no usable races, nothing to do")
        return 1
    for name, df in races.items():
        log.info(
            "%s: %d clean laps, %d drivers, %d stints",
            name,
            len(df),
            df["driver"].nunique(),
            df["stint_id"].nunique(),
        )

    preds, by_race_rows = {}, []
    for scheme in SCHEMES:
        per_race = []
        for name, df in races.items():
            log.info("evaluating %s | %s", name, scheme)
            p = evaluate_race(df, scheme, args.alpha, seed=args.seed)
            per_race.append(p)
            s = summarise(p, args.alpha, n_boot=args.n_boot, seed=args.seed)
            s.insert(0, "scheme", scheme)
            s.insert(0, "race", name)
            by_race_rows.append(s)
        preds[scheme] = pd.concat(per_race)
    pooled = {
        sc: summarise(preds[sc], args.alpha, n_boot=args.n_boot, seed=args.seed) for sc in SCHEMES
    }
    by_race = pd.concat(by_race_rows)[
        ["race", "scheme", "model", "n", "RMSE_s", "coverage", "mean_width_s", "interval_score"]
    ]

    fits, par_rows, trimmed = {}, [], {}
    for name, df in races.items():
        model = PhysicsLapModel(args.alpha).fit(df)
        fits[name] = (df, model)
        trimmed[name] = model.n_trimmed
        pp = model.physical_params()
        pp.insert(0, "race", name)
        par_rows.append(pp)
    params = pd.concat(par_rows)

    # machine-readable outputs
    pd.concat([s.assign(scheme=sc) for sc, s in pooled.items()]).to_csv(
        f"{args.out}/validation_pooled.csv", index=False
    )
    by_race.to_csv(f"{args.out}/validation_by_race.csv", index=False)
    params.to_csv(f"{args.out}/physical_params.csv", index=False)
    pd.concat([coverage_by_group(preds[sc]).assign(scheme=sc) for sc in SCHEMES]).to_csv(
        f"{args.out}/coverage_by_compound.csv", index=False
    )
    for name, log_df in droplogs.items():
        if log_df is not None:
            log_df.to_csv(f"{args.out}/cleaning_log_{name.replace(' ', '_')}.csv", index=False)

    figures = {
        "Fitted tyre degradation with parameter uncertainty": os.path.basename(
            plots.fig_degradation(fits, f"{args.out}/degradation.png")
        ),
        "Late-stint forecasts on held-out laps": os.path.basename(
            plots.fig_forecast(preds["late_stint"], f"{args.out}/late_stint_forecast.png")
        ),
        "Residual diagnostics and coverage by compound (unseen stints)": os.path.basename(
            plots.fig_diagnostics(preds["stint_holdout"], args.alpha, f"{args.out}/diagnostics.png")
        ),
    }
    write_report(
        f"{args.out}/REPORT.md",
        alpha=args.alpha,
        races=races,
        droplogs=droplogs,
        pooled=pooled,
        by_race=by_race,
        params=params,
        figures=figures,
        synthetic=args.synthetic,
        n_trimmed=trimmed,
    )

    pd.set_option("display.width", 200)
    for sc, s in pooled.items():
        print(f"\n{sc}")
        print(
            s[["model", "n", "MAE_s", "RMSE_s", "coverage", "mean_width_s", "interval_score"]]
            .round(3)
            .to_string(index=False)
        )
    print(f"\nWrote tables, figures and REPORT.md to {args.out}/")
    return 0


def strategy(args) -> int:
    """Rank tyre strategies for one race with parameter uncertainty propagated."""
    os.makedirs(args.out, exist_ok=True)
    if args.synthetic:
        name, df = "synthetic-0", synthetic_race(seed=0, name="synthetic-0", outlier_frac=0.02)
    else:
        name = f"{args.year} {args.race}"
        df, _ = _load_one(args, args.year, args.race)
    model = PhysicsLapModel().fit(df)
    res = analyse_strategies(
        model,
        df,
        n_laps=args.laps,
        pit_loss=args.pit_loss,
        max_stops=args.max_stops,
        n_draws=args.draws,
        seed=args.seed,
        step=args.step,
    )
    res.table.to_csv(f"{args.out}/strategy.csv", index=False)
    plots.fig_strategy(res, f"{args.out}/strategy.png")
    head = res.table.head(8).round(3).to_string(index=False)
    with open(f"{args.out}/STRATEGY.md", "w", encoding="utf-8") as fh:
        fh.write(
            f"# Strategy what-if: {name}\n\n"
            f"Race distance {res.n_laps} laps, pit-lane time loss {res.pit_loss:.1f} s "
            "(an input, not an estimate).\n\n"
            "**Assumptions.** Model-based what-if on the fitted physics model: no traffic, "
            "safety cars, track position, tyre warm-up or engine modes; two different dry "
            "compounds required; stint order does not change race time in this model.\n\n"
            f"```\n{head}\n```\n\n![strategy](strategy.png)\n"
        )
    print(
        f"{name}: best plan {res.table.iloc[0]['plan']} "
        f"(P fastest {res.table.iloc[0]['P_fastest']:.0%})"
    )
    print(head)
    return 0


def source_check(args) -> int:
    """Diagnose the data source: does it return what the loader expects?"""
    info = describe(args.year, args.race, args.cache_dir or ".openf1_cache")
    s = info["session"]
    print(
        f"session {s.get('session_key')}: {s.get('country_name')} / {s.get('location')}, "
        f"{s.get('year')}"
    )
    for ep in ("laps", "stints", "pit", "drivers", "race_control"):
        print(f"{ep:>13}: {info[ep]['rows']:>5} rows  fields: {', '.join(info[ep]['fields'])}")
    return 0


def verify(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    vdf = run_verification(args.n_races, outlier_frac=args.outlier_frac)
    vdf.to_csv(f"{args.out}/verification.csv", index=False)
    plots.fig_verification(vdf, f"{args.out}/verification.png")
    cov = vdf.groupby("param")["covered"].mean()
    print("95% CI coverage of known parameters over", args.n_races, "synthetic races")
    print(cov.round(2).to_string())
    print(f"overall: {vdf['covered'].mean():.1%}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="f1lap", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="validate models on race data")
    r.add_argument("--year", type=int, default=2024)
    r.add_argument("--races", nargs="+", default=["Bahrain", "Spain", "Hungary"])
    r.add_argument("--synthetic", action="store_true", help="use generated data (no network)")
    r.add_argument("--synthetic-races", type=int, default=2)
    r.add_argument("--alpha", type=float, default=0.10, help="1 - nominal interval coverage")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--n-boot", type=int, default=300)
    r.add_argument(
        "--source",
        choices=["openf1", "fastf1"],
        default="openf1",
        help="openf1: free, no account (2023+). fastf1: needs an F1TV account",
    )
    r.add_argument("--cache-dir", default=None)
    r.add_argument("--out", default="results")
    r.set_defaults(func=run)
    st = sub.add_parser("strategy", help="rank tyre strategies with propagated uncertainty")
    st.add_argument("--year", type=int, default=2024)
    st.add_argument("--race", default="Bahrain")
    st.add_argument("--synthetic", action="store_true")
    st.add_argument("--laps", type=int, default=None, help="race distance (default: max lap seen)")
    st.add_argument("--pit-loss", type=float, default=22.0, help="pit-lane time loss in seconds")
    st.add_argument("--max-stops", type=int, default=2)
    st.add_argument("--draws", type=int, default=2000)
    st.add_argument("--step", type=int, default=1, help="stint-length grid step (speed)")
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--source", choices=["openf1", "fastf1"], default="openf1")
    st.add_argument("--cache-dir", default=None)
    st.add_argument("--out", default="results")
    st.set_defaults(func=strategy)
    sc = sub.add_parser("source-check", help="diagnose the OpenF1 data source for one race")
    sc.add_argument("--year", type=int, default=2024)
    sc.add_argument("--race", default="Bahrain")
    sc.add_argument("--cache-dir", default=None)
    sc.set_defaults(func=source_check)
    v = sub.add_parser("verify", help="check parameter recovery on synthetic races")
    v.add_argument("--n-races", type=int, default=30)
    v.add_argument("--outlier-frac", type=float, default=0.0)
    v.add_argument("--out", default="results")
    v.set_defaults(func=verify)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
