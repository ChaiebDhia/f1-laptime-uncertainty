"""Figures. All functions save a PNG and return its path."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from .data import COMPOUNDS  # noqa: E402
from .validation import MODEL_LABELS, coverage_by_group  # noqa: E402

COMPOUND_COLORS = {"SOFT": "#d62728", "MEDIUM": "#d8a800", "HARD": "#555555"}
MODEL_COLORS = {"physics": "#1f77b4", "hybrid": "#2ca02c", "gbm": "#d6604d"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_degradation(fits: dict, path):
    """fits: {race: (laps_df, PhysicsLapModel)}"""
    n = len(fits)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4), squeeze=False, sharey=True)
    ages = np.arange(1, 36)
    for ax, (name, (df, model)) in zip(axes[0], fits.items(), strict=True):
        draws = model.degradation_draws(ages)
        for i, c in enumerate(COMPOUNDS):
            lo, mid, hi = np.percentile(draws[:, i], [5, 50, 95], axis=0)
            ax.plot(ages, mid, color=COMPOUND_COLORS[c], label=c)
            ax.fill_between(ages, lo, hi, color=COMPOUND_COLORS[c], alpha=0.2)
            sel = df["compound"] == c
            if sel.any():
                ax.axvline(df.loc[sel, "tyre_age"].max(), color=COMPOUND_COLORS[c], ls=":", lw=1)
        ax.set_title(name)
        ax.set_xlabel("tyre age (laps)")
    axes[0][0].set_ylabel("time lost to degradation (s)")
    axes[0][0].legend(title="dotted: oldest tyre seen", frameon=False)
    fig.suptitle("Fitted tyre degradation with 90% parameter-uncertainty band")
    return _save(fig, path)


def fig_forecast(preds, path, race: str | None = None):
    """Held-out late-stint forecasts for the driver with the most held-out laps."""
    p = preds[preds["valid"]]
    if race:
        p = p[p["race"] == race]
    drv = p["driver"].value_counts().index[0]
    q = p[p["driver"] == drv]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, m in zip(axes, MODEL_LABELS, strict=True):
        for _, s in q.groupby("stint_id"):
            s = s.sort_values("lap")
            ax.fill_between(s["lap"], s[f"{m}_lo"], s[f"{m}_hi"], color=MODEL_COLORS[m], alpha=0.25)
            ax.plot(s["lap"], s[f"{m}_mu"], color=MODEL_COLORS[m])
            ax.scatter(s["lap"], s["lap_time"], color="k", s=12, zorder=3)
        ax.set_title(MODEL_LABELS[m], fontsize=10)
        ax.set_xlabel("race lap")
    axes[0].set_ylabel("lap time (s)")
    fig.suptitle(
        f"Held-out late-stint forecasts, driver {drv} (dots = observed, band = 90% interval)"
    )
    return _save(fig, path)


def fig_diagnostics(preds, alpha: float, path):
    """Residual structure and interval calibration of out-of-sample forecasts."""
    p = preds[preds["valid"]]
    res = p["lap_time"] - p["physics_mu"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    for c in COMPOUNDS:
        s = p["compound"] == c
        ax.scatter(p.loc[s, "tyre_age"], res[s], s=6, alpha=0.4, color=COMPOUND_COLORS[c], label=c)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("tyre age (laps)")
    ax.set_ylabel("residual, physics model (s)")
    ax.set_title("Residuals vs tyre age")
    ax.legend(frameon=False, markerscale=2)
    stats.probplot(res / res.std(), dist="norm", plot=axes[1])
    axes[1].set_title("Residual normal QQ-plot")
    cov = coverage_by_group(preds)
    comps = [c for c in COMPOUNDS if c in set(cov["compound"])]
    width = 0.8 / len(MODEL_LABELS)
    for j, (m, label) in enumerate(MODEL_LABELS.items()):
        vals = [
            cov[(cov["compound"] == c) & (cov["model"] == label)]["coverage"].iloc[0] for c in comps
        ]
        axes[2].bar(
            np.arange(len(comps)) + j * width, vals, width, color=MODEL_COLORS[m], label=label
        )
    axes[2].axhline(1 - alpha, color="k", ls="--", lw=1)
    axes[2].set_xticks(np.arange(len(comps)) + width)
    axes[2].set_xticklabels(comps)
    axes[2].set_ylim(0, 1.45)
    axes[2].set_ylabel("empirical coverage")
    axes[2].set_title(f"Interval coverage by compound (target {1 - alpha:.0%})")
    axes[2].legend(frameon=False, fontsize=7, loc="upper center")
    return _save(fig, path)


def fig_verification(vdf, path):
    """Estimated vs true parameter values (with 95% CIs) over many synthetic races."""
    params = list(vdf["param"].unique())
    fig, axes = plt.subplots(1, len(params), figsize=(2.3 * len(params), 3.6))
    for ax, name in zip(axes, params, strict=True):
        s = vdf[vdf["param"] == name].sort_values("race")
        colors = np.where(s["covered"], "#1f77b4", "#d62728")
        ax.errorbar(
            s["race"],
            s["estimate"],
            yerr=[s["estimate"] - s["ci95_low"], s["ci95_high"] - s["estimate"]],
            fmt="none",
            ecolor=colors,
            elinewidth=0.8,
            alpha=0.7,
        )
        ax.axhline(s["true"].iloc[0], color="k", lw=1)
        ax.set_title(f"{name}\n{s['covered'].mean():.0%} covered", fontsize=8)
        ax.set_xticks([])
    fig.suptitle(
        "Verification on synthetic races: 95% CIs vs known truth (black line); red = missed",
        fontsize=10,
    )
    return _save(fig, path)


def fig_strategy(result, path, top: int = 8):
    """Left: pit-window curve with uncertainty. Right: probability each plan is fastest."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax, c = axes[0], result.pit_curve
    if len(c):
        ax.plot(c["pit_lap"], c["mean_s"], color="#1f77b4")
        ax.fill_between(c["pit_lap"], c["p05_s"], c["p95_s"], color="#1f77b4", alpha=0.25)
        first = result.curve_label.split(" + ")[0]
        ax.set_xlabel(f"laps run on {first} before the stop")
        ax.set_ylabel("race time vs best mean one-stop (s)")
        ax.set_title(f"One-stop pit window, {result.curve_label} (90% band)")
    t = result.table.sort_values("P_fastest", ascending=False).head(top).iloc[::-1]
    colors = ["#bbbbbb" if e else "#1f77b4" for e in t["extrapolates_tyre_life"]]
    axes[1].barh(t["plan"], t["P_fastest"], color=colors)
    axes[1].set_xlabel("probability the plan is the fastest candidate")
    axes[1].set_title(
        f"Strategy ranking, pit loss {result.pit_loss:.0f} s (grey: extrapolates tyre life)"
    )
    return _save(fig, path)
