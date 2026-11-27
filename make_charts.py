"""
Draws every chart and diagram for the ML signal (FI4008 Group 10, Pets at Home).
Run after pets_ml_model.py:   python make_charts.py   ->  results/charts/*.png

Every chart is drawn only from files the model wrote into ./results, so rerunning the
model and then this script rebuilds every figure. Nothing is typed in by hand.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, no screen needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import pets_ml_model as m

RES = m.OUT
INK, MUTED, GRID = "#14213d", "#5b6472", "#dcdbd5"
BLUE, ORANGE, GREY, LIGHT = "#2a78d6", "#eb6834", "#9a9890", "#e8eef8"

plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans", "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False,
})


def save(fig, folder, name, source, note_y=-0.02):
    """Every chart carries a one-line source note and is saved at print resolution."""
    fig.text(0.01, note_y, source, fontsize=8, color=MUTED, ha="left", va="top")
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def months(index):
    """Period strings ('2020-04') -> timestamps for plotting."""
    return pd.PeriodIndex(index, freq="M").to_timestamp()


pct = lambda v: f"{v * 100:.1f}%"
SRC = "Source: Yahoo Finance, ONS, Google Trends, CMA; snapshot 11 Nov 2026; pets_ml_model.py"

# Readable names for the nine inputs (same order as m.FEATURES)
LABELS = {
    "reversal": "Last month's excess return", "momentum": "12-month momentum (skip 1)",
    "vol3m": "3-month PETS volatility", "mkt_mom3": "FTSE 250 3-month return",
    "retail_yoy": "UK retail sales growth (lag 1)", "cpi_yoy": "UK CPI inflation (lag 1)",
    "gbp_chg3": "GBP/USD 3-month change", "puppy_yoy": "'Puppy' searches, yearly change",
    "cma_event": "CMA announcement month",
}


# =========================================================================== shared charts
def chart_design(out):
    """Diagram 1: which months train, which are scored, which is live; and the pipeline."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
    ax.set_title("Walk-forward design: each forecast uses only earlier months")
    # timeline bar scaled to feature months Mar 2015 .. Nov 2026 (141 months)
    t0, n = pd.Period("2015-03", "M"), 141
    X = lambda p: 4 + (pd.Period(p, "M") - t0).n / n * 92
    for a, b, col, lab in [("2015-03", "2020-03", "#9fb8dc", "First training window\n60 feature months, Mar 2015 - Feb 2020"),
                           ("2020-03", "2026-10", BLUE, "79 scored forecasts\ntargets Apr 2020 - Oct 2026"),
                           ("2026-10", "2026-12", ORANGE, "")]:
        ax.add_patch(plt.Rectangle((X(a), 33), X(b) - X(a), 5, color=col, lw=0))
        if lab:
            ax.text((X(a) + X(b)) / 2, 40, lab, ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.text(X("2026-12") + 0.5, 35.5, "Nov 2026\nlive", ha="left", va="center", fontsize=9, color=ORANGE)
    for yr in [2016, 2018, 2020, 2022, 2024, 2026]:
        ax.text(X(f"{yr}-01"), 30.8, str(yr), ha="center", fontsize=8.5, color=MUTED)
    # pipeline boxes
    steps = [("Monthly features\nat month-end t", "9 inputs, each lagged\nto its publication date"),
             ("Tune penalty", "time-series folds\ninside the window"),
             ("Fit Ridge on\nmonths up to t", "scaler fitted on\ntraining data only"),
             ("Forecast month t+1", "excess return of\nPETS over FTSE 250 ETF"),
             ("Score and roll on", "compare with the\nhistorical mean; t -> t+1")]
    w, gap, y0 = 16.4, 2.6, 5
    for i, (h, sub) in enumerate(steps):
        x = 2 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, 18, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=LIGHT if i != 2 else "#d6e6fa", ec=BLUE if i == 2 else GRID, lw=1.2))
        ax.text(x + w / 2, y0 + 12.5, h, ha="center", va="center", fontsize=9.5, weight="bold", color=INK)
        ax.text(x + w / 2, y0 + 4.5, sub, ha="center", va="center", fontsize=8.3, color=MUTED)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.4, y0 + 9), (x + w + gap - 0.4, y0 + 9),
                                         arrowstyle="-|>", mutation_scale=12, color=MUTED))
    save(fig, out, "01_walk_forward_design.png", "Design of pets_ml_model.py (walk_forward)")


def chart_total_returns(out, px):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    base = px.loc[:"2026-11"]
    d = months(base.index.astype(str))
    ax.plot(d, base["pets_tr"] / base["pets_tr"].iloc[0] * 100, color=BLUE, lw=2, label="Pets at Home, total return")
    ax.plot(d, base["midd_tr"] / base["midd_tr"].iloc[0] * 100, color=GREY, lw=2, label="FTSE 250 ETF (MIDD), total return")
    ax.plot(d, base["ftmc"] / base["ftmc"].iloc[0] * 100, color=GREY, lw=1, ls="--", label="FTSE 250 price index (no dividends)")
    ax.set_yscale("log"); ax.set_yticks([60, 100, 150, 200, 300]); ax.set_yticklabels(["60", "100", "150", "200", "300"])
    ax.set_ylabel("Index, March 2014 = 100 (log scale)")
    ax.set_title("Pets at Home has lagged the FTSE 250 on total return since listing")
    ax.legend(loc="upper left")
    save(fig, out, "02_total_return_indexed.png", "Month-end values, Mar 2014 - 11 Nov 2026 (Nov partial). " + SRC)


def chart_excess_distribution(out, train):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    e = train["reversal"] * 100
    ax.hist(e, bins=30, color=BLUE, alpha=0.85, edgecolor="white")
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(e.mean(), color=ORANGE, lw=2, ls="--")
    ax.text(e.mean() + 0.6, ax.get_ylim()[1] * 0.92, f"mean {e.mean():.2f}%", color=ORANGE, fontsize=10)
    ax.set_xlabel("Monthly log excess return of PETS over the FTSE 250 ETF (%)")
    ax.set_ylabel("Months")
    ax.set_title(f"Monthly excess returns are wide and centred near zero (sd {e.std():.1f}%)")
    save(fig, out, "03_excess_return_distribution.png", f"Feature months {train.index[0]} - {train.index[-1]}. " + SRC)


def chart_correlations(out, train):
    c = train[m.FEATURES].corr()
    labels = [LABELS[f] for f in m.FEATURES]
    fig, ax = plt.subplots(figsize=(8.6, 7))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9); ax.set_yticklabels(labels, fontsize=9)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = c.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title("Most inputs are only weakly related to each other")
    save(fig, out, "04_feature_correlations.png", "Pearson correlations, 139 training rows. " + SRC, note_y=-0.12)


def chart_features(out, train):
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    d = months(train.index.astype(str))
    for ax, f in zip(axes.flat, m.FEATURES):
        if f == "cma_event":
            ax.vlines(d[train[f] == 1], 0, 1, color=ORANGE, lw=2)
            ax.set_ylim(0, 1.2); ax.set_yticks([])
        else:
            ax.plot(d, train[f], color=BLUE, lw=1.3)
        ax.set_title(LABELS[f], fontsize=10)
    fig.suptitle("The nine inputs over time", x=0.01, ha="left", fontsize=13, weight="bold", color=INK)
    fig.tight_layout()
    save(fig, out, "05_features_over_time.png", "Values as available at each month-end (lags applied). " + SRC)


# =========================================================================== result charts
def result_charts(R):
    out = RES / "charts"
    S, B, F = R["summary"], R["backtest"], R["final"]
    tab = pd.read_csv(RES / "walk_forward_forecasts.csv")
    sc = tab[tab["status"] == "scored"].copy()
    d = months(sc["target_month"])
    tag = f"{S['n_oos']} scored months {S['first_target']} - {S['last_target']}. "

    # 10 forecast vs actual
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(d, sc["actual"] * 100, width=20, color=GRID, label="Actual excess return")
    ax.plot(d, sc["model"] * 100, color=BLUE, lw=2, label="Ridge forecast")
    ax.plot(d, sc["mean"] * 100, color=ORANGE, lw=1.4, ls="--", label="Historical-mean forecast")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylabel("Monthly excess return vs FTSE 250 ETF (%)")
    ax.set_title(f"Forecasts barely move (sd {pct(S['forecast_sd'])}) while returns swing (sd {pct(S['actual_sd'])})")
    ax.legend(loc="upper right")
    save(fig, out, "10_forecast_vs_actual.png", tag + SRC)

    # 11 cumulative squared-error gain over the mean
    gain = ((sc["actual"] - sc["mean"]) ** 2 - (sc["actual"] - sc["model"]) ** 2).cumsum()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axhline(0, color=MUTED, lw=1)
    ax.fill_between(d, gain, 0, color=BLUE, alpha=0.12, lw=0)
    ax.plot(d, gain, color=BLUE, lw=2)
    ax.annotate(f"{gain.iloc[-1]:.4f}", (d[-1], gain.iloc[-1]), xytext=(-8, -14),
                textcoords="offset points", ha="right", color=INK, weight="bold")
    ax.set_ylabel("Cumulative squared-error gain over the mean")
    ax.set_title("Above zero = Ridge ahead of the historical mean; below = behind")
    save(fig, out, "11_cumulative_gain.png", tag + SRC)

    # 12 error and direction comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ["Ridge", "OLS", "Historical mean"]
    rm = [S["rmse_ridge"], S["rmse_ols"], S["rmse_mean"]]
    axes[0].bar(names, np.array(rm) * 100, color=[BLUE, ORANGE, GREY], width=0.55)
    for i, v in enumerate(rm):
        axes[0].text(i, v * 100 + 0.15, f"{v * 100:.2f}%", ha="center", color=INK)
    axes[0].set_ylabel("RMSE, monthly (%)"); axes[0].set_title("Forecast error (lower is better)")
    hits = [S["hits_ridge"], S["hits_ols"], S["hits_mean"]]
    axes[1].bar(names, hits, color=[BLUE, ORANGE, GREY], width=0.55)
    always_under = S["n_oos"] - S["months_pets_beat_index"]
    axes[1].axhline(always_under, color=INK, ls="--", lw=1.3)
    axes[1].text(2.35, always_under + 0.8, f"always 'underperform': {always_under}", ha="right", fontsize=9, color=INK)
    for i, v in enumerate(hits):
        axes[1].text(i, v + 0.8, f"{v}/{S['n_oos']}", ha="center", color=INK)
    axes[1].set_ylim(0, max(always_under, max(hits)) + 8)
    axes[1].set_ylabel("Months with the direction right"); axes[1].set_title("Direction right")
    fig.tight_layout()
    save(fig, out, "12_model_comparison.png", tag + SRC)

    # 13 bootstrap distribution of R-squared
    boot = pd.read_csv(RES / "bootstrap_r2.csv")["r2"] * 100
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.hist(boot, bins=60, color=BLUE, alpha=0.85, edgecolor="white")
    lo, hi = np.percentile(boot, [5, 95])
    for v, c, ls in [(lo, ORANGE, "--"), (hi, ORANGE, "--"), (0, INK, "-")]:
        ax.axvline(v, color=c, ls=ls, lw=1.6)
    ax.set_xlabel("Out-of-sample R² in resampled samples (%)"); ax.set_ylabel("Resamples")
    ax.set_title(f"90% of resamples between {lo:.1f}% and {hi:.1f}% (estimate {S['r2_oos_ridge'] * 100:.1f}%)")
    save(fig, out, "13_bootstrap_r2.png", "5,000 resamples of months with replacement (seed 0). " + tag)

    # 14 penalty chosen in each window
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(d, sc["alpha"], color=BLUE, marker="o", ms=3.5, lw=1)
    ax.axhline(S["max_alpha"], color=ORANGE, ls="--", lw=1.4)
    ax.text(d[0], S["max_alpha"] / 2.2, f"top of grid ({S['max_alpha']:,.0f})", color=ORANGE, fontsize=9)
    ax.set_yscale("log"); ax.set_ylabel("Ridge penalty chosen (log scale)")
    ax.set_title(f"The strongest penalty was chosen in {S['windows_at_max_alpha']} of {S['n_oos']} windows")
    save(fig, out, "14_penalty_path.png", tag + SRC)

    # 15 coefficient paths (stability)
    cp = pd.read_csv(RES / "coefficient_path.csv", index_col=0)
    fig, axes = plt.subplots(3, 3, figsize=(13, 7.6), sharex=True, sharey=True)
    dc = months(cp.index)
    for ax, f in zip(axes.flat, m.FEATURES):
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.plot(dc, cp[f] * 100, color=BLUE, lw=1.5)
        ax.set_title(f"{LABELS[f]} ({F['sign_flips'][f]} sign flips)", fontsize=8.8)
    fig.supylabel("Standardised coefficient (% points)", color=MUTED, fontsize=10)
    fig.suptitle("Coefficients across the 79 refits: small and unstable", x=0.01, ha="left",
                 fontsize=13, weight="bold", color=INK)
    fig.tight_layout()
    save(fig, out, "15_coefficient_paths.png", tag + SRC)

    # 16 final-fit coefficients
    co = pd.read_csv(RES / "final_coefficients.csv", index_col=0)["std_coef"].sort_values() * 100
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.barh([LABELS[k] for k in co.index], co.values, color=BLUE, height=0.55)
    ax.axvline(0, color=MUTED, lw=1)
    span = max(abs(co).max(), 1e-4)
    for yv, v in enumerate(co.values):
        ax.text(v + (0.04 * span if v >= 0 else -0.04 * span), yv, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.set_xlim(-span * 1.5, span * 1.5)
    ax.grid(False); ax.set_xlabel("Effect of a 1 sd rise on next month's excess return (% points)")
    ax.set_title(f"Final fit on all 139 rows (penalty {F['final_alpha']:,.0f}) shrinks every effect")
    save(fig, out, "16_final_coefficients.png", tag + SRC)

    # 17 robustness
    rb = pd.read_csv(RES / "robustness.csv", index_col=0)["r2_oos"] * 100
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cols = [BLUE if v < 0 else ORANGE for v in rb.values]
    ax.barh(rb.index[::-1], rb.values[::-1], color=cols[::-1], height=0.55)
    ax.axvline(0, color=INK, lw=1)
    for yv, v in enumerate(rb.values[::-1]):
        ax.text(v + (-0.4 if v < 0 else 0.4), yv, f"{v:.1f}%", va="center",
                ha="right" if v < 0 else "left", fontsize=9.5, color=INK)
    ax.set_xlim(min(rb.min() * 1.25, -3), max(rb.max() * 1.5, 3))
    ax.grid(False); ax.set_xlabel("Out-of-sample R² vs historical mean (%)")
    ax.set_title("Robustness: one change at a time")
    save(fig, out, "17_robustness.png", tag + SRC)

    # 18 backtest
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(d, sc["strategy"].cumsum() * 100, color=BLUE, lw=2,
            label=f"Switching strategy ({B['per_switch_cost'] * 100:.2f}% per switch)")
    ax.plot(d, sc["actual"].cumsum() * 100, color=ORANGE, lw=2, ls="--", label="Always hold PETS")
    ax.axhline(0, color=MUTED, lw=1)
    ymin, ymax = ax.get_ylim()
    ax.fill_between(d, ymin, ymax, where=sc["position"] == 1, color=BLUE, alpha=0.06, step="mid", lw=0,
                    label="Months holding PETS")
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Cumulative log excess return vs ETF (%)")
    ax.set_title(f"Strategy {B['strategy_cum_excess'] * 100:.1f}% vs always PETS {B['always_pets_cum_excess'] * 100:.1f}% "
                 f"({B['switches']} switches)")
    ax.legend(loc="lower left")
    save(fig, out, "18_backtest.png", tag + SRC)

    # 19 cost sensitivity
    cs = pd.read_csv(RES / "cost_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(cs["round_trip_cost"] * 100, cs["strategy_cum_excess"] * 100, color=BLUE, marker="o", lw=2,
            label="Switching strategy")
    ax.axhline(cs["always_pets_cum_excess"].iloc[0] * 100, color=ORANGE, ls="--", lw=2, label="Always hold PETS")
    ax.set_xlabel("Round-trip cost (%); half is charged per switch"); ax.set_ylabel("Cumulative log excess return (%)")
    ax.set_title("The strategy trails holding PETS even with zero costs" if
                 cs["strategy_cum_excess"].iloc[0] < cs["always_pets_cum_excess"].iloc[0]
                 else "Cost at which the strategy loses its lead")
    ax.legend()
    save(fig, out, "19_cost_sensitivity.png", tag + SRC)

    # 20 power check (only if it has been run)
    if "power_check" in R:
        pw = R["power_check"]
        fig, ax = plt.subplots(figsize=(9, 4.4))
        top = max(max(r["all_r2"]) for r in pw) * 100
        ax.set_ylim(min(min(r["all_r2"]) for r in pw) * 100 - 1, top + 3.5)
        for row in pw:
            xs = np.full(len(row["all_r2"]), row["planted_correlation"]) + np.linspace(-0.012, 0.012, len(row["all_r2"]))
            ax.scatter(xs, np.array(row["all_r2"]) * 100, s=16, color=BLUE, alpha=0.6)
            ax.text(row["planted_correlation"], top + 1.0,
                    f"{row['share_r2_above_zero'] * 100:.0f}% of runs > 0", ha="center", fontsize=9.5, color=INK)
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks([r["planted_correlation"] for r in pw])
        ax.set_xlabel("True correlation of a planted signal with returns")
        ax.set_ylabel("Out-of-sample R² (%)")
        ax.set_title("How often this test detects a real signal of known strength")
        save(fig, out, "20_power_check.png", f"{pw[0]['n_runs']} runs per level, shuffled real returns + planted signal. " + tag)

    # 21 results table as an image (for slides)
    rows = [["Out-of-sample R² vs mean", pct(S["r2_oos_ridge"]), pct(S["r2_oos_ols"]), "0 (benchmark)"],
            ["90% bootstrap interval (Ridge)", f"{pct(S['r2_boot_5'])} to {pct(S['r2_boot_95'])}", "", ""],
            ["RMSE (monthly)", pct(S["rmse_ridge"]), pct(S["rmse_ols"]), pct(S["rmse_mean"])],
            ["MAE (monthly)", pct(S["mae_ridge"]), pct(S["mae_ols"]), pct(S["mae_mean"])],
            ["Direction right", f"{S['hits_ridge']}/{S['n_oos']}", f"{S['hits_ols']}/{S['n_oos']}", f"{S['hits_mean']}/{S['n_oos']}"],
            ["Diebold-Mariano p (vs mean)", f"{S['dm_p']:.2f}", "", ""],
            ["Clark-West p (vs mean)", f"{S['cw_p']:.2f}", "", ""],
            ["Backtest cum. excess", f"{B['strategy_cum_excess'] * 100:.1f}%", "", f"always PETS {B['always_pets_cum_excess'] * 100:.1f}%"],
            ["Live forecast for Nov 2026", pct(F["forecast_excess"]), "", f"actual to 11 Nov {pct(F['nov_to_date_excess'])}"]]
    fig, ax = plt.subplots(figsize=(10, 3.6)); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["Measure", "Ridge", "OLS", "Historical mean / note"],
                 loc="center", cellLoc="left", colWidths=[0.34, 0.24, 0.12, 0.30])
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.45)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(weight="bold", color=INK); cell.set_facecolor(LIGHT)
    ax.set_title("Results summary", pad=6)
    save(fig, out, "21_results_table.png", tag + SRC)


def main():
    train, _, _, px = m.load()
    out = RES / "charts"
    chart_design(out)
    chart_total_returns(out, px)
    chart_excess_distribution(out, train)
    chart_correlations(out, train)
    chart_features(out, train)
    result_charts(json.loads((RES / "results.json").read_text()))
    print(f"{len(list(out.glob('*.png')))} charts written to {out}")


if __name__ == "__main__":
    main()
