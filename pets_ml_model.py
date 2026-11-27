"""
Pets at Home (LSE: PETS) — walk-forward Ridge signal for the FI4008 group pitch (Group 10).

This script produces every number in the "Machine learning signal" and "Interpreting
the model" sections of our group report. The report recommends SELL on valuation; the
ML signal gets a weight of 0% because, under the rule we agreed before seeing any
results, it failed to beat a simple historical-mean forecast.

Pre-set weighting rule (agreed before the first run):
  - out-of-sample R^2 > 0 AND significant      -> about 25% weight in the call
  - out-of-sample R^2 > 0 but not significant  -> supporting evidence only
  - out-of-sample R^2 <= 0                     -> 0% weight   <- this is what happened

What this script does:

- It predicts PETS next-month log excess return over the FTSE 250 (iShares FTSE 250
    ETF, total return) using an expanding-window Ridge regression, refit every month
    on past data only.
- It benchmarks against plain OLS and the expanding historical mean.
- It tests significance (Diebold-Mariano, Clark-West, bootstrap interval).
- Runs a simple switching backtest with robustness checks, and a final fit to
    generate the live November 2026 forecast and a 12-month tilt.
- Optional (--power): a power check that asks how often this test would detect a
    real signal of known strength.

Usage:
  python pets_ml_model.py            (about 1 minute; expects ./data, writes ./results)
  python pets_ml_model.py --power    (adds the power check, about 30 minutes)
  python make_charts.py              (charts into ./results/charts)
  python tests/test_pipeline.py      (checks: no look-ahead, placebo, data checksums)

Snapshot: data taken on 11 Nov 2026 (full details in DATA_SOURCES.md)

  data/prices_monthly.csv   Month-end total returns (pets_tr = PETS.L, midd_tr =
                            iShares FTSE 250 UCITS ETF MIDD.L), FTSE 250 price
                            index (ftmc), GBP/USD (gbpusd), and 63-day annualised
                            PETS volatility (vol63).

  data/ons_raw.json         ONS D7G7 (CPI annual %) and J5EK (retail sales volume).

  data/google_trends_gb.csv UK monthly search interest for "puppy" (and pet insurance).
                            Only "puppy" is used, as an input to all models;
                            "pet insurance" is not used.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Resolve paths relative to this script so it works regardless of working directory
HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "results"
OUT.mkdir(exist_ok=True)  # make sure output folder exists before saving csv/json

LAST_FULL_MONTH = "2026-10"      # Nov 2026 is incomplete in the data snapshot (ends 11 Nov)
START = 60                       # First 60 rows (Mar 2015 - Feb 2020) train the first model
ALPHAS = np.logspace(-2, 4, 30)  # Candidate Ridge penalties (0.01 ... 10,000) searched via CV
COST = 0.005                     # 0.5% round-trip cost (0.25% / half charged per switch)
SEED = 0                         # Fixed seed for the bootstrap and the power check

# Key UK CMA veterinary market review / investigation dates (month of release).
# CMA: the UK's regulator for competition, mergers and consumer protection.
# Checked on 23 Nov 2026 against gov.uk/cma-cases/veterinary-services-market-for-pets-review
CMA_MONTHS = [
    "2023-09",  # 7 Sep 2023: review launched
    "2024-03",  # 12 Mar 2024: formal market investigation proposed
    "2024-05",  # 23 May 2024: market investigation reference made
    "2025-06",  # 10 Jun 2025: timetable extended
    "2025-10",  # 15 Oct 2025: provisional decision
    "2026-03",  # 24 Mar 2026: final decision
    "2026-09",  # 22 Sep 2026: final orders
]

# Core feature set: mix of price/vol signals, macro, search trend and regulatory dummy
FEATURES = [
    "reversal", "momentum", "vol3m", "mkt_mom3",
    "retail_yoy", "cpi_yoy", "gbp_chg3", "puppy_yoy", "cma_event",
]

PRICE_ONLY = FEATURES[:4]  # purely market/price based subset for comparison

# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------

def load_inputs():
    """Read the frozen snapshot files (nothing is downloaded here)."""
    px = pd.read_csv(DATA / "prices_monthly.csv", index_col="ym")
    px.index = pd.PeriodIndex(px.index, freq="M")  # monthly periods make lags & shifts clean

    ons = json.loads((DATA / "ons_raw.json").read_text())
    to_series = lambda key: pd.Series(
        {pd.Period(ym, "M"): float(val) for ym, val in ons[key]["data"].items()}
    )
    cpi, rsi = to_series("cpi"), to_series("rsi")

    gt = pd.read_csv(DATA / "google_trends_gb.csv", index_col="ym")
    gt.index = pd.PeriodIndex(gt.index, freq="M")
    return px, cpi, rsi, gt


def build_features(px, cpi, rsi, gt, benchmark="midd_tr", last_full_month=LAST_FULL_MONTH):
    """Build features and the next-month target.

    Timing rule: each row is dated by its feature month t. Everything in the row must be
    known at the close of month t; the target is the excess return over month t+1.
    tests/test_pipeline.py checks this by cutting the data off at t and rebuilding.
    """
    # Log returns keep additions clean and avoid compounding quirks
    r_p = np.log(px["pets_tr"]).diff()
    r_m = np.log(px[benchmark]).diff()
    ex = r_p - r_m  # Monthly log excess return vs chosen benchmark

    X = pd.DataFrame(index=px.index)
    X["reversal"] = ex                                # Current month's excess return
    X["momentum"] = ex.shift(1).rolling(11).sum()     # Prior 11 months (t-11..t-1), skipping t
    X["vol3m"] = px["vol63"]                          # Recent realised volatility (63d annualised)
    X["mkt_mom3"] = r_m.rolling(3).sum()              # 3-month benchmark log return

    # Shift ONS series by 1 month: month m data typically lands in the third week of m+1
    X["retail_yoy"] = ((rsi / rsi.shift(12) - 1) * 100).reindex(px.index).shift(1)
    X["cpi_yoy"] = cpi.reindex(px.index).shift(1)

    X["gbp_chg3"] = np.log(px["gbpusd"]).diff(3)      # 3-month GBP/USD log move

    # YoY log change neutralises seasonal spikes and Google's relative 0-100 rescaling
    X["puppy_yoy"] = np.log(gt["puppy"] / gt["puppy"].shift(12)).reindex(px.index)
    X["cma_event"] = [1.0 if str(p) in CMA_MONTHS else 0.0 for p in px.index]  # event dummy

    # Trim to last complete month before shifting so partial-month returns don't enter y
    ex_full = ex.loc[:last_full_month]
    y = ex_full.shift(-1).rename("y")  # Next month's excess return

    full = pd.concat([X, y], axis=1).loc[:last_full_month]
    train = full.dropna()              # Only rows where both features and target exist
    latest = full[FEATURES].iloc[[-1]] # Feature row at end of the last full month
    return train, latest, ex_full


def load(benchmark="midd_tr"):
    """Load prices, macro series and search data; build features and next-month target.
    Report sample: 139 rows = feature months Mar 2015 - Sep 2026 (targets Apr 2015 - Oct 2026)."""
    px, cpi, rsi, gt = load_inputs()
    train, latest, ex_full = build_features(px, cpi, rsi, gt, benchmark)
    return train, latest, ex_full, px


# -----------------------------------------------------------------------------
# Model builders & walk-forward engine
# -----------------------------------------------------------------------------

def ridge():
    # Scaler fitted once per training window (not per tuning fold; see CHANGELOG).
    # RidgeCV then picks the penalty by 5-fold time-series CV inside that window.
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(5)))


def ols():
    # Unpenalised linear benchmark on scaled features (same scale as Ridge for comparison)
    return make_pipeline(StandardScaler(), LinearRegression())


def lasso():
    # L1 alternative to see if sparse feature selection beats L2 shrinkage
    return make_pipeline(StandardScaler(), LassoCV(cv=TimeSeriesSplit(5), max_iter=50000))


def walk_forward(data, cols, make_model=ridge, start=START, gap=0, target="y", keep_coefs=False):
    """Run an expanding-window forecast without peeking ahead.

    When predicting row t, only rows up to t - 1 - gap are available for fitting.
    Use gap > 0 for multi-month targets whose full outcome isn't known yet at t-1.
    With start=60 the first forecast is for Apr 2020 and the last for Oct 2026 (79 in all).
    """
    X, y = data[cols], data[target]
    rows, coefs = [], []

    for t in range(start, len(X)):
        # Slice strictly prior data so no future info leaks into training or scaling
        X_tr, y_tr = X.iloc[: t - gap], y.iloc[: t - gap]
        m = make_model().fit(X_tr, y_tr)
        o = ols().fit(X_tr, y_tr)

        rows.append({
            "feature_month": str(X.index[t]),            # month-end at which the forecast is made
            "train_first": str(X.index[0]),             # training window used for this forecast
            "train_last": str(X.index[len(X_tr) - 1]),
            "actual": y.iloc[t],
            "model": m.predict(X.iloc[[t]])[0],
            "ols": o.predict(X.iloc[[t]])[0],
            "mean": y_tr.mean(),                        # expanding historical-mean forecast
            "alpha": getattr(m[-1], "alpha_", np.nan),  # OLS/Lasso won't have alpha_
        })
        if keep_coefs:
            coefs.append(m[-1].coef_)

    res = pd.DataFrame(rows, index=X.index[start:])
    # Shift index forward so each row is stamped with the month being forecast
    res.index = (res.index + 1 + gap).rename("target_month")

    if keep_coefs:
        return res, pd.DataFrame(coefs, index=res.index, columns=cols)
    return res


# -----------------------------------------------------------------------------
# Evaluation metrics & backtest
# -----------------------------------------------------------------------------

def r2_oos(res, col="model"):
    """Out-of-sample R^2 relative to the expanding historical mean (Campbell & Thompson, 2008)."""
    sse_model = ((res["actual"] - res[col]) ** 2).sum()
    sse_bench = ((res["actual"] - res["mean"]) ** 2).sum()
    return 1 - sse_model / sse_bench  # negative means worse than expanding mean


def dm_test(res, col="model"):
    """One-sided Diebold-Mariano test (with HLN small-sample adjustment).

    Tests whether the model has lower squared forecast error than the expanding mean.
    """
    d = (res["actual"] - res["mean"]) ** 2 - (res["actual"] - res[col]) ** 2
    n = len(d)
    stat = d.mean() / np.sqrt(d.var(ddof=1) / n) * np.sqrt((n - 1) / n)
    p_val = 1 - stats.t.cdf(stat, n - 1)
    return stat, p_val


def cw_test(res, col="model"):
    """One-sided Clark-West test tailored for comparing nested forecasts."""
    # Clark-West adjusts for estimation noise when comparing against a nested null
    f = (
        (res["actual"] - res["mean"]) ** 2
        - ((res["actual"] - res[col]) ** 2 - (res["mean"] - res[col]) ** 2)
    )
    stat = f.mean() / np.sqrt(f.var(ddof=1) / len(f))
    p_val = 1 - stats.norm.cdf(stat)
    return stat, p_val


def significance_tables(res, col="model"):
    """Show the working behind R^2, Diebold-Mariano and Clark-West, so each number can be
    recalculated by hand (or in Excel) from results/significance_by_month.csv.

    Per month:  e_model2 = (actual - model)^2,  e_mean2 = (actual - mean)^2
                d  = e_mean2 - e_model2                    (DM loss differential; > 0 = model better)
                f  = e_mean2 - (e_model2 - (mean - model)^2)  (Clark-West adjusted differential)
    Tests:      DM = mean(d) / (sd(d)/sqrt(n)) * sqrt((n-1)/n),  p from t with n-1 df (one-sided)
                CW = mean(f) / (sd(f)/sqrt(n)),                  p from the normal (one-sided)
    """
    a, p, b = res["actual"], res[col], res["mean"]
    by_month = pd.DataFrame({
        "actual": a, "model_forecast": p, "mean_forecast": b,
        "e_model2": (a - p) ** 2, "e_mean2": (a - b) ** 2,
    })
    by_month["d_dm"] = by_month["e_mean2"] - by_month["e_model2"]
    by_month["f_cw"] = by_month["e_mean2"] - (by_month["e_model2"] - (b - p) ** 2)

    n = len(by_month)
    rows = []
    for name, col_, hln, dist in [("Diebold-Mariano (HLN)", "d_dm", True, "t"),
                                  ("Clark-West", "f_cw", False, "normal")]:
        x = by_month[col_]
        se = x.std(ddof=1) / np.sqrt(n)
        stat = x.mean() / se * (np.sqrt((n - 1) / n) if hln else 1.0)
        p_val = 1 - (stats.t.cdf(stat, n - 1) if dist == "t" else stats.norm.cdf(stat))
        rows.append({"test": name, "n": n, "mean": x.mean(), "sd": x.std(ddof=1),
                     "std_error": se, "hln_factor": np.sqrt((n - 1) / n) if hln else 1.0,
                     "statistic": stat, "reference": f"t({n - 1})" if dist == "t" else "N(0,1)",
                     "p_one_sided": p_val})
    summary_tab = pd.DataFrame(rows)
    # R^2 from the same columns, so the headline number can be checked too
    summary_tab.attrs["r2"] = 1 - by_month["e_model2"].sum() / by_month["e_mean2"].sum()
    return by_month, summary_tab


def bootstrap_r2(res, col="model", n=5000, seed=SEED):
    """Resample OOS months with replacement; returns all R^2 draws.
    Simplification: months are treated as independent (no block structure)."""
    rng = np.random.default_rng(seed)  # fixed seed so intervals are reproducible
    a, p, b = res["actual"].values, res[col].values, res["mean"].values
    idx = rng.integers(0, len(a), size=(n, len(a)))
    return 1 - ((a[idx] - p[idx]) ** 2).sum(1) / ((a[idx] - b[idx]) ** 2).sum(1)


def backtest(res, col="model", cost=COST):
    """Simple long-PETS vs long-FTSE-250 rule driven by positive predicted excess return.

    Holding the benchmark gives 0 excess return by construction. Each switch (entering
    or exiting PETS) pays half the round-trip transaction cost (0.25% per switch when COST=0.005).
    """
    pos = (res[col] > 0).astype(float)  # 1.0 = PETS, 0.0 = FTSE 250 benchmark
    switches = pos.diff().abs().fillna(pos.iloc[0])
    strat = pos * res["actual"] - switches * (cost / 2)

    ann_ir = lambda s: (s.mean() / s.std()) * np.sqrt(12) if s.std() > 0 else np.nan
    metrics = {
        "round_trip_cost": cost,
        "per_switch_cost": cost / 2,
        "strategy_cum_excess": strat.sum(),
        "always_pets_cum_excess": res["actual"].sum(),
        "strategy_info_ratio": ann_ir(strat),
        "always_pets_info_ratio": ann_ir(res["actual"]),
        "months_in_pets": int(pos.sum()),
        "switches": int(switches.sum()),
    }
    return metrics, strat, pos


def summary(res):
    """Collect headline accuracy, statistical test, and stability numbers."""
    a = res["actual"]
    err = lambda c: a - res[c]
    hits = lambda c: int((np.sign(res[c]) == np.sign(a)).sum())  # months with direction right
    dm_stat, dm_p = dm_test(res)
    cw_stat, cw_p = cw_test(res)
    boot_lo, boot_hi = np.percentile(bootstrap_r2(res), [5, 95])
    n_at_max_alpha = int(np.isclose(res["alpha"], ALPHAS.max(), rtol=0, atol=1e-12).sum())

    return {
        "n_oos": len(res),
        "first_target": str(res.index[0]),
        "last_target": str(res.index[-1]),
        "r2_oos_ridge": r2_oos(res),
        "r2_oos_ols": r2_oos(res, "ols"),
        "r2_boot_5": boot_lo,
        "r2_boot_95": boot_hi,
        "rmse_ridge": np.sqrt((err("model") ** 2).mean()),
        "rmse_ols": np.sqrt((err("ols") ** 2).mean()),
        "rmse_mean": np.sqrt((err("mean") ** 2).mean()),
        "mae_ridge": err("model").abs().mean(),
        "mae_ols": err("ols").abs().mean(),
        "mae_mean": err("mean").abs().mean(),
        "hits_ridge": hits("model"),
        "hits_ols": hits("ols"),
        "hits_mean": hits("mean"),
        "hit_ridge": hits("model") / len(res),
        "hit_ols": hits("ols") / len(res),
        "hit_mean": hits("mean") / len(res),
        "months_pets_beat_index": int((a > 0).sum()),  # "always underperform" is right in the rest
        "share_up_months": (a > 0).mean(),
        "dm_stat": dm_stat,
        "dm_p": dm_p,
        "cw_stat": cw_stat,
        "cw_p": cw_p,
        "median_alpha": res["alpha"].median(),
        "max_alpha": float(ALPHAS.max()),
        "windows_at_max_alpha": n_at_max_alpha,
        "forecast_sd": float(res["model"].std()),
        "actual_sd": float(a.std()),
    }


def power_check(rhos=(0.1, 0.2, 0.3), n_seeds=20):
    """How often would this exact pipeline detect a REAL signal of known strength?

    The real excess returns are shuffled (destroying any signal) and an artificial
    signal, proportional to the 'reversal' feature, is added with true correlation rho.
    The whole walk-forward is rerun and we count how often R^2 comes out above zero.
    This backs the report's statement that the test has limited power.
    """
    train, _, _, _ = load()
    z = ((train["reversal"] - train["reversal"].mean()) / train["reversal"].std()).values
    rows = []
    for rho in rhos:
        r2 = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            t = train.copy()
            noise = rng.permutation(train["y"].values)
            t["y"] = noise + rho / np.sqrt(1 - rho ** 2) * train["y"].std() * z
            r2.append(float(r2_oos(walk_forward(t, FEATURES))))
        rows.append({"planted_correlation": rho, "n_runs": n_seeds,
                     "median_r2_oos": float(np.median(r2)),
                     "share_r2_above_zero": float(np.mean(np.array(r2) > 0)),
                     "all_r2": r2})
        print(f"  rho={rho}: median R2 {rows[-1]['median_r2_oos']:.4f}, "
              f"share of runs > 0: {rows[-1]['share_r2_above_zero']:.2f}")
    return rows


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main(run_power=False):
    train, latest, ex, px = load()
    print(
        f"Training rows: {len(train)} ({train.index[0]} to {train.index[-1]}), "
        f"features: {len(FEATURES)}"
    )

    # Save underlying dataset slices for auditability
    train.to_csv(OUT / "model_dataset.csv")
    train.describe().T.to_csv(OUT / "summary_stats.csv")
    train[FEATURES].corr().to_csv(OUT / "feature_correlations.csv")

    # 1. Primary walk-forward evaluation and backtest
    res, coef_path = walk_forward(train, FEATURES, keep_coefs=True)
    S = summary(res)
    bt, strat, pos = backtest(res)
    pd.Series(bootstrap_r2(res), name="r2").to_csv(OUT / "bootstrap_r2.csv", index=False)

    # Working behind the test statistics and p-values (one row per scored month + summary)
    sig_by_month, sig_summary = significance_tables(res)
    sig_by_month.to_csv(OUT / "significance_by_month.csv")
    sig_summary.to_csv(OUT / "significance_summary.csv", index=False)
    print(f"Check: R2 from significance_by_month.csv = {sig_summary.attrs['r2'] * 100:.2f}%")

    # Cost sensitivity: same rule at different round-trip costs (0% ... 2%)
    pd.DataFrame([backtest(res, cost=c)[0] for c in [0.0, 0.002, 0.005, 0.01, 0.02]]) \
        .to_csv(OUT / "cost_sensitivity.csv", index=False)

    # 2. Robustness checks (varying one design choice at a time), intervals + lasso
    pandemic = pd.period_range("2020-03", "2021-06", freq="M")  # COVID lockdown/rebound window
    rob = {
        "Main model (Ridge, 9 features)": r2_oos(res),
        "Lasso instead of Ridge": r2_oos(walk_forward(train, FEATURES, lasso)),
        "Price features only (4)": r2_oos(walk_forward(train, PRICE_ONLY)),
        "Scored without Mar 2020 - Jun 2021": r2_oos(res[~res.index.isin(pandemic)]),
    }

    t2, _, _, _ = load("ftmc")  # price index instead of ETF total return
    rob["Benchmark = FTSE 250 price index"] = r2_oos(walk_forward(t2, FEATURES))

    t3 = train.copy()
    t3["y3"] = (ex.shift(-1) + ex.shift(-2) + ex.shift(-3)).reindex(t3.index)
    # gap=2 avoids overlapping 3-month returns leaking across train/test boundary (inner tuning folds have no gap)
    rob["3-month horizon"] = r2_oos(walk_forward(t3.dropna(), FEATURES, gap=2, target="y3"))

    pd.Series(rob, name="r2_oos").to_csv(OUT / "robustness.csv")

    # 3. Fit on all complete history to score the next month after LAST_FULL_MONTH
    final = ridge().fit(train[FEATURES], train["y"])
    coefs = pd.Series(final[-1].coef_, index=FEATURES)
    coefs.to_csv(OUT / "final_coefficients.csv", header=["std_coef"])

    fc = final.predict(latest)[0]
    # Feature contribution = standardised coef * scaled feature value
    contrib = pd.Series(final[-1].coef_ * final[0].transform(latest)[0], index=FEATURES)
    resid_sd = (res["actual"] - res["model"]).std()

    F = {
        "forecast_month": str(latest.index[0] + 1),
        "forecast_excess": fc,
        "final_alpha": final[-1].alpha_,
        "intercept": final[-1].intercept_,
        # Annualise the average of the last 5 OOS forecasts plus the new forecast.
        # Reported for completeness only: the model has no edge, so this is not used.
        "tilt_12m": np.mean(res["model"].tail(5).tolist() + [fc]) * 12,
        "band_12m": resid_sd * np.sqrt(12),   # rough +/- 1 sd range over 12 months
        "resid_sd_monthly": resid_sd,
        "sign_flips": {c: int((np.sign(coef_path[c]).diff().abs() > 0).sum()) for c in FEATURES},
        "latest_features": latest.iloc[0].to_dict(),
        "contributions": contrib.to_dict(),
    }

    # Check realised excess return so far in Nov 2026 if present in snapshot
    nov_period = pd.Period("2026-11", "M")
    if nov_period in px.index and pd.Period("2026-10", "M") in px.index:
        nov = px.loc[nov_period]
        oct_ = px.loc[pd.Period("2026-10", "M")]
        F["nov_to_date_excess"] = (
            np.log(nov["pets_tr"] / oct_["pets_tr"])
            - np.log(nov["midd_tr"] / oct_["midd_tr"])
        )

    # 4. Dated forecast table: one row per scored forecast, plus the live November row
    table = res.assign(position=pos, strategy=strat).copy()
    table.insert(0, "target_month", table.index.astype(str))
    table["status"] = "scored"
    live = {"target_month": F["forecast_month"], "feature_month": str(latest.index[0]),
            "train_first": str(train.index[0]), "train_last": str(train.index[-1]),
            "model": fc, "mean": train["y"].mean(), "alpha": F["final_alpha"],
            "status": "live: Nov 2026 incomplete in snapshot"}
    table = pd.concat([table, pd.DataFrame([live])], ignore_index=True)
    table.to_csv(OUT / "walk_forward_forecasts.csv", index=False)
    coef_path.to_csv(OUT / "coefficient_path.csv")

    out = {
        "sample": {"rows": len(train), "first_feature_month": str(train.index[0]),
                   "last_feature_month": str(train.index[-1]),
                   "first_training_window": f"{train.index[0]} to {train.index[START - 1]}"},
        "summary": S,
        "backtest": bt,
        "robustness": rob,
        "final": F,
        "coefficients": coefs.to_dict(),
    }

    # 5. Optional power check (slow); otherwise keep the result of an earlier run
    if run_power:
        print("\nPower check (this takes a while):")
        out["power_check"] = power_check()
    elif (OUT / "results.json").exists():
        old = json.loads((OUT / "results.json").read_text())
        if "power_check" in old:
            out["power_check"] = old["power_check"]
    (OUT / "results.json").write_text(json.dumps(out, indent=2, default=float))

    # Prints readable summary on console
    pct = lambda v: f"{v * 100:6.2f}%"
    print(f"\nOut-of-sample months: {S['n_oos']} ({S['first_target']} to {S['last_target']})")
    print(
        f"R2 out-of-sample   Ridge {pct(S['r2_oos_ridge'])}   OLS {pct(S['r2_oos_ols'])}"
        f"   (90% bootstrap for Ridge: {pct(S['r2_boot_5'])} to {pct(S['r2_boot_95'])})"
    )
    print(
        f"RMSE               Ridge {pct(S['rmse_ridge'])}   OLS {pct(S['rmse_ols'])}"
        f"   mean {pct(S['rmse_mean'])}"
    )
    print(
        f"MAE                Ridge {pct(S['mae_ridge'])}   OLS {pct(S['mae_ols'])}"
        f"   mean {pct(S['mae_mean'])}"
    )
    print(
        f"Direction right    Ridge {S['hits_ridge']}/{S['n_oos']}   OLS {S['hits_ols']}/{S['n_oos']}"
        f"   mean {S['hits_mean']}/{S['n_oos']}"
        f"   (PETS beat benchmark in {S['months_pets_beat_index']} months)"
    )
    print(
        f"Diebold-Mariano    stat {S['dm_stat']:.2f}  p {S['dm_p']:.3f}"
        f"     Clark-West  stat {S['cw_stat']:.2f}  p {S['cw_p']:.3f}"
    )
    print(
        f"Median Ridge penalty chosen: {S['median_alpha']:.1f}   "
        f"(at max {S['max_alpha']:.1f}: {S['windows_at_max_alpha']}/{S['n_oos']} windows)"
    )
    print(f"Forecast st. dev. {pct(S['forecast_sd'])} vs actual {pct(S['actual_sd'])}")

    print(f"\nBacktest (round-trip cost {COST * 100:.2f}%, per-switch cost {(COST / 2) * 100:.2f}%):")
    for k, v in bt.items():
        print(f"  {k:24s} {v:.4f}")

    print("\nRobustness (R2 out-of-sample):")
    for label, val in rob.items():
        print(f"  {label:38s} {pct(val)}")

    print("\nStandardised coefficients, final fit (x100):")
    print((coefs * 100).round(3).sort_values().to_string())

    print("\nSign changes along the walk-forward path:", F["sign_flips"])
    print(f"\nForecast for {F['forecast_month']}: {pct(fc)} excess return vs FTSE 250")
    print(
        f"12-month tilt (avg of last 6 forecasts x 12): {pct(F['tilt_12m'])}"
        f"   +/- {pct(F['band_12m'])} (1 st. dev.)"
    )
    if "nov_to_date_excess" in F:
        print(f"November to snapshot date, actual excess return so far: {pct(F['nov_to_date_excess'])}")
    if "power_check" in out:
        print("\nPower check (share of runs with R2 > 0):",
              {r["planted_correlation"]: r["share_r2_above_zero"] for r in out["power_check"]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", action="store_true", help="also run the slow power check")
    main(run_power=ap.parse_args().power)
