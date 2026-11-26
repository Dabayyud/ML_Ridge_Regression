"""
Pets at Home (LSE: PETS) — walk-forward Ridge signal for the group project pitch.
As mentioned in the presentation, the ML gets an assigned weight of 0% based on 
pre-determined rules.

What this script does:

- It predicts PETS next-month log excess return over the FTSE 250 using an
    expanding-window Ridge regression (refit monthly on past data only).
- It benchmarks against plain OLS and the expanding historical mean.
- Runs a simple switch backtest with robustness checks, and a
    final fit to generate the next-month forecast and 12-month tilt.

Usage:
  python pets_ml_model.py   (expects ./data and writes outputs to ./results)

Snapshot: strats 11 Nov 2026

  data/prices_monthly.csv   Month-end total returns (pets_tr = PETS.L, midd_tr =
                            iShares FTSE 250 UCITS ETF MIDD.L), FTSE 250 price
                            index (ftmc), GBP/USD (gbpusd), and 63-day annualised
                            PETS volatility (vol63).
                            
  data/ons_raw.json         ONS D7G7 (CPI annual %) and J5EK (retail sales volume).
  
  data/google_trends_gb.csv UK monthly search interest for "puppy" (and pet insurance).
# Only "puppy" is used, as an input to all models; "pet insurance" is not used.
"""

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

LAST_FULL_MONTH = "2026-10"      # Nov 2026 is incomplete in the data snapshot
START = 60                       # Minimum training months before first OOS forecast
ALPHAS = np.logspace(-2, 4, 30)  # Candidate Ridge penalties searched via CV
COST = 0.005                     # 0.5% round-trip cost (0.25% / half charged per switch)

# Key UK CMA veterinary market review / investigation dates (month of release)
# CMA: The UK's regulator for competition mergers & consumer protection.

CMA_MONTHS = [
    "2023-09",  # Review launched
    "2024-03",  # Formal market investigation proposed
    "2024-05",  # Market investigation reference made
    "2025-06",  # Timetable extended
    "2025-10",  # Provisional decision
    "2026-03",  # Final decision
    "2026-09",  # Final orders
]

# Core feature set: mix of price/vol signals, macro, search trend and regulatory dummy
FEATURES = [
    "reversal", "momentum", "vol3m", "mkt_mom3",
    "retail_yoy", "cpi_yoy", "gbp_chg3", "puppy_yoy", "cma_event",
]

PRICE_ONLY = FEATURES[:4]  # purely market/price based subset for comparison

# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ----------------------------------------------------------------------------

def load(benchmark="midd_tr"):
    """Load prices macro series and search data; build features and next-month target."""
    px = pd.read_csv(DATA / "prices_monthly.csv", index_col="ym")
    px.index = pd.PeriodIndex(px.index, freq="M")  # monthly periods make lags & shifts clean

    ons = json.loads((DATA / "ons_raw.json").read_text())
    to_series = lambda key: pd.Series(
        {pd.Period(ym, "M"): float(val) for ym, val in ons[key]["data"].items()}
    )
    cpi, rsi = to_series("cpi"), to_series("rsi")

    gt = pd.read_csv(DATA / "google_trends_gb.csv", index_col="ym")
    gt.index = pd.PeriodIndex(gt.index, freq="M")

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
    ex_full = ex.loc[:LAST_FULL_MONTH]
    y = ex_full.shift(-1).rename("y")  # Next month's excess return

    full = pd.concat([X, y], axis=1).loc[:LAST_FULL_MONTH]
    train = full.dropna()              # Only rows where both features and target exist
    latest = full[FEATURES].iloc[[-1]] # Feature row at end of LAST_FULL_MONTH
    return train, latest, ex_full, px


# -----------------------------------------------------------------------------
# Model builders & walk-forward engine
# -----------------------------------------------------------------------------

def ridge():
    # Scaler fitted once per training window (not per tuning fold; see CHANGELOG).
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(5)))


def ols():
    # Unpenalised linear benchmark on scaled features (same scale as Ridge for comparision)
    return make_pipeline(StandardScaler(), LinearRegression())


def lasso():
    # L1 alternative to see if sparse feature selection beats L2 shrinkage
    return make_pipeline(StandardScaler(), LassoCV(cv=TimeSeriesSplit(5), max_iter=50000))


def walk_forward(data, cols, make_model=ridge, start=START, gap=0, target="y", keep_coefs=False):
    """Run an expanding-window forecast without peeking ahead.

    When predicting row t, only rows up to t - 1 - gap are available for fitting.
    Use gap > 0 for multi-month targets whose full outcome isn't known yet at t-1.
    """
    X, y = data[cols], data[target]
    rows, coefs = [], []

    for t in range(start, len(X)):
        # Slice strictly prior data so no future info leaks into training or scaling
        X_tr, y_tr = X.iloc[: t - gap], y.iloc[: t - gap]
        m = make_model().fit(X_tr, y_tr)
        o = ols().fit(X_tr, y_tr)

        rows.append({
            "actual": y.iloc[t],
            "model": m.predict(X.iloc[[t]])[0],
            "ols": o.predict(X.iloc[[t]])[0],
            "mean": y_tr.mean(),
            "alpha": getattr(m[-1], "alpha_", np.nan),  # OLS/Lasso won't have alpha_
        })
        if keep_coefs:
            coefs.append(m[-1].coef_)

    res = pd.DataFrame(rows, index=X.index[start:])
    # Shift index forward by 1 month so each row is stamped with the month being forecast
    res.index = (res.index + 1).rename("target_month")

    if keep_coefs:
        return res, pd.DataFrame(coefs, index=res.index, columns=cols)
    return res


# -----------------------------------------------------------------------------
# Evaluation metrics & backtest
# -----------------------------------------------------------------------------

def r2_oos(res, col="model"):
    """Out-of-sample R^2 relative to the expanding historical mean."""
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


def bootstrap_r2(res, col="model", n=5000, seed=0):
    """Resample OOS errors with replacement to get a 5th-95th percentile R^2 band."""
    rng = np.random.default_rng(seed)  # fixed seed so intervals are reproducible
    a, p, b = res["actual"].values, res[col].values, res["mean"].values
    idx = rng.integers(0, len(a), size=(n, len(a)))
    r2_draws = 1 - ((a[idx] - p[idx]) ** 2).sum(1) / ((a[idx] - b[idx]) ** 2).sum(1)
    return np.percentile(r2_draws, [5, 95])


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
    return metrics, strat


def summary(res):
    """Collect headline accuracy, statistical test, and stability numbers."""
    a = res["actual"]
    err = lambda c: a - res[c]
    dm_stat, dm_p = dm_test(res)
    cw_stat, cw_p = cw_test(res)
    boot_lo, boot_hi = bootstrap_r2(res)
    n_at_max_alpha = int(np.isclose(res["alpha"], ALPHAS.max(), rtol=0, atol=1e-12).sum())

    return {
        "n_oos": len(res),
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
        "hit_ridge": (np.sign(res["model"]) == np.sign(a)).mean(),
        "hit_ols": (np.sign(res["ols"]) == np.sign(a)).mean(),
        "hit_mean": (np.sign(res["mean"]) == np.sign(a)).mean(),
        "share_up_months": (a > 0).mean(),
        "dm_stat": dm_stat,
        "dm_p": dm_p,
        "cw_stat": cw_stat,
        "cw_p": cw_p,
        "median_alpha": res["alpha"].median(),
        "max_alpha": float(ALPHAS.max()),
        "windows_at_max_alpha": n_at_max_alpha,
        "forecast_sd": float(res["model"].std()),
    }


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main():
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
    bt, strat = backtest(res)

    # Save dated walk-forward forecast table explicitly with target_month column
    forecast_table = res.assign(strategy=strat).copy()
    forecast_table.insert(0, "target_month", forecast_table.index.astype(str))
    forecast_table.to_csv(OUT / "walk_forward_forecasts.csv", index=False)
    coef_path.to_csv(OUT / "coefficient_path.csv")

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
        "forecast_sd": float(res["model"].std()),
        "windows_at_max_alpha": S["windows_at_max_alpha"],
        "final_alpha": final[-1].alpha_,
        "intercept": final[-1].intercept_,
        # Annualise the average of the last 5 OOS forecasts plus the new forecast
        "tilt_12m": np.mean(res["model"].tail(5).tolist() + [fc]) * 12,
        "band_12m": resid_sd * np.sqrt(12),
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

    out = {
        "summary": S,
        "backtest": bt,
        "robustness": rob,
        "final": F,
        "coefficients": coefs.to_dict(),
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, default=float))

    # Prints readable summary on console
    pct = lambda v: f"{v * 100:6.2f}%"
    print(f"\nOut-of-sample months: {S['n_oos']} ({res.index[0]} to {res.index[-1]})")
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
        f"Hit rate           Ridge {pct(S['hit_ridge'])}   OLS {pct(S['hit_ols'])}"
        f"   mean {pct(S['hit_mean'])}"
        f"   (PETS beat benchmark in {pct(S['share_up_months'])} of months)"
    )
    print(
        f"Diebold-Mariano    stat {S['dm_stat']:.2f}  p {S['dm_p']:.3f}"
        f"     Clark-West  stat {S['cw_stat']:.2f}  p {S['cw_p']:.3f}"
    )
    print(
        f"Median Ridge penalty chosen: {S['median_alpha']:.1f}   "
        f"(at max {S['max_alpha']:.1f}: {S['windows_at_max_alpha']}/{S['n_oos']} windows)"
    )
    print(f"Forecast st. dev. (OOS Ridge predictions): {pct(S['forecast_sd'])}")

    print(
        f"\nBacktest (round-trip cost {COST * 100:.2f}%, per-switch cost {(COST / 2) * 100:.2f}%):",
        {k: round(v, 3) for k, v in bt.items()},
    )

    print("\nRobustness (R2 out-of-sample):")
    for label, val in rob.items():
        print(f"  {label:38s} {pct(val)}")

    print("\nStandardised coefficients, final fit:")
    print((coefs * 100).round(3).sort_values().to_string())

    print("\nSign changes along the walk-forward path:", F["sign_flips"])
    print(f"\nForecast for {F['forecast_month']}: {pct(fc)} excess return vs FTSE 250")
    print(
        f"12-month tilt (avg of last 6 forecasts x 12): {pct(F['tilt_12m'])}"
        f"   +/- {pct(F['band_12m'])} (1 st. dev.)"
    )
    if "nov_to_date_excess" in F:
        print(f"November to snapshot date, actual excess return so far: {pct(F['nov_to_date_excess'])}")


if __name__ == "__main__":
    main()
