"""
Re-download the daily Yahoo Finance data and rebuild data/prices_monthly.csv in Python.

STATUS: written 23 Nov 2026 and NOT yet run, because the machine that built the snapshot
had no direct network access to Yahoo. The frozen snapshot (data/prices_monthly.csv) was
built on 11 Nov 2026 in a browser session using exactly the steps below. Run this on a
laptop with internet access to (a) save the raw daily files that the snapshot lacks and
(b) see how far today's Yahoo history differs from the snapshot.

    pip install yfinance
    python tools/rebuild_prices.py            # writes data/raw/*.csv and data/prices_monthly_rebuilt.csv

It never overwrites the frozen snapshot. Yahoo revises history from time to time, so
small differences are possible; the script prints the largest one per column.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
START, END = "2014-03-01", "2026-11-12"   # END is exclusive: last day 11 Nov 2026
TICKERS = {"PETS.L": "pets", "MIDD.L": "midd", "^FTMC": "ftmc", "GBPUSD=X": "gbpusd"}


def download(ticker):
    """Daily close (split-adjusted, NOT dividend-adjusted) and cash dividends on ex-dates."""
    h = yf.Ticker(ticker).history(start=START, end=END, auto_adjust=False, actions=True)
    h.index = h.index.tz_localize(None).normalize()
    out = h[["Close", "Volume", "Dividends"]].dropna(subset=["Close"])
    out.to_csv(RAW / f"{TICKERS[ticker]}_daily.csv")
    return out


def total_return(daily):
    """Dividends reinvested at the ex-date close:  TR_t = TR_{t-1} * (P_t + D_t) / P_{t-1}.
    Yahoo's own 'Adj Close' was not used: it under-adjusted London dividends in the snapshot."""
    p, d = daily["Close"], daily["Dividends"].fillna(0.0)
    growth = (p + d) / p.shift(1)
    return p.iloc[0] * growth.fillna(1.0).cumprod()


def month_end(series):
    """Value on the last trading day of each calendar month (that series' own calendar)."""
    return series.groupby(series.index.to_period("M")).last()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    d = {t: download(t) for t in TICKERS}
    pets_tr = total_return(d["PETS.L"])
    lr = np.log(pets_tr).diff()
    vol63 = lr.rolling(63).std(ddof=1) * np.sqrt(252)   # 63 daily returns, sample sd, annualised
    m = pd.DataFrame({
        "pets_close": month_end(d["PETS.L"]["Close"]).round(2),
        "pets_tr": month_end(pets_tr).round(3),
        "midd_tr": month_end(total_return(d["MIDD.L"])).round(2),
        "ftmc": month_end(d["^FTMC"]["Close"]).round(1),
        "gbpusd": month_end(d["GBPUSD=X"]["Close"]).round(4),
        "vol63": month_end(vol63).round(4),
    }).loc["2014-03":]
    m.index = m.index.astype(str).rename("ym")
    m.to_csv(ROOT / "data" / "prices_monthly_rebuilt.csv")

    snap = pd.read_csv(ROOT / "data" / "prices_monthly.csv", index_col="ym")
    common = snap.index.intersection(m.index)
    print("Months compared:", len(common))
    # returns, not levels, are what the model uses (levels depend on the starting value)
    for c in ["pets_tr", "midd_tr", "ftmc", "gbpusd"]:
        a = np.log(snap.loc[common, c]).diff()
        b = np.log(m.loc[common, c]).diff()
        print(f"  {c:8s} largest monthly-return difference: {(a - b).abs().max():.6f}")
    print(f"  vol63    largest difference: {(snap.loc[common, 'vol63'] - m.loc[common, 'vol63']).abs().max():.6f}")


if __name__ == "__main__":
    main()
