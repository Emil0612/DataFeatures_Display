"""Future-invariance check: features up to time t must not change when the data after t changes.

`perturb_after` keeps every record that was knowable before a cutoff and replaces everything
from the cutoff on with unrelated synthetic data. It also plants events exactly at the cutoff
(a split and a dividend announced at that instant, and a news item stamped at it), because the
boundary is where off-by-one leaks hide. If a feature at a bar labelled <= cutoff differs between
the original and the perturbed data, it used information from the future.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .pipeline import compute_features
from .schema import TZ
from .synthetic import MarketData, generate


def perturb_after(data: MarketData, cutoff: pd.Timestamp, seed: int = 12345) -> MarketData:
    """Copy of `data` with everything from `cutoff` on replaced by an unrelated dataset."""
    cutoff = pd.Timestamp(cutoff).tz_convert(TZ)
    other = generate(n_days=len(data.days), seed=seed)
    if list(other.days) != list(data.days):
        raise ValueError("perturb_after expects data from synthetic.generate (same trading calendar)")
    cutoff_day = cutoff.tz_localize(None).normalize()

    def splice(a: pd.DataFrame, b: pd.DataFrame, col: str) -> pd.DataFrame:
        return pd.concat([a.loc[a[col] < cutoff], b.loc[b[col] >= cutoff]], ignore_index=True)

    trades = splice(data.trades, other.trades, "timestamp").sort_values("timestamp", kind="stable")
    news = splice(data.news, other.news, "timestamp")
    planted_news = data.news.iloc[:0].copy()
    planted_news.loc[0] = [cutoff, "corporate", data.instruments.index[0], None, "RATING"]
    news = pd.concat([news, planted_news], ignore_index=True).sort_values("timestamp", kind="stable")

    index_bars = pd.concat([data.index_bars.loc[data.index_bars.index <= cutoff],
                            other.index_bars.loc[other.index_bars.index > cutoff]])

    # End-of-day values for the cutoff day are published after its close, so they are future too.
    eod = data.eod.copy()
    future = pd.to_datetime(eod["date"]) >= cutoff_day
    eod.loc[future, "mcap"] *= np.random.default_rng(seed).uniform(0.5, 2.0, int(future.sum()))

    next_day = cutoff_day + pd.offsets.BDay(2)
    splits = pd.concat(
        [data.splits.loc[data.splits["announced"] < cutoff],
         pd.DataFrame([{"symbol": s, "announced": cutoff, "ex_date": next_day, "new_shares": 3.0, "old_shares": 1.0}
                       for s in data.instruments.index])],
        ignore_index=True,
    )
    dividends = pd.concat(
        [data.dividends.loc[data.dividends["announced"] < cutoff],
         pd.DataFrame([{"symbol": s, "announced": cutoff, "ex_date": next_day, "amount": 9.99}
                       for s in data.instruments.index])],
        ignore_index=True,
    )
    return dataclasses.replace(
        data, trades=trades.reset_index(drop=True), news=news.reset_index(drop=True),
        index_bars=index_bars, eod=eod, splits=splits, dividends=dividends,
    )


def lookahead_violations(data: MarketData, symbol: str, cutoff: pd.Timestamp, seed: int = 12345) -> pd.DataFrame:
    """Features whose values at bars labelled <= cutoff change when the future is replaced.

    Returns one row per violating feature with the number of affected bars; empty if none.
    """
    cutoff = pd.Timestamp(cutoff).tz_convert(TZ)
    before = compute_features(data, symbol).loc[:cutoff]
    after = compute_features(perturb_after(data, cutoff, seed), symbol).loc[:cutoff]
    return compare(before, after)


def compare(a: pd.DataFrame, b: pd.DataFrame, rtol: float = 1e-9, atol: float = 1e-12) -> pd.DataFrame:
    """Columns where two feature frames differ (NaN equals NaN), with the count of differing rows."""
    rows = []
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(x):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            same = np.isclose(xv, yv, rtol=rtol, atol=atol) | (np.isnan(xv) & np.isnan(yv))
        else:
            same = (x.astype("string").fillna("<NA>") == y.astype("string").fillna("<NA>")).to_numpy()
        if not same.all():
            rows.append({"feature": col, "bars_changed": int((~same).sum()), "first_changed": a.index[~same][0]})
    return pd.DataFrame(rows, columns=["feature", "bars_changed", "first_changed"])
