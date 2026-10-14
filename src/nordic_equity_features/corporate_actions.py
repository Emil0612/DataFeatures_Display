"""Corporate actions: split adjustment and split / dividend event features.

Point-in-time rule: an event is only visible to bars labelled strictly after its announcement
timestamp. A dividend announced at 12:15 is invisible to the 12:15 bar and to every bar before it,
even on the same day. This needs real announcement timestamps. If a data source only has ex-dates,
these countdown features would encode the future and must not be used in a backtest.

Split adjustment is backward: prices before a split are scaled so they are comparable with prices
after it. That depends on which splits have happened, so adjusted history is always built as of a
moment (`as_of`), and only splits with an ex-date on or before that moment are applied.

Input tables:
    splits     symbol, announced (tz-aware), ex_date (date), new_shares, old_shares
    dividends  symbol, announced (tz-aware), ex_date (date), amount
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import TZ, trading_day

SPLIT_WINDOW_DAYS = 30


def _busdays(start: pd.DatetimeIndex, end: np.ndarray) -> np.ndarray:
    """Trading days (Mon-Fri) from each start day to each end day; NaN where end is missing."""
    end = np.asarray(end, dtype="datetime64[ns]")
    ok = ~np.isnat(end)
    out = np.full(len(start), np.nan)
    s = start.to_numpy(dtype="datetime64[D]")
    out[ok] = np.busday_count(s[ok], end[ok].astype("datetime64[D]"))
    return out


def _visible(labels: pd.DatetimeIndex, announced: pd.Series) -> np.ndarray:
    """Matrix [event, bar]: True where the event was announced strictly before the bar label."""
    ann = pd.DatetimeIndex(pd.to_datetime(announced)).tz_convert(TZ)
    return labels.to_numpy()[None, :] > ann.to_numpy()[:, None]


def split_factor(labels: pd.DatetimeIndex, splits: pd.DataFrame, as_of: pd.Timestamp) -> np.ndarray:
    """Backward price-adjustment factor for each bar, using splits that happened by `as_of`.

    The factor is the product of old_shares / new_shares over splits with ex-date after the bar's
    day and on or before the as-of day. A 2-for-1 split halves every earlier price.
    """
    day = trading_day(labels).to_numpy()
    as_of_day = trading_day(pd.DatetimeIndex([pd.Timestamp(as_of)]))[0]
    factor = np.ones(len(labels))
    for _, s in splits.iterrows():
        ex = pd.Timestamp(s["ex_date"]).normalize()
        if ex <= as_of_day:
            factor[day < ex.to_datetime64()] *= s["old_shares"] / s["new_shares"]
    return factor


def adjusted_ohlcv(bars: pd.DataFrame, splits: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Split-adjusted bars ao/ah/al/ac/av as seen at `as_of` (bars after `as_of` are dropped)."""
    hist = bars.loc[bars.index <= pd.Timestamp(as_of)]
    f = split_factor(hist.index, splits, as_of)
    return pd.DataFrame(
        {
            "ao": hist["open"] * f,
            "ah": hist["high"] * f,
            "al": hist["low"] * f,
            "ac": hist["close"] * f,
            "av": hist["volume"] / f,
        },
        index=hist.index,
    )


def split_features(bars: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Per-bar split event features, using only splits announced before each bar."""
    labels = pd.DatetimeIndex(bars.index)
    day = trading_day(labels)
    day_np = day.to_numpy()
    out = pd.DataFrame(index=labels)

    # Close-to-close log return with the split neutralised on the ex-day.
    on_ex = np.ones(len(labels))
    for _, s in splits.iterrows():
        ex = pd.Timestamp(s["ex_date"]).normalize().to_datetime64()
        first_bar_of_ex_day = (day_np == ex) & (np.r_[True, day_np[1:] != day_np[:-1]])
        on_ex[first_bar_of_ex_day] *= s["old_shares"] / s["new_shares"]
    prev_close = bars["close"].shift(1)
    out["split_adjusted_return"] = np.log(bars["close"] / (prev_close * on_ex))

    n = len(labels)
    next_ex = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    next_ratio = np.full(n, np.nan)
    next_factor = np.full(n, np.nan)
    last_ex = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    if len(splits):
        vis = _visible(labels, splits["announced"])
        for k, (_, s) in enumerate(splits.iterrows()):
            ex = pd.Timestamp(s["ex_date"]).normalize().to_datetime64()
            upcoming = vis[k] & (day_np < ex) & (np.isnat(next_ex) | (ex < next_ex))
            next_ex[upcoming] = ex
            next_ratio[upcoming] = s["new_shares"] / s["old_shares"]
            next_factor[upcoming] = s["old_shares"] / s["new_shares"]
            # A split that has gone ex is public information whether or not it was announced.
            past = (day_np >= ex) & (np.isnat(last_ex) | (ex > last_ex))
            last_ex[past] = ex

    out["stock_split_xday"] = _busdays(day, next_ex)
    out["stock_split_ratio"] = next_ratio
    out["stock_split_direction"] = np.select(
        [next_factor < 1, next_factor > 1], ["forward", "reverse"], default="none"
    )
    lookback = np.full(n, np.nan)
    has_past = ~np.isnat(last_ex)
    lookback[has_past] = np.busday_count(
        last_ex[has_past].astype("datetime64[D]"), day_np[has_past].astype("datetime64[D]")
    )
    out["stock_split_xday_lookback"] = lookback
    near = lambda v: np.nan_to_num(np.abs(v), nan=np.inf) <= SPLIT_WINDOW_DAYS  # noqa: E731
    out["is_split_window"] = near(out["stock_split_xday"].to_numpy()) | near(lookback)
    return out


def dividend_features(bars: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """Per-bar dividend features, using only dividends announced before each bar."""
    labels = pd.DatetimeIndex(bars.index)
    day = trading_day(labels)
    day_np = day.to_numpy()
    n = len(labels)

    # Previous day's closing price, for the dividend yield on the ex-date.
    daily_close = bars["close"].groupby(day).last()
    prev_day_close = daily_close.shift(1).reindex(day).to_numpy()

    amount = np.zeros(n)
    ex_flag = np.zeros(n)
    declared = np.zeros(n)
    next_ex = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    if len(dividends):
        vis = _visible(labels, dividends["announced"])
        for k, (_, d) in enumerate(dividends.iterrows()):
            ex = pd.Timestamp(d["ex_date"]).normalize().to_datetime64()
            on_ex = vis[k] & (day_np == ex)
            amount[on_ex] += d["amount"]
            ex_flag[on_ex] = 1.0
            upcoming = vis[k] & (day_np <= ex) & (np.isnat(next_ex) | (ex < next_ex))
            next_ex[upcoming] = ex
            ann_day = pd.Timestamp(d["announced"]).tz_convert(TZ).tz_localize(None).normalize().to_datetime64()
            declared[vis[k] & (day_np == ann_day)] = 1.0

    out = pd.DataFrame(index=labels)
    out["dividend_amount"] = amount
    out["dividend_pct"] = np.where(amount > 0, amount / prev_day_close, 0.0)
    out["dividend_ex_flag"] = ex_flag
    out["dividend_ex_date"] = pd.to_datetime(next_ex)
    out["dividend_declaration_flag"] = declared
    out["days_to_ex_dividend"] = _busdays(day, next_ex)
    return out
