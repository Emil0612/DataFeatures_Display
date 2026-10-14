"""Price bars and bar-level price features: returns, range, true range and volatility estimators.

Bars are built from cleaned trades on a fixed session grid (see schema.session_grid). A minute
without trades still gets a bar: its open/high/low/close carry the last close forward, and its
volume, turnover and trade count are zero.

Rolling windows run over the bar sequence and only look backwards, so every value at bar T uses
bars up to and including T.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import BARS_PER_DAY, INVALID_PRINTS, NON_LIT_PRINTS, BarBinner, has_bits, session_grid, trading_day

VOL_WINDOW = 30
ATR_WILDER_N = 14
MEDRV_SAMPLE_MIN = 5
ANNUALISE = np.sqrt(252 * BARS_PER_DAY)


def clean_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Drop cancelled trades and correction / trade-break prints."""
    keep = ~trades["cancelled"].fillna(False).astype(bool) & ~has_bits(trades["tradecode"], INVALID_PRINTS)
    clean = trades.loc[keep].sort_values("timestamp", kind="stable")
    return clean.assign(timestamp=pd.DatetimeIndex(clean["timestamp"]).as_unit("ns"))


def lit_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Cleaned trades that reflect the visible order book (no off-book, dark or delayed prints)."""
    clean = clean_trades(trades)
    return clean.loc[~has_bits(clean["tradecode"], NON_LIT_PRINTS)]


def build_bars(trades: pd.DataFrame, days) -> pd.DataFrame:
    """OHLCV bars on the session grid for `days`.

    Returns open/high/low/close, volume (shares), turnover (price x quantity) and trade_count.
    Empty bars are filled with the previous close (last observation carried forward).
    """
    b = BarBinner(trades["timestamp"], session_grid(days))
    price = trades["price"].to_numpy(dtype=float)
    qty = trades["quantity"].to_numpy(dtype=float)
    bars = pd.DataFrame(
        {
            "open": b.agg(price, "first"),
            "high": b.agg(price, "max"),
            "low": b.agg(price, "min"),
            "close": b.agg(price, "last"),
            "volume": b.sum(qty),
            "turnover": b.sum(price * qty),
            "trade_count": b.count(),
        }
    )
    close = bars["close"].ffill()
    for col in ("open", "high", "low"):
        bars[col] = bars[col].fillna(close)
    bars["close"] = close
    return bars


def price_features(bars: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    """Returns, range, true range and volatility features for each bar."""
    out = pd.DataFrame(index=bars.index)
    close, high, low, open_ = bars["close"], bars["high"], bars["low"], bars["open"]
    prev_close = close.shift(1)

    out["log_return"] = np.log(close / prev_close)
    out["return_pct"] = close / prev_close - 1.0
    out["bar_return"] = (close - open_) / open_
    out["bar_range"] = (high - low) / close
    out["bar_range_abs"] = high - low

    # True range (Wilder 1978) also counts a gap from the previous close.
    out["true_range"] = pd.concat([high, prev_close], axis=1).max(axis=1, skipna=False) - pd.concat(
        [low, prev_close], axis=1
    ).min(axis=1, skipna=False)
    out["ntr"] = out["true_range"] / prev_close
    out["atr"] = out["true_range"].rolling(window, min_periods=1).mean()

    # Volatility estimators ignore bars without trades: their zero return would read as calm
    # exactly when there is no information. The last estimate is carried across those bars.
    active = bars["trade_count"] > 0
    log_ret = out["log_return"].where(active)
    out["realised_vol"] = log_ret.rolling(window, min_periods=2).std().ffill()

    log_hl = np.log(high / low).where(active & (high > low))
    out["parkinson_vol"] = np.sqrt((log_hl**2 / (4 * np.log(2))).rolling(window, min_periods=1).mean()).ffill()

    log_co = np.log(close / open_).where(active & (high > low))
    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    gk = gk.where(gk >= 0)  # negative when the open-close move exceeds the range
    out["garman_klass_vol"] = np.sqrt(gk.rolling(window, min_periods=1).mean()).ffill()

    for col in ("realised_vol", "parkinson_vol", "garman_klass_vol"):
        out[f"{col}_ann"] = out[col] * ANNUALISE

    # Wilder's ATR: exponential smoothing with alpha = 1/14.
    out["atr_wilder"] = out["true_range"].ewm(alpha=1 / ATR_WILDER_N, adjust=False, min_periods=ATR_WILDER_N).mean()

    # Short vs medium horizon realised volatility: a regime-shift signal.
    for n in (10, 60, 120):
        out[f"volatility_{n}"] = log_ret.rolling(n, min_periods=2).std().ffill()
    out["vol10_to_vol60"] = out["volatility_10"] / out["volatility_60"].where(out["volatility_60"] > 0)
    out["medrv_vol"] = medrv_vol(bars)
    return out


def medrv_vol(bars: pd.DataFrame, sample_min: int = MEDRV_SAMPLE_MIN) -> pd.Series:
    """Previous day's jump-robust realised volatility (MedRV, Andersen, Dobrev & Schaumburg 2012).

    MedRV for day D needs all of day D's returns, so it is only known after the close. Every bar
    of day D carries the value for day D-1 (the sqrt of MedRV, on the scale of a daily std).
    """
    day = trading_day(bars.index)
    daily = {}
    for d, g in bars["close"].groupby(day):
        r = np.log(g.iloc[sample_min - 1::sample_min]).diff().dropna().abs().to_numpy()
        n = len(r)
        if n < 3:
            daily[d] = np.nan
            continue
        med = np.median(np.vstack([r[:-2], r[1:-1], r[2:]]), axis=0)
        const = np.pi / (6 - 4 * np.sqrt(3) + np.pi)
        daily[d] = np.sqrt(const * n / (n - 2) * np.sum(med**2))
    previous_day = pd.Series(daily).shift(1)
    return pd.Series(previous_day.reindex(day).to_numpy(), index=bars.index)
