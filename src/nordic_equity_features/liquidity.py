"""Liquidity and trading cost: intra-bar price distribution, implied spreads and Amihud illiquidity.

Order-book quotes are not part of the input, so spreads are inferred from trades:

* spread_bounce  - average price jump when the aggressor side flips (buy after sell or vice
                   versa); each flip crosses the spread once.
* spread_roll    - Roll (1984): 2 * sqrt(-cov(dP_t, dP_t-1)) within the bar; 0 when the price
                   trends (non-negative covariance) and the model does not apply.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import BarBinner, is_continuous, session_grid

BPS = 1e4
SPREAD_DIR_WINDOW = 30
AMIHUD_WINDOW = 30
REGIME_WINDOW = 30
REGIME_LAG = 3


def _weighted_moments(trades: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Volume-weighted mean, std, skewness and excess kurtosis of trade prices per bar."""
    b = BarBinner(trades["timestamp"], grid)
    p = trades["price"].to_numpy(dtype=float)
    q = trades["quantity"].to_numpy(dtype=float)
    w = b.sum(q)
    mean = b.sum(q * p) / w.where(w > 0)
    n = b.count()

    dev = np.full(len(p), np.nan)
    dev[b.valid] = p[b.valid] - mean.to_numpy()[b.code[b.valid]]
    var = b.sum(q * dev**2) / w
    sigma = np.sqrt(var)
    has_shape = sigma > 0
    skew = (b.sum(q * dev**3) / w / sigma**3).where(has_shape & (n >= 3))
    kurt = (b.sum(q * dev**4) / w / sigma**4 - 3.0).where(has_shape & (n >= 4))
    return pd.DataFrame({"vwap": mean, "vwad": sigma, "skew": skew, "kurt": kurt})


def _rolling_slope(y: pd.Series, window: int, min_points: int = 2) -> pd.Series:
    """Trailing OLS slope of y on time, fitted on the non-missing points of each full window."""
    m = y.notna().astype(float)
    yv = y.fillna(0.0)
    x = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    roll = lambda s: s.rolling(window, min_periods=window).sum()  # noqa: E731
    n, sx, sy = roll(m), roll(x * m), roll(yv)
    sxx, sxy = roll(x * x * m), roll(x * yv)
    denom = n * sxx - sx**2
    return ((n * sxy - sx * sy) / denom.where(denom > 0)).where(n >= min_points)


def liquidity_regime(spread_bps: pd.Series, window: int = REGIME_WINDOW, lag: int = REGIME_LAG) -> pd.Series:
    """Tight / normal / wide spread regime with hysteresis.

    Each bar is first classified against the trailing 33rd and 67th percentiles of the spread.
    The published regime only switches once the raw class has been the same for `lag` bars in a
    row, so single noisy bars do not flip it.
    """
    p33 = spread_bps.rolling(window, min_periods=1).quantile(0.33)
    p67 = spread_bps.rolling(window, min_periods=1).quantile(0.67)
    raw = np.where(spread_bps.isna(), "", np.where(spread_bps <= p33, "tight", np.where(spread_bps <= p67, "normal", "wide")))
    regime = np.empty(len(raw), dtype=object)
    current = None
    for i in range(len(raw)):
        recent = raw[max(0, i - lag):i]
        if len(recent) and recent[0] != "" and (recent == recent[0]).all():
            current = recent[0]
        regime[i] = current
    return pd.Series(regime, index=spread_bps.index, dtype="string")


def _flip_spreads(trades: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    signed = trades.loc[trades["aggressor"].isin(["buy", "sell"])]
    b = BarBinner(signed["timestamp"], grid)
    date = pd.DatetimeIndex(signed["timestamp"]).date
    side = signed["aggressor"].to_numpy()
    price = signed["price"].to_numpy(dtype=float)
    flip = np.r_[False, (side[1:] != side[:-1]) & (date[1:] == date[:-1])]
    bounce = np.abs(np.r_[np.nan, np.diff(price)])
    return pd.DataFrame(
        {
            "spread_bounce": b.agg(bounce, "mean", where=flip),
            "spread_min": b.agg(bounce, "min", where=flip),
            "spread_max": b.agg(bounce, "max", where=flip),
        }
    )


def _roll_spread(trades: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.Series:
    b = BarBinner(trades["timestamp"], grid)
    code = b.code
    price = trades["price"].to_numpy(dtype=float)
    same_bar = np.r_[False, code[1:] == code[:-1]]
    dp = np.where(same_bar, np.r_[np.nan, np.diff(price)], np.nan)
    dp_lag = np.where(np.r_[False, same_bar[1:] & same_bar[:-1]], np.r_[np.nan, dp[:-1]], np.nan)
    pair = ~np.isnan(dp) & ~np.isnan(dp_lag)
    n = b.count(where=pair)
    sa, sb, sab = b.sum(dp, where=pair), b.sum(dp_lag, where=pair), b.sum(dp * dp_lag, where=pair)
    cov = (sab - sa * sb / n) / (n - 1)
    roll = pd.Series(np.where(cov < 0, 2.0 * np.sqrt(-cov.clip(upper=0)), 0.0), index=b.grid)
    return roll.where(b.count() >= 4)


def liquidity_features(clean: pd.DataFrame, lit: pd.DataFrame, bars: pd.DataFrame, days) -> pd.DataFrame:
    """Per-bar liquidity features.

    `clean` is the cleaned trade stream, `lit` the lit subset used for spread inference (auction
    and off-book prints do not cross a visible spread), `bars` the lit price bars.
    """
    grid = session_grid(days)
    out = pd.DataFrame(index=grid)

    mom = _weighted_moments(clean, grid)
    traded = mom["vwap"].notna()
    close = BarBinner(clean["timestamp"], grid).agg(clean["price"].to_numpy(dtype=float), "last")
    out["vwap"] = mom["vwap"].ffill()
    out["vwap_dev"] = (close - out["vwap"]).where(traded)
    out["vwap_dev_bps"] = out["vwap_dev"] / out["vwap"] * BPS
    out["vwad"] = mom["vwad"].fillna(0.0)
    out["vwad_bps"] = out["vwad"] / out["vwap"] * BPS
    out["intrabar_skew"] = mom["skew"]
    out["intrabar_kurt"] = mom["kurt"]

    # Auction prints put a single-price volume cluster in the opening and closing bars.
    cont = _weighted_moments(clean.loc[is_continuous(clean["mmt"])], grid)
    out["vwap_continuous"] = cont["vwap"].ffill()
    out["vwad_continuous"] = cont["vwad"].fillna(0.0)

    lit_cont = lit.loc[is_continuous(lit["mmt"])]
    out = out.join(_flip_spreads(lit_cont, grid))
    out["spread_roll"] = _roll_spread(lit_cont, grid)
    out["spread_dir"] = _rolling_slope(out["spread_bounce"], SPREAD_DIR_WINDOW)
    out["spread_accel"] = out["spread_dir"].diff()
    out["spread_roll_valid"] = (out["spread_roll"] > 0).astype(float).where(out["spread_roll"].notna())

    for col in ("bounce", "min", "max"):
        name = "effective_spread_bps" if col == "bounce" else f"spread_{col}_bps"
        out[name] = out[f"spread_{col}"] / out["vwap"] * BPS

    # Execution cost: half the most recent effective spread, paid on one side of a round trip.
    log_ret = np.log(bars["close"] / bars["close"].shift(1))
    half_spread = out["effective_spread_bps"].ffill() / BPS / 2
    traded_bar = bars["volume"] > 0
    out["cost_adj_return"] = (log_ret - half_spread).where(traded_bar)
    out["cost_capture_ratio"] = (half_spread / log_ret.abs()).where(log_ret.abs() > 1e-8)
    regime = liquidity_regime(out["effective_spread_bps"].ffill())
    out["liquidity_regime"] = regime
    out["liquidity_regime_encoded"] = regime.map({"tight": 0.0, "normal": 1.0, "wide": 2.0}).astype(float)

    # Amihud (2002): absolute return per unit of turnover. Bars without price movement are left
    # out; their zero would read as perfect liquidity.
    log_ret = np.log(bars["close"] / bars["close"].shift(1))
    amihud = (log_ret.abs() / bars["turnover"]).where((bars["turnover"] > 0) & (log_ret != 0))
    mu = amihud.rolling(AMIHUD_WINDOW, min_periods=2).mean()
    sd = amihud.rolling(AMIHUD_WINDOW, min_periods=2).std()
    out["amihud"] = amihud
    out["amihud_zscore"] = (amihud - mu) / sd.where(sd > 0)
    return out
