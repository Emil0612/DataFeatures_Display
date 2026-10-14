"""Transaction metrics: trade counts and sizes, block trades and trade velocity within each bar.

Block trades are defined relative to the instrument's own history: a trade is a block if its size
is at or above the 95th percentile of trade sizes over the previous 20 trading days. The threshold
for day D uses only days strictly before D. A same-day percentile would leak the rest of the day's
trades into every morning bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import BarBinner, is_continuous, session_grid, trading_day

BLOCK_PERCENTILE = 0.95
BLOCK_WINDOW_DAYS = 20
BLOCK_MIN_DAYS = 5


def prior_window_threshold(
    trades: pd.DataFrame,
    days,
    percentile: float = BLOCK_PERCENTILE,
    window_days: int = BLOCK_WINDOW_DAYS,
    min_days: int = BLOCK_MIN_DAYS,
) -> pd.Series:
    """Block-size threshold for each day, from trade sizes on the previous `window_days` days.

    Only days with trades count towards the window. Days with fewer than `min_days` prior trading
    days get NaN, and every block feature for that day is NaN as well.
    """
    day_of_trade = trading_day(pd.DatetimeIndex(trades["timestamp"]))
    sizes_by_day = {d: g.to_numpy(dtype=float) for d, g in trades["quantity"].groupby(day_of_trade)}
    history = sorted(sizes_by_day)
    out = {}
    for day in pd.DatetimeIndex(pd.to_datetime(list(days))).normalize():
        prior = [d for d in history if d < day][-window_days:]
        if len(prior) < min_days:
            out[day] = np.nan
        else:
            out[day] = float(np.quantile(np.concatenate([sizes_by_day[d] for d in prior]), percentile))
    return pd.Series(out, name="block_threshold", dtype=float)


def _per_second_stats(trades: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """tps_avg, tps_max and tps_direction from per-second trade counts within each bar."""
    b = BarBinner(trades["timestamp"], grid)
    ts = pd.DatetimeIndex(trades["timestamp"])
    second = ts.second.to_numpy()[b.valid]
    code = b.code[b.valid]
    # Trades per (bar, second); seconds without trades are simply absent.
    per_second = pd.Series(1.0, index=code * 60 + second).groupby(level=0).sum()
    bar_of = per_second.index.to_numpy() // 60
    sec_of = per_second.index.to_numpy() % 60
    by_bar = per_second.groupby(bar_of)

    # Slope of an OLS fit of the 60 per-second counts (zeros included) on the second index.
    # Seconds without trades contribute nothing to the numerator, so the sum runs over active ones.
    slope_num = pd.Series((sec_of - 29.5) * per_second.to_numpy()).groupby(bar_of).sum()
    sxx = 60 * (60**2 - 1) / 12
    direction = (slope_num / sxx).where(by_bar.size() >= 3)

    stats = pd.DataFrame({"tps_avg": by_bar.mean(), "tps_max": by_bar.max(), "tps_direction": direction})
    stats = stats.reindex(range(len(grid)))
    stats.index = grid
    return stats.astype(float)


def transaction_features(trades: pd.DataFrame, days) -> pd.DataFrame:
    """Per-bar trade size distribution, block flow and velocity. `trades` must be cleaned."""
    grid = session_grid(days)
    b = BarBinner(trades["timestamp"], grid)
    qty = trades["quantity"].to_numpy(dtype=float)

    out = pd.DataFrame(index=b.grid)
    out["trade_count"] = b.count()
    volume = b.sum(qty)
    out["avg_trade_size"] = (volume / out["trade_count"]).where(out["trade_count"] > 0)
    out["max_trade_size"] = b.agg(qty, "max")
    out["trade_size_std"] = b.agg(qty, "std")

    threshold = prior_window_threshold(trades, days)
    theta = threshold.reindex(trading_day(pd.DatetimeIndex(trades["timestamp"]))).to_numpy()
    is_block = qty >= theta  # False where the threshold is NaN
    defined = pd.Series(threshold.reindex(trading_day(b.grid)).to_numpy(), index=b.grid).notna()
    block_qty = b.sum(qty, where=is_block)

    out["block_count"] = b.count(where=is_block).where(defined)
    out["block_trade"] = block_qty.where(defined)
    out["block_share"] = (block_qty / volume).where(defined & (volume > 0))

    out = out.join(_per_second_stats(trades, b.grid))
    cont_stats = _per_second_stats(trades.loc[is_continuous(trades["mmt"])], b.grid)
    out["tps_avg_continuous"] = cont_stats["tps_avg"]
    out["tps_max_continuous"] = cont_stats["tps_max"]
    return out
