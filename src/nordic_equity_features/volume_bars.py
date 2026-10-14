"""Volume-bar texture on the clock-bar grid.

A volume bar (Lopez de Prado 2018) closes each time cumulative turnover since the previous one
reaches a threshold. The emission runs within each day on the minute bars, so the state at minute T
depends only on turnover up to T. The features describe that state on the regular clock grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import trading_day

TURNOVER_THRESHOLD = 1_000_000.0
FILL_RATE_WINDOW_MIN = 15


def volume_bar_features(
    bars: pd.DataFrame, threshold: float = TURNOVER_THRESHOLD, window_min: int = FILL_RATE_WINDOW_MIN
) -> pd.DataFrame:
    """vol_bar_count_today_so_far, vol_bar_fill_rate_recent_15min and vol_bar_partial_fill_progress."""
    out = pd.DataFrame(index=bars.index, columns=[
        "vol_bar_count_today_so_far", "vol_bar_fill_rate_recent_15min", "vol_bar_partial_fill_progress",
    ], dtype=float)

    for _, day in bars.groupby(trading_day(bars.index)):
        cum = day["turnover"].to_numpy(dtype=float).cumsum()
        emitted = np.zeros(len(cum), dtype=bool)
        anchor_at = np.zeros(len(cum))
        anchor = 0.0
        for i, c in enumerate(cum):
            if c >= anchor + threshold:
                emitted[i] = True
                anchor = c  # overshoot is not carried into the next bar
            anchor_at[i] = anchor
        count = emitted.cumsum().astype(float)

        # Emissions per second over the trailing window; undefined until the window fits in the session.
        recent = pd.Series(emitted.astype(float)).rolling(window_min, min_periods=window_min).sum()
        out.loc[day.index, "vol_bar_count_today_so_far"] = count
        out.loc[day.index, "vol_bar_fill_rate_recent_15min"] = (recent / (window_min * 60)).to_numpy()
        out.loc[day.index, "vol_bar_partial_fill_progress"] = (cum - anchor_at) / threshold
    return out
