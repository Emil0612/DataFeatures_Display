"""Instrument metadata: market cap, listing age, listing count and country.

Market cap comes from end-of-day data, which is published after the close. During day D the most
recent value anyone could know is the one from the previous trading day, so the join takes the
last end-of-day row strictly before D.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import trading_day


def metadata_features(labels: pd.DatetimeIndex, eod: pd.DataFrame, instrument: dict) -> pd.DataFrame:
    """Per-bar metadata.

    `eod` has one row per trading day with columns date and mcap. `instrument` holds static
    fields: symbol, country, listing_date and n_listings.
    """
    day = trading_day(labels)
    bars = pd.DataFrame({"timestamp": labels, "day": day.astype("datetime64[ns]")})
    daily = eod[["date", "mcap"]].assign(
        date=lambda d: pd.to_datetime(d["date"]).dt.normalize().astype("datetime64[ns]")
    ).sort_values("date")
    merged = pd.merge_asof(
        bars, daily, left_on="day", right_on="date", direction="backward", allow_exact_matches=False
    )

    out = pd.DataFrame(index=labels)
    out["timestamp"] = labels
    out["symbol"] = instrument["symbol"]
    out["mcap"] = merged["mcap"].to_numpy()
    out["country"] = instrument["country"]
    listing = np.datetime64(pd.Timestamp(instrument["listing_date"]).date(), "D")
    out["days_since_listing"] = np.busday_count(listing, day.to_numpy(dtype="datetime64[D]")).astype(float)
    # A single current snapshot: not point-in-time (see the feature catalogue).
    out["n_listings"] = float(instrument["n_listings"])
    return out
