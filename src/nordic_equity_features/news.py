"""News features: latest news category and trailing news density.

A news item stamped at time e is attached to the first bar labelled strictly after e. News that
arrives before the open lands in the first bar of the session, not in the bars of the evening
before, and an item published at exactly 10:15:00 belongs to the 10:16 bar.

Input table: timestamp (tz-aware), scope ("corporate" / "sector" / "macro"), category.
Corporate news is about the instrument itself and sector news about its industry; macro news is
market-wide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import snap_to_next_bar

DENSITY_WINDOW = 60

CORPORATE_CATEGORIES = ("EARNINGS", "GUIDANCE", "M_AND_A", "MANAGEMENT", "PRODUCT", "OFFERING", "RATING")
MACRO_CATEGORIES = ("INFLATION", "EMPLOYMENT", "CENTRAL_BANK", "GROWTH", "GEOPOLITICAL")


def _density(snapped: pd.Series, grid: pd.DatetimeIndex, window: int) -> pd.Series:
    per_bar = snapped.value_counts().reindex(grid, fill_value=0).astype(float)
    return per_bar.rolling(window, min_periods=1).sum()


def _latest_category(items: pd.DataFrame, snapped: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """Category of the most recent item visible at each bar."""
    if snapped.empty:
        return pd.Series(pd.NA, index=grid, dtype="string")
    ordered = items.loc[snapped.index].assign(bar=snapped).sort_values("timestamp")
    last = ordered.groupby("bar")["category"].last()
    return last.reindex(grid).astype("string").ffill()


def news_features(news: pd.DataFrame, grid: pd.DatetimeIndex, window: int = DENSITY_WINDOW) -> pd.DataFrame:
    out = pd.DataFrame(index=grid)
    snapped = {}
    for scope in ("corporate", "sector", "macro"):
        items = news.loc[news["scope"] == scope]
        snapped[scope] = (items, snap_to_next_bar(items["timestamp"], grid))

    corp, corp_bar = snapped["corporate"]
    macro, macro_bar = snapped["macro"]
    out["news_corp_cat"] = _latest_category(corp, corp_bar, grid)
    out["news_macro_cat"] = _latest_category(macro, macro_bar, grid)

    out["news_corp_density"] = _density(corp_bar, grid, window)
    out["news_density_sector"] = _density(snapped["sector"][1], grid, window)
    out["news_density_ratio"] = out["news_corp_density"] / out["news_density_sector"].replace(0, np.nan)
    out["news_density_macro"] = _density(macro_bar, grid, window)
    return out
