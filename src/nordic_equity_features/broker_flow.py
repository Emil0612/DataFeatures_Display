"""Broker flow: who traded, using the buyer and seller broker codes on each trade.

Many venues do not publish broker codes; on those, every feature here is missing. Brokers are
mapped to a participant type (institutional / retail / market_maker) through a broker directory;
codes not in the directory count as "unknown".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import BarBinner, session_grid

BROKER_TYPES = ("institutional", "retail", "market_maker", "unknown")


def _side_stats(b: BarBinner, qty: np.ndarray, broker: np.ndarray) -> pd.DataFrame:
    """Per bar: top broker's share of the side, Herfindahl index and number of distinct brokers."""
    df = pd.DataFrame({"bar": b.code, "broker": broker, "q": qty})[b.valid]
    per_broker = df.groupby(["bar", "broker"])["q"].sum()
    share = per_broker / per_broker.groupby(level=0).transform("sum")
    by_bar = share.groupby(level=0)
    stats = pd.DataFrame(
        {"top": by_bar.max(), "hhi": (share**2).groupby(level=0).sum(), "n": by_bar.size().astype(float)}
    ).reindex(range(len(b.grid)))
    stats.index = b.grid
    return stats


def broker_features(clean: pd.DataFrame, directory: dict[str, str], days) -> pd.DataFrame:
    """Per-bar broker flow features. `directory` maps broker code -> participant type."""
    coded = clean.loc[clean["buyer"].notna() & clean["seller"].notna()]
    b = BarBinner(coded["timestamp"], session_grid(days))
    qty = coded["quantity"].to_numpy(dtype=float)
    buyer = coded["buyer"].to_numpy()
    seller = coded["seller"].to_numpy()
    buyer_type = coded["buyer"].map(directory).fillna("unknown").to_numpy()
    seller_type = coded["seller"].map(directory).fillna("unknown").to_numpy()
    out = pd.DataFrame(index=b.grid)

    total = b.sum(qty)
    has_flow = total > 0

    # Each trade has a buyer and a seller; averaging the two sides keeps the shares summing to 1.
    bought = {t: b.sum(qty, where=buyer_type == t) for t in BROKER_TYPES}
    sold = {t: b.sum(qty, where=seller_type == t) for t in BROKER_TYPES}
    for t in BROKER_TYPES:
        out[f"share_{t}"] = ((bought[t] + sold[t]) / 2 / total).where(has_flow)

    inst = (bought["institutional"] + sold["institutional"]) / 2
    retail = (bought["retail"] + sold["retail"]) / 2
    out["broker_ratio"] = ((inst - retail) / total).where(has_flow)
    out["inst_net_flow"] = (bought["institutional"] - sold["institutional"]).where(has_flow)
    out["retail_net_flow"] = (bought["retail"] - sold["retail"]).where(has_flow)

    # Signed flow by the type of the broker that initiated the trade.
    side = coded["aggressor"].to_numpy()
    aggressor_type = np.where(side == "buy", buyer_type, np.where(side == "sell", seller_type, "none"))
    signed_qty = qty * np.where(side == "buy", 1.0, np.where(side == "sell", -1.0, 0.0))
    for t in BROKER_TYPES:
        out[f"signed_vol_{t}"] = b.sum(signed_qty, where=aggressor_type == t).where(has_flow)
    out["inst_signed_share"] = (out["signed_vol_institutional"] / total).clip(-1, 1).where(has_flow)

    buy = _side_stats(b, qty, buyer)
    sell = _side_stats(b, qty, seller)
    out["top_buyer_share"] = buy["top"]
    out["top_seller_share"] = sell["top"]
    out["cross_trade_ratio"] = (b.sum(qty, where=buyer == seller) / total).where(has_flow)
    out["n_unique_buyers"] = buy["n"].fillna(0.0)
    out["n_unique_sellers"] = sell["n"].fillna(0.0)
    out["broker_skew"] = (out["n_unique_buyers"] - out["n_unique_sellers"]).where(has_flow)
    out["broker_hhi_buy"] = buy["hhi"]
    out["broker_hhi_sell"] = sell["hhi"]
    return out


def news_event_features(broker: pd.DataFrame, news_bars: pd.Series, window_bars: int = 5, baseline_bars: int = 60) -> pd.DataFrame:
    """Broker flow in the minutes after company news.

    A bar is in an event window if a news item became visible in it or in the previous
    `window_bars - 1` bars (items are snapped to the first bar after their timestamp). Only bars
    after the news count; a window that also covered the minutes before it would know the news
    was coming. The institutional-share baseline is the median over the trailing `baseline_bars`
    bars outside event windows.
    """
    grid = broker.index
    flag = np.zeros(len(grid), dtype=bool)
    for p in grid.get_indexer(pd.DatetimeIndex(news_bars.to_numpy())):
        if p >= 0:
            flag[p:p + window_bars] = True
    in_window = pd.Series(flag, index=grid)

    inst = broker["share_institutional"]
    baseline = inst.where(~in_window).rolling(baseline_bars, min_periods=10).median()
    return pd.DataFrame(
        {
            "in_event_window": in_window.astype(float),
            "news_flow_imb_event": broker["broker_ratio"].where(in_window),
            "news_flow_inst_share_event_delta": (inst - baseline).where(in_window),
        },
        index=grid,
    )
