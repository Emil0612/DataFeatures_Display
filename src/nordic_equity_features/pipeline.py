"""Assemble every feature family into one bar-level table for an instrument."""

from __future__ import annotations

import pandas as pd

from .bars import build_bars, clean_trades, lit_trades, price_features
from .broker_flow import broker_features, news_event_features
from .corporate_actions import dividend_features, split_features
from .cross_asset import index_features, relative_features
from .liquidity import liquidity_features
from .metadata import metadata_features
from .news import news_features
from .order_flow import order_flow_features
from .schema import session_grid, snap_to_next_bar
from .synthetic import MarketData
from .transactions import transaction_features
from .venue import venue_features
from .volume_bars import volume_bar_features

FAMILIES = (
    "bars", "transactions", "order_flow", "liquidity", "venue", "broker_flow",
    "volume_bars", "cross_asset", "corporate_actions", "metadata", "news",
)


def compute_features(data: MarketData, symbol: str) -> pd.DataFrame:
    """All features for `symbol`, one row per bar on the session grid of `data.days`."""
    days = data.days
    grid = session_grid(days)
    trades = data.trades.loc[data.trades["symbol"] == symbol]
    clean = clean_trades(trades)
    lit = lit_trades(trades)
    inst = data.instruments.loc[symbol]

    bars = build_bars(lit, days)
    parts = [
        bars[["open", "high", "low", "close", "volume", "turnover"]],
        price_features(bars),
        transaction_features(clean, days),
        order_flow_features(clean, days),
    ]
    liq = liquidity_features(clean, lit, bars, days)
    parts += [liq, venue_features(clean, days)]
    if inst["broker_coded"]:
        broker = broker_features(clean, data.brokers, days)
        corp_news = data.news.loc[(data.news["scope"] == "corporate") & (data.news["symbol"] == symbol)]
        parts += [broker, news_event_features(broker, snap_to_next_bar(corp_news["timestamp"], grid))]
    parts.append(volume_bar_features(bars))

    idx = data.index_bars.reindex(grid)
    parts.append(index_features(idx, data.index_name))
    parts.append(relative_features(bars["close"], liq["vwap"], idx["close"]))

    splits = data.splits.loc[data.splits["symbol"] == symbol]
    dividends = data.dividends.loc[data.dividends["symbol"] == symbol]
    parts += [split_features(bars, splits), dividend_features(bars, dividends)]

    eod = data.eod.loc[data.eod["symbol"] == symbol]
    parts.append(metadata_features(grid, eod, inst.to_dict()))

    news = data.news
    relevant = (
        ((news["scope"] == "corporate") & (news["symbol"] == symbol))
        | ((news["scope"] == "sector") & (news["sector"] == inst["sector"]))
        | (news["scope"] == "macro")
    )
    parts.append(news_features(news.loc[relevant], grid))

    out = pd.concat(parts, axis=1)
    duplicated = out.columns[out.columns.duplicated()].tolist()
    if duplicated:
        raise RuntimeError(f"feature name collision: {duplicated}")
    return out
