"""Order flow: aggressor-signed volume, price pressure, special-session flow and trade timing."""

from __future__ import annotations

import pandas as pd

from .schema import MMT_CONTINUOUS, BarBinner, session_grid, trading_mode


def order_flow_features(trades: pd.DataFrame, days) -> pd.DataFrame:
    """Per-bar order flow features. `trades` must be cleaned (see bars.clean_trades)."""
    b = BarBinner(trades["timestamp"], session_grid(days))
    qty = trades["quantity"].to_numpy(dtype=float)
    price = trades["price"].to_numpy(dtype=float)
    out = pd.DataFrame(index=b.grid)

    # Aggressor-signed volume: buyer-initiated counts +, seller-initiated counts -.
    # Trades without a known aggressor are left out of both sides.
    side = trades["aggressor"].to_numpy()
    buy = b.sum(qty, where=side == "buy")
    sell = b.sum(qty, where=side == "sell")
    signed_total = buy + sell
    out["volume_imbalance"] = ((buy - sell) / signed_total).where(signed_total > 0)
    out["net_volume_delta"] = buy - sell

    # VWAP minus close: positive when most volume traded above where the bar closed.
    volume = b.sum(qty)
    out["vt_divergence"] = b.sum(price * qty) / volume.where(volume > 0) - b.agg(price, "last")

    # Share of volume printed outside continuous trading (auctions, negotiated trades).
    mode = trading_mode(trades["mmt"])
    known = mode.notna().to_numpy()
    special = known & ~mode.isin(MMT_CONTINUOUS).fillna(False).to_numpy()
    known_qty = b.sum(qty, where=known)
    out["mmt_special_session_share"] = (b.sum(qty, where=special) / known_qty).where(known_qty > 0)

    out["trade_arrival_rate"] = b.count() / 60.0

    # Gaps between consecutive trades, attributed to the bar of the later trade. The first trade
    # of each day has no gap (the overnight break is not a trading gap).
    ts = pd.Series(pd.DatetimeIndex(trades["timestamp"]))
    gaps = ts.diff().dt.total_seconds().where(ts.dt.date == ts.shift(1).dt.date).to_numpy()
    out["mean_inter_trade_seconds"] = b.agg(gaps, "mean")
    out["max_inter_trade_seconds"] = b.agg(gaps, "max")
    return out
