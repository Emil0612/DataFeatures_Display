"""Venue decomposition: lit vs dark vs off-book flow, and how far the all-market price drifts from lit.

Each cleaned trade is put in exactly one class using its tradecode bits:

    dark      dark-pool bit set
    offbook   off-book bit set (and not dark)
    lit       any of the three on-book bits set (and neither of the above)
    residual  none of the classification bits set
"""

from __future__ import annotations

import pandas as pd

from .bars import build_bars
from .schema import ON_BOOK, BarBinner, TradeCode, has_bits, session_grid

BPS = 1e4
VENUES = ("lit", "dark", "offbook", "residual")


def classify_venue(trades: pd.DataFrame) -> pd.Series:
    tc = trades["tradecode"]
    dark = has_bits(tc, TradeCode.DARK_POOL)
    offbook = ~dark & has_bits(tc, TradeCode.OFF_BOOK)
    lit = ~dark & ~offbook & has_bits(tc, ON_BOOK)
    venue = pd.Series("residual", index=trades.index)
    venue[lit] = "lit"
    venue[offbook] = "offbook"
    venue[dark] = "dark"
    return venue


def venue_features(clean: pd.DataFrame, days) -> pd.DataFrame:
    """Per-bar venue volumes, shares and lit-vs-all price divergence. `clean` must be cleaned trades."""
    b = BarBinner(clean["timestamp"], session_grid(days))
    qty = clean["quantity"].to_numpy(dtype=float)
    venue = classify_venue(clean).to_numpy()
    out = pd.DataFrame(index=b.grid)

    total = b.sum(qty)
    for v in VENUES:
        vol = b.sum(qty, where=venue == v)
        out[f"volume_{v}"] = vol
        out[f"vshare_{v}"] = (vol / total).where(total > 0)

    # Bit shares straight from the tradecode (a print can carry several bits).
    tc = clean["tradecode"]
    for name, bit in (
        ("offbook_share", TradeCode.OFF_BOOK),
        ("dark_share", TradeCode.DARK_POOL),
        ("outside_spread_share", TradeCode.OUTSIDE_SPREAD),
    ):
        out[name] = (b.sum(qty, where=has_bits(tc, bit)) / total).where(total > 0)

    is_lit = venue == "lit"
    turnover = clean["price"].to_numpy(dtype=float) * qty
    total_turnover = b.sum(turnover)
    out["volume_lit_share"] = out["vshare_lit"]
    out["turnover_lit_share"] = (b.sum(turnover, where=is_lit) / total_turnover).where(total_turnover > 0)

    # Price divergence between the all-market bar and the lit-only bar. Delayed off-book prints
    # carry a stale execution price, so a contemporaneous variant drops them.
    all_bars = build_bars(clean, days)
    lit_bars = build_bars(clean.loc[is_lit], days)
    delayed_offbook = has_bits(tc, TradeCode.OFF_BOOK) & has_bits(tc, TradeCode.DELAYED)
    contemp_bars = build_bars(clean.loc[~delayed_offbook], days)

    out["close_divergence"] = all_bars["close"] - lit_bars["close"]
    out["close_divergence_bps"] = out["close_divergence"] / lit_bars["close"] * BPS
    out["close_divergence_contemporaneous"] = contemp_bars["close"] - lit_bars["close"]
    out["close_divergence_contemporaneous_bps"] = out["close_divergence_contemporaneous"] / lit_bars["close"] * BPS
    out["high_divergence"] = all_bars["high"] - lit_bars["high"]
    out["low_divergence"] = lit_bars["low"] - all_bars["low"]
    return out
