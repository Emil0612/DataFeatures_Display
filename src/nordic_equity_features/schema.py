"""Shared conventions: trade-record columns, tradecode bits, MMT decoding and the session clock.

Every feature module works on the same normalised trade table:

    timestamp   tz-aware execution time (Europe/Stockholm)
    symbol      instrument identifier
    price       trade price
    quantity    shares traded
    aggressor   "buy" / "sell" (side that crossed the spread), missing when unknown
    tradecode   integer bitfield describing the print (see TradeCode)
    mmt         Market Model Typology flags (FIX MMT); position 2 is the trading mode
    buyer       buying broker code, missing on venues that do not publish it
    seller      selling broker code, missing on venues that do not publish it
    cancelled   True if the trade was later cancelled

Bars follow one timing rule everywhere: a bar labelled T contains trades in [T - 1 min, T).
The label is the moment the bar is complete, so a feature stamped T only uses information that
existed before T. Event data (news, corporate actions, end-of-day values) is aligned to the same
clock: anything that happens at time e becomes visible in the first bar labelled strictly after e.
"""

from __future__ import annotations

import datetime as dt
from enum import IntFlag

import numpy as np
import pandas as pd

TZ = "Europe/Stockholm"
BAR = pd.Timedelta("1min")

# Nasdaq Stockholm main-market session: opening auction at 09:00, closing auction around 17:30.
SESSION_OPEN = dt.time(9, 0)
SESSION_CLOSE = dt.time(17, 30)
BARS_PER_DAY = 510  # (17:30 - 09:00) / 1 min

TRADE_COLUMNS = [
    "timestamp", "symbol", "price", "quantity", "aggressor",
    "tradecode", "mmt", "buyer", "seller", "cancelled",
]


class TradeCode(IntFlag):
    """Tradecode bitfield as published in the Millistream trade feed (subset used here)."""

    OUTSIDE_SPREAD = 2      # executed outside the displayed best bid/offer
    OFF_BOOK = 4            # negotiated / off-book execution
    CORRECTION = 8          # correction of an earlier print
    TRADE_BREAK = 16        # replacement for a broken (cancelled) trade
    UPDATES_HIGH_LOW = 32   # the three on-book bits: the print updates the order book's
    UPDATES_VOLUME = 64     #   high/low, volume/turnover and last price
    UPDATES_LAST = 128
    DELAYED = 512           # disseminated late (deferred publication)
    DARK_POOL = 1024        # executed in a dark pool


ON_BOOK = TradeCode.UPDATES_HIGH_LOW | TradeCode.UPDATES_VOLUME | TradeCode.UPDATES_LAST

# Prints that are not real trades in their own right and are always dropped.
INVALID_PRINTS = TradeCode.CORRECTION | TradeCode.TRADE_BREAK

# Prints excluded when building lit price bars: everything that does not reflect the visible
# order book at the time of execution.
NON_LIT_PRINTS = INVALID_PRINTS | TradeCode.OFF_BOOK | TradeCode.DELAYED | TradeCode.DARK_POOL

# MMT position 2 ("trading mode") values. Continuous trading is '2'; separators mean "not set".
MMT_CONTINUOUS = {"2", "-", "_", " "}
MMT_OPENING_AUCTION = "O"
MMT_CLOSING_AUCTION = "K"
MMT_PERIODIC_AUCTION = "P"
MMT_NEGOTIATED = "5"


def has_bits(tradecode: pd.Series, bits: int) -> pd.Series:
    """True where any of `bits` is set in the tradecode bitfield."""
    return (tradecode.fillna(0).astype("int64") & int(bits)) != 0


def trading_mode(mmt: pd.Series) -> pd.Series:
    """Position 2 of the MMT string (the trading mode); missing when the field is too short."""
    s = mmt.astype("string")
    return s.where(s.str.len() >= 2).str[1]


def is_continuous(mmt: pd.Series) -> pd.Series:
    """True for prints from continuous trading; False for auctions, negotiated or unknown."""
    return trading_mode(mmt).isin(MMT_CONTINUOUS).fillna(False).astype(bool)


def session_grid(days) -> pd.DatetimeIndex:
    """All bar labels for the given trading days: 09:01, 09:02, ..., 17:30 local time."""
    labels = []
    for day in pd.DatetimeIndex(pd.to_datetime(list(days))).normalize().unique():
        start = pd.Timestamp.combine(day.date(), SESSION_OPEN).tz_localize(TZ) + BAR
        labels.append(pd.date_range(start, periods=BARS_PER_DAY, freq=BAR, unit="ns"))
    if not labels:
        return pd.DatetimeIndex([], tz=TZ).as_unit("ns")
    return labels[0].append(labels[1:]) if len(labels) > 1 else labels[0]


def bar_label(timestamps) -> pd.DatetimeIndex:
    """Label of the bar each timestamp falls into: floor to the minute, then + 1 minute."""
    ts = pd.DatetimeIndex(timestamps).tz_convert(TZ).as_unit("ns")
    return ts.floor(BAR) + BAR


class BarBinner:
    """Assigns each trade to its bar on a session grid once, then aggregates per bar.

    Trades whose bar is not on the grid (outside the session) are ignored. Every aggregate comes
    back as a Series indexed by the grid; bars without (selected) trades get 0 for sums and counts
    and NaN for everything else.
    """

    def __init__(self, timestamps, grid: pd.DatetimeIndex):
        self.grid = grid.as_unit("ns")
        self.code = self.grid.get_indexer(bar_label(timestamps))
        self.valid = self.code >= 0

    def _mask(self, where) -> np.ndarray:
        return self.valid if where is None else self.valid & np.asarray(where, dtype=bool)

    def sum(self, values, where=None) -> pd.Series:
        m = self._mask(where)
        v = np.asarray(values, dtype=float)[m]
        return pd.Series(np.bincount(self.code[m], weights=v, minlength=len(self.grid)), index=self.grid)

    def count(self, where=None) -> pd.Series:
        m = self._mask(where)
        return pd.Series(np.bincount(self.code[m], minlength=len(self.grid)).astype(float), index=self.grid)

    def agg(self, values, how: str, where=None) -> pd.Series:
        """Per-bar max / min / first / last / mean / std of `values` (in trade order)."""
        m = self._mask(where)
        s = pd.Series(np.asarray(values)[m], index=self.code[m])
        out = getattr(s.groupby(level=0, sort=True), how)()
        return pd.Series(out.reindex(range(len(self.grid))).to_numpy(), index=self.grid)


def snap_to_next_bar(event_times: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """Map each event to the first bar labelled strictly after it.

    An event at exactly T lands in bar T + 1 min, because bar T only covers [T - 1 min, T).
    Pre-market events land in the day's first bar; events after the last bar are dropped.
    """
    times = pd.DatetimeIndex(pd.to_datetime(event_times)).tz_convert(TZ).as_unit("ns")
    grid = grid.as_unit("ns")
    pos = grid.searchsorted(times, side="right")
    keep = pos < len(grid)
    return pd.Series(grid[pos[keep]], index=np.asarray(event_times.index)[keep])


def trading_day(labels: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Calendar day (tz-naive midnight) of each bar label."""
    return pd.DatetimeIndex(labels).tz_convert(TZ).tz_localize(None).normalize()
