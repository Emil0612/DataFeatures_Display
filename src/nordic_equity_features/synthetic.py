"""Synthetic Nordic-style market data: fictional instruments, simulated ticks and events.

Nothing here is real. Tickers, companies, brokers, prices and news are invented, and dates are in
the 2030s so they cannot be mistaken for history. The generator reproduces the structure that the
features rely on, not any real market's statistics:

* a benchmark index and three stocks driven by a one-factor model, minute by minute
* Poisson trade arrivals with a U-shaped intraday profile, bid-ask bounce on a tick grid
* opening and closing auction prints, off-book (some published late), dark-pool, corrected and
  cancelled trades, each carrying the matching tradecode bits and MMT trading mode
* broker codes on one instrument only (most venues do not publish them)
* a 2-for-1 split, a cash dividend announced mid-session, and corporate / sector / macro news
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import BARS_PER_DAY, ON_BOOK, TZ, TradeCode, session_grid

START_DAY = "2030-01-07"
CONTINUOUS_MINUTES = 505  # 09:00-17:25, then the closing call until the uncross just before 17:30
INDEX_NAME = "nordidx"

BROKERS = {
    "INST01": "institutional", "INST02": "institutional", "INST03": "institutional", "INST04": "institutional",
    "RETL01": "retail", "RETL02": "retail", "RETL03": "retail",
    "MMKR01": "market_maker", "MMKR02": "market_maker",
}
UNLISTED_BROKERS = ["OTHR01", "OTHR02", "OTHR03"]


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    sector: str
    country: str
    listing_date: str
    n_listings: int
    shares_outstanding: float
    start_price: float
    beta: float
    idio_vol: float          # per-minute volatility of the stock-specific component
    trades_per_minute: float
    mean_size: float
    tick: float
    broker_coded: bool


INSTRUMENTS = (
    Instrument("NORDA", "Norda Industrial (fictional)", "Industrials", "SE", "2011-05-12", 3,
               450e6, 250.0, 1.1, 0.0006, 8.0, 300.0, 0.05, False),
    Instrument("FJORD", "Fjord Shipping (fictional)", "Energy", "NO", "2016-09-28", 2,
               120e6, 84.0, 0.8, 0.0009, 3.0, 500.0, 0.02, False),
    Instrument("SKOG", "Skog Biotech (fictional)", "Health Care", "SE", "2024-03-15", 1,
               40e6, 18.5, 1.3, 0.0015, 1.2, 900.0, 0.01, True),
)


@dataclass
class MarketData:
    """Everything the feature pipeline needs, for one or more instruments."""

    days: list[pd.Timestamp]
    trades: pd.DataFrame
    index_bars: pd.DataFrame
    index_name: str
    instruments: pd.DataFrame
    brokers: dict[str, str]
    splits: pd.DataFrame
    dividends: pd.DataFrame
    news: pd.DataFrame
    eod: pd.DataFrame
    meta: dict = field(default_factory=dict)


def trading_days(n_days: int, start: str = START_DAY) -> list[pd.Timestamp]:
    return list(pd.bdate_range(start, periods=n_days))


def _intraday_profile() -> np.ndarray:
    """U-shaped trade intensity: busy after the open and before the close."""
    m = np.arange(CONTINUOUS_MINUTES, dtype=float)
    return 1.0 + 2.0 * np.exp(-m / 25.0) + 1.5 * np.exp(-(CONTINUOUS_MINUTES - m) / 30.0)


def _mmt(mode: str) -> str:
    return "1" + mode + "-" * 12


def _ts(day: pd.Timestamp, seconds: np.ndarray) -> pd.DatetimeIndex:
    base = pd.Timestamp.combine(day.date(), pd.Timestamp("09:00").time()).tz_localize(TZ)
    return base + pd.to_timedelta(seconds, unit="s")


def _simulate_index(days, rng) -> tuple[pd.DataFrame, np.ndarray]:
    """Index minute bars on the session grid, and the index minute log-returns used as the factor."""
    n = len(days) * BARS_PER_DAY
    ret = rng.normal(0.0, 0.0004, n)
    ret[::BARS_PER_DAY] += rng.normal(0.0, 0.006, len(days))  # overnight gap lands in the first bar
    level = 1500.0 * np.exp(np.cumsum(ret))
    prev = np.r_[1500.0, level[:-1]]
    wiggle = np.abs(rng.normal(0.0, 0.0001, (2, n)))
    bars = pd.DataFrame(
        {
            "open": prev,
            "high": np.maximum(prev, level) * (1 + wiggle[0]),
            "low": np.minimum(prev, level) * (1 - wiggle[1]),
            "close": level,
        },
        index=session_grid(days),
    )
    return bars, ret


def _simulate_trades(inst: Instrument, days, factor: np.ndarray, rng, split_ex=None, dividend=None) -> pd.DataFrame:
    """Tick trades for one instrument over all days."""
    profile = _intraday_profile()
    log_p = np.log(inst.start_price)
    rows = []
    for d, day in enumerate(days):
        f = factor[d * BARS_PER_DAY:(d + 1) * BARS_PER_DAY]
        if split_ex is not None and day == split_ex:
            log_p += np.log(0.5)  # 2-for-1 split: the price halves at the open of the ex-day
        if dividend is not None and day == dividend[0]:
            log_p = np.log(np.exp(log_p) - dividend[1])  # price drops by the dividend on the ex-day
        minute_ret = inst.beta * f + rng.normal(0.0, inst.idio_vol, BARS_PER_DAY)
        path = log_p + np.r_[0.0, np.cumsum(minute_ret)]  # efficient log price at each minute edge
        log_p = path[-1]

        # Continuous trading.
        counts = rng.poisson(inst.trades_per_minute * profile)
        minute = np.repeat(np.arange(CONTINUOUS_MINUTES), counts)
        sec = minute * 60 + np.sort(rng.uniform(0, 60, counts.sum()))
        sec = np.sort(sec)
        frac = (sec / 60.0) - minute
        mid = np.exp(path[minute] + frac * (path[minute + 1] - path[minute]))
        n = len(sec)
        side = np.empty(n, dtype=object)
        state = rng.random() < 0.5
        flips = rng.random(n) < 0.4
        for i in range(n):
            if flips[i]:
                state = not state
            side[i] = "buy" if state else "sell"
        half_spread = inst.tick * rng.integers(1, 3, n)
        price = mid + np.where(side == "buy", half_spread, -half_spread)
        price = np.round(price / inst.tick) * inst.tick
        qty = np.maximum(1, np.round(rng.lognormal(np.log(inst.mean_size), 1.0, n)))
        qty[rng.random(n) < 0.01] *= 20  # occasional block

        kind = rng.choice(["lit", "offbook", "dark", "unclassified"], size=n, p=[0.875, 0.08, 0.04, 0.005])
        tradecode = np.where(kind == "lit", int(ON_BOOK), 0)
        tradecode = tradecode | np.where((kind == "lit") & (rng.random(n) < 0.01), int(TradeCode.OUTSIDE_SPREAD), 0)
        tradecode = tradecode | np.where(kind == "offbook", int(TradeCode.OFF_BOOK), 0)
        tradecode = tradecode | np.where(kind == "dark", int(TradeCode.DARK_POOL), 0)
        price = np.where(kind == "lit", price, np.round(mid / inst.tick) * inst.tick)
        qty = np.where(kind == "offbook", qty * 5, qty)
        mmt = np.where(kind == "offbook", _mmt("5"), _mmt("2"))
        aggressor = np.where(kind == "lit", side, None)

        # Some off-book trades are published late: stamped at publication, priced at execution.
        delayed = (kind == "offbook") & (rng.random(n) < 0.3)
        sec = np.where(delayed, np.minimum(sec + rng.uniform(60, 900, n), CONTINUOUS_MINUTES * 60 - 1), sec)
        tradecode = tradecode | np.where(delayed, int(TradeCode.DELAYED), 0)

        day_trades = pd.DataFrame(
            {
                "timestamp": _ts(day, sec),
                "price": price,
                "quantity": qty,
                "aggressor": aggressor,
                "tradecode": tradecode.astype("int64"),
                "mmt": mmt,
                "cancelled": rng.random(n) < 0.001,
            }
        )

        # Corrections re-print an earlier trade; they are not new volume.
        corr = day_trades.sample(frac=0.002, random_state=rng.integers(1 << 31))
        corr = corr.assign(
            tradecode=corr["tradecode"] | int(TradeCode.CORRECTION),
            timestamp=corr["timestamp"] + pd.to_timedelta(rng.uniform(1, 120, len(corr)), unit="s"),
        )

        # Opening and closing auctions: one uncross print each.
        avg = inst.mean_size * inst.trades_per_minute
        auctions = pd.DataFrame(
            {
                "timestamp": _ts(day, np.array([0.0, CONTINUOUS_MINUTES * 60 + 290.0])),
                "price": np.round(np.exp([path[0], path[CONTINUOUS_MINUTES]]) / inst.tick) * inst.tick,
                "quantity": np.round(avg * rng.uniform(15, 40, 2)),
                "aggressor": [None, None],
                "tradecode": [int(ON_BOOK)] * 2,
                "mmt": [_mmt("O"), _mmt("K")],
                "cancelled": [False, False],
            }
        )
        rows += [auctions.iloc[:1], day_trades, corr, auctions.iloc[1:]]

    trades = pd.concat(rows, ignore_index=True).sort_values("timestamp", kind="stable").reset_index(drop=True)
    trades["symbol"] = inst.symbol
    if inst.broker_coded:
        codes = list(BROKERS) + UNLISTED_BROKERS
        weights = np.array([6, 5, 4, 3, 6, 5, 4, 8, 7, 1, 1, 1], dtype=float)
        weights /= weights.sum()
        trades["buyer"] = rng.choice(codes, len(trades), p=weights)
        cross = rng.random(len(trades)) < 0.05
        trades["seller"] = np.where(cross, trades["buyer"], rng.choice(codes, len(trades), p=weights))
    else:
        trades["buyer"] = None
        trades["seller"] = None
    trades["aggressor"] = trades["aggressor"].astype("object")
    return trades


def _session_time(day: pd.Timestamp, hh_mm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day.date()} {hh_mm}").tz_localize(TZ)


def _simulate_news(days, instruments, rng) -> pd.DataFrame:
    from .news import CORPORATE_CATEGORIES, MACRO_CATEGORIES

    rows = []
    for day in days:
        base = _session_time(day, "00:00")
        for inst in instruments:
            for _ in range(rng.poisson(0.8)):
                # Pre-market, in-session and after-hours releases.
                start, end = [(7.0, 9.0), (9.0, 17.5), (17.5, 19.0)][rng.choice(3, p=[0.4, 0.5, 0.1])]
                rows.append((base + pd.Timedelta(hours=rng.uniform(start, end)), "corporate",
                             inst.symbol, inst.sector, rng.choice(CORPORATE_CATEGORIES)))
        for sector in sorted({i.sector for i in instruments}):
            for _ in range(rng.poisson(1.5)):
                rows.append((base + pd.Timedelta(hours=rng.uniform(7.0, 18.0)), "sector",
                             None, sector, rng.choice(CORPORATE_CATEGORIES)))
        for _ in range(rng.poisson(2.0)):
            rows.append((base + pd.Timedelta(hours=rng.uniform(8.0, 16.5)), "macro",
                         None, None, rng.choice(MACRO_CATEGORIES)))
    news = pd.DataFrame(rows, columns=["timestamp", "scope", "symbol", "sector", "category"])
    return news.sort_values("timestamp").reset_index(drop=True)


def generate(n_days: int = 25, seed: int = 7, instruments=INSTRUMENTS) -> MarketData:
    """Generate a complete synthetic dataset for `n_days` trading days."""
    rng = np.random.default_rng(seed)
    days = trading_days(n_days)
    index_bars, factor = _simulate_index(days, rng)

    # Events are placed relative to the window so short test datasets still contain them.
    split_ann, split_ex = days[max(1, n_days // 5)], days[int(n_days * 0.6)]
    div_ann, div_ex = days[max(1, n_days // 8)], days[int(n_days * 0.45)]
    splits = pd.DataFrame(
        [{"symbol": "FJORD", "announced": _session_time(split_ann, "08:30"), "ex_date": split_ex,
          "new_shares": 2.0, "old_shares": 1.0}]
    )
    dividends = pd.DataFrame(
        [{"symbol": "NORDA", "announced": _session_time(div_ann, "12:15"), "ex_date": div_ex, "amount": 4.5}]
    )

    trades, eod = [], []
    for inst in instruments:
        t = _simulate_trades(
            inst, days, factor, rng,
            split_ex=split_ex if inst.symbol == "FJORD" else None,
            dividend=(div_ex, 4.5) if inst.symbol == "NORDA" else None,
        )
        trades.append(t)
        close = t.groupby(t["timestamp"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize())["price"].last()
        shares = pd.Series(inst.shares_outstanding, index=close.index)
        if inst.symbol == "FJORD":
            shares[shares.index >= split_ex] *= 2
        eod.append(pd.DataFrame({"symbol": inst.symbol, "date": close.index, "close": close.to_numpy(),
                                 "mcap": (close * shares).to_numpy()}))

    instruments_df = pd.DataFrame(
        [{"symbol": i.symbol, "name": i.name, "sector": i.sector, "country": i.country,
          "listing_date": i.listing_date, "n_listings": i.n_listings, "broker_coded": i.broker_coded}
         for i in instruments]
    ).set_index("symbol", drop=False)

    trade_cols = ["timestamp", "symbol", "price", "quantity", "aggressor", "tradecode", "mmt", "buyer", "seller", "cancelled"]
    return MarketData(
        days=days,
        trades=pd.concat(trades, ignore_index=True)[trade_cols],
        index_bars=index_bars,
        index_name=INDEX_NAME,
        instruments=instruments_df,
        brokers=dict(BROKERS),
        splits=splits,
        dividends=dividends,
        news=_simulate_news(days, instruments, rng),
        eod=pd.concat(eod, ignore_index=True),
        meta={"seed": seed, "n_days": n_days},
    )
