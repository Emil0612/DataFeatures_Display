"""Targeted tests for each lookahead mechanism and the bar timing convention."""

import numpy as np
import pandas as pd

from nordic_equity_features.bars import build_bars
from nordic_equity_features.corporate_actions import adjusted_ohlcv, dividend_features, split_features
from nordic_equity_features.metadata import metadata_features
from nordic_equity_features.news import news_features
from nordic_equity_features.schema import TZ, session_grid, snap_to_next_bar
from nordic_equity_features.transactions import prior_window_threshold

from .conftest import at


def _trades(times, prices, qty=100.0):
    n = len(times)
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(times).tz_localize(TZ) if pd.DatetimeIndex(times).tz is None else times,
            "symbol": "TEST", "price": prices, "quantity": np.full(n, qty), "aggressor": "buy",
            "tradecode": 224, "mmt": "12------------", "buyer": None, "seller": None, "cancelled": False,
        }
    )


def test_bar_t_contains_trades_from_the_minute_before_t():
    day = pd.Timestamp("2030-01-07")
    trades = _trades(["2030-01-07 10:00:00", "2030-01-07 10:00:59.999", "2030-01-07 10:01:00"], [10.0, 11.0, 12.0])
    bars = build_bars(trades, [day])
    assert bars.loc[at(day, "10:01"), "close"] == 11.0
    assert bars.loc[at(day, "10:01"), "trade_count"] == 2
    assert bars.loc[at(day, "10:02"), "close"] == 12.0
    # Empty bars carry the last close forward and have no volume.
    assert bars.loc[at(day, "10:05"), "close"] == 12.0
    assert bars.loc[at(day, "10:05"), "volume"] == 0


def test_events_snap_to_the_next_bar():
    day = pd.Timestamp("2030-01-07")
    grid = session_grid([day])
    events = pd.Series(pd.DatetimeIndex(
        ["2030-01-07 07:45", "2030-01-07 10:15:00", "2030-01-07 10:15:00.000001", "2030-01-07 18:00"]
    ).tz_localize(TZ))
    snapped = snap_to_next_bar(events, grid)
    assert list(snapped) == [at(day, "09:01"), at(day, "10:16"), at(day, "10:16")]  # after-hours dropped


def test_block_threshold_uses_only_prior_days():
    days = pd.bdate_range("2030-01-07", periods=7)
    rows = []
    for i, day in enumerate(days):
        times = pd.date_range(f"{day.date()} 10:00", periods=50, freq="1min")
        rows.append(_trades(times, np.full(50, 10.0), qty=float(i + 1)))
    trades = pd.concat(rows, ignore_index=True)
    theta = prior_window_threshold(trades, days, min_days=5)
    assert theta.iloc[:5].isna().all()  # fewer than five prior days
    assert theta.iloc[5] == 5.0         # sizes 1..5 from the five prior days, never day 6's size 6
    # Changing today's trades does not move today's threshold.
    bigger = trades.copy()
    bigger.loc[pd.DatetimeIndex(bigger["timestamp"]).date == days[5].date(), "quantity"] = 1e9
    assert prior_window_threshold(bigger, days, min_days=5).iloc[5] == 5.0


def test_market_cap_is_yesterdays_close():
    days = pd.bdate_range("2030-01-07", periods=3)
    grid = session_grid(days)
    eod = pd.DataFrame({"date": days, "mcap": [100.0, 200.0, 300.0]})
    inst = {"symbol": "TEST", "country": "SE", "listing_date": "2020-01-01", "n_listings": 1}
    f = metadata_features(grid, eod, inst)
    assert f.loc[at(days[0], "09:01"):at(days[0], "17:30"), "mcap"].isna().all()
    assert (f.loc[at(days[1], "09:01"):at(days[1], "17:30"), "mcap"] == 100.0).all()
    assert (f.loc[at(days[2], "09:01"):at(days[2], "17:30"), "mcap"] == 200.0).all()


def test_dividend_is_invisible_until_announced():
    days = pd.bdate_range("2030-01-07", periods=6)
    bars = pd.DataFrame({"close": 100.0}, index=session_grid(days))
    div = pd.DataFrame([{"symbol": "TEST", "announced": at(days[1], "12:15"), "ex_date": days[4], "amount": 2.0}])
    f = dividend_features(bars, div)
    assert f.loc[:at(days[1], "12:15"), "dividend_ex_date"].isna().all()
    assert f.loc[at(days[1], "12:16"), "days_to_ex_dividend"] == 3
    # The declaration flag starts with the announcement, not at the open of that day.
    assert f.loc[at(days[1], "12:15"), "dividend_declaration_flag"] == 0
    assert f.loc[at(days[1], "12:16"), "dividend_declaration_flag"] == 1
    ex_day = f.loc[at(days[4], "09:01"):at(days[4], "17:30")]
    assert (ex_day["dividend_ex_flag"] == 1).all()
    assert np.allclose(ex_day["dividend_pct"], 0.02)


def test_split_features_and_as_of_adjustment():
    days = pd.bdate_range("2030-01-07", periods=6)
    grid = session_grid(days)
    close = np.where(grid < at(days[3], "09:00"), 100.0, 50.0)
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=grid)
    split = pd.DataFrame([{"symbol": "TEST", "announced": at(days[1], "08:30"),
                           "ex_date": days[3], "new_shares": 2.0, "old_shares": 1.0}])
    f = split_features(bars, split)
    assert f.loc[:at(days[0], "17:30"), "stock_split_xday"].isna().all()   # not yet announced
    assert f.loc[at(days[1], "09:01"), "stock_split_xday"] == 2
    assert f.loc[at(days[3], "09:01"), "stock_split_xday_lookback"] == 0
    assert f.loc[at(days[3], "09:01"), "split_adjusted_return"] == 0.0   # the halving is not a return

    # History as seen before the split is unadjusted; after it, earlier prices are halved.
    before = adjusted_ohlcv(bars, split, at(days[2], "17:30"))
    after = adjusted_ohlcv(bars, split, at(days[4], "17:30"))
    assert (before["ac"] == 100.0).all()
    assert (after["ac"] == 50.0).all()
    assert (after.loc[:at(days[2], "17:30"), "av"] == 2000.0).all()


def test_news_density_counts_only_past_items():
    days = [pd.Timestamp("2030-01-07")]
    grid = session_grid(days)
    news = pd.DataFrame({
        "timestamp": pd.DatetimeIndex(["2030-01-07 10:15:00", "2030-01-07 10:20:30"]).tz_localize(TZ),
        "scope": ["corporate", "corporate"], "symbol": "TEST", "sector": None, "category": ["EARNINGS", "RATING"],
    })
    f = news_features(news, grid)
    assert f.loc[at(days[0], "10:15"), "news_corp_density"] == 0
    assert f.loc[at(days[0], "10:16"), "news_corp_density"] == 1
    assert f.loc[at(days[0], "10:21"), "news_corp_cat"] == "RATING"
    assert f.loc[at(days[0], "10:20"), "news_corp_cat"] == "EARNINGS"
