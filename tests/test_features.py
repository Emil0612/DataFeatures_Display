"""Correctness of individual estimators on small hand-made inputs, plus bounds on the demo data."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso

from nordic_equity_features.cross_asset import rolling_lasso_beta
from nordic_equity_features.liquidity import _roll_spread, _weighted_moments
from nordic_equity_features.order_flow import order_flow_features
from nordic_equity_features.schema import TZ, TradeCode, session_grid
from nordic_equity_features.venue import classify_venue

from .conftest import SYMBOLS, at

DAY = pd.Timestamp("2030-01-07")


def _trades(rows):
    df = pd.DataFrame(rows, columns=["time", "price", "quantity", "aggressor"])
    return df.assign(
        timestamp=pd.DatetimeIndex(f"{DAY.date()} " + df["time"]).tz_localize(TZ),
        symbol="TEST", tradecode=224, mmt="12------------", buyer=None, seller=None, cancelled=False,
    ).drop(columns="time")


def test_volume_imbalance_and_vwap_divergence():
    t = _trades([("10:00:01", 10.0, 300, "buy"), ("10:00:30", 10.2, 100, "sell"), ("10:00:50", 10.1, 100, None)])
    f = order_flow_features(t, [DAY]).loc[at(DAY, "10:01")]
    assert f["volume_imbalance"] == pytest.approx((300 - 100) / 400)
    assert f["net_volume_delta"] == 200
    vwap = (10.0 * 300 + 10.2 * 100 + 10.1 * 100) / 500
    assert f["vt_divergence"] == pytest.approx(vwap - 10.1)


def test_weighted_moments():
    t = _trades([("10:00:01", 10.0, 100, "buy"), ("10:00:02", 11.0, 300, "buy"), ("10:00:03", 13.0, 100, "buy")])
    m = _weighted_moments(t, session_grid([DAY])).loc[at(DAY, "10:01")]
    w = np.array([100, 300, 100]); p = np.array([10.0, 11.0, 13.0])
    mean = np.average(p, weights=w)
    sd = np.sqrt(np.average((p - mean) ** 2, weights=w))
    assert m["vwap"] == pytest.approx(mean)
    assert m["vwad"] == pytest.approx(sd)
    assert m["skew"] == pytest.approx(np.average(((p - mean) / sd) ** 3, weights=w))


def test_roll_spread_recovers_a_pure_bounce():
    # Prices alternate between bid 9.95 and ask 10.05 around a constant value of 10.
    prices = [9.95, 10.05, 9.95, 10.05, 9.95, 10.05, 9.95, 10.05]
    t = _trades([(f"10:00:{i:02d}", p, 100, "buy") for i, p in enumerate(prices)])
    roll = _roll_spread(t, session_grid([DAY])).loc[at(DAY, "10:01")]
    dp = np.diff(prices)
    expected = 2 * np.sqrt(-np.cov(dp[1:], dp[:-1])[0, 1])
    assert roll == pytest.approx(expected)
    assert roll > 0


def test_closed_form_lasso_matches_scikit_learn():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 4e-4, 300))
    y = 0.9 * x + pd.Series(rng.normal(0, 5e-4, 300))
    beta = rolling_lasso_beta(y, x, window=60, alpha=3e-8)
    for end in (59, 150, 299):
        model = Lasso(alpha=3e-8, fit_intercept=True, max_iter=100_000, tol=1e-12)
        model.fit(x.iloc[end - 59:end + 1].to_frame(), y.iloc[end - 59:end + 1])
        assert beta.iloc[end] == pytest.approx(model.coef_[0], rel=1e-4, abs=1e-8)
    assert beta.iloc[:59].isna().all()


def test_venue_classification_priority():
    tc = pd.Series([int(TradeCode.DARK_POOL | TradeCode.OFF_BOOK), int(TradeCode.OFF_BOOK), 224, 0])
    assert classify_venue(pd.DataFrame({"tradecode": tc})).tolist() == ["dark", "offbook", "lit", "residual"]


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_bounds_on_demo_data(features, symbol):
    f = features[symbol]
    shares = [c for c in f.columns if c.startswith(("vshare_", "share_")) or c.endswith("_share")
              and c not in ("inst_signed_share",)]
    for col in shares:
        v = f[col].dropna()
        assert ((v >= -1e-12) & (v <= 1 + 1e-12)).all(), col
    assert f["volume_imbalance"].dropna().between(-1, 1).all()
    assert (f["high"] >= f["low"]).all()
    assert (f["vol_bar_partial_fill_progress"].dropna() >= 0).all()
    assert len(f) == 8 * 510
    assert f.index.is_monotonic_increasing and f.index.is_unique
