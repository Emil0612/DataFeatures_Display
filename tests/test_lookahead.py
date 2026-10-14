"""Every feature must be invariant to data that arrives after the bar it is stamped on."""

import pandas as pd
import pytest

import nordic_equity_features as nef
from nordic_equity_features.lookahead import compare, lookahead_violations, perturb_after
from nordic_equity_features.transactions import BLOCK_PERCENTILE

from .conftest import SYMBOLS, at

# Cutoffs chosen to hit the boundaries: mid-session, the first bar of a day, the exact minute of
# an intraday announcement, before the open, and the last bar of a day.
CUTOFFS = [
    ("mid_session", 5, "11:37"),
    ("first_bar", 4, "09:01"),
    ("announcement_minute", 1, "12:15"),
    ("pre_market", 4, "08:00"),
    ("last_bar", 6, "17:30"),
]


@pytest.mark.parametrize("symbol", SYMBOLS)
@pytest.mark.parametrize("name,day,hhmm", CUTOFFS, ids=[c[0] for c in CUTOFFS])
def test_features_ignore_the_future(data, symbol, name, day, hhmm):
    violations = lookahead_violations(data, symbol, at(data.days[day], hhmm))
    assert violations.empty, f"features changed by future data:\n{violations.to_string()}"


def test_perturbation_changes_the_future(data, features):
    """The check has teeth only if the replaced future actually moves the features."""
    cutoff = at(data.days[5], "11:37")
    after = nef.compute_features(perturb_after(data, cutoff), "NORDA").loc[cutoff + pd.Timedelta("1min"):]
    changed = compare(features["NORDA"].loc[after.index], after)
    assert len(changed) > 50


def _same_day_block_share(data, symbol):
    """A deliberately leaky block threshold: the 95th percentile of the whole current day."""
    f = nef.compute_features(data, symbol)
    trades = data.trades.loc[data.trades["symbol"] == symbol]
    day = trades["timestamp"].dt.tz_convert("Europe/Stockholm").dt.normalize()
    theta = trades.groupby(day)["quantity"].transform(lambda q: q.quantile(BLOCK_PERCENTILE))
    label = trades["timestamp"].dt.floor("1min") + pd.Timedelta("1min")
    blocks = trades["quantity"].where(trades["quantity"] >= theta, 0.0).groupby(label).sum()
    total = trades["quantity"].groupby(label).sum()
    return (blocks / total).reindex(f.index).to_frame("leaky_block_share")


def test_check_catches_a_same_day_threshold(data):
    cutoff = at(data.days[5], "11:37")
    before = _same_day_block_share(data, "NORDA").loc[:cutoff]
    after = _same_day_block_share(perturb_after(data, cutoff), "NORDA").loc[:cutoff]
    assert not compare(before, after).empty
