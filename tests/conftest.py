import pandas as pd
import pytest

import nordic_equity_features as nef

TZ = "Europe/Stockholm"
SYMBOLS = ("NORDA", "FJORD", "SKOG")


@pytest.fixture(scope="session")
def data():
    return nef.generate(n_days=8, seed=3)


@pytest.fixture(scope="session")
def features(data):
    return {s: nef.compute_features(data, s) for s in SYMBOLS}


def at(day, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{pd.Timestamp(day).date()} {hhmm}", tz=TZ)
