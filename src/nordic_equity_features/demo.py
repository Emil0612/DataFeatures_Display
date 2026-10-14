"""End-to-end demo on synthetic data: generate a market, compute every feature, prove no lookahead.

    uv run python -m nordic_equity_features.demo
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from .lookahead import lookahead_violations
from .pipeline import compute_features
from .schema import TZ
from .synthetic import generate

SHOWCASE = [
    "close", "log_return", "realised_vol", "volume_imbalance", "block_share", "vwap_dev_bps",
    "spread_bounce", "vshare_dark", "beta_ols", "mcap", "days_to_ex_dividend", "news_corp_density",
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=10, help="trading days to simulate (default 10)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    data = generate(n_days=args.days, seed=args.seed)
    print(f"Synthetic market: {args.days} trading days from {data.days[0].date()}, "
          f"{len(data.trades):,} trades, {len(data.news)} news items")
    for sym, n in data.trades.groupby("symbol").size().items():
        print(f"  {sym:6s} {n:>8,} trades   {data.instruments.loc[sym, 'name']}")
    print(f"  events: {len(data.splits)} split, {len(data.dividends)} dividend")

    features = {sym: compute_features(data, sym) for sym in data.instruments.index}
    f = features["NORDA"]
    print(f"\nFeatures for NORDA: {f.shape[1]} columns x {len(f):,} one-minute bars "
          f"({time.perf_counter() - t0:.1f}s so far)")

    ann = data.dividends.iloc[0]["announced"]
    window = f.loc[ann - pd.Timedelta("2min"):ann + pd.Timedelta("2min"), SHOWCASE]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 4):
        print(f"\nAround the dividend announcement at {ann:%Y-%m-%d %H:%M} "
              f"(visible from the next bar on):\n{window.to_string()}")

    cutoff = pd.Timestamp(f"{data.days[len(data.days) // 2].date()} 11:37", tz=TZ)
    print(f"\nLookahead check: replace all data from {cutoff:%Y-%m-%d %H:%M} on with unrelated data, "
          "recompute, and compare every feature up to the cutoff")
    ok = True
    for sym in data.instruments.index:
        violations = lookahead_violations(data, sym, cutoff)
        status = "OK, no feature changed" if violations.empty else f"FAILED\n{violations.to_string()}"
        ok &= violations.empty
        print(f"  {sym:6s} {status}")
    print(f"\nDone in {time.perf_counter() - t0:.1f}s")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
