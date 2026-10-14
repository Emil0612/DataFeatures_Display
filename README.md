# nordic-equity-features

Lookahead-safe intraday features for Nordic equities, built from tick-level trades.

> **Portfolio version.** This repository is a cleaned-up rebuild of feature-engineering work done
> during an internship. The licensed market data, internal documents and the original research
> notebooks have been removed. Everything here runs on **synthetic data**: fictional instruments,
> simulated ticks, dates in 2030.

## The problem

A trading model is only as honest as its inputs. Intraday features are easy to get subtly wrong:
a volume threshold computed over the whole day, a market cap that includes today's close, a news
item attached to the minute it was published rather than the minute after, a dividend that
"exists" before it was announced. Each of these lets the model see a little of the future, and
the backtest looks better than anything that could trade live.

This project turns raw trade prints into about 150 one-minute features across seven families,
and proves with tests that none of them uses information from after the bar it is stamped on.

## Pipeline

```
trade prints ──► clean ──► one-minute bars ──► feature families ──► one row per bar
(tradecode,        drop cancelled,   bar T = [T−1 min, T)    bars & volatility
 MMT flags,        corrections        empty minutes carry     transactions & order flow
 broker codes)                        the last close          liquidity & venue split
                                                              broker flow
benchmark index ─────────────────────────────────────────►    cross-asset (beta, residuals)
corporate actions, news, end-of-day ──► aligned to the    ►   corporate actions, news, metadata
                                        bar clock
```

| Family | What it measures | Examples |
|---|---|---|
| Bars & volatility | Returns, ranges, realised and range-based volatility | `log_return`, `true_range`, `parkinson_vol`, `medrv_vol` |
| Transactions & order flow | Trade sizes, block trades, velocity, aggressor imbalance | `block_share`, `tps_max`, `volume_imbalance` |
| Liquidity & venue | Spreads inferred from trades, price distribution in the bar, lit/dark/off-book split | `spread_roll`, `vwad`, `amihud_zscore`, `vshare_dark` |
| Broker flow | Who traded: institutional vs retail share, concentration | `broker_ratio`, `broker_hhi_buy` |
| Cross-asset | Benchmark returns, rolling beta, beta-adjusted residual returns | `beta_ols`, `beta_lasso`, `return_rs_ols` |
| Corporate actions | Split adjustment, split and dividend countdowns | `days_to_ex_dividend`, `stock_split_xday` |
| Metadata & news | Market cap, listing age, news categories and density | `mcap`, `news_corp_density` |

The full list (all 177 features of the original suite, with definitions, inputs and the
lookahead rule for each) is in **[docs/feature_catalogue.md](docs/feature_catalogue.md)**.

## How lookahead is prevented

| Situation | Rule |
|---|---|
| Bars | A bar labelled `T` contains trades in `[T − 1 min, T)`, so it is complete at `T` |
| Rolling statistics | Trailing windows that end at the current bar |
| Thresholds ("a block is a top-5% trade") | Estimated on the previous 20 trading days, never the current day |
| End-of-day values (market cap) | Day D uses day D−1, because D's close is unknown during D |
| Daily estimators (jump-robust MedRV volatility) | Shifted one day for the same reason |
| News | Attached to the first bar labelled strictly after the timestamp; pre-market news lands at 09:01 |
| Splits and dividends | Visible only after the announcement time, to the minute |
| Split-adjusted history | Built *as of* a moment, applying only splits that have happened by then |

The rules are enforced by a **future-invariance test**: keep everything that was knowable before
a cutoff, replace everything after it with an unrelated synthetic market, plant a split, a
dividend and a news item exactly at the cutoff, recompute, and require every feature at every bar
up to the cutoff to be unchanged. It runs for every feature and instrument at five boundary
cutoffs (mid-session, first bar, the minute of an announcement, before the open, last bar). A
deliberately leaky feature (a same-day block threshold) is included to show that the test
catches it.

## Run it

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras                        # install
uv run python -m nordic_equity_features.demo   # generate a market, compute features, run the lookahead check
uv run pytest                               # 36 tests, about 15 seconds
```

The demo prints the synthetic market, a slice of the feature table around a dividend
announcement (the countdown appears one bar after 12:15), and the lookahead check for each
instrument.

Notebooks (committed without outputs; run them top to bottom):

- `notebooks/01_walkthrough.ipynb`: from trade prints to features, with charts
- `notebooks/02_lookahead_safety.ipynb`: the timing rules, the general check, and a leak being caught

Dagster prototype: the same pipeline as four assets (market → benchmark index → features →
lookahead report). The last asset fails the run if any feature leaks.

```bash
uv run dagster job execute -m dagster_demo -j feature_pipeline
```

## Using it on real data

`compute_features(data, symbol)` takes a `MarketData` object: normalised trades, benchmark
index bars, corporate actions, news and end-of-day values (the columns are listed in
`src/nordic_equity_features/schema.py` and `synthetic.py`). The synthetic generator produces
exactly these tables. For real data, a loader maps the vendor's fields onto them; nothing
downstream changes.

The original work used tick data from **Millistream** (trades with tradecode bitfields, MMT
flags and broker codes), fundamentals from **Börsdata** and macro news from **Trading
Economics**, stored in DuckDB / MotherDuck.

### Data-quality method

Before trusting a field, the original work checked, per instrument and venue:

- how much of each field is actually populated (aggressor side, broker codes, venue, MMT)
- that tradecode and MMT decoding matched the vendor's documentation
- tick-level venue volumes reconciled against the vendor's end-of-day totals
- computed split adjustments against the vendor's own adjusted price series
- that every event table carried timestamps fine enough for point-in-time use

The specific findings describe a licensed product and are not included here.

## Repository layout

```
src/nordic_equity_features/
  schema.py              trade-record columns, tradecode bits, MMT decoding, bar clock
  bars.py                cleaning, one-minute bars, returns and volatility
  transactions.py        trade sizes, block trades (prior-window threshold), velocity
  order_flow.py          aggressor imbalance, special-session share, trade timing
  liquidity.py           VWAP/VWAD and moments, spreads from trades, Amihud, liquidity regime
  venue.py               lit / dark / off-book / residual split, lit-vs-all divergence
  broker_flow.py         broker-type shares, concentration, flow after news
  volume_bars.py         volume-bar progress on the clock grid
  cross_asset.py         index features, rolling OLS / LASSO beta, residual returns
  corporate_actions.py   split adjustment as of a time, split and dividend features
  metadata.py            market cap (previous day), listing age, country
  news.py                news categories and trailing density
  pipeline.py            compute_features: all families for one instrument
  lookahead.py           the future-invariance check
  synthetic.py           synthetic market generator
  demo.py                end-to-end demo
tests/                   lookahead, mechanisms, estimator correctness, catalogue consistency
notebooks/               two walkthroughs on synthetic data
dagster_demo/            Dagster assets and job
docs/feature_catalogue.md
```

## Limitations

- The synthetic market reproduces the *structure* the features rely on (auctions, venue flags,
  late prints, bid-ask bounce, events), not real market statistics. Feature values in the
  notebooks illustrate the mechanics and say nothing about real instruments.
- Features are engineered and tested for correctness and point-in-time safety, not for
  predictive power; selecting and validating them across instruments is the next stage.
- Quotes are not an input, so spreads are inferred from trades (bounce and Roll estimators).
- `n_listings` and `country` come from a current snapshot and are not point-in-time.
