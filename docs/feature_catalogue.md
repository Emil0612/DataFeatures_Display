# Feature catalogue

All 177 features of the original research suite, grouped as they were built, plus the few
columns this package adds. Every feature is computed on one-minute bars; a bar labelled `T`
covers `[T − 1 min, T)`.

**Status**
- **implemented**: produced by `compute_features` under the same name
- **implemented as `x`**: produced under a different name (renamed for clarity, or generic over the benchmark)
- **described**: definition only; not ported (the reason is given)

**Lookahead rules** (how each feature avoids future information)

| Rule | Meaning |
|---|---|
| bar | Uses only trades inside the bar, which are all in the past at `T` |
| trailing N | Rolling window over the last N bars, ending at the current bar |
| prior days | Uses only whole trading days before the current day |
| previous day | A daily value that is only known after the close, so day D carries day D−1's value |
| announced | Uses an event only after its announcement time (strictly before the bar label) |
| next bar | Event timestamps are attached to the first bar labelled strictly after them |
| as of | History is adjusted only for corporate actions that have happened by the as-of time |
| static | A current snapshot, not point-in-time (safe only if the value never changes) |

Inputs: **ticks** = cleaned trade prints, **lit** = lit (on-book) trades only, **index** = benchmark
index minute bars, **CA** = corporate actions, **EOD** = end-of-day data, **ref** = instrument reference data.

## Summary

| Group | Features | Implemented | Described |
|---|---:|---:|---:|
| Raw OHLCV | 17 | 17 | 0 |
| Microstructure | 14 | 12 | 2 |
| Volatility | 9 | 9 | 0 |
| Indices | 20 | 20 | 0 |
| Returns | 7 | 7 | 0 |
| Beta factor | 3 | 3 | 0 |
| Split-adjusted | 10 | 10 | 0 |
| Dividend | 6 | 6 | 0 |
| Volume profile | 9 | 9 | 0 |
| Volume duration | 3 | 3 | 0 |
| Transaction metrics | 11 | 11 | 0 |
| Order flow | 11 | 11 | 0 |
| Lit & dark | 16 | 16 | 0 |
| Broker flow | 23 | 23 | 0 |
| Spread metrics | 6 | 6 | 0 |
| Metadata | 4 | 4 | 0 |
| Venue | 2 | 2 | 0 |
| News | 6 | 6 | 0 |
| **Total** | **177** | **175** | **2** |

Some names appear in more than one group (for example `log_return`, `true_range`, `spread_dir`).
The original suite built them from different sources; the package computes each once.

## Raw OHLCV (17), module `bars`

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `log_return` | ln(close / previous close) | lit | bar | implemented |
| `return_pct` | close / previous close − 1 | lit | bar | implemented |
| `bar_return` | (close − open) / open | lit | bar | implemented |
| `bar_range` | (high − low) / close | lit | bar | implemented |
| `bar_range_abs` | high − low | lit | bar | implemented |
| `realised_vol` | Std of log returns over 30 bars with trades; carried across bars without trades | lit | trailing 30 | implemented |
| `parkinson_vol` | sqrt(mean of ln(H/L)² / (4 ln 2)) over 30 bars (Parkinson 1980) | lit | trailing 30 | implemented |
| `garman_klass_vol` | sqrt(mean of ½ln(H/L)² − (2 ln 2 − 1) ln(C/O)²) over 30 bars (Garman & Klass 1980) | lit | trailing 30 | implemented |
| `realised_vol_ann` | `realised_vol` × √(252 × 510) | lit | trailing 30 | implemented |
| `parkinson_vol_ann` | `parkinson_vol` × √(252 × 510) | lit | trailing 30 | implemented |
| `garman_klass_vol_ann` | `garman_klass_vol` × √(252 × 510) | lit | trailing 30 | implemented |
| `true_range` | max(high, previous close) − min(low, previous close) (Wilder 1978) | lit | bar | implemented |
| `atr` | Mean true range over 30 bars | lit | trailing 30 | implemented |
| `ntr` | True range / previous close | lit | bar | implemented |
| `trade_count` | Number of cleaned trades in the bar | ticks | bar | implemented |
| `avg_trade_size` | Shares per trade (the original also had a currency-per-trade variant) | ticks | bar | implemented |
| `turnover` | Σ price × quantity | lit | bar | implemented |

## Microstructure (14), module `liquidity`

Spreads are inferred from trades because the input has no quotes.

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `log_return` | See Raw OHLCV | lit | bar | implemented |
| `effective_spread_bps` | Bounce spread / VWAP × 10⁴ | lit | bar | implemented |
| `liq_spread_bps` | Effective spread scaled to a target order size with a square-root impact model | minute bars | trailing | described: built on spread and impact columns that the data vendor computes in advance |
| `impact_cost_bps` | Estimated market impact of a reference order, in bps | minute bars | trailing | described: same reason as `liq_spread_bps` |
| `spread_min_bps` | Smallest bounce in the bar / VWAP × 10⁴ | lit | bar | implemented |
| `spread_max_bps` | Largest bounce in the bar / VWAP × 10⁴ | lit | bar | implemented |
| `cost_adj_return` | Log return − half of the latest effective spread (original: liquidity-scaled spread) | lit | bar | implemented |
| `cost_capture_ratio` | Half-spread cost / \|log return\| | lit | bar | implemented |
| `liquidity_regime` | tight / normal / wide vs trailing 33rd/67th spread percentiles, switching only after 3 consistent bars | lit | trailing 30 | implemented |
| `liquidity_regime_encoded` | 0 / 1 / 2 encoding of `liquidity_regime` | lit | trailing 30 | implemented |
| `amihud_zscore` | 30-bar z-score of Amihud illiquidity \|r\| / turnover (Amihud 2002) | lit | trailing 30 | implemented |
| `spread_roll_valid` | 1 when the Roll covariance is negative (spread model applies), 0 when prices trend | lit | bar | implemented |
| `spread_dir` | OLS slope of the bounce spread over the last 30 bars | lit | trailing 30 | implemented |
| `spread_accel` | Change in `spread_dir` | lit | trailing 30 | implemented |

## Volatility (9), module `bars`

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `true_range` | See Raw OHLCV | lit | bar | implemented |
| `ntr` | See Raw OHLCV | lit | bar | implemented |
| `atr` | Wilder average true range, exponential smoothing with α = 1/14 | lit | trailing | implemented as `atr_wilder` |
| `volatility_10` | Std of log returns over 10 bars | lit | trailing 10 | implemented |
| `volatility_60` | Std of log returns over 60 bars | lit | trailing 60 | implemented |
| `volatility_120` | Std of log returns over 120 bars | lit | trailing 120 | implemented |
| `parkinson_vol` | See Raw OHLCV | lit | trailing 30 | implemented |
| `vol10_to_vol60` | `volatility_10` / `volatility_60`, a short-vs-medium horizon regime signal | lit | trailing 60 | implemented |
| `medrv_vol` | sqrt of daily MedRV on 5-minute returns (Andersen, Dobrev & Schaumburg 2012), jump-robust | lit | previous day | implemented |

## Indices (20), module `cross_asset`

OHLC and close-to-close return of four Nasdaq Nordic benchmarks. The package computes the same
five columns for any benchmark passed in; the demo uses a synthetic index, `nordidx`.

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `omxs30_open` | OMX Stockholm 30 open | index | bar | implemented as `<index>_open` |
| `omxs30_high` | OMX Stockholm 30 high | index | bar | implemented as `<index>_high` |
| `omxs30_low` | OMX Stockholm 30 low | index | bar | implemented as `<index>_low` |
| `omxs30_close` | OMX Stockholm 30 close | index | bar | implemented as `<index>_close` |
| `omxs30_return` | OMX Stockholm 30 close-to-close return | index | bar | implemented as `<index>_return` |
| `omxn40_open` | OMX Nordic 40 open | index | bar | implemented as `<index>_open` |
| `omxn40_high` | OMX Nordic 40 high | index | bar | implemented as `<index>_high` |
| `omxn40_low` | OMX Nordic 40 low | index | bar | implemented as `<index>_low` |
| `omxn40_close` | OMX Nordic 40 close | index | bar | implemented as `<index>_close` |
| `omxn40_return` | OMX Nordic 40 close-to-close return | index | bar | implemented as `<index>_return` |
| `omxspi_open` | OMX Stockholm All-Share (price index) open | index | bar | implemented as `<index>_open` |
| `omxspi_high` | OMX Stockholm All-Share (price index) high | index | bar | implemented as `<index>_high` |
| `omxspi_low` | OMX Stockholm All-Share (price index) low | index | bar | implemented as `<index>_low` |
| `omxspi_close` | OMX Stockholm All-Share (price index) close | index | bar | implemented as `<index>_close` |
| `omxspi_return` | OMX Stockholm All-Share (price index) close-to-close return | index | bar | implemented as `<index>_return` |
| `omxsgi_open` | OMX Stockholm All-Share (gross index) open | index | bar | implemented as `<index>_open` |
| `omxsgi_high` | OMX Stockholm All-Share (gross index) high | index | bar | implemented as `<index>_high` |
| `omxsgi_low` | OMX Stockholm All-Share (gross index) low | index | bar | implemented as `<index>_low` |
| `omxsgi_close` | OMX Stockholm All-Share (gross index) close | index | bar | implemented as `<index>_close` |
| `omxsgi_return` | OMX Stockholm All-Share (gross index) close-to-close return | index | bar | implemented as `<index>_return` |

Close-to-close is used rather than (close − open) / open because some index variants are only
recalculated once a minute, which makes the intra-bar return identically zero.

## Returns (7), module `cross_asset`

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `hm_return` | Stock close-to-close return (the original was named after its sample stock) | lit | bar | implemented as `stock_return` |
| `omxs30_return` | Benchmark close-to-close return | index | bar | implemented as `<index>_return` |
| `return_rs` | Stock return − benchmark return | lit, index | bar | implemented |
| `return_rs_ols` | r − β·m with β the 60-bar OLS slope | lit, index | trailing 60 | implemented |
| `return_rs_ols_r2` | r − β·m·ρ², the residual shrunk by how much the benchmark explains | lit, index | trailing 60 | implemented |
| `return_rs_lasso` | r − β_LASSO·m | lit, index | trailing 60 | implemented |
| `return_rs_vwap` | VWAP return − benchmark return | ticks, index | bar | implemented |

## Beta factor (3), module `cross_asset`

The original specification had the first two names swapped; the package uses names that say
what the column holds.

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `beta_factor_corr` | 60-bar OLS slope cov(r, m) / var(m) | lit, index | trailing 60 | implemented as `beta_ols` |
| `beta_factor_ols` | 60-bar Pearson correlation of r and m | lit, index | trailing 60 | implemented as `corr` |
| `beta_factor_lasso` | 60-bar LASSO slope, solved in closed form (soft-thresholded covariance) | lit, index | trailing 60 | implemented as `beta_lasso` |

## Split-adjusted (10), module `corporate_actions`

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `ao` | Split-adjusted open | lit, CA | as of | implemented in `adjusted_ohlcv` |
| `ah` | Split-adjusted high | lit, CA | as of | implemented in `adjusted_ohlcv` |
| `al` | Split-adjusted low | lit, CA | as of | implemented in `adjusted_ohlcv` |
| `ac` | Split-adjusted close | lit, CA | as of | implemented in `adjusted_ohlcv` |
| `av` | Split-adjusted volume | lit, CA | as of | implemented in `adjusted_ohlcv` |
| `stock_split_xday` | Trading days to the next announced split | CA | announced | implemented |
| `stock_split_ratio` | New shares per old share of that split | CA | announced | implemented |
| `stock_split_xday_lookback` | Trading days since the last split (0 on the ex-day) | CA | bar | implemented |
| `stock_split_direction` | forward / reverse / none for the next announced split | CA | announced | implemented |
| `is_split_window` | True within 30 trading days of a split on either side | CA | announced | implemented |

The split countdown needs real announcement timestamps. With ex-dates alone, a bar before the
announcement would already "know" the split, so the original flagged these as unsafe for
backtests. The package takes announcement times as an input and enforces them.

## Dividend (6), module `corporate_actions`

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `dividend_amount` | Cash dividend per share on the ex-date, else 0 | CA | announced | implemented |
| `dividend_pct` | Dividend / previous day's close on the ex-date | CA, lit | announced | implemented |
| `dividend_ex_flag` | 1 on the ex-date | CA | announced | implemented |
| `dividend_ex_date` | Next announced ex-date | CA | announced | implemented |
| `dividend_declaration_flag` | 1 on the announcement day, from the announcement on | CA | announced | implemented |
| `days_to_ex_dividend` | Trading days to the next announced ex-date | CA | announced | implemented |

## Volume profile (9), module `liquidity`

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `vwap` | Σ p·q / Σ q in the bar, carried forward on empty bars | ticks | bar | implemented |
| `vwap_dev` | close − VWAP | ticks | bar | implemented |
| `vwap_dev_bps` | (close − VWAP) / VWAP × 10⁴ | ticks | bar | implemented |
| `vwad` | Volume-weighted std of trade prices around VWAP | ticks | bar | implemented |
| `vwad_bps` | VWAD / VWAP × 10⁴ | ticks | bar | implemented |
| `vwap_continuous` | VWAP of continuous-trading prints only (MMT trading mode) | ticks | bar | implemented |
| `vwad_continuous` | VWAD of continuous-trading prints only | ticks | bar | implemented |
| `intrabar_skew` | Volume-weighted skewness of trade prices (≥ 3 trades) | ticks | bar | implemented |
| `intrabar_kurt` | Volume-weighted excess kurtosis of trade prices (≥ 4 trades) | ticks | bar | implemented |

## Volume duration (3), module `volume_bars`

Volume bars (López de Prado 2018) close each time cumulative turnover reaches a threshold.

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `vol_bar_count_today_so_far` | Volume bars completed since the open | lit | bar | implemented |
| `vol_bar_fill_rate_recent_15min` | Volume bars completed per second over the last 15 minutes | lit | trailing 15 | implemented |
| `vol_bar_partial_fill_progress` | Turnover in the open volume bar / threshold | lit | bar | implemented |

## Transaction metrics (11), module `transactions`

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `trade_count` | See Raw OHLCV | ticks | bar | implemented |
| `avg_trade_size` | See Raw OHLCV | ticks | bar | implemented |
| `block_trade` | Shares traded in block trades (size ≥ 95th percentile of the previous 20 days) | ticks | prior days | implemented |
| `block_count` | Number of block trades | ticks | prior days | implemented |
| `max_trade_size` | Largest trade in the bar | ticks | bar | implemented |
| `trade_size_std` | Std of trade sizes in the bar | ticks | bar | implemented |
| `tps_avg` | Mean trades per second over seconds with trades | ticks | bar | implemented |
| `tps_max` | Most trades in any one second | ticks | bar | implemented |
| `tps_direction` | OLS slope of the 60 per-second trade counts | ticks | bar | implemented |
| `tps_avg_continuous` | `tps_avg` without auction and negotiated prints | ticks | bar | implemented |
| `tps_max_continuous` | `tps_max` without auction and negotiated prints | ticks | bar | implemented |

## Order flow (11), modules `order_flow` and `venue`

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `volume_imbalance` | (buyer- − seller-initiated volume) / signed volume | ticks | bar | implemented |
| `net_volume_delta` | Buyer- − seller-initiated volume | ticks | bar | implemented |
| `vt_divergence` | VWAP − close | ticks | bar | implemented |
| `offbook_share` | Volume share with the off-book tradecode bit | ticks | bar | implemented |
| `dark_share` | Volume share with the dark-pool tradecode bit | ticks | bar | implemented |
| `outside_spread_share` | Volume share executed outside the best bid/offer | ticks | bar | implemented |
| `mmt_special_session_share` | Volume share outside continuous trading (MMT trading mode ≠ continuous) | ticks | bar | implemented |
| `trade_arrival_rate` | Trades per second | ticks | bar | implemented |
| `block_share` | Share of volume in block trades | ticks | prior days | implemented |
| `mean_inter_trade_seconds` | Mean gap between consecutive trades | ticks | bar | implemented |
| `max_inter_trade_seconds` | Longest gap between consecutive trades | ticks | bar | implemented |

The block threshold first used the current day's 95th percentile, which leaked the rest of the
day into every earlier bar. It now uses the previous 20 trading days, and
`tests/test_lookahead.py` shows that the check catches the same-day version.

## Lit & dark (16), module `venue`

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `close_divergence` | All-market close − lit close | ticks | bar | implemented |
| `close_divergence_bps` | The same in bps of the lit close | ticks | bar | implemented |
| `close_divergence_contemporaneous` | As above, without late-published off-book prints | ticks | bar | implemented |
| `close_divergence_contemporaneous_bps` | The same in bps | ticks | bar | implemented |
| `volume_lit_share` | Lit volume / total volume | ticks | bar | implemented |
| `high_divergence` | All-market high − lit high | ticks | bar | implemented |
| `low_divergence` | Lit low − all-market low | ticks | bar | implemented |
| `turnover_lit_share` | Lit turnover / total turnover | ticks | bar | implemented |
| `volume_lit` | Lit volume | ticks | bar | implemented |
| `volume_dark` | Dark-pool volume | ticks | bar | implemented |
| `volume_offbook` | Off-book volume | ticks | bar | implemented |
| `volume_residual` | Volume with no classification bits | ticks | bar | implemented |
| `vshare_lit` | Lit share of the four-way split | ticks | bar | implemented |
| `vshare_dark` | Dark share | ticks | bar | implemented |
| `vshare_offbook` | Off-book share | ticks | bar | implemented |
| `vshare_residual` | Residual share | ticks | bar | implemented |

## Broker flow (23), module `broker_flow`

Needs buyer and seller broker codes, which many venues do not publish; on such instruments
these columns are absent.

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `share_institutional` | (institutional buy + sell volume) / 2 / total | ticks, ref | bar | implemented |
| `share_retail` | The same for retail brokers | ticks, ref | bar | implemented |
| `share_market_maker` | The same for market makers | ticks, ref | bar | implemented |
| `share_unknown` | The same for brokers not in the directory | ticks, ref | bar | implemented |
| `broker_ratio` | (institutional − retail volume) / total | ticks, ref | bar | implemented |
| `inst_net_flow` | Institutional buy − sell volume | ticks, ref | bar | implemented |
| `retail_net_flow` | Retail buy − sell volume | ticks, ref | bar | implemented |
| `signed_vol_institutional` | Aggressor-signed volume where the initiator is institutional | ticks, ref | bar | implemented |
| `signed_vol_retail` | The same for retail | ticks, ref | bar | implemented |
| `signed_vol_market_maker` | The same for market makers | ticks, ref | bar | implemented |
| `signed_vol_unknown` | The same for unknown brokers | ticks, ref | bar | implemented |
| `inst_signed_share` | `signed_vol_institutional` / total, clipped to [−1, 1] | ticks, ref | bar | implemented |
| `top_buyer_share` | Largest single buyer's share of buy volume | ticks | bar | implemented |
| `top_seller_share` | Largest single seller's share of sell volume | ticks | bar | implemented |
| `cross_trade_ratio` | Volume where buyer = seller / total | ticks | bar | implemented |
| `broker_skew` | Distinct buyers − distinct sellers | ticks | bar | implemented |
| `broker_hhi_buy` | Herfindahl index of buy volume across brokers | ticks | bar | implemented |
| `broker_hhi_sell` | Herfindahl index of sell volume across brokers | ticks | bar | implemented |
| `n_unique_buyers` | Distinct buying brokers | ticks | bar | implemented |
| `n_unique_sellers` | Distinct selling brokers | ticks | bar | implemented |
| `in_event_window` | 1 in the 5 bars starting with the bar where company news became visible | news | next bar | implemented |
| `news_flow_imb_event` | `broker_ratio` inside event windows | ticks, news | next bar | implemented |
| `news_flow_inst_share_event_delta` | Institutional share − its trailing 60-bar median outside event windows | ticks, news | trailing 60 | implemented |

The original event window covered five minutes on both sides of each news item, and the baseline
was the median over the whole day. Both used future information. The package flags only bars
after the news and uses a trailing baseline.

## Spread metrics (6), module `liquidity`

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `spread_bounce` | Mean \|Δprice\| when the aggressor side flips (each flip crosses the spread) | lit | bar | implemented |
| `spread_min` | Smallest such bounce in the bar | lit | bar | implemented |
| `spread_max` | Largest such bounce in the bar | lit | bar | implemented |
| `spread_roll` | Roll (1984): 2√(−cov(Δp_t, Δp_t−1)) within the bar, 0 if the covariance is not negative | lit | bar | implemented |
| `spread_dir` | See Microstructure | lit | trailing 30 | implemented |
| `spread_accel` | See Microstructure | lit | trailing 30 | implemented |

## Metadata (4), module `metadata`

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `timestamp` | Bar label (end of the bar), Europe/Stockholm | — | bar | implemented |
| `insref` | Instrument identifier | ref | static | implemented as `symbol` |
| `mcap` | Market capitalisation at the previous day's close | EOD | previous day | implemented |
| `country` | Country of the issuer | ref | static | implemented |

## Venue (2), module `metadata`

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `n_listings` | Number of venues the instrument is listed on | ref | static | implemented |
| `days_since_listing` | Trading days since the first listing | ref | bar | implemented |

`n_listings` comes from a single current snapshot. It is not point-in-time and should only be
used where listings rarely change.

## News (6), module `news`

| Feature | Definition | Inputs | Lookahead | Status |
|---|---|---|---|---|
| `news_corp_cat` | Category of the latest company news (EARNINGS, GUIDANCE, M_AND_A, …) | news | next bar | implemented |
| `news_macro_cat` | Category of the latest macro news (INFLATION, EMPLOYMENT, CENTRAL_BANK, …) | news | next bar | implemented |
| `news_corp_density` | Company news items in the last 60 bars | news | next bar | implemented |
| `news_density_sector` | Sector news items in the last 60 bars | news | next bar | implemented |
| `news_density_ratio` | Company / sector density | news | next bar | implemented |
| `news_density_macro` | Macro news items in the last 60 bars | news | next bar | implemented |

## Added in the package

Columns that `compute_features` returns beyond the 177.

| Column | Definition | Lookahead |
|---|---|---|
| `open`, `high`, `low`, `close`, `volume` | Lit price bar | bar |
| `amihud` | \|log return\| / turnover, the input to `amihud_zscore` | bar |
| `split_adjusted_return` | Log return with the split factor removed on the ex-day | bar |

## References

- Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31–56.
- Andersen, T. G., Dobrev, D., Schaumburg, E. (2012). Jump-robust volatility estimation using nearest neighbor truncation. *Journal of Econometrics*, 169(1), 75–93.
- Garman, M. B., Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*, 53(1), 67–78.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 53(1), 61–65.
- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *Journal of Finance*, 39(4), 1127–1139.
- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
- FIX Trading Community. Market Model Typology (MMT). https://www.fixtrading.org/standards/mmt/
