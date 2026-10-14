"""Four assets: synthetic market -> benchmark index features -> per-instrument features -> lookahead report.

In production the first asset would read from the market-data warehouse instead of simulating;
everything downstream stays the same because it only sees the normalised tables.
"""

import pandas as pd
from dagster import AssetExecutionContext, AssetSelection, Config, Definitions, MetadataValue, asset, define_asset_job

from nordic_equity_features import MarketData, compute_features, generate
from nordic_equity_features.cross_asset import index_features
from nordic_equity_features.lookahead import lookahead_violations
from nordic_equity_features.schema import TZ


class MarketConfig(Config):
    n_days: int = 10
    seed: int = 7


@asset
def market_data(context: AssetExecutionContext, config: MarketConfig) -> MarketData:
    """Normalised trades, index bars, corporate actions, news and end-of-day data."""
    data = generate(n_days=config.n_days, seed=config.seed)
    context.add_output_metadata(
        {
            "trading_days": len(data.days),
            "trades": len(data.trades),
            "instruments": MetadataValue.json(list(data.instruments.index)),
        }
    )
    return data


@asset
def benchmark_index_features(context: AssetExecutionContext, market_data: MarketData) -> pd.DataFrame:
    """OHLC and close-to-close return of the benchmark index."""
    df = index_features(market_data.index_bars, market_data.index_name)
    context.add_output_metadata({"bars": len(df), "columns": MetadataValue.json(list(df.columns))})
    return df


@asset
def instrument_features(
    context: AssetExecutionContext, market_data: MarketData, benchmark_index_features: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """The full feature table for every instrument."""
    _ = benchmark_index_features  # dependency edge: instrument features embed the index columns
    out = {sym: compute_features(market_data, sym) for sym in market_data.instruments.index}
    context.add_output_metadata({sym: f"{f.shape[1]} features x {len(f)} bars" for sym, f in out.items()})
    return out


@asset
def lookahead_report(
    context: AssetExecutionContext, market_data: MarketData, instrument_features: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Future-invariance check at a mid-session cutoff; fails the run if any feature leaks."""
    cutoff = pd.Timestamp(f"{market_data.days[len(market_data.days) // 2].date()} 11:37", tz=TZ)
    rows = []
    for sym in instrument_features:
        v = lookahead_violations(market_data, sym, cutoff)
        rows.append({"symbol": sym, "cutoff": str(cutoff), "violations": len(v)})
    report = pd.DataFrame(rows)
    context.add_output_metadata({"report": MetadataValue.json(report.to_dict(orient="records"))})
    if report["violations"].sum():
        raise RuntimeError(f"lookahead detected:\n{report}")
    return report


feature_pipeline = define_asset_job("feature_pipeline", selection=AssetSelection.all())

defs = Definitions(
    assets=[market_data, benchmark_index_features, instrument_features, lookahead_report],
    jobs=[feature_pipeline],
)
