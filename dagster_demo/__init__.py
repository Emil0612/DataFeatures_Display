"""Dagster prototype: the feature pipeline as assets, running on synthetic data.

    uv run --extra dagster dagster job execute -m dagster_demo -j feature_pipeline
"""

from .definitions import defs

__all__ = ["defs"]
