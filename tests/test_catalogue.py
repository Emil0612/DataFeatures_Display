"""The feature catalogue and the code must agree."""

import re
from pathlib import Path

from nordic_equity_features.corporate_actions import adjusted_ohlcv

CATALOGUE = Path(__file__).resolve().parents[1] / "docs" / "feature_catalogue.md"
ROW = re.compile(r"^\| `([^`]+)` \|.*\| ([^|]+) \|\s*$")


def _sections():
    """{section title: [(feature, status), ...]} for the feature tables."""
    sections, current = {}, None
    for line in CATALOGUE.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current and (m := ROW.match(line)):
            sections[current].append((m.group(1), m.group(2).strip()))
    return {k: v for k, v in sections.items() if v and re.search(r"\(\d+\)", k)}


def _produced_name(feature, status, index_name):
    if status == "implemented":
        return feature
    if m := re.match(r"implemented as `([^`]+)`", status):
        return m.group(1).replace("<index>", index_name)
    if status == "implemented in `adjusted_ohlcv`":
        return f"adjusted_ohlcv:{feature}"
    assert status.startswith("described"), f"unknown status for {feature}: {status}"
    return None


def test_catalogue_names_all_177_features():
    sections = _sections()
    assert len(sections) == 18
    for title, rows in sections.items():
        assert int(re.search(r"\((\d+)\)", title).group(1)) == len(rows), title
    assert sum(len(rows) for rows in sections.values()) == 177


def test_every_implemented_feature_is_produced(data, features):
    produced = set().union(*(f.columns for f in features.values()))
    adjusted = set(adjusted_ohlcv(features["FJORD"][["open", "high", "low", "close", "volume"]],
                                  data.splits, features["FJORD"].index[-1]).columns)
    for rows in _sections().values():
        for feature, status in rows:
            name = _produced_name(feature, status, data.index_name)
            if name is None:
                continue
            if name.startswith("adjusted_ohlcv:"):
                assert name.split(":")[1] in adjusted
            else:
                assert name in produced, f"{feature} -> {name} is not produced"


def test_every_produced_column_is_catalogued(data, features):
    text = CATALOGUE.read_text(encoding="utf-8")
    named = set(re.findall(r"`([a-z0-9_<>]+)`", text))
    named = {n.replace("<index>", data.index_name) for n in named}
    produced = set().union(*(f.columns for f in features.values()))
    missing = sorted(produced - named)
    assert not missing, f"columns missing from the catalogue: {missing}"


def test_summary_table_matches_sections():
    sections = _sections()
    implemented = sum(not s.startswith("described") for rows in sections.values() for _, s in rows)
    described = sum(s.startswith("described") for rows in sections.values() for _, s in rows)
    text = CATALOGUE.read_text(encoding="utf-8")
    assert f"| **Total** | **177** | **{implemented}** | **{described}** |" in text
