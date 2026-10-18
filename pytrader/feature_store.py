"""FeatureStore: compute features for a ticker's price series, cache them,
and serve point-in-time queries.

The cache is the `FeatureStorage` (DuckDB implements it). Computing is
done once per (ticker, spec_hash); afterwards every query is a pure DB
read — no recompute, no network. This is what makes time-travel cheap:
`store.query(ticker, date, spec)` reuses the same cached panel that a run
produced, so the model sees exactly what did on the original date.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from pytrader.data import DailyPrice
from pytrader.features import FeatureSpec

if TYPE_CHECKING:
    from pytrader.storage import FeatureStorage


class FeatureStore:
    def __init__(self, storage: "FeatureStorage") -> None:
        self._storage = storage

    def _hash(self, spec: FeatureSpec) -> str:
        return spec.spec_hash()

    def has(self, ticker: str, spec: FeatureSpec) -> bool:
        return self._storage.spec_features(self._hash(spec)) != []

    def build(self, series: DailyPrice, spec: FeatureSpec) -> pd.DataFrame:
        """Compute features for one ticker and persist them.

        Returns the long panel (date, feature, value) for downstream
        training, or the wide form if the caller prefers.
        """
        h = self._hash(spec)
        # Compute the wide panel (index = dates, cols = feature names).
        wide = spec.compute(series.close).dropna(how="all")
        if len(wide) == 0:
            return pd.DataFrame(columns=["date"] + spec.column_names())
        panel = wide.reset_index()
        if "date" not in panel.columns:
            panel = panel.rename(columns={panel.columns[0]: "date"})
        panel["date"] = pd.to_datetime(panel["date"])
        self._storage.upsert_panel(series.ticker.upper(), h, panel)
        return panel

    def query(self, ticker: str, as_of: pd.Timestamp, spec: FeatureSpec) -> dict[str, float] | None:
        """Point-in-time feature vector for (ticker, date). None if absent."""
        return self._storage.query(ticker.upper(), as_of, self._hash(spec))

    def dates(self, ticker: str, spec: FeatureSpec) -> list[pd.Timestamp]:
        return self._storage.dates(ticker.upper(), self._hash(spec))

    def features(self, spec: FeatureSpec) -> list[str]:
        return self._storage.spec_features(self._hash(spec))
