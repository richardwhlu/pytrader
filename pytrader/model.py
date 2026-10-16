"""Model layer: thin XGBoost wrapper + artifact persistence.

The `Model` protocol is intentionally tiny so an alternative learner
(lightgbm, a stack, etc.) can slot in later without touching backtest/UI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Model": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def feature_names(self) -> list[str]: ...


class XGBoostModel:
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        early_stopping_rounds: int | None = 50,
    ) -> None:
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
            tree_method="hist",
        )
        self.early_stopping_rounds = early_stopping_rounds
        self._booster = None
        self._features: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "XGBoostModel":
        import xgboost as xgb
        self._features = list(X.columns)
        kwargs = dict(self.params, verbosity=0)
        eval_set = None
        if self.early_stopping_rounds is not None:
            kwargs["early_stopping_rounds"] = self.early_stopping_rounds
            eval_set = [(X, y)]
        self._booster = xgb.XGBRegressor(objective="reg:squarederror", **kwargs).fit(
            X, y, eval_set=eval_set
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("model not fit")
        return self._booster.predict(X)

    def feature_importance(self) -> list[tuple[str, float]]:
        if self._booster is None:
            return []
        imp = self._booster.feature_importances_
        return sorted(zip(self._features, imp), key=lambda p: -p[1])

    # ---- persistence -------------------------------------------------------

    def save(self, directory: Path, meta: dict | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / "model.json"
        self._booster.save_model(str(p))
        m = directory / "meta.json"
        m.write_text(json.dumps({
            "features": self._features,
            "params": self.params,
            **({"meta": meta} if meta else {}),
        }, indent=2))
        return p

    @classmethod
    def load(cls, directory: Path) -> "XGBoostModel":
        import json as _json
        import xgboost as xgb
        directory = Path(directory)
        meta = _json.loads((directory / "meta.json").read_text())
        m = cls()
        m._features = meta["features"]
        m._booster = xgb.XGBRegressor()
        m._booster.load_model(str(directory / "model.json"))
        m.meta = meta.get("meta", {})
        return m
