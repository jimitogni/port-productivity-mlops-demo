from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor


def train_baseline_model(X, y) -> DummyRegressor:
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)
    return model


def train_candidate_model(X, y, random_state: int = 42) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=180,
        max_depth=14,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model

