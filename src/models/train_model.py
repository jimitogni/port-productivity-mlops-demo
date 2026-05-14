from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score

from src.utils.logging import get_logger


LOGGER = get_logger(__name__)

_RF_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 12, 18],
    "max_features": ["sqrt", 0.5],
    "min_samples_leaf": [1, 3],
}


def train_baseline_model(X, y) -> DummyRegressor:
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)
    return model


def train_candidate_model(
    X, y, random_state: int = 42
) -> tuple[dict[str, object], str, dict]:
    """Train RandomForest (grid search) and XGBoost.

    Returns:
        (candidates, best_type, best_params) where `candidates` maps each
        trained model type to its fitted estimator, and `best_type` is the
        one with the lowest cross-validated MAE.
    """
    tscv = TimeSeriesSplit(n_splits=5)

    # Random Forest — grid search over compact parameter grid
    LOGGER.info("Starting RandomForest grid search (%d combinations × 5 folds)", _grid_size())
    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=random_state, n_jobs=-1),
        _RF_PARAM_GRID,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        refit=True,
        n_jobs=-1,
    )
    rf_search.fit(X, y)
    rf_cv_mae = -rf_search.best_score_
    LOGGER.info("RandomForest best CV MAE: %.4f | params: %s", rf_cv_mae, rf_search.best_params_)

    # XGBoost — fixed good defaults, CV score for fair comparison
    xgb_model, xgb_cv_mae = _train_xgboost(X, y, tscv, random_state)

    best_params = {
        "rf_cv_mae": round(rf_cv_mae, 4),
        "xgb_cv_mae": round(xgb_cv_mae, 4) if xgb_cv_mae != float("inf") else None,
        **{f"rf_{k}": str(v) for k, v in rf_search.best_params_.items()},
    }

    candidates: dict[str, object] = {"RandomForestRegressor": rf_search.best_estimator_}
    if xgb_model is not None:
        candidates["XGBRegressor"] = xgb_model

    if xgb_model is not None and xgb_cv_mae < rf_cv_mae:
        LOGGER.info("XGBoost wins (CV MAE %.4f < RF %.4f)", xgb_cv_mae, rf_cv_mae)
        return candidates, "XGBRegressor", best_params

    LOGGER.info("RandomForest wins (CV MAE %.4f)", rf_cv_mae)
    return candidates, "RandomForestRegressor", best_params


def _train_xgboost(X, y, tscv, random_state: int):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        LOGGER.warning("xgboost not installed; skipping XGBoost candidate")
        return None, float("inf")

    xgb = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    scores = cross_val_score(xgb, X, y, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_mae = float(-scores.mean())
    LOGGER.info("XGBoost CV MAE: %.4f (±%.4f)", cv_mae, scores.std())
    xgb.fit(X, y)
    return xgb, cv_mae


def _grid_size() -> int:
    total = 1
    for v in _RF_PARAM_GRID.values():
        total *= len(v)
    return total
