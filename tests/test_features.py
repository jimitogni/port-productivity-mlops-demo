from __future__ import annotations

from src.data.generate_synthetic_data import generate_daily_operational_forecast, generate_synthetic_data
from src.features.build_features import TARGET_COLUMN, build_inference_features, build_training_dataset, expected_feature_columns


def test_training_features_are_numeric_and_include_target():
    df = generate_synthetic_data("2024-01-01", "2024-02-01", seed=10)
    features = build_training_dataset(df)
    feature_columns = expected_feature_columns()
    assert TARGET_COLUMN in features.columns
    assert set(feature_columns).issubset(features.columns)
    assert features[feature_columns].isna().sum().sum() == 0
    assert len(features) == len(df) * 3


def test_inference_features_match_training_feature_columns():
    history = generate_synthetic_data("2024-01-01", "2024-03-01", seed=11)
    current = generate_daily_operational_forecast("2024-03-02", seed=12)
    features = build_inference_features(current, history)
    assert list(features[expected_feature_columns()].columns) == expected_feature_columns()
    assert features[expected_feature_columns()].isna().sum().sum() == 0

