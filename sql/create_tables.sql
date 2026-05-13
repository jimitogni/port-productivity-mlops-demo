CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    execution_date DATE NOT NULL,
    forecast_date DATE NOT NULL,
    forecast_horizon TEXT NOT NULL,
    terminal_id TEXT NOT NULL,
    predicted_productivity_tons_hour DOUBLE PRECISION NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_metadata (
    run_id TEXT PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    execution_date DATE NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    status TEXT NOT NULL,
    input_rows INTEGER NOT NULL,
    prediction_rows INTEGER NOT NULL,
    model_name TEXT,
    model_version TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS model_performance (
    id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    run_id TEXT,
    evaluation_date DATE NOT NULL,
    terminal_id TEXT,
    forecast_horizon TEXT,
    mae DOUBLE PRECISION,
    rmse DOUBLE PRECISION,
    r2 DOUBLE PRECISION,
    mape DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS monitoring_metrics (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    execution_date DATE NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

