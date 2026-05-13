SHELL := /bin/bash
PYTHON ?= $(shell if [ -x .venv/bin/python ] && .venv/bin/python -m pip --version >/dev/null 2>&1; then echo .venv/bin/python; else echo python3; fi)
BOOTSTRAP_PYTHON ?= python3.12
TODAY ?= 2026-05-12
IMAGE ?= port-productivity-mlops-demo:local
TEST_IMAGE ?= port-productivity-mlops-demo:test
COMPOSE := docker compose --env-file .env -f docker-compose.demo.yml
DOCKER_RUN_ENV := --env-file .env -e PROJECT_ROOT=/app -e DATA_DIR=/app/data -e REPORTS_DIR=/app/reports -e MODELS_DIR=/app/models -e PROMETHEUS_METRICS_PATH=/app/data/monitoring/port_productivity_metrics.prom -e MLFLOW_TRACKING_URI=$${DOCKER_MLFLOW_TRACKING_URI:-http://mlflow_server:5000}
DOCKER_RUN_VOLUMES := -v "$$(pwd)/data:/app/data" -v "$$(pwd)/reports:/app/reports" -v "$$(pwd)/models:/app/models"
DOCKER_RUN_NET := --network $${DOCKER_MLFLOW_NETWORK:-mlflow_default}

.PHONY: setup doctor venv install check-python-deps generate-data train predict validate-dags test compile monitoring-report promote-model run-api docker-build docker-run-api docker-test docker-generate-data docker-train docker-promote-model docker-predict docker-demo-normal compose-up compose-down compose-logs k8s-apply k8s-delete k8s-status k8s-logs demo-normal demo-missing-terminal demo-drift demo-prediction-drift demo-train-challenger demo-promote-model demo-failed-validation demo-recovery clean

setup:
	@if [ ! -f .env ]; then cp .env.example .env; fi
	mkdir -p data/raw data/processed data/predictions data/monitoring reports/evidently models

doctor:
	@echo "Using PYTHON=$(PYTHON)"
	@$(PYTHON) --version || true
	@$(PYTHON) -c "import pandas, numpy, sklearn, pytest, mlflow, evidently, fastapi, joblib; print('Python dependencies: ok')" 2>/dev/null || \
		(echo "Python dependencies are missing. Run 'make install' with Python 3.12 available, or use Docker targets such as 'make docker-demo-normal'."; exit 1)

venv:
	@command -v $(BOOTSTRAP_PYTHON) >/dev/null 2>&1 || \
		(echo "$(BOOTSTRAP_PYTHON) is required for local setup. Install Python 3.12 with venv support, or run 'make docker-demo-normal'."; exit 1)
	$(BOOTSTRAP_PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip

install: venv
	.venv/bin/python -m pip install -r requirements.txt

check-python-deps:
	@$(PYTHON) -c "import pandas, numpy, sklearn, pytest, mlflow, evidently, fastapi, joblib" 2>/dev/null || \
		(echo "Missing Python dependencies for $(PYTHON). Run 'make install' first, or use 'make docker-demo-normal'."; exit 1)

generate-data: setup
	@if $(PYTHON) -c "import pandas, numpy" >/dev/null 2>&1; then \
		$(PYTHON) -m src.data.generate_synthetic_data --start-date 2024-01-01 --end-date $(TODAY) --output data/raw/port_productivity.csv; \
	else \
		echo "Local Python dependencies are missing; running generate-data in Docker."; \
		$(MAKE) docker-generate-data; \
	fi

train: setup
	@if $(PYTHON) -c "import pandas, numpy, sklearn, mlflow, joblib" >/dev/null 2>&1; then \
		$(PYTHON) -m src.pipelines.training_pipeline --data-path data/raw/port_productivity.csv; \
	else \
		echo "Local Python dependencies are missing; running train in Docker."; \
		$(MAKE) docker-train; \
	fi

predict: setup
	@if $(PYTHON) -c "import pandas, numpy, sklearn, mlflow, joblib" >/dev/null 2>&1; then \
		$(PYTHON) -m src.pipelines.daily_prediction_pipeline --execution-date $(TODAY); \
	else \
		echo "Local Python dependencies are missing; running predict in Docker."; \
		$(MAKE) docker-predict; \
	fi

validate-dags:
	$(PYTHON) scripts/validate_airflow_dags.py

test:
	@if $(PYTHON) -c "import pandas, numpy, sklearn, pytest, mlflow, evidently, fastapi, joblib" >/dev/null 2>&1; then \
		$(PYTHON) -m pytest -q; \
	else \
		echo "Local Python dependencies are missing; running tests in Docker."; \
		$(MAKE) docker-test; \
	fi

compile:
	$(PYTHON) -m compileall src dags scripts tests

monitoring-report:
	$(PYTHON) -m src.pipelines.daily_prediction_pipeline --execution-date $(TODAY)

promote-model:
	@if $(PYTHON) -c "import pandas, numpy, sklearn, mlflow, joblib" >/dev/null 2>&1; then \
		$(PYTHON) -m src.models.promote_model; \
	else \
		echo "Local Python dependencies are missing; running promote-model in Docker."; \
		$(MAKE) docker-promote-model; \
	fi

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port $${API_PORT:-8015} --reload

docker-build:
	docker build -f docker/Dockerfile -t $(IMAGE) .

docker-run-api:
	docker run --rm $(DOCKER_RUN_ENV) $(DOCKER_RUN_NET) -p 8015:8000 $(DOCKER_RUN_VOLUMES) $(IMAGE)

docker-test:
	docker build -f docker/Dockerfile -t $(TEST_IMAGE) .
	docker run --rm $(TEST_IMAGE) python -m pytest -q

docker-generate-data: setup
	docker build -f docker/Dockerfile -t $(TEST_IMAGE) .
	docker run --rm $(DOCKER_RUN_ENV) $(DOCKER_RUN_VOLUMES) $(TEST_IMAGE) python -m src.data.generate_synthetic_data --start-date 2024-01-01 --end-date $(TODAY) --output /app/data/raw/port_productivity.csv

docker-train: setup
	docker build -f docker/Dockerfile -t $(TEST_IMAGE) .
	docker run --rm $(DOCKER_RUN_ENV) $(DOCKER_RUN_NET) $(DOCKER_RUN_VOLUMES) $(TEST_IMAGE) python -m src.pipelines.training_pipeline --data-path /app/data/raw/port_productivity.csv

docker-promote-model: setup
	docker build -f docker/Dockerfile -t $(TEST_IMAGE) .
	docker run --rm $(DOCKER_RUN_ENV) $(DOCKER_RUN_NET) $(DOCKER_RUN_VOLUMES) $(TEST_IMAGE) python -m src.models.promote_model

docker-predict: setup
	docker build -f docker/Dockerfile -t $(TEST_IMAGE) .
	docker run --rm $(DOCKER_RUN_ENV) $(DOCKER_RUN_NET) $(DOCKER_RUN_VOLUMES) $(TEST_IMAGE) python -m src.pipelines.daily_prediction_pipeline --execution-date $(TODAY)

docker-demo-normal: docker-test docker-generate-data docker-train docker-promote-model docker-predict

compose-up: setup
	$(COMPOSE) up -d --build

compose-down:
	$(COMPOSE) down

compose-logs:
	$(COMPOSE) logs -f --tail=200

k8s-apply:
	kubectl apply -f k8s/

k8s-delete:
	kubectl delete -f k8s/ --ignore-not-found=true

k8s-status:
	kubectl -n port-productivity-mlops get all

k8s-logs:
	kubectl -n port-productivity-mlops logs deployment/port-productivity-api --tail=200

demo-normal: generate-data train promote-model predict

demo-missing-terminal: generate-data train promote-model
	-$(PYTHON) -m src.pipelines.daily_prediction_pipeline --execution-date $(TODAY) --scenario missing_terminal

demo-drift: generate-data train promote-model
	$(PYTHON) -m src.pipelines.daily_prediction_pipeline --execution-date $(TODAY) --scenario drift

demo-prediction-drift: generate-data train promote-model
	$(PYTHON) -m src.pipelines.daily_prediction_pipeline --execution-date $(TODAY) --scenario prediction_drift

demo-train-challenger: setup
	$(PYTHON) -m src.data.generate_synthetic_data --start-date 2024-01-01 --end-date $(TODAY) --scenario drift --drift-start-date 2026-01-01 --output data/raw/port_productivity.csv
	$(PYTHON) -m src.pipelines.training_pipeline --data-path data/raw/port_productivity.csv

demo-promote-model: promote-model

demo-failed-validation: demo-missing-terminal

demo-recovery: demo-missing-terminal predict

clean:
	rm -rf data/raw/*.csv data/processed/*.csv data/predictions/*.csv data/monitoring/*.csv data/monitoring/*.prom reports/evidently/*.html models/*
	touch models/.gitkeep
