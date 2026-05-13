# Kubeflow Pipelines Extension

Kubeflow Pipelines is optional and represents a future Kubernetes-native evolution.

Airflow is better for the first version because:

- The business process is a scheduled daily batch forecast.
- The team needs clear operational DAGs, retries, logs, and simple reruns.
- Existing homelab infrastructure already has Airflow and MLflow running.

Kubeflow becomes more attractive when:

- The company wants a standardized Kubernetes ML platform.
- Pipeline steps need container-native isolation and artifact lineage.
- The ML platform team owns model training workflows across many projects.

Kubeflow may be too heavy for the first production version because it adds cluster dependencies, storage configuration, artifact stores, and platform operations before the daily batch business process has proven value.

Compile the example pipeline after installing the optional Kubeflow dependency:

```bash
pip install -r requirements-kubeflow.txt
python kubeflow/pipeline.py
```

The pipeline demonstrates the same lifecycle:

1. generate/load data
2. validate data
3. build features
4. train model
5. evaluate model
6. register/promote model in MLflow
7. generate monitoring report

This aligns with Kubernetes-based companies without making Kubernetes mandatory for the core Airflow demo.
