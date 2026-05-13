# Custom Model Server

The first KServe option uses the existing FastAPI container as a custom predictor. This is the pragmatic fallback because the model depends on feature engineering code, MLflow alias lookup, and recent terminal history, not just a raw sklearn artifact.

For a stricter KServe sklearn-server deployment, export a plain MLflow sklearn model with:

```bash
make train
make promote-model
python scripts/export_model_for_serving.py --output-dir models/serving
```

Then adapt `inferenceservice.yaml` to point to the exported artifact location.

