# Copilot Instructions

## Commands

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_predict.py

# Run a single test function
pytest tests/test_predict.py::test_fit_does_not_raise

# Format code
black .

# Install dependencies (uses uv)
uv pip install -e . --system
```

Tests require `CONFIG=./config/model1.yaml` to be set (already the default in the Dockerfile and CI).

## Architecture

Three Docker services defined in `compose.yaml`:
1. **mlflow** — starts first; all others wait for its health check (`http://mlflow:5000`)
2. **trainer** — runs `training.py` once to train and register the model in MLflow
3. **api** — starts after trainer completes; serves predictions via FastAPI on port 8000

**Training flow** (`src/farm_detection/models/training.py`):
- Loads config from `config/model1.yaml`
- Fits a `GNBWithEncoding` model (log-transform → StandardScaler → GaussianNB)
- MLflow `sklearn.autolog` registers the model directly in the registry under `farm-detection-gnb-model`
- The `LabelEncoder` is saved separately to `artifacts/gnb_model_label_encoder.joblib` and logged as an MLflow artifact — it is **not** captured by autolog

**Prediction flow** (`app.py`):
- At startup, loads the latest registered model version from MLflow and downloads the label encoder artifact separately
- `/predict` applies `Preprocessor.log_transform` before calling `model.predict`, then uses the label encoder for `inverse_transform`

**Why the label encoder is separate**: MLflow's sklearn autolog does not capture the `LabelEncoder` that lives outside the pipeline, so it must be persisted and loaded independently.

## Key Conventions

**`GNBWithEncoding` extends `Preprocessor`** — the model class inherits `log_transform` from `Preprocessor` and calls it inside both `fit` and `predict`. Any input DataFrame passed to either method will be mutated in place (humidity/rainfall columns are replaced). Always pass `X.copy()` when the original DataFrame needs to be preserved.

**Config-driven** — all paths (data, artifacts) and hyperparameters come from `config/model1.yaml`. The config path is injected via the `CONFIG` env var. Tests patch `load_config` directly rather than writing temp YAML files.

**Mocking pattern for API tests** — `app.py` runs MLflow calls at module import time, so `test_api_integration.py` patches all MLflow/joblib calls before importing the module inside the fixture, then removes `app` from `sys.modules` on teardown.

**CI runs on `dev` branch only** — the `ci.yaml` workflow triggers on push to `dev`. It runs black, pytest, and a full `docker compose up` smoke test in sequence.
