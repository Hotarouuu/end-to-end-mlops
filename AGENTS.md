# AGENTS.md

## Commands

```bash
# Install deps (use uv, not pip)
uv pip install -e . --system

# Run tests
pytest

# Lint + format
pylint -j 0 --fail-under=6.5 .
black .

# Pre-commit
pre-commit run --all-files

# Start all services (API + MLFlow + trainer)
docker compose up --build

# Start single service
docker compose up api    # or mlflow, trainer
```

## Architecture

- **Crop prediction API** (FastAPI) — `app.py` loads model from MLFlow registry at startup via lifespan context manager
- **Model loader** — `src/models/model.py` handles model import from MLFlow registry
- **Helpers** — `src/models/helpers.py` contains config loading utilities
- **Training pipeline** — `src/models/training.py` trains XGBoost, logs to MLFlow, saves decode map, auto-promotes to Production
- **MLFlow server** — experiment tracking + model registry at port 5000
- **API serves predictions** at port 8000 (`POST /predict`)

## Critical Gotchas

1. **API requires trained model in MLFlow Production stage** — `src/models/model.py` calls `client.get_latest_versions(model_name, stages=["Production"])` during FastAPI lifespan startup. Will crash if no model registered.

2. **Training must run before API** — `compose.yaml` has trainer commented out in api depends_on. Train first, then start API.

3. **SHAP installed from GitHub** — PyPI version incompatible with XGBoost 3.2.0. Dockerfile does: `uv pip install git+https://github.com/shap/shap.git --system`

4. **CONFIG env var** — `app.py` reads `os.getenv("CONFIG")` (default `./config/model1.yaml`).

5. **Integration tests empty** — `tests/test_integration.py` exists but is empty. Unit tests exist in `test_unit_api.py` and `test_unit_model.py`.

6. **numpy < 2.0.0** pinned in dependencies — compatibility constraint.

7. **xgboost==3.2.0** pinned exactly (not >=) in pyproject.toml — compatibility constraint.

8. **Model auto-promotion** — `training.py` has `promote_model_if_better()` function that promotes model to Production if LogLoss < 0.33 benchmark threshold.

## CI/CD

- **CI** (`.github/workflows/ci.yaml`): triggers on `dev` push → black → pylint → pytest → docker build test
- **CD** (`.github/workflows/cd.yaml`): triggers on `master` push/PR → deploys to EC2 via SSH → runs `docker compose up`
- **CD requires secrets**: `EC2_SSH_KEY`, `REMOTE_HOST`, `REMOTE_USER`

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI app, loads model from MLFlow at startup |
| `src/models/model.py` | Model import from MLFlow registry |
| `src/models/helpers.py` | Config loading utilities |
| `src/models/training.py` | Training pipeline, MLflow logging, SHAP analysis |
| `config/model1.yaml` | Model config, data paths, artifact names |
| `compose.yaml` | 3 services: api, mlflow, trainer |
| `data/Crop_recommendation.csv` | Training dataset |
| `tests/conftest.py` | Pytest fixtures, MLflow mocking |
| `tests/test_unit_api.py` | API endpoint unit tests |
| `tests/test_unit_model.py` | Model import unit tests |
| `tests/test_integration.py` | Integration tests (empty) |

## Config Structure (`config/model1.yaml`)

- `data.train_path`: CSV path
- `data.features`: 7 features (N, P, K, temperature, ph, humidity, rainfall)
- `data.target`: "label"
- `model.variables`: XGBoost params
- `artifacts.model_name`: "XGBOOST" (MLFlow registry name)
- `artifacts.decode_path`: "/DECODE_MAP/decode_map.pkl" (MLFlow artifact path)
