# Testing Report — End-to-End Farm Detection

**Date:** 2026-03-01
**File:** `reports/2026-03-01_TESTING_REPORT.md`
**Tool:** GitHub Copilot CLI

---

## Summary

| Category | Count |
|---|---|
| Test files created | 3 |
| Test files improved | 2 |
| Bugs fixed | 2 |
| Dependencies installed | 1 |
| Total tests (final) | **19 passing** |

---

## 1. API Integration Tests — `tests/test_api_integration.py` ✅ Created

New file with 6 tests for the FastAPI app using `TestClient` (no real server required).

### Dependency installed
```
httpx==0.28.1
```
Required by FastAPI's `TestClient` to simulate HTTP requests.

### Support file created: `tests/conftest.py`
Adds the project root to `sys.path` so that `app.py` is importable by the tests.

### Tests created

| Test | Endpoint | What it validates |
|---|---|---|
| `test_root_returns_welcome_message` | `GET /` | Status 200 and `message` field present |
| `test_predict_returns_prediction_and_label` | `POST /predict` | Status 200, `prediction` (int) and `label` (str) present |
| `test_predict_known_input_returns_expected_label` | `POST /predict` | Known input returns `prediction=17, label="papaya"` |
| `test_predict_missing_field_returns_422` | `POST /predict` | Missing required field returns HTTP 422 |
| `test_predict_invalid_type_returns_422` | `POST /predict` | Invalid field type returns HTTP 422 |
| `test_predict_different_valid_input` | `POST /predict` | Another valid payload returns 200 with prediction and label |

---

## 2. Improvements to Existing Tests

### 2.1 `tests/test_predict.py` — From 1 to 3 tests

**Changes:**
- Added `predictor` fixture with `scope="module"` — model loaded once for all tests (performance)
- Renamed `test_predict` → `test_predict_returns_expected_class_and_label` (descriptive name)
- Added `test_predict_returns_list_types`: validates output length and types
- Added `test_predict_raises_on_invalid_input`: ensures invalid input raises an exception (exercises the `try/except` block in the source code)

### 2.2 `tests/test_preprocessor.py` — From 2 to 5 tests

**Changes:**
- Removed unused import: `from typing import List`
- Added `sample_df` fixture with 2 samples (required for `StandardScaler` to work correctly with variance)
- `test_preprocessor` → `test_fit_transform_returns_float_columns`: per-column assertion instead of fixed list
- `test_preprocessor_transform` → `test_transform_output_dtype_is_float`: kept and corrected to use 2 samples
- Added `test_fit_transform_encodes_labels`: verifies correct label encoding on output
- Added `test_fit_transform_applies_log_to_humidity_and_rainfall`: validates that `log_transform` replaces `humidity` and `rainfall` with `humidity_log` and `rainfall_log`
- Added `test_fit_then_transform_matches_fit_transform`: ensures consistency between both APIs using `np.testing.assert_array_almost_equal`

---

## 3. Training Integration Tests — `tests/test_training_integration.py` ✅ Created

New file with 5 tests for the full training pipeline.

### Isolation strategy

| Dependency | Problem | Solution |
|---|---|---|
| MLflow (`http://mlflow:5000`) | Requires running server | `patch("farm_detection.models.training.mlflow")` with `autouse=True` |
| Artifacts (`./model/*.joblib`) | Would overwrite production models | `tmp_path` fixture redirects output to a temporary directory |
| Config YAML | Hardcoded production paths | `patch("farm_detection.models.training.load_config")` injects test config |
| Dataset | Real dataset (2200 rows) | Used as-is — acceptable runtime (~4s) |

### Tests created

| Test | What it validates |
|---|---|
| `test_train_completes_without_error` | Pipeline runs from start to finish without exception |
| `test_train_saves_model_and_preprocessor_to_disk` | Both `.joblib` files are created on disk |
| `test_train_saved_model_can_predict` | Saved model can make a prediction with a real sample |
| `test_train_saved_preprocessor_has_required_keys` | Saved dictionary contains `scaler` and `labelencoder` keys |
| `test_train_mlflow_run_is_started` | Code calls `mlflow.start_run()` and `set_experiment()` correctly |

---

## 4. Bugs Fixed

### Bug 1 — Config path in `.env`

**File:** `.env`

```diff
- CONFIG = "/config/model1.yaml"
+ CONFIG=./config/model1.yaml
```

**Root cause:** The absolute path `/config/model1.yaml` is a Docker-internal path. Locally (and in pytest), the file did not exist at that location.  
**Effect if not fixed:** `app.py` would raise `FileNotFoundError` when imported by the tests.

---

### Bug 2 — Wrong config keys in `app.py`

**File:** `app.py`

```diff
- model_path=config["model"]["path"],
- preprocessor_path=config["preprocessor"]["path"]
+ model_path=config["artifacts"]["model_path"],
+ preprocessor_path=config["artifacts"]["preprocessor_path"],
```

**Root cause:** The YAML stores artifact paths under the `artifacts` key, not `model`/`preprocessor`. The wrong key access would raise a `KeyError` on API startup.

---

## 5. Final Result

```
19 passed, 18 warnings in 1.26s
```

| File | Tests | Status |
|---|---|---|
| `tests/test_api_integration.py` | 6 | ✅ |
| `tests/test_predict.py` | 3 | ✅ |
| `tests/test_preprocessor.py` | 5 | ✅ |
| `tests/test_training_integration.py` | 5 | ✅ |

---

## 6. Files Changed

| File | Action |
|---|---|
| `tests/test_api_integration.py` | **Created** |
| `tests/conftest.py` | **Created** |
| `tests/test_training_integration.py` | **Created** |
| `tests/test_predict.py` | **Improved** |
| `tests/test_preprocessor.py` | **Improved** |
| `.env` | **Fixed** (config path) |
| `app.py` | **Fixed** (config YAML keys) |
| `pyproject.toml` | **Updated** automatically (`httpx` added via `uv add`) |
