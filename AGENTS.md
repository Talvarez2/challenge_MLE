# AGENTS.md — challenge_MLE

## Project Overview

Flight delay prediction API for SCL airport. An XGBoost model predicts whether a flight will be delayed (>15 min) based on airline, flight type, and month.

## Project Structure

```
challenge/              # Application code
  config.py             # Constants: feature list, thresholds, season ranges, model path
  preprocessing.py      # Feature engineering and data preparation
  model.py              # DelayModel class (XGBoost wrapper with save/load)
  api.py                # FastAPI app with /health and /predict endpoints
  __init__.py           # Exports the FastAPI app
tests/
  model/test_model.py   # Unit tests for preprocessing and model training
  api/test_api.py       # Integration tests for the API endpoints
  stress/api_stress.py  # Locust load tests
data/
  data.csv              # Training dataset
  model.pkl             # Serialized trained model
save_model.py           # Script to train and persist the model
Dockerfile              # Container image (non-root user, health check)
.dockerignore           # Minimizes Docker build context
.coveragerc             # Coverage configuration (70% threshold)
Makefile                # Build, test, and stress-test targets
.github/workflows/      # CI/CD pipelines (ci.yml, cd.yml)
```

## How to Run

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
make install

# Train model
python save_model.py

# Run API locally
uvicorn challenge.api:app --reload

# Run tests
make model-test
make api-test
make stress-test

# Docker
docker build -t flight-delay-api .
docker run -p 8000:8000 flight-delay-api
```

## Key Conventions

- **Constants**: UPPER_SNAKE_CASE in `config.py`. All magic values live there.
- **Type hints**: All function signatures must have type annotations. Use `X | Y` syntax (not `Union`).
- **Docstrings**: Google-style docstrings on all public functions and classes.
- **Validation**: API input validated via Pydantic models, not manual checks.
- **Naming**: Modules use snake_case. Private helpers prefixed with `_`.
- **Model loading**: Use `DelayModel.load_model()` classmethod, not raw pickle.
- **No macOS artifacts**: `.DS_Store` and `__MACOSX/` are gitignored.
