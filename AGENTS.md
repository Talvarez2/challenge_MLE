# AGENTS.md — challenge_MLE

## Project Overview

Flight delay prediction API for SCL airport. An XGBoost model predicts whether a flight will be delayed (>15 min) based on airline, flight type, and month. Built as an ML Engineering challenge covering model operationalization, API development, containerization, and CI/CD.

## Project Structure

```
challenge/              # Application code
  config.py             # Constants: feature list, thresholds, season ranges, model path
  preprocessing.py      # Feature engineering and data preparation
  model.py              # DelayModel class (XGBoost wrapper with save/load)
  api.py                # FastAPI app with /health and /predict endpoints
  __init__.py           # Exports the FastAPI app
  exploration.ipynb     # Original DS notebook (reference only)
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
serverless.yml          # AWS Lambda config (unused — Lambda too small for xgboost)
package.json            # Node deps for serverless framework (unused)
docs/challenge.md       # Design decisions and bug fixes documentation
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

## API Endpoints

- `GET /health` — Returns `{"status": "OK"}`
- `POST /predict` — Accepts `{"flights": [{"OPERA": str, "TIPOVUELO": "N"|"I", "MES": 1-12}]}`, returns `{"predict": [0|1, ...]}`

## Key Conventions

- **Constants**: UPPER_SNAKE_CASE in `config.py`. All magic values live there.
- **Type hints**: All function signatures have type annotations. Use `X | Y` union syntax.
- **Docstrings**: Google-style docstrings on all public functions and classes.
- **Validation**: API input validated via Pydantic models with `@validator` decorators.
- **Naming**: Modules use snake_case. Private helpers prefixed with `_`.
- **Model loading**: Use `DelayModel.load_model()` classmethod, not raw pickle.
- **Paths**: Use `pathlib.Path` for file paths (see `config.py`, `model.py`).
- **Dependencies**: Runtime deps in `requirements.txt`, test deps in `requirements-test.txt`, dev/viz deps in `requirements-dev.txt`.
- **No macOS artifacts**: `.DS_Store` and `__MACOSX/` are gitignored.
