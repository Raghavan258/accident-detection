# Accident Detection MLOps frontend

React interface for the existing FastAPI service. It reads MLflow, the report files, and the prediction endpoints. It does not train a model.

## Run

From the repository root, start the API with the local registry and reports:

```powershell
$env:PYTHONPATH = "src"
$env:MLFLOW_TRACKING_URI = "sqlite:///mlops/mlflow.db"
$env:IVERAS_REPORT_DIR = "reports/mlops"
$env:IVERAS_DB = "data/processed/accident_detection.db"
$env:IVERAS_FEATURES = "datasets/processed/features.parquet"
.\.venv\Scripts\python.exe -m uvicorn accident_detection.mlops.api:app --host 127.0.0.1 --port 8000
```

In `frontend/`:

```powershell
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The API base is `VITE_API_BASE` in `.env.development` (`http://127.0.0.1:8000`). CORS allows `http://localhost:5173` and `http://127.0.0.1:5173`.

## Checks

```powershell
npm test
npm run build
```

## API used

| Method | Path | Use |
| --- | --- | --- |
| GET | `/mlops/status` | Registry, runs, validation and test metrics, drift, quality |
| GET | `/mlops/health` | Live checks for MLflow, SQLite, reports, model load, Docker, cloud |
| GET | `/mlops/schema` | Twelve defensible feature fields |
| GET | `/mlops/examples` | Train rows nearest class median peaks |
| POST | `/predict` | Feature prediction |
| POST | `/predict/window` | Sensor-window prediction |
| GET | `/mlops/reports/drift` | Evidently HTML |
| POST | `/mlops/registry/production` | Move the production alias to a version that already passed the gate |

Prediction history is not stored. The overview shows N/A for the prediction count. Docker and cloud stay unavailable unless those checks succeed.
