# RAG Eval Pipeline

RAG evaluation console with a FastAPI backend and a React/Vite frontend.

## Directory Structure

```text
.
|-- rag_eval_pipeline/          # Python package
|   |-- api.py                  # CLI entry point; starts FastAPI with uvicorn
|   |-- api_app.py              # FastAPI app factory, health endpoint, static frontend serving
|   |-- api_models.py           # Pydantic request models
|   |-- api_routes/             # FastAPI route modules
|   |   |-- chat.py
|   |   |-- datasets.py
|   |   |-- pipeline.py
|   |   `-- reports.py
|   |-- api_services/           # Shared API state
|   |-- config.py
|   |-- evaluation.py
|   |-- generation.py
|   |-- http.py
|   |-- pipeline.py
|   `-- visualization.py
|-- web/                        # React/Vite/TypeScript frontend
|   |-- src/
|   |   |-- api/                # Typed API client and frontend types
|   |   |-- components/         # Shared console components
|   |   |-- pages/              # Datasets, Run, Records, Reports, Overview pages
|   |   |-- styles/
|   |   `-- App.tsx
|   |-- package.json
|   |-- README.md
|   `-- vite.config.ts
|-- scripts/                    # Pipeline, mock service, and helper scripts
|-- tests/                      # Python test suite
|-- data/                       # Local input/output data; mostly ignored by git
|-- reports/                    # Generated HTML reports; ignored by git
|-- docs/                       # Design notes and implementation plans
|-- requirements.txt
|-- requirements-dev.txt
`-- pyproject.toml
```

## Install

```powershell
python -m pip install -e .
python -m pip install -r requirements.txt
```

Install frontend dependencies:

```powershell
cd web
npm install
```

## Start App

Start the FastAPI backend:

```powershell
python -m rag_eval_pipeline.api --host 127.0.0.1 --port 9000
```

Use this address for normal usage:

```text
http://127.0.0.1:9000/datasets
```

The backend also serves the built React frontend from `web/dist`.

## Frontend Development Only

Use Vite only when editing React or CSS and you want hot reload. Run the backend first, then start Vite:

```powershell
cd web
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

Vite shows the same app and proxies API calls to the backend on port `9000`.
See `web/README.md` for frontend-specific commands and structure.

## Build Frontend For Backend Serving

```powershell
cd web
npm run build
```

After the build, the backend serves the React app from `web/dist` at `/` and `/datasets`.

## Useful Commands

Run Python tests:

```powershell
python -m pytest -q
```

Run frontend tests:

```powershell
cd web
npm test
```

Run frontend production build:

```powershell
cd web
npm run build
```

Dry-run the evaluation pipeline:

```powershell
python scripts/run_rag_eval_pipeline.py --config scripts/eval-cfg.yaml --dry-run
```

Run the evaluation pipeline:

```powershell
python scripts/run_rag_eval_pipeline.py --config scripts/eval-cfg.yaml
```
