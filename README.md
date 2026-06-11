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
|   |   |-- retrieval.py
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
|-- docs/                       # Project documentation
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

### Docker Compose

For the containerized app, build and start the backend, frontend, local
retrieval service, and mock OpenAI-compatible service:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\docker-start.ps1
```

Open:

```text
http://127.0.0.1:8080/datasets
```

The frontend is served by Nginx. The FastAPI backend is exposed on port `9000`,
the local retrieval service on `9380`, and the mock OpenAI-compatible service on
`8011`.

### Local Python

Start the FastAPI backend. This serves the API and the built React frontend on
port `9000`:

```powershell
python -m rag_eval_pipeline.api --host 127.0.0.1 --port 9000
```

Use this address for normal usage:

```text
http://127.0.0.1:9000/datasets
```

The backend also serves the built React frontend from `web/dist`.

## Local Knowledge Base Retrieval

The project can call any external retrieval API through `generation.retrieval_url`.
The `Datasets` page uses that retrieval URL when the user clicks
`Generate Answer`:

1. The browser calls `POST /api/datasets/generate-answer` on the backend.
2. The backend calls `generation.retrieval_url` to retrieve knowledge-base
   chunks.
3. The backend sends the question plus retrieved passages to the configured LLM.

The backend on port `9000` is not enough for grounded answer generation. A
retrieval service must also be reachable at the configured URL.

The repository includes a small sample chunk file at
`data/sample_chunks.jsonl` so a fresh clone can start the local retrieval
service immediately. For real evaluations, replace `local_retrieval.chunks_path`
with the JSONL file generated from your knowledge base.

```yaml
generation:
  retrieval_url: "http://127.0.0.1:9380/api/v1/retrieval"

local_retrieval:
  chunks_path: data/sample_chunks.jsonl
  cache_path: data/local_chunks.index.jsonl
  batch_size: 10
```

If `scripts/eval-cfg.yaml` contains a direct `evaluation.embedding_api_key`,
start the retrieval server directly:

```powershell
python .\scripts\local_chunks_retrieval_server.py --config .\scripts\eval-cfg.yaml
```

If the config uses `evaluation.embedding_api_key_env` instead, set that
environment variable before starting the server.

The first run embeds all chunks and writes `data/local_chunks.index.jsonl`.
Later runs reuse that cache unless you pass `--rebuild-index`.

Check whether the retrieval server is listening:

```powershell
Test-NetConnection 127.0.0.1 -Port 9380
```

Expected result:

```text
TcpTestSucceeded : True
```

Test retrieval directly:

```powershell
curl.exe -X POST http://127.0.0.1:9380/api/v1/retrieval `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"What is BSD?\",\"page_size\":5}"
```

Use the backend wrapper endpoint when callers need to preview retrieved chunks:

```text
POST http://127.0.0.1:9000/api/retrieval/chunks
```

Example body:

```json
{
  "question": "What is SM94?",
  "page_size": 5,
  "similarity_threshold": 0.1,
  "dataset_ids": ["123"],
  "document_ids": []
}
```

The backend reads retrieval defaults from `scripts/eval-cfg.yaml`, calls the
configured retrieval API, and returns normalized chunks with `id`, `text`, and
`source`.

### Common Retrieval Error

If the frontend shows an error like this after clicking `Generate Answer`:

```text
Retrieval request failed: <urlopen error [WinError 10061] ...>
```

it means the backend tried to connect to `generation.retrieval_url`, but no
service accepted the TCP connection. With the default config, the missing
service is:

```text
http://127.0.0.1:9380/api/v1/retrieval
```

Fix:

```powershell
python .\scripts\local_chunks_retrieval_server.py --config .\scripts\eval-cfg.yaml
Test-NetConnection 127.0.0.1 -Port 9380
```

This is different from an API key problem. An API key problem usually returns
HTTP `401` or `403`; `WinError 10061` means the retrieval service is not
listening or is refusing the connection.

### Datasets Page Retrieval Notes

- `Use dataset` selects the QA CSV that is previewed and saved to.
- `Preview: <name>` shows rows from that QA CSV only. It does not call the
  retrieval service.
- `Generate Answer` calls retrieval, then calls the LLM to draft the answer.
- `Retrieved passages` shows the chunks returned by retrieval for the last
  generated answer.

## Environment Variables

Set service URLs and API keys outside the repository before running generation
or evaluation. Do not commit real secrets.

```powershell
$env:RETRIEVAL_API_KEY="..."
$env:LLM_BASE_URL="https://your-openai-compatible-host/v1"
$env:LLM_API_KEY="..."
$env:EVAL_CHAT_BASE_URL="https://your-openai-compatible-host/v1"
$env:EVAL_CHAT_API_KEY="..."
$env:EMBEDDING_BASE_URL="https://your-embedding-host/v1"
$env:EMBEDDING_API_KEY="..."
```

Update `generation.llm_model`, `evaluation.chat_model`, and
`evaluation.embedding_model` in `scripts/eval-cfg.yaml` for your provider.

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
