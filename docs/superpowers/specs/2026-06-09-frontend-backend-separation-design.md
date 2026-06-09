# Frontend Backend Separation Design

## Purpose

Separate the current browser UI from the Python API server while keeping the existing RAG evaluation pipeline behavior intact. The first phase should move the `/datasets` console experience into a dedicated frontend under `web/`, expose the existing server behavior through clearer API boundaries, and avoid platform features that are not needed yet.

This design chooses the incremental "C" approach: split the frontend and backend now, keep local file-based data storage, and leave database-backed persistence, authentication, multi-user permissions, and queue infrastructure for later phases.

## Current Context

The project is currently a Python package with a lightweight HTTP server in `rag_eval_pipeline/api.py`. That file handles HTML rendering, request routing, upload parsing, report listing, pipeline job launching, chat calls, and response serialization in one module. Existing tests already cover upload APIs, HTTP helpers, visualization, and functional smoke paths.

The frontend is currently generated directly by `render_upload_page()` in the backend. The backend already exposes many useful API-like endpoints, including reports, datasets, evaluation runs, pipeline jobs, chat, manual QA entry, and QA generation.

## Recommended Stack

Use these technologies for the first split:

- Backend: FastAPI with Uvicorn.
- Frontend: React, Vite, and TypeScript.
- Frontend location: new `web/` directory at the repository root.
- Styling: plain CSS or CSS Modules for the first pass.
- Frontend data access: a small typed API client built on `fetch`.
- Data storage: keep the existing local `data/`, `reports/`, and `scripts/eval-cfg.yaml` paths.
- Development integration: Vite dev server proxies API calls to the FastAPI server.
- Production integration: FastAPI can serve the built frontend static assets after `web/` is built.

Do not introduce a database, background queue service, authentication provider, UI framework, or server-side frontend framework in this phase.

## Backend Architecture

Add a FastAPI-based app while preserving the existing pipeline modules and behavior.

Proposed structure:

```text
rag_eval_pipeline/
  api.py
  api_app.py
  api_models.py
  api_routes/
    __init__.py
    chat.py
    datasets.py
    pipeline.py
    reports.py
  api_services/
    __init__.py
    chat.py
    datasets.py
    pipeline.py
    reports.py
```

`api_app.py` owns app creation, middleware, static frontend mounting, and route registration. `api_models.py` defines Pydantic request and response models for the public API. Route modules stay thin: parse validated inputs, call services, and return response models. Service modules hold the reusable behavior that can be migrated from `api.py` in small pieces.

Keep `rag_eval_pipeline/api.py` as a compatibility entry point during migration. It can either continue to run the current server until replacement is complete or delegate to the new FastAPI app once the new API reaches parity.

## Frontend Architecture

Create `web/` as the independent frontend workspace.

Proposed structure:

```text
web/
  package.json
  index.html
  vite.config.ts
  tsconfig.json
  src/
    main.tsx
    App.tsx
    api/
      client.ts
      types.ts
    components/
      AppShell.tsx
      JobRail.tsx
      StatusBadge.tsx
    pages/
      Overview.tsx
      Reports.tsx
      EvaluationRuns.tsx
      RunEvaluation.tsx
      Datasets.tsx
    styles/
      globals.css
```

The initial UI should preserve the current console-oriented product shape from the existing RAG evaluation console design:

- Overview for latest report, latest run, active job, and recent activity.
- Reports for report listing, selected report preview, and report-aware chat.
- Evaluation Runs for persisted run history.
- Run Evaluation for quick and standard evaluation launch.
- Datasets for CSV/PDF upload, dataset preview, manual QA, answer generation, and bulk QA generation where already supported by the backend.
- Persistent job rail for active job status and logs.

Use React component state for the first pass. Keep shared state small and close to the components that need it. Add a heavier state or query library only after repeated request caching, polling, or invalidation logic becomes hard to maintain.

## API Surface

Expose a JSON API under `/api`. The first phase should keep endpoint names close to the current server to reduce migration risk:

```text
GET  /api/reports
GET  /api/eval-runs
GET  /api/datasets
GET  /api/datasets/preview
GET  /api/pipeline/status

POST /api/datasets/upload
POST /api/datasets/manual-qa
POST /api/datasets/generate-answer
POST /api/datasets/regenerate-answer
POST /api/datasets/generate-qa
POST /api/datasets/bulk-qa
POST /api/pipeline/run
POST /api/pipeline/cancel
POST /api/chat
POST /api/chat/save
```

New or materially changed responses should use a consistent envelope:

```json
{
  "ok": true,
  "data": {}
}
```

Errors should use the same shape with an error message:

```json
{
  "ok": false,
  "error": "Human-readable error message"
}
```

During migration, compatibility with current clients matters more than perfect endpoint redesign. If an existing endpoint response shape is already covered by tests, preserve that shape in the first FastAPI version or update tests with an explicit migration reason. Normalizing all legacy responses into the envelope can happen after route parity is proven.

## Data Flow

The frontend loads initial console data by calling reports, evaluation runs, datasets, and pipeline status endpoints. User actions call the relevant API endpoints and update the local page state from the response.

Pipeline runs remain backend-owned. The frontend launches a run and stores the returned `run_id` in component state or local storage. The job rail polls `/api/pipeline/status?run_id=...` until the job reaches a terminal state. The backend remains responsible for executing scripts, reading manifests, scanning report files, and guarding filesystem paths.

Reports remain served by the backend. The frontend can preview a selected report through an iframe or a dedicated report URL, while chat sends the selected report URL or report identifier back to the backend for context extraction.

## Error Handling

Backend services should translate validation and operational failures into stable API errors. FastAPI request validation errors should be converted into user-readable messages where possible.

The frontend should show errors inside the affected area:

- Upload errors in the Datasets page.
- Pipeline launch errors in Run Evaluation and the job rail.
- Job failure details in the job rail.
- Report loading errors in Reports.
- Chat errors beside the chat controls.

Errors should not replace the entire app shell unless the app cannot load at all.

## Testing

Backend tests should cover:

- FastAPI app creation.
- Core route parity for existing upload, reports, datasets, pipeline status, pipeline run, chat, and QA endpoints.
- Response envelope consistency for new or changed endpoints.
- Filesystem path safety for report and dataset access.
- Compatibility entry point behavior for `rag-eval-api`.

Frontend tests can start lightweight:

- API client response handling.
- App shell navigation rendering.
- Job rail status rendering.
- Main page rendering for Reports, Evaluation Runs, Run Evaluation, and Datasets.

Manual verification should cover:

- Starting the FastAPI server.
- Starting the Vite dev server with API proxy.
- Uploading a CSV dataset.
- Viewing reports.
- Launching a quick or standard evaluation against the local mock services.
- Watching job status progress.
- Building the frontend and serving the built static assets from FastAPI.

## Migration Plan

Implementation should be incremental:

1. Add FastAPI dependencies and a minimal `api_app.py` that can serve health/status and register route modules.
2. Extract backend service functions from `api.py` without changing behavior.
3. Recreate current `/api` behavior in FastAPI route modules.
4. Scaffold `web/` with React, Vite, and TypeScript.
5. Build the app shell and route-level pages.
6. Move console interactions from backend-rendered HTML into React.
7. Add static asset serving for the built frontend.
8. Keep or delegate the old `api.py` entry point so existing commands continue to work.
9. Update docs with development and production run commands.

## Out of Scope

This phase does not include:

- User accounts, roles, or authentication.
- Database-backed persistence.
- Distributed task queues.
- Multi-server deployment.
- Rewriting the RAG evaluation pipeline itself.
- Replacing local reports with a report database.
- A full UI redesign beyond preserving the approved evaluation console shape.

## Open Decisions Resolved

The first frontend workspace will be `web/`. The initial product shape is a local or internal RAG evaluation console rather than a full multi-user web platform. The preferred path is an incremental split, not a full rewrite.
