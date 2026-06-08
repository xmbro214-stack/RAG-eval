# Dataset Upload API Design

## Goal

Add a lightweight API service and upload page that let users upload an evaluation QA CSV. The uploaded CSV becomes a reusable input file for the existing RAG evaluation pipeline by setting both `queries_csv` and `golden_csv` to the saved path.

This feature only handles evaluation dataset upload and validation. It does not upload documents into RAGFlow, create vector indexes, or start a pipeline run.

## Current Context

The project is currently a CLI-based RAG evaluation pipeline. It reads:

- `queries_csv`: questions to send through retrieval and generation.
- `golden_csv`: reference answers used by evaluation metrics.
- `datasets`: RAGFlow dataset IDs used by retrieval.

The same QA CSV can serve as both `queries_csv` and `golden_csv` when it contains `query_id`, `query`, and `expected_answer`.

## API Service

Create a new module:

```text
rag_eval_pipeline/api.py
```

Start command:

```powershell
python -m rag_eval_pipeline.api --host 127.0.0.1 --port 9000
```

The service should use Python standard library HTTP primitives to avoid adding a new web framework dependency.

## Endpoint

### Upload Page

```text
GET /datasets
```

Return a self-contained HTML page for uploading evaluation datasets.

Page capabilities:

- Select a local `.csv` file.
- Enter an optional dataset name.
- Upload the file to `POST /api/datasets/upload`.
- Show upload success or validation errors inline.
- After success, display the saved path, row count, and a copyable config snippet.

Successful upload display:

```yaml
queries_csv: data/uploaded_datasets/demo.csv
golden_csv: data/uploaded_datasets/demo.csv
```

The page should use the same visual direction as the report page: deep blue background, AT&S mark in the top-left, restrained Apple-like spacing, and compact operational controls.

### Upload API

```text
POST /api/datasets/upload
```

Request type:

```text
multipart/form-data
```

Fields:

- `file`: required CSV file.
- `name`: optional dataset name. If omitted, the file stem is used.

Required CSV columns:

```text
query_id,query,expected_answer
```

Validation rules:

- Uploaded file name must end with `.csv`.
- Required columns must exist.
- At least one valid data row must exist.
- `query` and `expected_answer` must not be blank.
- Blank `query_id` is allowed and will be filled as `query_1`, `query_2`, etc.
- Dataset name must be slugified to prevent path traversal and invalid file names.

## Storage

Save validated files under:

```text
data/uploaded_datasets/{dataset_name}.csv
```

If a file with the same dataset name already exists, append a numeric suffix:

```text
demo.csv
demo_2.csv
demo_3.csv
```

The saved CSV should be normalized to UTF-8 and only include the required columns in this order:

```text
query_id,query,expected_answer
```

## Responses

Successful response:

```json
{
  "ok": true,
  "dataset_name": "demo",
  "path": "data/uploaded_datasets/demo.csv",
  "rows": 10,
  "columns": ["query_id", "query", "expected_answer"]
}
```

Failure response:

```json
{
  "ok": false,
  "error": "CSV missing required columns: expected_answer"
}
```

Use HTTP status codes:

- `200` for successful uploads.
- `400` for invalid requests or invalid CSV content.
- `404` for unknown routes.
- `405` for unsupported methods.

## Frontend Behavior

The upload page should be plain HTML, CSS, and JavaScript rendered by `rag_eval_pipeline/api.py`. It should not require a separate build step.

Interaction flow:

1. User opens `http://127.0.0.1:9000/datasets`.
2. User chooses a CSV and optionally enters a dataset name.
3. JavaScript submits a `FormData` request to `/api/datasets/upload`.
4. The page disables the upload button while the request is in progress.
5. On success, the page shows the returned metadata and config snippet.
6. On failure, the page shows the error message without reloading.

Accessibility and usability:

- File input and name input should have labels.
- Upload button should have loading and disabled states.
- Error text should be visually distinct.
- Long paths should wrap instead of overflowing.
- The page should work on desktop and narrow browser widths.

## Pipeline Integration

The upload endpoint returns a saved relative path. Users can copy that path into the existing config:

```yaml
queries_csv: data/uploaded_datasets/demo.csv
golden_csv: data/uploaded_datasets/demo.csv
```

The existing run command remains unchanged:

```powershell
python scripts/run_rag_eval_pipeline.py --config scripts/eval-cfg.yaml
```

This keeps upload concerns separate from pipeline execution.

## Testing

Add focused tests for:

- Successful CSV upload.
- Missing required column.
- Blank `query` or `expected_answer`.
- Dataset name slugification and safe output path.
- Duplicate dataset names creating suffixed files.
- `GET /datasets` returns an HTML page containing the upload form.

Tests should avoid opening network ports when possible by testing parsing, validation, and save helpers directly. A small HTTP-level test can be added if the handler remains simple to exercise in-process.

## Documentation

Update `README.md` with:

- How to start the API server.
- How to open the upload page.
- How to upload a CSV with PowerShell.
- How to use the returned path in `scripts/eval-cfg.yaml`.
