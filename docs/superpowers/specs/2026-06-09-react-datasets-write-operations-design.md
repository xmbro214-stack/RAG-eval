# React Datasets Write Operations Design

## Goal

Connect the React `Datasets` page to the existing FastAPI write routes for a small, complete dataset-management workflow:

- Upload a CSV or PDF dataset.
- Select an existing dataset and load its preview.
- Append one manual question and expected answer to the selected dataset.

Bulk Q&A generation and automated candidate generation are intentionally out of scope for this slice.

## Current Context

The backend already exposes the needed routes:

- `GET /api/datasets`
- `GET /api/datasets/preview?path=...`
- `POST /api/datasets/upload`
- `POST /api/datasets/manual-qa`

The React page currently lists datasets and renders inactive upload controls. The API client already has `listDatasets`, `previewDataset`, and `uploadDataset`, but it does not yet expose a manual Q&A save method.

## User Experience

The `Datasets` page stays as a single-page inline workflow.

At the top, the user can enter an optional dataset name, choose a `.csv` or `.pdf` file, and upload it. During upload, the upload button is disabled. On success, the page refreshes the dataset list, selects the uploaded dataset when the response includes a path, and shows a short success message.

The dataset list becomes selectable. Clicking a dataset stores it as the active dataset and calls the preview endpoint. The preview area shows the row count and a compact table of preview rows. If the dataset has no preview rows, the page shows an empty state instead of a broken table.

Below the preview, a manual Q&A form lets the user enter a question and expected answer for the selected dataset. The submit button is disabled until a dataset, question, and expected answer are available. On success, the form clears, the save result is shown, and the preview reloads so the new row is visible when included by the backend preview.

## Frontend Data Flow

`Datasets.tsx` owns these states:

- `datasets`: current dataset inventory.
- `selectedDataset`: the dataset selected by the user or upload success.
- `preview`: preview payload for the selected dataset.
- `uploadName`, `uploadFile`: upload form state.
- `question`, `expectedAnswer`: manual Q&A form state.
- `loadingDatasets`, `loadingPreview`, `uploading`, `savingQa`: request states.
- `error`, `notice`: user-facing request feedback.

Helper functions keep the page readable:

- `loadDatasets()`
- `loadPreview(dataset)`
- `handleUpload(event)`
- `handleSaveManualQa(event)`

`apiClient` gains:

```ts
saveManualQa(targetDataset: string, question: string, expectedAnswer: string)
```

It posts JSON to `/api/datasets/manual-qa` using the backend field names:

- `target_dataset`
- `question`
- `expected_answer`

## Error Handling

All API failures continue to flow through `ApiError`. The page shows a single error line near the relevant workflow area and leaves the existing data visible when possible. Form buttons are disabled while requests are in flight to avoid duplicate writes.

Upload validation happens in the browser before the request:

- A file is required.
- The optional dataset name is passed through unchanged, allowing the backend to normalize it.

Manual Q&A validation happens before the request:

- A dataset must be selected.
- Question and expected answer must contain non-whitespace text.

## Testing

Update `web/src/App.test.tsx` to cover the page workflow through the visible UI:

- The default page loads and lists datasets.
- Selecting a dataset calls preview and displays preview rows.
- Uploading a file calls `/api/datasets/upload`, refreshes datasets, and shows a success message.
- Saving manual Q&A posts the correct JSON payload and shows the appended query id.

Update `web/src/api/client.test.ts` to cover `saveManualQa` request shape and error handling through the shared `requestJson` path.

Existing backend route tests remain unchanged because the backend routes already cover the write behavior.

## Out Of Scope

- Bulk Q&A append.
- AI-generated Q&A candidates.
- Report-chat-to-dataset saving.
- Pagination or full dataset browsing.
- Reworking the overall console navigation or visual system.

## Acceptance Criteria

- A user can upload a dataset from the React `Datasets` page.
- A user can select a dataset and see a preview.
- A user can append one manual Q&A row to the selected dataset.
- Frontend tests pass.
- Frontend build passes.
- Backend tests continue to pass.
