# RAG Evaluation Console Design

## Purpose

Redesign the current `/datasets` page into a compact evaluation console for RAG pipeline work. The page should help users inspect historical reports, review evaluation runs, launch new evaluations, manage uploaded QA datasets, and monitor background jobs without leaving the browser.

This design covers the web UI and lightweight API shape only. It does not change RAGFlow retrieval behavior, evaluation metrics, document chunking, or model prompts.

## Recommended Layout

Use a three-column console layout:

- Left navigation for major work areas.
- Main workspace for the selected section.
- Right job rail for current background job status.

The UI should feel like an operational tool: restrained, dense, readable, and consistent with the existing AT&S blue theme. It should not use a landing-page hero, decorative cards, or large marketing-style sections.

## Navigation

Use these left-side sections:

1. Overview
2. Reports
3. Evaluation Runs
4. Run Evaluation
5. Datasets

Background jobs should not be a normal left navigation page by default. Job status is important enough to stay visible as a persistent right rail. A detailed job view can open from the rail when needed.

## Section Behavior

### Overview

The Overview section summarizes the current state:

- Latest report.
- Latest completed run.
- Active job status.
- Recent evaluation runs.
- Primary action to start a standard evaluation.

This is the default section when opening `/datasets`.

### Reports

Reports replaces the current report block:

- Fetch report list from the existing `GET /api/reports`.
- Show report name, modified time, and path.
- Keep iframe preview for selected HTML report.
- Preserve the existing report-aware chat behavior inside Reports, below or beside the selected report preview.

### Evaluation Runs

Evaluation Runs shows persisted pipeline history from `data/eval_runs/*/manifest.json`.

Each row should include:

- Run name.
- Dataset name.
- Page size.
- Similarity threshold.
- Generation status.
- Evaluation status.
- Generation and evaluation duration.
- Output CSV path.
- Report link when available.

This section needs a new lightweight endpoint, `GET /api/eval-runs`, that scans manifest files and returns normalized run/task records.

### Run Evaluation

Run Evaluation contains launch controls:

- Quick evaluation.
- Standard evaluation.
- Config path display.
- Full/custom mode remains available through the API but is not shown in the first UI pass.
- Dry-run and overwrite controls stay hidden in the first UI pass; quick and standard modes use the existing backend defaults.

Use the existing `POST /api/pipeline/run` endpoint for launch.

### Datasets

Datasets contains the existing CSV/PDF upload workflow:

- Dataset name input.
- CSV/PDF file picker.
- Upload result with saved path.
- Guidance that uploaded files are evaluation QA datasets, not RAGFlow knowledge-base documents.

The existing `POST /api/datasets/upload` endpoint remains the upload path.

## Persistent Job Rail

The right job rail should always show:

- Current job status: idle, starting, running, completed, failed.
- Progress bar.
- Completed task count and total task count.
- Current task label when available.
- Latest log line.
- Tail of recent logs.
- Link to generated report after completion.
- Error detail when failed.
- The first UI pass tracks one active job at a time: the latest job launched from the page.

Current backend job state already supports `GET /api/pipeline/status?run_id=...`, but progress is not explicit enough while a run is active. Implementation should add normalized job progress fields, such as:

```json
{
  "progress": {
    "total_tasks": 8,
    "completed_tasks": 5,
    "current_task": "trd / ps5_sim0p2",
    "percent": 62.5
  }
}
```

The progress can be derived from pipeline task events or from completed manifest entries. Prefer explicit progress updates in the pipeline/job layer over parsing display log text.

## Responsive Behavior

Desktop:

- Left navigation is fixed-width.
- Main workspace fills the center.
- Job rail is fixed-width on the right.

Tablet:

- Left navigation can remain visible if space allows.
- Job rail can collapse into a top or bottom status panel.

Mobile:

- Navigation becomes a compact top segmented control or menu.
- Job rail becomes a collapsible status panel below the section header.
- Tables should become stacked rows or horizontally scroll inside their own container.

## Visual Direction

Use an AT&S-aligned light console theme:

- Deep blue navigation.
- White and pale blue working surfaces.
- Green/yellow/red status accents.
- Compact typography.
- 8px or smaller border radii.
- Clear hover and selected states.

Avoid nested cards. Use panels for major work areas and compact rows for records. Buttons should be explicit commands, with icons added only if the existing frontend stack supports them cleanly.

## Error Handling

The UI should surface:

- Upload validation errors.
- Pipeline start failures.
- Job failures and final error messages.
- Missing reports.
- Missing or malformed manifest files.

Errors should stay inside the relevant section or job rail and should not replace the whole page.

## Testing

Add or update tests for:

- Rendering the navigation labels.
- Rendering the persistent job rail.
- `GET /api/eval-runs` manifest normalization.
- Pipeline status payload including progress.
- Existing upload, report list, and pipeline start behavior.

Manual browser verification should cover:

- Desktop console layout.
- Mobile collapse behavior.
- Starting a quick or mocked evaluation and watching progress update.
- Report selection after a completed job.

## Deferred Product Decisions

- Multi-job queue management beyond the latest active job.
- Exposing full/custom mode, dry-run, and overwrite controls to non-technical users.
- Persisting job records beyond the current API server process lifetime.
