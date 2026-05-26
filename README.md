# RAG Eval Pipeline

Lightweight tooling for running a RAGFlow/OpenAI-compatible evaluation loop:

1. Generate answers from a retrieval endpoint and chat-completions model.
2. Evaluate generated answers against a golden-answer CSV.
3. Compare runs across dataset, page-size, and similarity settings.
4. Render a self-contained HTML report.

This repository was derived from `open-rag-eval`, but the active code path is
now intentionally small and centered on the `rag_eval_pipeline` package.

## Install

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

The runtime dependencies are intentionally minimal: `omegaconf` plus optional
`.env` support through `python-dotenv`. HTTP calls use the Python standard
library.

## Configuration

Use `scripts/eval-cfg.yaml` as the pipeline template. It defines:

- input golden/query CSV paths
- output root and run name
- datasets to compare
- page-size and similarity grids
- retrieval endpoint settings
- OpenAI-compatible chat and embedding settings
- `generation.max_workers` and `evaluation.max_workers` for parallel API calls
- logging level

Secrets can be provided through environment variables referenced by `*_env`
fields in the YAML.

## Run

Dry-run the full grid without calling external services:

```bash
python scripts/run_rag_eval_pipeline.py --config scripts/eval-cfg.yaml --dry-run
```

Run only answer generation:

```bash
python scripts/run_rag_eval_pipeline.py --config scripts/eval-cfg.yaml --stage generate
```

Run only evaluation for existing generated answers:

```bash
python scripts/run_rag_eval_pipeline.py --config scripts/eval-cfg.yaml --stage eval
```

Run the full pipeline:

```bash
python scripts/run_rag_eval_pipeline.py --config scripts/eval-cfg.yaml
```

Outputs are written under:

```text
data/eval_runs/{run_name}/{dataset}/ps{page_size}_sim{similarity}/
  generated_answers.csv
  eval_result.csv
```

A `manifest.json` is also written under `data/eval_runs/{run_name}` for
resume/debugging.

## Visualize

Render a comparison report for a pipeline run:

```bash
python scripts/visualize_local_eval_results.py \
  --mode compare \
  --data-root data \
  --pipeline-run ragflow_grid \
  --output-html reports/local_eval_results_comparison.html
```

The report provides selectors for both `page_size` and `similarity`.

## Logging

Default logs use `INFO`. For HTTP payload summaries and detailed request
diagnostics, use:

```bash
python scripts/run_rag_eval_pipeline.py --config scripts/eval-cfg.yaml --log-level DEBUG
```

API keys are redacted in command logs.

## Tests

```bash
python -m py_compile rag_eval_pipeline/*.py scripts/*.py tests/*.py
python -m pytest
```

## Repository Layout

```text
rag_eval_pipeline/       Core package
scripts/                 Thin CLI wrappers and data-prep helper
tests/                   Tests for current pipeline behavior
data/qa_golden.csv       Small sample golden-answer CSV
```

Generated outputs and experiment results are ignored by git.
