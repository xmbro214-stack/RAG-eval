# Datasets List Layout Design

## Goal

Clean up the React `Datasets` page list so long dataset names and paths do not overlap.

## Problem

The current dataset row uses three inline columns for name, path, and row count. In the left list column, long dataset names and paths collide visually and make the page look broken.

## Design

Render each dataset row as a compact two-line item:

- Top line: dataset name on the left and a small rows badge on the right.
- Bottom line: dataset path in muted text with single-line ellipsis truncation.

The selected state remains a blue border with a light blue background. The dataset path is still available through the button accessible label and visible truncated text. Upload, preview, and manual Q&A behavior do not change.

## Verification

- Frontend tests pass.
- Frontend build passes.
- Backend tests continue to pass.
