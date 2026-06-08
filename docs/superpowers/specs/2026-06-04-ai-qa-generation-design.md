# AI-Assisted QA Generation Design

## Purpose

Reduce manual work in the Datasets section by letting users enter one question, generate a draft answer from retrieved passages, review or revise that answer, and then save the confirmed QA row into the target dataset.

This feature improves dataset authoring only. It does not change evaluation metrics, report rendering, or pipeline execution. It reuses the configured retrieval endpoint and chat model so the generated answer is grounded in retrieved context.

## Goals

- Make single-question assisted QA creation the main workflow for users who do not want to write standard answers manually.
- Keep manual single-row QA entry as a fallback.
- Let users inspect retrieved passages before saving.
- Let users review, edit, and regenerate the generated answer before saving.
- Reuse the existing dataset append behavior so saved rows keep the same CSV schema: `query_id`, `query`, `expected_answer`.
- Keep report-chat answers as a secondary source for QA candidates where existing UI already supports it.

## Non-Goals

- No fully automatic writes from model output to golden datasets.
- No batch question upload in the first implementation.
- No multi-row model generation from pasted source text in the first implementation.
- No quality scoring, deduplication engine, version history, or rollback in the first pass.
- No changes to evaluation scoring or pipeline configuration.
- No replacement for uploaded CSV/PDF dataset ingestion.

## Recommended UX

Update the current Add QA workspace so the method selector has two primary modes:

1. Manual
2. Generate Answer

The current report-chat-to-candidate behavior may remain as a secondary helper, but it should not dominate the first-pass dataset authoring workflow.

### Manual Mode

Keep the current behavior:

- Select target dataset.
- Enter one question.
- Enter one standard answer.
- Save QA.

The labels and current dataset preview remain useful and should be preserved.

### Generate Answer Mode

Primary workflow:

1. User selects a target dataset.
2. User enters one question.
3. User clicks Generate Answer.
4. Backend calls the configured retrieval service using the question.
5. Backend sends the question and retrieved passages to the configured chat model.
6. Frontend shows the retrieved passages in a collapsed details panel.
7. Frontend fills the answer textarea with the model answer.
8. User reviews or edits the answer.
9. User clicks Save QA.
10. Backend appends the confirmed question and answer to the selected target dataset.

Controls:

- Generate Answer
- Regenerate Answer
- Save QA
- Clear

Generated answers are never saved until the user explicitly confirms.

### Regenerate Answer

If the generated answer is not acceptable:

1. User clicks Regenerate Answer.
2. Backend receives the same question, current answer, and previous retrieved passages.
3. Backend asks the chat model to improve the answer while staying faithful to the passages.
4. Frontend replaces the answer textarea with the revised answer.
5. User can edit again before saving.

Regeneration should reuse the same retrieved passages by default. This avoids repeating the retrieval call and makes the user's review context stable.

### From Report Chat

Existing report-chat candidate behavior can remain as a secondary helper:

- A report question and answer can still be added to QA candidates.
- It does not write to a dataset automatically.
- The user must still confirm the save in the dataset workflow.

## API Design

### `POST /api/datasets/generate-answer`

Generates one draft answer from a user question and retrieved passages without writing files.

Request:

```json
{
  "question": "What is SM94 M3?",
  "language": "zh"
}
```

Response:

```json
{
  "ok": true,
  "question": "What is SM94 M3?",
  "expected_answer": "SM94 M3 is...",
  "passages": [
    {
      "text": "Retrieved passage text...",
      "score": 0.82,
      "source": "optional source metadata"
    }
  ]
}
```

Validation:

- `question` must not be empty.
- Retrieval URL must be configured and reachable.
- At least one usable passage should be extracted from the retrieval response.
- The generated answer must not be empty.
- If retrieval or model generation fails, return a clear error and do not save anything.

### `POST /api/datasets/regenerate-answer`

Improves one draft answer using the same question and retrieved passages.

Request:

```json
{
  "question": "What is SM94 M3?",
  "current_answer": "Previous draft answer...",
  "passages": [
    {
      "text": "Retrieved passage text..."
    }
  ],
  "language": "zh"
}
```

Response:

```json
{
  "ok": true,
  "expected_answer": "Improved answer..."
}
```

Validation:

- `question` and `current_answer` must not be empty.
- `passages` must contain at least one non-empty passage.
- The regenerated answer must not be empty.

### Existing `POST /api/datasets/manual-qa`

The save action should continue using the existing manual QA append endpoint:

```json
{
  "target_dataset": "data/qa_golden.csv",
  "question": "What is SM94 M3?",
  "expected_answer": "Confirmed answer..."
}
```

The backend keeps generating `query_id` values server-side using the existing `next_query_id` logic.

## Model Prompt Shape

The answer-generation prompt should ask for a single answer grounded only in retrieved passages:

```text
You are helping create a standard answer for a RAG evaluation dataset.
Answer the user's question using only the retrieved passages.
The answer must be factual, specific, and self-contained.
If the passages do not contain enough information, say that the retrieved passages are insufficient.
Return only the answer text.
```

The regeneration prompt should include the previous answer and ask the model to improve it while staying faithful to the same passages.

For Chinese UI language, the generated answer should be Chinese unless the question and retrieved passages are clearly English. The first implementation can keep the output language tied to the current page language.

## Data Flow

Generate Answer:

1. Frontend sends the question to `/api/datasets/generate-answer`.
2. Backend reads `generation.retrieval_url` and related retrieval settings from the current config.
3. Backend calls retrieval with the question.
4. Backend extracts a compact list of passages.
5. Backend calls the chat model with the question and passages.
6. Frontend stores the passages in memory and shows them in a collapsed details panel.
7. Frontend fills the answer textarea.
8. User edits or accepts the answer.
9. Frontend saves through `/api/datasets/manual-qa`.
10. Frontend refreshes dataset inventory and preview.

Regenerate Answer:

1. Frontend sends the question, current answer, and stored passages to `/api/datasets/regenerate-answer`.
2. Backend calls the chat model only.
3. Frontend replaces the answer textarea with the improved answer.
4. Save path remains `/api/datasets/manual-qa`.

## Error Handling

- Empty question: show "Please enter a question."
- Retrieval unavailable: show the configured retrieval URL and ask the user to start or fix the retrieval service.
- No retrieved passages: show "No usable retrieved passages were found."
- Model unavailable: show model connection error without modifying datasets.
- Regenerate without passages: require the user to generate an answer first.
- Save failure: preserve the question, answer, and passages so the user does not lose work.
- Successful save: show saved dataset path and refresh preview.

## Testing Plan

Backend tests:

- Generate Answer rejects empty question.
- Generate Answer calls retrieval with the question.
- Generate Answer extracts passages from a RAGFlow-like retrieval response.
- Generate Answer calls the chat model with the question and passages.
- Generate Answer returns passages and a non-empty answer.
- Generate Answer reports retrieval failures clearly.
- Regenerate Answer rejects missing passages.
- Regenerate Answer calls the chat model with the current answer and passages.
- Manual save still appends one confirmed QA row.

Frontend/static tests:

- Datasets page contains the Generate Answer mode.
- Generate Answer mode contains one question textarea, one answer textarea, generate/regenerate/save controls, and a retrieved passages details panel.
- Generate Answer mode calls `/api/datasets/generate-answer`.
- Regenerate calls `/api/datasets/regenerate-answer`.
- Manual QA still calls the existing single-row endpoint.

Functional smoke:

- Generate one answer with mocked retrieval and chat responses.
- Edit the generated answer.
- Save the confirmed QA row.
- Confirm dataset preview row count increases.

## Recommended First Implementation

Build this in one focused increment:

1. Add backend helpers and endpoints for `generate-answer` and `regenerate-answer`.
2. Replace the current AI-generate mode with the single-question Generate Answer mode.
3. Reuse the existing manual QA save endpoint for the final confirmed write.
4. Keep the retrieved passages panel collapsed by default.
5. Keep report-chat candidate behavior unchanged.

This keeps the main value small enough to test and avoids introducing batch review complexity before the single-question flow is proven.
