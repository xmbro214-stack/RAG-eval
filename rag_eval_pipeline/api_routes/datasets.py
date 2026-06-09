"""Dataset routes for the FastAPI app."""

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from rag_eval_pipeline import api
from rag_eval_pipeline.api_models import (
    BulkQaRequest,
    GenerateAnswerRequest,
    GenerateQaRequest,
    ManualQaRequest,
    RegenerateAnswerRequest,
)
from rag_eval_pipeline.api_services.state import ApiState

router = APIRouter()


def get_state(request: Request) -> ApiState:
    return request.app.state.rag_eval


def error_response(exc: api.UploadError, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": str(exc)}, status_code=status_code)


def resolve_manual_qa_dataset_path(value: str, state: ApiState):
    handler_class = api.make_handler(
        upload_root=state.upload_root,
        reports_root=state.reports_root,
        eval_runs_root=state.eval_runs_root,
        chat_supplement_csv_path=state.chat_supplement_csv_path,
        pipeline_config_path=state.pipeline_config_path,
        pipeline_jobs=state.jobs(),
    )
    return handler_class.resolve_manual_qa_dataset_path(handler_class, value)


@router.get("/api/datasets")
def list_datasets(request: Request) -> dict[str, object]:
    state = get_state(request)
    return {
        "ok": True,
        "datasets": api.list_datasets(state.upload_root, state.pipeline_config_path),
        "run_options": api.list_run_options(state.pipeline_config_path),
    }


@router.get("/api/datasets/preview")
def preview_dataset(path: str, request: Request):
    state = get_state(request)
    try:
        dataset_path = resolve_manual_qa_dataset_path(path, state)
        return api.dataset_preview_payload(dataset_path)
    except api.UploadError as exc:
        return error_response(exc)


async def multipart_to_saved_dataset(file: UploadFile, name: str, state: ApiState) -> dict[str, object]:
    content = await file.read()
    filename = file.filename or ""
    filename_lower = filename.lower()
    if filename_lower.endswith(".csv"):
        rows = api.parse_and_validate_csv(content)
        source_format = "csv"
    elif filename_lower.endswith(".pdf"):
        rows = api.parse_and_validate_pdf(content)
        source_format = "pdf"
    else:
        raise api.UploadError("Uploaded file must be a .csv or .pdf file")
    return api.save_dataset(rows, name.strip() or filename, upload_root=state.upload_root, source_format=source_format)


@router.post("/api/datasets/upload")
async def upload_dataset(request: Request, file: UploadFile = File(...), name: str = Form("")):
    state = get_state(request)
    try:
        return await multipart_to_saved_dataset(file, name, state)
    except api.UploadError as exc:
        return error_response(exc)


@router.post("/api/datasets/manual-qa")
def manual_qa(payload: ManualQaRequest, request: Request):
    state = get_state(request)
    try:
        dataset_path = resolve_manual_qa_dataset_path(payload.target_dataset, state)
        return api.append_qa_to_dataset(dataset_path, payload.question, payload.expected_answer)
    except api.UploadError as exc:
        return error_response(exc)


@router.post("/api/datasets/bulk-qa")
def bulk_qa(payload: BulkQaRequest, request: Request):
    state = get_state(request)
    try:
        dataset_path = resolve_manual_qa_dataset_path(payload.target_dataset, state)
        return api.append_bulk_qa_to_dataset(dataset_path, payload.rows)
    except api.UploadError as exc:
        return error_response(exc)


@router.post("/api/datasets/generate-qa")
def generate_qa(payload: GenerateQaRequest):
    try:
        return api.generate_qa_candidates(payload.source_text, payload.count, payload.language)
    except api.UploadError as exc:
        return error_response(exc)


@router.post("/api/datasets/generate-answer")
def generate_answer(payload: GenerateAnswerRequest, request: Request):
    state = get_state(request)
    try:
        return api.generate_answer_from_question(payload.question, payload.language, state.pipeline_config_path)
    except api.UploadError as exc:
        return error_response(exc)


@router.post("/api/datasets/regenerate-answer")
def regenerate_answer(payload: RegenerateAnswerRequest, request: Request):
    state = get_state(request)
    try:
        return api.regenerate_answer_from_passages(
            payload.question,
            payload.current_answer,
            payload.passages,
            payload.language,
            state.pipeline_config_path,
        )
    except api.UploadError as exc:
        return error_response(exc)
