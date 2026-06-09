"""Report-aware chat routes for the FastAPI app."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rag_eval_pipeline import api
from rag_eval_pipeline.api_models import ChatRequest, ChatSaveRequest
from rag_eval_pipeline.api_services.state import ApiState

router = APIRouter()


def get_state(request: Request) -> ApiState:
    return request.app.state.rag_eval


def error_response(exc: api.UploadError, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": str(exc)}, status_code=status_code)


@router.post("/api/chat")
def chat(payload: ChatRequest, request: Request):
    state = get_state(request)
    try:
        if not payload.question.strip():
            raise api.UploadError("Chat question is required")
        answer_payload = api.answer_chat_question(
            payload.question,
            payload.report_url,
            state.reports_root,
            payload.language,
        )
        return {"ok": True, **answer_payload, "reports": api.list_result_reports(state.reports_root)}
    except api.UploadError as exc:
        return error_response(exc)


@router.post("/api/chat/save")
def chat_save(payload: ChatSaveRequest, request: Request):
    state = get_state(request)
    try:
        supplement_payload = api.append_chat_supplement(
            payload.question,
            payload.answer,
            state.chat_supplement_csv_path,
        )
        return {"ok": True, **supplement_payload}
    except api.UploadError as exc:
        return error_response(exc)
