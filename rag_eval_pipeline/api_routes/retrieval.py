"""Retrieval preview routes for external knowledge-base chunks."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rag_eval_pipeline import api
from rag_eval_pipeline.api_models import RetrievalChunksRequest
from rag_eval_pipeline.api_services.state import ApiState

router = APIRouter()


def get_state(request: Request) -> ApiState:
    return request.app.state.rag_eval


def error_response(exc: api.UploadError, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": str(exc)}, status_code=status_code)


@router.post("/api/retrieval/chunks")
def retrieval_chunks(payload: RetrievalChunksRequest, request: Request):
    state = get_state(request)
    try:
        return api.retrieve_chunks_from_question(
            payload.question,
            state.pipeline_config_path,
            page_size=payload.page_size,
            similarity_threshold=payload.similarity_threshold,
            dataset_ids=payload.dataset_ids,
            document_ids=payload.document_ids,
            max_passages=payload.max_passages,
        )
    except api.UploadError as exc:
        return error_response(exc)
