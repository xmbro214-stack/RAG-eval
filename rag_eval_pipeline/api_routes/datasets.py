"""Dataset routes for the FastAPI app."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rag_eval_pipeline import api
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
