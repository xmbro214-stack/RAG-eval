"""Pipeline job routes for the FastAPI app."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rag_eval_pipeline import api
from rag_eval_pipeline.api_models import PipelineCancelRequest, PipelineRunRequest
from rag_eval_pipeline.api_services.state import ApiState

router = APIRouter()


def get_state(request: Request) -> ApiState:
    return request.app.state.rag_eval


def error_response(exc: api.UploadError, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": str(exc)}, status_code=status_code)


@router.get("/api/pipeline/status")
def pipeline_status(run_id: str, request: Request):
    state = get_state(request)
    job = state.jobs().get(run_id)
    if not run_id or job is None:
        return JSONResponse({"ok": False, "error": "Pipeline job not found"}, status_code=404)
    job["progress"] = api.job_progress(job)
    return {"ok": True, "job": job}


@router.post("/api/pipeline/run")
def pipeline_run(payload: PipelineRunRequest, request: Request):
    state = get_state(request)
    try:
        mode = payload.mode.strip().lower()
        config_path = api.pipeline_config_path(payload.config) if payload.config else state.pipeline_config_path
        selected_datasets = api.normalize_selected_datasets(payload.datasets)
        selected_page_sizes = api.normalize_page_sizes(payload.page_sizes)
        selected_similarity_thresholds = api.normalize_similarity_thresholds(payload.similarity_thresholds)
        overwrite = payload.overwrite
        stage = payload.stage
        dry_run = payload.dry_run
        if mode == "quick":
            config_path = api.create_quick_pipeline_config(
                config_path,
                "quick",
                selected_datasets,
                selected_page_sizes,
                selected_similarity_thresholds,
            )
            overwrite = True
        elif mode == "standard":
            config_path = api.create_standard_pipeline_config(
                config_path,
                "standard",
                selected_datasets,
                selected_page_sizes,
                selected_similarity_thresholds,
            )
            overwrite = True
        elif mode not in {"full", "all"}:
            raise api.UploadError("Pipeline mode must be full, standard, or quick")
        if api.stage_runs_generation(stage) and not dry_run:
            retrieval_error = api.check_pipeline_retrieval_service(config_path)
            if retrieval_error:
                raise api.UploadError(retrieval_error)
        return api.start_pipeline_job(
            config_path=config_path,
            stage=stage,
            dry_run=dry_run,
            overwrite=overwrite,
            job_store=state.jobs(),
        )
    except api.UploadError as exc:
        return error_response(exc)


@router.post("/api/pipeline/cancel")
def pipeline_cancel(payload: PipelineCancelRequest, request: Request):
    state = get_state(request)
    try:
        return api.cancel_pipeline_job(payload.run_id, state.jobs())
    except api.UploadError as exc:
        return error_response(exc)
