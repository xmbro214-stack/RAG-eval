"""Report and evaluation output routes."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from rag_eval_pipeline import api
from rag_eval_pipeline.api_services.state import ApiState

router = APIRouter()


def get_state(request: Request) -> ApiState:
    return request.app.state.rag_eval


@router.get("/api/reports")
def list_reports(request: Request) -> dict[str, object]:
    state = get_state(request)
    return {"ok": True, "reports": api.list_result_reports(state.reports_root)}


@router.get("/reports/{report_path:path}")
def get_report(report_path: str, request: Request):
    state = get_state(request)
    safe_path = api.safe_report_path(f"/reports/{report_path}", state.reports_root)
    if safe_path is None:
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    return HTMLResponse(api.inject_report_main_button(safe_path.read_text(encoding="utf-8")))


@router.get("/eval-output/{output_path:path}")
def get_eval_output(output_path: str, request: Request):
    state = get_state(request)
    safe_path = api.safe_eval_output_path(f"/eval-output/{output_path}", state.eval_runs_root)
    if safe_path is None:
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    return PlainTextResponse(safe_path.read_text(encoding="utf-8"), media_type="text/csv")


@router.get("/api/eval-runs")
def list_eval_runs(request: Request) -> dict[str, object]:
    state = get_state(request)
    return {"ok": True, "runs": api.list_eval_runs(state.eval_runs_root, state.reports_root)}
