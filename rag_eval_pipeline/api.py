"""Helpers for validating uploaded dataset CSV files."""

import argparse
import csv
import html
import inspect
import io
import json
import logging
import os
import re
import socket
import threading
import traceback
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = ROOT / "data" / "uploaded_datasets"
REPORTS_ROOT = ROOT / "reports"
EVAL_RUNS_ROOT = ROOT / "data" / "eval_runs"
GOLDEN_CSV_PATH = ROOT / "data" / "qa_golden.csv"
CHAT_SUPPLEMENT_CSV_PATH = ROOT / "data" / "qa_supplement.csv"
DEFAULT_PIPELINE_CONFIG = ROOT / "scripts" / "eval-cfg.yaml"
QUICK_PIPELINE_CONFIG_ROOT = ROOT / "data" / "eval_runs" / "_server_configs"
MOCK_OPENAI_BASE_URL = "http://127.0.0.1:8011/v1"
MOCK_OPENAI_API_KEY = "EMPTY"
MOCK_CHAT_MODEL = "mock-chat"
MOCK_EMBEDDING_MODEL = "mock-embedding"
STANDARD_PIPELINE_COMMAND_TIMEOUT_SECONDS = 3600
REQUIRED_COLUMNS = ("query_id", "query", "expected_answer")
LANGUAGE_OPTIONS = (
    ("de", "德语", "German"),
    ("en", "英语", "English"),
    ("zh", "中文", "Chinese"),
    ("ms", "马来西亚语", "Malay"),
)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_JSON_BYTES = 128 * 1024
REPORT_CONTEXT_CHARS = 12000
QA_GENERATION_COUNTS = {5, 10, 20}
PIPELINE_JOBS: dict[str, dict[str, Any]] = {}
PIPELINE_JOBS_LOCK = threading.Lock()
SECRET_ARGUMENT_FLAGS = {
    "--llm-api-key",
    "--retrieval-api-key",
    "--chat-api-key",
    "--embedding-api-key",
}
QA_LABEL_RE = re.compile(
    r"^\s*(?:\d+[.)、]\s*)?(?P<label>q|question|问题|a|answer|答案|回答)\s*[:：]\s*(?P<text>.*)$",
    re.IGNORECASE,
)


class UploadError(ValueError):
    """Raised when an uploaded dataset CSV is invalid."""


@dataclass
class MultipartFile:
    filename: str
    content: bytes
    content_type: str = ""


def split_header_parameters(value: str) -> list[str]:
    parts = []
    current = []
    in_quote = False
    escape_next = False
    for character in value:
        if escape_next:
            current.append(character)
            escape_next = False
            continue
        if character == "\\" and in_quote:
            escape_next = True
            continue
        if character == '"':
            in_quote = not in_quote
            current.append(character)
            continue
        if character == ";" and not in_quote:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(character)
    parts.append("".join(current).strip())
    return parts


def parse_header_parameters(value: str) -> tuple[str, dict[str, str]]:
    parts = split_header_parameters(value)
    main_value = parts[0].lower() if parts else ""
    params = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, raw_param_value = part.split("=", 1)
        param_value = raw_param_value.strip()
        if len(param_value) >= 2 and param_value[0] == '"' and param_value[-1] == '"':
            param_value = param_value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        params[key.strip().lower()] = param_value
    return main_value, params


def parse_content_type_boundary(content_type: str) -> str:
    mime_type, params = parse_header_parameters(content_type)
    if mime_type != "multipart/form-data":
        raise UploadError("Upload must use multipart/form-data")
    boundary = params.get("boundary", "")
    if not boundary:
        raise UploadError("Multipart form-data missing boundary")
    if "\r" in boundary or "\n" in boundary:
        raise UploadError("Multipart form-data boundary is malformed")
    return boundary


def parse_part_headers(header_bytes: bytes) -> dict[str, str]:
    headers = {}
    try:
        header_text = header_bytes.decode("iso-8859-1")
    except UnicodeDecodeError as exc:
        raise UploadError("Multipart part headers are malformed") from exc

    for line in header_text.split("\r\n"):
        if not line:
            continue
        if ":" not in line:
            raise UploadError("Multipart part header is malformed")
        header_name, header_value = line.split(":", 1)
        headers[header_name.strip().lower()] = header_value.strip()
    return headers


def add_form_value(form: dict[str, Any], name: str, value: Any) -> None:
    existing_value = form.get(name)
    if existing_value is None:
        form[name] = value
    elif isinstance(existing_value, list):
        existing_value.append(value)
    else:
        form[name] = [existing_value, value]


def parse_multipart_form(body: bytes, boundary: str) -> dict[str, Any]:
    if not boundary:
        raise UploadError("Multipart form-data missing boundary")

    boundary_bytes = boundary.encode("utf-8")
    delimiter = b"--" + boundary_bytes
    if delimiter not in body:
        raise UploadError("Multipart body missing boundary")

    form: dict[str, Any] = {}
    for index, part in enumerate(body.split(delimiter)):
        if index == 0:
            if part not in (b"", b"\r\n"):
                raise UploadError("Multipart body preamble is malformed")
            continue
        if part.startswith(b"--"):
            trailing = part[2:].strip()
            if trailing:
                raise UploadError("Multipart closing boundary is malformed")
            break
        if not part.startswith(b"\r\n"):
            raise UploadError("Multipart part is malformed")

        part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        if b"\r\n\r\n" not in part:
            raise UploadError("Multipart part missing headers")

        header_bytes, content = part.split(b"\r\n\r\n", 1)
        headers = parse_part_headers(header_bytes)
        disposition = headers.get("content-disposition", "")
        disposition_type, disposition_params = parse_header_parameters(disposition)
        if disposition_type != "form-data":
            raise UploadError("Multipart part missing form-data disposition")

        field_name = disposition_params.get("name", "")
        if not field_name:
            raise UploadError("Multipart part missing field name")

        filename = disposition_params.get("filename")
        if filename is None:
            add_form_value(form, field_name, content.decode("utf-8", errors="replace"))
        else:
            add_form_value(
                form,
                field_name,
                MultipartFile(
                    filename=Path(filename).name,
                    content=content,
                    content_type=headers.get("content-type", ""),
                ),
            )
    return form


def slugify_dataset_name(value: str) -> str:
    name = re.sub(r"\.[^./\\]+$", "", value.strip())
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return slug or "dataset"


def parse_and_validate_csv(content: bytes) -> list[dict[str, str]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UploadError("CSV must be UTF-8 encoded") from exc

    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing_columns:
        raise UploadError(f"CSV missing required columns: {', '.join(missing_columns)}")

    rows: list[dict[str, str]] = []
    for row_number, row in enumerate(reader, start=1):
        query = (row.get("query") or "").strip()
        expected_answer = (row.get("expected_answer") or "").strip()
        for field_name, field_value in (
            ("query", query),
            ("expected_answer", expected_answer),
        ):
            if not field_value:
                raise UploadError(f"CSV row {row_number} missing required value: {field_name}")

        query_id = (row.get("query_id") or "").strip() or f"query_{row_number}"
        rows.append(
            {
                "query_id": query_id,
                "query": query,
                "expected_answer": expected_answer,
            }
        )

    if not rows:
        raise UploadError("CSV must include at least one data row")

    return rows


def extract_text_from_pdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise UploadError("PDF upload requires pypdf; install dependencies from requirements.txt") from exc

    try:
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise UploadError("PDF could not be read as text") from exc

    if not text.strip():
        raise UploadError("PDF text could not be extracted; upload a text-based PDF or CSV")
    return text


def parse_qa_pairs_from_text(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current: dict[str, list[str]] | None = None
    active_field: str | None = None

    def flush_current() -> None:
        nonlocal current
        if current is None:
            return
        query = " ".join(current.get("query", [])).strip()
        expected_answer = " ".join(current.get("expected_answer", [])).strip()
        if not query or not expected_answer:
            raise UploadError("PDF QA pairs must include both question and answer text")
        rows.append(
            {
                "query_id": f"query_{len(rows) + 1}",
                "query": query,
                "expected_answer": expected_answer,
            }
        )
        current = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        label_match = QA_LABEL_RE.match(line)
        if label_match:
            label = label_match.group("label").lower()
            value = label_match.group("text").strip()
            if label in {"q", "question", "问题"}:
                flush_current()
                current = {"query": [], "expected_answer": []}
                active_field = "query"
            else:
                if current is None:
                    raise UploadError("PDF QA pairs must start with a question")
                active_field = "expected_answer"
            if value:
                current[active_field].append(value)
            continue

        if current is not None and active_field is not None:
            current[active_field].append(line)

    flush_current()

    if not rows:
        raise UploadError("PDF text must contain labeled QA pairs")
    return rows


def parse_and_validate_pdf(content: bytes) -> list[dict[str, str]]:
    return parse_qa_pairs_from_text(extract_text_from_pdf(content))


def unique_dataset_path(dataset_name: str, upload_root: Path = UPLOAD_ROOT) -> tuple[str, Path]:
    upload_root.mkdir(parents=True, exist_ok=True)
    base_name = slugify_dataset_name(dataset_name)
    candidate_name = base_name
    candidate_path = upload_root / f"{candidate_name}.csv"
    counter = 2
    while candidate_path.exists():
        candidate_name = f"{base_name}_{counter}"
        candidate_path = upload_root / f"{candidate_name}.csv"
        counter += 1
    return candidate_name, candidate_path


def relative_repo_path(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def pipeline_config_path(value: str | Path | None = None) -> Path:
    if value is None or str(value).strip() == "":
        return DEFAULT_PIPELINE_CONFIG
    path = Path(str(value).strip())
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def first_config_value(values: Any, field_name: str) -> Any:
    if not isinstance(values, list) or not values:
        raise UploadError(f"Quick evaluation requires at least one {field_name}")
    return values[0]


def safe_run_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "run"


def pipeline_data_path(value: Any) -> Path:
    path = Path(str(value or "").strip())
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def read_csv_rows_with_fields(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        return fieldnames, [dict(row) for row in reader]


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def limit_pipeline_qa(config: dict[str, Any], run_id: str, limit: int) -> None:
    queries_path = pipeline_data_path(config.get("queries_csv"))
    query_fields, query_rows = read_csv_rows_with_fields(queries_path)
    if not query_rows:
        raise UploadError("Quick evaluation requires at least one QA row")
    selected_rows = query_rows[:limit]
    selected_ids = {str(row.get("query_id") or "") for row in selected_rows}
    limited_queries = QUICK_PIPELINE_CONFIG_ROOT / f"{safe_run_id(run_id)}_qa.csv"
    write_csv_rows(limited_queries, query_fields, selected_rows)
    config["queries_csv"] = str(limited_queries)

    golden_path = pipeline_data_path(config.get("golden_csv") or config.get("queries_csv"))
    if golden_path == queries_path:
        config["golden_csv"] = str(limited_queries)
        return

    golden_fields, golden_rows = read_csv_rows_with_fields(golden_path)
    selected_golden = [
        row
        for row in golden_rows
        if str(row.get("query_id") or "") in selected_ids
    ]
    limited_golden = QUICK_PIPELINE_CONFIG_ROOT / f"{safe_run_id(run_id)}_golden.csv"
    write_csv_rows(limited_golden, golden_fields, selected_golden)
    config["golden_csv"] = str(limited_golden)


def apply_server_pipeline_overrides(config: dict[str, Any], *, run_name: str) -> None:
    output = dict(config.get("output") or {})
    output["run_name"] = run_name
    output["overwrite"] = True
    output["command_timeout_seconds"] = 300
    config["output"] = output

    grid = config.get("grid") or {}
    config["datasets"] = [first_config_value(config.get("datasets"), "dataset")]
    config["grid"] = {
        "page_sizes": [first_config_value(grid.get("page_sizes"), "page_size")],
        "similarity_thresholds": [first_config_value(grid.get("similarity_thresholds"), "similarity")],
    }

    generation = dict(config.get("generation") or {})
    generation["llm_base_url"] = MOCK_OPENAI_BASE_URL
    generation["llm_api_key"] = MOCK_OPENAI_API_KEY
    generation["llm_model"] = MOCK_CHAT_MODEL
    generation["repeat_query"] = 1
    generation["max_workers"] = 1
    generation["timeout"] = 30
    generation["retries"] = 0
    generation["max_tokens"] = 1024
    config["generation"] = generation

    evaluation = dict(config.get("evaluation") or {})
    evaluation["chat_base_url"] = MOCK_OPENAI_BASE_URL
    evaluation["chat_api_key"] = MOCK_OPENAI_API_KEY
    evaluation["chat_model"] = MOCK_CHAT_MODEL
    evaluation["embedding_base_url"] = MOCK_OPENAI_BASE_URL
    evaluation["embedding_api_key"] = MOCK_OPENAI_API_KEY
    evaluation["embedding_model"] = MOCK_EMBEDDING_MODEL
    evaluation["k_values"] = "1"
    evaluation["max_workers"] = 1
    evaluation["timeout"] = 30
    evaluation["retries"] = 0
    config["evaluation"] = evaluation


def normalize_selected_datasets(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    normalized: list[int] = []
    for item in value:
        try:
            normalized.append(int(item))
        except (TypeError, ValueError):
            continue
    return normalized


def normalize_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    normalized: list[float] = []
    for item in value:
        try:
            normalized.append(float(item))
        except (TypeError, ValueError):
            continue
    return normalized


def normalize_page_sizes(value: Any) -> list[int]:
    normalized = normalize_int_list(value)
    for item in normalized:
        if item < 1 or item > 20:
            raise UploadError("Page size must be between 1 and 20")
    return normalized


def normalize_similarity_thresholds(value: Any) -> list[float]:
    normalized = normalize_float_list(value)
    for item in normalized:
        if item < 0 or item > 0.5:
            raise UploadError("Similarity must be between 0 and 0.5")
    return normalized


def filter_config_datasets(config: dict[str, Any], selected_dataset_names: list[str] | None = None) -> None:
    if not selected_dataset_names:
        return
    selected = set(selected_dataset_names)
    datasets = [
        dataset
        for dataset in (config.get("datasets") or [])
        if isinstance(dataset, dict) and str(dataset.get("name") or dataset.get("id") or "") in selected
    ]
    if not datasets:
        raise UploadError("Select at least one configured dataset")
    config["datasets"] = datasets


def filter_config_grid(
    config: dict[str, Any],
    selected_page_sizes: list[int] | None = None,
    selected_similarity_thresholds: list[float] | None = None,
) -> None:
    if not selected_page_sizes and not selected_similarity_thresholds:
        return
    grid = dict(config.get("grid") or {})
    if selected_page_sizes:
        grid["page_sizes"] = list(selected_page_sizes)
    if selected_similarity_thresholds:
        grid["similarity_thresholds"] = list(selected_similarity_thresholds)
    config["grid"] = grid


def write_pipeline_config(config: dict[str, Any], prefix: str, run_id: str) -> Path:
    QUICK_PIPELINE_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    config_path = QUICK_PIPELINE_CONFIG_ROOT / f"{prefix}_{safe_run_id(run_id)}.yaml"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path


def stage_runs_generation(stage: str) -> bool:
    return stage.strip().lower() in {"", "all", "generate"}


def check_pipeline_retrieval_service(config_path: Path, timeout_seconds: float = 2.0) -> str | None:
    if not config_path.exists():
        return None
    from rag_eval_pipeline.config import load_config

    config = load_config(config_path)
    retrieval_url = str((config.get("generation") or {}).get("retrieval_url") or "").strip()
    if not retrieval_url:
        return "召回服务地址为空：请在配置文件中填写 generation.retrieval_url。"
    parsed = urlparse(retrieval_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return f"召回服务地址格式不正确：{retrieval_url}"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout_seconds):
            return None
    except OSError as exc:
        return (
            f"召回服务不可达：{retrieval_url}。"
            f"请启动 retrieval 服务，或修改 scripts/eval-cfg.yaml 中的 generation.retrieval_url。"
            f"底层错误：{exc}"
        )


def create_quick_pipeline_config(
    base_config_path: Path,
    run_id: str,
    selected_dataset_names: list[str] | None = None,
    selected_page_sizes: list[int] | None = None,
    selected_similarity_thresholds: list[float] | None = None,
) -> Path:
    from rag_eval_pipeline.config import load_config

    base_config = load_config(base_config_path)
    quick_config = json.loads(json.dumps(base_config))
    filter_config_datasets(quick_config, selected_dataset_names)
    filter_config_grid(quick_config, selected_page_sizes, selected_similarity_thresholds)
    apply_server_pipeline_overrides(quick_config, run_name="ragflow_quick")
    limit_pipeline_qa(quick_config, run_id, 10)
    return write_pipeline_config(quick_config, "quick", run_id)


def apply_pipeline_output_overrides(config: dict[str, Any], *, run_name: str) -> None:
    output = dict(config.get("output") or {})
    output["run_name"] = run_name
    output["overwrite"] = True
    output["command_timeout_seconds"] = STANDARD_PIPELINE_COMMAND_TIMEOUT_SECONDS
    config["output"] = output


def create_standard_pipeline_config(
    base_config_path: Path,
    run_id: str,
    selected_dataset_names: list[str] | None = None,
    selected_page_sizes: list[int] | None = None,
    selected_similarity_thresholds: list[float] | None = None,
) -> Path:
    from rag_eval_pipeline.config import load_config

    base_config = load_config(base_config_path)
    standard_config = json.loads(json.dumps(base_config))
    filter_config_datasets(standard_config, selected_dataset_names)
    filter_config_grid(standard_config, selected_page_sizes, selected_similarity_thresholds)
    apply_pipeline_output_overrides(standard_config, run_name="ragflow_standard")
    return write_pipeline_config(standard_config, "standard", run_id)


def pipeline_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def redact_secret_arguments(text: str) -> str:
    redacted = text
    for flag in SECRET_ARGUMENT_FLAGS:
        redacted = re.sub(
            rf"({re.escape(flag)}['\"]?,?\s+['\"]?)([^'\",\]\s]+)",
            rf"\1<redacted>",
            redacted,
        )
    return redacted


def run_pipeline_from_config(
    config_path: Path,
    stage: str,
    dry_run: bool,
    overwrite: bool,
    log_handler: logging.Handler | None = None,
    cancel_check=None,
) -> list[dict[str, Any]]:
    from rag_eval_pipeline.pipeline import configure_logging, load_config, run_pipeline

    config = load_config(str(config_path))
    configured_level = str((config.get("logging") or {}).get("level") or os.getenv("RAG_EVAL_LOG_LEVEL") or "INFO")
    configure_logging(configured_level)
    if log_handler is not None:
        logging.getLogger().addHandler(log_handler)
    try:
        return run_pipeline(config, stage=stage, dry_run=dry_run, overwrite=overwrite, cancel_check=cancel_check)
    finally:
        if log_handler is not None:
            logging.getLogger().removeHandler(log_handler)


def pipeline_run_name_from_eval_csv(eval_csv: Path) -> str:
    try:
        return eval_csv.parents[2].name
    except IndexError:
        return eval_csv.parent.name or "pipeline"


def render_pipeline_reports(
    entries: list[dict[str, Any]],
    *,
    reports_root: Path = REPORTS_ROOT,
) -> list[dict[str, Any]]:
    from rag_eval_pipeline.visualization import read_rows, render_html

    grouped_eval_csvs: dict[str, list[Path]] = {}
    for entry in entries:
        if entry.get("evaluation_status") != "completed":
            continue
        eval_result_value = entry.get("eval_result_csv")
        if not eval_result_value:
            continue
        eval_csv = Path(str(eval_result_value))
        if not eval_csv.exists():
            continue
        grouped_eval_csvs.setdefault(pipeline_run_name_from_eval_csv(eval_csv), []).append(eval_csv)

    rendered_reports: list[dict[str, Any]] = []
    reports_root.mkdir(parents=True, exist_ok=True)
    for run_name, eval_csvs in grouped_eval_csvs.items():
        rows: list[dict[str, str]] = []
        for eval_csv in eval_csvs:
            rows.extend(read_rows(str(eval_csv)))
        if not rows:
            continue

        report_name = f"{safe_run_id(run_name)}_eval.html"
        report_path = reports_root / report_name
        input_label = " | ".join(relative_repo_path(path) for path in eval_csvs)
        report_path.write_text(render_html(rows, input_label), encoding="utf-8")
        rendered_reports.append(
            {
                "name": report_name,
                "path": relative_repo_path(report_path),
                "url": f"/reports/{quote(report_name)}",
                "eval_result_csv": relative_repo_path(eval_csvs[0]),
                "eval_result_csvs": [relative_repo_path(path) for path in eval_csvs],
            }
        )
    return rendered_reports


TERMINAL_STAGE_STATUSES = {
    "completed",
    "dry_run",
    "skipped_existing",
    "missing_generated_answers",
    "failed",
    "not_requested",
}


def entry_is_finished(entry: dict[str, Any]) -> bool:
    evaluation_status = str(entry.get("evaluation_status") or "")
    if evaluation_status in {"completed", "dry_run", "skipped_existing", "failed", "missing_generated_answers"}:
        return True
    return False


def combo_from_entry(entry: dict[str, Any]) -> str:
    output_dir = str(entry.get("output_dir") or "")
    if output_dir:
        return Path(output_dir).name
    page_size = entry.get("page_size")
    similarity = entry.get("similarity_threshold")
    if page_size not in (None, "") and similarity not in (None, ""):
        similarity_text = str(similarity).replace("-", "neg").replace(".", "p")
        return f"ps{page_size}_sim{similarity_text}"
    return ""


def parse_task_log(message: str) -> dict[str, Any]:
    match = re.search(
        r"\[(?P<index>\d+)/(?P<total>\d+)\]\s+Task dataset=(?P<dataset>[^\s]+).*?output_dir=(?P<output_dir>\S+)",
        message or "",
    )
    if not match:
        return {}
    combo = Path(match.group("output_dir")).name
    return {
        "index": int(match.group("index")),
        "total": int(match.group("total")),
        "dataset_name": match.group("dataset"),
        "combo": combo,
        "current_task": f"{match.group('dataset')} / {combo}" if combo else match.group("dataset"),
    }


def job_progress(job: dict[str, Any]) -> dict[str, Any]:
    entries = list(job.get("entries") or [])
    log_task = parse_task_log(str(job.get("last_log") or ""))
    total_tasks = int(job.get("expected_tasks") or log_task.get("total") or len(entries) or 0)
    completed_tasks = sum(1 for entry in entries if entry_is_finished(entry))
    if not completed_tasks and log_task.get("index"):
        completed_tasks = max(int(log_task["index"]) - 1, 0)
    if str(job.get("status") or "") == "completed" and total_tasks:
        completed_tasks = total_tasks
    completed_tasks = min(completed_tasks, total_tasks) if total_tasks else completed_tasks
    current_task = str(job.get("current_task") or log_task.get("current_task") or "")
    if log_task.get("dataset_name") and (not log_task.get("combo") or not str(log_task.get("combo")).startswith("ps")):
        matching_entry = next(
            (entry for entry in reversed(entries) if str(entry.get("dataset_name") or "") == log_task["dataset_name"]),
            entries[-1] if entries else {},
        )
        combo = combo_from_entry(matching_entry)
        current_task = f"{log_task['dataset_name']} / {combo}" if combo else str(log_task["dataset_name"])
    percent = round((completed_tasks / total_tasks) * 100, 1) if total_tasks else 0.0
    return {
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "percent": percent,
        "current_task": current_task,
    }


def expected_pipeline_tasks(config_path: Path) -> int:
    try:
        from rag_eval_pipeline.pipeline import expand_tasks, load_config

        return len(expand_tasks(load_config(config_path)))
    except Exception:  # noqa: BLE001 - progress should not block job startup.
        return 0


def eval_output_url(output_path: Path) -> str:
    return f"/eval-output?path={quote(relative_repo_path(output_path), safe='')}"


def count_csv_rows(csv_path: Path) -> int:
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for row in reader if any(str(cell).strip() for cell in row))
    except OSError:
        return 0


def list_datasets(
    upload_root: Path = UPLOAD_ROOT,
    config_path: Path = DEFAULT_PIPELINE_CONFIG,
) -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    try:
        from rag_eval_pipeline.config import load_config

        config = load_config(config_path)
    except Exception:  # noqa: BLE001 - dataset inventory should still show uploaded files.
        config = {}

    config_csv_value = config.get("queries_csv") or config.get("golden_csv") or ""
    config_csv_path = Path(str(config_csv_value)).expanduser() if str(config_csv_value).strip() else config_path
    if not config_csv_path.is_absolute():
        config_csv_path = (ROOT / config_csv_path).resolve()
    config_rows: int | str = count_csv_rows(config_csv_path) if config_csv_path.suffix.lower() == ".csv" else ""

    for item in config.get("datasets") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("id") or "").strip()
        if not name:
            continue
        dataset_csv_value = item.get("queries_csv") or item.get("golden_csv") or config_csv_value
        dataset_csv_path = Path(str(dataset_csv_value)).expanduser() if str(dataset_csv_value).strip() else config_csv_path
        if not dataset_csv_path.is_absolute():
            dataset_csv_path = (ROOT / dataset_csv_path).resolve()
        dataset_rows: int | str = count_csv_rows(dataset_csv_path) if dataset_csv_path.suffix.lower() == ".csv" else config_rows
        datasets.append(
            {
                "name": name,
                "source": "config",
                "dataset_id": str(item.get("id") or ""),
                "rows": dataset_rows,
                "path": relative_repo_path(dataset_csv_path),
                "modified": "",
            }
        )

    default_golden_path = GOLDEN_CSV_PATH.resolve()
    should_offer_default_golden = config_path.resolve() == DEFAULT_PIPELINE_CONFIG.resolve()
    default_golden_relative_path = relative_repo_path(default_golden_path)
    if should_offer_default_golden and default_golden_path.is_file():
        datasets.append(
            {
                "name": default_golden_path.stem,
                "source": "config",
                "dataset_id": "",
                "rows": count_csv_rows(default_golden_path),
                "path": default_golden_relative_path,
                "modified": "",
                "runnable": False,
            }
        )

    if upload_root.exists():
        for csv_path in sorted(upload_root.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True):
            datasets.append(
                {
                    "name": csv_path.stem,
                    "source": "uploaded",
                    "dataset_id": "",
                    "rows": count_csv_rows(csv_path),
                    "path": relative_repo_path(csv_path),
                    "modified": datetime.fromtimestamp(csv_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                }
            )
    return datasets


def list_run_options(config_path: Path = DEFAULT_PIPELINE_CONFIG) -> dict[str, list[Any]]:
    try:
        from rag_eval_pipeline.config import load_config

        config = load_config(config_path)
    except Exception:  # noqa: BLE001 - UI can render empty choices.
        config = {}
    grid = config.get("grid") or {}
    return {
        "page_sizes": list(grid.get("page_sizes") or []),
        "similarity_thresholds": list(grid.get("similarity_thresholds") or []),
    }


def list_eval_runs(eval_runs_root: Path = EVAL_RUNS_ROOT, reports_root: Path = REPORTS_ROOT) -> list[dict[str, Any]]:
    if not eval_runs_root.exists():
        return []

    runs: list[dict[str, Any]] = []
    for manifest_path in sorted(eval_runs_root.glob("*/manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tasks = manifest.get("tasks")
        if not isinstance(tasks, list):
            continue
        run_name = manifest_path.parent.name
        updated_at = str(manifest.get("updated_at") or "")
        report_path = reports_root / f"{safe_run_id(run_name)}_eval.html"
        if report_path.is_file():
            report_modified = int(report_path.stat().st_mtime)
            report_url = f"/reports/{quote(report_path.name)}?v={report_modified}"
            report_relative_path = relative_repo_path(report_path)
        else:
            report_url = ""
            report_relative_path = ""
        for task in tasks:
            if not isinstance(task, dict):
                continue
            eval_csv_value = str(task.get("eval_result_csv") or "")
            eval_csv = Path(eval_csv_value) if eval_csv_value else None
            eval_csv_path = relative_repo_path(eval_csv) if eval_csv is not None else ""
            runs.append(
                {
                    "run_name": run_name,
                    "updated_at": updated_at,
                    "dataset_name": str(task.get("dataset_name") or ""),
                    "page_size": task.get("page_size"),
                    "similarity_threshold": task.get("similarity_threshold"),
                    "generation_status": str(task.get("generation_status") or ""),
                    "evaluation_status": str(task.get("evaluation_status") or ""),
                    "generation_seconds": task.get("generation_seconds"),
                    "evaluation_seconds": task.get("evaluation_seconds"),
                    "eval_result_csv": eval_csv_path,
                    "eval_result_url": eval_output_url(eval_csv) if eval_csv is not None else "",
                    "report_path": report_relative_path,
                    "report_url": report_url,
                    "combo": combo_from_entry(task),
                }
            )
    return runs


class PipelineJobLogHandler(logging.Handler):
    def __init__(self, job: dict[str, Any], max_lines: int = 40) -> None:
        super().__init__(level=logging.INFO)
        self.job = job
        self.max_lines = max_lines

    def emit(self, record: logging.LogRecord) -> None:
        message = redact_secret_arguments(record.getMessage())
        tail = self.job.setdefault("log_tail", [])
        tail.append(message)
        del tail[:-self.max_lines]
        self.job["last_log"] = message


def start_pipeline_job(
    *,
    config_path: Path,
    stage: str,
    dry_run: bool,
    overwrite: bool,
    job_store: dict[str, dict[str, Any]] | None = None,
    runner=run_pipeline_from_config,
    report_renderer=render_pipeline_reports,
    thread_factory=threading.Thread,
) -> dict[str, Any]:
    store = PIPELINE_JOBS if job_store is None else job_store
    run_id = uuid.uuid4().hex
    job = {
        "run_id": run_id,
        "status": "running",
        "config": relative_repo_path(config_path),
        "stage": stage,
        "dry_run": dry_run,
        "overwrite": overwrite,
        "expected_tasks": expected_pipeline_tasks(config_path),
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": "",
        "entries": [],
        "reports": [],
        "error": "",
        "last_log": "",
        "log_tail": [],
        "cancel_requested": False,
        "thread_alive": True,
    }
    store[run_id] = job

    def cancel_check() -> bool:
        return bool(job.get("cancel_requested"))

    def run_job() -> None:
        log_handler = PipelineJobLogHandler(job)
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        if root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)
        root_logger.addHandler(log_handler)
        try:
            runner_parameters = inspect.signature(runner).parameters
            runner_kwargs = {}
            if "log_handler" in runner_parameters:
                runner_kwargs["log_handler"] = log_handler
            if "cancel_check" in runner_parameters:
                runner_kwargs["cancel_check"] = cancel_check
            entries = runner(config_path, stage, dry_run, overwrite, **runner_kwargs)
            reports = report_renderer(entries)
        except Exception as exc:  # noqa: BLE001 - surface pipeline failures in job status.
            if job.get("cancel_requested"):
                job["status"] = "cancelled"
                job["error"] = "Cancelled by user."
                job["traceback"] = ""
            else:
                job["status"] = "failed"
                job["error"] = redact_secret_arguments(str(exc))
                job["traceback"] = redact_secret_arguments(traceback.format_exc(limit=8))
        else:
            job["status"] = "cancelled" if job.get("cancel_requested") else "completed"
            if job["status"] == "cancelled":
                job["error"] = "Cancelled by user."
            job["entries"] = entries
            job["reports"] = reports
        finally:
            job["progress"] = job_progress(job)
            root_logger.removeHandler(log_handler)
            root_logger.setLevel(previous_level)
            job["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            job["thread_alive"] = False

    thread = thread_factory(target=run_job, daemon=True)
    thread.start()
    job["progress"] = job_progress(job)
    return {"ok": True, "run_id": run_id, "status": job["status"], "job": job}

def cancel_pipeline_job(run_id: str, job_store: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    clean_run_id = str(run_id or "").strip()
    if not clean_run_id:
        raise UploadError("Pipeline run ID is required")
    store = PIPELINE_JOBS if job_store is None else job_store
    job = store.get(clean_run_id)
    if job is None:
        raise UploadError("Pipeline job not found")
    job["cancel_requested"] = True
    if job.get("status") in {"starting", "running", "cancelling"}:
        job["status"] = "cancelling"
    thread_alive = job.get("thread_alive", job.get("status") in {"starting", "running", "cancelling"})
    if not thread_alive and job.get("status") == "cancelling":
        job["status"] = "cancelled"
        job["error"] = "Cancelled by user."
    job["progress"] = job_progress(job)
    return {"ok": True, "run_id": clean_run_id, "status": job.get("status", ""), "job": job}


def save_dataset(
    rows: list[dict[str, str]],
    dataset_name: str,
    *,
    upload_root: Path = UPLOAD_ROOT,
    source_format: str = "csv",
) -> dict[str, Any]:
    while True:
        final_name, output_path = unique_dataset_path(dataset_name, upload_root)
        try:
            with output_path.open("x", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(REQUIRED_COLUMNS))
                writer.writeheader()
                writer.writerows(rows)
            break
        except FileExistsError:
            continue
    return {
        "ok": True,
        "dataset_name": final_name,
        "path": relative_repo_path(output_path),
        "rows": len(rows),
        "columns": list(REQUIRED_COLUMNS),
        "source_format": source_format,
    }


def next_query_id(rows: list[dict[str, str]]) -> str:
    highest = 0
    for row in rows:
        match = re.fullmatch(r"query_(\d+)", (row.get("query_id") or "").strip())
        if match:
            highest = max(highest, int(match.group(1)))
    return f"query_{highest + 1}"


def append_chat_supplement(
    question: str,
    answer: str,
    supplement_csv_path: Path = CHAT_SUPPLEMENT_CSV_PATH,
) -> dict[str, str]:
    clean_question = question.strip()
    clean_answer = answer.strip()
    if not clean_question:
        raise UploadError("Chat question is required")
    if not clean_answer:
        raise UploadError("Chat answer is required")

    supplement_csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if supplement_csv_path.exists() and supplement_csv_path.stat().st_size > 0:
        with supplement_csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames or []
            missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
            if missing_columns:
                raise UploadError(f"Chat supplement CSV missing required columns: {', '.join(missing_columns)}")
            rows = list(reader)

    query_id = next_query_id(rows)
    write_header = not supplement_csv_path.exists() or supplement_csv_path.stat().st_size == 0
    with supplement_csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(REQUIRED_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "query_id": query_id,
                "query": clean_question,
                "expected_answer": clean_answer,
            }
        )

    return {
        "supplemental_path": relative_repo_path(supplement_csv_path),
        "supplemental_query_id": query_id,
    }


def append_qa_to_dataset(dataset_csv_path: Path, question: str, expected_answer: str) -> dict[str, Any]:
    clean_question = question.strip()
    clean_answer = expected_answer.strip()
    if not clean_question:
        raise UploadError("Manual QA question is required")
    if not clean_answer:
        raise UploadError("Manual QA expected answer is required")
    if dataset_csv_path.suffix.lower() != ".csv":
        raise UploadError("Target dataset must be a CSV file")
    if not dataset_csv_path.exists() or not dataset_csv_path.is_file():
        raise UploadError("Target dataset was not found")

    with dataset_csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            raise UploadError(f"Target dataset CSV missing required columns: {', '.join(missing_columns)}")
        rows = list(reader)

    query_id = next_query_id(rows)
    with dataset_csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(REQUIRED_COLUMNS))
        writer.writerow(
            {
                "query_id": query_id,
                "query": clean_question,
                "expected_answer": clean_answer,
            }
        )

    return {
        "ok": True,
        "dataset_name": dataset_csv_path.stem,
        "path": relative_repo_path(dataset_csv_path),
        "rows": count_csv_rows(dataset_csv_path),
        "columns": list(REQUIRED_COLUMNS),
        "source_format": "manual",
        "appended_query_id": query_id,
    }


def read_dataset_rows(dataset_csv_path: Path) -> list[dict[str, str]]:
    if dataset_csv_path.suffix.lower() != ".csv":
        raise UploadError("Target dataset must be a CSV file")
    if not dataset_csv_path.exists() or not dataset_csv_path.is_file():
        raise UploadError("Target dataset was not found")

    with dataset_csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            raise UploadError(f"Target dataset CSV missing required columns: {', '.join(missing_columns)}")
        return list(reader)


def append_bulk_qa_to_dataset(dataset_csv_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise UploadError("Save requires at least one QA row")

    existing_rows = read_dataset_rows(dataset_csv_path)
    next_id_number = 1
    for row in existing_rows:
        match = re.fullmatch(r"query_(\d+)", (row.get("query_id") or "").strip())
        if match:
            next_id_number = max(next_id_number, int(match.group(1)) + 1)

    clean_rows: list[dict[str, str]] = []
    appended_query_ids: list[str] = []
    for item in rows:
        question = str(item.get("question", "")).strip()
        expected_answer = str(item.get("expected_answer", "")).strip()
        if not question or not expected_answer:
            raise UploadError("Each QA row requires a question and standard answer")
        query_id = f"query_{next_id_number}"
        next_id_number += 1
        appended_query_ids.append(query_id)
        clean_rows.append({
            "query_id": query_id,
            "query": question,
            "expected_answer": expected_answer,
        })

    with dataset_csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(REQUIRED_COLUMNS))
        if dataset_csv_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(clean_rows)

    return {
        "ok": True,
        "dataset_name": dataset_csv_path.stem,
        "path": relative_repo_path(dataset_csv_path),
        "rows": len(existing_rows) + len(clean_rows),
        "saved": len(clean_rows),
        "appended_query_ids": appended_query_ids,
        "columns": list(REQUIRED_COLUMNS),
        "source_format": "manual_bulk",
    }


def dataset_preview_payload(dataset_csv_path: Path, limit: int = 20) -> dict[str, Any]:
    if dataset_csv_path.suffix.lower() != ".csv":
        raise UploadError("Target dataset must be a CSV file")
    if not dataset_csv_path.exists() or not dataset_csv_path.is_file():
        raise UploadError("Target dataset was not found")
    with dataset_csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            raise UploadError(f"Target dataset CSV missing required columns: {', '.join(missing_columns)}")
        rows = [
            {column: str(row.get(column) or "") for column in REQUIRED_COLUMNS}
            for row in reader
        ]
    return {
        "ok": True,
        "path": relative_repo_path(dataset_csv_path),
        "rows": len(rows),
        "preview_rows": rows[:limit],
    }


def list_result_reports(reports_root: Path = REPORTS_ROOT) -> list[dict[str, str]]:
    if not reports_root.exists():
        return []

    reports = []
    for report_path in reports_root.glob("*.html"):
        if not report_path.is_file():
            continue
        modified_timestamp = report_path.stat().st_mtime
        reports.append(
            {
                "name": report_path.name,
                "path": relative_repo_path(report_path),
                "url": f"/reports/{quote(report_path.name)}?v={int(modified_timestamp)}",
                "modified": datetime.fromtimestamp(modified_timestamp).strftime("%Y-%m-%d %H:%M"),
            }
        )
    return sorted(reports, key=lambda item: (item["modified"], item["name"]), reverse=True)


def render_report_options(result_reports: list[dict[str, str]]) -> str:
    if not result_reports:
        return """
        <div class="report-workspace">
          <div class="result report-empty">No HTML reports found under reports/.</div>
          <div id="reportPath" hidden data-report-path=""></div>
          <div class="report-body-grid">
            <div class="report-preview-area">
              <iframe id="reportFrame" class="report-frame" title="Evaluation report preview" scrolling="auto" hidden></iframe>
            </div>
            <div class="report-chat-area">{{CHAT_PANEL}}</div>
          </div>
        </div>"""

    options = "\n".join(
        (
            f'<option value="{html.escape(report["url"], quote=True)}" '
            f'data-path="{html.escape(report["path"], quote=True)}">'
            f'{html.escape(report["name"])} - {html.escape(report["modified"])}</option>'
        )
        for report in result_reports
    )
    first_report = result_reports[0]
    return f"""
        <div class="report-workspace">
          <div class="report-control-strip">
            <label class="report-picker">
              <span data-i18n="reportFile">Report file</span>
              <select id="reportSelect">
                {options}
              </select>
            </label>
            <div id="reportPath" hidden data-report-path="{html.escape(first_report["path"], quote=True)}"></div>
          </div>
          <div class="report-body-grid">
            <div class="report-preview-area">
              <iframe id="reportFrame" class="report-frame" title="Evaluation report preview" src="{html.escape(first_report["url"], quote=True)}" scrolling="auto"></iframe>
            </div>
            <div class="report-chat-area">{{CHAT_PANEL}}</div>
          </div>
        </div>"""


def safe_report_path(request_path: str, reports_root: Path = REPORTS_ROOT) -> Path | None:
    parsed_path = urlparse(request_path).path
    if not parsed_path.startswith("/reports/"):
        return None

    report_name = unquote(parsed_path.removeprefix("/reports/"))
    if (
        not report_name
        or "/" in report_name
        or "\\" in report_name
        or Path(report_name).name != report_name
        or Path(report_name).suffix.lower() != ".html"
    ):
        return None

    report_path = (reports_root / report_name).resolve()
    try:
        report_path.relative_to(reports_root.resolve())
    except ValueError:
        return None
    if not report_path.is_file():
        return None
    return report_path


def inject_report_main_button(report_html: str) -> str:
    return_markup = """
<style>
  .rag-eval-main-return {
    position: fixed;
    right: 18px;
    bottom: 18px;
    z-index: 2147483647;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 38px;
    border-radius: 6px;
    padding: 0 14px;
    color: #ffffff;
    background: #176f95;
    box-shadow: 0 12px 30px rgba(6, 40, 70, 0.24);
    font: 700 14px/1.2 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    text-decoration: none;
  }
  .rag-eval-main-return:hover { background: #0b4f6c; }
</style>
<a class="rag-eval-main-return" href="/datasets">返回主界面</a>
"""
    if "rag-eval-main-return" in report_html:
        return report_html
    if re.search(r"</body\s*>", report_html, flags=re.IGNORECASE):
        return re.sub(r"</body\s*>", return_markup + "\n</body>", report_html, count=1, flags=re.IGNORECASE)
    if re.search(r"</html\s*>", report_html, flags=re.IGNORECASE):
        return re.sub(r"</html\s*>", return_markup + "\n</html>", report_html, count=1, flags=re.IGNORECASE)
    return report_html + return_markup


def safe_eval_output_path(request_path: str, eval_runs_root: Path = EVAL_RUNS_ROOT) -> Path | None:
    requested_path = (parse_qs(urlparse(request_path).query).get("path") or [""])[0].strip()
    if not requested_path:
        return None
    candidate = Path(requested_path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    try:
        output_path = candidate.resolve()
        output_path.relative_to(eval_runs_root.resolve())
    except (OSError, ValueError):
        return None
    if output_path.suffix.lower() not in {".csv", ".json", ".txt", ".log"}:
        return None
    return output_path if output_path.is_file() else None


def strip_html_to_text(html_text: str) -> str:
    without_scripts = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", html_text, flags=re.IGNORECASE | re.DOTALL)
    without_tags = re.sub(r"<[^>]+>", " ", without_scripts)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def extract_report_context(report_url: str, reports_root: Path = REPORTS_ROOT, limit: int = REPORT_CONTEXT_CHARS) -> str:
    if not report_url:
        return ""

    report_path = safe_report_path(report_url, reports_root)
    if report_path is None:
        raise UploadError("Selected report was not found")
    return strip_html_to_text(report_path.read_text(encoding="utf-8"))[:limit]


def chat_completions_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def chat_settings_from_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ if environ is None else environ
    base_url = env.get("LOCAL_LLM_BASE_URL", "").strip()
    model = env.get("LOCAL_LLM_MODEL", "").strip()
    api_key = env.get("LOCAL_LLM_API_KEY", "EMPTY").strip() or "EMPTY"
    if not base_url or not model:
        raise UploadError("Chat requires LOCAL_LLM_BASE_URL and LOCAL_LLM_MODEL")
    return {"base_url": base_url, "api_key": api_key, "model": model}


def normalize_response_language(value: str) -> tuple[str, str]:
    requested = (value or "").strip().lower()
    for code, _label, prompt_name in LANGUAGE_OPTIONS:
        if requested == code:
            return code, prompt_name
    return LANGUAGE_OPTIONS[0][0], LANGUAGE_OPTIONS[0][2]


def call_chat_completion(messages: list[dict[str, str]], settings: dict[str, str]) -> str:
    payload = {
        "model": settings["model"],
        "messages": messages,
        "temperature": 0.2,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        chat_completions_url(settings["base_url"]),
        data=body,
        headers={
            "Authorization": f"Bearer {settings['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise UploadError(f"Chat model request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise UploadError("Chat model returned invalid JSON") from exc

    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise UploadError("Chat model response missing assistant content") from exc


def parse_qa_candidates(text: str) -> list[dict[str, str]]:
    clean_text = text.strip()
    try:
        parsed = json.loads(clean_text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", clean_text, flags=re.DOTALL)
        if not match:
            raise UploadError("Candidate QA could not be parsed. Regenerate or edit manually.")
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise UploadError("Candidate QA could not be parsed. Regenerate or edit manually.") from exc
    if not isinstance(parsed, list):
        raise UploadError("Candidate QA response must be a JSON array")

    candidates: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        expected_answer = str(item.get("expected_answer", "")).strip()
        if question and expected_answer:
            candidates.append({"question": question, "expected_answer": expected_answer})
    if not candidates:
        raise UploadError("Candidate QA response did not include usable rows")
    return candidates


def build_qa_generation_messages(source_text: str, count: int, language: str) -> list[dict[str, str]]:
    _code, prompt_language = normalize_response_language(language)
    return [
        {
            "role": "system",
            "content": (
                "You create QA pairs for RAG evaluation datasets. "
                "Return strict JSON only. Do not include markdown or commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate {count} high-quality QA pairs from the source text. "
                f"Use {prompt_language} for the question when appropriate. "
                "Each question must be answerable from the source text. "
                "Each expected_answer must be factual, specific, and self-contained. "
                "Return only a JSON array with objects shaped exactly as "
                "{\"question\":\"...\",\"expected_answer\":\"...\"}.\n\n"
                f"Source text:\n{source_text}"
            ),
        },
    ]


def generate_qa_candidates(source_text: str, count: int, language: str = "zh") -> dict[str, Any]:
    clean_source_text = source_text.strip()
    if not clean_source_text:
        raise UploadError("Please provide source text.")
    if count not in QA_GENERATION_COUNTS:
        raise UploadError("QA generation count must be one of 5, 10, or 20")
    messages = build_qa_generation_messages(clean_source_text, count, language)
    response_text = call_chat_completion(messages, chat_settings_from_env())
    candidates = parse_qa_candidates(response_text)
    return {"ok": True, "candidates": candidates[:count]}


def load_pipeline_config(config_path: Path) -> dict[str, Any]:
    from rag_eval_pipeline.config import load_config

    return load_config(config_path)


def config_secret_value(section: dict[str, Any], key: str, env_key: str, default: str = "") -> str:
    value = str(section.get(key) or "").strip()
    if value:
        return value
    env_name = str(section.get(env_key) or "").strip()
    if env_name:
        return os.environ.get(env_name, "").strip() or default
    return default


def first_number(values: Any, default: int | float) -> int | float:
    if isinstance(values, list) and values:
        values = values[0]
    try:
        return int(values)
    except (TypeError, ValueError):
        try:
            return float(values)
        except (TypeError, ValueError):
            return default


def configured_dataset_ids(config: dict[str, Any], generation: dict[str, Any]) -> str:
    explicit_ids = str(generation.get("dataset_ids") or "").strip()
    if explicit_ids:
        return explicit_ids
    dataset_ids = [
        str(dataset.get("id") or "").strip()
        for dataset in (config.get("datasets") or [])
        if isinstance(dataset, dict) and str(dataset.get("id") or "").strip()
    ]
    return ",".join(dataset_ids)


def chat_settings_from_generation_config(config_path: Path) -> dict[str, str]:
    config = load_pipeline_config(config_path)
    generation = dict(config.get("generation") or {})
    base_url = str(generation.get("llm_base_url") or "").strip()
    model = str(generation.get("llm_model") or "").strip()
    api_key = config_secret_value(generation, "llm_api_key", "llm_api_key_env", "EMPTY") or "EMPTY"
    if not base_url or not model:
        raise UploadError("Generation model requires generation.llm_base_url and generation.llm_model.")
    return {"base_url": base_url, "api_key": api_key, "model": model}


def post_retrieval_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int = 30,
) -> dict[str, Any] | list[Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise UploadError(f"Retrieval request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise UploadError("Retrieval service returned invalid JSON") from exc


def retrieval_args_from_config(config_path: Path) -> argparse.Namespace:
    config = load_pipeline_config(config_path)
    generation = dict(config.get("generation") or {})
    grid = dict(config.get("grid") or {})
    retrieval_url = str(generation.get("retrieval_url") or "").strip()
    if not retrieval_url:
        raise UploadError("Retrieval URL is required: set generation.retrieval_url.")
    page_size = int(first_number(generation.get("page_size") or grid.get("page_sizes"), 5))
    similarity_threshold = float(first_number(generation.get("similarity_threshold") or grid.get("similarity_thresholds"), 0.0))
    auth_scheme = generation.get("retrieval_auth_scheme") if "retrieval_auth_scheme" in generation else "Bearer"
    return argparse.Namespace(
        retrieval_url=retrieval_url,
        retrieval_api_key=config_secret_value(generation, "retrieval_api_key", "retrieval_api_key_env"),
        retrieval_auth_header=generation.get("retrieval_auth_header") or "Authorization",
        retrieval_auth_scheme="" if auth_scheme is None else str(auth_scheme),
        retrieval_headers_json=generation.get("retrieval_headers_json"),
        retrieval_payload_json=generation.get("retrieval_payload_json"),
        retrieval_extra_body_json=generation.get("retrieval_extra_body_json"),
        dataset_ids=configured_dataset_ids(config, generation),
        document_ids=generation.get("document_ids"),
        page=int(generation.get("page") or 1),
        page_size=page_size,
        similarity_threshold=similarity_threshold,
        vector_similarity_weight=float(generation.get("vector_similarity_weight") or 0.3),
        top_k=int(generation.get("top_k") or 1024),
        keyword=bool(generation.get("keyword", False)),
        highlight=bool(generation.get("highlight", False)),
        cross_languages=generation.get("cross_languages"),
        use_kg=bool(generation.get("use_kg", False)),
        toc_enhance=bool(generation.get("toc_enhance", False)),
        rerank_id=generation.get("rerank_id"),
        metadata_condition_json=generation.get("metadata_condition_json"),
        max_passages=int(generation.get("max_passages") or page_size),
    )


def passage_payload(passages: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": passage.passage_id,
            "text": passage.text,
            "source": passage.raw_id,
        }
        for passage in passages
    ]


def retrieval_metadata(args: argparse.Namespace) -> dict[str, Any]:
    from rag_eval_pipeline.generation import parse_csv_list

    return {
        "url": args.retrieval_url,
        "dataset_ids": parse_csv_list(args.dataset_ids),
        "document_ids": parse_csv_list(args.document_ids),
        "page": args.page,
        "page_size": args.page_size,
        "similarity_threshold": args.similarity_threshold,
        "max_passages": args.max_passages,
    }


def retrieve_chunks_from_question(
    question: str,
    config_path: Path = DEFAULT_PIPELINE_CONFIG,
    *,
    page_size: int | None = None,
    similarity_threshold: float | None = None,
    dataset_ids: list[str] | None = None,
    document_ids: list[str] | None = None,
    max_passages: int | None = None,
) -> dict[str, Any]:
    from rag_eval_pipeline import generation as rag_generation

    clean_question = question.strip()
    if not clean_question:
        raise UploadError("Question is required.")

    args = retrieval_args_from_config(config_path)
    if page_size is not None:
        if page_size < 1:
            raise UploadError("page_size must be >= 1.")
        args.page_size = page_size
    if similarity_threshold is not None:
        args.similarity_threshold = similarity_threshold
    if dataset_ids:
        args.dataset_ids = ",".join(str(dataset_id).strip() for dataset_id in dataset_ids if str(dataset_id).strip())
    if document_ids:
        args.document_ids = ",".join(str(document_id).strip() for document_id in document_ids if str(document_id).strip())
    if max_passages is not None:
        if max_passages < 1:
            raise UploadError("max_passages must be >= 1.")
        args.max_passages = max_passages
    elif args.max_passages < 1:
        args.max_passages = args.page_size

    query = rag_generation.Query(query_id="retrieval_preview", query=clean_question)
    retrieval_payload = rag_generation.build_retrieval_payload(args, query)
    retrieval_headers = rag_generation.build_retrieval_headers(args)
    retrieval_json = post_retrieval_json(args.retrieval_url, retrieval_payload, retrieval_headers)
    passages = rag_generation.extract_passages(retrieval_json, args.max_passages)
    return {
        "ok": True,
        "question": clean_question,
        "chunks": passage_payload(passages),
        "retrieval": retrieval_metadata(args),
    }


def build_answer_generation_messages(question: str, passages: list[dict[str, Any]], language: str) -> list[dict[str, str]]:
    _code, prompt_language = normalize_response_language(language)
    context = "\n\n".join(
        f"[{index}] {str(passage.get('text') or '').strip()}"
        for index, passage in enumerate(passages, start=1)
        if str(passage.get("text") or "").strip()
    )
    return [
        {
            "role": "system",
            "content": (
                "You create standard answers for RAG evaluation datasets. "
                "Use only the retrieved passages. Return answer text only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Respond in {prompt_language}.\n"
                f"Question:\n{question}\n\n"
                f"Retrieved passages:\n{context}\n\n"
                "Write one factual, specific, self-contained standard answer. "
                "If the retrieved passages are insufficient, say so clearly."
            ),
        },
    ]


def generate_answer_from_question(
    question: str,
    language: str,
    config_path: Path = DEFAULT_PIPELINE_CONFIG,
) -> dict[str, Any]:
    from rag_eval_pipeline import generation as rag_generation

    clean_question = question.strip()
    if not clean_question:
        raise UploadError("Question is required.")
    args = retrieval_args_from_config(config_path)
    query = rag_generation.Query(query_id="manual_question", query=clean_question)
    retrieval_payload = rag_generation.build_retrieval_payload(args, query)
    retrieval_headers = rag_generation.build_retrieval_headers(args)
    retrieval_json = post_retrieval_json(args.retrieval_url, retrieval_payload, retrieval_headers)
    passages = rag_generation.extract_passages(retrieval_json, args.max_passages)
    if not passages:
        raise UploadError("No usable retrieved passages were found.")
    public_passages = passage_payload(passages)
    answer = call_chat_completion(
        build_answer_generation_messages(clean_question, public_passages, language),
        chat_settings_from_generation_config(config_path),
    ).strip()
    if not answer:
        raise UploadError("Model returned an empty answer.")
    return {"ok": True, "question": clean_question, "expected_answer": answer, "passages": public_passages}


def build_answer_regeneration_messages(
    question: str,
    current_answer: str,
    passages: list[dict[str, Any]],
    language: str,
) -> list[dict[str, str]]:
    messages = build_answer_generation_messages(question, passages, language)
    messages[-1]["content"] += (
        f"\n\nCurrent answer:\n{current_answer}\n\n"
        "Improve the current answer while staying faithful to the same retrieved passages."
    )
    return messages


def regenerate_answer_from_passages(
    question: str,
    current_answer: str,
    passages: list[dict[str, Any]],
    language: str,
    config_path: Path = DEFAULT_PIPELINE_CONFIG,
) -> dict[str, Any]:
    clean_question = question.strip()
    clean_answer = current_answer.strip()
    clean_passages = [
        {"text": str(passage.get("text") or "").strip(), "source": str(passage.get("source") or "").strip()}
        for passage in passages
        if isinstance(passage, dict) and str(passage.get("text") or "").strip()
    ]
    if not clean_question:
        raise UploadError("Question is required.")
    if not clean_answer:
        raise UploadError("Current answer is required.")
    if not clean_passages:
        raise UploadError("Retrieved passages are required before regenerating.")
    answer = call_chat_completion(
        build_answer_regeneration_messages(clean_question, clean_answer, clean_passages, language),
        chat_settings_from_generation_config(config_path),
    ).strip()
    if not answer:
        raise UploadError("Model returned an empty answer.")
    return {"ok": True, "expected_answer": answer}


def answer_chat_question(
    question: str,
    report_url: str,
    reports_root: Path = REPORTS_ROOT,
    language: str = "de",
) -> dict[str, str]:
    clean_question = question.strip()
    if not clean_question:
        raise UploadError("Chat question is required")

    language_code, language_name = normalize_response_language(language)
    report_context = extract_report_context(report_url, reports_root)
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions about RAG evaluation reports. "
                "Use the supplied report context when it is relevant, and be concise. "
                f"Respond in {language_name}."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{clean_question}\n\nCurrent report context:\n{report_context or '(no report selected)'}",
        },
    ]
    settings = chat_settings_from_env()
    answer = call_chat_completion(messages, settings)
    report_path = ""
    if report_url:
        resolved_report_path = safe_report_path(report_url, reports_root)
        if resolved_report_path is not None:
            report_path = relative_repo_path(resolved_report_path)
    return {"answer": answer, "model": settings["model"], "report_path": report_path, "language": language_code}


def render_upload_page(result_reports: list[dict[str, str]] | None = None) -> str:
    if result_reports is None:
        result_reports = list_result_reports()
    report_options = render_report_options(result_reports)
    language_options = "\n".join(
        f'              <option value="{html.escape(code, quote=True)}">{html.escape(label)}</option>'
        for code, label, _prompt_name in LANGUAGE_OPTIONS
    )
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AT&amp;S Dataset Upload</title>
  <style>
    :root {
      --ats-blue: #176f95;
      --ats-blue-dark: #0b4f6c;
      --ats-blue-deep: #08384c;
      --ats-blue-soft: #e6f5fa;
      --ats-cyan: #1aa6c8;
      --ats-teal: #1aa6c8;
      --ats-blue-100: #e6f5fa;
      --ats-blue-300: #8fbfd1;
      --ink: #0d3445;
      --muted: #4f6f7c;
      --line: #c6dce5;
      --error: #b3261e;
      --success: #176f95;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background:
        linear-gradient(180deg, rgba(0, 167, 200, 0.18), rgba(0, 167, 200, 0) 32rem),
        linear-gradient(135deg, var(--ats-blue-deep) 0%, var(--ats-blue-dark) 45%, var(--ats-blue) 100%);
    }

    main {
      width: min(1200px, calc(100% - 32px));
      margin: 0 auto;
      padding: 48px 0;
    }

    .brand {
      color: #ffffff;
      margin-bottom: 32px;
    }

    .brand-top {
      display: flex;
      justify-content: flex-start;
      margin-bottom: 18px;
    }

    .mark {
      font-size: clamp(2rem, 8vw, 4.5rem);
      line-height: 1;
      font-weight: 700;
      letter-spacing: 0;
    }

    .subtitle {
      max-width: 680px;
      margin: 12px 0 0;
      color: #dbeeff;
      font-size: 1.05rem;
    }

    .panel {
      display: grid;
      gap: 24px;
      padding: 28px;
      border: 1px solid var(--line);
      border-top: 4px solid var(--ats-cyan);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 18px 44px rgba(6, 40, 70, 0.18);
    }

    .workspace {
      display: grid;
      gap: 24px;
      align-items: start;
    }

    .report-content-stack {
      display: grid;
      gap: 24px;
      align-items: start;
    }

    .report-main {
      display: grid;
      gap: 24px;
      min-width: 0;
    }

    .report-toolbar,
    .dialog-titlebar {
      display: flex;
      gap: 16px;
      align-items: flex-start;
      justify-content: space-between;
    }

    .report-toolbar {
      margin: -28px -28px 0;
      padding: 22px 28px;
      border-radius: 8px 8px 0 0;
      background: linear-gradient(135deg, var(--ats-blue-dark), var(--ats-blue));
      color: #ffffff;
    }

    .report-toolbar .section-heading {
      color: #ffffff;
    }

    .report-toolbar .section-note {
      color: #dbeeff;
    }

    .section-heading {
      margin: 0;
      color: var(--ats-blue-dark);
      font-size: 1.2rem;
    }

    .section-note {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 0.95rem;
    }

    form {
      display: grid;
      gap: 18px;
    }

    label {
      display: grid;
      gap: 8px;
      font-weight: 700;
      color: var(--ats-blue-dark);
    }

    input,
    textarea,
    select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px 14px;
      font: inherit;
      color: var(--ink);
      background: #ffffff;
    }

    input:focus,
    textarea:focus,
    select:focus {
      outline: 3px solid rgba(24, 167, 181, 0.25);
      border-color: var(--ats-teal);
    }

    textarea {
      min-height: 120px;
      resize: vertical;
    }

    button {
      justify-self: start;
      min-height: 44px;
      border: 0;
      border-radius: 6px;
      padding: 0 18px;
      font: inherit;
      font-weight: 700;
      color: #ffffff;
      background: var(--ats-blue);
      cursor: pointer;
      box-shadow: 0 8px 18px rgba(0, 47, 95, 0.18);
    }

    button:hover {
      background: var(--ats-blue-dark);
    }

    button[disabled] {
      cursor: wait;
      opacity: 0.72;
    }

    .ghost-button {
      border: 1px solid var(--line);
      color: var(--ats-blue-dark);
      background: #ffffff;
    }

    .ghost-button:hover {
      color: var(--ats-blue);
      border-color: var(--ats-cyan);
      background: var(--ats-blue-soft);
    }

    .back-button {
      min-height: 38px;
      border: 1px solid rgba(255, 255, 255, 0.58);
      padding: 0 14px;
      color: #ffffff;
      background: rgba(23, 111, 149, 0.62);
    }

    .back-button:hover {
      border-color: #ffffff;
      background: rgba(0, 167, 200, 0.32);
    }

    .toolbar-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
      align-items: flex-end;
    }

    .language-select {
      min-width: 180px;
      color: #ffffff;
    }

    .language-select span {
      color: #dbeeff;
      font-size: 0.88rem;
    }

    .result {
      min-height: 72px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 16px;
      background: var(--ats-blue-soft);
      overflow-wrap: anywhere;
    }

    .result:empty::before {
      content: "Saved under data/uploaded_datasets/. Exact dataset path will appear here after upload.";
      color: var(--muted);
    }

    .pipeline-result:empty::before {
      content: "Pipeline server status will appear here.";
      color: var(--muted);
    }

    .result.error {
      color: var(--error);
      background: #fff4f2;
      border-color: #f2b8b5;
    }

    .result.success {
      color: var(--ats-blue-dark);
      background: var(--ats-blue-100);
      border-color: var(--ats-blue-300);
    }

    .result pre {
      margin: 12px 0 0;
      padding: 12px;
      border-radius: 6px;
      color: var(--ink);
      background: #ffffff;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .report-path {
      border: 1px solid var(--line);
      border-left: 4px solid var(--ats-cyan);
      border-radius: 6px;
      padding: 12px 14px;
      color: var(--ats-blue-dark);
      background: var(--ats-blue-soft);
      overflow-wrap: anywhere;
      font-weight: 700;
    }

    .report-frame {
      width: 100%;
      min-height: 720px;
      border: 2px solid var(--ats-blue-dark);
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
    }

    .report-empty {
      min-height: auto;
    }

    dialog {
      width: min(620px, calc(100% - 32px));
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0;
      color: var(--ink);
      background: #ffffff;
      box-shadow: 0 24px 70px rgba(6, 40, 70, 0.34);
    }

    dialog::backdrop {
      background: rgba(6, 40, 70, 0.48);
    }

    .dialog-window {
      display: grid;
      gap: 22px;
      padding: 26px;
    }

    .chat-answer {
      min-height: 120px;
      white-space: pre-wrap;
    }

    .chat-answer:empty::before {
      content: "Model answer will appear here.";
      color: var(--muted);
    }

    .chat-window {
      padding: 0;
      border-top: 1px solid var(--line);
      padding-top: 22px;
      overflow: hidden;
    }

    .chat-window .section-heading {
      color: var(--ats-blue-dark);
    }

    .chat-window form {
      margin-top: 14px;
      gap: 12px;
    }

    .chat-window textarea {
      min-height: 92px;
    }

    @media (max-width: 900px) {
      .report-toolbar,
      .dialog-titlebar {
        display: grid;
      }
    }

    @media (max-width: 640px) {
      main {
        width: min(100% - 24px, 960px);
        padding: 28px 0;
      }

      .panel {
        padding: 20px;
      }

      button {
        width: 100%;
      }

      .chat-window {
        padding-top: 22px;
      }
    }
  </style>
</head>
<body>
  <main>
    <header class="brand">
      <div class="brand-top">
        <button id="backButton" class="back-button" type="button" data-i18n="viewReport">View report</button>
      </div>
      <div class="mark">AT&amp;S</div>
      <p class="subtitle" data-i18n="subtitle">Upload CSV datasets or text-based PDF QA pairs for RAG evaluation runs.</p>
    </header>

    <div class="workspace">
      <section id="reports" class="panel report-panel" aria-label="Report analysis">
        <div class="report-toolbar">
          <div>
            <h2 class="section-heading" data-i18n="evaluationReports">Report analysis</h2>
            <p class="section-note" data-i18n="reportsNote">Preview HTML reports and ask focused questions about the selected report.</p>
          </div>
          <div class="toolbar-actions">
            <label class="language-select">
              <span data-i18n="pageLanguage">Page language</span>
              <select id="pageLanguage">
{{LANGUAGE_OPTIONS}}
              </select>
            </label>
            <button id="quickPipelineButton" type="button" data-i18n="quickEvaluation" data-ready-text="Quick evaluation" data-loading-text="Starting..." data-running-text="Running...">Quick evaluation</button>
            <button id="runPipelineButton" type="button" data-i18n="runEvaluation" data-ready-text="Run evaluation" data-loading-text="Starting..." data-running-text="Running...">Run evaluation</button>
            <button id="openUploadDialog" type="button" data-i18n="uploadDataset">Upload dataset</button>
          </div>
        </div>
        <div id="pipelineResult" class="result pipeline-result" role="status" aria-live="polite"></div>

        <div class="report-content-stack">
          <div class="report-main">
            {{REPORT_OPTIONS}}
          </div>
        </div>
      </section>
    </div>

    <dialog id="uploadDialog" aria-labelledby="uploadDialogTitle">
      <div class="dialog-window">
        <div class="dialog-titlebar">
          <div>
            <h2 id="uploadDialogTitle" class="section-heading" data-i18n="uploadDatasetWindow">Upload dataset window</h2>
            <p class="section-note" data-i18n="uploadWindowNote">CSV stays as-is. PDF QA pairs are converted to evaluation CSV.</p>
          </div>
          <button id="closeUploadDialog" class="ghost-button" type="button" data-i18n="close">Close</button>
        </div>
        <form id="datasetUploadForm">
          <label>
            <span data-i18n="datasetName">Dataset name</span>
            <input type="text" name="name" autocomplete="off" placeholder="quarterly-support-qa">
          </label>

          <label>
            <span data-i18n="csvOrPdfFile">CSV or PDF file</span>
            <input type="file" name="file" accept=".csv,.pdf,text/csv,application/pdf" required>
          </label>

          <button type="submit" data-i18n="uploadDataset" data-ready-text="Upload dataset" data-loading-text="Uploading...">Upload dataset</button>
        </form>

        <div id="uploadResult" class="result" role="status" aria-live="polite"></div>
      </div>
    </dialog>
  </main>

  <script>
    const form = document.getElementById("datasetUploadForm");
    const result = document.getElementById("uploadResult");
    const button = form.querySelector("button[type='submit']");
    const manualQaForm = document.getElementById("manualQaForm");
    const manualQaDatasetSelect = document.getElementById("manualQaDatasetSelect");
    const manualQaResult = document.getElementById("manualQaResult");
    const manualQaButton = manualQaForm ? manualQaForm.querySelector("button[type='submit']") : null;
    const manualQaQuestionTextarea = document.getElementById("manualQaQuestionTextarea");
    const manualQaAnswerTextarea = document.getElementById("manualQaAnswerTextarea");
    const uploadDialog = document.getElementById("uploadDialog");
    const openUploadDialog = document.getElementById("openUploadDialog");
    const closeUploadDialog = document.getElementById("closeUploadDialog");
    const askModelForm = document.getElementById("askModelForm");
    const chatAnswer = document.getElementById("chatAnswer");
    const askModelButton = askModelForm.querySelector("button[type='submit']");
    const runPipelineButton = document.getElementById("runPipelineButton");
    const pipelineResult = document.getElementById("pipelineResult");
    const pageLanguage = document.getElementById("pageLanguage");
    const backButton = document.getElementById("backButton");
    const reportSelect = document.getElementById("reportSelect");
    const reportPath = document.getElementById("reportPath");
    const reportFrame = document.getElementById("reportFrame");

    function escapeHTML(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    const uiTranslations = {
      de: {
        viewReport: "Bericht ansehen",
        subtitle: "CSV-Datensatze oder textbasierte PDF-QA-Paare fur RAG-Auswertungen hochladen.",
        evaluationReports: "Berichtsanalyse",
        reportsNote: "HTML-Berichte voranzeigen und gezielte Fragen zum ausgewaehlten Bericht stellen.",
        pageLanguage: "Seitensprache",
        runEvaluation: "Starten",
        pipelineRunning: "Laeuft...",
        uploadDataset: "Datensatz hochladen",
        uploadDatasetWindow: "Fenster zum Hochladen",
        uploadWindowNote: "CSV bleibt unverandert. PDF-QA-Paare werden in Evaluierungs-CSV umgewandelt.",
        close: "Schliessen",
        datasetName: "Datensatzname",
        csvOrPdfFile: "CSV- oder PDF-Datei",
        reportFile: "Berichtsdatei",
        reportPath: "Berichtspfad",
        askModel: "Modell fragen",
        askModelNote: "Fragen Sie zum ausgewahlten Bericht. Diese Antworten werden nicht gespeichert.",
        question: "Frage",
        questionPlaceholder: "Was zeigt dieser Bericht?",
        manualQuestionPlaceholder: "Geben Sie die Frage ein, die Sie hinzufuegen moechten",
        modelPlaceholder: "Die Modellantwort wird hier angezeigt.",
        pipelinePlaceholder: "Der Pipeline-Serverstatus wird hier angezeigt."
      },
      en: {
        viewReport: "View report",
        subtitle: "Upload CSV datasets or text-based PDF QA pairs for RAG evaluation runs.",
        evaluationReports: "Report analysis",
        reportsNote: "Preview HTML reports and ask focused questions about the selected report.",
        pageLanguage: "Page language",
        runEvaluation: "Run",
        pipelineRunning: "Running...",
        uploadDataset: "Upload dataset",
        uploadDatasetWindow: "Upload dataset window",
        uploadWindowNote: "CSV stays as-is. PDF QA pairs are converted to evaluation CSV.",
        close: "Close",
        datasetName: "Dataset name",
        csvOrPdfFile: "CSV or PDF file",
        reportFile: "Report file",
        reportPath: "Report path",
        askModel: "Ask model",
        askModelNote: "Ask about the selected report. Answers are not saved.",
        question: "Question",
        questionPlaceholder: "What does this report show?",
        manualQuestionPlaceholder: "Enter the question you want to add",
        modelPlaceholder: "Model answer will appear here.",
        pipelinePlaceholder: "Pipeline server status will appear here."
      },
      zh: {
        viewReport: "查看报告",
        subtitle: "上传 CSV 数据集或文本型 PDF QA 对，用于 RAG 评估运行。",
        evaluationReports: "报告分析",
        reportsNote: "围绕 HTML 报告进行预览和问答分析。",
        pageLanguage: "页面语言",
        runEvaluation: "运行",
        pipelineRunning: "运行中...",
        uploadDataset: "上传数据集",
        uploadDatasetWindow: "上传数据集窗口",
        uploadWindowNote: "CSV 保持原逻辑。PDF QA 对会转换为评估 CSV。",
        close: "关闭",
        datasetName: "数据集名称",
        csvOrPdfFile: "CSV 或 PDF 文件",
        reportFile: "报告文件",
        reportPath: "报告路径",
        askModel: "询问模型",
        askModelNote: "围绕所选报告临时提问，回答不会保存。",
        question: "问题",
        questionPlaceholder: "这份报告显示了什么？",
        manualQuestionPlaceholder: "请输入您想要添加的问题",
        modelPlaceholder: "模型回答将显示在这里。",
        pipelinePlaceholder: "Pipeline 服务状态将显示在这里。"
      },
      ms: {
        viewReport: "Lihat laporan",
        subtitle: "Muat naik dataset CSV atau pasangan QA PDF berasaskan teks untuk penilaian RAG.",
        evaluationReports: "Analisis laporan",
        reportsNote: "Pratonton laporan HTML dan tanya soalan khusus tentang laporan yang dipilih.",
        pageLanguage: "Bahasa halaman",
        runEvaluation: "Jalankan",
        pipelineRunning: "Sedang berjalan...",
        uploadDataset: "Muat naik dataset",
        uploadDatasetWindow: "Tetingkap muat naik dataset",
        uploadWindowNote: "CSV kekal seperti asal. Pasangan QA PDF ditukar kepada CSV penilaian.",
        close: "Tutup",
        datasetName: "Nama dataset",
        csvOrPdfFile: "Fail CSV atau PDF",
        reportFile: "Fail laporan",
        reportPath: "Laluan laporan",
        askModel: "Tanya model",
        askModelNote: "Tanya tentang laporan yang dipilih. Jawapan tidak disimpan.",
        question: "Soalan",
        questionPlaceholder: "Apakah yang ditunjukkan oleh laporan ini?",
        manualQuestionPlaceholder: "Masukkan soalan yang ingin anda tambah",
        modelPlaceholder: "Jawapan model akan dipaparkan di sini.",
        pipelinePlaceholder: "Status pelayan pipeline akan dipaparkan di sini."
      }
    };

    const reportTranslations = {
      de: {
        "RAG Evaluation": "RAG-Auswertung",
        "Source:": "Quelle:",
        "self-contained report for RAG evaluation metrics": "eigenstaendiger Bericht fuer RAG-Auswertungsmetriken",
        "Rows": "Zeilen",
        "Evaluation runs": "Auswertungsläufe",
        "Queries": "Abfragen",
        "Unique query_id count": "Anzahl eindeutiger query_id",
        "Recall avg": "Recall-Durchschnitt",
        "Mean of factual/vital/assignment recall": "Mittelwert aus Fakten-, Kernpunkt- und Zuweisungs-Recall",
        "Factual recall": "Fakten-Recall",
        "Golden claim coverage": "Abdeckung der Referenzaussagen",
        "Vital nuggets": "Wichtige Kernpunkte",
        "Key nugget coverage": "Abdeckung wichtiger Kernpunkte",
        "Nugget assignment": "Kernpunkt-Zuordnung",
        "Faithfulness": "Quellentreue",
        "Claims supported by retrieved context": "Aussagen durch gefundenen Kontext gestuetzt",
        "Query Summary": "Abfrageuebersicht",
        "Run Details": "Laufdetails",
        "All query_id": "Alle query_id",
        "All recall bands": "Alle Recall-Bereiche",
        "Search query text": "Abfragetext suchen",
        "Recall average": "Recall-Durchschnitt",
        "Factual precision": "Fakten-Praezision",
        "Factual F1": "Fakten-F1",
        "Semantic similarity": "Semantische Aehnlichkeit",
        "Citation F1": "Zitations-F1",
        "Source support": "Quellenstuetzung",
        "Generated answer preview": "Vorschau der generierten Antwort",
        "Expected answer preview": "Vorschau der erwarteten Antwort",
        "UMBRELA passage scores": "UMBRELA-Passagewerte",
        "Retrieval precision/AP/MRR": "Retrieval-Praezision/AP/MRR",
        "Nuggets and assignments": "Kernpunkte und Zuordnungen",
        "Faithfulness claims": "Quellentreue-Aussagen",
        "Faithfulness verdicts": "Quellentreue-Urteile",
        "Unsupported claims": "Nicht gestuetzte Aussagen",
        "Citation scores": "Zitationswerte",
        "No-answer score": "Keine-Antwort-Wert",
        "Generated claims": "Generierte Aussagen",
        "Expected claims": "Erwartete Aussagen",
        "Precision verdicts": "Praezisionsurteile",
        "Recall verdicts": "Recall-Urteile",
        "query_id": "query_id",
        "runs": "Laeufe",
        "query": "Abfrage",
        "recall avg": "Recall-Durchschnitt",
        "nugget assignment": "Kernpunkt-Zuordnung"
      },
      en: {},
      zh: {
        "RAG Evaluation": "RAG 评估",
        "Source:": "来源：",
        "self-contained report for RAG evaluation metrics": "RAG 评估指标自包含报告",
        "Rows": "行数",
        "Evaluation runs": "评估运行",
        "Queries": "问题",
        "Unique query_id count": "唯一 query_id 数量",
        "Recall avg": "召回均值",
        "Mean of factual/vital/assignment recall": "事实/关键点/分配召回的均值",
        "Factual recall": "事实召回",
        "Golden claim coverage": "标准事实覆盖率",
        "Vital nuggets": "关键要点",
        "Key nugget coverage": "关键要点覆盖率",
        "Nugget assignment": "要点分配",
        "Faithfulness": "忠实度",
        "Claims supported by retrieved context": "由检索上下文支持的声明",
        "Query Summary": "问题汇总",
        "Run Details": "运行详情",
        "All query_id": "全部 query_id",
        "All recall bands": "全部召回区间",
        "Search query text": "搜索问题文本",
        "Recall average": "召回均值",
        "Factual precision": "事实精确率",
        "Factual F1": "事实 F1",
        "Semantic similarity": "语义相似度",
        "Citation F1": "引用 F1",
        "Source support": "来源支持",
        "Generated answer preview": "生成答案预览",
        "Expected answer preview": "期望答案预览",
        "UMBRELA passage scores": "UMBRELA 段落评分",
        "Retrieval precision/AP/MRR": "检索 Precision/AP/MRR",
        "Nuggets and assignments": "要点与分配",
        "Faithfulness claims": "忠实度声明",
        "Faithfulness verdicts": "忠实度判定",
        "Unsupported claims": "未支持声明",
        "Citation scores": "引用评分",
        "No-answer score": "无答案评分",
        "Generated claims": "生成声明",
        "Expected claims": "期望声明",
        "Precision verdicts": "精确率判定",
        "Recall verdicts": "召回判定",
        "query_id": "query_id",
        "runs": "运行数",
        "query": "问题",
        "recall avg": "召回均值",
        "nugget assignment": "要点分配"
      },
      ms: {
        "RAG Evaluation": "Penilaian RAG",
        "Source:": "Sumber:",
        "self-contained report for RAG evaluation metrics": "laporan kendiri untuk metrik penilaian RAG",
        "Rows": "Baris",
        "Evaluation runs": "Larian penilaian",
        "Queries": "Pertanyaan",
        "Unique query_id count": "Bilangan query_id unik",
        "Recall avg": "Purata recall",
        "Mean of factual/vital/assignment recall": "Purata recall fakta/inti/padanan",
        "Factual recall": "Recall fakta",
        "Golden claim coverage": "Liputan dakwaan rujukan",
        "Vital nuggets": "Inti penting",
        "Key nugget coverage": "Liputan inti utama",
        "Nugget assignment": "Padanan inti",
        "Faithfulness": "Kesetiaan",
        "Claims supported by retrieved context": "Dakwaan disokong oleh konteks carian",
        "Query Summary": "Ringkasan pertanyaan",
        "Run Details": "Butiran larian",
        "All query_id": "Semua query_id",
        "All recall bands": "Semua julat recall",
        "Search query text": "Cari teks pertanyaan",
        "Recall average": "Purata recall",
        "Factual precision": "Ketepatan fakta",
        "Factual F1": "F1 fakta",
        "Semantic similarity": "Kesamaan semantik",
        "Citation F1": "F1 sitasi",
        "Source support": "Sokongan sumber",
        "Generated answer preview": "Pratonton jawapan dijana",
        "Expected answer preview": "Pratonton jawapan dijangka",
        "UMBRELA passage scores": "Skor petikan UMBRELA",
        "Retrieval precision/AP/MRR": "Ketepatan carian/AP/MRR",
        "Nuggets and assignments": "Inti dan padanan",
        "Faithfulness claims": "Dakwaan kesetiaan",
        "Faithfulness verdicts": "Keputusan kesetiaan",
        "Unsupported claims": "Dakwaan tidak disokong",
        "Citation scores": "Skor sitasi",
        "No-answer score": "Skor tiada jawapan",
        "Generated claims": "Dakwaan dijana",
        "Expected claims": "Dakwaan dijangka",
        "Precision verdicts": "Keputusan ketepatan",
        "Recall verdicts": "Keputusan recall",
        "query_id": "query_id",
        "runs": "larian",
        "query": "pertanyaan",
        "recall avg": "purata recall",
        "nugget assignment": "padanan inti"
      }
    };

    function currentTranslations() {
      return uiTranslations[pageLanguage.value] || uiTranslations.en;
    }

    function applyReportLanguage(language) {
      if (!reportFrame) {
        return;
      }
      let frameDocument;
      try {
        frameDocument = reportFrame.contentDocument || reportFrame.contentWindow.document;
      } catch (error) {
        return;
      }
      if (!frameDocument?.body) {
        return;
      }
      const translations = reportTranslations[language] || {};
      const walker = frameDocument.createTreeWalker(
        frameDocument.body,
        NodeFilter.SHOW_TEXT,
        {
          acceptNode(node) {
            const parent = node.parentElement;
            if (!parent || ["SCRIPT", "STYLE"].includes(parent.tagName)) {
              return NodeFilter.FILTER_REJECT;
            }
            return node.nodeValue.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
          }
        }
      );
      const textNodes = [];
      while (walker.nextNode()) {
        textNodes.push(walker.currentNode);
      }
      textNodes.forEach((node) => {
        if (!node.__ragEvalOriginalText) {
          node.__ragEvalOriginalText = node.nodeValue.trim();
        }
        const original = node.__ragEvalOriginalText;
        const translated = translations[original] || original
          .replace("Source:", translations["Source:"] || "Source:")
          .replace(
            "self-contained report for RAG evaluation metrics",
            translations["self-contained report for RAG evaluation metrics"] ||
              "self-contained report for RAG evaluation metrics"
          );
        node.nodeValue = node.nodeValue.replace(node.nodeValue.trim(), translated);
      });
      frameDocument.querySelectorAll("[placeholder]").forEach((element) => {
        if (!element.dataset.ragEvalOriginalPlaceholder) {
          element.dataset.ragEvalOriginalPlaceholder = element.getAttribute("placeholder") || "";
        }
        const original = element.dataset.ragEvalOriginalPlaceholder;
        element.setAttribute("placeholder", translations[original] || original);
      });
    }

    function applyPageLanguage(language) {
      pageLanguage.value = language;
      const translations = currentTranslations();
      document.querySelectorAll("[data-i18n]").forEach((element) => {
        const key = element.dataset.i18n;
        if (translations[key]) {
          element.textContent = translations[key];
        }
      });
      document.querySelectorAll("[data-i18n-placeholder]").forEach((element) => {
        const key = element.dataset.i18nPlaceholder;
        if (translations[key]) {
          element.setAttribute("placeholder", translations[key]);
        }
      });
      quickPipelineButton.dataset.readyText = translations.quickEvaluation;
      quickPipelineButton.dataset.loadingText = language === "zh" ? "启动中..." : language === "ms" ? "Memulakan..." : language === "de" ? "Startet..." : "Starting...";
      quickPipelineButton.dataset.runningText = translations.pipelineRunning;
      runPipelineButton.dataset.readyText = translations.runEvaluation;
      runPipelineButton.dataset.loadingText = quickPipelineButton.dataset.loadingText;
      runPipelineButton.dataset.runningText = translations.pipelineRunning;
      if (!quickPipelineButton.disabled) {
        quickPipelineButton.textContent = translations.quickEvaluation;
      }
      if (!runPipelineButton.disabled) {
        runPipelineButton.textContent = translations.runEvaluation;
      }
      button.dataset.readyText = translations.uploadDataset;
      button.dataset.loadingText = language === "zh" ? "上传中..." : language === "ms" ? "Memuat naik..." : language === "de" ? "Wird hochgeladen..." : "Uploading...";
      if (!button.disabled) {
        button.textContent = translations.uploadDataset;
      }
      askModelButton.dataset.readyText = translations.askModel;
      askModelButton.dataset.loadingText = language === "zh" ? "思考中..." : language === "ms" ? "Berfikir..." : language === "de" ? "Denkt nach..." : "Thinking...";
      if (!askModelButton.disabled) {
        askModelButton.textContent = translations.askModel;
      }
      if (reportPath?.dataset.reportPath) {
        reportPath.textContent = translations.reportPath + ": " + reportPath.dataset.reportPath;
      }
      if (!chatAnswer.textContent.trim()) {
        chatAnswer.textContent = translations.modelPlaceholder;
      }
      if (!pipelineResult.textContent.trim()) {
        pipelineResult.textContent = translations.pipelinePlaceholder;
      }
      applyReportLanguage(language);
      window.localStorage.setItem("ragEvalPageLanguage", language);
    }

    function setResult(type, html) {
      result.className = "result " + type;
      result.innerHTML = html;
    }

    function setPipelineResult(type, html) {
      pipelineResult.className = "result pipeline-result " + type;
      pipelineResult.innerHTML = html;
    }

    function setPipelineButtons(disabled) {
      quickPipelineButton.disabled = disabled;
      runPipelineButton.disabled = disabled;
      if (!disabled) {
        quickPipelineButton.textContent = quickPipelineButton.dataset.readyText;
        runPipelineButton.textContent = runPipelineButton.dataset.readyText;
      }
    }

    async function pollPipelineStatus(runId) {
      const response = await fetch("/api/pipeline/status?run_id=" + encodeURIComponent(runId));
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || payload.ok === false) {
        throw new Error(payload.error || "Pipeline status request failed.");
      }
      const job = payload.job || {};
      const renderedReport = Array.isArray(job.reports) && job.reports.length ? job.reports[0] : null;
      setPipelineResult(
        job.status === "failed" ? "error" : "success",
        "<strong>Pipeline " + escapeHTML(job.status || "unknown") + "</strong>" +
          "<div>Run ID: " + escapeHTML(runId) + "</div>" +
          "<div>Config: " + escapeHTML(job.config || "") + "</div>" +
          (job.last_log ? "<div><strong>Latest log:</strong> " + escapeHTML(job.last_log) + "</div>" : "") +
          (renderedReport ? "<div><strong>HTML report:</strong> " + escapeHTML(renderedReport.path || "") + "</div>" : "") +
          (renderedReport?.eval_result_csv ? "<div><strong>Result CSV:</strong> " + escapeHTML(renderedReport.eval_result_csv) + "</div>" : "") +
          (Array.isArray(job.log_tail) && job.log_tail.length
            ? "<pre>" + escapeHTML(job.log_tail.slice(-10).join("\\n")) + "</pre>"
            : "") +
          (job.error ? "<pre>" + escapeHTML(job.error) + "</pre>" : "")
      );
      if (job.status === "running") {
        window.setTimeout(() => pollPipelineStatus(runId).catch((error) => {
          setPipelineResult("error", "<strong>Error:</strong> " + escapeHTML(error.message));
        }), 1500);
      } else {
        await refreshReportList(renderedReport?.path || "");
        if (reportFrame && typeof resizeReportFrame === "function") {
          resizeReportFrame();
        }
        setPipelineButtons(false);
      }
    }

    function viewReport() {
      const selectedReportUrl = reportSelect?.selectedOptions[0]?.value || "";
      if (window.history.length > 1) {
        window.history.back();
        return;
      }
      if (selectedReportUrl) {
        window.location.href = selectedReportUrl;
        return;
      }
      window.location.href = "/datasets#reports";
    }

    async function refreshReportList(preferredPath = "") {
      if (!reportSelect || !reportPath) {
        return;
      }
      const selectedPath = reportSelect.selectedOptions[0]?.dataset.path || "";
      const response = await fetch("/api/reports");
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || payload.ok === false || !Array.isArray(payload.reports)) {
        return;
      }
      const reports = payload.reports;
      reportSelect.innerHTML = reports.map((report) => {
        const selected = report.path === (preferredPath || selectedPath) ? " selected" : "";
        return "<option value=\\"" + escapeHTML(report.url) + "\\" data-path=\\"" +
          escapeHTML(report.path) + "\\"" + selected + ">" +
          escapeHTML(report.name + " - " + report.modified) + "</option>";
      }).join("");
      const selectedOption = reportSelect.selectedOptions[0];
      if (selectedOption) {
        reportPath.dataset.reportPath = selectedOption.dataset.path;
        reportPath.textContent = currentTranslations().reportPath + ": " + selectedOption.dataset.path;
        if (reportFrame && reportFrame.getAttribute("src") !== selectedOption.value) {
          reportFrame.style.height = "720px";
          reportFrame.src = selectedOption.value;
          reportFrame.hidden = false;
        }
      }
    }

    openUploadDialog.addEventListener("click", () => {
      if (uploadDialog.showModal) {
        uploadDialog.showModal();
      } else {
        uploadDialog.setAttribute("open", "");
      }
    });

    async function startPipeline(mode, activeButton) {
      setPipelineButtons(true);
      activeButton.textContent = activeButton.dataset.loadingText;
      setPipelineResult("", "Starting pipeline server job...");
      try {
        const requestBody = mode === "quick"
          ? {config: "scripts/eval-cfg.yaml", "mode": "quick", stage: "all", dry_run: false, overwrite: false}
          : {config: "scripts/eval-cfg.yaml", "mode": "standard", stage: "all", dry_run: false, overwrite: false};
        const response = await fetch("/api/pipeline/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(requestBody)
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || payload.ok === false) {
          throw new Error(payload.error || "Pipeline start failed.");
        }
        setPipelineResult(
          "success",
          "<strong>Pipeline " + escapeHTML(payload.status || "running") + "</strong>" +
            "<div>Run ID: " + escapeHTML(payload.run_id || "") + "</div>"
        );
        activeButton.textContent = activeButton.dataset.runningText;
        await pollPipelineStatus(payload.run_id);
      } catch (error) {
        setPipelineResult("error", "<strong>Error:</strong> " + escapeHTML(error.message));
        setPipelineButtons(false);
      }
    }

    quickPipelineButton.addEventListener("click", () => {
      startPipeline("quick", quickPipelineButton);
    });

    runPipelineButton.addEventListener("click", () => {
      startPipeline("standard", runPipelineButton);
    });

    backButton.addEventListener("click", viewReport);

    pageLanguage.addEventListener("change", () => {
      applyPageLanguage(pageLanguage.value);
    });

    closeUploadDialog.addEventListener("click", () => {
      uploadDialog.close();
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const formData = new FormData(form);
      button.disabled = true;
      button.textContent = button.dataset.loadingText;
      setResult("", "Uploading dataset...");

      try {
        const response = await fetch("/api/datasets/upload", {
          method: "POST",
          body: formData
        });
        const payload = await response.json().catch(() => ({}));

        if (!response.ok || payload.ok === false) {
          throw new Error(payload.error || "Dataset upload failed.");
        }

        const path = payload.path || "";
        const rows = payload.rows ?? payload.row_count ?? "";
        const sourceFormat = payload.source_format || "";
        const config = [
          "queries_csv: " + path,
          "golden_csv: " + path
        ].join("\\n");

        setResult(
          "success",
          "<strong>Upload complete</strong>" +
            "<div>Saved dataset path: " + escapeHTML(path) + "</div>" +
            "<div>Source: " + escapeHTML(sourceFormat.toUpperCase()) + "</div>" +
            "<div>Rows: " + escapeHTML(rows) + "</div>" +
            "<pre>" + escapeHTML(config) + "</pre>"
        );
        await refreshReportList();
      } catch (error) {
        setResult("error", "<strong>Error:</strong> " + escapeHTML(error.message));
      } finally {
        button.disabled = false;
        button.textContent = button.dataset.readyText;
      }
    });

    if (reportSelect && reportFrame && reportPath) {
      function resizeReportFrame() {
        try {
          const frameDocument = reportFrame.contentDocument || reportFrame.contentWindow.document;
          const height = Math.max(
            frameDocument.documentElement.scrollHeight,
            frameDocument.body.scrollHeight
          );
          reportFrame.style.height = height + "px";
        } catch (error) {
          reportFrame.style.height = "720px";
        }
      }

      reportFrame.addEventListener("load", () => {
        applyReportLanguage(pageLanguage.value);
        resizeReportFrame();
      });
      window.addEventListener("resize", resizeReportFrame);

      reportSelect.addEventListener("change", () => {
        const option = reportSelect.selectedOptions[0];
        reportFrame.style.height = "720px";
        reportFrame.src = option.value;
        reportPath.dataset.reportPath = option.dataset.path;
        reportPath.textContent = currentTranslations().reportPath + ": " + option.dataset.path;
      });

      askModelForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const question = new FormData(askModelForm).get("question") || "";
        const language = pageLanguage.value || "en";
        const reportUrl = reportSelect.selectedOptions[0]?.value || "";
        askModelButton.disabled = true;
        askModelButton.textContent = askModelButton.dataset.loadingText;
        chatAnswer.className = "result chat-answer";
        chatAnswer.textContent = "Thinking...";

        try {
          const response = await fetch("/api/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({question, report_url: reportUrl, language})
          });
          const payload = await response.json().catch(() => ({}));
          if (!response.ok || payload.ok === false) {
            throw new Error(payload.error || "Model request failed.");
          }
          chatAnswer.className = "result success chat-answer";
          chatAnswer.innerHTML =
            "<strong>" + escapeHTML(payload.model || "Model") + "</strong>\\n" +
            escapeHTML(payload.answer || "") +
            "\\n\\nSaved QA supplement: " +
            escapeHTML(payload.supplemental_path || "") +
            (payload.supplemental_query_id ? " (" + escapeHTML(payload.supplemental_query_id) + ")" : "");
          await refreshReportList();
          resizeReportFrame();
        } catch (error) {
          chatAnswer.className = "result error chat-answer";
          chatAnswer.innerHTML = "<strong>Error:</strong> " + escapeHTML(error.message);
        } finally {
          askModelButton.disabled = false;
          askModelButton.textContent = askModelButton.dataset.readyText;
        }
      });
    }
    setQaMode("manual");
    renderQaCandidates();
    applyPageLanguage(window.localStorage.getItem("ragEvalPageLanguage") || "en");
  </script>
</body>
</html>""".replace(
        "{{REPORT_OPTIONS}}",
        report_options.replace("{{CHAT_PANEL}}", "{CHAT_PANEL}").replace(
            "{CHAT_PANEL}",
            """
        <section id="chatWindow" class="chat-window report-chat" aria-label="Ask model">
          <h2 class="section-heading" data-i18n="askModel">Ask model</h2>
          <form id="askModelForm">
            <label>
              <span data-i18n="question">Question</span>
              <textarea name="question" placeholder="What does this report show?" data-i18n-placeholder="questionPlaceholder" required></textarea>
            </label>
            <button type="submit" data-i18n="askModel" data-ready-text="Ask model" data-loading-text="Thinking...">Ask model</button>
          </form>
          <div id="chatAnswer" class="result chat-answer" role="status" aria-live="polite"></div>
          <button id="addReportAnswerToQaButton" class="secondary-button" type="button" data-i18n="addReportAnswerToQa" disabled>Fill QA</button>
        </section>""",
        ),
    ).replace("{{LANGUAGE_OPTIONS}}", language_options)


def render_console_chat_panel() -> str:
    return """
        <section id="chatWindow" class="chat-window report-chat" aria-label="Ask model">
          <h2 class="section-heading" data-i18n="askModel">Ask model</h2>
          <form id="askModelForm">
            <label>
              <span data-i18n="question">Question</span>
              <textarea name="question" placeholder="What does this report show?" data-i18n-placeholder="questionPlaceholder" required></textarea>
            </label>
            <button type="submit" data-i18n="askModel" data-ready-text="Ask model" data-loading-text="Thinking...">Ask model</button>
          </form>
          <div id="chatAnswer" class="result chat-answer" role="status" aria-live="polite"></div>
          <button id="addReportAnswerToQaButton" class="secondary-button" type="button" data-i18n="addReportAnswerToQa" disabled>Fill QA</button>
        </section>"""


def render_upload_page(result_reports: list[dict[str, str]] | None = None) -> str:
    if result_reports is None:
        result_reports = list_result_reports()
    report_options = (
        render_report_options(result_reports)
        .replace("{{CHAT_PANEL}}", render_console_chat_panel())
        .replace("{CHAT_PANEL}", render_console_chat_panel())
    )
    language_options = "\n".join(
        f'                <option value="{html.escape(code, quote=True)}">{html.escape(label)}</option>'
        for code, label, _prompt_name in LANGUAGE_OPTIONS
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AT&amp;S RAG Evaluation Console</title>
  <style>
    :root {{
      --ats-blue: #004b93;
      --ats-blue-dark: #003b74;
      --ats-blue-deep: #002f5f;
      --ats-blue-soft: #eaf3fb;
      --ats-cyan: #00a7c8;
      --ats-teal: #00a7c8;
      --ats-blue-50: #f6f9fd;
      --ats-blue-100: #edf4fb;
      --ats-blue-200: #d8e4f0;
      --ats-blue-300: #b7cce1;
      --ats-blue-700: #005bac;
      --ink: #0b2540;
      --muted: #526b86;
      --line: #d8e4f0;
      --surface: #ffffff;
      --bg: #f2fbfd;
      --paper: #fbfdff;
      --shadow-soft: 0 14px 36px rgba(8, 56, 76, 0.11);
      --error: #b3261e;
      --success: #207c50;
      --warning: #9a6b00;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      min-height: 100vh;
      font-family: -apple-system, BlinkMacSystemFont, Inter, "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        linear-gradient(180deg, rgba(255, 255, 255, .94), rgba(230, 245, 250, .82)),
        linear-gradient(90deg, rgba(23, 111, 149, .072) 1px, transparent 1px),
        linear-gradient(180deg, rgba(23, 111, 149, .05) 1px, transparent 1px);
      background-size: auto, 48px 48px, 48px 48px;
    }}

    button, input, textarea, select {{ font: inherit; }}

    button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 0;
      min-height: 38px;
      border: 0;
      border-radius: 6px;
      padding: 0 14px;
      color: #ffffff;
      background: var(--ats-blue);
      cursor: pointer;
      font-weight: 700;
      line-height: 1.2;
      text-align: center;
      white-space: normal;
      overflow-wrap: anywhere;
      box-shadow: 0 8px 18px rgba(0, 47, 95, 0.10);
      transition: background .18s ease, border-color .18s ease, color .18s ease, box-shadow .18s ease, transform .18s ease;
    }}

    button:hover {{ background: var(--ats-blue-dark); transform: translateY(-1px); }}
    button[disabled] {{ cursor: wait; opacity: .72; }}

    .ghost-button {{
      border: 1px solid var(--line);
      color: var(--ats-blue-dark);
      background: #ffffff;
    }}

    .ghost-button:hover {{
      color: var(--ats-blue);
      border-color: var(--ats-cyan);
      background: var(--ats-blue-soft);
    }}

    .app-shell {{
      display: grid;
      grid-template-columns: 216px minmax(0, 1fr);
      min-height: 100vh;
    }}

    .sidebar-nav {{
      position: sticky;
      top: 0;
      height: 100vh;
      padding: 18px 12px;
      color: #ffffff;
      background: linear-gradient(135deg, var(--ats-blue-dark), var(--ats-blue));
      border-right: 1px solid rgba(255, 255, 255, .12);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}

    .brand-lockup {{
      padding: 8px 10px 18px;
      border-bottom: 1px solid rgba(255, 255, 255, .24);
      margin-bottom: 8px;
      min-width: 0;
    }}

    .mark {{
      width: 74px;
      min-height: 42px;
      display: grid;
      place-items: center;
      border: 1px solid rgba(255,255,255,.62);
      border-radius: 8px;
      margin-bottom: 10px;
      background: rgba(255, 255, 255, .08);
      font-weight: 800;
    }}

    .brand-title {{ margin: 0; font-size: 16px; line-height: 1.18; overflow-wrap: anywhere; }}

    .nav-button {{
      display: flex;
      align-items: center;
      justify-content: center;
      width: 100%;
      min-height: 40px;
      border: 0;
      border-radius: 6px;
      padding: 9px 11px;
      text-align: center;
      color: #dbeeff;
      background: transparent;
      box-shadow: none;
    }}

    .nav-button::before {{
      content: "";
      width: 6px;
      height: 6px;
      margin-right: 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, .32);
    }}

    .nav-button:hover, .nav-button.active {{
      color: #ffffff;
      background: rgba(255, 255, 255, .12);
      transform: none;
    }}

    .nav-button.active::before {{
      background: #ffffff;
      box-shadow: 0 0 0 4px rgba(159, 203, 234, .22);
    }}

    .sidebar-footer {{
      margin-top: auto;
      padding: 12px 10px 0;
      border-top: 1px solid rgba(255, 255, 255, .24);
    }}

    .language-select {{
      display: grid;
      gap: 6px;
      color: #dbeeff;
      font-size: 12px;
    }}

    .workspace {{
      display: grid;
      gap: 18px;
      align-content: start;
      min-width: 0;
      padding: 22px clamp(18px, 2vw, 28px);
      background: transparent;
    }}

    .topbar {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
      width: 100%;
      max-width: 1320px;
      justify-self: center;
    }}

    .topbar > div {{ min-width: 0; }}

    .topbar-actions {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 10px;
      margin-left: auto;
      min-width: 0;
    }}

    .job-status-shell {{
      display: flex;
      justify-content: flex-end;
      width: auto;
      max-width: min(360px, 46vw);
      margin: 0;
      min-width: 0;
    }}

    .back-button {{
      min-height: 38px;
      border: 1px solid var(--line);
      color: var(--ats-blue-dark);
      background: #ffffff;
    }}

    .console-section {{
      display: none;
      gap: 16px;
    }}

    .console-section.active {{ display: grid; }}

    .panel {{
      display: grid;
      gap: 16px;
      width: 100%;
      max-width: 1320px;
      justify-self: center;
      padding: 20px;
      border: 1px solid var(--line);
      border-top: 4px solid var(--ats-cyan);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.98);
      box-shadow: var(--shadow-soft);
    }}

    .report-panel {{
      width: 100%;
      max-width: 1320px;
      justify-self: center;
      align-self: start;
      padding: 0;
      overflow: hidden;
      border-top: 0;
    }}

    .report-toolbar {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: center;
      flex-wrap: wrap;
      padding: 16px 18px;
      border-bottom: 1px solid rgba(183, 204, 225, .78);
      background: #fbfdff;
    }}

    .report-toolbar .section-heading {{
      font-size: 16px;
    }}

    .section-heading {{
      margin: 0;
      color: var(--ats-blue-dark);
      font-size: 1.15rem;
      line-height: 1.2;
      overflow-wrap: anywhere;
      text-wrap: pretty;
    }}

    .page-title {{
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0;
    }}

    .section-note {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: .92rem;
      line-height: 1.45;
      overflow-wrap: anywhere;
      text-wrap: pretty;
    }}

    .overview-board {{
      display: grid;
      gap: 16px;
      align-items: stretch;
    }}

    .overview-summary-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      min-width: 0;
    }}

    .overview-card, .overview-activity-strip {{
      display: grid;
      gap: 10px;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: #f8fbff;
    }}

    .overview-card span, .overview-activity-strip span {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}

    .overview-card strong, .overview-activity-strip strong {{
      display: block;
      color: var(--ats-blue-dark);
      font-size: 18px;
      line-height: 1.18;
      overflow-wrap: anywhere;
    }}

    .overview-count-card strong {{
      font-size: 30px;
      line-height: 1;
    }}

    .overview-activity-strip {{
      grid-template-columns: minmax(120px, .25fr) minmax(0, 1fr);
      align-items: center;
    }}

    .overview-actions {{
      justify-content: flex-end;
    }}

    form, label {{
      display: grid;
      gap: 10px;
    }}

    label {{
      color: var(--ats-blue-dark);
      font-weight: 700;
      min-width: 0;
      overflow-wrap: anywhere;
    }}

    input, textarea, select {{
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      color: var(--ink);
      background: #ffffff;
      line-height: 1.35;
    }}

    select {{
      border-color: var(--line);
      background: #ffffff;
      cursor: pointer;
    }}

    input:focus, textarea:focus, select:focus {{
      outline: 3px solid rgba(0, 167, 200, 0.2);
      border-color: var(--ats-cyan);
    }}

    textarea {{ min-height: 100px; resize: vertical; }}

    .toolbar-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      min-width: 0;
    }}

    .toolbar-actions button {{
      flex: 0 1 auto;
      max-width: 100%;
    }}

    .report-content-stack, .report-main, .report-workspace {{
      display: grid;
      gap: 0;
      min-width: 0;
    }}

    .report-control-strip {{
      display: grid;
      grid-template-columns: minmax(260px, 520px) minmax(0, 1fr);
      gap: 12px;
      align-items: center;
      min-width: 0;
      padding: 14px 18px;
      border-bottom: 1px solid rgba(183, 204, 225, .78);
      background: #ffffff;
    }}

    .report-picker {{
      gap: 6px;
    }}

    .report-picker > span {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
      text-transform: uppercase;
    }}

    .report-path {{
      border: 1px solid var(--line);
      border-left: 4px solid var(--ats-cyan);
      border-radius: 6px;
      padding: 12px 14px;
      color: var(--ats-blue-dark);
      background: var(--ats-blue-soft);
      overflow-wrap: anywhere;
      font-weight: 700;
    }}

    .report-body-grid {{
      display: grid;
      grid-template-columns: minmax(620px, 1fr) minmax(300px, 330px);
      gap: 0;
      align-items: start;
      min-width: 0;
    }}

    .report-preview-area, .report-chat-area {{
      min-width: 0;
    }}

    .report-preview-area {{
      padding: 14px 0 14px 18px;
    }}

    .report-chat-area {{
      padding: 14px 18px;
      border-left: 1px solid rgba(183, 204, 225, .78);
      background: #fbfdff;
    }}

    .report-frame {{
      display: block;
      width: 100%;
      height: clamp(520px, calc(100vh - 248px), 760px);
      min-height: 0;
      border: 1px solid rgba(183, 204, 225, .9);
      border-radius: 8px;
      background: #ffffff;
      overflow: auto;
    }}

    .chat-window {{
      border-top: 1px solid var(--line);
      padding-top: 18px;
    }}

    .report-chat {{
      position: sticky;
      top: 16px;
      display: grid;
      gap: 10px;
      min-width: 0;
      border-top: 0;
      border-left: 0;
      padding-top: 0;
      padding-left: 0;
    }}

    .report-chat .section-heading {{
      font-size: 15px;
    }}

    .report-chat form {{
      gap: 10px;
    }}

    .report-chat textarea {{
      min-height: 98px;
    }}

    .report-chat button[type="submit"] {{
      width: 100%;
    }}

    .result {{
      min-height: 64px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
      background: var(--ats-blue-soft);
      overflow-wrap: anywhere;
    }}

    .result:empty::before {{
      content: "Saved under data/uploaded_datasets/. Exact dataset path will appear here after upload.";
      color: var(--muted);
    }}

    .pipeline-result:empty::before {{
      content: "Pipeline server status will appear here.";
      color: var(--muted);
    }}

    .result.error {{ color: var(--error); background: #fff4f2; border-color: #f2b8b5; }}
    .result.success {{ color: var(--ats-blue-dark); background: var(--ats-blue-100); border-color: var(--ats-blue-300); }}
    .result pre {{ margin: 10px 0 0; white-space: pre-wrap; }}

    .records-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}

    .records-table th, .records-table td {{
      padding: 9px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
      overflow-wrap: anywhere;
    }}

    .records-table th {{
      color: var(--ats-blue-dark);
      background: #eef4f8;
    }}

    .records-table tbody tr:last-child td {{
      border-bottom: 0;
    }}

    .dataset-inventory {{
      display: grid;
      gap: 10px;
      min-width: 0;
    }}

    #datasets .panel {{
      width: 100%;
      max-width: 1360px;
      justify-self: center;
      padding: 0;
      overflow: hidden;
      border: 0;
      border-top: 0;
      border-radius: 0;
      background: transparent;
      box-shadow: none;
    }}

    .dataset-section-stack {{
      --qa-surface: #f6f9fd;
      --qa-surface-strong: var(--ats-blue-100);
      --qa-border: rgba(216, 228, 240, .9);
      --qa-border-solid: #d8e4f0;
      --qa-accent: var(--ats-blue-700);
      --qa-accent-soft: #eef6ff;
      --qa-amber: var(--ats-blue-700);
      --qa-card-radius: 16px;
      --qa-button-radius: 10px;
      --qa-control-height: 44px;
      --qa-control-radius: 10px;
      --qa-primary-action-width: 150px;
      --qa-primary-action-height: 52px;
      --qa-shadow: 0 10px 26px rgba(30, 49, 68, .08);
      display: grid;
      gap: 16px;
      min-width: 0;
      background: transparent;
    }}

    .dataset-action-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
      padding: 16px 18px;
      border-bottom: 1px solid rgba(183, 204, 225, .78);
      background: #fbfdff;
      min-width: 0;
    }}

    .dataset-action-header .section-heading {{
      font-size: 16px;
    }}

    .dataset-create-grid {{
      display: grid;
      gap: 18px;
      min-width: 0;
      padding: 20px 24px 24px;
      background: #f6f9fd;
    }}

    .dataset-flow-card {{
      min-width: 0;
      border: 1px solid var(--qa-border-solid);
      border-radius: var(--qa-card-radius);
      background: #ffffff;
      box-shadow: var(--qa-shadow);
    }}

    .dataset-step-card {{
      display: grid;
      gap: 16px;
      padding: 22px 24px;
    }}

    .dataset-step-dataset {{
      min-height: 96px;
      gap: 14px;
      padding: 18px 24px;
      box-shadow: 0 8px 24px rgba(15, 45, 80, .04);
    }}

    .dataset-step-heading {{
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
      color: #0b355b;
      font-size: 16px;
      font-weight: 600;
    }}

    .dataset-step-heading small {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}

    .dataset-step-title {{
      color: #0b2f5b;
      font-size: 16px;
      font-weight: 600;
      line-height: 1.2;
    }}

    .dataset-step-dataset .dataset-step-heading {{
      align-self: start;
    }}

    .dataset-step-badge {{
      display: inline-grid;
      place-items: center;
      width: 22px;
      height: 22px;
      border-radius: 999px;
      color: var(--ats-blue);
      background: #eaf2fb;
      font-size: 12px;
      font-weight: 900;
      line-height: 1;
      flex: 0 0 auto;
      box-shadow: 0 6px 14px rgba(0, 75, 147, .10);
    }}

    .dataset-inventory-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      min-width: 0;
    }}

    .dataset-list-title {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }}

    .dataset-list-actions {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      margin-left: auto;
      flex-wrap: wrap;
    }}

    .dataset-list-actions button {{
      min-height: 32px;
      padding: 0 12px;
      white-space: nowrap;
    }}

    .manual-qa-panel {{
      display: grid;
      gap: 16px;
      min-width: 0;
    }}

    .dataset-workbench {{
      position: relative;
      padding: 22px;
      border: 1px solid var(--qa-border-solid);
      border-radius: var(--qa-card-radius);
      background: #ffffff;
      box-shadow: var(--qa-shadow);
      overflow: hidden;
    }}

    .dataset-workbench::before {{
      content: "";
      position: absolute;
      inset: 0 0 auto;
      height: 1px;
      background: #d8e4f0;
    }}

    .manual-qa-panel > .section-heading {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 15px;
    }}

    .manual-qa-form {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      grid-template-areas:
        "body"
        "generate";
      gap: 12px;
      min-width: 0;
    }}

    .qa-setup-grid {{
      display: grid;
      grid-template-columns: minmax(360px, 1fr) minmax(240px, .52fr);
      gap: 14px;
      align-items: stretch;
      min-width: 0;
    }}

    .dataset-module-row {{
      align-items: stretch;
    }}

    .qa-mode-selector {{
      display: grid;
      grid-template-columns: minmax(64px, max-content) minmax(0, 1fr);
      gap: 12px;
      align-items: start;
      width: 100%;
      max-width: 100%;
      min-width: 0;
      padding: 14px;
      border: 1px solid rgba(183, 204, 225, .84);
      border-radius: 8px;
      background: #f8fbff;
    }}

    .qa-mode-selector > span {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
      white-space: nowrap;
    }}

    .qa-mode-selector select {{
      min-height: 36px;
    }}

    .qa-mode-panel {{
      min-width: 0;
    }}

    .ai-qa-panel,
    .report-qa-panel,
    .qa-candidate-panel {{
      display: grid;
      gap: 12px;
      min-width: 0;
      padding: 14px;
      border: 1px solid rgba(183, 204, 225, .84);
      border-radius: 8px;
      background: #fbfdff;
    }}

    .ai-qa-form {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(120px, 160px) minmax(160px, 220px);
      gap: 12px;
      align-items: end;
      min-width: 0;
    }}

    .ai-qa-form textarea {{
      min-height: 96px;
      resize: vertical;
    }}

    .answer-generation-form {{
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      align-items: start;
    }}

    .answer-generation-actions {{
      display: grid;
      grid-template-columns: var(--qa-primary-action-width) 132px 118px;
      gap: 16px;
      align-items: center;
    }}

    #generateAnswerButton {{
      width: var(--qa-primary-action-width);
      min-height: var(--qa-primary-action-height);
      height: var(--qa-primary-action-height);
      color: #ffffff;
      background: #005bac;
      border: 1px solid var(--ats-blue-700);
      border-radius: var(--qa-control-radius);
      box-shadow: 0 8px 18px rgba(0, 91, 172, .16);
    }}

    #generateAnswerButton:hover {{
      color: #ffffff;
      background: #004a98;
      border-color: #004a98;
      box-shadow: 0 10px 22px rgba(0, 91, 172, .18);
    }}

    #generateAnswerButton::before {{
      content: "✦";
      margin-right: 8px;
      font-size: 13px;
      line-height: 1;
    }}

    #regenerateAnswerButton,
    #clearGeneratedAnswerButton {{
      min-height: var(--qa-control-height);
      color: var(--ats-blue-dark);
      background: #ffffff;
      border: 1px solid #d8e4f0;
      border-radius: var(--qa-control-radius);
      box-shadow: none;
    }}

    #regenerateAnswerButton {{
      width: 132px;
    }}

    #clearGeneratedAnswerButton {{
      width: 118px;
    }}

    #regenerateAnswerButton::before {{
      content: "↻";
      margin-right: 8px;
    }}

    #clearGeneratedAnswerButton::before {{
      content: "⌫";
      margin-right: 8px;
    }}

    #regenerateAnswerButton:hover,
    #clearGeneratedAnswerButton:hover {{
      background: #ffffff;
      border-color: rgba(127, 153, 175, .5);
      box-shadow: 0 6px 14px rgba(21, 50, 76, .06);
    }}

    .qa-context-bar {{
      grid-area: context;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(220px, .72fr) minmax(160px, 220px);
      gap: 18px;
      align-items: center;
      padding: 0;
      border-bottom: 0;
    }}

    .dataset-step-toolbar {{
      display: grid;
      grid-template-columns: minmax(360px, 470px) minmax(220px, 1fr) minmax(160px, 220px);
      gap: 24px;
      align-items: center;
      min-width: 0;
      padding-left: 0;
    }}

    .dataset-step-dataset .dataset-step-toolbar {{
      grid-area: auto;
      width: 100%;
      justify-self: stretch;
    }}

    .qa-dataset-picker {{
      min-width: 0;
    }}

    .qa-dataset-control {{
      display: grid;
      grid-template-columns: max-content minmax(260px, 360px);
      gap: 14px;
      align-items: center;
      min-width: 0;
    }}

    .dataset-step-dataset .qa-dataset-control {{
      justify-self: start;
    }}

    .qa-dataset-control > span:first-child {{
      color: var(--ats-blue-dark);
      font-size: 13px;
      font-weight: 800;
      white-space: nowrap;
    }}

    .qa-dataset-control select {{
      min-height: var(--qa-control-height);
      border-radius: var(--qa-control-radius);
    }}

    .qa-dataset-upload-button {{
      min-height: var(--qa-control-height);
      width: 100%;
      align-self: end;
      white-space: nowrap;
    }}

    .dataset-step-dataset .qa-dataset-upload-button {{
      color: var(--ats-blue);
      background: #ffffff;
      border: 1px solid rgba(23, 111, 149, .64);
      box-shadow: none;
    }}

    .dataset-step-dataset .qa-dataset-upload-button:hover {{
      color: #ffffff;
      background: var(--ats-blue);
      border-color: var(--ats-blue);
      box-shadow: 0 8px 18px rgba(23, 111, 149, .16);
    }}

    .dataset-upload-cell {{
      display: grid;
      gap: 8px;
      justify-items: stretch;
      min-width: 0;
    }}

    .dataset-upload-cell small {{
      color: var(--muted);
      font-size: 11px;
      text-align: center;
    }}

    #datasets input,
    #datasets textarea,
    #datasets select {{
      border-color: #d8e4f0;
      border-radius: var(--qa-control-radius);
      background: #ffffff;
    }}

    #datasets button {{
      min-height: var(--qa-control-height);
      border-radius: var(--qa-control-radius);
      font-size: 14px;
      font-weight: 600;
    }}

    .dataset-file-summary {{
      display: grid;
      grid-template-columns: 28px minmax(0, 1fr);
      gap: 10px;
      align-items: center;
      min-width: 0;
      min-height: var(--qa-control-height);
      padding: 7px 10px;
      border: 1px solid #d8e4f0;
      border-radius: 8px;
      color: var(--ats-blue-dark);
      background: #f8fbfd;
    }}

    .dataset-file-icon {{
      display: inline-grid;
      place-items: center;
      width: 28px;
      height: 28px;
      border-radius: 6px;
      color: var(--ats-blue);
      background: #eef6fb;
      font-size: 13px;
      font-weight: 900;
    }}

    .dataset-file-summary strong {{
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 14px;
      line-height: 1.25;
    }}

    .qa-reference-grid {{
      grid-area: references;
      display: block;
      min-width: 0;
    }}

    .qa-save-panel {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(150px, 220px);
      gap: 12px;
      align-items: center;
      min-width: 0;
      padding: 0;
    }}

    .qa-save-panel button[type='submit'] {{
      min-height: var(--qa-control-height);
      box-shadow: 0 10px 22px rgba(0, 47, 95, .16);
    }}

    .qa-save-panel button[type='submit']:hover {{
      background: #003e72;
      box-shadow: 0 12px 26px rgba(0, 47, 95, .2);
    }}

    .qa-reference-grid .dataset-preview-details {{
      grid-area: auto;
    }}

    .retrieved-passages-list {{
      display: grid;
      gap: 10px;
      padding: 12px;
    }}

    .retrieved-passage-item {{
      padding: 10px 12px;
      border: 1px solid rgba(174, 194, 211, .66);
      border-radius: 8px;
      background: #ffffff;
      color: var(--ink);
      font-size: 12px;
      line-height: 1.5;
      overflow-wrap: anywhere;
    }}

    .report-qa-panel p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }}

    .dataset-preview-details,
    .qa-candidate-panel,
    .dataset-list-panel {{
      border: 1px solid var(--qa-border);
      border-radius: 8px;
      background: var(--qa-surface);
      min-width: 0;
    }}

    .dataset-list-panel,
    .dataset-preview-details {{
      background: rgba(255, 255, 255, .72);
      box-shadow: 0 10px 24px rgba(21, 50, 76, .04);
    }}

    .dataset-action-card {{
      gap: 12px;
      box-shadow: 0 14px 28px rgba(21, 50, 76, .06);
    }}

    .dataset-step-actions {{
      gap: 16px;
      grid-template-columns: minmax(0, 1fr);
      grid-template-areas:
        "heading"
        "actions"
        "references";
      align-items: stretch;
    }}

    .dataset-step-actions > .dataset-step-heading {{
      grid-area: heading;
      align-self: start;
      justify-self: start;
    }}

    .dataset-step-actions > .dataset-action-main {{
      grid-area: actions;
    }}

    .dataset-step-actions > .qa-reference-grid {{
      grid-area: references;
    }}

    .dataset-step-edit .manual-qa-row-body {{
      width: 100%;
      grid-area: auto;
      justify-self: stretch;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
      align-items: stretch;
      justify-content: stretch;
    }}

    .dataset-step-edit .qa-question-card,
    .dataset-step-edit .qa-answer-card {{
      width: 100%;
      justify-self: stretch;
    }}

    .dataset-action-main {{
      min-width: 0;
      width: 100%;
    }}

    .dataset-action-row {{
      display: grid;
      grid-template-columns: minmax(360px, auto) minmax(64px, 1fr) minmax(480px, 620px);
      align-items: center;
      gap: 30px;
      min-width: 0;
      width: 100%;
    }}

    .dataset-action-section {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-sizing: border-box;
      gap: 16px;
      min-width: 0;
      min-height: 50px;
      padding: 0;
      border: 0;
      border-radius: 0;
      background: transparent;
    }}

    .dataset-action-buttons {{
      display: flex;
      gap: 16px;
      align-items: center;
      min-width: 0;
      flex: 0 0 auto;
      justify-content: flex-end;
    }}

    .dataset-action-buttons button {{
      min-width: 118px;
    }}

    .dataset-generation-section .dataset-action-buttons {{
      width: auto;
    }}

    .dataset-generation-section {{
      flex: 0 0 auto;
    }}

    .dataset-save-section {{
      grid-column: 3;
      justify-self: end;
      justify-content: flex-end;
      margin-left: auto;
      width: max-content;
      background: transparent;
    }}

    .dataset-save-section .qa-save-panel {{
      display: flex;
      justify-content: flex-end;
      flex: 0 0 auto;
      width: auto;
      max-width: none;
      margin-left: auto;
    }}

    .dataset-action-card .qa-reference-grid {{
      display: block;
    }}

    .dataset-action-card .retrieved-passages-details {{
      width: 100%;
      max-width: none;
      box-shadow: none;
    }}

    .dataset-action-card .dataset-preview-summary {{
      min-height: 52px;
      padding: 11px 16px;
      border-radius: 8px;
      background: #fbfdff;
      border: 1px solid rgba(183, 204, 225, .72);
    }}

    .dataset-action-card .dataset-preview-summary {{
      justify-content: flex-start;
    }}

    .dataset-action-card .dataset-preview-summary::after {{
      margin-left: auto;
    }}

    .dataset-preview-summary-title {{
      color: #0b2f5b;
      font-size: 14px;
      font-weight: 700;
    }}

    .dataset-preview-summary-title::before {{
      content: "▤";
      margin-right: 8px;
      color: var(--ats-blue-700);
      font-size: 14px;
    }}

    .dataset-action-card .dataset-preview-summary small {{
      color: #5f7390;
      font-size: 12px;
      font-weight: 500;
    }}

    .dataset-save-shell {{
      display: grid;
      grid-template-columns: var(--qa-primary-action-width);
      gap: 0;
      align-items: stretch;
      justify-content: end;
      min-width: 0;
      width: auto;
      min-height: var(--qa-primary-action-height);
      padding: 0;
      border: 0;
      border-radius: 0;
      background: transparent;
      flex: 0 0 auto;
      max-width: none;
    }}

    .dataset-save-info-card {{
      display: grid;
      grid-template-columns: 22px minmax(0, 1fr);
      gap: 12px;
      align-items: center;
      min-width: 0;
      min-height: 52px;
      padding: 8px 14px;
      border: 1px solid rgba(183, 204, 225, .72);
      border-radius: 12px;
      background: #f8fbff;
    }}

    .dataset-save-state-dot {{
      display: inline-grid;
      place-items: center;
      width: 22px;
      height: 22px;
      border-radius: 999px;
      color: var(--ats-blue);
      background: #eef6ff;
      font-size: 13px;
      font-weight: 900;
    }}

    .dataset-save-button-group {{
      display: flex;
      align-items: stretch;
      justify-content: flex-end;
      justify-self: end;
      width: var(--qa-primary-action-width);
      min-width: 0;
      margin-left: auto;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 10px 22px rgba(0, 47, 95, .16);
    }}

    .dataset-save-button-group button {{
      border-radius: 8px;
      box-shadow: none;
    }}

    .dataset-save-button-group button[type='submit'] {{
      width: var(--qa-primary-action-width);
      min-height: var(--qa-primary-action-height);
      height: var(--qa-primary-action-height);
      padding: 0 14px 0 18px;
      justify-content: space-between;
    }}

    .dataset-save-button-group button[type='submit']::after {{
      content: "⌄";
      margin-left: 10px;
      font-size: 14px;
      line-height: 1;
      opacity: .92;
    }}

    .dataset-preview-details {{
      grid-area: preview;
    }}

    .retrieved-passages-details {{
      grid-area: passages;
    }}

    .dataset-secondary-panel {{
      background: var(--qa-surface);
    }}

    .dataset-secondary-preview {{
      border-left-color: var(--qa-accent);
    }}

    .dataset-secondary-candidates {{
      border-left-color: var(--qa-accent);
    }}

    .dataset-secondary-inventory {{
      border-left-color: var(--qa-accent);
    }}

    .dataset-preview-summary,
    .qa-candidate-summary,
    .dataset-list-summary {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      min-height: 42px;
      padding: 10px 12px;
      color: #0b355b;
      font-size: 13px;
      font-weight: 800;
      cursor: pointer;
      min-width: 0;
    }}

    .dataset-preview-summary::-webkit-details-marker,
    .qa-candidate-summary::-webkit-details-marker {{
      display: none;
    }}

    .dataset-preview-summary::after,
    .qa-candidate-summary::after {{
      content: "⌄";
      display: inline-grid;
      place-items: center;
      width: 22px;
      height: 22px;
      border-radius: 999px;
      color: var(--ats-blue);
      background: #e7f0f7;
      font-weight: 900;
    }}

    details[open] > .dataset-preview-summary::after,
    details[open] > .qa-candidate-summary::after {{
      content: "⌃";
    }}

    .qa-candidate-summary strong,
    .dataset-list-summary strong {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 24px;
      min-height: 22px;
      border-radius: 999px;
      padding: 0 8px;
      color: var(--ats-blue-dark);
      background: #eaf2fb;
      font-size: 12px;
    }}

    .dataset-list-summary {{
      cursor: default;
      padding: 4px 0 0;
    }}

    .qa-candidate-actions {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}

    .qa-candidate-actions .secondary-button {{
      color: var(--ats-blue-dark);
      background: #ffffff;
      border: 1px solid var(--line);
      box-shadow: none;
    }}

    .qa-candidate-table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      min-width: 760px;
    }}

    .qa-candidate-table th,
    .qa-candidate-table td {{
      border-bottom: 1px solid rgba(183, 204, 225, .76);
      padding: 10px;
      text-align: left;
      vertical-align: top;
    }}

    .qa-candidate-table th {{
      color: #174166;
      background: #eef6f3;
      font-size: 12px;
      font-weight: 800;
    }}

    .qa-candidate-table th:first-child,
    .qa-candidate-table td:first-child {{
      width: 52px;
      text-align: center;
    }}

    .qa-candidate-table th:last-child,
    .qa-candidate-table td:last-child {{
      width: 88px;
      text-align: center;
    }}

    .qa-candidate-table textarea {{
      min-height: 76px;
      resize: vertical;
      font-size: 12px;
      line-height: 1.45;
    }}

    .qa-candidate-delete {{
      min-height: 32px;
      padding: 0 10px;
      color: var(--error);
      background: #fff7f6;
      border: 1px solid #f2b8b5;
      box-shadow: none;
    }}

    .qa-candidate-empty {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}

    .manual-qa-row {{
      display: grid;
      gap: 12px;
      min-width: 0;
    }}

    .manual-qa-row-top {{
      grid-template-columns: minmax(0, 1fr);
      align-content: start;
      padding: 14px;
      border: 1px solid var(--qa-border);
      border-radius: 8px;
      background: var(--qa-surface);
    }}

    .manual-qa-row-body {{
      grid-area: body;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      align-items: start;
    }}

    .dataset-module {{
      min-width: 0;
      border: 0;
      border-left: 0;
      border-radius: 0;
      background: transparent;
    }}

    .dataset-module-dataset {{
      background: transparent;
    }}

    .dataset-module-method {{
      background: transparent;
    }}

    .dataset-module-save {{
      background: transparent;
    }}

    .dataset-module-entry {{
      padding: 0;
      border: 0;
      border-radius: 0;
      background: transparent;
    }}

    .qa-input-panel {{
      border: 0;
      box-shadow: none;
    }}

    .qa-card {{
      position: relative;
      min-width: 0;
      border: 1px solid var(--qa-border-solid);
      border-radius: 8px;
      padding: 16px;
      background: #ffffff;
    }}

    .manual-qa-form textarea {{
      min-height: 142px;
      resize: vertical;
    }}

    .qa-card textarea {{
      min-height: 140px;
      border-color: #d8e4f0;
      border-radius: 10px;
      box-shadow: inset 0 1px 2px rgba(21, 50, 76, .03);
    }}

    .manual-qa-form textarea:focus,
    .manual-qa-form select:focus {{
      border-color: rgba(0, 167, 200, .7);
      box-shadow: 0 0 0 3px rgba(0, 167, 200, .14);
      outline: 0;
    }}

    .manual-qa-row-body label {{
      gap: 8px;
    }}

    .qa-field-label {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }}

    .qa-field-mark {{
      display: inline-grid;
      place-items: center;
      width: 22px;
      height: 22px;
      border-radius: 999px;
      color: #ffffff;
      background: var(--ats-blue);
      font-size: 12px;
      font-weight: 800;
      line-height: 1;
    }}

    .qa-character-count {{
      justify-self: end;
      color: var(--muted);
      font-size: 11px;
      font-weight: 650;
    }}

    .manual-qa-selected-path {{
      min-width: 0;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      line-height: 1.45;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .dataset-step-toolbar > .manual-qa-selected-path {{
      display: none;
    }}

    .dataset-rows-table td {{
      vertical-align: middle;
    }}

    .dataset-row-question,
    .dataset-row-answer {{
      max-width: 360px;
      color: var(--ink);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .dataset-row-answer {{
      color: var(--muted);
    }}

    .dataset-row-actions {{
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: flex-end;
    }}

    .dataset-row-actions button,
    .dataset-view-all-button {{
      min-width: 34px;
      min-height: 34px;
      padding: 0 11px;
      color: var(--ats-blue);
      background: #ffffff;
      border: 1px solid rgba(174, 194, 211, .72);
      box-shadow: none;
    }}

    .dataset-row-actions button {{
      width: 34px;
      padding: 0;
    }}

    .dataset-view-all-button::after {{
      content: "›";
      margin-left: 8px;
      font-size: 16px;
      line-height: 1;
    }}

    .qa-context-bar button {{
      min-height: 40px;
      width: 100%;
    }}

    #manualQaResult {{
      min-height: 0;
      padding: 12px 14px;
    }}

    #manualQaResult:empty {{
      display: none;
    }}

    #manualQaResult:empty::before {{
      content: "";
    }}

    .manual-qa-save-target {{
      display: grid;
      gap: 2px;
      align-content: center;
      min-width: 0;
      min-height: 0;
      padding: 0;
      border: 0;
      border-radius: 0;
      background: transparent;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}

    .manual-qa-save-target small {{
      color: #5f7390;
      font-size: 12px;
      font-weight: 500;
      max-width: none;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .manual-qa-save-target strong {{
      color: var(--ats-blue-dark);
      font-weight: 760;
    }}

    .dataset-save-status {{
      color: #0b2f5b;
      font-size: 13px;
      font-weight: 700;
    }}

    .save-target-title {{
      color: #0b2f5b;
      font-weight: 700;
    }}

    .dataset-preview-meta {{
      color: var(--muted);
      font-weight: 600;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}

    .dataset-list-panel {{
      display: grid;
      gap: 14px;
      min-width: 0;
      padding: 18px 22px;
      background: #ffffff;
    }}

    .dataset-path-cell {{
      max-width: 420px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}

    #datasets .table-scroll {{
      border: 1px solid rgba(216, 228, 240, .78);
      border-radius: 8px;
      overflow: auto;
      background: #ffffff;
    }}

    #datasets .records-table {{
      font-size: 13px;
    }}

    #datasets .records-table th {{
      padding: 12px 16px;
      color: #0b2f5b;
      background: #f4f8fc;
      position: sticky;
      top: 0;
      z-index: 1;
      border-bottom-color: #d8e4f0;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0;
    }}

    #datasets .records-table td {{
      padding: 14px 16px;
      border-bottom-color: rgba(216, 228, 240, .82);
      background: #ffffff;
    }}

    #datasets .records-table tbody tr:hover td {{
      background: #fbfdff;
    }}

    .dataset-name-cell {{
      color: var(--ats-blue-dark);
      font-weight: 750;
    }}

    .dataset-name-cell small {{
      color: var(--muted);
      font-weight: 650;
    }}

    .dataset-source-badge {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 750;
      line-height: 1.2;
      white-space: nowrap;
    }}

    .dataset-source-badge.config {{
      color: var(--ats-blue-dark);
      background: var(--ats-blue-100);
    }}

    .dataset-source-badge.uploaded {{
      color: var(--ats-blue);
      background: #f2f8fd;
      border: 1px solid rgba(23, 111, 149, .24);
    }}

    .run-dataset-panel {{
      display: grid;
      gap: 10px;
      min-width: 0;
    }}

    .run-control-grid {{
      display: grid;
      gap: 18px;
      min-width: 0;
    }}

    .run-control-grid > .run-dataset-panel {{
      padding: 14px;
      border: 1px solid rgba(183, 204, 225, .82);
      border-radius: 8px;
      background: #fbfdff;
    }}

    .run-parameter-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      align-items: end;
      min-width: 0;
    }}

    .run-dataset-choices {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      min-width: 0;
    }}

    .run-parameter-control {{
      display: grid;
      grid-template-columns: minmax(120px, 1fr) minmax(100px, .72fr);
      gap: 10px;
      min-width: 0;
    }}

    .run-dataset-choice {{
      display: flex;
      gap: 10px;
      align-items: center;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #f8fbff;
      color: var(--ats-blue-dark);
      cursor: pointer;
    }}

    .run-submit-panel {{
      display: grid;
      gap: 10px;
      min-width: 0;
    }}

    .run-submit-panel::before {{
      content: "";
      min-height: calc(1.15rem * 1.2);
    }}

    .run-action-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(96px, .34fr);
      gap: 10px;
      align-items: end;
      min-width: 0;
    }}

    .run-submit-panel button {{
      width: 100%;
      min-height: 44px;
    }}

    .terminate-button {{
      border: 1px solid #f2b8b5;
      color: var(--error);
      background: #fff7f6;
      box-shadow: none;
    }}

    .terminate-button:hover:not([disabled]) {{
      color: #ffffff;
      background: var(--error);
    }}

    .terminate-button[disabled] {{
      cursor: not-allowed;
      color: #9aa9b5;
      border-color: var(--line);
      background: #f7fafc;
      opacity: .78;
    }}

    .run-dataset-choice input {{
      width: auto;
      min-width: 18px;
    }}

    .run-dataset-choice span {{
      min-width: 0;
      overflow-wrap: anywhere;
      font-weight: 700;
    }}

    .output-cell {{
      display: grid;
      gap: 6px;
      min-width: 132px;
    }}

    .output-action {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: fit-content;
      max-width: 100%;
      min-height: 30px;
      border-radius: 6px;
      padding: 0 10px;
      color: #ffffff;
      background: var(--ats-blue);
      font-weight: 700;
      line-height: 1.2;
      text-decoration: none;
      white-space: normal;
      overflow-wrap: anywhere;
    }}

    .output-action:hover {{ background: var(--ats-blue-dark); }}

    .status-pill {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border-radius: 999px;
      padding: 0 9px;
      color: var(--ats-blue-dark);
      background: var(--ats-blue-soft);
      font-size: 12px;
      font-weight: 700;
      line-height: 1.2;
    }}

    .status-completed, .status-dry_run, .status-skipped_existing {{
      color: var(--ats-blue-dark);
      background: var(--ats-blue-100);
    }}

    .status-failed, .status-missing_generated_answers {{
      color: var(--error);
      background: #fff4f2;
    }}

    .status-running, .status-starting {{
      color: var(--ats-blue);
      background: #f2f8fd;
    }}

    .combo-stack, .row-actions {{
      display: grid;
      gap: 6px;
      min-width: 0;
    }}

    .combo-stack small {{
      color: var(--muted);
      overflow-wrap: anywhere;
    }}

    .report-action {{
      width: fit-content;
      max-width: 100%;
      border: 1px solid var(--line);
      color: var(--ats-blue-dark);
      background: #ffffff;
    }}

    .report-action:hover {{
      color: var(--ats-blue);
      border-color: var(--ats-cyan);
      background: var(--ats-blue-soft);
    }}

    .job-status-bar {{
      display: flex;
      width: fit-content;
      max-width: 100%;
      gap: 10px;
      align-items: center;
      margin: 0;
      padding: 6px 10px;
      border: 1px solid rgba(183, 204, 225, .72);
      border-radius: 999px;
      background: rgba(255, 255, 255, .72);
      box-shadow: none;
      backdrop-filter: blur(12px);
      min-width: 0;
    }}

    .job-status-main, .job-status-details {{
      display: flex;
      align-items: center;
      gap: 6px;
      min-width: 0;
    }}

    .job-status-title {{
      color: var(--ats-blue-dark);
      font-size: 12px;
      font-weight: 700;
      line-height: 1.2;
      white-space: nowrap;
    }}

    .job-status-progress {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }}

    .progress-track {{
      flex: 0 0 104px;
      width: 104px;
      max-width: 104px;
      height: 4px;
      border-radius: 999px;
      overflow: hidden;
      background: #dce8f3;
    }}

    .progress-fill {{
      display: block;
      width: 0%;
      height: 100%;
      border-radius: inherit;
      background: #9fb2c4;
      transition: width .28s ease, background .22s ease;
    }}

    .job-status-bar[data-status="starting"] .progress-fill,
    .job-status-bar[data-status="running"] .progress-fill {{
      background: linear-gradient(90deg, var(--ats-blue-dark), var(--ats-cyan), var(--ats-blue));
      background-size: 180% 100%;
      animation: progressFlow 1.4s linear infinite;
    }}

    .job-status-bar[data-status="cancelling"] .progress-fill {{
      background: linear-gradient(90deg, var(--ats-blue-deep), var(--ats-blue), var(--ats-blue-deep));
      background-size: 180% 100%;
      animation: progressFlow 1.1s linear infinite;
    }}

    .job-status-bar[data-status="completed"] .progress-fill {{
      background: var(--ats-blue);
    }}

    .job-status-bar[data-status="cancelled"] .progress-fill {{
      background: #7baeca;
    }}

    .job-status-bar[data-status="failed"] .progress-fill {{
      background: #ff3b30;
    }}

    @keyframes progressFlow {{
      from {{ background-position: 0% 50%; }}
      to {{ background-position: 180% 50%; }}
    }}

    .job-meta {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.2;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      overflow-wrap: anywhere;
    }}

    .job-status-details:empty, #jobCurrentTask:empty {{
      display: none;
    }}

    dialog {{
      width: min(620px, calc(100% - 32px));
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0;
      color: var(--ink);
      background: #ffffff;
      box-shadow: 0 24px 70px rgba(6, 40, 70, 0.34);
    }}

    dialog::backdrop {{ background: rgba(6, 40, 70, 0.48); }}
    .dialog-window {{ display: grid; gap: 22px; padding: 26px; }}
    .dialog-titlebar {{ display: flex; justify-content: space-between; gap: 16px; align-items: start; flex-wrap: wrap; }}
    [hidden] {{ display: none !important; }}
    .chat-answer {{
      min-height: 110px;
      white-space: pre-wrap;
      background: #f8fbff;
    }}
    .chat-answer:empty::before {{ content: "Model answer will appear here."; color: var(--muted); }}
    .chat-save-actions {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .chat-save-actions button {{
      min-height: 36px;
      padding: 8px 12px;
    }}

    .table-scroll {{
      min-width: 0;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }}

    @media (max-width: 1180px) {{
      .app-shell {{ grid-template-columns: 196px minmax(0, 1fr); }}
      .job-status-bar {{ margin-left: 0; }}
      .report-body-grid {{ grid-template-columns: 1fr; }}
      .report-preview-area {{ padding: 14px 18px 0; }}
      .report-chat-area {{ border-left: 0; border-top: 1px solid rgba(183, 204, 225, .78); }}
      .report-chat {{ position: static; min-width: 0; }}
      .manual-qa-form {{ grid-template-columns: 1fr; grid-template-areas: "context" "body" "generate" "save" "references"; }}
      .qa-context-bar, .qa-dataset-control, .qa-save-panel, .qa-reference-grid {{ grid-template-columns: 1fr; }}
      .dataset-action-row {{ grid-template-columns: minmax(0, 1fr); }}
      .dataset-save-section {{ grid-column: auto; justify-self: end; width: max-content; }}
      .dataset-action-section {{ align-items: stretch; }}
      .dataset-save-shell {{ width: auto; flex-basis: auto; }}
    }}

    @media (max-width: 760px) {{
      .app-shell {{ display: block; }}
      .sidebar-nav {{ position: static; height: auto; }}
      .topbar-actions {{ width: 100%; justify-content: flex-start; margin-left: 0; }}
      .job-status-shell {{ max-width: 100%; }}
      .job-status-bar {{ width: auto; flex-wrap: wrap; border-radius: 14px; }}
      .progress-track {{ flex-basis: 96px; width: 96px; max-width: 96px; }}
      .overview-summary-grid {{ grid-template-columns: 1fr; }}
      .overview-activity-strip {{ grid-template-columns: 1fr; }}
      .overview-actions {{ justify-content: stretch; }}
      .report-control-strip {{ grid-template-columns: 1fr; }}
      .report-toolbar, .report-control-strip, .report-preview-area, .report-chat-area {{ padding-left: 14px; padding-right: 14px; }}
      .manual-qa-row-top, .manual-qa-row-body {{ grid-template-columns: 1fr; }}
      .answer-generation-actions, .dataset-action-buttons, .dataset-action-section {{ display: grid; grid-template-columns: 1fr; }}
      .dataset-save-section {{ width: 100%; justify-self: stretch; }}
      .dataset-save-shell {{ grid-template-columns: 1fr; }}
      .dataset-save-button-group {{ width: 100%; }}
      .dataset-save-button-group button[type='submit'] {{ width: 100%; }}
      .manual-qa-form textarea {{ min-height: 132px; }}
      .dataset-action-header, .dataset-create-grid, .dataset-list-panel {{ padding-left: 14px; padding-right: 14px; }}
      .run-parameter-grid {{ grid-template-columns: 1fr; }}
      .run-parameter-control {{ grid-template-columns: 1fr; }}
      .run-submit-panel::before {{ display: none; }}
      .run-action-row {{ grid-template-columns: 1fr; }}
      .toolbar-actions {{ display: grid; }}
      button {{ width: 100%; }}
      .workspace {{ padding: 14px; }}
      .records-table {{ min-width: 720px; }}
      .table-scroll {{ overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar-nav" aria-label="Primary navigation">
      <div class="brand-lockup">
        <div class="mark">AT&amp;S</div>
        <h1 class="brand-title">RAG Evaluation</h1>
        <span hidden>RAG 评估</span>
        <span hidden>RAG 팀뮌</span>
      </div>
      <button class="nav-button active" type="button" data-section-target="datasets" data-i18n="navDatasets">Dataset</button>
      <button class="nav-button" type="button" data-section-target="run-evaluation" data-i18n="navRunEvaluation">Run</button>
      <button class="nav-button" type="button" data-section-target="eval-runs" data-i18n="navEvalRuns">Records</button>
      <button class="nav-button" type="button" data-section-target="reports" data-i18n="navReports">Reports</button>
      <button class="nav-button" type="button" data-section-target="overview" data-i18n="navOverview">Overview</button>
      <div class="sidebar-footer">
        <label class="language-select">
          <span data-i18n="pageLanguage">Page language</span>
          <select id="pageLanguage">
{language_options}
          </select>
        </label>
      </div>
    </aside>

    <main class="workspace">
      <div class="topbar">
        <div>
          <h2 class="section-heading page-title" data-i18n="consoleTitle">RAG Evaluation Console</h2>
        </div>
        <div class="topbar-actions">
          <div class="job-status-shell">
            <section id="jobStatusBar" class="job-status-bar" data-status="idle" aria-label="Background job">
              <div class="job-status-main">
                <span class="job-status-title" data-i18n="backgroundJob">Background Job</span>
                <span id="jobProgressText" class="job-meta" data-i18n="jobIdle">Idle</span>
              </div>
              <div class="job-status-progress">
                <div class="progress-track"><span id="jobProgressBar" class="progress-fill"></span></div>
                <div id="jobCurrentTask" class="job-meta"></div>
              </div>
              <div id="jobReportLink" class="job-meta job-status-details"></div>
            </section>
          </div>
          <button id="backButton" class="back-button" type="button" data-i18n="viewReport">View report</button>
        </div>
      </div>

      <section id="overview" class="console-section">
        <section class="panel">
          <div>
            <h2 class="section-heading" data-i18n="overview">Overview</h2>
          </div>
          <div class="overview-board">
            <div class="overview-summary-grid">
              <div class="overview-card"><span data-i18n="latestReport">Latest report</span><strong id="overviewLatestReport">-</strong></div>
              <div class="overview-card"><span data-i18n="latestEval">Latest eval</span><strong id="overviewLatestRun">-</strong></div>
              <div class="overview-card overview-count-card"><span data-i18n="evalRecords">Eval records</span><strong id="overviewRunCount">0</strong></div>
            </div>
            <div class="overview-activity-strip">
              <span data-i18n="activeJob">Active job</span>
              <strong id="overviewJobStatus" data-status="idle">Idle</strong>
            </div>
            <div class="toolbar-actions overview-actions">
              <button type="button" data-section-target="reports" data-i18n="openReports">Open reports</button>
              <button type="button" data-section-target="run-evaluation" data-i18n="runStandardEval">Run standard eval</button>
            </div>
          </div>
        </section>
      </section>

      <section id="reports" class="console-section">
        <section class="panel report-panel" aria-label="Report analysis">
          <div class="report-toolbar">
            <div>
              <h2 class="section-heading" data-i18n="evaluationReports">Report analysis</h2>
            </div>
            <div class="toolbar-actions">
              <button id="refreshReportsButton" type="button" data-i18n="refreshReports">Refresh reports</button>
            </div>
          </div>
          <div class="report-content-stack">
            <div class="report-main">
              {report_options}
            </div>
          </div>
        </section>
      </section>

      <section id="eval-runs" class="console-section">
        <section class="panel">
          <div class="report-toolbar">
            <div>
              <h2 class="section-heading" data-i18n="evalRuns">Evaluation Runs</h2>
            </div>
            <button id="refreshEvalRunsButton" type="button" data-i18n="refreshRecords">Refresh records</button>
          </div>
          <div class="table-scroll">
            <table id="evalRunsTable" class="records-table">
              <thead>
                <tr>
                  <th data-i18n="runHeader">Run</th>
                  <th data-i18n="datasetHeader">Dataset</th>
                  <th data-i18n="comboHeader">Combo</th>
                  <th data-i18n="generationHeader">Generation</th>
                  <th data-i18n="evaluationHeader">Evaluation</th>
                  <th data-i18n="secondsHeader">Seconds</th>
                  <th data-i18n="outputHeader">Output</th>
                </tr>
              </thead>
              <tbody><tr><td colspan="7" data-i18n="loadingEvalRecords">Loading evaluation records...</td></tr></tbody>
            </table>
          </div>
        </section>
      </section>

      <section id="run-evaluation" class="console-section">
        <section class="panel">
          <div>
            <h2 class="section-heading" data-i18n="runEvaluation">Run evaluation</h2>
          </div>
          <div class="run-control-grid">
            <div class="run-dataset-panel">
              <h3 class="section-heading" data-i18n="runDatasets">Run datasets</h3>
              <div id="runDatasetChoices" class="run-dataset-choices" aria-live="polite"></div>
            </div>
            <div class="run-parameter-grid">
              <div class="run-dataset-panel">
                <h3 class="section-heading" data-i18n="runPageSizes">Recall chunks</h3>
                <div class="run-parameter-control">
                  <select id="runPageSizeSelect" aria-label="Recall chunk preset"></select>
                  <input id="runPageSizeCustom" class="run-custom-input" type="number" min="1" max="20" step="1" placeholder="1-20" hidden>
                </div>
              </div>
              <div class="run-dataset-panel">
                <h3 class="section-heading" data-i18n="runSimilarities">Similarity</h3>
                <div class="run-parameter-control">
                  <select id="runSimilaritySelect" aria-label="Similarity preset"></select>
                  <input id="runSimilarityCustom" class="run-custom-input" type="number" min="0" max="0.5" step="0.01" placeholder="0-0.5" hidden>
                </div>
              </div>
              <div class="run-submit-panel">
                <div class="run-action-row">
                  <button id="runPipelineButton" type="button" data-i18n="runEvaluation" data-ready-text="Run" data-loading-text="Starting..." data-running-text="Running...">Run</button>
                  <button id="terminatePipelineButton" class="terminate-button" type="button" data-i18n="terminateRun" disabled>Terminate</button>
                </div>
              </div>
            </div>
          </div>
          <div id="pipelineResult" class="result pipeline-result" role="status" aria-live="polite"></div>
        </section>
      </section>

      <section id="datasets" class="console-section active">
        <section class="panel">
          <div class="dataset-section-stack">
            <div class="dataset-create-grid">
              <div class="dataset-flow-card dataset-step-card dataset-step-dataset">
                <div class="dataset-step-heading">
                  <span class="dataset-step-badge">1</span>
                  <span class="dataset-step-title">添加问答</span>
                </div>
                <div class="dataset-step-toolbar qa-context-bar dataset-module dataset-module-dataset">
                  <label class="qa-dataset-picker qa-dataset-control">
                    <span data-i18n="targetDataset">操作数据集</span>
                    <select id="manualQaDatasetSelect" name="target_dataset" required></select>
                  </label>
                  <span id="manualQaDatasetPath" class="manual-qa-selected-path"></span>
                  <div class="dataset-file-summary">
                    <span class="dataset-file-icon" aria-hidden="true">▣</span>
                    <span>
                      <strong id="datasetFileName">-</strong>
                      <small id="datasetFileMeta">0 rows selected</small>
                    </span>
                  </div>
                  <div class="dataset-upload-cell">
                    <button id="openUploadDialog" type="button" class="qa-dataset-upload-button" data-i18n="uploadDataset">Upload dataset</button>
                  </div>
                </div>
              </div>
              <form id="manualQaForm" class="manual-qa-form">
                <div class="manual-qa-panel dataset-workbench dataset-flow-card dataset-step-card dataset-step-edit">
                  <div class="dataset-step-heading">
                    <span class="dataset-step-badge">2</span>
                    <span class="dataset-step-title">编辑问答</span>
                  </div>
                  <div class="manual-qa-row manual-qa-row-body qa-input-panel dataset-module dataset-module-entry">
                    <label class="qa-card qa-question-card">
                      <span class="qa-field-label"><span data-i18n="question">Question</span><span class="qa-field-mark">Q</span></span>
                      <textarea id="manualQaQuestionTextarea" name="question" placeholder="请输入您想要添加的问题..." required></textarea>
                      <span id="manualQaQuestionCount" class="qa-character-count">0/2000</span>
                    </label>
                    <label class="qa-card qa-answer-card">
                      <span class="qa-field-label"><span>答案（标准答案）</span><span class="qa-field-mark">A</span></span>
                      <textarea id="manualQaAnswerTextarea" name="expected_answer" placeholder="请输入标准答案（Golden answer）..." required></textarea>
                      <span id="manualQaAnswerCount" class="qa-character-count">0/4000</span>
                    </label>
                  </div>
                </div>
                <div class="dataset-action-card dataset-flow-card dataset-step-card dataset-step-actions">
                  <div class="dataset-step-heading">
                    <span class="dataset-step-badge">3</span>
                    <span>生成与操作</span>
                  </div>
                  <div class="dataset-action-main dataset-action-row">
                    <section class="dataset-action-section dataset-generation-section">
                      <div class="answer-generation-actions dataset-action-buttons">
                        <button id="generateAnswerButton" type="button" data-i18n="generateAnswer" data-ready-text="Generate answer" data-loading-text="Generating...">Generate answer</button>
                        <button id="regenerateAnswerButton" class="secondary-button" type="button" data-i18n="regenerateAnswer" disabled>Regenerate</button>
                        <button id="clearGeneratedAnswerButton" class="secondary-button" type="button" data-i18n="clearGeneratedAnswer">Clear</button>
                      </div>
                    </section>
                    <section class="dataset-action-section dataset-save-section">
                      <div class="qa-save-panel dataset-module dataset-module-save">
                        <div class="dataset-save-shell">
                          <div class="dataset-save-button-group">
                            <button type="submit" data-i18n="saveManualQa" data-ready-text="Save Q&A" data-loading-text="Saving...">保存问答</button>
                          </div>
                        </div>
                      </div>
                    </section>
                  </div>
                  <div class="qa-reference-grid">
                    <details id="retrievedPassagesDetails" class="dataset-preview-details retrieved-passages-details dataset-secondary-panel dataset-secondary-preview">
                    <summary class="dataset-preview-summary">
                      <span class="dataset-preview-summary-title">召回片段</span>
                      <small>展开查看模型检索到的相关片段信息</small>
                    </summary>
                    <div id="retrievedPassagesList" class="retrieved-passages-list" aria-live="polite"></div>
                    </details>
                  </div>
                </div>
              </form>
              <div id="manualQaResult" class="result" role="status" aria-live="polite"></div>
            </div>
            <section id="datasetInventoryDetails" class="dataset-list-panel dataset-flow-card dataset-step-card dataset-step-list">
              <div class="dataset-list-summary dataset-step-heading">
                <span class="dataset-list-title">
                  <span class="dataset-step-badge">4</span>
                  <span data-i18n="currentDatasets">数据列表</span>
                  <strong id="datasetInventoryCount">0</strong>
                </span>
                <span class="dataset-list-actions">
                  <button id="refreshDatasetsButton" type="button" data-i18n="refreshDatasets" hidden>Refresh datasets</button>
                  <button type="button" class="dataset-view-all-button">查看全部</button>
                </span>
              </div>
              <div class="dataset-inventory">
                <div class="table-scroll">
                  <table id="datasetRowsTable" class="records-table dataset-rows-table">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>问题预览</th>
                        <th>答案预览</th>
                        <th data-i18n="datasetHeader">Dataset</th>
                        <th>操作</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr><td colspan="5" data-i18n="loadingDatasets">Loading datasets...</td></tr>
                    </tbody>
                  </table>
                  <table id="datasetsTable" class="records-table dataset-inventory-table" hidden>
                    <tbody></tbody>
                  </table>
                </div>
              </div>
            </section>
          </div>
        </section>
      </section>
    </main>
  </div>

  <dialog id="uploadDialog" aria-labelledby="uploadDialogTitle">
    <div class="dialog-window">
      <div class="dialog-titlebar">
        <div>
          <h2 id="uploadDialogTitle" class="section-heading" data-i18n="uploadDatasetWindow">Upload dataset window</h2>
        </div>
        <button id="closeUploadDialog" class="ghost-button" type="button" data-i18n="close">Close</button>
      </div>
      <form id="datasetUploadForm">
        <label>
          <span data-i18n="datasetName">Dataset name</span>
          <input type="text" name="name" autocomplete="off" placeholder="quarterly-support-qa">
        </label>
        <label>
          <span data-i18n="csvOrPdfFile">CSV or PDF file</span>
          <input type="file" name="file" accept=".csv,.pdf,text/csv,application/pdf" required>
        </label>
        <button type="submit" data-i18n="uploadDataset" data-ready-text="Upload dataset" data-loading-text="Uploading...">Upload dataset</button>
      </form>
      <div id="uploadResult" class="result" role="status" aria-live="polite">
        <div data-i18n="uploadWaitingPath">Saved dataset path: waiting for upload.</div>
        <div data-i18n="uploadSavedUnder">Saved under data/uploaded_datasets/.</div>
      </div>
    </div>
  </dialog>

  <script>
    const form = document.getElementById("datasetUploadForm");
    const result = document.getElementById("uploadResult");
    const button = form.querySelector("button[type='submit']");
    const manualQaForm = document.getElementById("manualQaForm");
    const manualQaDatasetSelect = document.getElementById("manualQaDatasetSelect");
    const manualQaResult = document.getElementById("manualQaResult");
    const manualQaButton = manualQaForm ? manualQaForm.querySelector("button[type='submit']") : null;
    const manualQaSaveTarget = document.getElementById("manualQaSaveTarget");
    const uploadDialog = document.getElementById("uploadDialog");
    const openUploadDialog = document.getElementById("openUploadDialog");
    const closeUploadDialog = document.getElementById("closeUploadDialog");
    const askModelForm = document.getElementById("askModelForm");
    const chatAnswer = document.getElementById("chatAnswer");
    const askModelButton = askModelForm ? askModelForm.querySelector("button[type='submit']") : null;
    const addReportAnswerToQaButton = document.getElementById("addReportAnswerToQaButton");
    const generateAnswerButton = document.getElementById("generateAnswerButton");
    const regenerateAnswerButton = document.getElementById("regenerateAnswerButton");
    const clearGeneratedAnswerButton = document.getElementById("clearGeneratedAnswerButton");
    const retrievedPassagesList = document.getElementById("retrievedPassagesList");
    const retrievedPassagesDetails = document.getElementById("retrievedPassagesDetails");
    const runPipelineButton = document.getElementById("runPipelineButton");
    const terminatePipelineButton = document.getElementById("terminatePipelineButton");
    const pipelineResult = document.getElementById("pipelineResult");
    const jobStatusBar = document.getElementById("jobStatusBar");
    const pageLanguage = document.getElementById("pageLanguage");
    const backButton = document.getElementById("backButton");
    const reportSelect = document.getElementById("reportSelect");
    const reportPath = document.getElementById("reportPath");
    const reportFrame = document.getElementById("reportFrame");
    const evalRunsTable = document.getElementById("evalRunsTable");
    const datasetsTable = document.getElementById("datasetsTable");
    const datasetRowsTable = document.getElementById("datasetRowsTable");
    const datasetRowsTableBody = datasetRowsTable ? datasetRowsTable.querySelector("tbody") : null;
    const datasetInventoryCount = document.getElementById("datasetInventoryCount");
    const datasetViewAllButton = document.querySelector(".dataset-view-all-button");
    const runDatasetChoices = document.getElementById("runDatasetChoices");
    const manualQaDatasetPath = document.getElementById("manualQaDatasetPath");
    const datasetFileName = document.getElementById("datasetFileName");
    const datasetFileMeta = document.getElementById("datasetFileMeta");
    const datasetSaveStatus = document.getElementById("datasetSaveStatus");
    const manualQaQuestionTextarea = document.getElementById("manualQaQuestionTextarea");
    const manualQaAnswerTextarea = document.getElementById("manualQaAnswerTextarea");
    const manualQaQuestionCount = document.getElementById("manualQaQuestionCount");
    const manualQaAnswerCount = document.getElementById("manualQaAnswerCount");
    const runPageSizeSelect = document.getElementById("runPageSizeSelect");
    const runPageSizeCustom = document.getElementById("runPageSizeCustom");
    const runSimilaritySelect = document.getElementById("runSimilaritySelect");
    const runSimilarityCustom = document.getElementById("runSimilarityCustom");
    const jobProgressBar = document.getElementById("jobProgressBar");
    const jobProgressText = document.getElementById("jobProgressText");
    const jobCurrentTask = document.getElementById("jobCurrentTask");
    const jobReportLink = document.getElementById("jobReportLink");
    const overviewLatestReport = document.getElementById("overviewLatestReport");
    const overviewRunCount = document.getElementById("overviewRunCount");
    const overviewLatestRun = document.getElementById("overviewLatestRun");
    const overviewJobStatus = document.getElementById("overviewJobStatus");
    const uiTranslations = {{
      de: {{
        viewReport: "Bericht ansehen",
        overview: "Uebersicht",
        evalRuns: "Auswertungslaeufe",
        navDatasets: "Daten",
        navRunEvaluation: "Starten",
        navEvalRuns: "Laeufe",
        navReports: "Berichte",
        navOverview: "Uebersicht",
        consoleTitle: "RAG-Auswertungskonsole",
        latestReport: "Neuester Bericht",
        evalRecords: "Auswertungsdaten",
        latestEval: "Neueste Auswertung",
        activeJob: "Aktiver Job",
        openReports: "Berichte oeffnen",
        runStandardEval: "Standardauswertung starten",
        evaluationReports: "Berichtsanalyse",
        refreshReports: "Berichte aktualisieren",
        refreshRecords: "Daten aktualisieren",
        currentDatasets: "Aktuelle Datensaetze",
        refreshDatasets: "Datensaetze aktualisieren",
        manualQaDataset: "Manueller QA-Datensatz",
        targetDataset: "Datensatz verwalten",
        saveTargetPrefix: "Speichert in: ",
        standardAnswer: "Standardantwort",
        saveManualQa: "Q&A speichern",
        manualQaSaving: "Speichert...",
        manualQaSaved: "QA angehaengt",
        manualQaFailed: "QA konnte nicht gespeichert werden.",
        addQa: "QA hinzufuegen",
        addMethod: "Methode",
        manualAdd: "Manuell",
        aiGenerate: "KI",
        generateAnswerMode: "Antwort erzeugen",
        generateAnswer: "Antwort erzeugen",
        regenerateAnswer: "Neu erzeugen",
        clearGeneratedAnswer: "Leeren",
        retrievedPassages: "Gefundene Passagen",
        generatedAnswer: "Erzeugte Antwort",
        generatingAnswer: "Antwort wird erzeugt...",
        answerGenerationFailed: "Antworterzeugung fehlgeschlagen.",
        fromReportChat: "Bericht-QA",
        sourceText: "Quelltext",
        sourceTextPlaceholder: "Dokumentinhalt oder Berichtnotizen einfuegen",
        candidateCount: "Anzahl",
        generateCandidateQa: "QA erzeugen",
        clearCandidates: "Leeren",
        candidateQa: "Zu speichern",
        saveSelectedQa: "Auswahl speichern",
        select: "Auswahl",
        actions: "Aktionen",
        reportQaHint: "Frage im Bericht stellen und die Modellantwort ins QA-Formular uebernehmen.",
        noCandidates: "Noch keine Kandidaten.",
        generatingQa: "QA wird erzeugt...",
        qaGenerationFailed: "QA-Erzeugung fehlgeschlagen.",
        selectQaRows: "Mindestens einen QA-Kandidaten auswaehlen.",
        selectedQaSaved: "Ausgewaehlte QA gespeichert",
        deleteCandidate: "Loeschen",
        addReportAnswerToQa: "In QA uebernehmen",
        addedToQaCandidates: "In QA uebernommen",
        runDatasets: "Datensaetze ausfuehren",
        runPageSizes: "Abrufsegmente",
        runSimilarities: "Aehnlichkeit",
        sourceHeader: "Quelle",
        modifiedHeader: "Geaendert",
        pathHeader: "Pfad",
        datasetSourceConfig: "Konfiguration",
        datasetSourceUploaded: "Hochgeladen",
        loadingDatasets: "Datensaetze werden geladen...",
        unableLoadDatasets: "Datensaetze konnten nicht geladen werden.",
        noDatasetsFound: "Keine Datensaetze gefunden.",
        noRunnableDatasets: "Keine konfigurierten Datensaetze verfuegbar.",
        selectRunDataset: "Mindestens einen Datensatz auswaehlen.",
        noRunOptions: "Keine Parameter verfuegbar.",
        selectRunParameter: "Mindestens eine Seitengroesse und eine Aehnlichkeit auswaehlen.",
        otherOption: "Andere",
        runHeader: "Lauf",
        datasetHeader: "Datensatz",
        comboHeader: "Kombination",
        generationHeader: "Generierung",
        evaluationHeader: "Auswertung",
        secondsHeader: "Sekunden",
        outputHeader: "Ausgabe",
        openOutputFile: "Datei oeffnen",
        pageLanguage: "Seitensprache",
        runEvaluation: "Starten",
        pipelineStarting: "Startet...",
        pipelineRunning: "Laeuft...",
        terminateRun: "Stoppen",
        terminatingPipelineJob: "Pipeline wird gestoppt...",
        pipeline: "Pipeline",
        runId: "Lauf-ID",
        config: "Konfiguration",
        startingPipelineJob: "Pipeline-Serverjob wird gestartet...",
        pipelineStartFailed: "Pipeline-Start fehlgeschlagen.",
        pipelineStatusRequestFailed: "Pipeline-Statusanfrage fehlgeschlagen.",
        uploadDataset: "Datensatz hochladen",
        uploadDatasetWindow: "Fenster zum Hochladen",
        uploadLoading: "Wird hochgeladen...",
        uploadingDataset: "Datensatz wird hochgeladen...",
        uploadFailed: "Upload fehlgeschlagen.",
        close: "Schliessen",
        datasetName: "Datensatzname",
        csvOrPdfFile: "CSV- oder PDF-Datei",
        uploadWaitingPath: "Gespeicherter Datensatzpfad: wartet auf Upload.",
        uploadSavedUnder: "Gespeichert unter data/uploaded_datasets/.",
        uploaded: "Datensatz hochgeladen",
        rows: "Zeilen",
        savedDatasetPath: "Gespeicherter Datensatzpfad",
        savedUnder: "Gespeichert unter data/uploaded_datasets/.",
        askModel: "Modell fragen",
        askLoading: "Denkt nach...",
        question: "Frage",
        questionPlaceholder: "Was zeigt dieser Bericht?",
        manualQuestionPlaceholder: "Geben Sie die Frage ein, die Sie hinzufuegen moechten",
        thinking: "Denkt nach...",
        chatRequestFailed: "Chat-Anfrage fehlgeschlagen.",
        pipelinePlaceholder: "Der Pipeline-Serverstatus wird hier angezeigt.",
        backgroundJob: "Hintergrundjob",
        jobIdle: "Leerlauf",
        currentTask: "Aktuelle Aufgabe",
        htmlReport: "HTML-Bericht",
        openReport: "Bericht oeffnen",
        loadingEvalRecords: "Auswertungsdaten werden geladen...",
        unableLoadEvalRecords: "Auswertungsdaten konnten nicht geladen werden.",
        noEvalRecordsFound: "Keine Auswertungsdaten gefunden.",
        taskUnit: "Aufgaben",
        error: "Fehler",
        reportPath: "Berichtspfad",
        statusLabels: {{ idle: "Leerlauf", starting: "Startet", running: "Laeuft", cancelling: "Stoppt", cancelled: "Gestoppt", completed: "Abgeschlossen", failed: "Fehlgeschlagen" }}
      }},
      en: {{
        viewReport: "View report",
        overview: "Overview",
        evalRuns: "Evaluation Runs",
        navDatasets: "Dataset",
        navRunEvaluation: "Run",
        navEvalRuns: "Records",
        navReports: "Reports",
        navOverview: "Overview",
        consoleTitle: "RAG Evaluation Console",
        latestReport: "Latest report",
        evalRecords: "Eval records",
        latestEval: "Latest eval",
        activeJob: "Active job",
        openReports: "Open reports",
        runStandardEval: "Run standard eval",
        evaluationReports: "Report analysis",
        refreshReports: "Refresh reports",
        refreshRecords: "Refresh records",
        currentDatasets: "Datasets",
        refreshDatasets: "Refresh datasets",
        manualQaDataset: "Manual QA dataset",
        targetDataset: "Dataset actions",
        saveTargetPrefix: "Will save to: ",
        standardAnswer: "Answer",
        saveManualQa: "Save Q&A",
        manualQaSaving: "Saving...",
        manualQaSaved: "Appended QA",
        manualQaFailed: "Unable to save QA dataset.",
        addQa: "Add Q&A",
        addMethod: "Method",
        manualAdd: "Manual",
        aiGenerate: "AI",
        generateAnswerMode: "Generate answer",
        generateAnswer: "Generate answer",
        regenerateAnswer: "Regenerate",
        clearGeneratedAnswer: "Clear",
        retrievedPassages: "Retrieved passages",
        generatedAnswer: "Generated answer",
        generatingAnswer: "Generating answer...",
        answerGenerationFailed: "Answer generation failed.",
        fromReportChat: "Report Q&A",
        sourceText: "Source text",
        sourceTextPlaceholder: "Paste document content or report notes",
        candidateCount: "Count",
        generateCandidateQa: "Generate QA",
        clearCandidates: "Clear",
        candidateQa: "To save",
        saveSelectedQa: "Save selected",
        select: "Select",
        actions: "Actions",
        reportQaHint: "Ask a question in the report page, then fill the answer into the QA form.",
        noCandidates: "No candidates yet.",
        generatingQa: "Generating QA...",
        qaGenerationFailed: "QA generation failed.",
        selectQaRows: "Select at least one QA candidate.",
        selectedQaSaved: "Saved selected QA",
        deleteCandidate: "Delete",
        addReportAnswerToQa: "Fill QA",
        addedToQaCandidates: "Filled into QA",
        runDatasets: "Run datasets",
        runPageSizes: "Recall chunks",
        runSimilarities: "Similarity",
        sourceHeader: "Source",
        modifiedHeader: "Modified",
        pathHeader: "Path",
        datasetSourceConfig: "Config",
        datasetSourceUploaded: "Uploaded",
        loadingDatasets: "Loading datasets...",
        unableLoadDatasets: "Unable to load datasets.",
        noDatasetsFound: "No datasets found.",
        noRunnableDatasets: "No configured datasets available.",
        selectRunDataset: "Select at least one dataset.",
        noRunOptions: "No parameters available.",
        selectRunParameter: "Select at least one page size and one similarity.",
        otherOption: "Other",
        runHeader: "Run",
        datasetHeader: "Dataset",
        comboHeader: "Combo",
        generationHeader: "Generation",
        evaluationHeader: "Evaluation",
        secondsHeader: "Seconds",
        outputHeader: "Output",
        openOutputFile: "Open file",
        pageLanguage: "Page language",
        runEvaluation: "Run",
        pipelineStarting: "Starting...",
        pipelineRunning: "Running...",
        terminateRun: "Stop",
        terminatingPipelineJob: "Stopping pipeline job...",
        pipeline: "Pipeline",
        runId: "Run ID",
        config: "Config",
        startingPipelineJob: "Starting pipeline server job...",
        pipelineStartFailed: "Pipeline start failed.",
        pipelineStatusRequestFailed: "Pipeline status request failed.",
        uploadDataset: "Upload dataset",
        uploadDatasetWindow: "Upload dataset window",
        uploadLoading: "Uploading...",
        uploadingDataset: "Uploading dataset...",
        uploadFailed: "Upload failed.",
        close: "Close",
        datasetName: "Dataset name",
        csvOrPdfFile: "CSV or PDF file",
        uploadWaitingPath: "Saved dataset path: waiting for upload.",
        uploadSavedUnder: "Saved under data/uploaded_datasets/.",
        uploaded: "Uploaded dataset",
        rows: "Rows",
        savedDatasetPath: "Saved dataset path",
        savedUnder: "Saved under data/uploaded_datasets/.",
        askModel: "Ask model",
        askLoading: "Thinking...",
        question: "Question",
        questionPlaceholder: "What does this report show?",
        manualQuestionPlaceholder: "Enter the question you want to add",
        thinking: "Thinking...",
        chatRequestFailed: "Chat request failed.",
        pipelinePlaceholder: "Pipeline server status will appear here.",
        backgroundJob: "Background Job",
        jobIdle: "Idle",
        currentTask: "Current task",
        htmlReport: "HTML report",
        openReport: "Open report",
        loadingEvalRecords: "Loading evaluation records...",
        unableLoadEvalRecords: "Unable to load evaluation records.",
        noEvalRecordsFound: "No evaluation records found.",
        taskUnit: "tasks",
        error: "Error",
        reportPath: "Report path",
        statusLabels: {{ idle: "Idle", starting: "Starting", running: "Running", cancelling: "Stopping", cancelled: "Stopped", completed: "Completed", failed: "Failed" }}
      }},
      zh: {{
        viewReport: "查看报告",
        overview: "总览",
        evalRuns: "评估记录",
        navDatasets: "数据集",
        navRunEvaluation: "运行",
        navEvalRuns: "记录",
        navReports: "报告",
        navOverview: "总览",
        consoleTitle: "RAG 评估控制台",
        latestReport: "最新报告",
        evalRecords: "评估记录",
        latestEval: "最近评估",
        activeJob: "活动任务",
        openReports: "打开报告",
        runStandardEval: "运行标准评估",
        evaluationReports: "报告分析",
        refreshReports: "刷新报告",
        refreshRecords: "刷新记录",
        currentDatasets: "当前数据集",
        refreshDatasets: "刷新数据集",
        manualQaDataset: "手动 QA 数据集",
        targetDataset: "操作数据集",
        saveTargetPrefix: "将保存到：",
        standardAnswer: "标准答案",
        saveManualQa: "保存问答",
        manualQaSaving: "保存中...",
        manualQaSaved: "QA 已追加",
        manualQaFailed: "无法保存 QA 数据集。",
        runDatasets: "运行数据集",
        runPageSizes: "召回片段",
        runSimilarities: "相似度",
        sourceHeader: "来源",
        modifiedHeader: "修改时间",
        pathHeader: "路径",
        datasetSourceConfig: "配置",
        datasetSourceUploaded: "已上传",
        loadingDatasets: "正在加载数据集...",
        unableLoadDatasets: "无法加载数据集。",
        noDatasetsFound: "没有找到数据集。",
        noRunnableDatasets: "没有可运行的配置数据集。",
        selectRunDataset: "请至少选择一个数据集。",
        noRunOptions: "没有可用参数。",
        selectRunParameter: "请至少选择一个 page size 和一个 similarity。",
        otherOption: "其他",
        runHeader: "运行",
        datasetHeader: "数据集",
        comboHeader: "组合",
        generationHeader: "生成",
        evaluationHeader: "评估",
        secondsHeader: "耗时",
        outputHeader: "输出",
        openOutputFile: "打开文件",
        pageLanguage: "页面语言",
        runEvaluation: "运行",
        pipelineStarting: "启动中...",
        pipelineRunning: "运行中...",
        pipeline: "Pipeline",
        runId: "运行 ID",
        config: "配置",
        startingPipelineJob: "正在启动 Pipeline 后台任务...",
        pipelineStartFailed: "Pipeline 启动失败。",
        pipelineStatusRequestFailed: "Pipeline 状态请求失败。",
        uploadDataset: "上传数据集",
        uploadDatasetWindow: "上传数据集窗口",
        uploadLoading: "上传中...",
        uploadingDataset: "正在上传数据集...",
        uploadFailed: "上传失败。",
        close: "关闭",
        datasetName: "数据集名称",
        csvOrPdfFile: "CSV 或 PDF 文件",
        uploadWaitingPath: "数据集保存路径：等待上传。",
        uploadSavedUnder: "保存位置：data/uploaded_datasets/。",
        uploaded: "数据集已上传",
        rows: "行数",
        savedDatasetPath: "数据集保存路径",
        savedUnder: "保存位置：data/uploaded_datasets/。",
        askModel: "询问模型",
        askLoading: "思考中...",
        question: "问题",
        questionPlaceholder: "这份报告显示了什么？",
        manualQuestionPlaceholder: "请输入您想要添加的问题",
        thinking: "思考中...",
        chatRequestFailed: "Chat 请求失败。",
        pipelinePlaceholder: "Pipeline 服务状态将显示在这里。",
        backgroundJob: "后台任务",
        jobIdle: "空闲",
        currentTask: "当前任务",
        htmlReport: "HTML 报告",
        openReport: "打开报告",
        loadingEvalRecords: "正在加载评估记录...",
        unableLoadEvalRecords: "无法加载评估记录。",
        noEvalRecordsFound: "没有找到评估记录。",
        taskUnit: "个任务",
        error: "错误",
        reportPath: "报告路径",
        currentDatasets: "数据列表",
        standardAnswer: "答案",
        addQa: "添加问答",
        addMethod: "方式",
        manualAdd: "手动",
        aiGenerate: "AI 生成",
        generateAnswerMode: "生成答案",
        generateAnswer: "生成答案",
        regenerateAnswer: "重新生成",
        clearGeneratedAnswer: "清空",
        retrievedPassages: "召回片段",
        generatedAnswer: "生成答案",
        generatingAnswer: "正在生成答案...",
        answerGenerationFailed: "答案生成失败。",
        fromReportChat: "报告问答",
        sourceText: "来源文本",
        sourceTextPlaceholder: "粘贴文档内容或报告摘要",
        candidateCount: "数量",
        generateCandidateQa: "生成问答",
        clearCandidates: "清空",
        candidateQa: "待保存",
        saveSelectedQa: "保存选中",
        select: "选择",
        actions: "操作",
        reportQaHint: "先在报告页面询问模型，再把回答填入问答表单。",
        noCandidates: "暂无待保存内容。",
        generatingQa: "正在生成问答...",
        qaGenerationFailed: "问答生成失败。",
        selectQaRows: "请至少选择一条待保存内容。",
        selectedQaSaved: "已保存选中内容",
        deleteCandidate: "删除",
        addReportAnswerToQa: "填入问答",
        addedToQaCandidates: "已填入问答",
        statusLabels: {{ idle: "空闲", starting: "启动中", running: "运行中", cancelling: "终止中", cancelled: "已终止", completed: "已完成", failed: "失败" }}
      }},
      ms: {{
        viewReport: "Lihat laporan",
        overview: "Gambaran keseluruhan",
        evalRuns: "Rekod penilaian",
        navDatasets: "Dataset",
        navRunEvaluation: "Jalan",
        navEvalRuns: "Rekod",
        navReports: "Laporan",
        navOverview: "Ringkas",
        consoleTitle: "Konsol Penilaian RAG",
        latestReport: "Laporan terkini",
        evalRecords: "Rekod penilaian",
        latestEval: "Penilaian terkini",
        activeJob: "Kerja aktif",
        openReports: "Buka laporan",
        runStandardEval: "Jalankan penilaian standard",
        evaluationReports: "Analisis laporan",
        refreshReports: "Segar semula laporan",
        refreshRecords: "Segar semula rekod",
        currentDatasets: "Dataset semasa",
        refreshDatasets: "Segar semula dataset",
        manualQaDataset: "Dataset QA manual",
        targetDataset: "Urus dataset",
        saveTargetPrefix: "Akan disimpan ke: ",
        standardAnswer: "Jawapan standard",
        saveManualQa: "Simpan Q&A",
        manualQaSaving: "Menyimpan...",
        manualQaSaved: "QA ditambah",
        manualQaFailed: "Tidak dapat menyimpan dataset QA.",
        addQa: "Tambah QA",
        addMethod: "Kaedah",
        manualAdd: "Manual",
        aiGenerate: "AI",
        generateAnswerMode: "Jana jawapan",
        generateAnswer: "Jana jawapan",
        regenerateAnswer: "Jana semula",
        clearGeneratedAnswer: "Kosongkan",
        retrievedPassages: "Petikan carian",
        generatedAnswer: "Jawapan dijana",
        generatingAnswer: "Menjana jawapan...",
        answerGenerationFailed: "Penjanaan jawapan gagal.",
        fromReportChat: "QA laporan",
        sourceText: "Teks sumber",
        sourceTextPlaceholder: "Tampal kandungan dokumen atau nota laporan",
        candidateCount: "Bilangan",
        generateCandidateQa: "Jana QA",
        clearCandidates: "Kosongkan",
        candidateQa: "Untuk simpan",
        saveSelectedQa: "Simpan pilihan",
        select: "Pilih",
        actions: "Tindakan",
        reportQaHint: "Tanya soalan pada laporan, kemudian isikan jawapan model ke borang QA.",
        noCandidates: "Belum ada calon.",
        generatingQa: "Menjana QA...",
        qaGenerationFailed: "Penjanaan QA gagal.",
        selectQaRows: "Pilih sekurang-kurangnya satu calon QA.",
        selectedQaSaved: "QA pilihan disimpan",
        deleteCandidate: "Padam",
        addReportAnswerToQa: "Isi QA",
        addedToQaCandidates: "Diisi ke QA",
        runDatasets: "Jalankan dataset",
        runPageSizes: "Serpihan panggilan balik",
        runSimilarities: "Kesamaan",
        sourceHeader: "Sumber",
        modifiedHeader: "Diubah",
        pathHeader: "Laluan",
        datasetSourceConfig: "Konfigurasi",
        datasetSourceUploaded: "Dimuat naik",
        loadingDatasets: "Memuatkan dataset...",
        unableLoadDatasets: "Tidak dapat memuatkan dataset.",
        noDatasetsFound: "Tiada dataset ditemui.",
        noRunnableDatasets: "Tiada dataset konfigurasi tersedia.",
        selectRunDataset: "Pilih sekurang-kurangnya satu dataset.",
        noRunOptions: "Tiada parameter tersedia.",
        selectRunParameter: "Pilih sekurang-kurangnya satu saiz halaman dan satu kesamaan.",
        otherOption: "Lain",
        runHeader: "Larian",
        datasetHeader: "Dataset",
        comboHeader: "Gabungan",
        generationHeader: "Penjanaan",
        evaluationHeader: "Penilaian",
        secondsHeader: "Saat",
        outputHeader: "Output",
        openOutputFile: "Buka fail",
        pageLanguage: "Bahasa halaman",
        runEvaluation: "Jalankan",
        pipelineStarting: "Memulakan...",
        pipelineRunning: "Sedang berjalan...",
        terminateRun: "Hentikan",
        terminatingPipelineJob: "Menghentikan kerja pipeline...",
        pipeline: "Pipeline",
        runId: "ID larian",
        config: "Konfigurasi",
        startingPipelineJob: "Memulakan kerja pelayan pipeline...",
        pipelineStartFailed: "Gagal memulakan pipeline.",
        pipelineStatusRequestFailed: "Permintaan status pipeline gagal.",
        uploadDataset: "Muat naik dataset",
        uploadDatasetWindow: "Tetingkap muat naik dataset",
        uploadLoading: "Memuat naik...",
        uploadingDataset: "Memuat naik dataset...",
        uploadFailed: "Muat naik gagal.",
        close: "Tutup",
        datasetName: "Nama dataset",
        csvOrPdfFile: "Fail CSV atau PDF",
        uploadWaitingPath: "Laluan dataset tersimpan: menunggu muat naik.",
        uploadSavedUnder: "Disimpan di data/uploaded_datasets/.",
        uploaded: "Dataset dimuat naik",
        rows: "Baris",
        savedDatasetPath: "Laluan dataset tersimpan",
        savedUnder: "Disimpan di data/uploaded_datasets/.",
        askModel: "Tanya model",
        askLoading: "Berfikir...",
        question: "Soalan",
        questionPlaceholder: "Apakah yang ditunjukkan oleh laporan ini?",
        manualQuestionPlaceholder: "Masukkan soalan yang ingin anda tambah",
        thinking: "Berfikir...",
        chatRequestFailed: "Permintaan chat gagal.",
        pipelinePlaceholder: "Status pelayan pipeline akan dipaparkan di sini.",
        backgroundJob: "Kerja Latar Belakang",
        jobIdle: "Melahu",
        currentTask: "Tugas semasa",
        htmlReport: "Laporan HTML",
        openReport: "Buka laporan",
        loadingEvalRecords: "Memuatkan rekod penilaian...",
        unableLoadEvalRecords: "Tidak dapat memuatkan rekod penilaian.",
        noEvalRecordsFound: "Tiada rekod penilaian ditemui.",
        taskUnit: "tugas",
        error: "Ralat",
        reportPath: "Laluan laporan",
        statusLabels: {{ idle: "Melahu", starting: "Memulakan", running: "Sedang berjalan", cancelling: "Menghentikan", cancelled: "Dihentikan", completed: "Selesai", failed: "Gagal" }}
      }}
    }};
    const reportTranslations = {{
      de: {{"RAG Evaluation": "RAG-Auswertung"}},
      en: {{"RAG Evaluation": "RAG Evaluation"}},
      zh: {{"RAG Evaluation": "RAG 评估"}},
      ms: {{"RAG Evaluation": "Penilaian RAG"}}
    }};

    let activeRunId = "";
    let currentDatasets = [];
    let currentRunOptions = {{page_sizes: [], similarity_thresholds: []}};
    let generatedAnswerPassages = [];
    let datasetRowsExpanded = false;
    let lastDatasetRowsPayload = null;
    let lastReportQuestion = "";
    let lastReportAnswer = "";
    const OTHER_RUN_OPTION = "__other__";

    function escapeHTML(value) {{
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({{
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }}[char]));
    }}

    function switchSection(sectionId) {{
      document.querySelectorAll(".console-section").forEach((section) => {{
        section.classList.toggle("active", section.id === sectionId);
      }});
      document.querySelectorAll("[data-section-target]").forEach((item) => {{
        item.classList.toggle("active", item.dataset.sectionTarget === sectionId);
      }});
      if (sectionId === "eval-runs") refreshEvalRuns();
      if (sectionId === "reports") refreshReportList();
      if (sectionId === "datasets") refreshDatasets();
      if (sectionId === "run-evaluation") refreshDatasets();
    }}

    document.querySelectorAll("[data-section-target]").forEach((item) => {{
      item.addEventListener("click", () => switchSection(item.dataset.sectionTarget));
    }});

    function currentTranslations() {{
      return uiTranslations[pageLanguage.value] || uiTranslations.en;
    }}

    function t(key) {{
      const translations = currentTranslations();
      return translations[key] || uiTranslations.en[key] || key;
    }}

    function statusText(status) {{
      const labels = currentTranslations().statusLabels || uiTranslations.en.statusLabels;
      return labels[status] || status;
    }}

    function statusClass(status) {{
      return String(status || "").toLowerCase().replace(/[^a-z0-9_-]+/g, "_") || "unknown";
    }}

    function statusBadge(status) {{
      return "<span class=\\"status-pill status-" + escapeHTML(statusClass(status)) + "\\">" + escapeHTML(statusText(status || "-")) + "</span>";
    }}

    function updateOverviewLatestRun(run) {{
      if (!overviewLatestRun) return;
      if (!run) {{
        overviewLatestRun.textContent = "-";
        return;
      }}
      overviewLatestRun.textContent = [run.dataset_name || "-", run.combo || "-", statusText(run.evaluation_status || "")]
        .filter(Boolean)
        .join(" / ");
    }}

    function applyReportLanguage(language) {{
      return reportTranslations[language] || reportTranslations.en;
    }}

    function applyPageLanguage(language) {{
      if (uiTranslations[language]) pageLanguage.value = language;
      const translations = currentTranslations();
      document.querySelectorAll("[data-i18n]").forEach((element) => {{
        const key = element.dataset.i18n;
        const value = translations[key] || uiTranslations.en[key];
        if (value) element.textContent = value;
      }});
      document.querySelectorAll("[data-i18n-placeholder]").forEach((element) => {{
        const key = element.dataset.i18nPlaceholder;
        const value = translations[key] || uiTranslations.en[key];
        if (value) element.setAttribute("placeholder", value);
      }});
      runPipelineButton.dataset.readyText = translations.runEvaluation;
      runPipelineButton.dataset.loadingText = translations.pipelineStarting;
      runPipelineButton.dataset.runningText = translations.pipelineRunning;
      terminatePipelineButton.textContent = translations.terminateRun || "终止";
      button.dataset.readyText = translations.uploadDataset;
      button.dataset.loadingText = translations.uploadLoading;
      if (manualQaButton) {{
        manualQaButton.dataset.readyText = translations.saveManualQa;
        manualQaButton.dataset.loadingText = translations.manualQaSaving;
      }}
      if (askModelButton) {{
        askModelButton.dataset.readyText = translations.askModel;
        askModelButton.dataset.loadingText = translations.askLoading;
      }}
      if (generateAnswerButton) {{
        generateAnswerButton.dataset.readyText = t("generateAnswer");
        generateAnswerButton.dataset.loadingText = t("generatingAnswer");
      }}
      if (!runPipelineButton.disabled) runPipelineButton.textContent = translations.runEvaluation;
      if (!button.disabled) button.textContent = translations.uploadDataset;
      if (manualQaButton && !manualQaButton.disabled) manualQaButton.textContent = translations.saveManualQa;
      if (askModelButton && !askModelButton.disabled) askModelButton.textContent = translations.askModel;
      if (generateAnswerButton && !generateAnswerButton.disabled) generateAnswerButton.textContent = t("generateAnswer");
      if (overviewJobStatus.dataset.status) overviewJobStatus.textContent = statusText(overviewJobStatus.dataset.status);
      if (!pipelineResult.textContent.trim()) pipelineResult.textContent = translations.pipelinePlaceholder;
      renderDatasetInventory(currentDatasets);
      renderManualQaDatasetChoices(currentDatasets);
      renderRunDatasetChoices(currentDatasets);
      renderRunParameterChoices(currentRunOptions);
      applyReportLanguage(pageLanguage.value);
      window.localStorage.setItem("ragEvalPageLanguage", pageLanguage.value);
    }}

    function setResult(type, html) {{
      result.className = "result " + type;
      result.innerHTML = html;
    }}

    function setPipelineResult(type, html) {{
      pipelineResult.className = "result pipeline-result " + type;
      pipelineResult.innerHTML = html;
    }}

    function setManualQaResult(type, html) {{
      if (!manualQaResult) return;
      manualQaResult.className = "result " + type;
      manualQaResult.innerHTML = html;
    }}

    function renderRetrievedPassages(passages) {{
      if (!retrievedPassagesList) return;
      if (!passages || !passages.length) {{
        retrievedPassagesList.innerHTML = "<div class=\\"dataset-preview-empty\\">" + escapeHTML(t("retrievedPassages")) + ": 0</div>";
        if (regenerateAnswerButton) regenerateAnswerButton.disabled = true;
        return;
      }}
      retrievedPassagesList.innerHTML = passages.map((passage, index) => (
        "<div class=\\"retrieved-passage-item\\">" +
          "<strong>#" + escapeHTML(index + 1) + "</strong> " +
          escapeHTML(passage.text || "") +
          (passage.source ? "<div class=\\"dataset-preview-meta\\">" + escapeHTML(passage.source) + "</div>" : "") +
        "</div>"
      )).join("");
      if (regenerateAnswerButton) regenerateAnswerButton.disabled = false;
    }}

    function setPipelineButtons(disabled) {{
      runPipelineButton.disabled = disabled;
      const canTerminate = disabled && Boolean(activeRunId);
      terminatePipelineButton.disabled = !canTerminate;
      if (!disabled) {{
        runPipelineButton.textContent = runPipelineButton.dataset.readyText;
      }}
    }}

    function renderJobStatusBar(job = {{}}) {{
      const progress = job.progress || {{}};
      const percent = Number(progress.percent || 0);
      const completed = Number(progress.completed_tasks || 0);
      const total = Number(progress.total_tasks || 0);
      jobProgressBar.style.width = Math.max(0, Math.min(percent, 100)) + "%";
      const status = job.status || "idle";
      jobStatusBar.dataset.status = status;
      overviewJobStatus.dataset.status = status;
      overviewJobStatus.textContent = statusText(status);
      jobProgressText.textContent = total
        ? statusText(status) + " - " + completed + " / " + total + " " + t("taskUnit") + " (" + percent.toFixed(1) + "%)"
        : status === "idle" ? t("jobIdle") : statusText(status);
      jobCurrentTask.textContent = progress.current_task ? t("currentTask") + ": " + progress.current_task : "";
      const renderedReport = Array.isArray(job.reports) && job.reports.length ? job.reports[0] : null;
      jobReportLink.innerHTML = renderedReport
        ? t("htmlReport") + ": <a href='" + escapeHTML(renderedReport.url || "") + "'>" + escapeHTML(renderedReport.path || renderedReport.name || t("openReport")) + "</a>"
        : "";
    }}

    async function pollPipelineStatus(runId) {{
      const response = await fetch("/api/pipeline/status?run_id=" + encodeURIComponent(runId));
      const payload = await response.json().catch(() => ({{}}));
      if (!response.ok || payload.ok === false) throw new Error(payload.error || t("pipelineStatusRequestFailed"));
      const job = payload.job || {{}};
      renderJobStatusBar(job);
      const renderedReport = Array.isArray(job.reports) && job.reports.length ? job.reports[0] : null;
      setPipelineResult(
        job.status === "failed" ? "error" : "success",
        "<strong>" + escapeHTML(t("pipeline")) + " " + escapeHTML(statusText(job.status || "unknown")) + "</strong>" +
          "<div>" + escapeHTML(t("runId")) + ": " + escapeHTML(job.run_id || runId) + "</div>" +
          "<div>" + escapeHTML(t("config")) + ": " + escapeHTML(job.config || "") + "</div>" +
          (renderedReport ? "<div>" + escapeHTML(t("htmlReport")) + ": <a href='" + escapeHTML(renderedReport.url || "") + "'>" + escapeHTML(renderedReport.path || renderedReport.name || t("openReport")) + "</a></div>" : "") +
          (job.error ? "<pre>" + escapeHTML(job.error) + "</pre>" : "")
      );
      if (job.status === "running" || job.status === "cancelling") {{
        await new Promise((resolve) => setTimeout(resolve, 2000));
        return pollPipelineStatus(runId);
      }}
      activeRunId = "";
      setPipelineButtons(false);
      await refreshEvalRuns();
      if (renderedReport) await refreshReportList(renderedReport.path || "");
    }}

    async function terminatePipeline() {{
      if (!activeRunId || terminatePipelineButton.disabled) return;
      terminatePipelineButton.disabled = true;
      setPipelineResult("", t("terminatingPipelineJob"));
      renderJobStatusBar({{status: "cancelling", progress: {{percent: 0}}}});
      try {{
        const response = await fetch("/api/pipeline/cancel", {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify({{run_id: activeRunId}})
        }});
        const payload = await response.json().catch(() => ({{}}));
        if (!response.ok || payload.ok === false) throw new Error(payload.error || t("pipelineStatusRequestFailed"));
        renderJobStatusBar(payload.job || {{status: payload.status || "cancelling"}});
        await pollPipelineStatus(activeRunId);
      }} catch (error) {{
        setPipelineResult("error", "<strong>" + escapeHTML(t("error")) + ":</strong> " + escapeHTML(error.message));
        setPipelineButtons(true);
      }}
    }}

    function viewReport() {{
      const selectedReportUrl = reportSelect?.selectedOptions[0]?.value || "";
      if (selectedReportUrl) {{
        window.location.href = selectedReportUrl;
        return;
      }}
      window.location.href = "/datasets#reports";
    }}

    function resizeReportFrame() {{
      if (!reportFrame) return;
      reportFrame.style.height = "clamp(520px, calc(100vh - 248px), 760px)";
    }}

    async function refreshReportList(preferredPath = "") {{
      if (!reportSelect || !reportPath) return;
      const selectedPath = reportSelect.selectedOptions[0]?.dataset.path || "";
      const response = await fetch("/api/reports");
      const payload = await response.json().catch(() => ({{}}));
      if (!response.ok || payload.ok === false || !Array.isArray(payload.reports)) return;
      const reports = payload.reports;
      if (overviewLatestReport) overviewLatestReport.textContent = reports[0]?.name || "-";
      reportSelect.innerHTML = reports.map((report) => {{
        const selected = report.path === (preferredPath || selectedPath) ? " selected" : "";
        return "<option value=\\"" + escapeHTML(report.url) + "\\" data-path=\\"" +
          escapeHTML(report.path) + "\\"" + selected + ">" +
          escapeHTML(report.name + " - " + report.modified) + "</option>";
      }}).join("");
      const selectedOption = reportSelect.selectedOptions[0];
      if (selectedOption) {{
        reportPath.dataset.reportPath = selectedOption.dataset.path;
        reportPath.textContent = t("reportPath") + ": " + selectedOption.dataset.path;
        if (reportFrame && reportFrame.getAttribute("src") !== selectedOption.value) {{
          reportFrame.src = selectedOption.value;
          reportFrame.hidden = false;
          resizeReportFrame();
        }}
      }}
    }}

    async function refreshEvalRuns() {{
      const response = await fetch("/api/eval-runs");
      const payload = await response.json().catch(() => ({{}}));
      const tbody = evalRunsTable.querySelector("tbody");
      if (!response.ok || payload.ok === false || !Array.isArray(payload.runs)) {{
        tbody.innerHTML = "<tr><td colspan='7'>" + escapeHTML(t("unableLoadEvalRecords")) + "</td></tr>";
        return;
      }}
      overviewRunCount.textContent = payload.runs.length;
      if (!payload.runs.length) {{
        updateOverviewLatestRun(null);
        tbody.innerHTML = "<tr><td colspan='7'>" + escapeHTML(t("noEvalRecordsFound")) + "</td></tr>";
        return;
      }}
      updateOverviewLatestRun(payload.runs[0]);
      tbody.innerHTML = payload.runs.map((run) => {{
        const seconds = [run.generation_seconds, run.evaluation_seconds]
          .filter((value) => value !== null && value !== undefined && value !== "")
          .map((value) => Number(value).toFixed(1) + "s").join(" / ");
        const outputCell = run.eval_result_url
          ? "<div class=\\"output-cell\\"><a class=\\"output-action\\" href=\\"" + escapeHTML(run.eval_result_url) + "\\" target=\\"_blank\\" rel=\\"noopener\\" title=\\"" + escapeHTML(run.eval_result_csv || "") + "\\" aria-label=\\"" + escapeHTML(t("openOutputFile") + ": " + (run.eval_result_csv || "")) + "\\">" + escapeHTML(t("openOutputFile")) + "</a></div>"
          : escapeHTML(run.eval_result_csv || "-");
        const comboCell = "<div class=\\"combo-stack\\"><strong>ps" + escapeHTML(run.page_size) + "</strong><small>sim " + escapeHTML(run.similarity_threshold) + "</small><small>" + escapeHTML(run.combo || "-") + "</small></div>";
        const reportAction = run.report_url
          ? "<a class=\\"output-action report-action\\" href=\\"" + escapeHTML(run.report_url) + "\\" title=\\"" + escapeHTML(run.report_path || "") + "\\">" + escapeHTML(t("openReport")) + "</a>"
          : "";
        const actionsCell = reportAction
          ? "<div class=\\"row-actions\\">" + outputCell + reportAction + "</div>"
          : outputCell;
        return "<tr>" +
          "<td>" + escapeHTML(run.run_name) + "<br><small>" + escapeHTML(run.updated_at) + "</small></td>" +
          "<td>" + escapeHTML(run.dataset_name) + "</td>" +
          "<td>" + comboCell + "</td>" +
          "<td>" + statusBadge(run.generation_status) + "</td>" +
          "<td>" + statusBadge(run.evaluation_status) + "</td>" +
          "<td>" + escapeHTML(seconds || "-") + "</td>" +
          "<td>" + actionsCell + "</td>" +
        "</tr>";
      }}).join("");
    }}

    function datasetSourceText(dataset) {{
      return dataset.source === "config" ? t("datasetSourceConfig") : t("datasetSourceUploaded");
    }}

    function selectedRunDatasets() {{
      if (!runDatasetChoices) return [];
      return Array.from(runDatasetChoices.querySelectorAll("input[type='checkbox']:checked"))
        .map((input) => input.value)
        .filter(Boolean);
    }}

    function selectedRunPageSizes() {{
      if (runPageSizeSelect?.value === OTHER_RUN_OPTION) {{
        const customValue = validatedPageSizeValue(runPageSizeCustom?.value);
        return customValue === null ? [] : [customValue];
      }}
      const selectedValue = validatedPageSizeValue(runPageSizeSelect?.value);
      return selectedValue === null ? [] : [selectedValue];
    }}

    function selectedRunSimilarities() {{
      if (runSimilaritySelect?.value === OTHER_RUN_OPTION) {{
        const customValue = validatedSimilarityValue(runSimilarityCustom?.value);
        return customValue === null ? [] : [customValue];
      }}
      const selectedValue = validatedSimilarityValue(runSimilaritySelect?.value);
      return selectedValue === null ? [] : [selectedValue];
    }}

    function validatedPageSizeValue(value) {{
      if (value === null || value === undefined || String(value).trim() === "") return null;
      const numberValue = Number(value);
      if (!Number.isInteger(numberValue) || numberValue < 1 || numberValue > 20) return null;
      return numberValue;
    }}

    function validatedSimilarityValue(value) {{
      if (value === null || value === undefined || String(value).trim() === "") return null;
      const numberValue = Number(value);
      if (!Number.isFinite(numberValue) || numberValue < 0 || numberValue > 0.5) return null;
      return numberValue;
    }}

    function renderRunDatasetChoices(datasets) {{
      if (!runDatasetChoices) return;
      const previousSelection = new Set(selectedRunDatasets());
      const runnableDatasets = (Array.isArray(datasets) ? datasets : []).filter((dataset) => dataset.source === "config" && dataset.runnable !== false);
      if (!runnableDatasets.length) {{
        runDatasetChoices.innerHTML = "<div class=\\"result\\">" + escapeHTML(t("noRunnableDatasets")) + "</div>";
        return;
      }}
      runDatasetChoices.innerHTML = runnableDatasets.map((dataset) => {{
        const name = dataset.name || dataset.dataset_id || "";
        const shouldCheck = previousSelection.size ? previousSelection.has(name) : true;
        return "<label class=\\"run-dataset-choice\\">" +
          "<input type=\\"checkbox\\" name=\\"runDataset\\" value=\\"" + escapeHTML(name) + "\\"" + (shouldCheck ? " checked=\\"checked\\"" : "") + ">" +
          "<span>" + escapeHTML(name || "-") + (dataset.dataset_id ? "<br><small>ID: " + escapeHTML(dataset.dataset_id) + "</small>" : "") + "</span>" +
        "</label>";
      }}).join("");
    }}

    function renderRunOptionGroup(selectElement, values) {{
      if (!selectElement) return;
      const options = Array.isArray(values) ? values : [];
      if (!options.length) {{
        selectElement.innerHTML = "<option value=\\"" + OTHER_RUN_OPTION + "\\">" + escapeHTML(t("otherOption")) + "</option>";
        return;
      }}
      const currentValue = selectElement.value;
      selectElement.innerHTML = options.map((value) => {{
        const optionValue = String(value);
        return "<option value=\\"" + escapeHTML(optionValue) + "\\">" + escapeHTML(optionValue) + "</option>";
      }}).join("") + "<option value=\\"" + OTHER_RUN_OPTION + "\\">" + escapeHTML(t("otherOption")) + "</option>";
      if (currentValue && options.map(String).includes(currentValue)) selectElement.value = currentValue;
      else selectElement.selectedIndex = 0;
    }}

    function renderRunParameterChoices(options = {{}}) {{
      renderRunOptionGroup(runPageSizeSelect, options.page_sizes || []);
      renderRunOptionGroup(runSimilaritySelect, options.similarity_thresholds || []);
      toggleRunCustomInput(runPageSizeSelect, runPageSizeCustom);
      toggleRunCustomInput(runSimilaritySelect, runSimilarityCustom);
    }}

    function toggleRunCustomInput(selectElement, inputElement) {{
      if (!selectElement || !inputElement) return;
      const showCustom = selectElement.value === OTHER_RUN_OPTION;
      inputElement.hidden = !showCustom;
      if (!showCustom) inputElement.value = "";
    }}

    function renderDatasetInventory(datasets) {{
      if (!datasetsTable) return;
      const tbody = datasetsTable.querySelector("tbody");
      if (datasetInventoryCount) datasetInventoryCount.textContent = String(Array.isArray(datasets) ? datasets.length : 0);
      if (!Array.isArray(datasets) || !datasets.length) {{
        tbody.innerHTML = "<tr><td colspan='5'>" + escapeHTML(t("noDatasetsFound")) + "</td></tr>";
        return;
      }}
      tbody.innerHTML = datasets.map((dataset) => {{
        const nameCell = escapeHTML(dataset.name || "-") +
          (dataset.dataset_id ? "<br><small>ID: " + escapeHTML(dataset.dataset_id) + "</small>" : "");
        const sourceClass = dataset.source === "config" ? "config" : "uploaded";
        return "<tr>" +
          "<td class=\\"dataset-name-cell\\">" + nameCell + "</td>" +
          "<td><span class=\\"dataset-source-badge " + sourceClass + "\\">" + escapeHTML(datasetSourceText(dataset)) + "</span></td>" +
          "<td>" + escapeHTML(dataset.rows === "" ? "-" : dataset.rows) + "</td>" +
          "<td>" + escapeHTML(dataset.modified || "-") + "</td>" +
          "<td class=\\"dataset-path-cell\\">" + escapeHTML(dataset.path || "-") + "</td>" +
        "</tr>";
      }}).join("");
    }}

    function updateDatasetFileSummary() {{
      const selectedPath = manualQaDatasetSelect?.value || "";
      if (datasetFileName) datasetFileName.textContent = selectedPath || "-";
      if (datasetFileMeta) datasetFileMeta.textContent = selectedPath ? "Ready for Q&A save" : "0 rows selected";
    }}

    function compactCellText(value, maxLength = 72) {{
      const text = String(value ?? "").replace(/\\s+/g, " ").trim();
      return text.length > maxLength ? text.slice(0, maxLength - 1) + "..." : text;
    }}

    function renderDatasetRowsPreview(payload) {{
      if (!datasetRowsTableBody) return;
      const selectedPath = manualQaDatasetSelect?.value || "";
      const rows = payload && Array.isArray(payload.preview_rows) ? payload.preview_rows : [];
      lastDatasetRowsPayload = payload || null;
      if (datasetInventoryCount) datasetInventoryCount.textContent = String(payload && Number.isFinite(Number(payload.rows)) ? payload.rows : rows.length);
      if (!rows.length) {{
        if (datasetViewAllButton) datasetViewAllButton.hidden = true;
        datasetRowsTableBody.innerHTML = "<tr><td colspan='5'>" + escapeHTML(t("noDatasetsFound")) + "</td></tr>";
        return;
      }}
      if (datasetViewAllButton) {{
        datasetViewAllButton.hidden = rows.length <= 3;
        datasetViewAllButton.textContent = datasetRowsExpanded ? "收起" : "查看全部";
      }}
      const visibleRows = datasetRowsExpanded ? rows : rows.slice(0, 1);
      datasetRowsTableBody.innerHTML = visibleRows.map((row, index) => {{
        const queryId = row.query_id || String(index + 1);
        return "<tr>" +
          "<td>" + escapeHTML(queryId) + "</td>" +
          "<td class=\\"dataset-row-question\\" title=\\"" + escapeHTML(row.query || "") + "\\">" + escapeHTML(compactCellText(row.query || "", 92)) + "</td>" +
          "<td class=\\"dataset-row-answer\\" title=\\"" + escapeHTML(row.expected_answer || "") + "\\">" + escapeHTML(compactCellText(row.expected_answer || "", 110)) + "</td>" +
          "<td class=\\"dataset-path-cell\\">" + escapeHTML(selectedPath || "-") + "</td>" +
          "<td class=\\"dataset-row-actions\\"><button type=\\"button\\" aria-label=\\"查看当前问答\\" title=\\"查看\\">◎</button><button type=\\"button\\" aria-label=\\"更多操作\\" title=\\"更多\\">⋮</button></td>" +
        "</tr>";
      }}).join("");
    }}

    async function loadDatasetRowsPreview() {{
      if (!manualQaDatasetSelect || !manualQaDatasetSelect.value || !datasetRowsTableBody) return;
      datasetRowsExpanded = false;
      if (datasetViewAllButton) datasetViewAllButton.hidden = true;
      datasetRowsTableBody.innerHTML = "<tr><td colspan='5'>" + escapeHTML(t("loadingDatasets")) + "</td></tr>";
      try {{
        const response = await fetch("/api/datasets/preview?path=" + encodeURIComponent(manualQaDatasetSelect.value));
        const payload = await response.json().catch(() => ({{}}));
        if (!response.ok || payload.ok === false) throw new Error(payload.error || t("unableLoadDatasets"));
        renderDatasetRowsPreview(payload);
      }} catch (error) {{
        datasetRowsTableBody.innerHTML = "<tr><td colspan='5'>" + escapeHTML(t("unableLoadDatasets")) + "</td></tr>";
      }}
    }}

    function updateManualQaCharacterCounts() {{
      if (manualQaQuestionCount && manualQaQuestionTextarea) {{
        manualQaQuestionCount.textContent = String(manualQaQuestionTextarea.value.length) + "/2000";
      }}
      if (manualQaAnswerCount && manualQaAnswerTextarea) {{
        manualQaAnswerCount.textContent = String(manualQaAnswerTextarea.value.length) + "/4000";
      }}
    }}

    function renderManualQaDatasetChoices(datasets) {{
      if (!manualQaDatasetSelect) return;
      const writableDatasets = Array.isArray(datasets)
        ? datasets.filter((dataset) => String(dataset.path || "").toLowerCase().endsWith(".csv"))
        : [];
      if (!writableDatasets.length) {{
        manualQaDatasetSelect.innerHTML = "<option value=\\"\\">" + escapeHTML(t("noDatasetsFound")) + "</option>";
        manualQaDatasetSelect.disabled = true;
        if (manualQaButton) manualQaButton.disabled = true;
        if (manualQaDatasetPath) manualQaDatasetPath.textContent = "";
        return;
      }}
      const selectedValue = manualQaDatasetSelect.value;
      manualQaDatasetSelect.disabled = false;
      if (manualQaButton) manualQaButton.disabled = false;
      manualQaDatasetSelect.innerHTML = writableDatasets.map((dataset) => {{
        const path = String(dataset.path || "");
        const selected = path === selectedValue ? " selected" : "";
        const label = dataset.name || path || "dataset";
        return "<option value=\\"" + escapeHTML(path) + "\\" title=\\"" + escapeHTML(path) + "\\"" + selected + ">" +
          escapeHTML(label) +
          "</option>";
      }}).join("");
      updateManualQaDatasetPath();
      updateDatasetFileSummary();
    }}

    function updateManualQaDatasetPath() {{
      if (!manualQaDatasetPath || !manualQaDatasetSelect) return;
      const selectedPath = manualQaDatasetSelect.value || "";
      manualQaDatasetPath.textContent = selectedPath;
      manualQaDatasetPath.title = selectedPath;
      updateDatasetFileSummary();
    }}

    async function refreshDatasets() {{
      if (!datasetsTable) return;
      const tbody = datasetsTable.querySelector("tbody");
      tbody.innerHTML = "<tr><td colspan='5'>" + escapeHTML(t("loadingDatasets")) + "</td></tr>";
      try {{
        const response = await fetch("/api/datasets");
        const payload = await response.json().catch(() => ({{}}));
        if (!response.ok || payload.ok === false || !Array.isArray(payload.datasets)) {{
          throw new Error(payload.error || t("unableLoadDatasets"));
        }}
        currentDatasets = payload.datasets;
        currentRunOptions = payload.run_options || {{page_sizes: [], similarity_thresholds: []}};
        renderDatasetInventory(currentDatasets);
        renderManualQaDatasetChoices(currentDatasets);
        renderRunDatasetChoices(currentDatasets);
        renderRunParameterChoices(currentRunOptions);
        await loadDatasetRowsPreview();
      }} catch (error) {{
        tbody.innerHTML = "<tr><td colspan='5'>" + escapeHTML(t("unableLoadDatasets")) + "</td></tr>";
      }}
    }}

    async function startPipeline(activeButton) {{
      const selectedDatasets = selectedRunDatasets();
      if (!selectedDatasets.length) {{
        setPipelineResult("error", "<strong>" + escapeHTML(t("error")) + ":</strong> " + escapeHTML(t("selectRunDataset")));
        return;
      }}
      const selectedPageSizes = selectedRunPageSizes();
      const selectedSimilarities = selectedRunSimilarities();
      if (!selectedPageSizes.length || !selectedSimilarities.length) {{
        setPipelineResult("error", "<strong>" + escapeHTML(t("error")) + ":</strong> " + escapeHTML(t("selectRunParameter")));
        return;
      }}
      setPipelineButtons(true);
      activeButton.textContent = activeButton.dataset.loadingText;
      setPipelineResult("", t("startingPipelineJob"));
      renderJobStatusBar({{status: "starting", progress: {{percent: 0}}}});
      try {{
        const requestBody = {{"config": "scripts/eval-cfg.yaml", "mode": "standard", "stage": "all", "dry_run": false, "overwrite": false, datasets: selectedRunDatasets(), page_sizes: selectedRunPageSizes(), similarity_thresholds: selectedRunSimilarities()}};
        const response = await fetch("/api/pipeline/run", {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify(requestBody)
        }});
        const payload = await response.json().catch(() => ({{}}));
        if (!response.ok || payload.ok === false) throw new Error(payload.error || t("pipelineStartFailed"));
        activeRunId = payload.run_id || "";
        setPipelineButtons(true);
        renderJobStatusBar(payload.job || {{status: payload.status || "running"}});
        setPipelineResult("success", "<strong>" + escapeHTML(t("pipeline")) + " " + escapeHTML(statusText(payload.status || "running")) + "</strong><div>" + escapeHTML(t("runId")) + ": " + escapeHTML(activeRunId) + "</div>");
        activeButton.textContent = activeButton.dataset.runningText;
        await pollPipelineStatus(activeRunId);
      }} catch (error) {{
        setPipelineResult("error", "<strong>" + escapeHTML(t("error")) + ":</strong> " + escapeHTML(error.message));
        renderJobStatusBar({{status: "failed", error: error.message, progress: {{percent: 0}}}});
        setPipelineButtons(false);
      }}
    }}

    openUploadDialog.addEventListener("click", () => {{
      if (uploadDialog.showModal) uploadDialog.showModal();
      else uploadDialog.setAttribute("open", "");
    }});
    closeUploadDialog.addEventListener("click", () => uploadDialog.close());
    runPipelineButton.addEventListener("click", () => startPipeline(runPipelineButton));
    terminatePipelineButton.addEventListener("click", terminatePipeline);
    backButton.addEventListener("click", viewReport);
    pageLanguage.addEventListener("change", () => applyPageLanguage(pageLanguage.value));
    document.getElementById("refreshReportsButton").addEventListener("click", () => refreshReportList());
    document.getElementById("refreshEvalRunsButton").addEventListener("click", () => refreshEvalRuns());
    document.querySelectorAll(".dataset-list-actions button").forEach((actionButton) => {{
      actionButton.addEventListener("click", (event) => event.stopPropagation());
    }});
    document.getElementById("refreshDatasetsButton").addEventListener("click", (event) => {{
      event.preventDefault();
      event.stopPropagation();
      refreshDatasets();
    }});
    if (datasetViewAllButton) {{
      datasetViewAllButton.addEventListener("click", () => {{
        datasetRowsExpanded = !datasetRowsExpanded;
        if (lastDatasetRowsPayload) renderDatasetRowsPreview(lastDatasetRowsPayload);
      }});
    }}
    if (manualQaDatasetSelect) manualQaDatasetSelect.addEventListener("change", () => {{
      updateManualQaDatasetPath();
      loadDatasetRowsPreview();
    }});
    runPageSizeSelect.addEventListener("change", () => toggleRunCustomInput(runPageSizeSelect, runPageSizeCustom));
    runSimilaritySelect.addEventListener("change", () => toggleRunCustomInput(runSimilaritySelect, runSimilarityCustom));

    if (reportSelect) {{
      reportSelect.addEventListener("change", () => {{
        const option = reportSelect.selectedOptions[0];
        if (!option) return;
        reportPath.dataset.reportPath = option.dataset.path;
        reportPath.textContent = t("reportPath") + ": " + option.dataset.path;
        reportFrame.src = option.value;
        reportFrame.hidden = false;
        resizeReportFrame();
      }});
    }}

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      button.disabled = true;
      button.textContent = button.dataset.loadingText;
      setResult("", t("uploadingDataset"));
      try {{
        const response = await fetch("/api/datasets/upload", {{
          method: "POST",
          body: new FormData(form)
        }});
        const payload = await response.json().catch(() => ({{}}));
        if (!response.ok || payload.ok === false) throw new Error(payload.error || t("uploadFailed"));
        const path = payload.path || "";
        setResult(
          "success",
          "<strong>" + escapeHTML(t("uploaded")) + " " + escapeHTML(payload.dataset_name || "dataset") + "</strong>" +
            "<div>" + escapeHTML(t("rows")) + ": " + escapeHTML(payload.rows || 0) + "</div>" +
            "<div>" + escapeHTML(t("savedDatasetPath")) + ": " + escapeHTML(path) + "</div>" +
            "<div>" + escapeHTML(t("savedUnder")) + "</div>"
        );
        form.reset();
        await refreshDatasets();
      }} catch (error) {{
        setResult("error", "<strong>" + escapeHTML(t("error")) + ":</strong> " + escapeHTML(error.message));
      }} finally {{
        button.disabled = false;
        button.textContent = button.dataset.readyText;
      }}
    }});

    if (manualQaForm) {{
      manualQaForm.addEventListener("submit", async (event) => {{
        event.preventDefault();
        manualQaButton.disabled = true;
        manualQaButton.textContent = manualQaButton.dataset.loadingText;
        setManualQaResult("", t("manualQaSaving"));
        try {{
          const formData = new FormData(manualQaForm);
          const targetDataset = manualQaDatasetSelect?.value || formData.get("target_dataset") || "";
          const response = await fetch("/api/datasets/manual-qa", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify({{
              target_dataset: targetDataset,
              question: formData.get("question") || "",
              expected_answer: formData.get("expected_answer") || ""
            }})
          }});
          const payload = await response.json().catch(() => ({{}}));
          if (!response.ok || payload.ok === false) throw new Error(payload.error || t("manualQaFailed"));
          setManualQaResult(
            "success",
            "<strong>" + escapeHTML(t("manualQaSaved")) + " " + escapeHTML(payload.dataset_name || "manual_qa") + "</strong>" +
              "<div>" + escapeHTML(t("rows")) + ": " + escapeHTML(payload.rows || 0) + "</div>" +
              "<div>query_id: " + escapeHTML(payload.appended_query_id || "") + "</div>" +
              "<div>" + escapeHTML(t("savedDatasetPath")) + ": " + escapeHTML(payload.path || "") + "</div>"
          );
          manualQaForm.reset();
          if (manualQaDatasetSelect) manualQaDatasetSelect.value = targetDataset;
          generatedAnswerPassages = [];
          renderRetrievedPassages([]);
          await refreshDatasets();
          updateManualQaDatasetPath();
          updateManualQaCharacterCounts();
        }} catch (error) {{
          setManualQaResult("error", "<strong>" + escapeHTML(t("error")) + ":</strong> " + escapeHTML(error.message));
        }} finally {{
          manualQaButton.disabled = false;
          manualQaButton.textContent = manualQaButton.dataset.readyText;
        }}
      }});
    }}

    if (generateAnswerButton) {{
      generateAnswerButton.addEventListener("click", async () => {{
        generateAnswerButton.disabled = true;
        generateAnswerButton.textContent = generateAnswerButton.dataset.loadingText;
        setManualQaResult("", t("generatingAnswer"));
        try {{
          const formData = new FormData(manualQaForm);
          const response = await fetch("/api/datasets/generate-answer", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify({{
              question: formData.get("question") || "",
              language: pageLanguage.value || "en"
            }})
          }});
          const payload = await response.json().catch(() => ({{}}));
          if (!response.ok || payload.ok === false) throw new Error(payload.error || t("answerGenerationFailed"));
          manualQaAnswerTextarea.value = payload.expected_answer || "";
          updateManualQaCharacterCounts();
          generatedAnswerPassages = payload.passages || [];
          renderRetrievedPassages(generatedAnswerPassages);
          setManualQaResult("success", escapeHTML(t("generatedAnswer")));
        }} catch (error) {{
          setManualQaResult("error", "<strong>" + escapeHTML(t("error")) + ":</strong> " + escapeHTML(error.message));
        }} finally {{
          generateAnswerButton.disabled = false;
          generateAnswerButton.textContent = generateAnswerButton.dataset.readyText;
        }}
      }});
    }}

    if (regenerateAnswerButton) {{
      regenerateAnswerButton.addEventListener("click", async () => {{
        regenerateAnswerButton.disabled = true;
        setManualQaResult("", t("generatingAnswer"));
        try {{
          const formData = new FormData(manualQaForm);
          const response = await fetch("/api/datasets/regenerate-answer", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify({{
              question: formData.get("question") || "",
              current_answer: manualQaAnswerTextarea.value || "",
              passages: generatedAnswerPassages,
              language: pageLanguage.value || "en"
            }})
          }});
          const payload = await response.json().catch(() => ({{}}));
          if (!response.ok || payload.ok === false) throw new Error(payload.error || t("answerGenerationFailed"));
          manualQaAnswerTextarea.value = payload.expected_answer || "";
          updateManualQaCharacterCounts();
          setManualQaResult("success", escapeHTML(t("generatedAnswer")));
        }} catch (error) {{
          setManualQaResult("error", "<strong>" + escapeHTML(t("error")) + ":</strong> " + escapeHTML(error.message));
        }} finally {{
          regenerateAnswerButton.disabled = generatedAnswerPassages.length === 0;
        }}
      }});
    }}

    if (clearGeneratedAnswerButton) {{
      clearGeneratedAnswerButton.addEventListener("click", () => {{
        const targetDataset = manualQaDatasetSelect?.value || "";
        if (manualQaForm) manualQaForm.reset();
        if (manualQaDatasetSelect) manualQaDatasetSelect.value = targetDataset;
        if (manualQaQuestionTextarea) manualQaQuestionTextarea.value = "";
        if (manualQaAnswerTextarea) manualQaAnswerTextarea.value = "";
        updateManualQaDatasetPath();
        updateManualQaCharacterCounts();
        generatedAnswerPassages = [];
        renderRetrievedPassages([]);
        setManualQaResult("", "");
      }});
    }}

    if (manualQaAnswerTextarea) {{
      manualQaAnswerTextarea.addEventListener("input", () => {{
        updateManualQaCharacterCounts();
        if (!manualQaAnswerTextarea.value.trim() && regenerateAnswerButton) regenerateAnswerButton.disabled = generatedAnswerPassages.length === 0;
      }});
    }}

    if (manualQaQuestionTextarea) {{
      manualQaQuestionTextarea.addEventListener("input", updateManualQaCharacterCounts);
    }}

    if (askModelForm) {{
      askModelForm.addEventListener("submit", async (event) => {{
        event.preventDefault();
        askModelButton.disabled = true;
        askModelButton.textContent = askModelButton.dataset.loadingText;
        chatAnswer.className = "result chat-answer";
        chatAnswer.textContent = t("thinking");
        try {{
          const question = new FormData(askModelForm).get("question") || "";
          const language = pageLanguage.value || "en";
          const reportUrl = reportSelect?.selectedOptions[0]?.value || "";
          const response = await fetch("/api/chat", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify({{question, report_url: reportUrl, language}})
          }});
          const payload = await response.json().catch(() => ({{}}));
          if (!response.ok || payload.ok === false) throw new Error(payload.error || t("chatRequestFailed"));
          chatAnswer.className = "result chat-answer success";
          chatAnswer.textContent = payload.answer || "";
          lastReportQuestion = String(question || "").trim();
          lastReportAnswer = String(payload.answer || "").trim();
          if (addReportAnswerToQaButton) addReportAnswerToQaButton.disabled = !(lastReportQuestion && lastReportAnswer);
        }} catch (error) {{
          lastReportQuestion = "";
          lastReportAnswer = "";
          if (addReportAnswerToQaButton) addReportAnswerToQaButton.disabled = true;
          chatAnswer.className = "result chat-answer error";
          chatAnswer.textContent = t("error") + ": " + error.message;
        }} finally {{
          askModelButton.disabled = false;
          askModelButton.textContent = askModelButton.dataset.readyText;
        }}
      }});
    }}

    if (addReportAnswerToQaButton) {{
      addReportAnswerToQaButton.addEventListener("click", () => {{
        if (!lastReportQuestion || !lastReportAnswer) return;
        manualQaQuestionTextarea.value = lastReportQuestion;
        manualQaAnswerTextarea.value = lastReportAnswer;
        updateManualQaCharacterCounts();
        generatedAnswerPassages = [];
        renderRetrievedPassages([]);
        setManualQaResult("success", escapeHTML(t("addedToQaCandidates")));
        addReportAnswerToQaButton.disabled = true;
      }});
    }}

    applyPageLanguage(window.localStorage.getItem("ragEvalPageLanguage") || "en");
    renderJobStatusBar({{status: "idle", progress: {{percent: 0}}}});
    refreshReportList();
    refreshEvalRuns();
    refreshDatasets();
    updateManualQaCharacterCounts();
  </script>
</body>
</html>"""


class DatasetUploadHandler(BaseHTTPRequestHandler):
    server_version = "RagEvalDatasetAPI/0.1"
    upload_root = UPLOAD_ROOT
    reports_root = REPORTS_ROOT
    eval_runs_root = EVAL_RUNS_ROOT
    chat_supplement_csv_path = CHAT_SUPPLEMENT_CSV_PATH
    pipeline_config_path = DEFAULT_PIPELINE_CONFIG
    pipeline_jobs = PIPELINE_JOBS

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        parsed_path = parsed.path
        if parsed_path == "/datasets":
            self.write_html(render_upload_page(list_result_reports(self.reports_root)))
            return
        if parsed_path == "/api/reports":
            self.write_json({"ok": True, "reports": list_result_reports(self.reports_root)})
            return
        if parsed_path == "/api/eval-runs":
            self.write_json({"ok": True, "runs": list_eval_runs(self.eval_runs_root)})
            return
        if parsed_path == "/api/datasets":
            self.write_json(
                {
                    "ok": True,
                    "datasets": list_datasets(self.upload_root, self.pipeline_config_path),
                    "run_options": list_run_options(self.pipeline_config_path),
                }
            )
            return
        if parsed_path == "/api/datasets/preview":
            dataset_path_value = (parse_qs(parsed.query).get("path") or [""])[0]
            try:
                payload = dataset_preview_payload(self.resolve_manual_qa_dataset_path(dataset_path_value))
            except UploadError as exc:
                self.write_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.write_json(payload)
            return
        if parsed_path == "/eval-output":
            output_path = safe_eval_output_path(self.path, self.eval_runs_root)
            if output_path is None:
                self.write_json({"ok": False, "error": "Not found"}, status=404)
                return
            self.write_text(output_path.read_text(encoding="utf-8"), content_type="text/csv")
            return
        if parsed_path == "/api/pipeline/status":
            run_id = (parse_qs(parsed.query).get("run_id") or [""])[0]
            job = self.pipeline_jobs.get(run_id)
            if not run_id or job is None:
                self.write_json({"ok": False, "error": "Pipeline job not found"}, status=404)
                return
            job["progress"] = job_progress(job)
            self.write_json({"ok": True, "job": job})
            return
        if parsed_path.startswith("/reports/"):
            report_path = safe_report_path(self.path, self.reports_root)
            if report_path is None:
                self.write_json({"ok": False, "error": "Not found"}, status=404)
                return
            self.write_html(inject_report_main_button(report_path.read_text(encoding="utf-8")))
            return
        self.write_json({"ok": False, "error": "Not found"}, status=404)

    def do_HEAD(self) -> None:
        self.write_method_not_allowed()

    def do_POST(self) -> None:
        parsed_path = urlparse(self.path).path
        if parsed_path == "/api/chat":
            try:
                payload = self.handle_chat()
            except UploadError as exc:
                self.write_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.write_json(payload)
            return

        if parsed_path == "/api/chat/save":
            try:
                payload = self.handle_chat_save()
            except UploadError as exc:
                self.write_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.write_json(payload)
            return

        if parsed_path == "/api/pipeline/run":
            try:
                payload = self.handle_pipeline_run()
            except UploadError as exc:
                self.write_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.write_json(payload)
            return

        if parsed_path == "/api/pipeline/cancel":
            try:
                payload = self.handle_pipeline_cancel()
            except UploadError as exc:
                self.write_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.write_json(payload)
            return

        if parsed_path == "/api/datasets/manual-qa":
            try:
                payload = self.handle_manual_qa()
            except UploadError as exc:
                self.write_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.write_json(payload)
            return

        if parsed_path == "/api/datasets/generate-answer":
            try:
                payload = self.handle_generate_answer()
            except UploadError as exc:
                self.write_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.write_json(payload)
            return

        if parsed_path == "/api/datasets/regenerate-answer":
            try:
                payload = self.handle_regenerate_answer()
            except UploadError as exc:
                self.write_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.write_json(payload)
            return

        if parsed_path == "/api/datasets/generate-qa":
            try:
                payload = self.handle_generate_qa()
            except UploadError as exc:
                self.write_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.write_json(payload)
            return

        if parsed_path == "/api/datasets/bulk-qa":
            try:
                payload = self.handle_bulk_qa()
            except UploadError as exc:
                self.write_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.write_json(payload)
            return

        if parsed_path != "/api/datasets/upload":
            self.write_json({"ok": False, "error": "Not found"}, status=404)
            return

        try:
            payload = self.handle_upload()
        except UploadError as exc:
            self.write_json({"ok": False, "error": str(exc)}, status=400)
            return

        self.write_json(payload)

    def do_PUT(self) -> None:
        self.write_method_not_allowed()

    def do_DELETE(self) -> None:
        self.write_method_not_allowed()

    def do_PATCH(self) -> None:
        self.write_method_not_allowed()

    def do_OPTIONS(self) -> None:
        self.write_method_not_allowed()

    def handle_upload(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        boundary = parse_content_type_boundary(content_type)
        content_length_header = self.headers.get("Content-Length")
        if content_length_header is None:
            raise UploadError("Content-Length header is required")
        try:
            content_length = int(content_length_header)
        except ValueError as exc:
            raise UploadError("Content-Length header must be a valid integer") from exc
        if content_length < 0:
            raise UploadError("Content-Length header must be a valid integer")
        if content_length > MAX_UPLOAD_BYTES:
            raise UploadError(f"Upload is too large; maximum size is {MAX_UPLOAD_BYTES} bytes")

        body = self.rfile.read(content_length)
        if len(body) != content_length:
            raise UploadError("Request body ended before Content-Length bytes were read")

        form = parse_multipart_form(body, boundary)
        file_item = form.get("file")
        if file_item is None:
            raise UploadError("Upload requires a file field")
        if isinstance(file_item, list):
            if not file_item:
                raise UploadError("Upload requires a file field")
            file_item = file_item[0]
        if not isinstance(file_item, MultipartFile):
            raise UploadError("Upload requires a file field")

        filename = file_item.filename
        if not filename:
            raise UploadError("Upload requires a file field")
        filename_lower = filename.lower()
        if filename_lower.endswith(".csv"):
            source_format = "csv"
        elif filename_lower.endswith(".pdf"):
            source_format = "pdf"
        else:
            raise UploadError("Uploaded file must be a .csv or .pdf file")

        name = form.get("name", "")
        if isinstance(name, list):
            name = name[0] if name else ""
        if isinstance(name, MultipartFile):
            name = ""
        dataset_name = str(name).strip() or filename

        if source_format == "csv":
            rows = parse_and_validate_csv(file_item.content)
        else:
            rows = parse_and_validate_pdf(file_item.content)
        return save_dataset(
            rows,
            dataset_name,
            upload_root=self.upload_root,
            source_format=source_format,
        )

    def handle_manual_qa(self) -> dict[str, Any]:
        payload = self.read_json_body()
        dataset_csv_path = self.resolve_manual_qa_dataset_path(str(payload.get("target_dataset", "")))
        question = str(payload.get("question", "")).strip()
        expected_answer = str(payload.get("expected_answer", "")).strip()
        return append_qa_to_dataset(dataset_csv_path, question, expected_answer)

    def handle_generate_answer(self) -> dict[str, Any]:
        payload = self.read_json_body()
        return generate_answer_from_question(
            str(payload.get("question", "")),
            str(payload.get("language", "")),
            self.pipeline_config_path,
        )

    def handle_regenerate_answer(self) -> dict[str, Any]:
        payload = self.read_json_body()
        passages = payload.get("passages")
        if not isinstance(passages, list):
            raise UploadError("Retrieved passages are required before regenerating.")
        return regenerate_answer_from_passages(
            str(payload.get("question", "")),
            str(payload.get("current_answer", "")),
            passages,
            str(payload.get("language", "")),
            self.pipeline_config_path,
        )

    def handle_generate_qa(self) -> dict[str, Any]:
        payload = self.read_json_body()
        try:
            count = int(payload.get("count") or 5)
        except (TypeError, ValueError) as exc:
            raise UploadError("QA generation count must be one of 5, 10, or 20") from exc
        return generate_qa_candidates(
            str(payload.get("source_text", "")),
            count,
            str(payload.get("language", "")),
        )

    def handle_bulk_qa(self) -> dict[str, Any]:
        payload = self.read_json_body()
        dataset_csv_path = self.resolve_manual_qa_dataset_path(str(payload.get("target_dataset", "")))
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise UploadError("QA rows must be a list")
        return append_bulk_qa_to_dataset(dataset_csv_path, rows)

    def resolve_manual_qa_dataset_path(self, value: str) -> Path:
        clean_value = value.strip()
        if not clean_value:
            raise UploadError("Target dataset is required")
        candidate = Path(clean_value)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        candidate = candidate.resolve()
        allowed_roots = [
            ROOT.resolve(),
            self.upload_root.resolve(),
            self.pipeline_config_path.resolve().parent,
        ]
        if not any(candidate == root or root in candidate.parents for root in allowed_roots):
            raise UploadError("Target dataset path is not allowed")
        return candidate

    def read_json_body(self) -> dict[str, Any]:
        content_length_header = self.headers.get("Content-Length")
        if content_length_header is None:
            raise UploadError("Content-Length header is required")
        try:
            content_length = int(content_length_header)
        except ValueError as exc:
            raise UploadError("Content-Length header must be a valid integer") from exc
        if content_length < 0:
            raise UploadError("Content-Length header must be a valid integer")
        if content_length > MAX_JSON_BYTES:
            raise UploadError(f"Request body is too large; maximum size is {MAX_JSON_BYTES} bytes")

        body = self.rfile.read(content_length)
        if len(body) != content_length:
            raise UploadError("Request body ended before Content-Length bytes were read")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UploadError("Request body must be valid UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise UploadError("Request body must be a JSON object")
        return payload

    def handle_chat(self) -> dict[str, Any]:
        payload = self.read_json_body()
        question = str(payload.get("question", ""))
        if not question.strip():
            raise UploadError("Chat question is required")
        answer_payload = answer_chat_question(
            question,
            str(payload.get("report_url", "")),
            self.reports_root,
            str(payload.get("language", "")),
        )
        return {
            "ok": True,
            **answer_payload,
            "reports": list_result_reports(self.reports_root),
        }

    def handle_chat_save(self) -> dict[str, Any]:
        payload = self.read_json_body()
        supplement_payload = append_chat_supplement(
            str(payload.get("question", "")),
            str(payload.get("answer", "")),
            self.chat_supplement_csv_path,
        )
        return {"ok": True, **supplement_payload}

    def handle_pipeline_run(self) -> dict[str, Any]:
        payload = self.read_json_body()
        mode = str(payload.get("mode") or "full").strip().lower()
        config_path = pipeline_config_path(payload.get("config"))
        selected_datasets = normalize_selected_datasets(payload.get("datasets"))
        selected_page_sizes = normalize_page_sizes(payload.get("page_sizes"))
        selected_similarity_thresholds = normalize_similarity_thresholds(payload.get("similarity_thresholds"))
        overwrite = pipeline_bool(payload.get("overwrite", False))
        stage = str(payload.get("stage") or "all")
        dry_run = pipeline_bool(payload.get("dry_run", False))
        if mode == "quick":
            config_path = create_quick_pipeline_config(
                config_path,
                "quick",
                selected_datasets,
                selected_page_sizes,
                selected_similarity_thresholds,
            )
            overwrite = True
        elif mode == "standard":
            config_path = create_standard_pipeline_config(
                config_path,
                "standard",
                selected_datasets,
                selected_page_sizes,
                selected_similarity_thresholds,
            )
            overwrite = True
        elif mode not in {"full", "all"}:
            raise UploadError("Pipeline mode must be full, standard, or quick")
        if stage_runs_generation(stage) and not dry_run:
            retrieval_error = check_pipeline_retrieval_service(config_path)
            if retrieval_error:
                raise UploadError(retrieval_error)
        return start_pipeline_job(
            config_path=config_path,
            stage=stage,
            dry_run=dry_run,
            overwrite=overwrite,
            job_store=self.pipeline_jobs,
        )

    def handle_pipeline_cancel(self) -> dict[str, Any]:
        payload = self.read_json_body()
        return cancel_pipeline_job(str(payload.get("run_id", "")), self.pipeline_jobs)

    def write_html(self, html_text: str, status: int = 200) -> None:
        body = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_text(self, text: str, status: int = 200, content_type: str = "text/plain") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_method_not_allowed(self) -> None:
        body = json.dumps({"ok": False, "error": "Method not allowed"}).encode("utf-8")
        self.send_response(405)
        self.send_header("Allow", "GET, POST")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def make_handler(
    upload_root: Path = UPLOAD_ROOT,
    reports_root: Path = REPORTS_ROOT,
    eval_runs_root: Path = EVAL_RUNS_ROOT,
    chat_supplement_csv_path: Path = CHAT_SUPPLEMENT_CSV_PATH,
    pipeline_config_path: Path = DEFAULT_PIPELINE_CONFIG,
    pipeline_jobs: dict[str, dict[str, Any]] | None = None,
) -> type[DatasetUploadHandler]:
    class ConfiguredDatasetUploadHandler(DatasetUploadHandler):
        pass

    ConfiguredDatasetUploadHandler.upload_root = upload_root
    ConfiguredDatasetUploadHandler.reports_root = reports_root
    ConfiguredDatasetUploadHandler.eval_runs_root = eval_runs_root
    ConfiguredDatasetUploadHandler.chat_supplement_csv_path = chat_supplement_csv_path
    ConfiguredDatasetUploadHandler.pipeline_config_path = pipeline_config_path
    ConfiguredDatasetUploadHandler.pipeline_jobs = PIPELINE_JOBS if pipeline_jobs is None else pipeline_jobs
    return ConfiguredDatasetUploadHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG eval dataset upload API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--upload-root", type=Path, default=UPLOAD_ROOT)
    parser.add_argument("--reports-root", type=Path, default=REPORTS_ROOT)
    parser.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CONFIG)
    return parser.parse_args()


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def main() -> None:
    load_dotenv_if_available()
    args = parse_args()
    import uvicorn

    from rag_eval_pipeline.api_app import create_app
    from rag_eval_pipeline.api_services.state import ApiState

    app = create_app(
        state=ApiState(
            upload_root=args.upload_root,
            reports_root=args.reports_root,
            pipeline_config_path=args.pipeline_config,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
