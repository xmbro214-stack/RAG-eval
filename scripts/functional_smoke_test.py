"""Functional smoke tests for the local RAG evaluation console.

The default checks are read-only and safe to run repeatedly against a local
server. Pass ``--include-write-checks`` to exercise upload and manual QA save.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:9000"


class CheckFailure(RuntimeError):
    """Raised when a functional check fails."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


class HttpClient:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout

    def get_text(self, path: str) -> str:
        status, body, _content_type = self._request("GET", path)
        if not 200 <= status < 300:
            raise CheckFailure(f"GET {path} returned HTTP {status}")
        return body.decode("utf-8", errors="replace")

    def get_json(self, path: str) -> dict[str, Any]:
        return self._json_response("GET", path)

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        return self._json_response("POST", path, body, {"Content-Type": "application/json"})

    def post_multipart(
        self,
        path: str,
        fields: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
    ) -> dict[str, Any]:
        boundary = f"----rag-eval-functional-{uuid.uuid4().hex}"
        body = _build_multipart_body(boundary, fields, files)
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        return self._json_response("POST", path, body, headers)

    def _json_response(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        status, response_body, _content_type = self._request(method, path, body, headers)
        if not 200 <= status < 300:
            message = response_body.decode("utf-8", errors="replace")
            raise CheckFailure(f"{method} {path} returned HTTP {status}: {message}")
        try:
            payload = json.loads(response_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise CheckFailure(f"{method} {path} did not return valid JSON") from exc
        if not isinstance(payload, dict):
            raise CheckFailure(f"{method} {path} returned non-object JSON")
        return payload

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes, str]:
        url = urljoin(self.base_url, path.lstrip("/"))
        request = Request(url, data=body, headers=headers or {}, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.status, response.read(), response.headers.get("Content-Type", "")
        except HTTPError as exc:
            return exc.code, exc.read(), exc.headers.get("Content-Type", "")
        except URLError as exc:
            raise CheckFailure(f"{method} {path} failed: {exc.reason}") from exc


def run_smoke_checks(
    client: Any,
    *,
    include_write_checks: bool = False,
    preview_dataset_path: str | None = None,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    page_html = client.get_text("/datasets")
    _require_console_page(page_html)
    results.append(CheckResult("console page", True, "React shell and assets are present"))

    datasets_payload = client.get_json("/api/datasets")
    datasets = _require_datasets_payload(datasets_payload)
    results.append(CheckResult("datasets api", True, f"{len(datasets)} datasets returned"))

    preview_path = preview_dataset_path or _first_dataset_path(datasets)
    preview_payload = client.get_json(f"/api/datasets/preview?path={quote(preview_path, safe='')}")
    _require_preview_payload(preview_payload)
    results.append(CheckResult("dataset preview", True, f"{preview_payload.get('rows', 0)} rows previewed"))

    reports_payload = client.get_json("/api/reports")
    reports = _require_list_payload(reports_payload, "reports")
    results.append(CheckResult("reports api", True, f"{len(reports)} reports returned"))

    runs_payload = client.get_json("/api/eval-runs")
    runs = _require_list_payload(runs_payload, "runs")
    results.append(CheckResult("eval runs api", True, f"{len(runs)} runs returned"))

    if include_write_checks:
        upload_payload = _check_dataset_upload(client)
        results.append(CheckResult("dataset upload", True, upload_payload["path"]))

        try:
            manual_payload = _check_manual_qa_save(client, str(upload_payload["path"]))
            results.append(CheckResult("manual QA save", True, f"rows={manual_payload.get('rows', '?')}"))
        finally:
            _cleanup_uploaded_smoke_dataset(str(upload_payload["path"]))

    return results


def _require_console_page(page_html: str) -> None:
    if "RAG Evaluation Console" not in page_html:
        raise CheckFailure("console page is missing the RAG Evaluation Console title")
    if not re.search(r"<div\s+[^>]*id=[\"']root[\"']", page_html):
        raise CheckFailure("console page is missing the React root")
    if not re.search(r"<script\s+[^>]*type=[\"']module[\"'][^>]*src=[\"']/assets/[^\"']+\.js[\"']", page_html):
        raise CheckFailure("console page is missing the React JavaScript asset")
    if not re.search(r"<link\s+[^>]*href=[\"']/assets/[^\"']+\.css[\"']", page_html):
        raise CheckFailure("console page is missing the React stylesheet asset")


def _require_datasets_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    _require_ok(payload, "/api/datasets")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise CheckFailure("/api/datasets returned no datasets")
    for dataset in datasets:
        if not isinstance(dataset, dict) or not dataset.get("name") or not dataset.get("path"):
            raise CheckFailure("/api/datasets returned a dataset without name/path")
    if not isinstance(payload.get("run_options"), dict):
        raise CheckFailure("/api/datasets is missing run_options")
    return datasets


def _first_dataset_path(datasets: list[dict[str, Any]]) -> str:
    for dataset in datasets:
        path = dataset.get("path")
        if isinstance(path, str) and path:
            return path
    raise CheckFailure("no previewable dataset path was returned")


def _require_preview_payload(payload: dict[str, Any]) -> None:
    _require_ok(payload, "/api/datasets/preview")
    if not isinstance(payload.get("rows"), int):
        raise CheckFailure("/api/datasets/preview is missing integer rows")
    if not isinstance(payload.get("preview_rows"), list):
        raise CheckFailure("/api/datasets/preview is missing preview_rows")


def _require_list_payload(payload: dict[str, Any], key: str) -> list[Any]:
    _require_ok(payload, f"/api/{key}")
    items = payload.get(key)
    if not isinstance(items, list):
        raise CheckFailure(f"/api/{key} returned non-list {key}")
    return items


def _require_ok(payload: dict[str, Any], endpoint: str) -> None:
    if payload.get("ok") is not True:
        raise CheckFailure(f"{endpoint} returned ok=false: {payload.get('error', 'unknown error')}")


def _check_dataset_upload(client: Any) -> dict[str, Any]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_name = f"functional_smoke_{stamp}"
    csv_content = (
        "query_id,query,expected_answer\n"
        "query_1,Functional smoke question?,Functional smoke answer.\n"
    ).encode("utf-8")
    payload = client.post_multipart(
        "/api/datasets/upload",
        {"name": dataset_name},
        {"file": (f"{dataset_name}.csv", csv_content, "text/csv")},
    )
    _require_ok(payload, "/api/datasets/upload")
    if not payload.get("path") or payload.get("rows") != 1:
        raise CheckFailure("/api/datasets/upload returned an unexpected payload")
    return payload


def _check_manual_qa_save(client: Any, target_dataset: str) -> dict[str, Any]:
    payload = client.post_json(
        "/api/datasets/manual-qa",
        {
            "target_dataset": target_dataset,
            "question": "Functional smoke follow-up question?",
            "expected_answer": "Functional smoke follow-up answer.",
        },
    )
    _require_ok(payload, "/api/datasets/manual-qa")
    if not payload.get("path"):
        raise CheckFailure("/api/datasets/manual-qa returned no dataset path")
    return payload


def _cleanup_uploaded_smoke_dataset(upload_path: str) -> None:
    path = Path(upload_path)
    if path.is_absolute():
        target = path
    else:
        target = Path.cwd() / path
    if target.name.startswith("functional_smoke_") and target.suffix.lower() == ".csv":
        target.unlink(missing_ok=True)


def _build_multipart_body(
    boundary: str,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    for name, (filename, content, content_type) in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run functional smoke checks against the RAG evaluation console.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Console base URL. Default: {DEFAULT_BASE_URL}")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds. Default: 10")
    parser.add_argument(
        "--preview-dataset-path",
        default=None,
        help="Dataset path to preview. Defaults to the first dataset returned by /api/datasets.",
    )
    parser.add_argument(
        "--include-write-checks",
        action="store_true",
        help="Also test dataset upload and manual QA save. This creates a small uploaded dataset.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = HttpClient(args.base_url, timeout=args.timeout)
    try:
        results = run_smoke_checks(
            client,
            include_write_checks=args.include_write_checks,
            preview_dataset_path=args.preview_dataset_path,
        )
    except CheckFailure as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    for result in results:
        print(f"[OK] {result.name}: {result.detail}")
    print(f"Functional smoke checks passed against {args.base_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
