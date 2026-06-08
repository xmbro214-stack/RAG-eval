"""Small local retrieval server for smoke-testing the RAG eval pipeline.

It serves a RAGFlow-like ``POST /api/v1/retrieval`` endpoint using rows from a
golden-answer CSV. This is intended for local pipeline verification when a real
RAGFlow service is not available.
"""

from __future__ import annotations

import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_CSV = ROOT / "data" / "qa_golden.csv"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def choose_rows(rows: list[dict[str, str]], question: str, page_size: int) -> list[dict[str, str]]:
    question_lc = question.lower()
    matches = [
        row for row in rows
        if row.get("query", "").lower() in question_lc
        or question_lc in row.get("query", "").lower()
    ]
    selected = matches or rows
    return selected[:max(page_size, 1)]


def make_handler(rows: list[dict[str, str]]) -> type[BaseHTTPRequestHandler]:
    class MockRetrievalHandler(BaseHTTPRequestHandler):
        server_version = "RagEvalMockRetrieval/0.1"

        def do_GET(self) -> None:  # noqa: N802
            self.write_json({
                "service": "rag-eval mock retrieval",
                "status": "ok",
                "endpoint": "POST /api/v1/retrieval",
                "rows": len(rows),
            })

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/api/v1/retrieval":
                self.write_json({"code": 404, "message": "not found"}, status=404)
                return

            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
                payload = json.loads(body.decode("utf-8")) if body else {}
            except json.JSONDecodeError as exc:
                self.write_json({"code": 400, "message": f"invalid JSON: {exc}"}, status=400)
                return

            question = str(payload.get("question") or payload.get("query") or "")
            page_size = int(payload.get("page_size") or 3)
            selected = choose_rows(rows, question, page_size)
            chunks = [
                {
                    "id": f"mock-{row.get('query_id') or idx}",
                    "content": row.get("expected_answer") or row.get("query") or "",
                    "document_id": "mock-golden-csv",
                    "query_id": row.get("query_id", ""),
                }
                for idx, row in enumerate(selected, start=1)
            ]
            self.write_json({"code": 0, "data": {"chunks": chunks}, "message": "success"})

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

        def write_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return MockRetrievalHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9380)
    parser.add_argument("--golden-csv", type=Path, default=DEFAULT_GOLDEN_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.golden_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.golden_csv}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(rows))
    print(f"Mock retrieval listening on http://{args.host}:{args.port}/api/v1/retrieval")
    server.serve_forever()


if __name__ == "__main__":
    main()
