"""OpenAI-compatible mock server for local RAG eval smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DEFAULT_ANSWER = "SM94 is a solder resist surface dent or depression [1]."
DEFAULT_NUGGETS = [
    "SM94 is a solder resist surface dent or depression",
    "SM94 M1 refers to dents on plastic body or C4 areas",
    "SM94 M3 refers to top-side solder resist undulation",
    "Reject criteria include visible SR dents greater than 40 mils or 1 mm",
]


def extract_user_prompt(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def parse_claims_from_prompt(prompt: str) -> list[str]:
    match = re.search(r"Claims:\s*(\[.*?\])\s*$", prompt, flags=re.DOTALL)
    if not match:
        return DEFAULT_NUGGETS[:2]
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return DEFAULT_NUGGETS[:2]
    return [str(item) for item in parsed if str(item).strip()]


def chat_response(prompt: str) -> str:
    if "Return only a JSON array" in prompt and "expected_answer" in prompt:
        return json.dumps([
            {
                "question": "What is SM94?",
                "expected_answer": "SM94 is a solder resist surface dent or depression.",
            },
            {
                "question": "What is SM94 M3?",
                "expected_answer": "SM94 M3 refers to top-side solder resist undulation.",
            },
        ])
    if "Return only one integer: 0, 1, 2, or 3" in prompt:
        return "3"
    if "Update the list of atomic nuggets" in prompt:
        return json.dumps(DEFAULT_NUGGETS)
    if 'Label each nugget as "vital" or "okay"' in prompt:
        count = prompt.count('", "') + 1 if "Nugget List:" in prompt else len(DEFAULT_NUGGETS)
        count = max(1, min(count, len(DEFAULT_NUGGETS)))
        return json.dumps(["vital"] * count)
    if "For each nugget, label whether it is captured" in prompt:
        count = prompt.count('", "') + 1 if "Nugget List:" in prompt else len(DEFAULT_NUGGETS)
        count = max(1, min(count, len(DEFAULT_NUGGETS)))
        return json.dumps(["support"] * count)
    if "Evaluate whether the statement is supported by the citation" in prompt:
        return json.dumps({"support": "full_support"})
    if "Determine whether the answer is an attempt to answer the query" in prompt:
        return json.dumps({"answered": "yes"})
    if "Score how factually supported the generated answer is by the source passages" in prompt:
        return json.dumps({"score": 1.0})
    if "Extract all atomic factual claims from the text" in prompt:
        return json.dumps({"claims": DEFAULT_NUGGETS[:3]})
    if "For each claim, determine whether it is" in prompt:
        claims = parse_claims_from_prompt(prompt)
        return json.dumps({
            "verdicts": [
                {"claim": claim, "verdict": "entailment"}
                for claim in claims
            ]
        })
    return DEFAULT_ANSWER


def embedding_for_text(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [byte / 255.0 for byte in digest[:8]]
    return values or [1.0]


class MockOpenAIHandler(BaseHTTPRequestHandler):
    server_version = "RagEvalMockOpenAI/0.1"

    def do_GET(self) -> None:  # noqa: N802
        self.write_json({"service": "rag-eval mock OpenAI", "status": "ok"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
            payload = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError as exc:
            self.write_json({"error": f"invalid JSON: {exc}"}, status=400)
            return

        path = self.path.rstrip("/")
        if path.endswith("/chat/completions"):
            content = chat_response(extract_user_prompt(payload))
            self.write_json({
                "id": "mock-chatcmpl",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 32,
                    "completion_tokens": max(1, len(content.split())),
                    "total_tokens": 32 + max(1, len(content.split())),
                },
            })
            return

        if path.endswith("/embeddings"):
            inputs = payload.get("input") or []
            if isinstance(inputs, str):
                inputs = [inputs]
            self.write_json({
                "object": "list",
                "data": [
                    {"object": "embedding", "index": idx, "embedding": embedding_for_text(str(text))}
                    for idx, text in enumerate(inputs)
                ],
                "model": payload.get("model") or "mock-embedding",
            })
            return

        self.write_json({"error": "not found"}, status=404)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), MockOpenAIHandler)
    print(f"Mock OpenAI listening on http://{args.host}:{args.port}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
