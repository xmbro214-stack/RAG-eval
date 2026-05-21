"""Generate Open RAG Eval answer CSVs from an HTTP retrieval API.

The script reads a query CSV, calls a retrieval endpoint for each query, asks an
OpenAI-compatible chat model to answer from the retrieved passages, and writes
the CSV shape expected by Open RAG Eval:

    query_id,query,query_run,passage_id,passage,generated_answer

Default retrieval payloads are compatible with RAGFlow
``POST /api/v1/retrieval`` endpoints:

    {
        "dataset_ids": ["dataset_id"],
        "question": query,
        "document_ids": [],
        "page": 1,
        "page_size": 30,
        "similarity_threshold": 0.2,
        "vector_similarity_weight": 0.3,
        "top_k": 1024,
        "keyword": false,
        "highlight": false,
        "cross_languages": [],
        "use_kg": false,
        "toc_enhance": false
    }

Use ``--retrieval-payload-json`` when your API has different field names.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


FIELDNAMES = [
    "query_id",
    "query",
    "query_run",
    "passage_id",
    "passage",
    "generated_answer",
]

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful RAG assistant. Answer the user question using only the "
    "provided retrieved passages. Cite supporting passages with bracketed "
    "numbers like [1]. If the passages do not contain enough information, say "
    "that there is not enough information to answer."
)

DEFAULT_PROMPT_TEMPLATE = """Question:
{query}

Retrieved passages:
{context}

Write a concise answer grounded in the retrieved passages. Include citations in
the form [1], [2] for the passages that support each claim.
"""


@dataclass(frozen=True)
class Query:
    query_id: str
    query: str


@dataclass(frozen=True)
class Passage:
    passage_id: str
    text: str
    raw_id: str = ""


class HTTPJSONClient:
    def __init__(self, timeout: int, retries: int) -> None:
        self.timeout = timeout
        self.retries = retries

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any] | list[Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            req = request.Request(url, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout) as resp:
                    text = resp.read().decode("utf-8")
                    return json.loads(text) if text else {}
            except (error.HTTPError, error.URLError, TimeoutError) as exc:
                last_error = describe_http_error(exc)
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Request failed for {url}: {last_error}")


class OpenAICompatibleChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int,
        retries: int,
        temperature: float,
        max_tokens: int | None,
    ) -> None:
        self.chat_url = chat_completions_url(base_url)
        self.api_key = api_key
        self.model = model
        self.http = HTTPJSONClient(timeout=timeout, retries=retries)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        data = self.http.post_json(
            self.chat_url,
            payload,
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        if not isinstance(data, dict):
            raise ValueError(f"Expected chat JSON object, got {type(data).__name__}")
        return data["choices"][0]["message"]["content"].strip()


def chat_completions_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def describe_http_error(exc: Exception) -> str:
    if isinstance(exc, error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        if body:
            return f"HTTPError {exc.code} {exc.reason}: {body}"
        return f"HTTPError {exc.code} {exc.reason}"
    return str(exc)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def read_queries(path: str) -> list[Query]:
    queries: list[Query] = []
    with open(path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for idx, row in enumerate(reader, start=1):
            query_text = (row.get("query") or "").strip()
            if not query_text:
                log(f"Skipping row {idx}: missing query")
                continue
            query_id = (row.get("query_id") or "").strip() or f"query_{idx}"
            queries.append(Query(query_id=query_id, query=query_text))
    return queries


def write_rows(path: str, rows: list[dict[str, str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json_arg(value: str | None, arg_name: str) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON: {exc}") from exc


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def build_template_replacements(args: argparse.Namespace, query: Query) -> dict[str, Any]:
    return {
        "{query}": query.query,
        "{query_id}": query.query_id,
        "{page}": args.page,
        "{page_size}": args.page_size,
        "{similarity_threshold}": args.similarity_threshold,
        "{vector_similarity_weight}": args.vector_similarity_weight,
        "{top_k}": args.top_k,
    }


def apply_payload_template(value: Any, replacements: dict[str, Any]) -> Any:
    if isinstance(value, str):
        if value in replacements:
            return replacements[value]
        result = value
        for placeholder, replacement in replacements.items():
            result = result.replace(placeholder, str(replacement))
        return result
    if isinstance(value, list):
        return [apply_payload_template(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            str(apply_payload_template(key, replacements)): apply_payload_template(item, replacements)
            for key, item in value.items()
        }
    return value


def build_retrieval_payload(args: argparse.Namespace, query: Query) -> dict[str, Any]:
    template = load_json_arg(args.retrieval_payload_json, "--retrieval-payload-json")
    if template is not None:
        payload = apply_payload_template(template, build_template_replacements(args, query))
        if not isinstance(payload, dict):
            raise ValueError("--retrieval-payload-json must render to a JSON object")
        return payload

    dataset_ids = parse_csv_list(args.dataset_ids)
    if not dataset_ids:
        raise ValueError("--dataset-ids is required for the default RAGFlow retrieval payload")

    document_ids = parse_csv_list(args.document_ids)
    cross_languages = parse_csv_list(args.cross_languages)
    metadata_condition = load_json_arg(args.metadata_condition_json, "--metadata-condition-json")

    payload: dict[str, Any] = {
        "dataset_ids": dataset_ids,
        "question": query.query,
        "document_ids": document_ids,
        "page": args.page,
        "page_size": args.page_size,
        "similarity_threshold": args.similarity_threshold,
        "vector_similarity_weight": args.vector_similarity_weight,
        "top_k": args.top_k,
        "keyword": args.keyword,
        "highlight": args.highlight,
        "cross_languages": cross_languages,
        "use_kg": args.use_kg,
        "toc_enhance": args.toc_enhance,
    }
    if args.rerank_id:
        payload["rerank_id"] = args.rerank_id
    if metadata_condition is not None:
        if not isinstance(metadata_condition, dict):
            raise ValueError("--metadata-condition-json must be a JSON object")
        payload["metadata_condition"] = metadata_condition

    extra_body = load_json_arg(args.retrieval_extra_body_json, "--retrieval-extra-body-json")
    if extra_body:
        if not isinstance(extra_body, dict):
            raise ValueError("--retrieval-extra-body-json must be a JSON object")
        payload.update(extra_body)
    return payload


def build_retrieval_headers(args: argparse.Namespace) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    extra_headers = load_json_arg(args.retrieval_headers_json, "--retrieval-headers-json")
    if extra_headers:
        if not isinstance(extra_headers, dict):
            raise ValueError("--retrieval-headers-json must be a JSON object")
        headers.update({str(key): str(value) for key, value in extra_headers.items()})

    api_key = args.retrieval_api_key
    if api_key:
        if args.retrieval_auth_scheme:
            headers[args.retrieval_auth_header] = f"{args.retrieval_auth_scheme} {api_key}"
        else:
            headers[args.retrieval_auth_header] = api_key
    return headers


def find_first_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        return []

    preferred_keys = (
        "chunks",
        "search_results",
        "retrieval_results",
        "results",
        "documents",
        "passages",
        "retrieved_passages",
    )
    for key in preferred_keys:
        item = value.get(key)
        if isinstance(item, list):
            return item

    data = value.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return find_first_list(data)

    for item in value.values():
        found = find_first_list(item)
        if found:
            return found
    return []


def normalize_retrieval_response(response_json: dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
    if isinstance(response_json, dict) and "code" in response_json:
        code = response_json.get("code")
        if code != 0:
            message = response_json.get("message", "")
            raise RuntimeError(f"Retrieval API returned code={code}: {message}")
        data = response_json.get("data")
        if isinstance(data, (dict, list)):
            return data
    return response_json


def text_from_result(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    if not isinstance(result, dict):
        return ""

    text_keys = (
        "content",
        "text",
        "passage",
        "page_content",
        "chunk_content",
        "document",
        "body",
    )
    for key in text_keys:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for nested_key in ("doc", "document", "metadata"):
        nested = result.get(nested_key)
        if isinstance(nested, dict):
            text = text_from_result(nested)
            if text:
                return text
    return ""


def raw_id_from_result(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    for key in ("id", "chunk_id", "document_id", "doc_id", "passage_id"):
        value = result.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def extract_passages(response_json: dict[str, Any] | list[Any], max_passages: int) -> list[Passage]:
    retrieval_data = normalize_retrieval_response(response_json)
    results = find_first_list(retrieval_data)
    passages: list[Passage] = []
    for idx, result in enumerate(results, start=1):
        text = text_from_result(result)
        if not text:
            continue
        passages.append(
            Passage(
                passage_id=f"[{len(passages) + 1}]",
                text=text,
                raw_id=raw_id_from_result(result),
            )
        )
        if len(passages) >= max_passages:
            break
    return passages


def format_context(passages: list[Passage]) -> str:
    if not passages:
        return "No retrieved passages."
    return "\n\n".join(f"{passage.passage_id} {passage.text}" for passage in passages)


def render_prompt(template: str, query: Query, passages: list[Passage]) -> str:
    return template.format(query=query.query, query_id=query.query_id, context=format_context(passages))


def rows_for_result(
    query: Query,
    run_idx: int,
    passages: list[Passage],
    generated_answer: str,
) -> list[dict[str, str]]:
    if not passages:
        return [{
            "query_id": query.query_id,
            "query": query.query,
            "query_run": str(run_idx),
            "passage_id": "NA",
            "passage": "NA",
            "generated_answer": generated_answer,
        }]

    rows = []
    for idx, passage in enumerate(passages, start=1):
        rows.append({
            "query_id": query.query_id,
            "query": query.query,
            "query_run": str(run_idx),
            "passage_id": passage.passage_id,
            "passage": passage.text,
            "generated_answer": generated_answer if idx == 1 else "",
        })
    return rows


def process_query_run(
    query: Query,
    run_idx: int,
    args: argparse.Namespace,
    retrieval_http: HTTPJSONClient,
    llm_client: OpenAICompatibleChatClient,
    retrieval_headers: dict[str, str],
    prompt_template: str,
) -> list[dict[str, str]]:
    try:
        payload = build_retrieval_payload(args, query)
        retrieval_json = retrieval_http.post_json(
            args.retrieval_url,
            payload,
            retrieval_headers,
        )
        max_passages = args.max_passages or args.page_size
        passages = extract_passages(retrieval_json, max_passages)
        prompt = render_prompt(prompt_template, query, passages)
        generated_answer = llm_client.chat(args.system_prompt, prompt)
        return rows_for_result(query, run_idx, passages, generated_answer)
    except Exception as exc:
        log(f"Error processing {query.query_id} run {run_idx}: {exc}")
        return [{
            "query_id": query.query_id,
            "query": query.query,
            "query_run": str(run_idx),
            "passage_id": "ERROR",
            "passage": f"Runtime error: {exc}",
            "generated_answer": "API_ERROR",
        }]


def load_prompt_template(path: str | None) -> str:
    if not path:
        return DEFAULT_PROMPT_TEMPLATE
    return Path(path).read_text(encoding="utf-8")


def generate(args: argparse.Namespace) -> None:
    started_at = time.perf_counter()
    queries = read_queries(args.queries_csv)
    if not queries:
        raise ValueError(f"No queries found in {args.queries_csv}")

    if not args.llm_api_key:
        raise ValueError("Missing LLM API key. Pass --llm-api-key or set OPENAI_API_KEY.")

    retrieval_http = HTTPJSONClient(timeout=args.timeout, retries=args.retries)
    llm_client = OpenAICompatibleChatClient(
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        model=args.llm_model,
        timeout=args.timeout,
        retries=args.retries,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    retrieval_headers = build_retrieval_headers(args)
    prompt_template = load_prompt_template(args.prompt_template_file)

    repeated = [
        (query_idx, query, run_idx)
        for query_idx, query in enumerate(queries)
        for run_idx in range(1, args.repeat_query + 1)
    ]
    log(
        f"Generating answers: queries={len(queries)}, runs={len(repeated)}, "
        f"page_size={args.page_size}, top_k={args.top_k}, max_workers={args.max_workers}"
    )

    indexed_rows: list[tuple[tuple[int, int], dict[str, str]]] = []
    if args.max_workers <= 1:
        for completed, (query_idx, query, run_idx) in enumerate(repeated, start=1):
            log(f"[{completed}/{len(repeated)}] {query.query_id} run {run_idx}")
            rows = process_query_run(
                query,
                run_idx,
                args,
                retrieval_http,
                llm_client,
                retrieval_headers,
                prompt_template,
            )
            for row in rows:
                indexed_rows.append(((query_idx, run_idx), row))
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_index = {
                executor.submit(
                    process_query_run,
                    query,
                    run_idx,
                    args,
                    retrieval_http,
                    llm_client,
                    retrieval_headers,
                    prompt_template,
                ): (query_idx, run_idx)
                for query_idx, query, run_idx in repeated
            }
            completed = 0
            for future in as_completed(future_to_index):
                completed += 1
                query_idx, run_idx = future_to_index[future]
                log(f"[{completed}/{len(repeated)}] completed query_index={query_idx} run={run_idx}")
                for row in future.result():
                    indexed_rows.append(((query_idx, run_idx), row))

    indexed_rows.sort(key=lambda item: (item[0][0], item[0][1], row_order(item[1])))
    rows = [row for _, row in indexed_rows]
    write_rows(args.output_csv, rows)
    log(f"Wrote {len(rows)} rows to {args.output_csv} in {time.perf_counter() - started_at:.2f}s")


def row_order(row: dict[str, str]) -> int:
    passage_id = row.get("passage_id", "")
    if passage_id.startswith("[") and passage_id.endswith("]"):
        try:
            return int(passage_id[1:-1])
        except ValueError:
            return 0
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Open RAG Eval answer CSVs via HTTP retrieval + OpenAI-compatible LLM."
    )
    parser.add_argument("--queries-csv", required=True, help="Input CSV with query and optional query_id columns.")
    parser.add_argument("--output-csv", required=True, help="Output generated answer CSV path.")

    parser.add_argument("--retrieval-url", required=True, help="HTTP retrieval endpoint, e.g. http://ip:port/api/v1/retrieval.")
    parser.add_argument("--retrieval-api-key", default=os.getenv("RETRIEVAL_API_KEY"), help="Optional retrieval API key.")
    parser.add_argument("--retrieval-auth-header", default="Authorization", help="Header used for retrieval API auth.")
    parser.add_argument("--retrieval-auth-scheme", default="Bearer", help="Auth scheme prefix. Use '' for raw key.")
    parser.add_argument("--retrieval-headers-json", help="Extra retrieval headers as a JSON object.")
    parser.add_argument(
        "--retrieval-payload-json",
        help=(
            "Full retrieval payload template as JSON. Supports placeholders "
            "'{query}', '{query_id}', '{page}', '{page_size}', "
            "'{similarity_threshold}', '{vector_similarity_weight}', '{top_k}'."
        ),
    )
    parser.add_argument("--retrieval-extra-body-json", help="Extra fields merged into the default retrieval payload.")
    parser.add_argument("--dataset-ids", help="Comma-separated RAGFlow dataset/knowledge-base IDs. Required unless --retrieval-payload-json is used.")
    parser.add_argument("--document-ids", help="Comma-separated document IDs to restrict retrieval.")
    parser.add_argument("--page", type=int, default=1, help="RAGFlow retrieval page number.")
    parser.add_argument("--page-size", type=int, default=30, help="RAGFlow chunks returned per page.")
    parser.add_argument("--similarity-threshold", type=float, default=0.2, help="RAGFlow similarity threshold.")
    parser.add_argument("--vector-similarity-weight", type=float, default=0.3, help="RAGFlow vector similarity weight.")
    parser.add_argument("--top-k", type=int, default=1024, help="RAGFlow retrieval candidate limit.")
    parser.add_argument("--rerank-id", help="Optional RAGFlow rerank model ID.")
    parser.add_argument("--keyword", type=parse_bool, default=False, help="Whether RAGFlow should use LLM keyword expansion.")
    parser.add_argument("--highlight", type=parse_bool, default=False, help="Whether RAGFlow should return highlighted content.")
    parser.add_argument("--cross-languages", help="Comma-separated cross-language translation targets, e.g. English,Chinese.")
    parser.add_argument("--metadata-condition-json", help="RAGFlow metadata_condition as a JSON object. Ignored by RAGFlow when document_ids are supplied.")
    parser.add_argument("--use-kg", type=parse_bool, default=False, help="Whether RAGFlow should also use KG retrieval.")
    parser.add_argument("--toc-enhance", type=parse_bool, default=False, help="Whether RAGFlow should enhance retrieval with TOC.")
    parser.add_argument("--max-passages", type=int, help="Maximum retrieved passages to write and send to the LLM. Defaults to --page-size.")

    parser.add_argument("--llm-base-url", required=True, help="OpenAI-compatible base URL, e.g. http://ip:port/v1.")
    parser.add_argument("--llm-api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI-compatible API key.")
    parser.add_argument("--llm-model", required=True, help="Chat model name.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--prompt-template-file", help="Optional prompt template file with {query}, {query_id}, {context}.")

    parser.add_argument("--repeat-query", type=int, default=1, help="How many times to run each query.")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel workers. Use 1 for sequential runs.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per HTTP request.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat_query < 1:
        raise ValueError("--repeat-query must be >= 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.page < 1:
        raise ValueError("--page must be >= 1")
    if args.page_size < 1:
        raise ValueError("--page-size must be >= 1")
    if args.max_passages is not None and args.max_passages < 1:
        raise ValueError("--max-passages must be >= 1")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    generate(args)


if __name__ == "__main__":
    main()
