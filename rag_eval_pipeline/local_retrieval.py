"""Local chunk retrieval backed by OpenAI-compatible embeddings.

This module turns a JSONL chunk file into a small in-process vector index and
serves results in the RAGFlow-like shape already consumed by the generation
pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib import request

from rag_eval_pipeline.config import config_value


LOGGER = logging.getLogger("rag_eval.local_retrieval")
INDEX_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source: str
    title: str
    page_start: int
    page_end: int
    text: str

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: ChunkRecord
    vector: list[float]


class OpenAIEmbeddingClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: int = 60,
        retries: int = 1,
    ) -> None:
        self.embedding_url = embeddings_url(base_url)
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            req = request.Request(self.embedding_url, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout) as resp:
                    response_text = resp.read().decode("utf-8")
                    data = json.loads(response_text)
                    return [item["embedding"] for item in data["data"]]
            except Exception as exc:  # noqa: BLE001 - surface API errors with context.
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Embedding request failed: {last_error}")


class LocalChunkRetriever:
    def __init__(self, index: list[EmbeddedChunk], embedding_client: Any) -> None:
        self.index = index
        self.embedding_client = embedding_client

    def retrieve(
        self,
        question: str,
        *,
        page_size: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        query = question.strip()
        if not query:
            return []

        query_vector = self.embedding_client.embeddings([query])[0]
        scored = [
            (cosine_similarity(query_vector, item.vector), item.chunk)
            for item in self.index
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        results = []
        for score, chunk in scored:
            if score < similarity_threshold:
                continue
            metadata = chunk.metadata | {"score": score}
            results.append(
                {
                    "id": chunk.chunk_id,
                    "content": chunk.text,
                    "document_id": chunk.source,
                    "metadata": metadata,
                }
            )
            if len(results) >= max(page_size, 1):
                break
        return results


def embeddings_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/embeddings"):
        return base_url
    return f"{base_url}/embeddings"


def load_chunks(path: str | Path) -> list[ChunkRecord]:
    chunk_path = Path(path)
    chunks: list[ChunkRecord] = []
    with chunk_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            if not is_meaningful_chunk_text(text):
                LOGGER.debug("Skipping non-meaningful chunk text at %s:%s", chunk_path, line_number)
                continue
            chunks.append(
                ChunkRecord(
                    chunk_id=str(row["chunk_id"]),
                    source=str(row.get("source") or ""),
                    title=str(row.get("title") or ""),
                    page_start=int(row.get("page_start") or 0),
                    page_end=int(row.get("page_end") or row.get("page_start") or 0),
                    text=text,
                )
            )
    if not chunks:
        raise ValueError(f"No chunks loaded from {chunk_path}")
    return chunks


def is_meaningful_chunk_text(text: str) -> bool:
    return sum(1 for character in text if character.isalnum()) >= 2


def build_index(
    chunks: Iterable[ChunkRecord],
    embedding_client: Any,
    *,
    batch_size: int = 16,
) -> list[EmbeddedChunk]:
    chunk_list = list(chunks)
    index: list[EmbeddedChunk] = []
    for start in range(0, len(chunk_list), max(batch_size, 1)):
        batch = chunk_list[start : start + max(batch_size, 1)]
        vectors = embedding_client.embeddings([chunk.text for chunk in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding API returned a different number of vectors than inputs")
        index.extend(EmbeddedChunk(chunk=chunk, vector=list(vector)) for chunk, vector in zip(batch, vectors))
        LOGGER.info("Embedded chunks %s/%s", min(start + len(batch), len(chunk_list)), len(chunk_list))
    return index


def save_index(path: str | Path, index: list[EmbeddedChunk], *, model: str) -> None:
    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as handle:
        manifest = {
            "type": "manifest",
            "model": model,
            "schema_version": INDEX_SCHEMA_VERSION,
        }
        handle.write(json.dumps(manifest, ensure_ascii=False) + "\n")
        for item in index:
            row = {
                "chunk_id": item.chunk.chunk_id,
                "source": item.chunk.source,
                "title": item.chunk.title,
                "page_start": item.chunk.page_start,
                "page_end": item.chunk.page_end,
                "text": item.chunk.text,
                "vector": item.vector,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_index(path: str | Path, *, expected_model: str | None = None) -> list[EmbeddedChunk]:
    index_path = Path(path)
    rows = index_path.read_text(encoding="utf-8").splitlines()
    if not rows:
        raise ValueError(f"Empty index cache: {index_path}")

    start_index = 0
    first = json.loads(rows[0])
    if first.get("type") == "manifest":
        start_index = 1
        model = first.get("model")
        if expected_model and model != expected_model:
            raise ValueError(f"Index cache model mismatch: expected {expected_model}, got {model}")
        schema_version = int(first.get("schema_version") or 1)
        if schema_version != INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"Index cache schema mismatch: expected {INDEX_SCHEMA_VERSION}, got {schema_version}"
            )

    index: list[EmbeddedChunk] = []
    for row_text in rows[start_index:]:
        if not row_text.strip():
            continue
        row = json.loads(row_text)
        chunk = ChunkRecord(
            chunk_id=str(row["chunk_id"]),
            source=str(row.get("source") or ""),
            title=str(row.get("title") or ""),
            page_start=int(row.get("page_start") or 0),
            page_end=int(row.get("page_end") or row.get("page_start") or 0),
            text=str(row.get("text") or ""),
        )
        index.append(EmbeddedChunk(chunk=chunk, vector=[float(value) for value in row["vector"]]))
    if not index:
        raise ValueError(f"No vectors loaded from {index_path}")
    return index


def load_or_build_index(
    *,
    chunks_path: str | Path,
    cache_path: str | Path,
    embedding_client: Any,
    embedding_model: str,
    rebuild: bool = False,
    batch_size: int = 16,
) -> list[EmbeddedChunk]:
    cache = Path(cache_path)
    if cache.exists() and not rebuild:
        try:
            LOGGER.info("Loading local retrieval index cache: %s", cache)
            return load_index(cache, expected_model=embedding_model)
        except Exception as exc:  # noqa: BLE001 - corrupted caches should rebuild.
            LOGGER.warning("Ignoring unusable index cache %s: %s", cache, exc)

    chunks = load_chunks(chunks_path)
    LOGGER.info("Building local retrieval index from %s chunks", len(chunks))
    index = build_index(chunks, embedding_client, batch_size=batch_size)
    save_index(cache, index, model=embedding_model)
    return index


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def retrieval_response(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"code": 0, "data": {"chunks": chunks}, "message": "success"}


def local_retrieval_settings(config: dict[str, Any], root: Path) -> dict[str, Any]:
    section = config.get("local_retrieval") or {}
    evaluation = config.get("evaluation") or {}
    chunks_path = Path(str(section.get("chunks_path") or root.parent / "rag_chunk" / "chunks" / "chunks.jsonl"))
    cache_path = Path(str(section.get("cache_path") or root / "data" / "local_chunks.index.jsonl"))
    return {
        "chunks_path": chunks_path if chunks_path.is_absolute() else root / chunks_path,
        "cache_path": cache_path if cache_path.is_absolute() else root / cache_path,
        "embedding_base_url": config_value(evaluation, "embedding_base_url"),
        "embedding_api_key": config_value(evaluation, "embedding_api_key"),
        "embedding_model": config_value(evaluation, "embedding_model"),
        "timeout": int(section.get("timeout") or evaluation.get("timeout") or 60),
        "retries": int(section.get("retries") or evaluation.get("retries") or 1),
        "batch_size": int(section.get("batch_size") or 16),
        "rebuild": bool(section.get("rebuild") or False),
    }
