import json
import math
from pathlib import Path

import pytest

from rag_eval_pipeline.local_retrieval import (
    LocalChunkRetriever,
    build_index,
    cosine_similarity,
    load_chunks,
    retrieval_response,
)


class FakeEmbeddingClient:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embeddings(self, texts):
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_load_chunks_reads_jsonl_schema(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        chunks_path,
        [
            {
                "chunk_id": "c1",
                "source": "manual.pdf",
                "title": "清洁维护",
                "page_start": 10,
                "page_end": 11,
                "text": "雷达摄像头需要清洁。",
            }
        ],
    )

    chunks = load_chunks(chunks_path)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c1"
    assert chunks[0].metadata["page_start"] == 10
    assert chunks[0].text == "雷达摄像头需要清洁。"


def test_load_chunks_skips_punctuation_only_rows(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        chunks_path,
        [
            {
                "chunk_id": "divider",
                "source": "manual.pdf",
                "title": "",
                "page_start": 1,
                "page_end": 1,
                "text": "---",
            },
            {
                "chunk_id": "meaningful",
                "source": "manual.pdf",
                "title": "Usage",
                "page_start": 2,
                "page_end": 2,
                "text": "Check vehicle status before use.",
            },
        ],
    )

    chunks = load_chunks(chunks_path)

    assert [chunk.chunk_id for chunk in chunks] == ["meaningful"]


def test_cosine_similarity_orders_related_chunks():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 1], [1, 1]) == pytest.approx(1.0)
    assert math.isfinite(cosine_similarity([0, 0], [1, 0]))


def test_retriever_returns_top_chunks_with_metadata():
    chunks = [
        {
            "chunk_id": "battery",
            "source": "manual.pdf",
            "title": "充电",
            "page_start": 30,
            "page_end": 30,
            "text": "车辆充电前请检查充电口。",
        },
        {
            "chunk_id": "camera",
            "source": "manual.pdf",
            "title": "清洁维护",
            "page_start": 206,
            "page_end": 206,
            "text": "雷达或摄像头窗口表面有污渍时，请使用清水清洗。",
        },
    ]
    client = FakeEmbeddingClient(
        {
            chunks[0]["text"]: [1.0, 0.0],
            chunks[1]["text"]: [0.0, 1.0],
            "摄像头脏污怎么处理？": [0.0, 1.0],
        }
    )
    index = build_index(load_chunks_from_rows(chunks), client, batch_size=2)
    retriever = LocalChunkRetriever(index, client)

    results = retriever.retrieve("摄像头脏污怎么处理？", page_size=1)

    assert len(results) == 1
    assert results[0]["id"] == "camera"
    assert results[0]["content"].startswith("雷达或摄像头")
    assert results[0]["document_id"] == "manual.pdf"
    assert results[0]["metadata"]["title"] == "清洁维护"
    assert results[0]["metadata"]["page_start"] == 206
    assert results[0]["metadata"]["score"] == pytest.approx(1.0)


def test_retrieval_response_matches_ragflow_shape():
    chunks = [
        {
            "id": "camera",
            "content": "雷达或摄像头窗口表面有污渍时，请使用清水清洗。",
            "document_id": "manual.pdf",
            "metadata": {"score": 0.9},
        }
    ]

    payload = retrieval_response(chunks)

    assert payload == {"code": 0, "data": {"chunks": chunks}, "message": "success"}


def load_chunks_from_rows(rows):
    path = Path("unused")
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    temp_path = Path.cwd() / "__local_retrieval_test_chunks.jsonl"
    temp_path.write_text(text, encoding="utf-8")
    try:
        return load_chunks(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
