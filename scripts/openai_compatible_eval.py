"""Lightweight OpenAI-compatible evaluator for local RAG result CSVs.

This script intentionally avoids importing open_rag_eval so it can run with
only the Python standard library. It reproduces the main Open RAG Eval score
families against precomputed RAG outputs:

- UMBRELA-style retrieval scores and precision/AP/MRR
- AutoNugget-style answer coverage scores
- Citation support scores
- No-answer detection
- Golden-answer semantic similarity via an OpenAI-compatible embedding endpoint
- Golden-answer factual correctness via an OpenAI-compatible chat endpoint

The original project's hallucination score uses HHEM/Vectara factuality
backends. This script provides an LLM-judge factuality score in the same output
column so downstream CSV analysis stays simple.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from urllib import error, request


NO_INFO_NUGGET = "Not enough information, no answer found"


@dataclass
class GeneratedAnswerPart:
    text: str
    citations: list[str]


@dataclass
class RAGRun:
    query_id: str
    query: str
    query_run: str
    retrieved_passages: dict[str, str]
    generated_answer_raw: str
    generated_answer_parts: list[GeneratedAnswerPart]


class OpenAICompatibleClient:
    def __init__(
        self,
        chat_base_url: str,
        chat_api_key: str,
        chat_model: str,
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_model: str | None = None,
        timeout: int = 120,
        retries: int = 2,
    ) -> None:
        self.chat_url = chat_base_url.rstrip("/") + "/chat/completions"
        self.chat_api_key = chat_api_key
        self.chat_model = chat_model
        self.embedding_url = (
            embedding_base_url.rstrip("/") + "/embeddings"
            if embedding_base_url else None
        )
        self.embedding_api_key = embedding_api_key or chat_api_key
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.retries = retries
        self.input_tokens = 0
        self.output_tokens = 0

    def chat(self, prompt: str, max_tokens: int | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise evaluation judge. Follow the requested output format exactly.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        data = self._post_json(self.chat_url, self.chat_api_key, payload)
        usage = data.get("usage") or {}
        self.input_tokens += int(usage.get("prompt_tokens") or 0)
        self.output_tokens += int(usage.get("completion_tokens") or 0)
        return data["choices"][0]["message"]["content"].strip()

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        if not self.embedding_url or not self.embedding_model:
            raise ValueError("Embedding endpoint/model is required for semantic similarity.")
        payload = {"model": self.embedding_model, "input": texts}
        data = self._post_json(self.embedding_url, self.embedding_api_key, payload)
        return [item["embedding"] for item in data["data"]]

    def _post_json(self, url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            req = request.Request(url, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (error.HTTPError, error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Request failed for {url}: {last_error}")


def normalize_citations(text: str) -> str:
    return re.sub(r"\[ID:(\d+)\]", r"[\1]", text, flags=re.IGNORECASE)


def parse_generated_answer(text: str) -> list[GeneratedAnswerPart]:
    text = normalize_citations(text)

    def expand_multi(match: re.Match[str]) -> str:
        return "".join(f"[{num}]" for num in re.findall(r"\d+", match.group()))

    text = re.sub(r"\[\d+(?:,\s*\d+)+\]", expand_multi, text)
    text = re.sub(r"\]\s*,\s*\[", "][", text)
    citation_blocks = list(re.finditer(r"(?:\[\d+\])+", text))
    if not citation_blocks:
        return [GeneratedAnswerPart(text=text.strip(), citations=[])]

    parts: list[GeneratedAnswerPart] = []
    for idx, block in enumerate(citation_blocks):
        text_start = 0 if idx == 0 else citation_blocks[idx - 1].end()
        text_part = text[text_start:block.start()].strip()
        citations = re.findall(r"\[\d+\]", block.group())
        if text_part:
            parts.append(GeneratedAnswerPart(text=text_part, citations=citations))

    tail = text[citation_blocks[-1].end():].strip()
    if len(tail) > 1:
        parts.append(GeneratedAnswerPart(text=tail, citations=[]))
    return parts


def generated_text(parts: list[GeneratedAnswerPart]) -> str:
    return " ".join(part.text for part in parts).strip()


def load_rag_runs(path: str) -> list[RAGRun]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            qid = row.get("query_id", "").strip()
            run_id = row.get("query_run", "1").strip() or "1"
            grouped[(qid, run_id)].append(row)

    runs: list[RAGRun] = []
    for (qid, run_id), rows in grouped.items():
        query = rows[0]["query"]
        passages = {
            row["passage_id"]: row["passage"]
            for row in rows
            if row.get("passage_id") and row.get("passage")
        }
        answer = next(
            (row.get("generated_answer", "").strip() for row in rows
             if row.get("generated_answer", "").strip()),
            "",
        )
        if not answer:
            continue
        runs.append(RAGRun(qid, query, run_id, passages, answer, parse_generated_answer(answer)))
    return runs


def load_golden(path: str) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {row["query_id"]: row for row in csv.DictReader(f)}


def parse_json_obj(text: str) -> dict[str, Any]:
    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group())
    raise ValueError(f"Expected JSON object, got: {text[:300]}")


def parse_list(text: str) -> list[str]:
    text = strip_code_fences(text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Expected list, got: {text[:300]}")
        value = ast.literal_eval(match.group())
    if not isinstance(value, list):
        raise ValueError(f"Expected list, got: {type(value).__name__}")
    return [str(item).strip() for item in value if str(item).strip()]


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|python)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_umbrela(client: OpenAICompatibleClient, run: RAGRun, k_values: list[int]) -> dict[str, Any]:
    scores: dict[str, int] = {}
    for pid, passage in run.retrieved_passages.items():
        prompt = f"""
Given a query and a passage, provide a score on an integer scale of 0 to 3:
0 = passage has nothing to do with the query
1 = passage is related but does not answer it
2 = passage contains some answer, but unclear or mixed with extra information
3 = passage is dedicated to the query and contains the exact answer

Return only one integer: 0, 1, 2, or 3.

<query>
{run.query}
</query>
<passage>
{passage}
</passage>
"""
        raw = client.chat(prompt, max_tokens=8)
        match = re.search(r"[0-3]", raw)
        if not match:
            raise ValueError(f"Could not parse UMBRELA score from: {raw}")
        scores[pid] = int(match.group())

    binary = [1 if score >= 2 else 0 for score in scores.values()]
    retrieval_scores: dict[str, Any] = {"precision@": {}, "AP@": {}}
    for k in k_values:
        if k > len(binary):
            continue
        rel_at_k = sum(binary[:k])
        retrieval_scores["precision@"][str(k)] = rel_at_k / k
        retrieval_scores["AP@"][str(k)] = average_precision(binary[:k], rel_at_k)
    retrieval_scores["MRR"] = mrr(binary)
    return {
        "umbrela_scores": scores,
        "retrieval_scores": retrieval_scores,
        "mean_umbrela_score": sum(scores.values()) / len(scores) if scores else 0.0,
    }


def average_precision(binary: list[int], total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    precisions = []
    relevant_so_far = 0
    for idx, is_relevant in enumerate(binary, start=1):
        if is_relevant:
            relevant_so_far += 1
            precisions.append(relevant_so_far / idx)
    return sum(precisions) / len(precisions) if precisions else 0.0


def mrr(binary: list[int]) -> float:
    for idx, is_relevant in enumerate(binary, start=1):
        if is_relevant:
            return 1.0 / idx
    return 0.0


def compute_autonugget(client: OpenAICompatibleClient, run: RAGRun, umbrela_scores: dict[str, int]) -> dict[str, Any]:
    filtered = [
        passage
        for pid, passage in run.retrieved_passages.items()
        if umbrela_scores.get(pid, 0) >= 1
    ]
    context = "\n".join(f"[{idx + 1}] {text}" for idx, text in enumerate(filtered))
    nuggets: list[str] = []
    for _ in range(5):
        prompt = f"""
Update the list of atomic nuggets of information so they best provide all information required for the query.
Use only the provided context and the existing nugget list.
Return JSON array only, with at most 30 short strings. If context lacks relevant information, return ["{NO_INFO_NUGGET}"].

Search Query: {run.query}

Context:
{context}

Initial Nugget List: {json.dumps(nuggets, ensure_ascii=False)}
"""
        nuggets = parse_list(client.chat(prompt))
        if len(nuggets) >= 30:
            nuggets = nuggets[:30]
            break

    labels: list[str] = []
    for chunk in chunks(nuggets, 10):
        prompt = f"""
Label each nugget as "vital" or "okay" for the search query.
Return JSON array only, same order and same length.
If the nugget is "{NO_INFO_NUGGET}", label it "vital".

Search Query: {run.query}
Nugget List: {json.dumps(chunk, ensure_ascii=False)}
"""
        labels.extend(normalize_values(parse_list(client.chat(prompt)), {"vital", "okay"}))

    sorted_pairs = sorted(zip(nuggets, labels), key=lambda pair: pair[1] == "okay")[:20]
    sorted_nuggets = [pair[0] for pair in sorted_pairs]
    sorted_labels = [pair[1] for pair in sorted_pairs]
    answer = generated_text(run.generated_answer_parts)

    assignments: list[str] = []
    for chunk in chunks(sorted_nuggets, 10):
        prompt = f"""
For each nugget, label whether it is captured by the generated answer.
Allowed labels: "support", "partial_support", "not_support".
Return JSON array only, same order and same length.

Search Query: {run.query}

Generated Answer:
{answer}

Nugget List: {json.dumps(chunk, ensure_ascii=False)}
"""
        assignments.extend(
            normalize_values(parse_list(client.chat(prompt)), {"support", "partial_support", "not_support"})
        )

    nuggetizer_scores = evaluate_nuggets(sorted_nuggets, sorted_labels, assignments)
    score_map = {"support": 1.0, "partial_support": 0.5, "not_support": 0.0}
    assignment_scores = [score_map.get(item, 0.0) for item in assignments]
    return {
        "nuggetizer_scores": nuggetizer_scores,
        "nuggets": sorted_nuggets,
        "labels": sorted_labels,
        "assignments": assignments,
        "assignment_scores": assignment_scores,
        "mean_nugget_assignment_score": (
            sum(assignment_scores) / len(assignment_scores) if assignment_scores else 0.0
        ),
        "vital_nuggetizer_score": nuggetizer_scores.get("Vital", 0.5),
    }


def evaluate_nuggets(nuggets: list[str], labels: list[str], assignments: list[str]) -> dict[str, float]:
    score_map = {"support": 1.0, "partial_support": 0.5, "not_support": 0.0}
    vital_scores: list[float] = []
    okay_scores: list[float] = []
    strict_vital_scores: list[float] = []
    strict_okay_scores: list[float] = []
    all_scores: list[float] = []
    all_strict_scores: list[float] = []
    for label, assignment in zip(labels, assignments):
        score = score_map.get(assignment, 0.0)
        strict = 1.0 if assignment == "support" else 0.0
        all_scores.append(score)
        all_strict_scores.append(strict)
        if label == "vital":
            vital_scores.append(score)
            strict_vital_scores.append(strict)
        elif label == "okay":
            okay_scores.append(score)
            strict_okay_scores.append(strict)

    num_nuggets = max(len(nuggets), 1)
    num_vital = max(len(vital_scores), 1)
    num_okay = max(len(okay_scores), 1)
    return {
        "All": sum(all_scores) / num_nuggets,
        "All Strict": sum(all_strict_scores) / num_nuggets,
        "Vital": sum(vital_scores) / num_vital,
        "Vital Strict": sum(strict_vital_scores) / num_vital,
        "Weighted": (sum(vital_scores) + 0.5 * sum(okay_scores)) / (num_vital + 0.5 * num_okay),
        "Weighted Strict": (sum(strict_vital_scores) + 0.5 * sum(strict_okay_scores)) / (num_vital + 0.5 * num_okay),
    }


def compute_citation(client: OpenAICompatibleClient, run: RAGRun) -> dict[str, Any]:
    score_map = {"full_support": 1.0, "partial_support": 0.5, "no_support": 0.0}
    citation_to_scores: dict[str, list[float]] = defaultdict(list)
    part_to_scores: dict[str, list[float]] = defaultdict(list)
    for idx, part in enumerate(run.generated_answer_parts, start=1):
        if not part.citations:
            part_to_scores[f"part_score_{idx}"] = []
            continue
        for citation in part.citations:
            passage = run.retrieved_passages.get(citation, "")
            if not passage:
                continue
            prompt = f"""
Evaluate whether the statement is supported by the citation.
Allowed labels: "full_support", "partial_support", "no_support".
Return JSON only: {{"support": "full_support"}}

Statement:
{part.text}

Citation:
{passage}
"""
            support = normalize_values([parse_json_obj(client.chat(prompt)).get("support", "")], set(score_map))[0]
            score = score_map[support]
            citation_to_scores[f"citation_score_{citation}"].append(score)
            part_to_scores[f"part_score_{idx}"].append(score)

    citation_avg = {key: sum(vals) / len(vals) if vals else 0.0 for key, vals in citation_to_scores.items()}
    part_avg = {key: sum(vals) / len(vals) if vals else 0.0 for key, vals in part_to_scores.items()}
    precision = sum(citation_avg.values()) / len(citation_avg) if citation_avg else 0.0
    recall = sum(part_avg.values()) / len(part_avg) if part_avg else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        **citation_avg,
        **part_avg,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "f1": f1,
    }


def compute_no_answer(client: OpenAICompatibleClient, run: RAGRun) -> dict[str, str]:
    prompt = f"""
Determine whether the answer is an attempt to answer the query.
Do not judge correctness. If it attempts to answer, return yes. If it says it cannot answer or lacks information, return no.
Return JSON only: {{"answered": "yes"}}

Query:
{run.query}

Answer:
{generated_text(run.generated_answer_parts)}
"""
    answered = parse_json_obj(client.chat(prompt)).get("answered", "")
    answered = normalize_values([answered], {"yes", "no"})[0]
    return {"query_answered": answered}


def compute_hallucination_llm(client: OpenAICompatibleClient, run: RAGRun) -> float:
    sources = "\n\n".join(run.retrieved_passages.values())
    answer = generated_text(run.generated_answer_parts)
    prompt = f"""
Score how factually supported the generated answer is by the source passages.
Return JSON only: {{"score": 0.0}}
score must be between 0 and 1, where 1 means fully supported and 0 means unsupported/contradicted.

Source passages:
{sources}

Generated answer:
{answer}
"""
    score = float(parse_json_obj(client.chat(prompt)).get("score", 0.0))
    return max(0.0, min(1.0, score))


def compute_golden(client: OpenAICompatibleClient, run: RAGRun, expected_answer: str) -> dict[str, Any]:
    answer = generated_text(run.generated_answer_parts)
    emb = client.embeddings([answer, expected_answer])
    semantic = cosine_similarity(emb[0], emb[1])
    generated_claims = extract_claims(client, answer)
    expected_claims = extract_claims(client, expected_answer)
    if not generated_claims or not expected_claims:
        precision = recall = f1 = 0.0
        precision_verdicts: list[dict[str, str]] = []
        recall_verdicts: list[dict[str, str]] = []
    else:
        precision_verdicts = verify_claims(client, generated_claims, expected_answer)
        recall_verdicts = verify_claims(client, expected_claims, answer)
        precision = entailment_ratio(precision_verdicts)
        recall = entailment_ratio(recall_verdicts)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "semantic_similarity": semantic,
        "factual_correctness_precision": precision,
        "factual_correctness_recall": recall,
        "factual_correctness_f1": f1,
        "generated_claims": generated_claims,
        "expected_claims": expected_claims,
        "precision_verdicts": precision_verdicts,
        "recall_verdicts": recall_verdicts,
    }


def extract_claims(client: OpenAICompatibleClient, text: str) -> list[str]:
    prompt = f"""
Extract all atomic factual claims from the text.
Each claim should be a single verifiable statement.
Return JSON only: {{"claims": ["claim 1", "claim 2"]}}

Text:
{text}
"""
    claims = parse_json_obj(client.chat(prompt)).get("claims", [])
    return [str(claim).strip() for claim in claims if str(claim).strip()]


def verify_claims(client: OpenAICompatibleClient, claims: list[str], reference: str) -> list[dict[str, str]]:
    prompt = f"""
For each claim, determine whether it is "entailment", "contradiction", or "neutral" with respect to the reference text.
Return JSON only: {{"verdicts": [{{"claim": "...", "verdict": "entailment"}}]}}

Reference text:
{reference}

Claims:
{json.dumps(claims, ensure_ascii=False)}
"""
    verdicts = parse_json_obj(client.chat(prompt)).get("verdicts", [])
    normalized = []
    for item in verdicts:
        if not isinstance(item, dict):
            continue
        verdict = normalize_values([item.get("verdict", "")], {"entailment", "contradiction", "neutral"})[0]
        normalized.append({"claim": str(item.get("claim", "")), "verdict": verdict})
    return normalized


def entailment_ratio(verdicts: list[dict[str, str]]) -> float:
    if not verdicts:
        return 0.0
    return sum(1 for item in verdicts if item.get("verdict") == "entailment") / len(verdicts)


def normalize_values(values: list[str], allowed: set[str]) -> list[str]:
    normalized = []
    for value in values:
        lowered = str(value).strip().lower()
        lowered = lowered.replace("-", "_").replace(" ", "_")
        if lowered not in allowed:
            match = next((item for item in allowed if item in lowered), None)
            if match is None:
                raise ValueError(f"Value {value!r} is not one of {sorted(allowed)}")
            lowered = match
        normalized.append(lowered)
    return normalized


def chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[idx:idx + size] for idx in range(0, len(items), size)]


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def evaluate(args: argparse.Namespace) -> None:
    client = OpenAICompatibleClient(
        chat_base_url=args.chat_base_url,
        chat_api_key=args.chat_api_key,
        chat_model=args.chat_model,
        embedding_base_url=args.embedding_base_url,
        embedding_api_key=args.embedding_api_key,
        embedding_model=args.embedding_model,
        timeout=args.timeout,
        retries=args.retries,
    )
    golden = load_golden(args.golden_csv) if args.golden_csv else {}
    runs = load_rag_runs(args.answers_csv)
    k_values = [int(k) for k in args.k_values.split(",") if k.strip()]

    rows: list[dict[str, Any]] = []
    total = len(runs)
    for idx, run in enumerate(runs, start=1):
        print(f"[{idx}/{total}] evaluating {run.query_id} run {run.query_run}: {run.query}", file=sys.stderr)
        client.input_tokens = 0
        client.output_tokens = 0

        row: dict[str, Any] = {
            "query_id": run.query_id,
            "query": run.query,
            "query_run": run.query_run,
            "generated_answer": generated_text(run.generated_answer_parts),
        }

        umbrella = compute_umbrela(client, run, k_values)
        autonugget = compute_autonugget(client, run, umbrella["umbrela_scores"])
        citation = compute_citation(client, run)
        no_answer = compute_no_answer(client, run)
        hallucination = compute_hallucination_llm(client, run)

        row.update({
            "retrieval_score_umbrela_scores": json_cell(umbrella["umbrela_scores"]),
            "retrieval_score_precision_metrics": json_cell(umbrella["retrieval_scores"]),
            "retrieval_score_mean_umbrela_score": umbrella["mean_umbrela_score"],
            "generation_score_autonugget_scores": json_cell({
                "nuggetizer_scores": autonugget["nuggetizer_scores"],
                "nuggets": autonugget["nuggets"],
                "labels": autonugget["labels"],
                "assignments": autonugget["assignments"],
                "assignment_scores": autonugget["assignment_scores"],
            }),
            "generation_score_mean_nugget_assignment_score": autonugget["mean_nugget_assignment_score"],
            "generation_score_vital_nuggetizer_score": autonugget["vital_nuggetizer_score"],
            "generation_score_hallucination_score": hallucination,
            "generation_score_citation_scores": json_cell(citation),
            "generation_score_citation_f1_score": citation["f1"],
            "generation_score_no_answer_score": json_cell(no_answer),
        })

        if run.query_id in golden:
            expected = golden[run.query_id]["expected_answer"]
            golden_scores = compute_golden(client, run, expected)
            row.update({
                "generation_score_semantic_similarity": golden_scores["semantic_similarity"],
                "generation_score_factual_correctness_precision": golden_scores["factual_correctness_precision"],
                "generation_score_factual_correctness_recall": golden_scores["factual_correctness_recall"],
                "generation_score_factual_correctness_f1": golden_scores["factual_correctness_f1"],
                "generation_score_expected_answer": expected,
                "generation_score_generated_claims": json_cell(golden_scores["generated_claims"]),
                "generation_score_expected_claims": json_cell(golden_scores["expected_claims"]),
                "generation_score_precision_verdicts": json_cell(golden_scores["precision_verdicts"]),
                "generation_score_recall_verdicts": json_cell(golden_scores["recall_verdicts"]),
            })

        row["total_input_tokens"] = client.input_tokens
        row["total_output_tokens"] = client.output_tokens
        row["total_tokens"] = client.input_tokens + client.output_tokens
        rows.append(row)

    write_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}", file=sys.stderr)


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def env_or_default(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG CSV with OpenAI-compatible local models.")
    parser.add_argument("--answers-csv", default="data/chat.csv")
    parser.add_argument("--golden-csv", default="data/qa_golden.csv")
    parser.add_argument("--output-csv", default="data/local_eval_results.csv")
    parser.add_argument("--chat-base-url", default=env_or_default("LOCAL_LLM_BASE_URL"))
    parser.add_argument("--chat-api-key", default=env_or_default("LOCAL_LLM_API_KEY", "EMPTY"))
    parser.add_argument("--chat-model", default=env_or_default("LOCAL_LLM_MODEL"))
    parser.add_argument("--embedding-base-url", default=env_or_default("LOCAL_EMBEDDING_BASE_URL") or env_or_default("LOCAL_LLM_BASE_URL"))
    parser.add_argument("--embedding-api-key", default=env_or_default("LOCAL_EMBEDDING_API_KEY") or env_or_default("LOCAL_LLM_API_KEY", "EMPTY"))
    parser.add_argument("--embedding-model", default=env_or_default("LOCAL_EMBEDDING_MODEL"))
    parser.add_argument("--k-values", default="1,3,5")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    required = {
        "--chat-base-url or LOCAL_LLM_BASE_URL": args.chat_base_url,
        "--chat-model or LOCAL_LLM_MODEL": args.chat_model,
        "--embedding-base-url or LOCAL_EMBEDDING_BASE_URL": args.embedding_base_url,
        "--embedding-model or LOCAL_EMBEDDING_MODEL": args.embedding_model,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error("Missing required settings: " + ", ".join(missing))
    return args


if __name__ == "__main__":
    evaluate(parse_args())
