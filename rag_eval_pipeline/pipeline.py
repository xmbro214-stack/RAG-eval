"""Run answer generation and OpenAI-compatible evaluation from a YAML grid.

The pipeline is intentionally a thin orchestrator around:

* scripts/generate_answers_http_rag.py
* scripts/openai_compatible_eval.py

It expands dataset/page-size/similarity combinations, writes each run into a
stable output folder, and records progress in a manifest for resumable runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from rag_eval_pipeline.config import config_value, load_config


ROOT = Path(__file__).resolve().parents[1]
GENERATION_SCRIPT = ROOT / "scripts" / "generate_answers_http_rag.py"
EVALUATION_SCRIPT = ROOT / "scripts" / "openai_compatible_eval.py"
VALID_STAGES = ("all", "generate", "eval")
LOGGER = logging.getLogger("rag_eval.pipeline")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"
SECRET_FLAGS = {
    "--llm-api-key",
    "--retrieval-api-key",
    "--chat-api-key",
    "--embedding-api-key",
}


@dataclass(frozen=True)
class PipelineTask:
    dataset_id: str
    dataset_name: str
    page_size: int
    similarity_threshold: float
    output_dir: Path
    generated_answers_csv: Path
    eval_result_csv: Path

    @property
    def key(self) -> str:
        return f"{self.dataset_name}/{self.output_dir.name}"


def slugify_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "dataset"


def similarity_slug(value: float) -> str:
    text = f"{value:g}".replace("-", "neg").replace(".", "p")
    return f"sim{text}"


def output_root(config: dict[str, Any]) -> Path:
    output = config.get("output") or {}
    root = Path(str(output.get("root") or "data/eval_runs"))
    run_name = slugify_name(str(output.get("run_name") or "ragflow_grid"))
    if not root.is_absolute():
        root = ROOT / root
    return root / run_name


def expand_tasks(config: dict[str, Any]) -> list[PipelineTask]:
    datasets = config.get("datasets") or []
    grid = config.get("grid") or {}
    page_sizes = grid.get("page_sizes") or []
    similarities = grid.get("similarity_thresholds") or []
    base = output_root(config)

    tasks: list[PipelineTask] = []
    for dataset in datasets:
        dataset_id = str(dataset.get("id", "")).strip()
        dataset_name = slugify_name(str(dataset.get("name") or dataset_id))
        for page_size in page_sizes:
            for similarity in similarities:
                page_size_int = int(page_size)
                similarity_float = float(similarity)
                combo_dir = base / dataset_name / f"ps{page_size_int}_{similarity_slug(similarity_float)}"
                tasks.append(
                    PipelineTask(
                        dataset_id=dataset_id,
                        dataset_name=dataset_name,
                        page_size=page_size_int,
                        similarity_threshold=similarity_float,
                        output_dir=combo_dir,
                        generated_answers_csv=combo_dir / "generated_answers.csv",
                        eval_result_csv=combo_dir / "eval_result.csv",
                    )
                )
    return tasks


def validate_config(config: dict[str, Any], stage: str) -> list[str]:
    missing: list[str] = []
    if not config.get("queries_csv"):
        missing.append("queries_csv")
    if stage in ("all", "eval") and not config.get("golden_csv"):
        missing.append("golden_csv")

    datasets = config.get("datasets") or []
    if not datasets:
        missing.append("datasets")
    for idx, dataset in enumerate(datasets):
        if not str(dataset.get("id", "")).strip():
            missing.append(f"datasets[{idx}].id")

    grid = config.get("grid") or {}
    if not grid.get("page_sizes"):
        missing.append("grid.page_sizes")
    if not grid.get("similarity_thresholds"):
        missing.append("grid.similarity_thresholds")

    generation = config.get("generation") or {}
    if stage in ("all", "generate"):
        for key in ("retrieval_url", "llm_base_url", "llm_model"):
            if config_value(generation, key) in (None, ""):
                missing.append(f"generation.{key}")
        if config_value(generation, "llm_api_key") in (None, ""):
            missing.append("generation.llm_api_key or generation.llm_api_key_env")

    evaluation = config.get("evaluation") or {}
    if stage in ("all", "eval"):
        for key in ("chat_base_url", "chat_model", "embedding_base_url", "embedding_model"):
            if config_value(evaluation, key) in (None, ""):
                missing.append(f"evaluation.{key} or evaluation.{key}_env")

    return missing


def add_optional_arg(command: list[str], flag: str, value: Any) -> None:
    if value not in (None, ""):
        command.extend([flag, str(value)])


def build_generation_command(config: dict[str, Any], task: PipelineTask) -> list[str]:
    generation = config.get("generation") or {}
    log_level = config_value(generation, "log_level", default=config_value(config.get("logging") or {}, "level"))
    command = [
        sys.executable,
        str(GENERATION_SCRIPT),
        "--queries-csv",
        str(resolve_repo_path(config["queries_csv"])),
        "--output-csv",
        str(task.generated_answers_csv),
        "--dataset-ids",
        task.dataset_id,
        "--page-size",
        str(task.page_size),
        "--similarity-threshold",
        str(task.similarity_threshold),
        "--retrieval-url",
        str(config_value(generation, "retrieval_url")),
        "--llm-base-url",
        str(config_value(generation, "llm_base_url")),
        "--llm-api-key",
        str(config_value(generation, "llm_api_key")),
        "--llm-model",
        str(config_value(generation, "llm_model")),
    ]

    optional_flags = {
        "--retrieval-api-key": config_value(generation, "retrieval_api_key"),
        "--retrieval-auth-header": generation.get("retrieval_auth_header"),
        "--retrieval-auth-scheme": generation.get("retrieval_auth_scheme"),
        "--retrieval-headers-json": generation.get("retrieval_headers_json"),
        "--retrieval-payload-json": generation.get("retrieval_payload_json"),
        "--retrieval-extra-body-json": generation.get("retrieval_extra_body_json"),
        "--document-ids": generation.get("document_ids"),
        "--page": generation.get("page"),
        "--vector-similarity-weight": generation.get("vector_similarity_weight"),
        "--top-k": generation.get("top_k"),
        "--rerank-id": generation.get("rerank_id"),
        "--keyword": generation.get("keyword"),
        "--highlight": generation.get("highlight"),
        "--cross-languages": generation.get("cross_languages"),
        "--metadata-condition-json": generation.get("metadata_condition_json"),
        "--use-kg": generation.get("use_kg"),
        "--toc-enhance": generation.get("toc_enhance"),
        "--max-passages": generation.get("max_passages"),
        "--temperature": generation.get("temperature"),
        "--max-tokens": generation.get("max_tokens"),
        "--system-prompt": generation.get("system_prompt"),
        "--prompt-template-file": generation.get("prompt_template_file"),
        "--repeat-query": generation.get("repeat_query"),
        "--max-workers": generation.get("max_workers"),
        "--timeout": generation.get("timeout"),
        "--retries": generation.get("retries"),
        "--log-level": log_level,
    }
    for flag, value in optional_flags.items():
        add_optional_arg(command, flag, value)
    return command


def build_evaluation_command(config: dict[str, Any], task: PipelineTask) -> list[str]:
    evaluation = config.get("evaluation") or {}
    log_level = config_value(evaluation, "log_level", default=config_value(config.get("logging") or {}, "level"))
    command = [
        sys.executable,
        str(EVALUATION_SCRIPT),
        "--answers-csv",
        str(task.generated_answers_csv),
        "--golden-csv",
        str(resolve_repo_path(config["golden_csv"])),
        "--output-csv",
        str(task.eval_result_csv),
        "--chat-base-url",
        str(config_value(evaluation, "chat_base_url")),
        "--chat-model",
        str(config_value(evaluation, "chat_model")),
        "--embedding-base-url",
        str(config_value(evaluation, "embedding_base_url")),
        "--embedding-model",
        str(config_value(evaluation, "embedding_model")),
    ]
    optional_flags = {
        "--chat-api-key": config_value(evaluation, "chat_api_key", default="EMPTY"),
        "--embedding-api-key": config_value(evaluation, "embedding_api_key"),
        "--k-values": evaluation.get("k_values"),
        "--max-workers": evaluation.get("max_workers"),
        "--timeout": evaluation.get("timeout"),
        "--retries": evaluation.get("retries"),
        "--log-level": log_level,
    }
    for flag, value in optional_flags.items():
        add_optional_arg(command, flag, value)
    return command


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def should_overwrite(config: dict[str, Any], cli_overwrite: bool) -> bool:
    return cli_overwrite or bool((config.get("output") or {}).get("overwrite", False))


def manifest_entry(task: PipelineTask) -> dict[str, Any]:
    item = asdict(task)
    item["output_dir"] = str(task.output_dir)
    item["generated_answers_csv"] = str(task.generated_answers_csv)
    item["eval_result_csv"] = str(task.eval_result_csv)
    return item


def write_manifest(path: Path, entries: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tasks": list(entries),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def log(message: str) -> None:
    LOGGER.info(message)


def redacted_command(command: list[str]) -> str:
    redacted: list[str] = []
    redact_next = False
    for part in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        redacted.append(part)
        if part in SECRET_FLAGS:
            redact_next = True
    return " ".join(redacted)


def run_command(command: list[str], dry_run: bool) -> float:
    started_at = time.perf_counter()
    command_text = "$ " + redacted_command(command)
    if dry_run:
        log(command_text)
    else:
        LOGGER.debug(command_text)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)
    return time.perf_counter() - started_at


def run_pipeline(config: dict[str, Any], *, stage: str, dry_run: bool, overwrite: bool) -> list[dict[str, Any]]:
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage {stage!r}; expected one of {', '.join(VALID_STAGES)}")

    missing = validate_config(config, stage)
    if missing:
        raise ValueError("Missing required settings: " + ", ".join(missing))

    tasks = expand_tasks(config)
    if not tasks:
        raise ValueError("No pipeline tasks were generated from the config.")

    do_overwrite = should_overwrite(config, overwrite)
    entries: list[dict[str, Any]] = []
    manifest_path = output_root(config) / "manifest.json"
    log(
        "Pipeline started: "
        f"stage={stage}, dry_run={dry_run}, overwrite={do_overwrite}, "
        f"tasks={len(tasks)}, output_root={output_root(config)}"
    )

    for index, task in enumerate(tasks, start=1):
        log(
            f"[{index}/{len(tasks)}] Task dataset={task.dataset_name} "
            f"(id={task.dataset_id}), page_size={task.page_size}, "
            f"similarity={task.similarity_threshold}, output_dir={task.output_dir}"
        )
        if not dry_run:
            task.output_dir.mkdir(parents=True, exist_ok=True)
        entry = manifest_entry(task)
        entry["generation_status"] = "not_requested"
        entry["evaluation_status"] = "not_requested"
        entry["generation_seconds"] = None
        entry["evaluation_seconds"] = None

        if stage in ("all", "generate"):
            if task.generated_answers_csv.exists() and not do_overwrite:
                entry["generation_status"] = "skipped_existing"
                log(f"[{index}/{len(tasks)}] Generate skipped: existing {task.generated_answers_csv}")
            else:
                command = build_generation_command(config, task)
                entry["generation_command"] = command
                try:
                    log(f"[{index}/{len(tasks)}] Generate start -> {task.generated_answers_csv}")
                    entry["generation_seconds"] = run_command(command, dry_run)
                    entry["generation_status"] = "dry_run" if dry_run else "completed"
                    log(
                        f"[{index}/{len(tasks)}] Generate {entry['generation_status']} "
                        f"in {entry['generation_seconds']:.2f}s"
                    )
                except subprocess.CalledProcessError as exc:
                    entry["generation_status"] = "failed"
                    entry["generation_error"] = str(exc)
                    LOGGER.error(f"[{index}/{len(tasks)}] Generate failed: {exc}")
                    entries.append(entry)
                    write_manifest(manifest_path, entries)
                    raise

        if stage in ("all", "eval"):
            if not dry_run and not task.generated_answers_csv.exists():
                entry["evaluation_status"] = "missing_generated_answers"
                log(f"[{index}/{len(tasks)}] Eval skipped: missing {task.generated_answers_csv}")
            elif task.eval_result_csv.exists() and not do_overwrite:
                entry["evaluation_status"] = "skipped_existing"
                log(f"[{index}/{len(tasks)}] Eval skipped: existing {task.eval_result_csv}")
            else:
                command = build_evaluation_command(config, task)
                entry["evaluation_command"] = command
                try:
                    log(f"[{index}/{len(tasks)}] Eval start -> {task.eval_result_csv}")
                    entry["evaluation_seconds"] = run_command(command, dry_run)
                    entry["evaluation_status"] = "dry_run" if dry_run else "completed"
                    log(
                        f"[{index}/{len(tasks)}] Eval {entry['evaluation_status']} "
                        f"in {entry['evaluation_seconds']:.2f}s"
                    )
                except subprocess.CalledProcessError as exc:
                    entry["evaluation_status"] = "failed"
                    entry["evaluation_error"] = str(exc)
                    LOGGER.error(f"[{index}/{len(tasks)}] Eval failed: {exc}")
                    entries.append(entry)
                    write_manifest(manifest_path, entries)
                    raise

        entries.append(entry)
        if not dry_run:
            write_manifest(manifest_path, entries)

    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YAML-driven RAG generation/evaluation grid.")
    parser.add_argument("--config", default="scripts/eval-cfg.yaml", help="Pipeline YAML config path.")
    parser.add_argument("--stage", choices=VALID_STAGES, default="all", help="Pipeline stage to run.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated/eval outputs.")
    parser.add_argument("--log-level", default=os.getenv("RAG_EVAL_LOG_LEVEL"), help="Logging level: DEBUG, INFO, WARNING, ERROR.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configured_level = args.log_level or str((config.get("logging") or {}).get("level") or "INFO")
    configure_logging(configured_level)
    entries = run_pipeline(
        config,
        stage=args.stage,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    LOGGER.info(f"Pipeline tasks processed: {len(entries)}")


if __name__ == "__main__":
    main()
