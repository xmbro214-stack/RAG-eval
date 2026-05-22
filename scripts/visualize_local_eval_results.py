"""Generate a self-contained HTML viewer for local evaluation results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = [
    ("retrieval_score_mean_umbrela_score", "UMBRELA", "retrieval"),
    ("generation_score_vital_nuggetizer_score", "Vital nuggets", "recall"),
    ("generation_score_mean_nugget_assignment_score", "Nugget assignment", "recall"),
    ("generation_score_factual_correctness_recall", "Factual recall", "recall"),
    ("generation_score_factual_correctness_precision", "Factual precision", "quality"),
    ("generation_score_factual_correctness_f1", "Factual F1", "quality"),
    ("generation_score_semantic_similarity", "Semantic similarity", "quality"),
    ("generation_score_faithfulness_score", "Faithfulness", "support"),
    ("generation_score_citation_f1_score", "Citation F1", "citation"),
    ("generation_score_hallucination_score", "Source support", "support"),
]

DETAIL_COLUMNS = [
    ("retrieval_score_umbrela_scores", "UMBRELA passage scores"),
    ("retrieval_score_precision_metrics", "Retrieval precision/AP/MRR"),
    ("retrieval_score_ndcg_metrics", "Retrieval NDCG@K"),
    ("generation_score_autonugget_scores", "Nuggets and assignments"),
    ("generation_score_faithfulness_claims", "Faithfulness claims"),
    ("generation_score_faithfulness_verdicts", "Faithfulness verdicts"),
    ("generation_score_unsupported_claims", "Unsupported claims"),
    ("generation_score_citation_scores", "Citation scores"),
    ("generation_score_no_answer_score", "No-answer score"),
    ("generation_score_generated_claims", "Generated claims"),
    ("generation_score_expected_claims", "Expected claims"),
    ("generation_score_precision_verdicts", "Precision verdicts"),
    ("generation_score_recall_verdicts", "Recall verdicts"),
]

DATASETS = [
    ("custom", "Custom chunking"),
    ("trd", "TRD chunking"),
]

DEFAULT_PIPELINE_RUN = "ragflow_grid"


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def percent(value: float | None) -> str:
    return "0%" if value is None else f"{max(0, min(value, 1)) * 100:.1f}%"


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def metric_value(row: dict[str, str], column: str) -> float | None:
    return to_float(row.get(column))


def recall_average(row: dict[str, str]) -> float | None:
    values = [
        metric_value(row, "generation_score_factual_correctness_recall"),
        metric_value(row, "generation_score_vital_nuggetizer_score"),
        metric_value(row, "generation_score_mean_nugget_assignment_score"),
    ]
    values = [v for v in values if v is not None]
    return mean(values)


def score_class(value: float | None) -> str:
    if value is None:
        return "score-missing"
    if value >= 0.8:
        return "score-good"
    if value >= 0.5:
        return "score-medium"
    return "score-low"


def parse_jsonish(value: str) -> Any:
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def pretty_json(value: str) -> str:
    parsed = parse_jsonish(value)
    if isinstance(parsed, (dict, list)):
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    return str(parsed)


def truncate(text: str, limit: int = 180) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def summarize_overall(rows: list[dict[str, str]]) -> dict[str, float | None]:
    summary = {}
    for column, _, _ in METRICS:
        values = [metric_value(row, column) for row in rows]
        summary[column] = mean([v for v in values if v is not None])
    recall_values = [recall_average(row) for row in rows]
    summary["recall_average"] = mean([v for v in recall_values if v is not None])
    token_values = [metric_value(row, "total_tokens") for row in rows]
    summary["total_tokens"] = mean([v for v in token_values if v is not None])
    return summary


def answered_count(rows: list[dict[str, str]]) -> int:
    count = 0
    for row in rows:
        parsed = parse_jsonish(row.get("generation_score_no_answer_score", ""))
        if isinstance(parsed, dict) and parsed.get("query_answered") == "yes":
            count += 1
    return count


def summarize_by_query(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("query_id", "")].append(row)

    summaries = []
    for query_id, group in grouped.items():
        item: dict[str, Any] = {
            "query_id": query_id,
            "query": group[0].get("query", ""),
            "runs": len(group),
        }
        for column, _, _ in METRICS:
            values = [metric_value(row, column) for row in group]
            item[column] = mean([v for v in values if v is not None])
        recall_values = [recall_average(row) for row in group]
        item["recall_average"] = mean([v for v in recall_values if v is not None])
        summaries.append(item)
    return sorted(summaries, key=lambda item: item["query_id"])


def render_metric_bar(label: str, value: float | None, extra_class: str = "") -> str:
    safe_label = html.escape(label)
    value_text = fmt(value)
    return f"""
      <div class="metric-row {extra_class}">
        <div class="metric-head">
          <span>{safe_label}</span>
          <strong>{value_text}</strong>
        </div>
        <div class="bar"><span class="{score_class(value)}" style="width: {percent(value)}"></span></div>
      </div>
    """


def render_summary_cards(summary: dict[str, float | None], total_rows: int, total_queries: int) -> str:
    cards = [
        ("Rows", str(total_rows), "Evaluation runs"),
        ("Queries", str(total_queries), "Unique query_id count"),
        ("Recall avg", fmt(summary["recall_average"]), "Mean of factual/vital/assignment recall"),
        ("Factual recall", fmt(summary["generation_score_factual_correctness_recall"]), "Golden claim coverage"),
        ("Vital nuggets", fmt(summary["generation_score_vital_nuggetizer_score"]), "Key nugget coverage"),
        ("Faithfulness", fmt(summary["generation_score_faithfulness_score"]), "Claims supported by retrieved context"),
    ]
    return "\n".join(
        f"""
        <section class="summary-card">
          <span>{html.escape(label)}</span>
          <strong>{html.escape(value)}</strong>
          <small>{html.escape(note)}</small>
        </section>
        """
        for label, value, note in cards
    )


def render_query_table(query_summaries: list[dict[str, Any]]) -> str:
    rows = []
    for item in query_summaries:
        recall_avg = item["recall_average"]
        rows.append(
            f"""
            <tr>
              <td>{html.escape(item['query_id'])}</td>
              <td>{item['runs']}</td>
              <td>{html.escape(truncate(item['query'], 120))}</td>
              <td class="{score_class(recall_avg)}">{fmt(recall_avg)}</td>
              <td>{fmt(item['generation_score_factual_correctness_recall'])}</td>
              <td>{fmt(item['generation_score_vital_nuggetizer_score'])}</td>
              <td>{fmt(item['generation_score_mean_nugget_assignment_score'])}</td>
              <td>{fmt(item['retrieval_score_mean_umbrela_score'])}</td>
            </tr>
            """
        )
    return f"""
      <table>
        <thead>
          <tr>
            <th>query_id</th>
            <th>runs</th>
            <th>query</th>
            <th>recall avg</th>
            <th>factual recall</th>
            <th>vital nuggets</th>
            <th>nugget assignment</th>
            <th>UMBRELA</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    """


def render_detail(row: dict[str, str], row_number: int) -> str:
    recall_avg = recall_average(row)
    metric_bars = [render_metric_bar("Recall average", recall_avg, "featured")]
    metric_bars.extend(
        render_metric_bar(label, metric_value(row, column))
        for column, label, _ in METRICS
    )

    detail_blocks = []
    for column, label in DETAIL_COLUMNS:
        if column not in row or not row[column]:
            continue
        detail_blocks.append(
            f"""
            <details>
              <summary>{html.escape(label)}</summary>
              <pre>{html.escape(pretty_json(row[column]))}</pre>
            </details>
            """
        )

    query_id = html.escape(row.get("query_id", ""))
    query_run = html.escape(row.get("query_run", ""))
    query = html.escape(row.get("query", ""))
    generated_preview = html.escape(truncate(row.get("generated_answer", ""), 420))
    expected_preview = html.escape(truncate(row.get("generation_score_expected_answer", ""), 420))

    return f"""
      <article class="run-card" data-query="{query_id.lower()}" data-band="{score_class(recall_avg)}">
        <div class="run-title">
          <div>
            <span class="eyebrow">Run {row_number}</span>
            <h3>{query_id} · run {query_run}</h3>
            <p>{query}</p>
          </div>
          <strong class="pill {score_class(recall_avg)}">recall {fmt(recall_avg)}</strong>
        </div>
        <div class="metric-grid">{''.join(metric_bars)}</div>
        <div class="preview-grid">
          <section>
            <h4>Generated answer preview</h4>
            <p>{generated_preview or 'N/A'}</p>
          </section>
          <section>
            <h4>Expected answer preview</h4>
            <p>{expected_preview or 'N/A'}</p>
          </section>
        </div>
        <div class="details-grid">{''.join(detail_blocks)}</div>
      </article>
    """


def render_html(rows: list[dict[str, str]], input_csv: str) -> str:
    summary = summarize_overall(rows)
    query_summaries = summarize_by_query(rows)
    run_cards = "\n".join(render_detail(row, idx) for idx, row in enumerate(rows, start=1))
    query_options = "\n".join(
        f"<option value=\"{html.escape(item['query_id'].lower())}\">{html.escape(item['query_id'])}</option>"
        for item in query_summaries
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local Eval Results Viewer</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --green: #238636;
      --yellow: #b7791f;
      --red: #c2410c;
      --blue: #1f5eff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      padding: 28px 32px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    header h1 {{ margin: 0 0 8px; font-size: 28px; }}
    header p {{ margin: 0; color: var(--muted); }}
    main {{ padding: 24px 32px 40px; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .summary-card, .panel, .run-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .summary-card {{ padding: 16px; }}
    .summary-card span, .eyebrow {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .summary-card strong {{ display: block; margin: 6px 0; font-size: 28px; }}
    .summary-card small {{ color: var(--muted); }}
    .panel {{ padding: 18px; margin-bottom: 24px; overflow-x: auto; }}
    .panel h2 {{ margin: 0 0 14px; font-size: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 9px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; background: #fafbfc; }}
    .toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }}
    select, input {{
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: white;
      min-width: 180px;
    }}
    .run-card {{ padding: 18px; margin-bottom: 16px; }}
    .run-title {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 14px;
    }}
    .run-title h3 {{ margin: 4px 0 6px; font-size: 18px; }}
    .run-title p {{ margin: 0; color: var(--muted); }}
    .pill {{
      border-radius: 999px;
      padding: 6px 10px;
      white-space: nowrap;
      background: #eef2ff;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .metric-row {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }}
    .metric-row.featured {{ border-color: #9eb4ff; background: #f5f7ff; }}
    .metric-head {{ display: flex; justify-content: space-between; gap: 8px; font-size: 13px; }}
    .bar {{ height: 8px; background: #edf0f5; border-radius: 999px; overflow: hidden; margin-top: 8px; }}
    .bar span {{ display: block; height: 100%; border-radius: inherit; }}
    .score-good, .bar .score-good {{ color: var(--green); background: #2da44e; }}
    .score-medium, .bar .score-medium {{ color: var(--yellow); background: #d29922; }}
    .score-low, .bar .score-low {{ color: var(--red); background: #f97316; }}
    .score-missing, .bar .score-missing {{ color: var(--muted); background: #c7ced8; }}
    .preview-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .preview-grid section {{
      border-left: 3px solid var(--line);
      padding-left: 12px;
    }}
    .preview-grid h4 {{ margin: 0 0 6px; font-size: 13px; color: var(--muted); }}
    .preview-grid p {{ margin: 0; line-height: 1.45; }}
    .details-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 10px;
    }}
    details {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: #fbfcfe;
    }}
    summary {{ cursor: pointer; font-weight: 600; }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
      color: #243447;
    }}
    .hidden {{ display: none; }}
    @media (max-width: 720px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .run-title {{ display: block; }}
      .pill {{ display: inline-block; margin-top: 10px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Local Eval Results Viewer</h1>
    <p>Source: {html.escape(input_csv)} · self-contained report for RAG evaluation metrics</p>
  </header>
  <main>
    <section class="summary-grid">
      {render_summary_cards(summary, len(rows), len(query_summaries))}
    </section>
    <section class="panel">
      <h2>按 Query 聚合</h2>
      {render_query_table(query_summaries)}
    </section>
    <section class="panel">
      <h2>Run 明细</h2>
      <div class="toolbar">
        <select id="queryFilter">
          <option value="">All query_id</option>
          {query_options}
        </select>
        <select id="bandFilter">
          <option value="">All recall bands</option>
          <option value="score-good">Good ≥ 0.8</option>
          <option value="score-medium">Medium 0.5-0.8</option>
          <option value="score-low">Low &lt; 0.5</option>
        </select>
        <input id="searchBox" type="search" placeholder="Search query text">
      </div>
      <div id="runCards">{run_cards}</div>
    </section>
  </main>
  <script>
    const queryFilter = document.getElementById('queryFilter');
    const bandFilter = document.getElementById('bandFilter');
    const searchBox = document.getElementById('searchBox');
    const cards = Array.from(document.querySelectorAll('.run-card'));
    function applyFilters() {{
      const query = queryFilter.value;
      const band = bandFilter.value;
      const term = searchBox.value.trim().toLowerCase();
      cards.forEach(card => {{
        const queryOk = !query || card.dataset.query === query;
        const bandOk = !band || card.dataset.band === band;
        const textOk = !term || card.textContent.toLowerCase().includes(term);
        card.classList.toggle('hidden', !(queryOk && bandOk && textOk));
      }});
    }}
    queryFilter.addEventListener('change', applyFilters);
    bandFilter.addEventListener('change', applyFilters);
    searchBox.addEventListener('input', applyFilters);
  </script>
</body>
</html>
"""


def delta_text(left: float | None, right: float | None) -> str:
    if left is None or right is None:
        return "N/A"
    return f"{left - right:+.3f}"


def delta_class(left: float | None, right: float | None) -> str:
    if left is None or right is None:
        return "delta-missing"
    diff = left - right
    if diff > 0.02:
        return "delta-up"
    if diff < -0.02:
        return "delta-down"
    return "delta-flat"


def combo_sort_key(item: str) -> tuple[float, str]:
    try:
        return (float(item), item)
    except ValueError:
        return (float("inf"), item)


def parse_pipeline_combo_name(name: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"ps(?P<page_size>\d+)_sim(?P<similarity>.+)", name)
    if not match:
        return None
    similarity = match.group("similarity").replace("neg", "-").replace("p", ".")
    return match.group("page_size"), similarity


def dataset_payload(path: Path, label: str) -> dict[str, Any]:
    rows = read_rows(str(path))
    return {
        "label": label,
        "path": str(path),
        "rows": rows,
        "summary": summarize_overall(rows),
        "queries": summarize_by_query(rows),
        "answered": answered_count(rows),
    }


def read_pipeline_compare_inputs(run_root: Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    results: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for dataset, label in DATASETS:
        dataset_root = run_root / dataset
        if not dataset_root.exists():
            continue
        for combo_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
            parsed = parse_pipeline_combo_name(combo_dir.name)
            if not parsed:
                continue
            page_size, similarity = parsed
            path = combo_dir / "eval_result.csv"
            if not path.exists():
                continue
            results.setdefault(page_size, {}).setdefault(similarity, {})[dataset] = dataset_payload(path, label)

    complete_results: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    expected = {dataset for dataset, _ in DATASETS}
    for page_size, similarity_data in results.items():
        for similarity, page_data in similarity_data.items():
            if expected.issubset(page_data):
                complete_results.setdefault(page_size, {})[similarity] = page_data
    return complete_results


def read_compare_inputs(root: Path, pipeline_run: str = DEFAULT_PIPELINE_RUN) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    run_root = root / "data" / "eval_runs" / pipeline_run
    pipeline_results = read_pipeline_compare_inputs(run_root)
    if not pipeline_results:
        raise FileNotFoundError(f"Missing pipeline comparison inputs under: {run_root}")
    return pipeline_results


def render_compare_cards(page_data: dict[str, dict[str, Any]]) -> str:
    cards = []
    for dataset, label in DATASETS:
        data = page_data[dataset]
        summary = data["summary"]
        rows = data["rows"]
        queries = data["queries"]
        cards.append(
            f"""
            <section class="compare-card">
              <div>
                <span class="eyebrow">{html.escape(label)}</span>
                <strong>{fmt(summary['recall_average'])}</strong>
                <small>Recall avg · {len(rows)} runs · {len(queries)} queries · answered {data['answered']}/{len(rows)}</small>
              </div>
              <div class="card-metrics">
                <span>Factual {fmt(summary['generation_score_factual_correctness_recall'])}</span>
                <span>Vital {fmt(summary['generation_score_vital_nuggetizer_score'])}</span>
                <span>Nugget {fmt(summary['generation_score_mean_nugget_assignment_score'])}</span>
                <span>Tokens {fmt(summary['total_tokens'], 0)}</span>
              </div>
            </section>
            """
        )

    left = page_data["custom"]["summary"]["recall_average"]
    right = page_data["trd"]["summary"]["recall_average"]
    cards.append(
        f"""
        <section class="compare-card highlight">
          <span class="eyebrow">Custom - TRD</span>
          <strong class="{delta_class(left, right)}">{delta_text(left, right)}</strong>
          <small>Recall avg delta for this setting</small>
        </section>
        """
    )
    return "\n".join(cards)


def render_metric_comparison(page_data: dict[str, dict[str, Any]]) -> str:
    custom = page_data["custom"]["summary"]
    trd = page_data["trd"]["summary"]
    rows = [
        ("recall_average", "Recall average", "recall"),
        *[(column, label, group) for column, label, group in METRICS],
        ("total_tokens", "Avg total tokens", "cost"),
    ]
    rendered = []
    for column, label, group in rows:
        left = custom.get(column)
        right = trd.get(column)
        rendered.append(
            f"""
            <tr>
              <td><span class="metric-group">{html.escape(group)}</span>{html.escape(label)}</td>
              <td>{fmt(left, 0) if column == 'total_tokens' else fmt(left)}</td>
              <td>{fmt(right, 0) if column == 'total_tokens' else fmt(right)}</td>
              <td class="{delta_class(left, right)}">{delta_text(left, right)}</td>
            </tr>
            """
        )
    return f"""
      <table>
        <thead>
          <tr>
            <th>metric</th>
            <th>custom</th>
            <th>trd</th>
            <th>delta</th>
          </tr>
        </thead>
        <tbody>{''.join(rendered)}</tbody>
      </table>
    """


def query_map(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["query_id"]: item for item in data["queries"]}


def render_query_comparison(page_data: dict[str, dict[str, Any]]) -> str:
    custom_queries = query_map(page_data["custom"])
    trd_queries = query_map(page_data["trd"])
    rows = []
    for query_id in sorted(set(custom_queries) | set(trd_queries)):
        custom = custom_queries.get(query_id, {})
        trd = trd_queries.get(query_id, {})
        query = custom.get("query") or trd.get("query") or ""
        left = custom.get("recall_average")
        right = trd.get("recall_average")
        rows.append(
            f"""
            <tr>
              <td>{html.escape(query_id)}</td>
              <td>{html.escape(truncate(query, 110))}</td>
              <td class="{score_class(left)}">{fmt(left)}</td>
              <td class="{score_class(right)}">{fmt(right)}</td>
              <td class="{delta_class(left, right)}">{delta_text(left, right)}</td>
              <td>{fmt(custom.get('generation_score_factual_correctness_recall'))} / {fmt(trd.get('generation_score_factual_correctness_recall'))}</td>
              <td>{fmt(custom.get('generation_score_vital_nuggetizer_score'))} / {fmt(trd.get('generation_score_vital_nuggetizer_score'))}</td>
              <td>{fmt(custom.get('retrieval_score_mean_umbrela_score'))} / {fmt(trd.get('retrieval_score_mean_umbrela_score'))}</td>
            </tr>
            """
        )
    return f"""
      <table>
        <thead>
          <tr>
            <th>query_id</th>
            <th>query</th>
            <th>custom recall</th>
            <th>trd recall</th>
            <th>delta</th>
            <th>factual C/T</th>
            <th>vital C/T</th>
            <th>UMBRELA C/T</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    """


def row_map(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("query_id", "")].append(row)
    return dict(grouped)


def render_compact_run(row: dict[str, str] | None, label: str) -> str:
    if row is None:
        return f"""
          <section class="side-run missing">
            <h4>{html.escape(label)}</h4>
            <p>No result for this query.</p>
          </section>
        """

    recall_avg = recall_average(row)
    generated_preview = html.escape(truncate(row.get("generated_answer", ""), 360))
    expected_preview = html.escape(truncate(row.get("generation_score_expected_answer", ""), 260))
    bars = [
        render_metric_bar("Recall average", recall_avg, "featured"),
        render_metric_bar("Factual recall", metric_value(row, "generation_score_factual_correctness_recall")),
        render_metric_bar("Vital nuggets", metric_value(row, "generation_score_vital_nuggetizer_score")),
        render_metric_bar("Nugget assignment", metric_value(row, "generation_score_mean_nugget_assignment_score")),
        render_metric_bar("UMBRELA", metric_value(row, "retrieval_score_mean_umbrela_score")),
    ]
    no_answer = pretty_json(row.get("generation_score_no_answer_score", ""))
    return f"""
      <section class="side-run">
        <div class="side-title">
          <h4>{html.escape(label)}</h4>
          <strong class="pill {score_class(recall_avg)}">{fmt(recall_avg)}</strong>
        </div>
        <div class="metric-grid compact">{''.join(bars)}</div>
        <div class="answer-preview">
          <span>Generated</span>
          <p>{generated_preview or 'N/A'}</p>
        </div>
        <details>
          <summary>No-answer score</summary>
          <pre>{html.escape(no_answer)}</pre>
        </details>
        <details>
          <summary>Expected answer preview</summary>
          <pre>{expected_preview or 'N/A'}</pre>
        </details>
      </section>
    """


def render_detail_comparison(page_data: dict[str, dict[str, Any]]) -> str:
    custom_rows = row_map(page_data["custom"]["rows"])
    trd_rows = row_map(page_data["trd"]["rows"])
    cards = []
    for query_id in sorted(set(custom_rows) | set(trd_rows)):
        custom = custom_rows.get(query_id, [None])[0]
        trd = trd_rows.get(query_id, [None])[0]
        query = (custom or trd or {}).get("query", "")
        custom_avg = recall_average(custom) if custom else None
        trd_avg = recall_average(trd) if trd else None
        cards.append(
            f"""
            <article class="compare-detail" data-query="{html.escape(query_id.lower())}" data-text="{html.escape((query_id + ' ' + query).lower())}">
              <div class="detail-head">
                <div>
                  <span class="eyebrow">{html.escape(query_id)}</span>
                  <h3>{html.escape(query)}</h3>
                </div>
                <strong class="{delta_class(custom_avg, trd_avg)}">delta {delta_text(custom_avg, trd_avg)}</strong>
              </div>
              <div class="side-grid">
                {render_compact_run(custom, "Custom")}
                {render_compact_run(trd, "TRD")}
              </div>
            </article>
            """
        )
    return "\n".join(cards)


def render_combo_section(page_size: str, similarity: str, page_data: dict[str, dict[str, Any]]) -> str:
    return f"""
      <section class="page-section" data-page-size="{html.escape(page_size)}" data-similarity="{html.escape(similarity)}">
        <section class="compare-grid">
          {render_compare_cards(page_data)}
        </section>
        <section class="panel two-col">
          <div>
            <h2>指标对比</h2>
            {render_metric_comparison(page_data)}
          </div>
          <div>
            <h2>Query 对比</h2>
            {render_query_comparison(page_data)}
          </div>
        </section>
        <section class="panel">
          <h2>并排明细</h2>
          <div class="toolbar">
            <select class="queryFilter">
              <option value="">All query_id</option>
              {''.join(f'<option value="{html.escape(item["query_id"].lower())}">{html.escape(item["query_id"])}</option>' for item in page_data["custom"]["queries"])}
            </select>
            <input class="searchBox" type="search" placeholder="Search query text">
          </div>
          <div class="detailCards">{render_detail_comparison(page_data)}</div>
        </section>
      </section>
    """


def render_compare_html(results: dict[str, dict[str, dict[str, dict[str, Any]]]]) -> str:
    similarities = sorted(
        {similarity for similarity_data in results.values() for similarity in similarity_data},
        key=combo_sort_key,
    )
    page_options = "\n".join(
        f'<option value="{html.escape(page_size)}">page-size {html.escape(page_size)}</option>'
        for page_size in sorted(results, key=combo_sort_key)
    )
    similarity_options = "\n".join(
        f'<option value="{html.escape(similarity)}">similarity {html.escape(similarity)}</option>'
        for similarity in similarities
    )
    sections = "\n".join(
        render_combo_section(page_size, similarity, page_data)
        for page_size in sorted(results, key=combo_sort_key)
        for similarity, page_data in sorted(results[page_size].items(), key=lambda item: combo_sort_key(item[0]))
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Eval Results Comparison</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --green: #238636;
      --yellow: #b7791f;
      --red: #c2410c;
      --blue: #1f5eff;
      --purple: #7c3aed;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: center;
      padding: 22px 32px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, .96);
      backdrop-filter: blur(10px);
    }}
    header h1 {{ margin: 0 0 6px; font-size: 26px; }}
    header p {{ margin: 0; color: var(--muted); }}
    main {{ padding: 24px 32px 42px; }}
    select, input {{
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: white;
      min-width: 180px;
    }}
    .selector-row {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .page-section.hidden, .hidden {{ display: none; }}
    .compare-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .compare-card, .panel, .compare-detail, .side-run {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .compare-card {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 16px;
    }}
    .compare-card.highlight {{ border-color: #b8c7ff; background: #f5f7ff; }}
    .compare-card strong {{ display: block; margin: 6px 0; font-size: 30px; }}
    .compare-card small, .eyebrow {{ color: var(--muted); font-size: 12px; }}
    .eyebrow {{ text-transform: uppercase; letter-spacing: .04em; }}
    .card-metrics {{
      display: grid;
      gap: 6px;
      min-width: 132px;
      color: var(--muted);
      font-size: 13px;
    }}
    .panel {{ padding: 18px; margin-bottom: 20px; overflow-x: auto; }}
    .panel h2 {{ margin: 0 0 14px; font-size: 18px; }}
    .two-col {{
      display: grid;
      grid-template-columns: minmax(420px, .8fr) minmax(620px, 1.2fr);
      gap: 18px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 9px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; background: #fafbfc; }}
    .metric-group {{
      display: inline-block;
      min-width: 64px;
      margin-right: 8px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }}
    .compare-detail {{ padding: 16px; margin-bottom: 14px; }}
    .detail-head, .side-title, .metric-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
    }}
    .detail-head h3 {{ margin: 4px 0 14px; font-size: 18px; }}
    .side-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(280px, 1fr));
      gap: 12px;
    }}
    .side-run {{ padding: 14px; }}
    .side-run h4 {{ margin: 0 0 10px; font-size: 16px; }}
    .side-run.missing {{ color: var(--muted); }}
    .pill {{
      border-radius: 999px;
      padding: 5px 9px;
      white-space: nowrap;
      background: #eef2ff;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .metric-grid.compact {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
    .metric-row {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
    }}
    .metric-row.featured {{ border-color: #9eb4ff; background: #f5f7ff; }}
    .metric-head {{ font-size: 13px; }}
    .bar {{ height: 8px; background: #edf0f5; border-radius: 999px; overflow: hidden; margin-top: 8px; }}
    .bar span {{ display: block; height: 100%; border-radius: inherit; }}
    .score-good, .bar .score-good {{ color: var(--green); background: #2da44e; }}
    .score-medium, .bar .score-medium {{ color: var(--yellow); background: #d29922; }}
    .score-low, .bar .score-low {{ color: var(--red); background: #f97316; }}
    .score-missing, .bar .score-missing {{ color: var(--muted); background: #c7ced8; }}
    .delta-up {{ color: var(--green); }}
    .delta-down {{ color: var(--red); }}
    .delta-flat, .delta-missing {{ color: var(--muted); }}
    .answer-preview {{
      border-left: 3px solid var(--line);
      padding-left: 12px;
      margin: 10px 0;
    }}
    .answer-preview span {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .answer-preview p {{ margin: 5px 0 0; line-height: 1.45; }}
    details {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 11px;
      background: #fbfcfe;
      margin-top: 8px;
    }}
    summary {{ cursor: pointer; font-weight: 600; }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
      color: #243447;
    }}
    @media (max-width: 980px) {{
      header {{ display: block; }}
      header select {{ margin-top: 12px; }}
      .two-col, .side-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 720px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .compare-card {{ display: block; }}
      .card-metrics {{ margin-top: 10px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Eval Results Comparison</h1>
      <p>Compare Custom and TRD chunking across page-size and similarity settings.</p>
    </div>
    <div class="selector-row">
      <select id="pageSizeSelect" aria-label="Select page-size">
        {page_options}
      </select>
      <select id="similaritySelect" aria-label="Select similarity">
        {similarity_options}
      </select>
    </div>
  </header>
  <main>{sections}</main>
  <script>
    const pageSizeSelect = document.getElementById('pageSizeSelect');
    const similaritySelect = document.getElementById('similaritySelect');
    const sections = Array.from(document.querySelectorAll('.page-section'));
    const available = new Map();
    sections.forEach(section => {{
      const pageSize = section.dataset.pageSize;
      const similarity = section.dataset.similarity;
      if (!available.has(pageSize)) available.set(pageSize, new Set());
      available.get(pageSize).add(similarity);
    }});

    function syncSimilarityOptions() {{
      const selectedPageSize = pageSizeSelect.value;
      const allowed = available.get(selectedPageSize) || new Set();
      let firstEnabled = null;
      Array.from(similaritySelect.options).forEach(option => {{
        const enabled = allowed.has(option.value);
        option.disabled = !enabled;
        option.hidden = !enabled;
        if (enabled && firstEnabled === null) firstEnabled = option.value;
      }});
      if (!allowed.has(similaritySelect.value) && firstEnabled !== null) {{
        similaritySelect.value = firstEnabled;
      }}
    }}

    function applySelection() {{
      syncSimilarityOptions();
      const selectedPageSize = pageSizeSelect.value;
      const selectedSimilarity = similaritySelect.value;
      sections.forEach(section => {{
        section.classList.toggle(
          'hidden',
          section.dataset.pageSize !== selectedPageSize || section.dataset.similarity !== selectedSimilarity
        );
      }});
    }}

    function wireSectionFilters() {{
      sections.forEach(section => {{
        const queryFilter = section.querySelector('.queryFilter');
        const searchBox = section.querySelector('.searchBox');
        const cards = Array.from(section.querySelectorAll('.compare-detail'));
        function applyFilters() {{
          const query = queryFilter.value;
          const term = searchBox.value.trim().toLowerCase();
          cards.forEach(card => {{
            const queryOk = !query || card.dataset.query === query;
            const textOk = !term || card.dataset.text.includes(term) || card.textContent.toLowerCase().includes(term);
            card.classList.toggle('hidden', !(queryOk && textOk));
          }});
        }}
        queryFilter.addEventListener('change', applyFilters);
        searchBox.addEventListener('input', applyFilters);
      }});
    }}

    pageSizeSelect.addEventListener('change', applySelection);
    similaritySelect.addEventListener('change', applySelection);
    wireSectionFilters();
    applySelection();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an HTML viewer for local eval result CSV files.")
    parser.add_argument(
        "--mode",
        choices=["compare", "single"],
        default="compare",
        help="compare reads data/custom and data/trd cfg files; single renders one CSV.",
    )
    parser.add_argument("--input-csv", default="data/local_eval_results.csv")
    parser.add_argument("--output-html", default="reports/local_eval_results_comparison.html")
    parser.add_argument("--pipeline-run", default=DEFAULT_PIPELINE_RUN, help="Run name under data/eval_runs for compare mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        rows = read_rows(args.input_csv)
        if not rows:
            raise ValueError(f"No rows found in {args.input_csv}")
        output_path.write_text(render_html(rows, args.input_csv), encoding="utf-8")
        print(f"Wrote {len(rows)} rows to {output_path}")
        return

    results = read_compare_inputs(Path.cwd(), args.pipeline_run)
    output_path.write_text(render_compare_html(results), encoding="utf-8")
    total_rows = sum(
        len(dataset_data["rows"])
        for similarity_data in results.values()
        for page_data in similarity_data.values()
        for dataset_data in page_data.values()
    )
    print(f"Wrote comparison report with {total_rows} rows to {output_path}")


if __name__ == "__main__":
    main()
