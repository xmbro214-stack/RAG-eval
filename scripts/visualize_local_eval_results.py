"""Generate a self-contained HTML viewer for local evaluation results."""

from __future__ import annotations

import argparse
import csv
import html
import json
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
    ("generation_score_citation_f1_score", "Citation F1", "citation"),
    ("generation_score_hallucination_score", "Source support", "support"),
]

DETAIL_COLUMNS = [
    ("retrieval_score_umbrela_scores", "UMBRELA passage scores"),
    ("retrieval_score_precision_metrics", "Retrieval precision/AP/MRR"),
    ("generation_score_autonugget_scores", "Nuggets and assignments"),
    ("generation_score_citation_scores", "Citation scores"),
    ("generation_score_no_answer_score", "No-answer score"),
    ("generation_score_generated_claims", "Generated claims"),
    ("generation_score_expected_claims", "Expected claims"),
    ("generation_score_precision_verdicts", "Precision verdicts"),
    ("generation_score_recall_verdicts", "Recall verdicts"),
]


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
    return summary


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
        ("Citation F1", fmt(summary["generation_score_citation_f1_score"]), "Citation support quality"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an HTML viewer for local eval result CSV files.")
    parser.add_argument("--input-csv", default="data/local_eval_results.csv")
    parser.add_argument("--output-html", default="reports/local_eval_results_viewer.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.input_csv}")
    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(rows, args.input_csv), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
