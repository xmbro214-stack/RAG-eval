"""Summarize recall-oriented metrics from local evaluation results."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict


METRICS = [
    (
        "generation_score_factual_correctness_recall",
        "factual_recall",
    ),
    (
        "generation_score_vital_nuggetizer_score",
        "vital_nugget_recall",
    ),
    (
        "generation_score_mean_nugget_assignment_score",
        "mean_nugget_assignment",
    ),
]


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "p25": percentile(ordered, 0.25),
        "p75": percentile(ordered, 0.75),
        "max": ordered[-1],
    }


def percentile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def score_band(value: float) -> str:
    if value >= 0.8:
        return "good"
    if value >= 0.5:
        return "medium"
    return "low"


def print_metric_table(title: str, stats: dict[str, dict[str, float]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"{'metric':<28} {'n':>3} {'mean':>8} {'median':>8} "
        f"{'p25':>8} {'p75':>8} {'min':>8} {'max':>8}"
    )
    for _, label in METRICS:
        item = stats.get(label, {})
        if not item:
            continue
        print(
            f"{label:<28} {int(item['count']):>3} "
            f"{item['mean']:>8.4f} {item['median']:>8.4f} "
            f"{item['p25']:>8.4f} {item['p75']:>8.4f} "
            f"{item['min']:>8.4f} {item['max']:>8.4f}"
        )


def build_slim_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    slim_rows = []
    for row in rows:
        slim = {
            "query_id": row.get("query_id", ""),
            "query": row.get("query", ""),
            "query_run": row.get("query_run", ""),
        }
        values = []
        for source_col, label in METRICS:
            value = to_float(row.get(source_col, ""))
            slim[label] = "" if value is None else f"{value:.6f}"
            if value is not None:
                values.append(value)
        if values:
            avg = statistics.mean(values)
            slim["recall_metric_avg"] = f"{avg:.6f}"
            slim["recall_metric_band"] = score_band(avg)
        else:
            slim["recall_metric_avg"] = ""
            slim["recall_metric_band"] = ""
        slim_rows.append(slim)
    return slim_rows


def write_csv(path: str, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(args: argparse.Namespace) -> None:
    rows = read_rows(args.input_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.input_csv}")

    slim_rows = build_slim_rows(rows)
    write_csv(args.output_csv, slim_rows)

    overall_stats = {}
    for _, label in METRICS:
        values = [to_float(row[label]) for row in slim_rows]
        overall_stats[label] = describe([v for v in values if v is not None])
    print_metric_table("Overall Recall-Oriented Metrics", overall_stats)

    grouped = defaultdict(list)
    for row in slim_rows:
        grouped[row["query_id"]].append(row)

    print("\nBy Query")
    print("--------")
    print(
        f"{'query_id':<10} {'runs':>4} {'factual':>8} {'vital':>8} "
        f"{'assign':>8} {'avg':>8} query"
    )
    for query_id, group_rows in grouped.items():
        means = {}
        for _, label in METRICS:
            values = [to_float(row[label]) for row in group_rows]
            values = [v for v in values if v is not None]
            means[label] = statistics.mean(values) if values else None
        avg_values = [v for v in means.values() if v is not None]
        avg = statistics.mean(avg_values) if avg_values else None
        query = group_rows[0]["query"]
        print(
            f"{query_id:<10} {len(group_rows):>4} "
            f"{fmt(means['factual_recall']):>8} "
            f"{fmt(means['vital_nugget_recall']):>8} "
            f"{fmt(means['mean_nugget_assignment']):>8} "
            f"{fmt(avg):>8} {query}"
        )

    low_rows = sorted(
        [
            row for row in slim_rows
            if row["recall_metric_avg"] and float(row["recall_metric_avg"]) < args.low_threshold
        ],
        key=lambda row: float(row["recall_metric_avg"]),
    )
    print(f"\nLow Recall Samples (< {args.low_threshold:.2f})")
    print("--------------------------------")
    if not low_rows:
        print("None")
    for row in low_rows:
        print(
            f"{row['query_id']} run {row['query_run']} "
            f"avg={row['recall_metric_avg']} "
            f"factual={row['factual_recall']} "
            f"vital={row['vital_nugget_recall']} "
            f"assign={row['mean_nugget_assignment']} "
            f"- {row['query']}"
        )

    print(f"\nWrote slim metrics CSV to {args.output_csv}")


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize recall-oriented metrics from local_eval_results.csv."
    )
    parser.add_argument("--input-csv", default="data/local_eval_results.csv")
    parser.add_argument("--output-csv", default="data/recall_metrics_summary.csv")
    parser.add_argument("--low-threshold", type=float, default=0.6)
    return parser.parse_args()


if __name__ == "__main__":
    summarize(parse_args())
