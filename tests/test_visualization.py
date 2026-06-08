import csv
import tempfile
import unittest
from pathlib import Path

from rag_eval_pipeline.visualization import (
    read_compare_inputs,
    render_compare_html,
    render_html,
    resolve_compare_run_root,
)


HEADER = [
    "query_id",
    "query",
    "query_run",
    "generated_answer",
    "generation_score_factual_correctness_recall",
    "generation_score_vital_nuggetizer_score",
    "generation_score_mean_nugget_assignment_score",
    "retrieval_score_mean_umbrela_score",
    "retrieval_score_umbrela_scores",
    "total_tokens",
]
UMBRELLA_DETAIL = (
    '{"p1":{"topic_relevance":2,"answer_support":3,"coverage":2,'
    '"directness":1,"has_contradiction":false,"noise_level":"low","final_score":3}}'
)
ROW = ["q1", "What?", "1", "Answer", "0.8", "0.7", "0.6", "0.9", UMBRELLA_DETAIL, "100"]


def write_eval_result(root: Path, dataset: str, combo: str = "ps10_sim0p1") -> None:
    out_dir = root / "data" / "eval_runs" / "ragflow_grid" / dataset / combo
    out_dir.mkdir(parents=True)
    with (out_dir / "eval_result.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerow(ROW)


class TestVisualization(unittest.TestCase):
    def test_reads_pipeline_outputs_and_renders_selectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in ["custom", "trd"]:
                write_eval_result(root, dataset)

            results = read_compare_inputs(root)
            rendered = render_compare_html(results)

            self.assertEqual(sorted(results), ["10"])
            self.assertEqual(sorted(results["10"]), ["0.1"])
            self.assertIn("similaritySelect", rendered)
            self.assertIn('data-similarity="0.1"', rendered)
            self.assertIn("topic_relevance", rendered)
            self.assertIn("final_score", rendered)

    def test_reads_custom_comparison_dataset_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = [("baseline", "Baseline"), ("experiment", "Experiment")]
            for dataset, _ in datasets:
                write_eval_result(root, dataset)

            results = read_compare_inputs(root, datasets=datasets)
            rendered = render_compare_html(results, datasets)

            self.assertEqual(sorted(results["10"]["0.1"]), ["baseline", "experiment"])
            self.assertIn("Baseline - Experiment", rendered)
            self.assertIn("Baseline recall", rendered)
            self.assertIn("Experiment recall", rendered)

    def test_single_report_uses_light_ats_blue_theme(self):
        rows = [dict(zip(HEADER, ROW))]

        rendered = render_html(rows, "data/eval_result.csv")

        self.assertIn("--bg: #f4f9fd;", rendered)
        self.assertIn("--surface: #ffffff;", rendered)
        self.assertIn("--ink: #0b2540;", rendered)
        self.assertNotIn("--bg: #07111f;", rendered)

    def test_single_report_collapses_source_files(self):
        rows = [dict(zip(HEADER, ROW))]

        rendered = render_html(rows, "data/a.csv | data/b.csv")

        self.assertIn('<details class="source-details">', rendered)
        self.assertIn("<summary>查看来源文件 (2)</summary>", rendered)
        self.assertIn("<code>data/a.csv</code>", rendered)
        self.assertIn("<code>data/b.csv</code>", rendered)
        self.assertNotIn("<p>Source: data/a.csv | data/b.csv", rendered)

    def test_single_report_header_omits_ats_mark(self):
        rows = [dict(zip(HEADER, ROW))]

        rendered = render_html(rows, "data/eval_result.csv")

        self.assertIn("<h1>RAG Evaluation</h1>", rendered)
        self.assertNotIn("brand-mark", rendered)
        self.assertNotIn("AT&S logo", rendered)

    def test_compare_input_root_accepts_data_directory(self):
        root = Path("/tmp/project")

        self.assertEqual(
            resolve_compare_run_root(root / "data", "ragflow_grid"),
            root / "data" / "eval_runs" / "ragflow_grid",
        )


if __name__ == "__main__":
    unittest.main()
