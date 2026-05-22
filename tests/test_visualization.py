import csv
import tempfile
import unittest
from pathlib import Path

from rag_eval_pipeline.visualization import read_compare_inputs, render_compare_html


class TestVisualization(unittest.TestCase):
    def test_reads_pipeline_outputs_and_renders_selectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = [
                "query_id",
                "query",
                "query_run",
                "generated_answer",
                "generation_score_factual_correctness_recall",
                "generation_score_vital_nuggetizer_score",
                "generation_score_mean_nugget_assignment_score",
                "retrieval_score_mean_umbrela_score",
                "total_tokens",
            ]
            row = ["q1", "What?", "1", "Answer", "0.8", "0.7", "0.6", "0.9", "100"]
            for dataset in ["custom", "trd"]:
                out_dir = root / "data" / "eval_runs" / "ragflow_grid" / dataset / "ps10_sim0p1"
                out_dir.mkdir(parents=True)
                with (out_dir / "eval_result.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerow(row)

            results = read_compare_inputs(root)
            rendered = render_compare_html(results)

            self.assertEqual(sorted(results), ["10"])
            self.assertEqual(sorted(results["10"]), ["0.1"])
            self.assertIn("similaritySelect", rendered)
            self.assertIn('data-similarity="0.1"', rendered)


if __name__ == "__main__":
    unittest.main()
