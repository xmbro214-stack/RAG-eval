import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_rag_eval_pipeline as pipeline


def sample_config(root: str) -> dict:
    return {
        "queries_csv": "data/qa_golden.csv",
        "golden_csv": "data/qa_golden.csv",
        "output": {
            "root": root,
            "run_name": "ragflow_grid",
            "overwrite": False,
        },
        "datasets": [
            {"id": "123", "name": "custom"},
            {"id": "345", "name": "trd"},
        ],
        "grid": {
            "page_sizes": [10, 20],
            "similarity_thresholds": [0.1, 0.25],
        },
        "generation": {
            "retrieval_url": "http://retrieval.test",
            "llm_base_url": "http://llm.test/v1",
            "llm_api_key": "test-key",
            "llm_model": "test-model",
            "top_k": 1024,
            "rerank_id": "reranker",
            "max_tokens": 1000000,
            "repeat_query": 2,
        },
        "evaluation": {
            "chat_base_url": "http://judge.test/v1",
            "chat_api_key": "judge-key",
            "chat_model": "judge-model",
            "embedding_base_url": "http://embed.test/v1",
            "embedding_api_key": "embed-key",
            "embedding_model": "embed-model",
            "k_values": "1,3,5",
        },
    }


class TestRagEvalPipeline(unittest.TestCase):
    def test_load_config_reads_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "eval-cfg.yaml"
            config_path.write_text(
                "queries_csv: data/qa_golden.csv\n"
                "golden_csv: data/qa_golden.csv\n"
                "output:\n"
                f"  root: {tmp}\n"
                "  run_name: yaml_case\n"
                "datasets:\n"
                "  - id: '123'\n"
                "    name: custom\n"
                "grid:\n"
                "  page_sizes: [10]\n"
                "  similarity_thresholds: [0.1]\n"
                "generation:\n"
                "  retrieval_url: http://retrieval.test\n"
                "  llm_base_url: http://llm.test/v1\n"
                "  llm_api_key: test-key\n"
                "  llm_model: test-model\n"
                "evaluation:\n"
                "  chat_base_url: http://judge.test/v1\n"
                "  chat_model: judge-model\n"
                "  embedding_base_url: http://embed.test/v1\n"
                "  embedding_model: embed-model\n",
                encoding="utf-8",
            )
            config = pipeline.load_config(config_path)
            self.assertEqual(config["datasets"][0]["name"], "custom")
            self.assertEqual(config["grid"]["page_sizes"], [10])

    def test_expand_tasks_cross_product_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = pipeline.expand_tasks(sample_config(tmp))
            self.assertEqual(len(tasks), 8)
            first = tasks[0]
            self.assertEqual(first.dataset_id, "123")
            self.assertEqual(first.dataset_name, "custom")
            self.assertEqual(first.page_size, 10)
            self.assertEqual(first.similarity_threshold, 0.1)
            self.assertEqual(first.output_dir.name, "ps10_sim0p1")
            self.assertEqual(tasks[1].output_dir.name, "ps10_sim0p25")

    def test_validate_config_reports_missing_required_values(self):
        config = sample_config("/tmp/out")
        config["generation"].pop("llm_api_key")
        missing = pipeline.validate_config(config, "generate")
        self.assertIn("generation.llm_api_key or generation.llm_api_key_env", missing)

    def test_env_reference_resolves_when_direct_value_missing(self):
        with patch.dict(os.environ, {"PIPELINE_TEST_MODEL": "env-model"}):
            section = {"llm_model_env": "PIPELINE_TEST_MODEL"}
            self.assertEqual(pipeline.config_value(section, "llm_model"), "env-model")

    def test_generation_and_evaluation_commands_include_expected_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = sample_config(tmp)
            task = pipeline.expand_tasks(config)[0]

            gen_cmd = pipeline.build_generation_command(config, task)
            self.assertIn("--dataset-ids", gen_cmd)
            self.assertEqual(gen_cmd[gen_cmd.index("--dataset-ids") + 1], "123")
            self.assertEqual(gen_cmd[gen_cmd.index("--page-size") + 1], "10")
            self.assertEqual(gen_cmd[gen_cmd.index("--similarity-threshold") + 1], "0.1")
            self.assertIn("--rerank-id", gen_cmd)

            eval_cmd = pipeline.build_evaluation_command(config, task)
            self.assertIn("--answers-csv", eval_cmd)
            self.assertEqual(eval_cmd[eval_cmd.index("--answers-csv") + 1], str(task.generated_answers_csv))
            self.assertEqual(eval_cmd[eval_cmd.index("--output-csv") + 1], str(task.eval_result_csv))

    def test_null_optional_values_are_not_added_to_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = sample_config(tmp)
            config["generation"]["max_tokens"] = None
            task = pipeline.expand_tasks(config)[0]

            gen_cmd = pipeline.build_generation_command(config, task)
            self.assertNotIn("--max-tokens", gen_cmd)

    def test_run_pipeline_dry_run_returns_entries_for_requested_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = sample_config(tmp)
            config["datasets"] = [{"id": "123", "name": "custom"}]
            config["grid"] = {"page_sizes": [10], "similarity_thresholds": [0.1]}
            with self.assertLogs("rag_eval.pipeline", level="INFO") as logs:
                entries = pipeline.run_pipeline(config, stage="generate", dry_run=True, overwrite=False)

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["generation_status"], "dry_run")
            self.assertEqual(entries[0]["evaluation_status"], "not_requested")
            output = "\n".join(logs.output)
            self.assertIn("Pipeline started: stage=generate", output)
            self.assertIn("[1/1] Task dataset=custom", output)
            self.assertIn("--max-tokens", output)
            self.assertIn("--llm-api-key <redacted>", output)
            self.assertNotIn("test-key", output)

            manifest_path = Path(tmp) / "ragflow_grid" / "manifest.json"
            self.assertFalse(manifest_path.exists())

    def test_existing_outputs_are_skipped_when_not_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = sample_config(tmp)
            config["datasets"] = [{"id": "123", "name": "custom"}]
            config["grid"] = {"page_sizes": [10], "similarity_thresholds": [0.1]}
            task = pipeline.expand_tasks(config)[0]
            task.output_dir.mkdir(parents=True)
            task.generated_answers_csv.write_text("already here\n", encoding="utf-8")

            entries = pipeline.run_pipeline(config, stage="generate", dry_run=False, overwrite=False)
            self.assertEqual(entries[0]["generation_status"], "skipped_existing")

    def test_eval_stage_only_evaluates_existing_generated_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = sample_config(tmp)
            config["datasets"] = [{"id": "123", "name": "custom"}]
            config["grid"] = {"page_sizes": [10], "similarity_thresholds": [0.1]}
            task = pipeline.expand_tasks(config)[0]
            task.output_dir.mkdir(parents=True)
            task.generated_answers_csv.write_text("query_id,query,query_run,passage_id,passage,generated_answer\n", encoding="utf-8")

            with patch.object(pipeline.subprocess, "run") as mocked_run:
                entries = pipeline.run_pipeline(config, stage="eval", dry_run=False, overwrite=False)

            self.assertEqual(entries[0]["generation_status"], "not_requested")
            self.assertEqual(entries[0]["evaluation_status"], "completed")
            mocked_run.assert_called_once()
            self.assertIn("--answers-csv", mocked_run.call_args.args[0])

    def test_failed_subprocess_is_recorded_in_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = sample_config(tmp)
            config["datasets"] = [{"id": "123", "name": "custom"}]
            config["grid"] = {"page_sizes": [10], "similarity_thresholds": [0.1]}

            with patch.object(
                pipeline,
                "run_command",
                side_effect=subprocess.CalledProcessError(1, ["generate"]),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    pipeline.run_pipeline(config, stage="generate", dry_run=False, overwrite=False)

            manifest_path = Path(tmp) / "ragflow_grid" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["tasks"][0]["generation_status"], "failed")
            self.assertIn("generation_error", manifest["tasks"][0])


if __name__ == "__main__":
    unittest.main()
