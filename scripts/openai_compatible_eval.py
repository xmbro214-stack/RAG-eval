"""CLI wrapper for rag_eval_pipeline.evaluation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_eval_pipeline.evaluation import main


if __name__ == "__main__":
    main()
