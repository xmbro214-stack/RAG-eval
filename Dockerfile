FROM python:3.12-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY rag_eval_pipeline ./rag_eval_pipeline
COPY scripts ./scripts

RUN mkdir -p /app/data/uploaded_datasets /app/data/eval_runs /app/reports /app/logs

COPY data/qa_golden.csv ./data/qa_golden.csv

RUN python -m pip install --upgrade pip \
    && python -m pip install .

EXPOSE 9000

CMD ["python", "-m", "rag_eval_pipeline.api", "--host", "0.0.0.0", "--port", "9000"]
