import io
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pytest

from rag_eval_pipeline import api


def test_slugify_dataset_name_keeps_safe_name():
    assert api.slugify_dataset_name("demo qa v1.csv") == "demo_qa_v1"
    assert api.slugify_dataset_name("../bad/name") == "bad_name"
    assert api.slugify_dataset_name("   ") == "dataset"


def test_parse_and_validate_csv_normalizes_rows():
    content = "query_id,query,expected_answer,extra\n,What is A?,Answer A,ignored\nq2,What is B?,Answer B,ignored\n"

    rows = api.parse_and_validate_csv(content.encode("utf-8"))

    assert rows == [
        {"query_id": "query_1", "query": "What is A?", "expected_answer": "Answer A"},
        {"query_id": "q2", "query": "What is B?", "expected_answer": "Answer B"},
    ]


def test_parse_and_validate_csv_rejects_missing_column():
    content = "query_id,query\nq1,What is A?\n"

    with pytest.raises(api.UploadError) as exc:
        api.parse_and_validate_csv(content.encode("utf-8"))

    assert "expected_answer" in str(exc.value)


def test_parse_and_validate_csv_rejects_blank_required_values():
    content = "query_id,query,expected_answer\nq1,,Answer A\n"

    with pytest.raises(api.UploadError) as exc:
        api.parse_and_validate_csv(content.encode("utf-8"))

    assert "row 1" in str(exc.value)
    assert "query" in str(exc.value)


def test_parse_and_validate_csv_rejects_non_utf8_bytes():
    with pytest.raises(api.UploadError) as exc:
        api.parse_and_validate_csv(b"\xff\xfe\xfa")

    assert "UTF-8" in str(exc.value)


def test_parse_qa_pairs_from_text_reads_english_and_chinese_labels():
    text = """
    Q: What is A?
    A: Answer A

    问题：什么是 B？
    答案：答案 B
    """

    rows = api.parse_qa_pairs_from_text(text)

    assert rows == [
        {"query_id": "query_1", "query": "What is A?", "expected_answer": "Answer A"},
        {"query_id": "query_2", "query": "什么是 B？", "expected_answer": "答案 B"},
    ]


def test_parse_qa_pairs_from_text_rejects_missing_pairs():
    with pytest.raises(api.UploadError) as exc:
        api.parse_qa_pairs_from_text("This PDF has no labeled QA pairs.")

    assert "QA pairs" in str(exc.value)


def test_parse_and_validate_pdf_extracts_qa_rows(monkeypatch):
    def fake_extract_text_from_pdf(content):
        assert content == b"%PDF fake"
        return "Question: What is A?\nAnswer: Answer A\n"

    monkeypatch.setattr(api, "extract_text_from_pdf", fake_extract_text_from_pdf)

    rows = api.parse_and_validate_pdf(b"%PDF fake")

    assert rows == [
        {"query_id": "query_1", "query": "What is A?", "expected_answer": "Answer A"},
    ]


def test_save_dataset_writes_normalized_csv(tmp_path):
    rows = [{"query_id": "q1", "query": "What is A?", "expected_answer": "Answer A"}]

    payload = api.save_dataset(rows, "demo", upload_root=tmp_path)

    assert payload["ok"] is True
    assert payload["dataset_name"] == "demo"
    assert payload["rows"] == 1
    assert payload["columns"] == ["query_id", "query", "expected_answer"]
    assert (tmp_path / "demo.csv").read_text(encoding="utf-8").splitlines() == [
        "query_id,query,expected_answer",
        "q1,What is A?,Answer A",
    ]


def test_save_dataset_appends_suffix_for_duplicate_names(tmp_path):
    rows = [{"query_id": "q1", "query": "What is A?", "expected_answer": "Answer A"}]
    (tmp_path / "demo.csv").write_text("existing\n", encoding="utf-8")

    payload = api.save_dataset(rows, "demo", upload_root=tmp_path)

    assert payload["dataset_name"] == "demo_2"
    assert (tmp_path / "demo_2.csv").exists()


def test_save_dataset_retries_when_exclusive_create_loses_race(tmp_path, monkeypatch):
    rows = [{"query_id": "q1", "query": "What is A?", "expected_answer": "Answer A"}]
    existing_path = tmp_path / "demo.csv"
    existing_path.write_text("existing\n", encoding="utf-8")
    real_unique_dataset_path = api.unique_dataset_path
    calls = []

    def raced_unique_dataset_path(dataset_name, upload_root=api.UPLOAD_ROOT):
        if not calls:
            calls.append("race")
            return "demo", existing_path
        return real_unique_dataset_path(dataset_name, upload_root)

    monkeypatch.setattr(api, "unique_dataset_path", raced_unique_dataset_path)

    payload = api.save_dataset(rows, "demo", upload_root=tmp_path)

    assert payload["dataset_name"] == "demo_2"
    assert existing_path.read_text(encoding="utf-8") == "existing\n"
    assert (tmp_path / "demo_2.csv").read_text(encoding="utf-8").splitlines() == [
        "query_id,query,expected_answer",
        "q1,What is A?,Answer A",
    ]


def test_append_chat_supplement_adds_next_qa_row(tmp_path):
    supplement_csv = tmp_path / "qa_supplement.csv"
    supplement_csv.write_text(
        "query_id,query,expected_answer\nquery_1,Existing question,Existing answer\n",
        encoding="utf-8",
    )

    payload = api.append_chat_supplement(
        "What should we add?",
        "Add this answer.",
        supplement_csv,
    )

    assert payload == {
        "supplemental_path": api.relative_repo_path(supplement_csv),
        "supplemental_query_id": "query_2",
    }
    assert supplement_csv.read_text(encoding="utf-8").splitlines() == [
        "query_id,query,expected_answer",
        "query_1,Existing question,Existing answer",
        "query_2,What should we add?,Add this answer.",
    ]


def test_chat_supplement_default_path_is_separate_from_golden_dataset():
    assert api.CHAT_SUPPLEMENT_CSV_PATH == api.ROOT / "data" / "qa_supplement.csv"
    assert api.CHAT_SUPPLEMENT_CSV_PATH != api.GOLDEN_CSV_PATH


def test_render_upload_page_contains_form_and_brand():
    page = api.render_upload_page(
        result_reports=[
            {
                "name": "local_smoke_eval.html",
                "path": "reports/local_smoke_eval.html",
                "url": "/reports/local_smoke_eval.html",
                "modified": "2026-05-28 09:00",
            }
        ]
    )

    assert "AT&amp;S" in page
    assert 'id="datasetUploadForm"' in page
    assert 'id="manualQaForm"' in page
    assert 'id="manualQaDatasetSelect"' in page
    assert 'id="manualQaDatasetPreview"' not in page
    assert 'name="target_dataset"' in page
    assert 'name="question"' in page
    assert 'name="expected_answer"' in page
    assert 'id="manualQaResult"' in page
    assert 'name="file"' in page
    assert 'name="name"' in page
    assert 'accept=".csv,.pdf,text/csv,application/pdf"' in page
    assert "CSV or PDF" in page
    assert "Saved dataset path" in page
    assert "Saved under data/uploaded_datasets/" in page
    assert "/api/datasets/upload" in page
    assert "/api/datasets/manual-qa" in page
    assert 'id="uploadDialog"' in page
    assert 'id="openUploadDialog"' in page
    assert 'id="backButton"' in page
    assert ">View report<" in page
    assert "viewReport" in page
    assert "selectedReportUrl" in page
    assert "window.location.href = selectedReportUrl" in page
    assert "#176f95" in page
    assert "#1aa6c8" in page
    assert "border-top: 3px solid var(--ats-cyan)" in page
    assert "linear-gradient(180deg, var(--ats-blue-deep), var(--ats-blue-dark) 54%, var(--ats-blue));" in page
    assert "Upload dataset window" in page
    assert "Report analysis" in page
    assert 'id="pageLanguage"' in page
    assert "uiTranslations" in page
    assert "applyPageLanguage" in page
    assert "reportTranslations" in page
    assert "applyReportLanguage" in page
    assert "RAG Evaluation" in page
    assert "RAG-Auswertung" in page
    assert "RAG 评估" in page
    assert "Penilaian RAG" in page
    page_language_order = [
        page.index('value="de">德语</option>'),
        page.index('value="en">英语</option>'),
        page.index('value="zh">中文</option>'),
        page.index('value="ms">马来西亚语</option>'),
    ]
    assert page_language_order == sorted(page_language_order)
    assert 'id="runPipelineButton"' in page
    assert 'id="quickPipelineButton"' not in page
    assert "Quick evaluation" not in page
    assert '"mode": "quick"' not in page
    assert '"mode": "standard"' in page
    assert 'data-running-text="Running..."' in page
    assert "运行中..." in page
    assert 'id="pipelineResult"' in page
    assert "/api/pipeline/run" in page
    assert "/api/pipeline/status" in page
    assert "job.log_tail" not in page
    assert "Latest log" not in page
    assert "job.reports" in page
    assert "HTML report" in page
    assert "refreshReportList(renderedReport" in page
    assert "reports/local_smoke_eval.html" in page
    assert 'id="reportFrame"' in page
    assert "resizeReportFrame" in page
    assert 'scrolling="auto"' in page
    assert 'id="chatWindow"' in page
    assert 'class="report-content-stack"' in page
    assert 'class="report-main"' in page
    assert 'class="chat-window report-chat"' in page
    assert 'class="panel chat-window"' not in page
    assert ".workspace {" in page
    assert "grid-template-columns: minmax(0, 1fr) minmax(300px, 360px);" not in page


def test_render_upload_page_contains_console_navigation_and_compact_job_status():
    page = api.render_upload_page(result_reports=[])

    assert 'class="app-shell"' in page
    assert 'class="sidebar-nav"' in page
    assert 'data-section-target="overview"' in page
    assert 'data-section-target="reports"' in page
    assert 'data-section-target="eval-runs"' in page
    assert 'data-section-target="run-evaluation"' in page
    assert 'data-section-target="datasets"' in page
    assert 'id="jobStatusBar"' in page
    assert 'class="topbar-actions"' in page
    assert 'class="topbar-actions">\n          <div class="job-status-shell">' in page
    assert 'class="job-status-shell"' in page
    assert 'class="job-status-bar"' in page
    assert 'class="job-status-main"' in page
    assert "job-status-details" in page
    assert 'id="jobRail"' not in page
    assert 'class="job-rail"' not in page
    assert 'id="jobProgressBar"' in page
    assert 'id="evalRunsTable"' in page
    assert "/api/eval-runs" in page
    assert "grid-template-columns: 216px minmax(0, 1fr);" in page
    assert "grid-template-columns: 216px minmax(0, 1fr) 324px;" not in page
    assert "width: fit-content;" in page
    assert ".job-status-shell {" in page
    assert ".topbar-actions {" in page
    assert "margin-left: auto;" in page
    assert "padding: 6px 10px;" in page
    assert "border-radius: 999px;" in page
    assert "height: 4px;" in page
    assert "box-shadow: none;" in page
    assert "width: 104px;" in page
    assert "max-width: 104px;" in page
    assert "data-status=\"idle\"" in page
    assert "jobStatusBar.dataset.status = status;" in page
    assert "@keyframes progressFlow" in page
    assert ".job-status-bar[data-status=\"running\"] .progress-fill" in page
    assert ".job-status-bar[data-status=\"cancelling\"] .progress-fill" in page
    assert ".job-status-bar[data-status=\"completed\"] .progress-fill" in page
    assert ".job-status-bar[data-status=\"failed\"] .progress-fill" in page
    assert ">Idle<" in page
    assert "Idle. No background job is running." not in page
    assert "当前没有后台任务运行" not in page


def test_console_navigation_follows_user_workflow_order():
    page = api.render_upload_page(result_reports=[])

    nav_targets = [
        'data-section-target="datasets" data-i18n="navDatasets">Dataset</button>',
        'data-section-target="run-evaluation" data-i18n="navRunEvaluation">Run</button>',
        'data-section-target="eval-runs" data-i18n="navEvalRuns">Records</button>',
        'data-section-target="reports" data-i18n="navReports">Reports</button>',
        'data-section-target="overview" data-i18n="navOverview">Overview</button>',
    ]
    positions = [page.index(target) for target in nav_targets]
    assert positions == sorted(positions)
    assert '<button class="nav-button active" type="button" data-section-target="datasets"' in page
    assert '<section id="datasets" class="console-section active">' in page
    assert '<section id="overview" class="console-section active">' not in page


def test_dataset_section_shows_existing_dataset_inventory():
    page = api.render_upload_page(result_reports=[])

    assert 'id="datasetsTable"' in page
    assert 'data-i18n="currentDatasets">数据列表</' in page
    assert 'id="refreshDatasetsButton"' in page
    assert "/api/datasets" in page
    assert "refreshDatasets" in page
    assert "renderDatasetInventory" in page
    assert "dataset.source === \"config\"" in page
    assert "dataset.path || \"-\"" in page
    assert 'class="dataset-step-title">添加问答</span>' in page
    assert 'data-i18n="manualAdd">手动</' not in page
    assert 'data-i18n="generateAnswerMode"' not in page
    assert 'id="generateAnswerButton"' in page
    assert 'data-i18n="fromReportChat">报告问答</' not in page
    assert 'data-i18n="targetDataset">操作数据集</' in page
    assert 'manualQaForm.addEventListener("submit", async (event)' in page
    assert 'const manualQaForm = document.getElementById("manualQaForm");' in page
    assert 'const manualQaDatasetSelect = document.getElementById("manualQaDatasetSelect");' in page
    assert 'const manualQaButton = manualQaForm ? manualQaForm.querySelector("button[type=\'submit\']") : null;' in page
    assert "renderManualQaDatasetChoices(currentDatasets)" in page
    assert "loadManualQaDatasetPreview()" not in page
    assert 'manualQaDatasetSelect.addEventListener("change", loadManualQaDatasetPreview)' not in page
    assert 'fetch("/api/datasets/preview?path="' in page
    assert 'fetch("/api/datasets/manual-qa"' in page


def test_dataset_section_uses_clean_split_layout_for_manual_qa():
    page = api.render_upload_page(result_reports=[])

    assert 'class="dataset-section-stack"' in page
    assert 'class="dataset-action-header"' not in page
    assert 'class="dataset-flow-card dataset-step-card dataset-step-dataset"' in page
    assert 'class="qa-setup-grid dataset-module-row"' not in page
    assert 'class="dataset-step-toolbar qa-context-bar dataset-module dataset-module-dataset"' in page
    assert 'class="qa-dataset-picker qa-dataset-control"' in page
    assert 'class="qa-dataset-upload-button"' in page
    assert 'data-i18n="targetDataset">操作数据集<' in page
    assert 'class="qa-save-panel dataset-module dataset-module-save"' in page
    assert 'class="manual-qa-row manual-qa-row-top dataset-module dataset-module-dataset"' not in page
    assert 'class="manual-qa-row manual-qa-row-body qa-input-panel dataset-module dataset-module-entry"' in page
    assert 'class="manual-qa-actions manual-qa-save-row"' not in page
    assert 'class="qa-reference-grid"' in page
    assert ".dataset-section-stack {" in page
    assert ".dataset-flow-card {" in page
    assert ".dataset-step-badge {" in page
    assert 'grid-template-areas:' in page
    assert '"body"' in page
    assert '"generate"' in page
    assert '"context context"' not in page
    assert '"save save"' not in page
    assert '"actions actions"' not in page
    assert '"preview preview"' not in page
    assert '"passages passages"' not in page
    assert "grid-template-columns: minmax(0, 1fr);" in page
    assert ".qa-context-bar {" in page
    assert ".qa-dataset-control {" in page
    assert ".qa-dataset-upload-button {" in page
    assert ".qa-save-panel {" in page
    assert ".qa-reference-grid {" in page
    assert ".dataset-list-panel {" in page


def test_dataset_page_separates_each_window_visually():
    page = api.render_upload_page(result_reports=[])

    assert 'class="manual-qa-panel dataset-workbench dataset-flow-card dataset-step-card dataset-step-edit"' in page
    assert 'class="qa-setup-grid dataset-module-row"' not in page
    assert 'class="dataset-step-toolbar qa-context-bar dataset-module dataset-module-dataset"' in page
    assert 'class="qa-dataset-picker qa-dataset-control"' in page
    assert 'class="qa-dataset-upload-button"' in page
    assert 'class="qa-save-panel dataset-module dataset-module-save"' in page
    assert 'class="manual-qa-row manual-qa-row-top dataset-module dataset-module-dataset"' not in page
    assert 'class="qa-mode-selector dataset-module dataset-module-method"' not in page
    assert 'class="manual-qa-row manual-qa-row-body qa-input-panel dataset-module dataset-module-entry"' in page
    assert 'class="qa-reference-grid"' in page
    assert 'id="manualQaPreviewDetails"' not in page
    assert 'class="dataset-preview-details retrieved-passages-details dataset-secondary-panel dataset-secondary-preview"' in page
    assert 'class="qa-candidate-panel dataset-secondary-panel dataset-secondary-candidates"' not in page
    assert 'class="dataset-list-panel dataset-flow-card dataset-step-card dataset-step-list"' in page
    for css_selector in [
        ".dataset-workbench {",
        ".dataset-flow-card {",
        ".dataset-step-heading {",
        ".dataset-step-badge {",
        ".dataset-module {",
        ".dataset-module-dataset {",
        ".dataset-module-method {",
        ".dataset-module-entry {",
        ".dataset-module-save {",
        ".qa-context-bar {",
        ".qa-dataset-upload-button {",
        ".qa-save-panel {",
        ".qa-reference-grid {",
        ".dataset-secondary-panel {",
        ".dataset-secondary-preview {",
        ".dataset-secondary-candidates {",
        ".dataset-secondary-inventory {",
    ]:
        assert css_selector in page


def test_dataset_workbench_uses_flat_simplified_layout():
    page = api.render_upload_page(result_reports=[])

    assert ".dataset-module {\n      min-width: 0;\n      border: 0;\n      border-left: 0;\n      border-radius: 0;\n      background: transparent;" in page
    input_panel_block = page[page.index(".qa-input-panel {"):page.index(".manual-qa-form textarea {")]
    assert "border: 1px solid var(--qa-border-solid);" in input_panel_block
    assert "box-shadow: none;" in input_panel_block
    generate_button_block = page[page.index("#generateAnswerButton {"):page.index("#regenerateAnswerButton,")]
    assert "color: #ffffff;" in generate_button_block
    assert "background: var(--ats-blue-700);" in generate_button_block
    assert ".qa-save-panel button[type='submit'] {\n      min-height: var(--qa-control-height);" in page
    assert ".manual-qa-panel > .section-heading::before" not in page
    assert "border-left: 4px solid var(--qa-accent);" not in page


def test_dataset_page_has_refined_enterprise_saas_visual_treatment():
    page = api.render_upload_page(result_reports=[])

    assert "--qa-surface: var(--ats-blue-50);" in page
    assert "--qa-border-solid: #d8e4f0;" in page
    assert "--qa-card-radius: 8px;" in page
    assert ".dataset-create-grid {\n      display: grid;\n      gap: 18px;\n      min-width: 0;\n      padding: 20px 24px 24px;\n      background: transparent;" in page
    assert ".dataset-workbench {\n      position: relative;" in page
    assert ".dataset-workbench::before {" in page
    assert "background: #d8e4f0;" in page
    assert "box-shadow: var(--qa-shadow);" in page
    assert ".manual-qa-form textarea:focus" in page
    assert "box-shadow: 0 0 0 3px rgba(0, 167, 200, .14);" in page
    assert "#generateAnswerButton:hover" in page
    assert ".qa-save-panel button[type='submit']:hover" in page
    assert ".dataset-list-panel,\n    .dataset-preview-details {" in page


def test_dataset_page_uses_four_step_workbench_from_reference():
    page = api.render_upload_page(result_reports=[])

    for step_number, step_class in [
        ("1", "dataset-step-dataset"),
        ("2", "dataset-step-edit"),
        ("3", "dataset-step-actions"),
        ("4", "dataset-step-list"),
    ]:
        assert f'class="dataset-step-badge">{step_number}</span>' in page
        assert step_class in page
    assert 'class="dataset-file-summary"' in page
    assert 'id="datasetFileName"' in page
    assert 'id="datasetFileMeta"' not in page
    assert 'class="qa-card qa-question-card"' in page
    assert 'class="qa-card qa-answer-card"' in page
    assert 'id="manualQaQuestionCount"' in page
    assert 'id="manualQaAnswerCount"' in page
    assert 'class="dataset-action-card dataset-flow-card dataset-step-card dataset-step-actions"' in page
    assert 'id="datasetSaveStatus"' not in page
    assert 'id="datasetRowsTable"' in page


def test_dataset_selector_step_matches_compact_reference_toolbar():
    page = api.render_upload_page(result_reports=[])

    assert 'class="dataset-step-toolbar qa-context-bar dataset-module dataset-module-dataset"' in page
    assert ".dataset-step-dataset {\n      min-height: 96px;\n      gap: 14px;\n      padding: 18px 24px;" in page
    assert ".dataset-step-toolbar {\n      display: grid;\n      grid-template-columns: minmax(360px, 470px) minmax(220px, 1fr) minmax(160px, 220px);" in page
    assert "padding-left: 0;" in page
    assert "padding-left: 76px;" not in page
    assert ".dataset-step-dataset .dataset-step-toolbar {\n      grid-area: auto;" in page
    assert "justify-self: stretch;" in page
    assert ".dataset-step-dataset .qa-dataset-control {\n      justify-self: start;" in page
    assert ".dataset-step-dataset .dataset-step-heading {\n      align-self: start;" in page
    assert ".dataset-step-dataset .qa-dataset-upload-button {\n      color: var(--ats-blue);" in page
    assert "background: #ffffff;" in page
    assert "box-shadow: none;" in page


def test_dataset_page_polishes_step_alignment_and_list_toolbar():
    page = api.render_upload_page(result_reports=[])

    assert ".dataset-step-edit .manual-qa-row-body {\n      width: 100%;" in page
    assert "grid-area: auto;" in page
    assert ".dataset-step-edit .qa-question-card,\n    .dataset-step-edit .qa-answer-card {\n      width: 100%;" in page
    assert ".dataset-action-main {\n      min-width: 0;" in page
    assert ".dataset-action-row {\n      display: grid;" in page
    assert ".dataset-action-buttons {\n      display: flex;" in page
    assert ".dataset-action-card .qa-reference-grid {\n      display: block;" in page
    assert 'class="dataset-action-main dataset-action-row"' in page
    assert 'class="dataset-action-section dataset-generation-section"' in page
    assert 'class="dataset-action-section dataset-save-section"' in page
    assert 'dataset-action-buttons' in page
    assert 'class="dataset-list-title"' in page
    assert 'class="dataset-list-actions"' in page
    assert '<div class="dataset-inventory-header">' not in page
    assert '<h3 class="section-heading" data-i18n="currentDatasets">数据列表</h3>' not in page


def test_dataset_generation_actions_stack_below_step_title():
    page = api.render_upload_page(result_reports=[])

    assert 'class="dataset-action-main dataset-action-row"' in page
    assert 'class="dataset-action-section dataset-generation-section"' in page
    assert 'class="dataset-action-section dataset-save-section"' in page
    action_card = page[page.index('class="dataset-action-card dataset-flow-card dataset-step-card dataset-step-actions"'):page.index('class="qa-reference-grid"')]
    assert action_card.index('class="dataset-step-heading"') < action_card.index('class="dataset-action-main dataset-action-row"')
    assert action_card.index('class="dataset-action-section dataset-generation-section"') < action_card.index('class="dataset-action-section dataset-save-section"')
    assert 'class="dataset-action-section-copy"' not in action_card
    assert 'class="answer-generation-actions dataset-action-buttons"' in page
    assert 'class="dataset-save-shell"' in page
    assert 'class="dataset-save-state-dot" aria-hidden="true"' not in page
    assert 'class="dataset-save-button-group"' in page
    assert 'class="dataset-save-menu-button" type="button" aria-label="保存选项"' not in page
    assert ".dataset-step-actions {\n      gap: 16px;\n      grid-template-columns: minmax(0, 1fr);\n      grid-template-areas:\n        \"heading\"\n        \"actions\"\n        \"references\";" in page
    assert ".dataset-step-actions > .dataset-step-heading {\n      grid-area: heading;" in page
    assert ".dataset-step-actions > .dataset-action-main {\n      grid-area: actions;" in page
    assert ".dataset-step-actions > .qa-reference-grid {\n      grid-area: references;" in page
    assert ".dataset-action-row {\n      display: grid;\n      grid-template-columns: minmax(360px, auto) minmax(64px, 1fr) minmax(480px, 620px);\n      align-items: center;" in page
    assert ".dataset-action-section {\n      display: flex;\n      justify-content: space-between;\n      align-items: center;" in page
    assert ".dataset-save-shell {\n      display: grid;" in page
    assert ".dataset-save-info-card {\n      display: grid;" not in page
    assert ".dataset-action-card .retrieved-passages-details {\n      width: 100%;\n      max-width: none;" in page
    assert ".dataset-action-card .dataset-preview-summary {\n      min-height: 52px;" in page


def test_dataset_generation_and_save_sections_are_full_width_and_low_noise():
    page = api.render_upload_page(result_reports=[])

    assert ".dataset-action-row {\n      display: grid;\n      grid-template-columns: minmax(360px, auto) minmax(64px, 1fr) minmax(480px, 620px);\n      align-items: center;" in page
    assert ".dataset-action-section {\n      display: flex;\n      justify-content: space-between;\n      align-items: center;" in page
    assert "box-sizing: border-box;" in page
    assert ".dataset-generation-section {\n      flex: 0 0 auto;" in page
    assert ".dataset-generation-section .dataset-action-buttons {\n      width: auto;" in page
    assert ".dataset-save-section {\n      grid-column: 3;\n      justify-self: end;" in page
    assert ".dataset-action-buttons button {\n      min-width: 118px;" in page
    assert "#generateAnswerButton {\n      width: var(--qa-primary-action-width);" in page
    assert ".dataset-save-shell {\n      display: grid;\n      grid-template-columns: var(--qa-primary-action-width);\n      gap: 0;\n      align-items: stretch;\n      justify-content: end;\n      min-width: 0;\n      width: auto;\n      min-height: var(--qa-primary-action-height);\n      padding: 0;\n      border: 0;\n      border-radius: 0;\n      background: transparent;" in page
    assert ".dataset-save-info-card {\n      display: grid;" not in page
    assert ".dataset-save-button-group button[type='submit'] {\n      width: var(--qa-primary-action-width);" in page
    assert ".dataset-list-panel {\n      display: grid;\n      gap: 14px;\n      min-width: 0;\n      padding: 18px 22px;" in page
    assert ".qa-card textarea {\n      min-height: 140px;" in page


def test_dataset_save_panel_anchors_to_right_edge_of_action_row():
    page = api.render_upload_page(result_reports=[])

    assert ".dataset-save-section {\n      grid-column: 3;\n      justify-self: end;" in page
    assert ".dataset-save-section .qa-save-panel {\n      display: flex;\n      justify-content: flex-end;\n      flex: 0 0 auto;\n      width: auto;\n      max-width: none;\n      margin-left: auto;" in page
    assert ".dataset-save-shell {\n      display: grid;\n      grid-template-columns: var(--qa-primary-action-width);\n      gap: 0;\n      align-items: stretch;\n      justify-content: end;\n      min-width: 0;\n      width: auto;\n      min-height: var(--qa-primary-action-height);\n      padding: 0;" in page


def test_dataset_save_ui_uses_compact_submit_button():
    page = api.render_upload_page(result_reports=[])

    save_card = page[page.index('class="dataset-save-shell"'):page.index('class="qa-reference-grid"')]
    assert 'class="dataset-save-info-card"' not in save_card
    assert 'class="manual-qa-save-target"' not in save_card
    assert 'class="dataset-save-button-group"' in save_card
    assert ".dataset-save-shell {\n      display: grid;\n      grid-template-columns: var(--qa-primary-action-width);\n      gap: 0;\n      align-items: stretch;\n      justify-content: end;\n      min-width: 0;\n      width: auto;\n      min-height: var(--qa-primary-action-height);\n      padding: 0;\n      border: 0;\n      border-radius: 0;\n      background: transparent;" in page
    assert ".dataset-save-info-card {\n      display: grid;" not in page
    assert ".manual-qa-save-target small {" not in page
    assert ".dataset-save-button-group {\n      display: flex;\n      align-items: stretch;\n      justify-content: flex-end;\n      justify-self: end;\n      width: var(--qa-primary-action-width);" in page
    assert ".dataset-save-button-group button[type='submit'] {\n      width: var(--qa-primary-action-width);" in page


def test_dataset_final_reference_polishes_minor_visual_details():
    page = api.render_upload_page(result_reports=[])

    assert ".dataset-step-badge {\n      display: inline-grid;" in page
    assert "color: var(--ats-blue);" in page
    assert "background: #eaf2fb;" in page
    assert "box-shadow: 0 6px 14px rgba(23, 111, 149, .13);" in page
    assert ".dataset-preview-summary::after,\n    .qa-candidate-summary::after {\n      content: \"⌄\";" in page
    assert "details[open] > .dataset-preview-summary::after,\n    details[open] > .qa-candidate-summary::after {\n      content: \"⌃\";" in page
    assert ".dataset-view-all-button::after {\n      content: \"›\";" in page
    assert 'class="dataset-save-state-dot" aria-hidden="true">▱</span>' not in page
    assert ".dataset-save-state-dot {\n      display: inline-grid;" not in page
    assert "color: var(--ats-blue);" in page
    assert "--qa-accent-soft: #eef6ff;" in page
    assert 'aria-label=\\"查看当前问答\\" title=\\"查看\\">◎</button>' in page


def test_dataset_final_screen_detail_density_is_polished():
    page = api.render_upload_page(result_reports=[])

    assert ".dataset-create-grid {\n      display: grid;\n      gap: 18px;" in page
    assert ".dataset-step-card {\n      display: grid;\n      gap: 16px;\n      padding: 22px 24px;" in page
    assert ".dataset-step-actions {\n      gap: 16px;" in page
    assert ".dataset-action-row {\n      display: grid;\n      grid-template-columns: minmax(360px, auto) minmax(64px, 1fr) minmax(480px, 620px);\n      align-items: center;" in page
    assert ".dataset-save-section .qa-save-panel {\n      display: flex;\n      justify-content: flex-end;\n      flex: 0 0 auto;\n      width: auto;" in page
    assert ".dataset-save-shell {\n      display: grid;\n      grid-template-columns: var(--qa-primary-action-width);\n      gap: 0;\n      align-items: stretch;\n      justify-content: end;\n      min-width: 0;\n      width: auto;" in page
    assert ".dataset-action-card .dataset-preview-summary {\n      min-height: 52px;\n      padding: 11px 16px;" in page
    assert ".dataset-list-panel {\n      display: grid;\n      gap: 14px;\n      min-width: 0;\n      padding: 18px 22px;" in page
    assert "#datasets .records-table {\n      font-size: 13px;" in page
    assert "#datasets .records-table td {\n      padding: 14px 16px;" in page
    assert ".dataset-row-actions button,\n    .dataset-view-all-button {\n      min-width: 34px;\n      min-height: 34px;" in page


def test_dataset_console_uses_consistent_controls_and_lighter_inventory():
    page = api.render_upload_page(result_reports=[])

    assert "--qa-control-height: 44px;" in page
    assert "--qa-control-radius: 8px;" in page
    assert "#datasets input,\n    #datasets textarea,\n    #datasets select {\n      border-color: #d8e4f0;\n      border-radius: var(--qa-control-radius);" in page
    assert "#datasets button {\n      min-height: var(--qa-control-height);\n      border-radius: var(--qa-control-radius);" in page
    assert ".answer-generation-actions {\n      display: grid;\n      grid-template-columns: var(--qa-primary-action-width) 132px 118px;" in page
    assert "#generateAnswerButton {\n      width: var(--qa-primary-action-width);\n      min-height: var(--qa-primary-action-height);" in page
    assert "#regenerateAnswerButton,\n    #clearGeneratedAnswerButton {\n      min-height: var(--qa-control-height);" in page
    assert "#regenerateAnswerButton {\n      width: 132px;" in page
    assert "#clearGeneratedAnswerButton {\n      width: 118px;" in page
    assert ".dataset-save-button-group button[type='submit'] {\n      width: var(--qa-primary-action-width);\n      min-height: var(--qa-primary-action-height);" in page
    assert "#datasets .table-scroll {\n      border: 1px solid rgba(216, 228, 240, .78);" in page
    assert "#datasets .records-table th {\n      position: sticky;\n      top: 0;\n      z-index: 1;\n      padding: 12px 16px;\n      color: #0b2f5b;\n      background: #eef4f8;" in page
    assert "#datasets .records-table tbody tr:hover td {\n      background: #f7fafc;" in page


def test_dataset_console_matches_enterprise_saas_workbench_spec():
    page = api.render_upload_page(result_reports=[])

    assert "--ats-blue-50: #f2fbfd;" in page
    assert "--qa-card-radius: 8px;" in page
    assert "--qa-button-radius: 8px;" in page
    assert "--qa-border-solid: #d8e4f0;" in page
    assert ".dataset-flow-card {\n      min-width: 0;\n      border: 1px solid var(--qa-border-solid);\n      border-radius: var(--qa-card-radius);" in page
    assert ".dataset-step-dataset {\n      min-height: 96px;" in page
    assert 'class="dataset-step-title">添加问答</span>' in page
    assert 'class="dataset-step-title">编辑问答</span>' in page
    assert '答案（标准答案）' in page
    assert 'placeholder="请输入您想要添加的问题..."' in page
    assert 'placeholder="请输入标准答案（Golden answer）..."' in page
    assert 'class="save-target-title">保存到数据集</span>' not in page
    assert '将当前问答保存到' not in page
    assert 'class="dataset-preview-summary-title">召回片段</span>' in page
    assert '展开查看模型检索到的相关片段信息' in page
    assert '<th data-i18n="modifiedHeader">Modified</th>' not in page
    assert '<tr><td colspan="5" data-i18n="loadingDatasets">Loading datasets...</td></tr>' in page
    assert "const visibleRows = datasetRowsExpanded ? rows : rows.slice(0, 1);" in page
    assert "datasetRowsTableBody.innerHTML = \"<tr><td colspan='5'>\"" in page


def test_dataset_list_view_all_is_real_preview_toggle_without_extra_collapse_affordance():
    page = api.render_upload_page(result_reports=[])

    assert '<section id="datasetInventoryDetails" class="dataset-list-panel dataset-flow-card dataset-step-card dataset-step-list">' in page
    assert '<details id="datasetInventoryDetails"' not in page
    assert '<summary class="dataset-list-summary dataset-step-heading">' not in page
    assert '<div class="dataset-list-summary dataset-step-heading">' in page
    assert ".dataset-list-summary::after" not in page
    assert "details[open] > .dataset-list-summary::after" not in page
    assert 'const datasetViewAllButton = document.querySelector(".dataset-view-all-button");' in page
    assert "let datasetRowsExpanded = false;" in page
    assert "let lastDatasetRowsPayload = null;" in page
    assert "const visibleRows = datasetRowsExpanded ? rows : rows.slice(0, 1);" in page
    assert "datasetViewAllButton.textContent = datasetRowsExpanded ? \"收起\" : \"查看全部\";" in page
    assert 'datasetViewAllButton.addEventListener("click", () => {' in page


def test_retrieved_passages_stay_collapsed_after_generation_and_row_actions_are_clear():
    page = api.render_upload_page(result_reports=[])

    assert 'id="retrievedPassagesDetails" class="dataset-preview-details retrieved-passages-details dataset-secondary-panel dataset-secondary-preview">' in page
    assert "retrievedPassagesDetails.open = true;" not in page
    assert 'aria-label=\\"查看当前问答\\" title=\\"查看\\"' in page
    assert 'aria-label=\\"更多操作\\" title=\\"更多\\"' in page
    assert 'aria-label=\\"View row\\"' not in page
    assert 'aria-label=\\"More\\"' not in page


def test_dataset_actions_use_stable_single_row_toolbar_layout():
    page = api.render_upload_page(result_reports=[])

    assert ".dataset-step-dataset {\n      min-height: 96px;" in page
    assert ".dataset-action-row {\n      display: grid;" in page
    assert ".dataset-action-section {\n      display: flex;\n      justify-content: space-between;\n      align-items: center;" in page
    assert ".dataset-action-buttons {\n      display: flex;" in page
    assert ".dataset-save-shell {\n      display: grid;" in page
    assert ".dataset-save-info-card {\n      display: grid;" not in page
    assert ".dataset-save-button-group {\n      display: flex;" in page
    assert ".dataset-save-button-group button[type='submit']::after" in page
    assert "class=\"dataset-save-menu-button\"" not in page
    assert ".dataset-action-card .dataset-preview-summary {\n      min-height: 52px;" in page
    assert ".dataset-row-actions button,\n    .dataset-view-all-button {\n      min-width: 34px;\n      min-height: 34px;" in page
    assert "writing-mode" not in page


def test_dataset_rows_table_loads_selected_dataset_preview():
    page = api.render_upload_page(result_reports=[])

    assert 'const datasetRowsTable = document.getElementById("datasetRowsTable");' in page
    assert 'const datasetRowsTableBody = datasetRowsTable ? datasetRowsTable.querySelector("tbody") : null;' in page
    assert 'async function loadDatasetRowsPreview()' in page
    assert 'fetch("/api/datasets/preview?path=" + encodeURIComponent(manualQaDatasetSelect.value))' in page
    assert 'renderDatasetRowsPreview(payload)' in page
    assert 'manualQaDatasetSelect.addEventListener("change", () => {' in page
    assert 'loadDatasetRowsPreview();' in page
    assert 'class=\\"dataset-row-question\\"' in page
    assert 'class=\\"dataset-row-actions\\"' in page


def test_dataset_selector_is_compact_inline_control():
    page = api.render_upload_page(result_reports=[])

    assert ".qa-context-bar {\n      grid-area: context;\n      display: grid;\n      grid-template-columns: minmax(0, 1fr) minmax(220px, .72fr) minmax(160px, 220px);" in page
    assert ".dataset-step-toolbar {\n      display: grid;\n      grid-template-columns: minmax(360px, 470px) minmax(220px, 1fr) minmax(160px, 220px);" in page
    assert ".qa-dataset-control {\n      display: grid;\n      grid-template-columns: max-content minmax(260px, 360px);" in page
    assert ".qa-dataset-control > span:first-child {" in page
    assert ".manual-qa-selected-path {\n      min-width: 0;" in page
    assert ".dataset-step-toolbar > .manual-qa-selected-path {\n      display: none;" in page
    assert "white-space: nowrap;" in page
    assert "text-overflow: ellipsis;" in page
    context_block = page[page.index(".qa-context-bar {"):page.index(".qa-dataset-picker {")]
    assert "min-height: 42px;" not in context_block
    assert ".qa-context-bar {\n      grid-area: context;\n      display: grid;\n      grid-template-columns: minmax(280px, 1fr);" not in page


def test_dataset_toolbar_integrates_upload_action_with_selection():
    page = api.render_upload_page(result_reports=[])

    assert '<div class="dataset-action-header">' not in page
    assert '<h2 class="section-heading" data-i18n="uploadDataset">Upload dataset</h2>' not in page
    assert 'id="openUploadDialog" type="button" class="qa-dataset-upload-button" data-i18n="uploadDataset"' in page
    assert page.index('class="dataset-step-toolbar qa-context-bar dataset-module dataset-module-dataset"') < page.index('id="manualQaDatasetSelect"')
    assert page.index('id="manualQaDatasetSelect"') < page.index('id="openUploadDialog"')
    assert page.index('id="openUploadDialog"') < page.index('class="manual-qa-row manual-qa-row-body qa-input-panel dataset-module dataset-module-entry"')


def test_dataset_selector_translation_describes_dataset_operation():
    page = api.render_upload_page(result_reports=[])

    assert 'targetDataset: "Dataset actions"' in page
    assert 'targetDataset: "操作数据集"' in page
    assert 'targetDataset: "Datensatz verwalten"' in page
    assert 'targetDataset: "Urus dataset"' in page
    assert 'targetDataset: "数据集"' not in page


def test_dataset_preview_rows_are_removed_from_manual_qa_page():
    page = api.render_upload_page(result_reports=[])

    assert "function truncateText(value, maxLength = 140)" not in page
    assert 'class=\\"dataset-preview-list\\"' not in page
    assert 'class=\\"dataset-preview-item\\"' not in page
    assert 'class=\\"dataset-preview-answer\\"' not in page
    assert '<table><thead><tr><th>query_id</th>' not in page
    assert ".dataset-preview-list {" not in page
    assert ".dataset-preview-answer {" not in page
    assert "-webkit-line-clamp: 2;" not in page


def test_run_section_allows_clicking_config_datasets_to_select_run_targets():
    page = api.render_upload_page(result_reports=[])

    assert 'id="runDatasetChoices"' in page
    assert 'data-i18n="runDatasets">Run datasets</' in page
    assert "renderRunDatasetChoices" in page
    assert "dataset.source === \"config\"" in page
    assert 'type=\\"checkbox\\"' in page
    assert 'checked=\\"checked\\"' in page
    assert "selectedRunDatasets" in page
    assert 'datasets: selectedRunDatasets()' in page


def test_run_section_allows_clicking_grid_parameters_for_each_run():
    page = api.render_upload_page(result_reports=[])

    assert 'id="runPageSizeSelect"' in page
    assert 'id="runPageSizeCustom"' in page
    assert 'id="runSimilaritySelect"' in page
    assert 'id="runSimilarityCustom"' in page
    assert 'class="run-custom-input"' in page
    assert 'placeholder="1-20" hidden' in page
    assert 'placeholder="0-0.5" hidden' in page
    assert 'const OTHER_RUN_OPTION = "__other__";' in page
    assert '<option value=\\"" + OTHER_RUN_OPTION + "\\">' in page
    assert "otherOption" in page
    assert 'min="1"' in page
    assert 'max="20"' in page
    assert 'min="0"' in page
    assert 'max="0.5"' in page
    assert 'step="1"' in page
    assert 'step="0.01"' in page
    assert 'data-i18n="runPageSizes">Recall chunks</' in page
    assert 'aria-label="Recall chunk preset"' in page
    assert 'runPageSizes: "召回片段"' in page
    assert 'data-i18n="runSimilarities">Similarity</' in page
    assert "renderRunParameterChoices" in page
    assert "toggleRunCustomInput" in page
    assert 'if (sectionId === "run-evaluation") refreshDatasets();' in page
    assert "selectElement.selectedIndex = 0;" in page
    assert "currentValue === OTHER_RUN_OPTION" not in page
    assert "selectedRunPageSizes" in page
    assert "selectedRunSimilarities" in page
    assert "validatedPageSizeValue" in page
    assert "validatedSimilarityValue" in page
    assert 'page_sizes: selectedRunPageSizes()' in page
    assert 'similarity_thresholds: selectedRunSimilarities()' in page


def test_run_parameter_selects_have_clean_focus_borders():
    page = api.render_upload_page(result_reports=[])

    assert "select {" in page
    assert "appearance: none;" not in page
    assert "background: #ffffff;" in page
    assert "cursor: pointer;" in page
    assert "select:focus {" in page
    assert "outline: 3px solid rgba(0, 167, 200, 0.2);" in page
    assert "border-color: var(--ats-cyan);" in page


def test_reports_section_is_labeled_as_report_analysis():
    page = api.render_upload_page(result_reports=[])

    assert 'data-section-target="reports"' in page
    assert 'data-i18n="evaluationReports">Report analysis</' in page
    assert 'aria-label="Report analysis"' in page
    assert "报告分析" in page
    assert "围绕 HTML 报告进行预览和问答分析" not in page
    assert "Evaluation reports" not in page


def test_console_language_switch_updates_all_i18n_marked_elements():
    page = api.render_upload_page(result_reports=[])

    assert 'document.querySelectorAll("[data-i18n]")' in page
    assert 'document.querySelectorAll("[data-i18n-placeholder]")' in page
    assert "element.dataset.i18n" in page
    assert "element.dataset.i18nPlaceholder" in page
    for key in [
        "viewReport",
        "navDatasets",
        "navRunEvaluation",
        "navEvalRuns",
        "navReports",
        "navOverview",
        "evaluationReports",
        "pageLanguage",
        "runEvaluation",
        "uploadDataset",
        "uploadDatasetWindow",
        "close",
        "datasetName",
        "csvOrPdfFile",
        "askModel",
        "question",
        "questionPlaceholder",
        "pipelinePlaceholder",
    ]:
        assert key in page


def test_console_chinese_language_covers_static_and_dynamic_console_copy():
    page = api.render_upload_page(result_reports=[])

    for key in [
        "overview",
        "evalRuns",
        "consoleTitle",
        "latestReport",
        "evalRecords",
        "activeJob",
        "openReports",
        "runStandardEval",
        "refreshReports",
        "refreshRecords",
        "runHeader",
        "datasetHeader",
        "comboHeader",
        "generationHeader",
        "evaluationHeader",
        "secondsHeader",
        "outputHeader",
        "backgroundJob",
        "uploadWaitingPath",
        "uploadSavedUnder",
    ]:
        assert f'data-i18n="{key}"' in page

    for key in [
        "jobIdle",
        "loadingEvalRecords",
        "unableLoadEvalRecords",
        "noEvalRecordsFound",
        "currentTask",
        "htmlReport",
        "pipeline",
        "runId",
        "config",
        "startingPipelineJob",
        "uploadingDataset",
        "uploaded",
        "rows",
        "savedDatasetPath",
        "reportPath",
        "thinking",
    ]:
        assert key in page

    assert "评估记录" in page
    assert "后台任务" in page
    assert "正在加载评估记录" in page
    assert "数据集已上传" in page


def test_console_uses_view_report_button_and_chinese_label():
    page = api.render_upload_page(result_reports=[])

    assert 'id="backButton"' in page
    assert 'data-i18n="viewReport"' in page
    assert ">View report<" in page
    assert "查看报告" in page
    assert "function viewReport()" in page
    assert 'backButton.addEventListener("click", viewReport)' in page


def test_console_buttons_and_forms_have_event_handlers():
    page = api.render_upload_page(result_reports=[])

    assert 'document.querySelectorAll("[data-section-target]")' in page
    assert 'item.addEventListener("click", () => switchSection(item.dataset.sectionTarget))' in page
    assert 'openUploadDialog.addEventListener("click"' in page
    assert "uploadDialog.showModal" in page
    assert 'closeUploadDialog.addEventListener("click", () => uploadDialog.close())' in page
    assert 'runPipelineButton.addEventListener("click", () => startPipeline(runPipelineButton))' in page
    assert 'quickPipelineButton' not in page
    assert 'mode === "quick"' not in page
    assert '"mode": "standard"' in page
    assert 'backButton.addEventListener("click", viewReport)' in page
    assert 'pageLanguage.addEventListener("change", () => applyPageLanguage(pageLanguage.value))' in page
    assert 'document.getElementById("refreshReportsButton").addEventListener("click", () => refreshReportList())' in page
    assert 'document.getElementById("refreshEvalRunsButton").addEventListener("click", () => refreshEvalRuns())' in page
    assert 'form.addEventListener("submit", async (event)' in page
    assert 'askModelForm.addEventListener("submit", async (event)' in page


def test_render_upload_page_contains_unified_qa_editor():
    page = api.render_upload_page(result_reports=[])

    assert 'id="qaModeSelect"' not in page
    assert 'id="qaModeManual"' not in page
    assert 'id="qaModeAi"' not in page
    assert 'id="qaModeReport"' not in page
    assert 'id="aiQaPanel"' not in page
    assert 'id="reportQaPanel"' not in page
    assert 'id="qaCandidatePanel"' not in page
    assert 'id="qaCandidateRows"' not in page
    assert 'id="manualQaQuestionTextarea"' in page
    assert 'id="manualQaAnswerTextarea"' in page
    assert 'id="generateAnswerButton"' in page
    assert 'id="regenerateAnswerButton"' in page
    assert 'id="clearGeneratedAnswerButton"' in page
    assert 'id="saveSelectedQaButton"' not in page
    assert 'id="clearQaCandidatesButton"' not in page
    assert "addQaCandidates" not in page
    assert "renderQaCandidates" not in page


def test_dataset_unified_qa_editor_generates_answer_inside_same_form():
    page = api.render_upload_page(result_reports=[])

    assert 'value="ai" data-i18n="generateAnswerMode"' not in page
    assert 'id="generateAnswerForm"' not in page
    assert 'id="manualQaForm"' in page
    assert 'name="question"' in page
    assert 'id="manualQaAnswerTextarea"' in page
    assert 'id="generateAnswerButton"' in page
    assert 'id="regenerateAnswerButton"' in page
    assert 'id="saveGeneratedAnswerButton"' not in page
    assert 'id="clearGeneratedAnswerButton"' in page
    assert 'id="retrievedPassagesDetails"' in page
    assert 'id="retrievedPassagesList"' in page
    assert 'fetch("/api/datasets/generate-answer"' in page
    assert 'fetch("/api/datasets/regenerate-answer"' in page
    assert 'fetch("/api/datasets/generate-qa"' not in page


def test_dataset_qa_page_uses_progressive_disclosure_for_secondary_tools():
    page = api.render_upload_page(result_reports=[])

    assert 'id="manualQaPreviewDetails"' not in page
    assert 'id="manualQaDatasetPreview"' not in page
    assert 'id="retrievedPassagesDetails"' in page
    assert '<summary class="dataset-preview-summary"' in page
    assert 'id="qaCandidatePanel"' not in page
    assert '<section id="datasetInventoryDetails" class="dataset-list-panel dataset-flow-card dataset-step-card dataset-step-list">' in page
    assert '<div class="dataset-list-summary dataset-step-heading">' in page
    assert 'class="qa-mode-tabs"' not in page


def test_dataset_window_uses_short_plain_labels():
    page = api.render_upload_page(result_reports=[])

    assert 'class="dataset-step-title">添加问答</span>' in page
    assert 'class="dataset-step-title">编辑问答</span>' in page
    assert 'data-i18n="addMethod">' not in page
    assert 'data-i18n="manualAdd">手动<' not in page
    assert 'data-i18n="generateAnswer"' in page
    assert 'data-i18n="fromReportChat">' not in page
    assert 'data-i18n="targetDataset">操作数据集<' in page
    assert 'data-i18n="previewDatasetSummary"' not in page
    assert 'data-i18n="candidateQa">' not in page
    assert 'id="manualQaSaveTarget"' not in page
    assert 'id="datasetSaveStatus" class="save-target-title">保存到数据集</span>' not in page
    assert '将当前问答保存到 <strong></strong>' not in page
    assert 'data-i18n="saveManualQa" data-ready-text="Save Q&A" data-loading-text="Saving...">保存问答<' in page
    assert 'data-i18n="saveSelectedQa">' not in page
    assert 'data-i18n="clearCandidates">' not in page
    assert 'data-i18n="currentDatasets">数据列表<' in page
    assert 'Add QA' not in page
    assert 'Add method' not in page
    assert 'View dataset content' not in page
    assert 'Candidate QA' not in page


def test_dataset_window_translation_keys_cover_all_languages():
    page = api.render_upload_page(result_reports=[])
    required_keys = [
        "currentDatasets",
        "refreshDatasets",
        "targetDataset",
        "saveTargetPrefix",
        "standardAnswer",
        "saveManualQa",
        "addQa",
        "manualAdd",
        "generateAnswer",
        "regenerateAnswer",
        "clearGeneratedAnswer",
        "retrievedPassages",
        "generatedAnswer",
        "generatingAnswer",
        "answerGenerationFailed",
        "sourceText",
        "sourceTextPlaceholder",
        "candidateCount",
        "generateCandidateQa",
        "reportQaHint",
        "addReportAnswerToQa",
        "addedToQaCandidates",
        "datasetHeader",
        "sourceHeader",
        "rows",
        "modifiedHeader",
        "pathHeader",
        "datasetSourceConfig",
        "datasetSourceUploaded",
        "loadingDatasets",
        "unableLoadDatasets",
        "noDatasetsFound",
    ]

    language_markers = {
        "de": "      de: {",
        "en": "      en: {",
        "zh": "      zh: {",
        "ms": "      ms: {",
    }
    marker_positions = {language: page.index(marker) for language, marker in language_markers.items()}
    language_order = sorted(marker_positions.items(), key=lambda item: item[1])
    blocks = {}
    for index, (language, start) in enumerate(language_order):
        end = language_order[index + 1][1] if index + 1 < len(language_order) else page.index("    };\n    const reportTranslations")
        blocks[language] = page[start:end]

    missing = {
        language: [key for key in required_keys if f"{key}:" not in block]
        for language, block in blocks.items()
    }

    assert missing == {"de": [], "en": [], "zh": [], "ms": []}


def test_dataset_add_qa_controls_follow_user_input_order():
    page = api.render_upload_page(result_reports=[])

    assert 'class="dataset-step-toolbar qa-context-bar dataset-module dataset-module-dataset"' in page
    assert 'class="qa-save-panel dataset-module dataset-module-save"' in page
    assert 'class="manual-qa-actions manual-qa-save-row"' not in page
    assert '"body"' in page
    assert '"generate"' in page
    assert '"save save"' not in page

    positions = [
        page.index('class="dataset-step-toolbar qa-context-bar dataset-module dataset-module-dataset"'),
        page.index('class="manual-qa-panel dataset-workbench dataset-flow-card dataset-step-card dataset-step-edit"'),
        page.index('class="manual-qa-row manual-qa-row-body qa-input-panel dataset-module dataset-module-entry"'),
        page.index('class="dataset-action-card dataset-flow-card dataset-step-card dataset-step-actions"'),
        page.index('id="generateAnswerButton"'),
        page.index('class="qa-save-panel dataset-module dataset-module-save"'),
        page.index('class="qa-reference-grid"'),
        page.index('id="datasetInventoryDetails"'),
    ]
    assert positions == sorted(positions)


def test_dataset_qa_textareas_are_tall_enough_for_long_answers():
    page = api.render_upload_page(result_reports=[])

    assert ".manual-qa-form textarea {\n      min-height: 142px;" in page
    assert "resize: vertical;" in page
    assert ".manual-qa-row-body {\n      grid-area: body;\n      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);" in page
    assert ".answer-generation-actions, .dataset-action-buttons, .dataset-action-section { display: grid; grid-template-columns: 1fr; }" in page
    assert ".manual-qa-form textarea { min-height: 132px; }" in page


def test_render_upload_page_calls_generate_and_bulk_qa_endpoints():
    page = api.render_upload_page(result_reports=[])

    assert 'fetch("/api/datasets/generate-answer"' in page
    assert 'fetch("/api/datasets/regenerate-answer"' in page
    assert 'fetch("/api/datasets/generate-qa"' not in page
    assert 'fetch("/api/datasets/bulk-qa"' not in page
    assert "selectedQaRows" not in page
    assert "target_dataset: targetDataset" in page


def test_dataset_generate_answer_javascript_preserves_review_before_save():
    page = api.render_upload_page(result_reports=[])

    assert 'let generatedAnswerPassages = [];' in page
    assert 'function renderRetrievedPassages(passages)' in page
    assert 'generateAnswerButton.addEventListener("click", async ()' in page
    assert 'regenerateAnswerButton.addEventListener("click", async ()' in page
    assert 'manualQaAnswerTextarea.value = payload.expected_answer || "";' in page
    assert 'generatedAnswerPassages = payload.passages || [];' in page
    assert 'current_answer: manualQaAnswerTextarea.value || ""' in page
    assert 'question: formData.get("question") || ""' in page
    assert 'expected_answer: formData.get("expected_answer") || ""' in page
    assert 'manualQaAnswerTextarea.addEventListener("input"' in page


def test_chat_panel_is_temporary_and_does_not_offer_save_actions():
    page = api.render_upload_page(result_reports=[])

    assert 'id="chatSaveActions" class="chat-save-actions" hidden' not in page
    assert 'id="saveChatSupplementButton"' not in page
    assert 'id="discardChatSupplementButton"' not in page
    assert 'let pendingChatSupplement = null;' not in page
    assert 'fetch("/api/chat/save"' not in page
    assert 'payload.supplemental_path ?' not in page


def test_report_chat_can_fill_unified_qa_editor():
    page = api.render_upload_page(result_reports=[])

    assert 'id="addReportAnswerToQaButton"' in page
    assert "let lastReportQuestion = \"\";" in page
    assert "let lastReportAnswer = \"\";" in page
    assert "manualQaQuestionTextarea.value = lastReportQuestion;" in page
    assert "manualQaAnswerTextarea.value = lastReportAnswer;" in page
    assert "addQaCandidates" not in page
    assert 'fetch("/api/chat/save"' not in page


def test_run_section_uses_symmetric_control_grid():
    page = api.render_upload_page(result_reports=[])

    assert 'class="run-control-grid"' in page
    assert 'class="run-submit-panel"' in page
    assert 'class="run-action-row"' in page
    assert 'id="runDatasetChoices"' in page
    assert 'id="runPipelineButton"' in page
    assert 'id="terminatePipelineButton"' in page
    assert 'data-i18n="terminateRun"' in page
    assert 'fetch("/api/pipeline/cancel"' in page
    assert "terminatePipelineButton.disabled = !canTerminate;" in page
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in page
    assert ".run-submit-panel button {" in page


def test_console_layout_keeps_text_inside_controls_and_panels():
    page = api.render_upload_page(result_reports=[])

    assert "font-family: -apple-system, BlinkMacSystemFont" in page
    assert "button {" in page
    assert "display: inline-flex;" in page
    assert "white-space: normal;" in page
    assert "overflow-wrap: anywhere;" in page
    assert "text-wrap: pretty;" in page
    assert ".topbar > div { min-width: 0; }" in page
    assert ".nav-button {" in page
    assert "align-items: center;" in page
    assert ".overview-card strong, .overview-activity-strip strong {" in page
    assert ".table-scroll {" in page
    assert "overflow-x: auto;" in page
    assert ".records-table th, .records-table td {" in page
    assert ".dialog-titlebar {" in page
    assert "flex-wrap: wrap;" in page


def test_sidebar_navigation_labels_are_centered():
    page = api.render_upload_page(result_reports=[])
    nav_button_css = re.search(r"\.nav-button \{(?P<body>.*?)\n    \}", page, re.S)

    assert nav_button_css
    css = nav_button_css.group("body")
    assert "display: flex;" in css
    assert "align-items: center;" in css
    assert "justify-content: flex-start;" in css
    assert "text-align: left;" in css


def test_records_table_cells_are_vertically_centered():
    page = api.render_upload_page(result_reports=[])
    records_cell_css = re.search(
        r"\.records-table th, \.records-table td \{(?P<body>.*?)\n    \}",
        page,
        re.S,
    )

    assert records_cell_css
    css = records_cell_css.group("body")
    assert "vertical-align: middle;" in css
    assert "vertical-align: top;" not in css


def test_eval_runs_table_renders_output_open_file_button():
    page = api.render_upload_page(result_reports=[])

    assert 'class=\\"output-action\\"' in page
    assert 'target=\\"_blank\\"' in page
    assert 'rel=\\"noopener\\"' in page
    assert 'title=\\"" + escapeHTML(run.eval_result_csv || "") + "\\"' in page
    assert 'aria-label=\\"" + escapeHTML(t("openOutputFile") + ": " + (run.eval_result_csv || "")) + "\\"' in page
    assert 't("openOutputFile")' in page
    assert "run.eval_result_url" in page
    assert 'class=\\"output-path\\"' not in page
    assert 'escapeHTML(run.eval_result_csv || "-") + "</small>' not in page
    assert "打开文件" in page


def test_eval_runs_report_action_opens_same_page_with_cache_busted_url():
    page = api.render_upload_page(result_reports=[])

    assert "run.report_url" in page
    assert 'class=\\"output-action report-action\\"' in page
    assert 'href=\\"" + escapeHTML(run.report_url) + "\\"' in page
    assert 'report-action\\" href=\\"" + escapeHTML(run.report_url) + "\\" target=' not in page


def test_console_overview_records_and_run_modules_follow_reviewed_flow():
    page = api.render_upload_page(result_reports=[])

    assert 'class="overview-board"' in page
    assert 'class="overview-summary-grid"' in page
    assert 'class="overview-activity-strip"' in page
    assert 'class="overview-card overview-count-card"' in page
    assert 'class="overview-status-panel"' not in page
    assert 'class="overview-action-panel"' not in page
    assert 'class="metric-card"' not in page
    assert 'id="overviewLatestRun"' in page
    assert 'data-i18n="latestEval"' in page
    assert "updateOverviewLatestRun" in page
    assert "payload.runs[0]" in page
    assert 'status-pill status-' in page
    assert "statusBadge(run.generation_status)" in page
    assert "statusBadge(run.evaluation_status)" in page
    assert 'class=\\"combo-stack\\"' in page
    assert 'class=\\"row-actions\\"' in page
    assert "run.report_url" in page
    assert 'class="advanced-panel"' not in page
    assert 'data-i18n="advancedConfig"' not in page
    assert 'data-i18n="configPath"' not in page
    assert 'data-i18n="stage"' not in page
    assert "最近评估" in page


def test_console_removes_nonessential_explanatory_chrome():
    page = api.render_upload_page(result_reports=[])

    assert 'class="subtitle"' not in page
    assert 'data-i18n="consoleNote"' not in page
    assert 'data-i18n="overviewNote"' not in page
    assert 'data-i18n="reportsNote"' not in page
    assert 'data-i18n="evalRunsNote"' not in page
    assert 'data-i18n="runEvaluationNote"' not in page
    assert 'data-i18n="uploadDatasetNote"' not in page
    assert 'data-i18n="backgroundJobNote"' not in page
    assert 'data-i18n="uploadWindowNote"' not in page
    assert 'data-i18n="askModelNote"' not in page
    assert 'id="jobLogTail"' not in page
    assert 'class="log-box"' not in page
    assert 'class="advanced-panel"' not in page


def test_report_section_uses_clean_preview_and_chat_workspace():
    page = api.render_upload_page(
        result_reports=[
            {
                "name": "ragflow_standard_eval.html",
                "path": "reports/ragflow_standard_eval.html",
                "url": "/reports/ragflow_standard_eval.html?v=1",
                "modified": "2026-06-02 10:00",
            }
        ]
    )

    assert 'class="report-workspace"' in page
    assert 'class="report-control-strip"' in page
    assert 'class="report-picker"' in page
    assert 'class="report-body-grid"' in page
    assert 'class="report-preview-area"' in page
    assert 'class="report-chat-area"' in page
    assert 'class="report-path report-path-inline"' not in page
    assert 'id="reportPath" hidden' in page
    assert ".report-body-grid {" in page
    assert ".report-chat-area {" in page
    assert ".report-frame {" in page
    assert "border: 1px solid var(--line);" in page
    assert 'class="panel chat-window"' not in page


def test_report_analysis_layout_is_balanced_at_desktop_scale():
    page = api.render_upload_page(result_reports=[])

    assert ".report-panel {" in page
    assert "max-width: 1320px;" in page
    assert "justify-self: center;" in page
    assert "grid-template-columns: minmax(620px, 1fr) minmax(300px, 330px);" in page
    assert "height: clamp(520px, calc(100vh - 248px), 760px);" in page
    assert 'reportFrame.style.height = "clamp(520px, calc(100vh - 248px), 760px)";' in page
    assert "overflow: auto;" in page
    assert ".report-chat-area {" in page


def test_job_progress_counts_finished_entries():
    job = {
        "status": "running",
        "entries": [
            {
                "dataset_name": "custom",
                "output_dir": "x/ps5_sim0p1",
                "generation_status": "completed",
                "evaluation_status": "completed",
            },
            {
                "dataset_name": "trd",
                "output_dir": "x/ps5_sim0p1",
                "generation_status": "completed",
                "evaluation_status": "not_requested",
            },
        ],
        "expected_tasks": 4,
        "last_log": "[3/4] Task dataset=trd (id=345), page_size=5, similarity=0.2, output_dir=x",
    }

    progress = api.job_progress(job)

    assert progress == {
        "total_tasks": 4,
        "completed_tasks": 1,
        "percent": 25.0,
        "current_task": "trd / ps5_sim0p1",
    }


def test_list_eval_runs_reads_pipeline_manifests(tmp_path):
    run_root = tmp_path / "ragflow_standard"
    run_root.mkdir(parents=True)
    eval_csv = run_root / "custom" / "ps5_sim0p1" / "eval_result.csv"
    eval_csv.parent.mkdir(parents=True)
    eval_csv.write_text("query_id\nq1\n", encoding="utf-8")
    manifest = {
        "updated_at": "2026-06-02T08:00:00+0800",
        "tasks": [
            {
                "dataset_name": "custom",
                "page_size": 5,
                "similarity_threshold": 0.1,
                "generation_status": "completed",
                "evaluation_status": "completed",
                "generation_seconds": 2.5,
                "evaluation_seconds": 7.5,
                "eval_result_csv": str(eval_csv),
            }
        ],
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    runs = api.list_eval_runs(tmp_path, reports_root=tmp_path / "reports")

    assert runs == [
        {
            "run_name": "ragflow_standard",
            "updated_at": "2026-06-02T08:00:00+0800",
            "dataset_name": "custom",
            "page_size": 5,
            "similarity_threshold": 0.1,
            "generation_status": "completed",
            "evaluation_status": "completed",
            "generation_seconds": 2.5,
            "evaluation_seconds": 7.5,
            "eval_result_csv": api.relative_repo_path(eval_csv),
            "eval_result_url": f"/eval-output?path={quote(api.relative_repo_path(eval_csv), safe='')}",
            "report_path": "",
            "report_url": "",
            "combo": "ps5_sim0p1",
        }
    ]
    page = api.render_upload_page()
    language_order = [
        page.index('value="de">德语</option>'),
        page.index('value="en">英语</option>'),
        page.index('value="zh">中文</option>'),
        page.index('value="ms">马来西亚语</option>'),
    ]
    assert language_order == sorted(language_order)
    assert "Model answer will appear here." in page
    assert "Each answer is appended to data/qa_golden.csv." not in page
    assert "Saved QA supplement" not in page
    assert 'fetch("/api/chat/save"' not in page
    assert "/api/chat" in page
    assert "refreshReportList" in page


def test_list_eval_runs_includes_matching_report_link(tmp_path):
    eval_runs_root = tmp_path / "eval_runs"
    reports_root = tmp_path / "reports"
    run_root = eval_runs_root / "ragflow_standard"
    run_root.mkdir(parents=True)
    reports_root.mkdir()
    eval_csv = run_root / "custom" / "ps5_sim0p1" / "eval_result.csv"
    eval_csv.parent.mkdir(parents=True)
    eval_csv.write_text("query_id\nq1\n", encoding="utf-8")
    (reports_root / "ragflow_standard_eval.html").write_text("<html>report</html>", encoding="utf-8")
    (run_root / "manifest.json").write_text(
        json.dumps(
            {
                "updated_at": "2026-06-02T08:00:00+0800",
                "tasks": [
                    {
                        "dataset_name": "custom",
                        "page_size": 5,
                        "similarity_threshold": 0.1,
                        "generation_status": "completed",
                        "evaluation_status": "completed",
                        "eval_result_csv": str(eval_csv),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    runs = api.list_eval_runs(eval_runs_root, reports_root=reports_root)

    assert runs[0]["report_path"] == api.relative_repo_path(reports_root / "ragflow_standard_eval.html")
    assert runs[0]["report_url"].startswith("/reports/ragflow_standard_eval.html?v=")


def test_default_upload_root_matches_documented_dataset_path():
    assert api.UPLOAD_ROOT == api.ROOT / "data" / "uploaded_datasets"
    assert api.relative_repo_path(api.UPLOAD_ROOT / "demo.csv") == "data/uploaded_datasets/demo.csv"


def test_list_datasets_reads_config_and_uploaded_csvs(tmp_path):
    config_path = tmp_path / "eval-cfg.yaml"
    queries_csv = tmp_path / "qa_golden.csv"
    queries_csv.write_text("query_id,query,expected_answer\nquery_1,a,b\n", encoding="utf-8")
    config_path.write_text(
        f'queries_csv: "{queries_csv.as_posix()}"\ngolden_csv: "{queries_csv.as_posix()}"\ndatasets:\n  - id: "123"\n    name: custom\n  - id: "345"\n    name: trd\n',
        encoding="utf-8",
    )
    upload_root = tmp_path / "uploaded"
    upload_root.mkdir()
    uploaded = upload_root / "support_qa.csv"
    uploaded.write_text("query_id,query,expected_answer\nq1,a,b\nq2,c,d\n", encoding="utf-8")

    datasets = api.list_datasets(upload_root=upload_root, config_path=config_path)

    assert datasets == [
        {
            "name": "custom",
            "source": "config",
            "dataset_id": "123",
            "rows": 1,
            "path": api.relative_repo_path(queries_csv),
            "modified": "",
        },
        {
            "name": "trd",
            "source": "config",
            "dataset_id": "345",
            "rows": 1,
            "path": api.relative_repo_path(queries_csv),
            "modified": "",
        },
        {
            "name": "support_qa",
            "source": "uploaded",
            "dataset_id": "",
            "rows": 2,
            "path": api.relative_repo_path(uploaded),
            "modified": datetime.fromtimestamp(uploaded.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        },
    ]


def test_list_datasets_includes_default_golden_csv_for_qa_operations(tmp_path):
    datasets = api.list_datasets(upload_root=tmp_path / "missing", config_path=api.DEFAULT_PIPELINE_CONFIG)

    assert {
        "name": "qa_golden",
        "source": "config",
        "dataset_id": "",
        "rows": api.count_csv_rows(api.GOLDEN_CSV_PATH),
        "path": "data/qa_golden.csv",
        "modified": "",
        "runnable": False,
    } in datasets


def test_run_dataset_choices_ignore_non_runnable_config_datasets():
    page = api.render_upload_page(result_reports=[])

    assert 'dataset.source === "config" && dataset.runnable !== false' in page


def test_list_result_reports_finds_html_reports(tmp_path):
    report = tmp_path / "local_smoke_eval.html"
    report.write_text("<html>report</html>", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    reports = api.list_result_reports(tmp_path)

    assert reports[0]["url"].startswith("/reports/local_smoke_eval.html?v=")
    assert reports == [
        {
            "name": "local_smoke_eval.html",
            "path": api.relative_repo_path(report),
            "url": reports[0]["url"],
            "modified": reports[0]["modified"],
        }
    ]


def test_render_upload_page_shows_empty_report_state():
    page = api.render_upload_page(result_reports=[])

    assert "No HTML reports found under reports/." in page


def test_extract_report_context_strips_html_and_limits_text(tmp_path):
    report = tmp_path / "report.html"
    report.write_text("<html><body><h1>Report</h1><script>ignore()</script><p>Metric A</p></body></html>", encoding="utf-8")

    context = api.extract_report_context("/reports/report.html", tmp_path, limit=20)

    assert context == "Report Metric A"


def test_chat_completion_url_accepts_base_or_full_endpoint():
    assert api.chat_completions_url("http://localhost:8000/v1") == "http://localhost:8000/v1/chat/completions"
    assert api.chat_completions_url("http://localhost:8000/v1/chat/completions") == "http://localhost:8000/v1/chat/completions"


def test_normalize_response_language_defaults_and_accepts_supported_values():
    assert api.normalize_response_language("de") == ("de", "German")
    assert api.normalize_response_language("en") == ("en", "English")
    assert api.normalize_response_language("zh") == ("zh", "Chinese")
    assert api.normalize_response_language("ms") == ("ms", "Malay")
    assert api.normalize_response_language("unknown") == ("de", "German")


def build_multipart_body(boundary, parts):
    body = bytearray()
    for headers, content in parts:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        for header_name, header_value in headers.items():
            body.extend(f"{header_name}: {header_value}\r\n".encode("utf-8"))
        body.extend(b"\r\n")
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body)


def test_parse_content_type_boundary_rejects_missing_boundary():
    with pytest.raises(api.UploadError) as exc:
        api.parse_content_type_boundary("multipart/form-data")

    assert "boundary" in str(exc.value)


def test_parse_multipart_form_reads_name_and_file():
    boundary = "upload-boundary"
    csv_body = b"query_id,query,expected_answer\nq1,What is A?,Answer A\n"
    body = build_multipart_body(
        boundary,
        [
            (
                {"Content-Disposition": 'form-data; name="name"'},
                b"Support QA",
            ),
            (
                {
                    "Content-Disposition": 'form-data; name="file"; filename="demo.csv"',
                    "Content-Type": "text/csv",
                },
                csv_body,
            ),
        ],
    )

    form = api.parse_multipart_form(body, boundary)

    assert form["name"] == "Support QA"
    assert form["file"].filename == "demo.csv"
    assert form["file"].content == csv_body


class DummyUploadHandler(api.DatasetUploadHandler):
    def __init__(self):
        self.rfile = io.BytesIO()
        self.wfile = io.BytesIO()
        self.headers = {}
        self.upload_root = api.UPLOAD_ROOT
        self.reports_root = api.REPORTS_ROOT
        self.chat_supplement_csv_path = api.CHAT_SUPPLEMENT_CSV_PATH
        self.pipeline_jobs = {}
        self.status = None
        self.sent_headers = []
        self.ended = False

    def send_response(self, code, message=None):
        self.status = code

    def send_header(self, keyword, value):
        self.sent_headers.append((keyword, value))

    def end_headers(self):
        self.ended = True


def test_write_json_sets_status_content_type_and_body():
    handler = DummyUploadHandler()

    api.DatasetUploadHandler.write_json(handler, {"ok": True}, status=201)

    assert handler.status == 201
    assert ("Content-Type", "application/json; charset=utf-8") in handler.sent_headers
    assert handler.ended is True
    assert json.loads(handler.wfile.getvalue().decode("utf-8")) == {"ok": True}


def test_write_html_sets_status_content_type_and_body():
    handler = DummyUploadHandler()
    html = api.render_upload_page()

    handler.write_html(html, status=202)

    assert handler.status == 202
    assert ("Content-Type", "text/html; charset=utf-8") in handler.sent_headers
    assert handler.ended is True
    assert html.encode("utf-8") in handler.wfile.getvalue()


def test_write_method_not_allowed_sets_allow_header():
    handler = DummyUploadHandler()

    handler.write_method_not_allowed()

    assert handler.status == 405
    assert ("Allow", "GET, POST") in handler.sent_headers
    assert json.loads(handler.wfile.getvalue().decode("utf-8")) == {
        "ok": False,
        "error": "Method not allowed",
    }


def test_do_head_uses_json_method_not_allowed_response():
    handler = DummyUploadHandler()

    handler.do_HEAD()

    assert handler.status == 405
    assert ("Allow", "GET, POST") in handler.sent_headers
    assert json.loads(handler.wfile.getvalue().decode("utf-8")) == {
        "ok": False,
        "error": "Method not allowed",
    }


def test_do_get_serves_report_html(tmp_path):
    (tmp_path / "local_smoke_eval.html").write_text("<html><body>Report</body></html>", encoding="utf-8")
    handler = DummyUploadHandler()
    handler.reports_root = tmp_path
    handler.path = "/reports/local_smoke_eval.html"

    handler.do_GET()

    assert handler.status == 200
    assert ("Content-Type", "text/html; charset=utf-8") in handler.sent_headers
    html = handler.wfile.getvalue().decode("utf-8")
    assert "Report" in html
    assert 'class="rag-eval-main-return"' in html
    assert 'href="/datasets"' in html
    assert "返回主界面" in html


def test_inject_report_main_button_appends_button_without_body():
    html = api.inject_report_main_button("<html>Report</html>")

    assert html.endswith("</html>")
    assert 'href="/datasets"' in html
    assert "返回主界面" in html


def test_do_get_rejects_report_path_traversal(tmp_path):
    handler = DummyUploadHandler()
    handler.reports_root = tmp_path
    handler.path = "/reports/../README.md"

    handler.do_GET()

    assert handler.status == 404


def test_do_get_returns_reports_json(tmp_path):
    (tmp_path / "local_smoke_eval.html").write_text("<html>Report</html>", encoding="utf-8")
    handler = DummyUploadHandler()
    handler.reports_root = tmp_path
    handler.path = "/api/reports"

    handler.do_GET()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload["ok"] is True
    assert payload["reports"][0]["path"].endswith("local_smoke_eval.html")


def test_do_get_returns_eval_runs_json(tmp_path):
    run_root = tmp_path / "ragflow_standard"
    run_root.mkdir(parents=True)
    (run_root / "manifest.json").write_text(
        json.dumps(
            {
                "updated_at": "2026-06-02T08:00:00+0800",
                "tasks": [
                    {
                        "dataset_name": "custom",
                        "page_size": 5,
                        "similarity_threshold": 0.1,
                        "generation_status": "completed",
                        "evaluation_status": "completed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    handler = DummyUploadHandler()
    handler.eval_runs_root = tmp_path
    handler.path = "/api/eval-runs"

    handler.do_GET()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload["ok"] is True
    assert payload["runs"][0]["run_name"] == "ragflow_standard"


def test_do_get_returns_datasets_json(tmp_path):
    config_path = tmp_path / "eval-cfg.yaml"
    config_path.write_text(
        'datasets:\n  - id: "123"\n    name: custom\n'
        "grid:\n  page_sizes: [5, 10]\n  similarity_thresholds: [0.1, 0.2]\n",
        encoding="utf-8",
    )
    upload_root = tmp_path / "uploaded"
    upload_root.mkdir()
    (upload_root / "demo.csv").write_text("query_id,query,expected_answer\nq1,a,b\n", encoding="utf-8")
    handler = DummyUploadHandler()
    handler.upload_root = upload_root
    handler.pipeline_config_path = config_path
    handler.path = "/api/datasets"

    handler.do_GET()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload["ok"] is True
    assert [dataset["name"] for dataset in payload["datasets"]] == ["custom", "demo"]
    assert payload["run_options"] == {"page_sizes": [5, 10], "similarity_thresholds": [0.1, 0.2]}


def test_do_get_returns_dataset_preview_json(tmp_path):
    dataset_csv = tmp_path / "support_qa.csv"
    dataset_csv.write_text(
        "query_id,query,expected_answer\n"
        "query_1,What is A?,Answer A\n"
        "query_2,What is B?,Answer B\n",
        encoding="utf-8",
    )
    handler = DummyUploadHandler()
    handler.upload_root = tmp_path
    handler.path = f"/api/datasets/preview?path={quote(dataset_csv.as_posix(), safe='')}"

    handler.do_GET()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload == {
        "ok": True,
        "path": api.relative_repo_path(dataset_csv),
        "rows": 2,
        "preview_rows": [
            {"query_id": "query_1", "query": "What is A?", "expected_answer": "Answer A"},
            {"query_id": "query_2", "query": "What is B?", "expected_answer": "Answer B"},
        ],
    }


def test_do_get_serves_eval_output_csv(tmp_path):
    output_csv = tmp_path / "ragflow_standard" / "custom" / "ps5_sim0p1" / "eval_result.csv"
    output_csv.parent.mkdir(parents=True)
    output_csv.write_text("query_id,score\nq1,1\n", encoding="utf-8")
    handler = DummyUploadHandler()
    handler.eval_runs_root = tmp_path
    handler.path = f"/eval-output?path={quote(output_csv.as_posix(), safe='')}"

    handler.do_GET()

    assert handler.status == 200
    assert ("Content-Type", "text/csv; charset=utf-8") in handler.sent_headers
    assert handler.wfile.getvalue().decode("utf-8") == "query_id,score\nq1,1\n"


def test_do_get_rejects_eval_output_outside_eval_runs_root(tmp_path):
    outside_csv = tmp_path.parent / "outside_eval_result.csv"
    outside_csv.write_text("query_id\nq1\n", encoding="utf-8")
    handler = DummyUploadHandler()
    handler.eval_runs_root = tmp_path
    handler.path = f"/eval-output?path={quote(outside_csv.as_posix(), safe='')}"

    try:
        handler.do_GET()
    finally:
        outside_csv.unlink(missing_ok=True)

    assert handler.status == 404


def test_start_pipeline_job_runs_runner_and_records_completion(tmp_path):
    calls = []
    job_store = {}

    class ImmediateThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            self.target()

    def fake_runner(config_path, stage, dry_run, overwrite):
        calls.append((config_path, stage, dry_run, overwrite))
        return [{"dataset_name": "mock", "generation_status": "completed"}]

    payload = api.start_pipeline_job(
        config_path=tmp_path / "eval-cfg.yaml",
        stage="eval",
        dry_run=True,
        overwrite=False,
        job_store=job_store,
        runner=fake_runner,
        thread_factory=ImmediateThread,
    )

    assert payload["ok"] is True
    assert payload["status"] == "completed"
    assert calls == [(tmp_path / "eval-cfg.yaml", "eval", True, False)]
    assert job_store[payload["run_id"]]["entries"] == [{"dataset_name": "mock", "generation_status": "completed"}]


def test_start_pipeline_job_records_log_tail(tmp_path):
    job_store = {}

    class ImmediateThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            self.target()

    def fake_runner(config_path, stage, dry_run, overwrite):
        logging.getLogger("rag_eval.pipeline").info("Task 1 started")
        logging.getLogger("rag_eval.evaluate").info("AutoNugget generation scoring")
        return []

    payload = api.start_pipeline_job(
        config_path=tmp_path / "eval-cfg.yaml",
        stage="eval",
        dry_run=False,
        overwrite=False,
        job_store=job_store,
        runner=fake_runner,
        thread_factory=ImmediateThread,
    )

    job = job_store[payload["run_id"]]
    assert job["status"] == "completed"
    assert job["last_log"] == "AutoNugget generation scoring"
    assert job["log_tail"][-2:] == ["Task 1 started", "AutoNugget generation scoring"]


def test_render_pipeline_reports_writes_html_from_eval_entries(tmp_path):
    eval_csv = tmp_path / "data" / "eval_runs" / "ragflow_quick" / "custom" / "ps10_sim0p1" / "eval_result.csv"
    eval_csv.parent.mkdir(parents=True)
    eval_csv.write_text(
        "query_id,query,query_run,generated_answer,"
        "generation_score_factual_correctness_recall,"
        "generation_score_vital_nuggetizer_score,"
        "generation_score_mean_nugget_assignment_score,"
        "retrieval_score_mean_umbrela_score,total_tokens\n"
        "query_1,What is SM94?,1,SM94 answer,1,1,1,3,100\n",
        encoding="utf-8",
    )

    reports = api.render_pipeline_reports(
        [
            {
                "evaluation_status": "completed",
                "eval_result_csv": str(eval_csv),
            }
        ],
        reports_root=tmp_path / "reports",
    )

    report_path = tmp_path / "reports" / "ragflow_quick_eval.html"
    assert reports[0]["name"] == "ragflow_quick_eval.html"
    assert reports[0]["path"] == api.relative_repo_path(report_path)
    assert reports[0]["url"] == "/reports/ragflow_quick_eval.html"
    assert reports[0]["eval_result_csv"] == api.relative_repo_path(eval_csv)
    assert "RAG Evaluation" in report_path.read_text(encoding="utf-8")
    assert "SM94 answer" in report_path.read_text(encoding="utf-8")


def test_start_pipeline_job_records_rendered_reports(tmp_path):
    job_store = {}

    class ImmediateThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            self.target()

    entries = [{"eval_result_csv": str(tmp_path / "eval_result.csv"), "evaluation_status": "completed"}]

    def fake_runner(config_path, stage, dry_run, overwrite):
        return entries

    def fake_report_renderer(rendered_entries):
        assert rendered_entries is entries
        return [{"path": "reports/ragflow_quick_eval.html", "url": "/reports/ragflow_quick_eval.html"}]

    payload = api.start_pipeline_job(
        config_path=tmp_path / "eval-cfg.yaml",
        stage="all",
        dry_run=False,
        overwrite=False,
        job_store=job_store,
        runner=fake_runner,
        thread_factory=ImmediateThread,
        report_renderer=fake_report_renderer,
    )

    job = job_store[payload["run_id"]]
    assert job["status"] == "completed"
    assert job["reports"] == [{"path": "reports/ragflow_quick_eval.html", "url": "/reports/ragflow_quick_eval.html"}]


def test_start_pipeline_job_redacts_secret_flags_in_failure(tmp_path):
    job_store = {}

    class ImmediateThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            self.target()

    def fake_runner(config_path, stage, dry_run, overwrite):
        raise RuntimeError("--chat-api-key secret-chat --embedding-api-key secret-embed")

    payload = api.start_pipeline_job(
        config_path=tmp_path / "eval-cfg.yaml",
        stage="eval",
        dry_run=False,
        overwrite=False,
        job_store=job_store,
        runner=fake_runner,
        thread_factory=ImmediateThread,
    )

    job = job_store[payload["run_id"]]
    assert job["status"] == "failed"
    assert "secret-chat" not in job["error"]
    assert "secret-embed" not in job["error"]
    assert "--chat-api-key <redacted>" in job["error"]
    assert "--embedding-api-key <redacted>" in job["error"]


def test_create_quick_pipeline_config_limits_matrix_and_repeat_query(tmp_path, monkeypatch):
    queries_csv = tmp_path / "qa_golden.csv"
    rows = [
        "query_id,query,expected_answer",
        *[f"query_{idx},Question {idx}?,Answer {idx}" for idx in range(1, 13)],
    ]
    queries_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    base_config = tmp_path / "eval-cfg.yaml"
    base_config.write_text(
        json.dumps(
            {
                "queries_csv": str(queries_csv),
                "golden_csv": str(queries_csv),
                "output": {"root": "data/eval_runs", "run_name": "ragflow_grid", "overwrite": False},
                "datasets": [{"id": "123", "name": "custom"}, {"id": "345", "name": "trd"}],
                "grid": {"page_sizes": [10, 20], "similarity_thresholds": [0.1, 0.2]},
                "generation": {"repeat_query": 4, "max_workers": 2, "llm_base_url": "https://real-llm/v1"},
                "evaluation": {
                    "max_workers": 2,
                    "chat_base_url": "https://real-judge/v1",
                    "embedding_base_url": "https://real-embed/v1",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "QUICK_PIPELINE_CONFIG_ROOT", tmp_path / "server-configs")

    quick_config_path = api.create_quick_pipeline_config(base_config, "run-1")
    quick_config = json.loads(quick_config_path.read_text(encoding="utf-8"))

    assert quick_config_path.name == "quick_run-1.yaml"
    assert quick_config["output"]["run_name"] == "ragflow_quick"
    assert quick_config["output"]["overwrite"] is True
    assert quick_config["datasets"] == [{"id": "123", "name": "custom"}]
    assert quick_config["grid"] == {"page_sizes": [10], "similarity_thresholds": [0.1]}
    assert quick_config["generation"]["repeat_query"] == 1
    assert quick_config["generation"]["max_workers"] == 1
    assert quick_config["generation"]["llm_base_url"] == "http://127.0.0.1:8011/v1"
    assert quick_config["evaluation"]["max_workers"] == 1
    assert quick_config["evaluation"]["chat_base_url"] == "http://127.0.0.1:8011/v1"
    assert quick_config["evaluation"]["embedding_base_url"] == "http://127.0.0.1:8011/v1"
    assert quick_config["evaluation"]["k_values"] == "1"
    quick_rows = Path(quick_config["queries_csv"]).read_text(encoding="utf-8").splitlines()
    assert quick_rows == rows[:11]
    assert quick_config["golden_csv"] == quick_config["queries_csv"]


def test_create_standard_pipeline_config_preserves_full_matrix_and_real_models(tmp_path, monkeypatch):
    queries_csv = tmp_path / "qa_golden.csv"
    queries_csv.write_text(
        "query_id,query,expected_answer\n"
        "query_1,What is SM94?,SM94 answer\n"
        "query_2,What is CU64?,CU64 answer\n",
        encoding="utf-8",
    )
    base_config = tmp_path / "eval-cfg.yaml"
    base_config.write_text(
        json.dumps(
            {
                "queries_csv": str(queries_csv),
                "golden_csv": str(queries_csv),
                "output": {"root": "data/eval_runs", "run_name": "ragflow_grid", "command_timeout_seconds": 300},
                "datasets": [{"id": "123", "name": "custom"}, {"id": "345", "name": "trd"}],
                "grid": {"page_sizes": [10, 20], "similarity_thresholds": [0.1, 0.2]},
                "generation": {"repeat_query": 4, "max_workers": 2, "llm_base_url": "https://real-llm/v1"},
                "evaluation": {
                    "max_workers": 2,
                    "chat_base_url": "https://real-judge/v1",
                    "embedding_base_url": "https://real-embed/v1",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "QUICK_PIPELINE_CONFIG_ROOT", tmp_path / "server-configs")

    standard_config_path = api.create_standard_pipeline_config(base_config, "run-2")
    standard_config = json.loads(standard_config_path.read_text(encoding="utf-8"))

    assert standard_config_path.name == "standard_run-2.yaml"
    assert standard_config["output"]["run_name"] == "ragflow_standard"
    assert standard_config["output"]["overwrite"] is True
    assert standard_config["output"]["command_timeout_seconds"] == 3600
    assert standard_config["datasets"] == [{"id": "123", "name": "custom"}, {"id": "345", "name": "trd"}]
    assert standard_config["grid"] == {"page_sizes": [10, 20], "similarity_thresholds": [0.1, 0.2]}
    assert standard_config["generation"]["repeat_query"] == 4
    assert standard_config["generation"]["llm_base_url"] == "https://real-llm/v1"
    assert standard_config["evaluation"]["chat_base_url"] == "https://real-judge/v1"
    assert standard_config["evaluation"]["embedding_base_url"] == "https://real-embed/v1"
    assert standard_config["queries_csv"] == str(queries_csv)
    assert standard_config["golden_csv"] == str(queries_csv)


def test_create_standard_pipeline_config_can_filter_selected_datasets(tmp_path, monkeypatch):
    base_config = tmp_path / "eval-cfg.yaml"
    base_config.write_text(
        json.dumps(
            {
                "output": {"root": "data/eval_runs", "run_name": "ragflow_grid"},
                "datasets": [{"id": "123", "name": "custom"}, {"id": "345", "name": "trd"}],
                "grid": {"page_sizes": [5], "similarity_thresholds": [0.1]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "QUICK_PIPELINE_CONFIG_ROOT", tmp_path / "server-configs")

    standard_config_path = api.create_standard_pipeline_config(base_config, "run-3", ["trd"])
    standard_config = json.loads(standard_config_path.read_text(encoding="utf-8"))

    assert standard_config["datasets"] == [{"id": "345", "name": "trd"}]


def test_create_standard_pipeline_config_can_filter_selected_grid_parameters(tmp_path, monkeypatch):
    base_config = tmp_path / "eval-cfg.yaml"
    base_config.write_text(
        json.dumps(
            {
                "output": {"root": "data/eval_runs", "run_name": "ragflow_grid"},
                "datasets": [{"id": "123", "name": "custom"}],
                "grid": {"page_sizes": [5, 10, 20], "similarity_thresholds": [0.1, 0.2]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "QUICK_PIPELINE_CONFIG_ROOT", tmp_path / "server-configs")

    standard_config_path = api.create_standard_pipeline_config(
        base_config,
        "run-grid",
        selected_page_sizes=[10, 20],
        selected_similarity_thresholds=[0.2],
    )
    standard_config = json.loads(standard_config_path.read_text(encoding="utf-8"))

    assert standard_config["grid"] == {"page_sizes": [10, 20], "similarity_thresholds": [0.2]}


def test_create_standard_pipeline_config_accepts_custom_page_size(tmp_path, monkeypatch):
    base_config = tmp_path / "eval-cfg.yaml"
    base_config.write_text(
        json.dumps(
            {
                "output": {"root": "data/eval_runs", "run_name": "ragflow_grid"},
                "datasets": [{"id": "123", "name": "custom"}],
                "grid": {"page_sizes": [5, 10], "similarity_thresholds": [0.1, 0.2]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "QUICK_PIPELINE_CONFIG_ROOT", tmp_path / "server-configs")

    standard_config_path = api.create_standard_pipeline_config(
        base_config,
        "run-custom-grid",
        selected_page_sizes=[2],
        selected_similarity_thresholds=[0.1],
    )
    standard_config = json.loads(standard_config_path.read_text(encoding="utf-8"))

    assert standard_config["grid"] == {"page_sizes": [2], "similarity_thresholds": [0.1]}


def test_grid_parameter_validation_rejects_values_outside_ui_ranges():
    with pytest.raises(api.UploadError):
        api.normalize_page_sizes([0])
    with pytest.raises(api.UploadError):
        api.normalize_page_sizes([21])
    with pytest.raises(api.UploadError):
        api.normalize_similarity_thresholds([-0.1])
    with pytest.raises(api.UploadError):
        api.normalize_similarity_thresholds([0.6])

    assert api.normalize_page_sizes(["1", "20"]) == [1, 20]
    assert api.normalize_similarity_thresholds(["0", "0.5"]) == [0.0, 0.5]


def test_create_quick_pipeline_config_can_filter_selected_datasets(tmp_path, monkeypatch):
    queries_csv = tmp_path / "qa_golden.csv"
    queries_csv.write_text("query_id,query,expected_answer\nq1,a,b\n", encoding="utf-8")
    base_config = tmp_path / "eval-cfg.yaml"
    base_config.write_text(
        json.dumps(
            {
                "queries_csv": str(queries_csv),
                "golden_csv": str(queries_csv),
                "output": {"root": "data/eval_runs", "run_name": "ragflow_grid"},
                "datasets": [{"id": "123", "name": "custom"}, {"id": "345", "name": "trd"}],
                "grid": {"page_sizes": [5], "similarity_thresholds": [0.1]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "QUICK_PIPELINE_CONFIG_ROOT", tmp_path / "server-configs")

    quick_config_path = api.create_quick_pipeline_config(base_config, "run-4", ["trd"])
    quick_config = json.loads(quick_config_path.read_text(encoding="utf-8"))

    assert quick_config["datasets"] == [{"id": "345", "name": "trd"}]


def test_create_quick_pipeline_config_uses_first_selected_grid_parameter(tmp_path, monkeypatch):
    queries_csv = tmp_path / "qa_golden.csv"
    queries_csv.write_text("query_id,query,expected_answer\nq1,a,b\n", encoding="utf-8")
    base_config = tmp_path / "eval-cfg.yaml"
    base_config.write_text(
        json.dumps(
            {
                "queries_csv": str(queries_csv),
                "golden_csv": str(queries_csv),
                "output": {"root": "data/eval_runs", "run_name": "ragflow_grid"},
                "datasets": [{"id": "123", "name": "custom"}],
                "grid": {"page_sizes": [5, 10, 20], "similarity_thresholds": [0.1, 0.2]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "QUICK_PIPELINE_CONFIG_ROOT", tmp_path / "server-configs")

    quick_config_path = api.create_quick_pipeline_config(
        base_config,
        "run-grid",
        selected_page_sizes=[10, 20],
        selected_similarity_thresholds=[0.2],
    )
    quick_config = json.loads(quick_config_path.read_text(encoding="utf-8"))

    assert quick_config["grid"] == {"page_sizes": [10], "similarity_thresholds": [0.2]}


def test_do_post_pipeline_run_starts_job(tmp_path, monkeypatch):
    handler = DummyUploadHandler()
    handler.path = "/api/pipeline/run"
    body = json.dumps(
        {
            "config": str(tmp_path / "eval-cfg.yaml"),
            "stage": "eval",
            "dry_run": True,
            "overwrite": False,
        }
    ).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    def fake_start_pipeline_job(config_path, stage, dry_run, overwrite, job_store=None):
        assert config_path == tmp_path / "eval-cfg.yaml"
        assert stage == "eval"
        assert dry_run is True
        assert overwrite is False
        assert job_store is handler.pipeline_jobs
        return {"ok": True, "run_id": "run-1", "status": "running"}

    monkeypatch.setattr(api, "start_pipeline_job", fake_start_pipeline_job)

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload == {"ok": True, "run_id": "run-1", "status": "running"}


def test_do_post_pipeline_run_quick_mode_uses_quick_config(tmp_path, monkeypatch):
    handler = DummyUploadHandler()
    handler.path = "/api/pipeline/run"
    body = json.dumps(
        {
            "config": str(tmp_path / "eval-cfg.yaml"),
            "mode": "quick",
            "stage": "all",
            "dry_run": False,
            "overwrite": False,
            "datasets": ["trd"],
            "page_sizes": [10],
            "similarity_thresholds": [0.2],
        }
    ).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    quick_config = tmp_path / "quick.yaml"

    def fake_create_quick_pipeline_config(
        base_config_path,
        run_id,
        selected_datasets=None,
        selected_page_sizes=None,
        selected_similarity_thresholds=None,
    ):
        assert base_config_path == tmp_path / "eval-cfg.yaml"
        assert run_id == "quick"
        assert selected_datasets == ["trd"]
        assert selected_page_sizes == [10]
        assert selected_similarity_thresholds == [0.2]
        return quick_config

    def fake_start_pipeline_job(config_path, stage, dry_run, overwrite, job_store=None):
        assert config_path == quick_config
        assert stage == "all"
        assert dry_run is False
        assert overwrite is True
        assert job_store is handler.pipeline_jobs
        return {"ok": True, "run_id": "run-quick", "status": "running"}

    monkeypatch.setattr(api, "create_quick_pipeline_config", fake_create_quick_pipeline_config)
    monkeypatch.setattr(api, "start_pipeline_job", fake_start_pipeline_job)

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload == {"ok": True, "run_id": "run-quick", "status": "running"}


def test_do_post_pipeline_run_standard_mode_uses_standard_config(tmp_path, monkeypatch):
    handler = DummyUploadHandler()
    handler.path = "/api/pipeline/run"
    body = json.dumps(
        {
            "config": str(tmp_path / "eval-cfg.yaml"),
            "mode": "standard",
            "stage": "all",
            "dry_run": False,
            "overwrite": False,
            "datasets": ["custom"],
            "page_sizes": [5, 10],
            "similarity_thresholds": [0.1],
        }
    ).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    standard_config = tmp_path / "standard.yaml"

    def fake_create_standard_pipeline_config(
        base_config_path,
        run_id,
        selected_datasets=None,
        selected_page_sizes=None,
        selected_similarity_thresholds=None,
    ):
        assert base_config_path == tmp_path / "eval-cfg.yaml"
        assert run_id == "standard"
        assert selected_datasets == ["custom"]
        assert selected_page_sizes == [5, 10]
        assert selected_similarity_thresholds == [0.1]
        return standard_config

    def fake_start_pipeline_job(config_path, stage, dry_run, overwrite, job_store=None):
        assert config_path == standard_config
        assert stage == "all"
        assert dry_run is False
        assert overwrite is True
        assert job_store is handler.pipeline_jobs
        return {"ok": True, "run_id": "run-standard", "status": "running"}

    monkeypatch.setattr(api, "create_standard_pipeline_config", fake_create_standard_pipeline_config)
    monkeypatch.setattr(api, "start_pipeline_job", fake_start_pipeline_job)

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload == {"ok": True, "run_id": "run-standard", "status": "running"}


def test_do_post_pipeline_run_standard_mode_checks_retrieval_before_start(tmp_path, monkeypatch):
    handler = DummyUploadHandler()
    handler.path = "/api/pipeline/run"
    body = json.dumps(
        {
            "config": str(tmp_path / "eval-cfg.yaml"),
            "mode": "standard",
            "stage": "all",
            "dry_run": False,
            "overwrite": False,
            "datasets": ["custom"],
            "page_sizes": [2],
            "similarity_thresholds": [0.1],
        }
    ).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    standard_config = tmp_path / "standard.yaml"
    standard_config.write_text(
        json.dumps({"generation": {"retrieval_url": "http://localhost:9380/api/v1/retrieval"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(api, "create_standard_pipeline_config", lambda *args, **kwargs: standard_config)
    monkeypatch.setattr(api, "check_pipeline_retrieval_service", lambda config_path: "召回服务不可达: http://localhost:9380/api/v1/retrieval")

    def fail_start_pipeline_job(*args, **kwargs):
        raise AssertionError("pipeline job should not start when retrieval is unreachable")

    monkeypatch.setattr(api, "start_pipeline_job", fail_start_pipeline_job)

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 400
    assert payload["ok"] is False
    assert "召回服务不可达" in payload["error"]


def test_do_post_pipeline_run_accepts_custom_page_size(tmp_path, monkeypatch):
    handler = DummyUploadHandler()
    handler.path = "/api/pipeline/run"
    body = json.dumps(
        {
            "config": str(tmp_path / "eval-cfg.yaml"),
            "mode": "standard",
            "stage": "all",
            "dry_run": False,
            "overwrite": False,
            "datasets": ["custom"],
            "page_sizes": [2],
            "similarity_thresholds": [0.1],
        }
    ).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    standard_config = tmp_path / "standard.yaml"

    def fake_create_standard_pipeline_config(
        base_config_path,
        run_id,
        selected_datasets=None,
        selected_page_sizes=None,
        selected_similarity_thresholds=None,
    ):
        assert selected_page_sizes == [2]
        assert selected_similarity_thresholds == [0.1]
        return standard_config

    def fake_start_pipeline_job(config_path, stage, dry_run, overwrite, job_store=None):
        assert config_path == standard_config
        return {"ok": True, "run_id": "run-custom", "status": "running"}

    monkeypatch.setattr(api, "create_standard_pipeline_config", fake_create_standard_pipeline_config)
    monkeypatch.setattr(api, "start_pipeline_job", fake_start_pipeline_job)

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload == {"ok": True, "run_id": "run-custom", "status": "running"}


def test_do_get_pipeline_status_returns_job():
    handler = DummyUploadHandler()
    handler.pipeline_jobs = {
        "run-1": {
            "run_id": "run-1",
            "status": "completed",
            "config": "scripts/eval-cfg.yaml",
            "entries": [{"dataset_name": "mock"}],
        }
    }
    handler.path = "/api/pipeline/status?run_id=run-1"

    handler.do_GET()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload["ok"] is True
    assert payload["job"]["status"] == "completed"
    assert payload["job"]["entries"] == [{"dataset_name": "mock"}]


def test_do_post_pipeline_cancel_marks_running_job_cancelling():
    handler = DummyUploadHandler()
    handler.pipeline_jobs = {"run-1": {"run_id": "run-1", "status": "running", "progress": {"percent": 40}}}
    handler.path = "/api/pipeline/cancel"
    body = json.dumps({"run_id": "run-1"}).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload["ok"] is True
    assert payload["job"]["status"] == "cancelling"
    assert payload["job"]["cancel_requested"] is True


def test_start_pipeline_job_marks_cancelled_when_runner_observes_cancel(tmp_path):
    job_store = {}

    class ImmediateThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            self.target()

    def fake_runner(config_path, stage, dry_run, overwrite, cancel_check):
        job = next(iter(job_store.values()))
        job["cancel_requested"] = True
        assert cancel_check() is True
        raise RuntimeError("runner stopped")

    payload = api.start_pipeline_job(
        config_path=tmp_path / "eval-cfg.yaml",
        stage="all",
        dry_run=False,
        overwrite=False,
        job_store=job_store,
        runner=fake_runner,
        thread_factory=ImmediateThread,
    )

    job = job_store[payload["run_id"]]
    assert job["status"] == "cancelled"
    assert job["error"] == "Cancelled by user."


def test_do_post_chat_returns_answer_and_report_list(tmp_path, monkeypatch):
    (tmp_path / "local_smoke_eval.html").write_text("<html>Report</html>", encoding="utf-8")
    supplement_csv = tmp_path / "qa_supplement.csv"
    handler = DummyUploadHandler()
    handler.reports_root = tmp_path
    handler.chat_supplement_csv_path = supplement_csv
    handler.path = "/api/chat"
    body = json.dumps({"question": "What changed?", "report_url": "/reports/local_smoke_eval.html", "language": "zh"}).encode(
        "utf-8"
    )
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    def fake_answer_chat_question(question, report_url, reports_root, language):
        assert question == "What changed?"
        assert report_url == "/reports/local_smoke_eval.html"
        assert reports_root == tmp_path
        assert language == "zh"
        return {"answer": "Recall improved.", "model": "mock-chat", "report_path": "reports/local_smoke_eval.html"}

    monkeypatch.setattr(api, "answer_chat_question", fake_answer_chat_question)

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload["ok"] is True
    assert payload["answer"] == "Recall improved."
    assert "supplemental_path" not in payload
    assert "supplemental_query_id" not in payload
    assert payload["reports"][0]["path"].endswith("local_smoke_eval.html")
    assert not supplement_csv.exists()


def test_do_post_chat_save_writes_supplement_csv(tmp_path):
    supplement_csv = tmp_path / "qa_supplement.csv"
    handler = DummyUploadHandler()
    handler.chat_supplement_csv_path = supplement_csv
    handler.path = "/api/chat/save"
    body = json.dumps({"question": "What changed?", "answer": "Recall improved."}).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload == {
        "ok": True,
        "supplemental_path": api.relative_repo_path(supplement_csv),
        "supplemental_query_id": "query_1",
    }
    assert supplement_csv.read_text(encoding="utf-8").splitlines() == [
        "query_id,query,expected_answer",
        "query_1,What changed?,Recall improved.",
    ]


def test_parse_qa_candidates_accepts_json_array():
    candidates = api.parse_qa_candidates(
        '[{"question":"What is SM94?","expected_answer":"SM94 is an SR dent."}]'
    )

    assert candidates == [
        {"question": "What is SM94?", "expected_answer": "SM94 is an SR dent."}
    ]


def test_parse_qa_candidates_rejects_invalid_json():
    with pytest.raises(api.UploadError) as exc:
        api.parse_qa_candidates("not json")

    assert "Candidate QA could not be parsed" in str(exc.value)


def test_generate_qa_candidates_calls_chat_model(monkeypatch):
    captured_messages = []

    def fake_call_chat_completion(messages, settings):
        captured_messages.extend(messages)
        return json.dumps([
            {"question": "What is SM94?", "expected_answer": "SM94 is an SR dent."}
        ])

    monkeypatch.setattr(api, "call_chat_completion", fake_call_chat_completion)
    monkeypatch.setattr(
        api,
        "chat_settings_from_env",
        lambda: {"base_url": "http://model.test/v1", "api_key": "EMPTY", "model": "mock-chat"},
    )

    payload = api.generate_qa_candidates("SM94 is a solder resist dent.", 5, "zh")

    assert payload["ok"] is True
    assert payload["candidates"][0]["question"] == "What is SM94?"
    assert "Return only a JSON array" in captured_messages[-1]["content"]


def test_generate_qa_candidates_rejects_empty_source_text():
    with pytest.raises(api.UploadError) as exc:
        api.generate_qa_candidates("   ", 5, "zh")

    assert "source text" in str(exc.value)


def test_generate_qa_candidates_rejects_invalid_count():
    with pytest.raises(api.UploadError) as exc:
        api.generate_qa_candidates("SM94 text", 7, "zh")

    assert "count" in str(exc.value)


def test_generate_answer_from_question_calls_retrieval_and_chat(monkeypatch, tmp_path):
    config_path = tmp_path / "eval-cfg.yaml"
    config_path.write_text(
        "generation:\n"
        "  retrieval_url: http://retrieval.test/api/v1/retrieval\n"
        "  retrieval_api_key: retrieval-key\n"
        "  dataset_ids: ds-1\n"
        "  page_size: 2\n"
        "  similarity_threshold: 0.1\n"
        "  llm_base_url: http://model.test/v1\n"
        "  llm_api_key: model-key\n"
        "  llm_model: mock-chat\n",
        encoding="utf-8",
    )
    captured_retrieval = {}
    captured_messages = []

    def fake_post_json(url, payload, headers, timeout_seconds=30):
        captured_retrieval.update({"url": url, "payload": payload, "headers": headers})
        return {
            "code": 0,
            "data": {
                "chunks": [
                    {"content": "SM94 M3 is top side SR undulation.", "score": 0.91, "document_name": "defects.pdf"},
                    {"content": "It is rejected by the defect criteria.", "score": 0.83},
                ]
            },
        }

    def fake_call_chat_completion(messages, settings):
        captured_messages.extend(messages)
        assert settings == {"base_url": "http://model.test/v1", "api_key": "model-key", "model": "mock-chat"}
        return "SM94 M3 is top side solder resist undulation and is rejected by the criteria."

    monkeypatch.setattr(api, "post_retrieval_json", fake_post_json)
    monkeypatch.setattr(api, "call_chat_completion", fake_call_chat_completion)

    payload = api.generate_answer_from_question("What is SM94 M3?", "zh", config_path)

    assert payload["ok"] is True
    assert payload["question"] == "What is SM94 M3?"
    assert payload["expected_answer"].startswith("SM94 M3")
    assert len(payload["passages"]) == 2
    assert captured_retrieval["url"] == "http://retrieval.test/api/v1/retrieval"
    assert captured_retrieval["payload"]["question"] == "What is SM94 M3?"
    assert captured_retrieval["headers"]["Authorization"] == "Bearer retrieval-key"
    assert "retrieved passages" in captured_messages[-1]["content"].lower()


def test_generate_answer_from_question_rejects_empty_question(tmp_path):
    config_path = tmp_path / "eval-cfg.yaml"
    config_path.write_text("generation:\n  retrieval_url: http://retrieval.test\n", encoding="utf-8")

    with pytest.raises(api.UploadError) as exc:
        api.generate_answer_from_question("   ", "zh", config_path)

    assert "question" in str(exc.value).lower()


def test_generate_answer_from_question_rejects_missing_retrieval_url(tmp_path):
    config_path = tmp_path / "eval-cfg.yaml"
    config_path.write_text("generation: {}\n", encoding="utf-8")

    with pytest.raises(api.UploadError) as exc:
        api.generate_answer_from_question("What is SM94?", "zh", config_path)

    assert "retrieval" in str(exc.value).lower()


def test_generate_answer_from_question_rejects_empty_passages(monkeypatch, tmp_path):
    config_path = tmp_path / "eval-cfg.yaml"
    config_path.write_text(
        "generation:\n"
        "  retrieval_url: http://retrieval.test/api/v1/retrieval\n"
        "  retrieval_payload_json: '{\"question\":\"{query}\"}'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        api,
        "post_retrieval_json",
        lambda url, payload, headers, timeout_seconds=30: {"data": {"chunks": []}},
    )

    with pytest.raises(api.UploadError) as exc:
        api.generate_answer_from_question("What is SM94?", "zh", config_path)

    assert "passages" in str(exc.value).lower()


def test_retrieve_chunks_from_question_calls_retrieval(monkeypatch, tmp_path):
    config_path = tmp_path / "eval-cfg.yaml"
    config_path.write_text(
        "datasets:\n"
        "  - id: ds-from-config\n"
        "    name: smoke\n"
        "grid:\n"
        "  page_sizes: [5]\n"
        "  similarity_thresholds: [0.1]\n"
        "generation:\n"
        "  retrieval_url: http://retrieval.test/api/v1/retrieval\n"
        "  retrieval_api_key_env: TEST_RETRIEVAL_API_KEY\n"
        "  vector_similarity_weight: 0.4\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_RETRIEVAL_API_KEY", "retrieval-key")
    captured_retrieval = {}

    def fake_post_json(url, payload, headers, timeout_seconds=30):
        captured_retrieval.update({"url": url, "payload": payload, "headers": headers})
        return {
            "code": 0,
            "data": {
                "chunks": [
                    {"id": "chunk-1", "content": "First retrieved chunk."},
                    {"id": "chunk-2", "content": "Second retrieved chunk."},
                ]
            },
        }

    monkeypatch.setattr(api, "post_retrieval_json", fake_post_json)

    payload = api.retrieve_chunks_from_question(
        " What is SM94? ",
        config_path,
        page_size=2,
        similarity_threshold=0.2,
        dataset_ids=["ds-override"],
        document_ids=["doc-1"],
        max_passages=1,
    )

    assert payload["ok"] is True
    assert payload["question"] == "What is SM94?"
    assert payload["chunks"] == [{"id": "[1]", "text": "First retrieved chunk.", "source": "chunk-1"}]
    assert payload["retrieval"]["page_size"] == 2
    assert payload["retrieval"]["similarity_threshold"] == 0.2
    assert captured_retrieval["url"] == "http://retrieval.test/api/v1/retrieval"
    assert captured_retrieval["payload"]["question"] == "What is SM94?"
    assert captured_retrieval["payload"]["dataset_ids"] == ["ds-override"]
    assert captured_retrieval["payload"]["document_ids"] == ["doc-1"]
    assert captured_retrieval["headers"]["Authorization"] == "Bearer retrieval-key"


def test_regenerate_answer_from_passages_calls_chat(monkeypatch, tmp_path):
    config_path = tmp_path / "eval-cfg.yaml"
    config_path.write_text(
        "generation:\n"
        "  llm_base_url: http://model.test/v1\n"
        "  llm_api_key: model-key\n"
        "  llm_model: mock-chat\n",
        encoding="utf-8",
    )
    captured_messages = []

    def fake_call_chat_completion(messages, settings):
        captured_messages.extend(messages)
        assert settings == {"base_url": "http://model.test/v1", "api_key": "model-key", "model": "mock-chat"}
        return "Improved answer based only on the retrieved passages."

    monkeypatch.setattr(api, "call_chat_completion", fake_call_chat_completion)

    payload = api.regenerate_answer_from_passages(
        "What is SM94 M3?",
        "Old answer",
        [{"text": "SM94 M3 is top side SR undulation."}],
        "zh",
        config_path,
    )

    assert payload == {"ok": True, "expected_answer": "Improved answer based only on the retrieved passages."}
    assert "Old answer" in captured_messages[-1]["content"]


def test_regenerate_answer_from_passages_rejects_missing_passages():
    with pytest.raises(api.UploadError) as exc:
        api.regenerate_answer_from_passages("What is SM94?", "Old answer", [], "zh")

    assert "passages" in str(exc.value).lower()


def test_handle_chat_rejects_blank_question():
    handler = DummyUploadHandler()
    body = json.dumps({"question": "   "}).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    with pytest.raises(api.UploadError) as exc:
        handler.handle_chat()

    assert "question" in str(exc.value)


def test_make_handler_sets_upload_root(tmp_path):
    reports_root = tmp_path / "reports"
    supplement_csv = tmp_path / "qa_supplement.csv"
    handler_class = api.make_handler(
        upload_root=tmp_path,
        reports_root=reports_root,
        chat_supplement_csv_path=supplement_csv,
    )

    assert issubclass(handler_class, api.DatasetUploadHandler)
    assert handler_class.upload_root == tmp_path
    assert handler_class.reports_root == reports_root
    assert handler_class.chat_supplement_csv_path == supplement_csv


def test_handle_upload_rejects_missing_content_length():
    handler = DummyUploadHandler()
    handler.headers = {"Content-Type": "multipart/form-data; boundary=upload-boundary"}

    with pytest.raises(api.UploadError) as exc:
        handler.handle_upload()

    assert "Content-Length" in str(exc.value)


def test_handle_upload_rejects_body_larger_than_limit():
    handler = DummyUploadHandler()
    handler.headers = {
        "Content-Type": "multipart/form-data; boundary=upload-boundary",
        "Content-Length": str(api.MAX_UPLOAD_BYTES + 1),
    }

    with pytest.raises(api.UploadError) as exc:
        handler.handle_upload()

    assert "too large" in str(exc.value)


def test_handle_upload_saves_multipart_csv(tmp_path):
    boundary = "upload-boundary"
    csv_body = b"query_id,query,expected_answer\nq1,What is A?,Answer A\n"
    body = build_multipart_body(
        boundary,
        [
            ({"Content-Disposition": 'form-data; name="name"'}, b"Support QA"),
            ({"Content-Disposition": 'form-data; name="file"; filename="demo.csv"'}, csv_body),
        ],
    )
    handler = DummyUploadHandler()
    handler.upload_root = tmp_path
    handler.rfile = io.BytesIO(body)
    handler.headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }

    payload = handler.handle_upload()

    assert payload["ok"] is True
    assert payload["dataset_name"] == "support_qa"
    assert (tmp_path / "support_qa.csv").exists()


def test_handle_upload_saves_multipart_pdf_as_csv(tmp_path, monkeypatch):
    boundary = "upload-boundary"
    pdf_body = b"%PDF fake"
    body = build_multipart_body(
        boundary,
        [
            ({"Content-Disposition": 'form-data; name="name"'}, b"Support QA"),
            (
                {
                    "Content-Disposition": 'form-data; name="file"; filename="qa-pairs.pdf"',
                    "Content-Type": "application/pdf",
                },
                pdf_body,
            ),
        ],
    )
    handler = DummyUploadHandler()
    handler.upload_root = tmp_path
    handler.rfile = io.BytesIO(body)
    handler.headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }

    monkeypatch.setattr(
        api,
        "extract_text_from_pdf",
        lambda content: "Question: What is A?\nAnswer: Answer A\n",
    )

    payload = handler.handle_upload()

    assert payload["ok"] is True
    assert payload["dataset_name"] == "support_qa"
    assert payload["source_format"] == "pdf"
    assert (tmp_path / "support_qa.csv").read_text(encoding="utf-8").splitlines() == [
        "query_id,query,expected_answer",
        "query_1,What is A?,Answer A",
    ]


def test_handle_manual_qa_appends_row_to_target_dataset(tmp_path):
    target_csv = tmp_path / "support_qa.csv"
    target_csv.write_text(
        "query_id,query,expected_answer\nquery_1,Existing question,Existing answer\n",
        encoding="utf-8",
    )
    handler = DummyUploadHandler()
    handler.upload_root = tmp_path
    body = json.dumps(
        {
            "target_dataset": str(target_csv),
            "question": "What is SM94?",
            "expected_answer": "A support module.",
        }
    ).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    payload = handler.handle_manual_qa()

    assert payload["ok"] is True
    assert payload["dataset_name"] == "support_qa"
    assert payload["source_format"] == "manual"
    assert payload["appended_query_id"] == "query_2"
    assert payload["rows"] == 2
    assert target_csv.read_text(encoding="utf-8").splitlines() == [
        "query_id,query,expected_answer",
        "query_1,Existing question,Existing answer",
        "query_2,What is SM94?,A support module.",
    ]


def test_append_bulk_qa_to_dataset_appends_multiple_rows(tmp_path):
    target_csv = tmp_path / "support_qa.csv"
    target_csv.write_text(
        "query_id,query,expected_answer\n"
        "query_1,Existing question,Existing answer\n",
        encoding="utf-8",
    )

    payload = api.append_bulk_qa_to_dataset(
        target_csv,
        [
            {"question": "What is SM94?", "expected_answer": "SM94 is an SR dent."},
            {"question": "What is SM94 M3?", "expected_answer": "SM94 M3 is top-side SR undulation."},
        ],
    )

    assert payload["ok"] is True
    assert payload["dataset_name"] == "support_qa"
    assert payload["source_format"] == "manual_bulk"
    assert payload["saved"] == 2
    assert payload["appended_query_ids"] == ["query_2", "query_3"]
    assert payload["rows"] == 3
    assert target_csv.read_text(encoding="utf-8").splitlines() == [
        "query_id,query,expected_answer",
        "query_1,Existing question,Existing answer",
        "query_2,What is SM94?,SM94 is an SR dent.",
        "query_3,What is SM94 M3?,SM94 M3 is top-side SR undulation.",
    ]


def test_append_bulk_qa_to_dataset_rejects_empty_rows(tmp_path):
    target_csv = tmp_path / "support_qa.csv"
    target_csv.write_text("query_id,query,expected_answer\n", encoding="utf-8")

    with pytest.raises(api.UploadError) as exc:
        api.append_bulk_qa_to_dataset(target_csv, [])

    assert "at least one QA row" in str(exc.value)


def test_append_bulk_qa_to_dataset_rejects_blank_question_or_answer(tmp_path):
    target_csv = tmp_path / "support_qa.csv"
    target_csv.write_text("query_id,query,expected_answer\n", encoding="utf-8")

    with pytest.raises(api.UploadError) as exc:
        api.append_bulk_qa_to_dataset(
            target_csv,
            [{"question": "What is SM94?", "expected_answer": "   "}],
        )

    assert "question and standard answer" in str(exc.value)


def test_do_post_manual_qa_saves_dataset(tmp_path):
    target_csv = tmp_path / "support_qa.csv"
    target_csv.write_text("query_id,query,expected_answer\n", encoding="utf-8")
    handler = DummyUploadHandler()
    handler.upload_root = tmp_path
    handler.path = "/api/datasets/manual-qa"
    body = json.dumps(
        {"target_dataset": str(target_csv), "question": "What is A?", "expected_answer": "Answer A"}
    ).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload["dataset_name"] == "support_qa"
    assert payload["path"] == api.relative_repo_path(target_csv)


def test_do_post_generate_qa_returns_candidates(monkeypatch):
    handler = DummyUploadHandler()
    handler.path = "/api/datasets/generate-qa"
    body = json.dumps({"source_text": "SM94 text", "count": 5, "language": "zh"}).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    def fake_generate_qa_candidates(source_text, count, language):
        assert source_text == "SM94 text"
        assert count == 5
        assert language == "zh"
        return {"ok": True, "candidates": [{"question": "What is SM94?", "expected_answer": "SM94 is an SR dent."}]}

    monkeypatch.setattr(api, "generate_qa_candidates", fake_generate_qa_candidates)

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload["ok"] is True
    assert payload["candidates"][0]["question"] == "What is SM94?"


def test_do_post_generate_answer_returns_answer(monkeypatch):
    handler = DummyUploadHandler()
    handler.path = "/api/datasets/generate-answer"
    body = json.dumps({"question": "What is SM94 M3?", "language": "zh"}).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    def fake_generate_answer_from_question(question, language, config_path):
        assert question == "What is SM94 M3?"
        assert language == "zh"
        assert config_path == handler.pipeline_config_path
        return {
            "ok": True,
            "question": question,
            "expected_answer": "Generated answer",
            "passages": [{"text": "Retrieved passage"}],
        }

    monkeypatch.setattr(api, "generate_answer_from_question", fake_generate_answer_from_question)

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload["expected_answer"] == "Generated answer"


def test_do_post_regenerate_answer_returns_answer(monkeypatch):
    handler = DummyUploadHandler()
    handler.path = "/api/datasets/regenerate-answer"
    body = json.dumps(
        {
            "question": "What is SM94 M3?",
            "current_answer": "Old answer",
            "passages": [{"text": "Retrieved passage"}],
            "language": "zh",
        }
    ).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    def fake_regenerate_answer_from_passages(question, current_answer, passages, language, config_path):
        assert question == "What is SM94 M3?"
        assert current_answer == "Old answer"
        assert passages == [{"text": "Retrieved passage"}]
        assert language == "zh"
        assert config_path == handler.pipeline_config_path
        return {"ok": True, "expected_answer": "Improved answer"}

    monkeypatch.setattr(api, "regenerate_answer_from_passages", fake_regenerate_answer_from_passages)

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload["expected_answer"] == "Improved answer"


def test_do_post_bulk_qa_saves_selected_rows(tmp_path):
    target_csv = tmp_path / "support_qa.csv"
    target_csv.write_text("query_id,query,expected_answer\n", encoding="utf-8")
    handler = DummyUploadHandler()
    handler.upload_root = tmp_path
    handler.path = "/api/datasets/bulk-qa"
    body = json.dumps(
        {
            "target_dataset": str(target_csv),
            "rows": [{"question": "What is SM94?", "expected_answer": "SM94 is an SR dent."}],
        }
    ).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload["ok"] is True
    assert payload["saved"] == 1
    assert payload["appended_query_ids"] == ["query_1"]


def test_handle_manual_qa_rejects_missing_question_or_answer():
    handler = DummyUploadHandler()
    body = json.dumps({"target_dataset": "data/qa_golden.csv", "question": "", "expected_answer": "Answer A"}).encode(
        "utf-8"
    )
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    with pytest.raises(api.UploadError) as exc:
        handler.handle_manual_qa()

    assert "question" in str(exc.value)
