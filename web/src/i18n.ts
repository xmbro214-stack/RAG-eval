export type LanguageCode = "de" | "en" | "zh" | "ms";

export type TranslationKey =
  | "activeJob"
  | "chooseDatasetFile"
  | "chooseFile"
  | "consoleSections"
  | "customRecallChunks"
  | "customSimilarityThreshold"
  | "datasetName"
  | "datasetNamePlaceholder"
  | "datasets"
  | "enterQuestionAndAnswer"
  | "enterQuestionBeforeGenerating"
  | "evaluationRuns"
  | "expectedAnswer"
  | "generateAnswer"
  | "generatedAnswer"
  | "generating"
  | "idle"
  | "job"
  | "jobStatus"
  | "language"
  | "manualQa"
  | "navDatasets"
  | "navOverview"
  | "navRecords"
  | "navReports"
  | "navRun"
  | "navDatasetsSubtitle"
  | "navOverviewSubtitle"
  | "navRecordsSubtitle"
  | "navReportsSubtitle"
  | "navRunSubtitle"
  | "noBackgroundJob"
  | "noConfiguredDatasets"
  | "noDatasetsFound"
  | "noFileSelected"
  | "noPreviewRows"
  | "noReport"
  | "noReportsFound"
  | "noRetrievedPassages"
  | "openReport"
  | "overview"
  | "preview"
  | "question"
  | "quickEvaluation"
  | "recallChunks"
  | "records"
  | "regenerateAnswer"
  | "regenerating"
  | "report"
  | "reportModified"
  | "reportPath"
  | "reports"
  | "reportDetails"
  | "retrievalSettings"
  | "retrievedPassages"
  | "rows"
  | "rowsLabel"
  | "runDatasets"
  | "runEvaluation"
  | "saveQa"
  | "saveTargetDataset"
  | "saving"
  | "selectDatasetBeforeSaving"
  | "selectDatasetPreview"
  | "selectReport"
  | "selectOneDataset"
  | "selectRetrievalSettings"
  | "similarityThreshold"
  | "standardEvaluation"
  | "started"
  | "uploadDataset"
  | "uploaded"
  | "uploading";

export type Translate = (key: TranslationKey) => string;

export const languageOptions: Array<{ code: LanguageCode; label: string }> = [
  { code: "de", label: "Deutsch" },
  { code: "en", label: "English" },
  { code: "zh", label: "中文" },
  { code: "ms", label: "Bahasa Melayu" }
];

const translations: Record<LanguageCode, Record<TranslationKey, string>> = {
  de: {
    activeJob: "Aktiver Job",
    chooseDatasetFile: "Wähle vor dem Upload eine Datensatzdatei.",
    chooseFile: "Datei wählen",
    consoleSections: "Konsolenbereiche",
    customRecallChunks: "Eigene Abrufsegmente",
    customSimilarityThreshold: "Eigene Ähnlichkeitsschwelle",
    datasetName: "Datensatzname",
    datasetNamePlaceholder: "Datensatzname (optional)",
    datasets: "Datensätze",
    enterQuestionAndAnswer: "Gib Frage und erwartete Antwort ein.",
    enterQuestionBeforeGenerating: "Gib vor dem Generieren eine Frage ein.",
    evaluationRuns: "Auswertungsläufe",
    expectedAnswer: "Erwartete Antwort",
    generateAnswer: "Antwort generieren",
    generatedAnswer: "Antwort generiert",
    generating: "Generiert...",
    idle: "Bereit",
    job: "Job",
    jobStatus: "Jobstatus",
    language: "Sprache",
    manualQa: "Manuelles Q&A",
    navDatasets: "Datensätze",
    navDatasetsSubtitle: "QA-Datensätze und fundierte Antworten verwalten",
    navOverview: "Überblick",
    navOverviewSubtitle: "Aktuellen Lauf und Systemstatus prüfen",
    navRecords: "Läufe",
    navRecordsSubtitle: "Auswertungshistorie und Artefakte prüfen",
    navReports: "Berichte",
    navReportsSubtitle: "Generierte Berichte öffnen und vergleichen",
    navRun: "Ausführen",
    navRunSubtitle: "Datensätze und Abrufeinstellungen konfigurieren",
    noBackgroundJob: "Es läuft kein Hintergrundjob.",
    noConfiguredDatasets: "Keine konfigurierten Datensätze verfügbar.",
    noDatasetsFound: "Keine Datensätze gefunden.",
    noFileSelected: "Keine Datei ausgewählt",
    noPreviewRows: "Keine Vorschauzeilen gefunden.",
    noReport: "Kein Bericht",
    noReportsFound: "Keine Berichte gefunden.",
    noRetrievedPassages: "Noch keine abgerufenen Passagen.",
    openReport: "Bericht öffnen",
    overview: "Überblick",
    preview: "Vorschau",
    question: "Frage",
    quickEvaluation: "Schnellauswertung",
    recallChunks: "Abrufsegmente",
    records: "Läufe",
    regenerateAnswer: "Antwort neu generieren",
    regenerating: "Generiert neu...",
    report: "Bericht",
    reportModified: "Geändert",
    reportPath: "Pfad",
    reports: "Berichte",
    reportDetails: "Berichtdetails",
    retrievalSettings: "Abrufeinstellungen",
    retrievedPassages: "Abgerufene Passagen",
    rows: "Zeilen",
    rowsLabel: "Zeilen",
    runDatasets: "Datensätze ausführen",
    runEvaluation: "Auswertung starten",
    saveQa: "Q&A speichern",
    saveTargetDataset: "Zieldatensatz",
    saving: "Speichert...",
    selectDatasetBeforeSaving: "Wähle vor dem Speichern von Q&A einen Datensatz.",
    selectDatasetPreview: "Wähle einen Datensatz, um Zeilen anzuzeigen.",
    selectReport: "Bericht auswahlen",
    selectOneDataset: "Wähle mindestens einen Datensatz.",
    selectRetrievalSettings: "Wähle Abrufsegmente und Ähnlichkeitsschwelle.",
    similarityThreshold: "Ähnlichkeitsschwelle",
    standardEvaluation: "Standardauswertung",
    started: "Gestartet",
    uploadDataset: "Datensatz hochladen",
    uploaded: "Hochgeladen",
    uploading: "Lädt hoch..."
  },
  en: {
    activeJob: "Active job",
    chooseDatasetFile: "Choose a dataset file before uploading.",
    chooseFile: "Choose file",
    consoleSections: "Console sections",
    customRecallChunks: "Custom recall chunks",
    customSimilarityThreshold: "Custom similarity threshold",
    datasetName: "Dataset name",
    datasetNamePlaceholder: "Dataset name (optional)",
    datasets: "Datasets",
    enterQuestionAndAnswer: "Enter both a question and expected answer.",
    enterQuestionBeforeGenerating: "Enter a question before generating an answer.",
    evaluationRuns: "Evaluation Runs",
    expectedAnswer: "Expected answer",
    generateAnswer: "Generate Answer",
    generatedAnswer: "Generated answer",
    generating: "Generating...",
    idle: "Idle",
    job: "Job",
    jobStatus: "Job status",
    language: "Language",
    manualQa: "Manual Q&A",
    navDatasets: "Datasets",
    navDatasetsSubtitle: "Manage QA datasets and grounded answers",
    navOverview: "Overview",
    navOverviewSubtitle: "Monitor the current run and console status",
    navRecords: "Records",
    navRecordsSubtitle: "Review evaluation history and run artifacts",
    navReports: "Reports",
    navReportsSubtitle: "Open generated reports and compare outputs",
    navRun: "Run",
    navRunSubtitle: "Configure datasets and retrieval settings",
    noBackgroundJob: "No background job is running.",
    noConfiguredDatasets: "No configured datasets available.",
    noDatasetsFound: "No datasets found.",
    noFileSelected: "No file selected",
    noPreviewRows: "No preview rows found.",
    noReport: "No report",
    noReportsFound: "No reports found.",
    noRetrievedPassages: "No retrieved passages yet.",
    openReport: "Open report",
    overview: "Overview",
    preview: "Preview",
    question: "Question",
    quickEvaluation: "Quick evaluation",
    recallChunks: "Recall chunks",
    records: "Records",
    regenerateAnswer: "Regenerate Answer",
    regenerating: "Regenerating...",
    report: "Report",
    reportModified: "Modified",
    reportPath: "Path",
    reports: "Reports",
    reportDetails: "Report details",
    retrievalSettings: "Retrieval settings",
    retrievedPassages: "Retrieved passages",
    rows: "rows",
    rowsLabel: "Rows",
    runDatasets: "Run datasets",
    runEvaluation: "Run Evaluation",
    saveQa: "Save Q&A",
    saveTargetDataset: "Save target dataset",
    saving: "Saving...",
    selectDatasetBeforeSaving: "Select a dataset before saving Q&A.",
    selectDatasetPreview: "Select a dataset to preview rows.",
    selectReport: "Select report",
    selectOneDataset: "Select at least one dataset.",
    selectRetrievalSettings: "Select recall chunks and similarity threshold.",
    similarityThreshold: "Similarity threshold",
    standardEvaluation: "Standard evaluation",
    started: "Started",
    uploadDataset: "Upload dataset",
    uploaded: "Uploaded",
    uploading: "Uploading..."
  },
  zh: {
    activeJob: "当前任务",
    chooseDatasetFile: "上传前请选择数据集文件。",
    chooseFile: "选择文件",
    consoleSections: "控制台模块",
    customRecallChunks: "自定义召回片段",
    customSimilarityThreshold: "自定义相似度阈值",
    datasetName: "数据集名称",
    datasetNamePlaceholder: "数据集名称（可选）",
    datasets: "数据集",
    enterQuestionAndAnswer: "请输入问题和期望回答。",
    enterQuestionBeforeGenerating: "生成回答前请输入问题。",
    evaluationRuns: "评估记录",
    expectedAnswer: "期望回答",
    generateAnswer: "生成回答",
    generatedAnswer: "已生成回答",
    generating: "生成中...",
    idle: "空闲",
    job: "任务",
    jobStatus: "任务状态",
    language: "语言",
    manualQa: "手动 Q&A",
    navDatasets: "数据集",
    navDatasetsSubtitle: "管理 QA 数据集和基于召回的回答",
    navOverview: "概览",
    navOverviewSubtitle: "查看当前运行和控制台状态",
    navRecords: "记录",
    navRecordsSubtitle: "查看评估历史和运行产物",
    navReports: "报告",
    navReportsSubtitle: "打开生成报告并对比输出",
    navRun: "运行",
    navRunSubtitle: "配置数据集和召回设置",
    noBackgroundJob: "当前没有后台任务。",
    noConfiguredDatasets: "没有可用的配置数据集。",
    noDatasetsFound: "未找到数据集。",
    noFileSelected: "未选择文件",
    noPreviewRows: "未找到预览行。",
    noReport: "无报告",
    noReportsFound: "未找到报告。",
    noRetrievedPassages: "暂无召回片段。",
    openReport: "打开报告",
    overview: "概览",
    preview: "预览",
    question: "问题",
    quickEvaluation: "快速评估",
    recallChunks: "召回片段",
    records: "记录",
    regenerateAnswer: "重新生成回答",
    regenerating: "重新生成中...",
    report: "报告",
    reportModified: "修改时间",
    reportPath: "路径",
    reports: "报告",
    reportDetails: "Report details",
    retrievalSettings: "召回设置",
    retrievedPassages: "召回片段",
    rows: "行",
    rowsLabel: "行",
    runDatasets: "运行数据集",
    runEvaluation: "运行评估",
    saveQa: "保存 Q&A",
    saveTargetDataset: "保存目标数据集",
    saving: "保存中...",
    selectDatasetBeforeSaving: "保存 Q&A 前请选择数据集。",
    selectDatasetPreview: "选择一个数据集以预览行。",
    selectReport: "Select report",
    selectOneDataset: "请至少选择一个数据集。",
    selectRetrievalSettings: "请选择召回片段和相似度阈值。",
    similarityThreshold: "相似度阈值",
    standardEvaluation: "标准评估",
    started: "已启动",
    uploadDataset: "上传数据集",
    uploaded: "已上传",
    uploading: "上传中..."
  },
  ms: {
    activeJob: "Tugas aktif",
    chooseDatasetFile: "Pilih fail dataset sebelum memuat naik.",
    chooseFile: "Pilih fail",
    consoleSections: "Bahagian konsol",
    customRecallChunks: "Serpihan carian tersuai",
    customSimilarityThreshold: "Ambang kesamaan tersuai",
    datasetName: "Nama dataset",
    datasetNamePlaceholder: "Nama dataset (pilihan)",
    datasets: "Dataset",
    enterQuestionAndAnswer: "Masukkan soalan dan jawapan jangkaan.",
    enterQuestionBeforeGenerating: "Masukkan soalan sebelum menjana jawapan.",
    evaluationRuns: "Rekod penilaian",
    expectedAnswer: "Jawapan jangkaan",
    generateAnswer: "Jana Jawapan",
    generatedAnswer: "Jawapan dijana",
    generating: "Menjana...",
    idle: "Melahu",
    job: "Tugas",
    jobStatus: "Status tugas",
    language: "Bahasa",
    manualQa: "Q&A manual",
    navDatasets: "Dataset",
    navDatasetsSubtitle: "Urus dataset QA dan jawapan berasaskan carian",
    navOverview: "Gambaran",
    navOverviewSubtitle: "Pantau larian semasa dan status konsol",
    navRecords: "Rekod",
    navRecordsSubtitle: "Semak sejarah penilaian dan artifak larian",
    navReports: "Laporan",
    navReportsSubtitle: "Buka laporan terjana dan bandingkan output",
    navRun: "Jalankan",
    navRunSubtitle: "Konfigurasi dataset dan tetapan carian",
    noBackgroundJob: "Tiada tugas latar belakang sedang berjalan.",
    noConfiguredDatasets: "Tiada dataset konfigurasi tersedia.",
    noDatasetsFound: "Tiada dataset ditemui.",
    noFileSelected: "Tiada fail dipilih",
    noPreviewRows: "Tiada baris pratonton ditemui.",
    noReport: "Tiada laporan",
    noReportsFound: "Tiada laporan ditemui.",
    noRetrievedPassages: "Belum ada petikan carian.",
    openReport: "Buka laporan",
    overview: "Gambaran",
    preview: "Pratonton",
    question: "Soalan",
    quickEvaluation: "Penilaian pantas",
    recallChunks: "Serpihan carian",
    records: "Rekod",
    regenerateAnswer: "Jana Semula Jawapan",
    regenerating: "Menjana semula...",
    report: "Laporan",
    reportModified: "Diubah",
    reportPath: "Laluan",
    reports: "Laporan",
    reportDetails: "Report details",
    retrievalSettings: "Tetapan carian",
    retrievedPassages: "Petikan carian",
    rows: "baris",
    rowsLabel: "Baris",
    runDatasets: "Jalankan dataset",
    runEvaluation: "Jalankan Penilaian",
    saveQa: "Simpan Q&A",
    saveTargetDataset: "Dataset sasaran",
    saving: "Menyimpan...",
    selectDatasetBeforeSaving: "Pilih dataset sebelum menyimpan Q&A.",
    selectDatasetPreview: "Pilih dataset untuk pratonton baris.",
    selectReport: "Select report",
    selectOneDataset: "Pilih sekurang-kurangnya satu dataset.",
    selectRetrievalSettings: "Pilih serpihan carian dan ambang kesamaan.",
    similarityThreshold: "Ambang kesamaan",
    standardEvaluation: "Penilaian standard",
    started: "Dimulakan",
    uploadDataset: "Muat naik dataset",
    uploaded: "Dimuat naik",
    uploading: "Memuat naik..."
  }
};

export function translate(language: LanguageCode, key: TranslationKey): string {
  return translations[language][key];
}
