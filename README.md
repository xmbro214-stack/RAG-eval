# RAG Eval Pipeline

轻量级 RAG 评估与可视化工具。当前主要运行方式是启动一个本地 Web Server，在浏览器中完成数据上传、评估运行、报告查看和模型问答补充。

## 快速启动

安装依赖：

```powershell
python -m pip install -e .
python -m pip install -r requirements.txt
```

启动服务：

```powershell
python -m rag_eval_pipeline.api --host 127.0.0.1 --port 9000
```

打开页面：

```text
http://127.0.0.1:9000/datasets
```

启动入口在 `rag_eval_pipeline/api.py` 的 `main()`，默认读取 `.env`，默认端口是 `9000`。

## 页面功能

`/datasets` 是当前主页面，前端由 `rag_eval_pipeline/api.py::render_upload_page()` 直接生成。

页面包含：

- 报告查看：选择并展示 `reports/*.html`
- 上传数据集：支持 CSV 和文本型 PDF
- 运行评估：点击 `Run evaluation` 后在后台执行 pipeline
- 提问模型：围绕当前报告提问，点击保存后才会写入 `data/qa_supplement.csv`
- 页面语言：支持德语、英语、中文、马来西亚语

`Ask model` 中已经取消单独的回答语言选择，模型回答语言会自动跟随顶部页面语言。

## 数据上传

上传接口：

```text
POST /api/datasets/upload
```

CSV 必须包含：

```csv
query_id,query,expected_answer
query_1,What is SM94?,Golden answer text
```

PDF 要求是可抽取文本的 QA 文档，支持类似格式：

```text
Question: What is SM94?
Answer: Golden answer text

问题：什么是 SM94？
答案：标准答案文本
```

上传后文件保存到：

```text
data/uploaded_datasets/
```

页面会显示实际保存路径，可写入评估配置：

```yaml
queries_csv: data/uploaded_datasets/your_dataset.csv
golden_csv: data/uploaded_datasets/your_dataset.csv
```

命令行上传示例：

```powershell
curl.exe -F "file=@data\qa_golden.csv" -F "name=qa_golden" http://127.0.0.1:9000/api/datasets/upload
```

## 评估运行

页面上的 `Run evaluation` 默认使用：

```text
scripts/eval-cfg.yaml
```

配置中当前默认数据文件是：

```yaml
queries_csv: data/qa_golden.csv
golden_csv: data/qa_golden.csv
```

评估输出通常位于：

```text
data/eval_runs/{run_name}/{dataset}/ps{page_size}_sim{similarity}/
  generated_answers.csv
  eval_result.csv

data/eval_runs/{run_name}/manifest.json
```

HTML 报告保存到：

```text
reports/*.html
```

页面会自动扫描并展示 `reports/` 下的 HTML 报告。

## 模型问答

`Ask model` 使用 OpenAI-compatible Chat API。需要配置：

```powershell
$env:LOCAL_LLM_BASE_URL="http://127.0.0.1:8011/v1"
$env:LOCAL_LLM_API_KEY="EMPTY"
$env:LOCAL_LLM_MODEL="mock-chat"
```

提问流程：

1. 前端把当前问题、当前报告地址和页面语言发送到 `/api/chat`
2. 后端提取当前 HTML 报告文本作为上下文
3. 后端调用 chat model
4. 前端点击保存后，回答写入 `data/qa_supplement.csv`；点击放弃则不写入

追加写入的数据可作为后续评估的 QA 补充。

## 后端接口

主要接口都在 `rag_eval_pipeline/api.py::DatasetUploadHandler`：

```text
GET  /datasets
GET  /api/reports
GET  /api/pipeline/status?run_id=...
GET  /reports/<report.html>

POST /api/datasets/upload
POST /api/datasets/manual-qa
POST /api/pipeline/run
POST /api/chat
POST /api/chat/save
```

默认路径：

```text
data/uploaded_datasets/   上传数据集
data/qa_supplement.csv    Ask model 手动保存的 QA
reports/                  HTML 评估报告
scripts/eval-cfg.yaml     默认评估配置
```

## 本地 Mock

没有真实 retrieval 或模型服务时，可以启动 mock：

```powershell
python scripts/mock_retrieval_server.py --host 127.0.0.1 --port 9380 --golden-csv data/qa_golden.csv
python scripts/mock_openai_server.py --host 127.0.0.1 --port 8011
```

然后启动页面服务：

```powershell
python -m rag_eval_pipeline.api --host 127.0.0.1 --port 9000
```

## 常用命令

仅检查 pipeline 命令：

```powershell
python scripts/run_rag_eval_pipeline.py --config scripts/eval-cfg.yaml --dry-run
```

完整运行 pipeline：

```powershell
python scripts/run_rag_eval_pipeline.py --config scripts/eval-cfg.yaml
```

运行测试：

```powershell
python -m pytest -q
```

语法检查：

```powershell
python -m py_compile rag_eval_pipeline\api.py
```
