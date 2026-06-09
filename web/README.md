# RAG Eval Web Frontend

React/Vite/TypeScript frontend for the RAG Eval console.

## Structure

```text
web/
|-- src/
|   |-- api/            # API client and shared frontend types
|   |-- components/     # Shell, job rail, and shared UI components
|   |-- pages/          # Datasets, Run, Records, Reports, Overview
|   |-- styles/         # Global console styles
|   |-- App.tsx
|   `-- main.tsx
|-- package.json
`-- vite.config.ts
```

## Development

Start the backend first from the repository root:

```powershell
python -m rag_eval_pipeline.api --host 127.0.0.1 --port 9000
```

Then start Vite from this directory:

```powershell
npm install
npm run dev
```

Open the frontend development server:

```text
http://127.0.0.1:5173
```

The Vite server shows the same app as the backend-served UI and proxies `/api`, `/reports`, and `/eval-output` requests to `http://127.0.0.1:9000`.

## Normal Usage

For normal usage, build the frontend and use the backend address:

```powershell
npm run build
```

```text
http://127.0.0.1:9000/datasets
```

The backend serves the built files from `web/dist`.

## Commands

```powershell
npm test
npm run build
npm run preview
```
