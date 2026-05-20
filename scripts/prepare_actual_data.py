"""Prepare local QA data for Open RAG Eval.

This converts data/QA List.xlsx to the CSV shape expected by run_eval and skips
the first data row, which is a sample QA record.
"""

from __future__ import annotations

import csv
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
XLSX_PATH = ROOT / "data" / "QA List.xlsx"
OUTPUT_PATH = ROOT / "data" / "qa_golden.csv"

NAMESPACES = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

# chat.csv contains two near-duplicate SM94 M3 root-cause queries. Both should
# be evaluated against the same golden answer row from the QA sheet.
QUERY_ID_OVERRIDES = {
    "What is SM94?": "query_1",
    "Please provide me detailed root cause of SM94 M3.": "query_2",
    "What is CU64 M1?": "query_4",
    "What is the root cause of CU64": "query_5",
    "Where is the CU64 M1 defect originating from?": "query_6",
}

QUERY_TEXT_OVERRIDES = {
    "query_2": "Please provide the detailed root cause of SM94 M3.",
}

EXTRA_ROWS = [
    {
        "query_id": "query_3",
        "query": "Please provide me with the detailed root cause of SM94 M3.",
        "source_question": "Please provide me detailed root cause of SM94 M3.",
    }
]


def _read_shared_strings(zip_file: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zip_file.namelist():
        return []

    root = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
    return [
        "".join(text.text or "" for text in item.findall(".//a:t", NAMESPACES))
        for item in root.findall("a:si", NAMESPACES)
    ]


def _read_first_sheet_rows(path: Path) -> list[list[str]]:
    with ZipFile(path) as zip_file:
        shared_strings = _read_shared_strings(zip_file)
        workbook = ET.fromstring(zip_file.read("xl/workbook.xml"))
        relationships = ET.fromstring(zip_file.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in relationships
        }

        first_sheet = workbook.find("a:sheets/a:sheet", NAMESPACES)
        if first_sheet is None:
            return []

        relationship_id = first_sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        target = rel_targets[relationship_id]
        sheet_path = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
        sheet = ET.fromstring(zip_file.read(sheet_path))

        rows = []
        for row in sheet.findall("a:sheetData/a:row", NAMESPACES):
            values = []
            for cell in row.findall("a:c", NAMESPACES):
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", NAMESPACES)
                value = "" if value_node is None else value_node.text or ""
                if cell_type == "s" and value:
                    value = shared_strings[int(value)]
                elif cell_type == "inlineStr":
                    value = "".join(
                        text.text or ""
                        for text in cell.findall(".//a:t", NAMESPACES)
                    )
                values.append(value)
            rows.append(values)

    return rows


def main() -> None:
    rows = _read_first_sheet_rows(XLSX_PATH)
    if len(rows) < 3:
        raise ValueError(f"No real QA rows found in {XLSX_PATH}")

    output_rows = []
    # rows[0] is the header, rows[1] is a sample row supplied for reference.
    for row in rows[2:]:
        if len(row) < 2:
            continue

        question, answer = row[:2]
        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            continue

        query_id = QUERY_ID_OVERRIDES.get(question)
        if not query_id:
            continue

        output_rows.append({
            "query_id": query_id,
            "query": QUERY_TEXT_OVERRIDES.get(query_id, question),
            "expected_answer": answer,
        })

        for extra in EXTRA_ROWS:
            if extra["source_question"] == question:
                output_rows.append({
                    "query_id": extra["query_id"],
                    "query": extra["query"],
                    "expected_answer": answer,
                })

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["query_id", "query", "expected_answer"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
