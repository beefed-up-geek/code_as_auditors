#!/usr/bin/env python3
"""Extract violated legal articles from case folders using GPT-4o."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

DATASET_ROOT = Path("dataset/PIPA/cases/original")
ENCODINGS = [
    "utf-8",
    "utf-8-sig",
    "cp949",
    "ms949",
    "euc-kr",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
]
OUTPUT_PATH = Path("case_violations.jsonl")
MODEL_NAME = "gpt-4o"


def read_text(path: Path) -> str:
    """Return file contents trying multiple encodings."""
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def collect_case_folders(root: Path) -> List[Path]:
    """Return sorted list of folders sitting next to case txt files."""
    folders = set()
    for txt in root.rglob("*.txt"):
        candidate = txt.parent / txt.stem
        if candidate.is_dir():
            folders.add(candidate)
    return sorted(folders)


def find_violation_files(folder: Path) -> List[Tuple[Path, str]]:
    """Return (path, text) pairs for files containing the word '위법'."""
    hits: List[Tuple[Path, str]] = []
    for file in sorted(folder.glob("*.txt")):
        text = read_text(file)
        if "위법" in text:
            hits.append((file, text))
    return hits


def build_prompt(folder: Path, files: List[Tuple[Path, str]], case_id: Optional[str]) -> str:
    """Create the user prompt with file contents."""
    header_parts = [
        "아래는 하나의 개인정보보호위원회 판결에서 '위법'이라는 표현이 포함된 텍스트 조각들입니다.",
        "보호법은 개인정보보호법과 동일함을 유념하세요.",
        "각 조각을 읽고 해당 판결에서 위법하다고 판단된 법과 조항을 찾아 JSON으로만 응답하세요.",
        "출력 형식은 예시와 같은 구조의 JSON 객체 하나여야 합니다.",
    ]
    if case_id:
        header_parts.append(f"사건 ID: {case_id}")
    header = "\n".join(header_parts)

    sections = []
    for file_path, text in files:
        sections.append(
            f"파일명: {file_path.name}\n{text.strip()}"
        )

    example = (
        "예시 출력:\n"
        "{""violated_articles"": [\n"
        "  {""law"": ""개인정보보호법"", ""id"": ""제24조 제1항""},\n"
        "  {""law"": ""시행령"", ""id"": ""제10조""},\n"
        "  {""law"": ""정보통신법"", ""id"": ""제34조 제3항 제2목""}\n"
        "]}\n"
    )

    return "\n\n".join([header, example, *sections])


def call_gpt(client: OpenAI, prompt: str) -> str:
    """Call GPT-4o and return its JSON response text."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a legal assistant who only outputs JSON. "
                    "Extract the violated legal articles described in the user message."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return response.choices[0].message.content.strip()


def load_metadata(folder: Path) -> Tuple[Dict[str, object], Path]:
    """Load metadata.json if present, otherwise return a default structure."""
    metadata_path = folder / "metadata.json"
    if metadata_path.exists():
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8")), metadata_path
        except json.JSONDecodeError:
            pass
    # Provide a default scaffold when metadata is missing or unreadable.
    data: Dict[str, object] = {"file_number": folder.parent.name}
    return data, metadata_path


def save_metadata(path: Path, data: Dict[str, object]) -> None:
    """Write metadata.json with pretty formatting."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please add it to your .env file.")

    client = OpenAI(api_key=api_key)

    case_folders = collect_case_folders(DATASET_ROOT)
    print(f"Discovered {len(case_folders)} case folders. Starting extraction…")

    output_file = OUTPUT_PATH.open("w", encoding="utf-8")
    skipped_no_word = 0
    failures = 0

    try:
        for folder in tqdm(case_folders, desc="Cases"):
            violation_files = find_violation_files(folder)
            if not violation_files:
                skipped_no_word += 1
                continue

            metadata, metadata_path = load_metadata(folder)
            case_id = metadata.get("id") if isinstance(metadata, dict) else None
            if isinstance(metadata, dict) and "file_number" not in metadata:
                metadata["file_number"] = folder.parent.name

            prompt = build_prompt(folder, violation_files, case_id)
            try:
                gpt_response = call_gpt(client, prompt)
            except Exception as exc:  # pylint: disable=broad-except
                failures += 1
                print(f"\nFailed to process {folder}: {exc}\n")
                continue

            result_json = json.loads(gpt_response)
            violated_articles = result_json.get("violated_articles", [])

            if isinstance(metadata, dict):
                metadata["violated_articles"] = violated_articles
                save_metadata(metadata_path, metadata)

            record = {
                "folder": str(folder),
                "case_id": case_id,
                "result": result_json,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        output_file.close()

    print("Run complete.")
    print(f"  Skipped (no '위법' text): {skipped_no_word}")
    print(f"  Failures: {failures}")
    print(f"  Results written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
