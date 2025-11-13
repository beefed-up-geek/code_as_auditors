#!/usr/bin/env python3
"""Fill missing business/content fields in case_violations_pipa.jsonl via GPT-5."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional

from tqdm import tqdm

from method.utils.llm_interface import llm_response

BASE_DIR = Path("dataset/PIPA/cases/original")
DATA_PATH = BASE_DIR / "case_violations_pipa.jsonl"
MODEL_NAME = "gpt-5"
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
MAX_CHARS = 6000  # prevent overly long payloads

SYSTEM_PROMPT = """
너는 JSON만 반환하는 전문가다. 출력은 반드시 {\n  \"business\": ...,\n  \"content\": ...\n} 형식의 단일 JSON 객체여야 한다.
응답자는 반드시 회사 또는 단체 명칭으로 작성한다.
데이터 처리에 대한 사실만 기술하되 저장 위치/형식, 암호화, 보관, 접근권한, 제공/수탁, 공개 흐름을 세밀하게 적시하라.
"개인정보 유출"이나 위탁·수탁, 그리고 공유 흐름이 드러나도록 하지만 부정적인 단어(예: 위반, 문제, 잘못)를 사용하지 말고 법적 표현도 배제하라.
항상 한국어로 작성하고, content는 3~5문장 수준의 단락으로 요약한다.
""".strip()

USER_TEMPLATE = """
다음은 개인정보보호위원회 사건의 원문 일부이다. 요구사항을 지켜 {{"business":"...","content":"..."}} JSON만 반환하라.

[예시 출력]
{{
  "business": "피심인 A사",
  "content": "피심인 A사는 구매 이력 DB를 미국 리전 S3에 평문 CSV로 저장하고, 본사와 위탁사만 접근하는 VPN 계정을 부여한다. 매일 새벽 2시에 로그를 보관하며, 주문 정보는 3개월 뒤 별도 아카이브에 이동시킨다. 고객센터와 물류 파트가 동일 계정으로 접속해 상담 녹취와 배송 메모를 동일 저장소에서 열람한다."
}}

[원문]
{case_text}
""".strip()


def read_text(path: Path) -> str:
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def gather_case_text(file_number: str, case_id: str) -> Optional[str]:
    base = BASE_DIR / file_number
    if not base.exists():
        return None

    # Prefer folder named exactly case_id
    case_dir = base / case_id
    if case_dir.is_dir():
        parts = sorted(case_dir.glob("*.txt"))
        if parts:
            text = "\n\n".join(read_text(p) for p in parts)
            return text

    # Fall back to files containing case_id in name directly under base
    direct_matches = list(base.glob(f"*{case_id}*.txt"))
    if direct_matches:
        return read_text(direct_matches[0])

    # Search recursively (safeguard)
    for txt in base.rglob("*.txt"):
        if case_id in txt.stem:
            return read_text(txt)

    return None


def trim_text(text: str) -> str:
    if len(text) <= MAX_CHARS:
        return text
    return text[:MAX_CHARS] + "\n\n[이후 내용 생략]"


def load_records() -> list[dict]:
    records: list[dict] = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def save_records(records: Iterable[dict]) -> None:
    with DATA_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    records = load_records()
    targets = [r for r in records if not r.get("business") or not r.get("content")]
    if not targets:
        print("모든 레코드에 business와 content가 채워져 있습니다.")
        return

    print(f"LLM 추출 대상: {len(targets)}건")

    for record in tqdm(targets, desc="LLM 추출 진행", unit="case"):
        file_number = str(record.get("file_number") or "").strip()
        case_id = record.get("case_id")
        if not file_number or not case_id:
            continue

        case_text = gather_case_text(file_number, case_id)
        if not case_text:
            continue

        prompt = USER_TEMPLATE.format(case_text=trim_text(case_text))
        try:
            response = llm_response(MODEL_NAME, SYSTEM_PROMPT, prompt)
        except Exception as exc:
            print(f"\n{case_id} 처리 실패: {exc}")
            continue

        business = (response or {}).get("business")
        content = (response or {}).get("content")
        if business:
            record["business"] = business.strip()
        if content:
            record["content"] = content.strip()

    save_records(records)
    print("처리가 완료되었습니다.")


if __name__ == "__main__":
    main()
