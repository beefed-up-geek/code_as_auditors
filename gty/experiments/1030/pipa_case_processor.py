import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


PROMPT_TEMPLATE = """ 
입력으로 주어진 사건 문서(심의·의결서 등)를 읽고, 그 안에서 드러난 사업자의 개인정보 처리 방식과 
위반된 법 조항을 JSON 구조로 정리하라. 

출력은 반드시 하나의 JSON 객체여야 하며, JSON의 필드 순서는 다음과 같아야 한다:

1. case_id: 문서에 명시된 사건번호 또는 안건번호를 그대로 기입한다. (예: "2021-013-103, 2021조일035")
2. business: 조사·처분의 대상이 된 기업명 또는 서비스명. (예: "넷플릭스", "구글", "카카오" 등)
3. violated_articles: 각 위반 법 조항의 id(법 종류 & 조, 항, 호, 목 등 세부 위치)와 reason(조문의 내용, 기업의 위반 행위, 위반 이유)
4. content: 문서에서 드러난 개인정보 수집·동의·이용·보관·처리·이전의 모든 과정을 
   객관적으로 기술하되(특히 위반 사항과 관련된 데이터 처리 단계들이 강조 되야함), 법 위반 여부나 법과 관련된 내용을 언급하지 않는다.

출력 형식 예시는 다음과 같다:

{
  "case_id": "2021-013-103",
  "business": "넷플릭스",
  "violated_articles": [
    { "id": {"law": "개인정보보호법", "id": "제39조의12 제2항"}, "reason": "..." },
    { "id": {"law": "시행령", "id": "제30조 제2항 제4호"}, "reason": "..." },
    { "id": {"law": "정보통신망법", "id": "제28조 제1항"}, "reason": "..." },
  ],
  "content": "..."
}

주의사항:
- 반드시 JSON만 출력하고, 다른 설명문이나 해설을 덧붙이지 않는다.
- case_id와 business는 문서에서 직접 추출 가능한 정보를 기반으로 작성한다.
- violated_articles 안에는 아래의 사건 문서에서 위법이라고 정확히 판별한 법률 조항들을 중복 없이 모두 포함해야한다. 
- violated_articles 안의 law에는 "개인정보보호법", "시행령" 또는 기타 법의 종류를 명시한다.  
- violated_articles 안의 id 들은 가능한 구체적으로 명시한다. 또한 조, 항, 호 목을 형식에 맞게 표시한다. (ex.제3조, 제4조의2 제1항, 제5조 제7항, 제3목) 
- reason은 조문의 내용, 기업의 위반 행위, 위반 이유를 연결하여 구체적이고 논리적이게 기술한다. 
- content에는 사업자의 데이터 처리 방식(수집 항목, 목적, 시점, 보관 기간, 국외 이전, 시스템 구조 등)을 최대한 구체적으로 포함한다.
- content에는 “위반”, “불법”, “위법하다” 등의 표현과 법에 대한 어떤 언급도 절대 사용하지 않는다.
- 하지만, 중요한 점은 content에는 violated_articles에 명시된 기업의 위반 행위들이 누락 없이 강조되어 들어가있어야한다. 

아래는 분석할 사건 문서의 전문이다. 
이 문서의 내용을 바탕으로 위 지침에 따라 오직 JSON만을 반환하라.

------ 사건 문서 시작 ------
{document}
------ 사건 문서 끝 ------
"""


def extract_output_text(response: Any) -> str:
    """Extract plain text from responses API result."""
    response_text = getattr(response, "output_text", None)

    if response_text:
        return response_text.strip()

    if hasattr(response, "model_dump"):
        response_payload = response.model_dump()
    elif isinstance(response, dict):
        response_payload = response
    else:
        response_payload = {}

    response_text = response_payload.get("output_text")
    if isinstance(response_text, str) and response_text.strip():
        return response_text.strip()

    output_items = response_payload.get("output") if isinstance(response_payload, dict) else None
    if output_items is None:
        output_items = getattr(response, "output", []) or []

    segments: List[str] = []
    for item in output_items:
        if isinstance(item, dict):
            content_list = item.get("content", [])
        else:
            content_list = getattr(item, "content", []) or []

        for segment in content_list:
            if isinstance(segment, dict):
                text_value = segment.get("text")
            else:
                text_value = getattr(segment, "text", None)
            if text_value:
                segments.append(text_value)

    return "".join(segments).strip()


def refine_json_with_agent(client: OpenAI, raw_text: str, relative_path: str) -> Dict[str, Any]:
    """Use chat completions JSON mode to clean up malformed JSON."""
    print(f"🔁 Retrying JSON parsing for {relative_path} via gpt-4o JSON agent.")
    messages = [
        {
            "role": "system",
            "content": "You are a JSON formatting assistant. Return valid JSON only with the same fields requested originally.",
        },
        {
            "role": "user",
            "content": (
                "다음 텍스트는 개인정보 사건 분석 결과지만 JSON 형식에 오류가 있습니다. "
                "원래의 필드와 구조(case_id, business, violated_articles(json 리스트 내용 유지), content)를 유지하면서 유효한 JSON 객체로 정제해 주세요. "
                "JSON 외의 다른 설명은 추가하지 마세요.\n\n"
                "----- 원본 텍스트 -----\n"
                f"{raw_text}\n"
                "----- 끝 -----"
            ),
        },
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "json_object"},
    )

    fixed_text = completion.choices[0].message.content
    if not fixed_text:
        raise ValueError("JSON parsing agent returned empty output.")

    try:
        return json.loads(fixed_text)
    except json.JSONDecodeError as exc:
        raise ValueError("JSON parsing agent returned invalid JSON.") from exc


def load_existing_results(output_path: Path) -> List[Dict[str, Any]]:
    if not output_path.exists():
        return []
    try:
        with output_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, list):
                return data
    except json.JSONDecodeError:
        pass
    return []


def ensure_processed_paths(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    processed: Dict[str, Dict[str, Any]] = {}
    for item in results:
        source_path = item.get("source_path")
        if isinstance(source_path, str):
            processed[source_path] = item
    return processed


def build_prompt(document_text: str) -> str:
    """PROMPT_TEMPLATE에 문서 삽입"""
    return PROMPT_TEMPLATE.replace("{document}", document_text)


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Check your .env file.")

    base_dir = Path("/Users/taeyoonkwack/Documents/code_as_auditors/dataset/PIPA/cases/original")
    output_path = Path("/Users/taeyoonkwack/Documents/code_as_auditors/persoanl_directory/gty/experiments/1030/pipa_cases_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_results = load_existing_results(output_path)
    processed_map = ensure_processed_paths(existing_results)

    all_txt_files = sorted(base_dir.rglob("*.txt"))
    client = OpenAI()
    aggregated_results = existing_results[:]

    for txt_file in tqdm(all_txt_files, desc="Processing case files"):
        relative_path = str(txt_file.relative_to(base_dir))
        if relative_path in processed_map:
            continue

        try:
            document_text = txt_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            document_text = txt_file.read_text(encoding="utf-8", errors="replace")

        prompt = build_prompt(document_text)

        # ✅ 안정적: responses API 기반 호출
        try:
            response = client.responses.create(
                model="gpt-5",
                input=prompt,
                reasoning={"effort": "low"},
                #response_format={"type": "json_object"},
            )

            response_text = extract_output_text(response)
            if not response_text:
                raise ValueError("Model returned empty output.")

            try:
                analysis = json.loads(response_text)
            except json.JSONDecodeError:
                analysis = refine_json_with_agent(client, response_text, relative_path)

        except Exception as e:
            print(f"❌ Error processing {relative_path}: {e}")
            continue

        result_entry = {
            "source_path": relative_path,
            "analysis": analysis,
        }

        aggregated_results.append(result_entry)
        processed_map[relative_path] = result_entry

        # 중간 저장
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(aggregated_results, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
