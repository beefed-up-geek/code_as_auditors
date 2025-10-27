import os
import json
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict

# 1️⃣ 환경 설정
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2️⃣ 파일 경로 설정
input_path = "/Users/taeyoonkwack/Documents/code_as_auditors/dataset/PIPA/law/decree.json"
output_path = input_path.replace(".json", "_definition.json")

# 3) 데이터 로드
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 4) parent -> children 매핑
children_map = defaultdict(list)
for item in data:
    if "parent" in item:
        children_map[item["parent"]].append(item)

# 5) 조(+하위) 텍스트 빌드
def build_article_text(article):
    lines = [f"[조항] {article['id']}"]
    if article.get("title"):
        lines.append(f"[제목] {article['title']}")
    if article.get("content"):
        lines.append(f"[내용]\n{article['content']}")

    def collect(pid):
        for child in children_map.get(pid, []):
            if child.get("content"):
                lines.append(f"{child['id']}: {child['content']}")
            collect(child["id"])
    collect(article["id"])

    return "\n".join(lines).strip()

# 6) 시스템 프롬프트 (‘이하’ 패턴 전용, 법 명시 규칙, 객체(JSON)로 반환, 멀티-정의 few-shot 포함)
SYSTEM_PROMPT = """너는 한국 법령 텍스트를 분석하는 전문가이다.
아래 입력은 한 조항과 그 하위 항목(항, 호, 목 등)을 모두 포함한다.
입력 텍스트에서 **'(이하 “…”라 한다)' 또는 '(이하 ‘…’라 한다)'** 패턴으로 약칭을 부여하는 **'이하' 정의**만을 찾아,
다음 **JSON 객체** 형태로만 반환하라(배열은 'items' 필드에 넣는다):

{
  "items": [
    {"definition_term":"...", "full_term":"..."},
    ...
  ]
}

규칙(매우 중요):
- 오직 '이하 …라 한다' 패턴만 추출한다. ('…란 …을 말한다' 등은 무시)
- definition_term: 괄호 안에 지정된 약칭(따옴표 없이 원문 그대로).
- full_term: 약칭이 지칭하는 **원문 내 전체 표현**을 자연스럽게 포함하되,
  - **content에 '법 제XX조', 'OO법 제X조', '시행령 제X조' 등 법/조항이 명시된 경우에만** 그 **법 문구를 그대로 포함**한다.
  - '이 법', '본 조' 같은 비특정 표현은 법 명시로 보지 않는다(포함 금지).
  - 정의의 **출처(예: 조항 ID)는 full_term에 절대 포함하지 말 것**.
- 여러 개가 있으면 모두 'items' 배열에 담아 반환한다.
- **이하 정의가 하나도 없으면 'items': [] 만 반환**한다.
- 반드시 JSON **객체**만 출력한다. 다른 텍스트/주석은 금지.

---

### Few-shot 예시 1 (법 조항 '명시'된 경우 → full_term에 법 포함)

입력:
[조항] 제49조
[내용]
이 법은 법 제49조에 따른 집단분쟁조정(이하 '집단분쟁조정'이라 한다) 절차를 규정한다.

출력:
{
  "items": [
    {
      "definition_term": "집단분쟁조정",
      "full_term": "법 제49조에 따른 집단분쟁조정"
    }
  ]
}

---

### Few-shot 예시 2 (법 조항 '명시' 없음 → full_term에 법 미포함)

입력:
[조항] 제3조
[내용]
국무총리 소속으로 개인정보 보호위원회(이하 '보호위원회'라 한다)를 둔다.

출력:
{
  "items": [
    {
      "definition_term": "보호위원회",
      "full_term": "국무총리 소속 개인정보 보호위원회"
    }
  ]
}

---

### Few-shot 예시 3 (이하 정의가 없는 경우 → 빈 items)

입력:
[조항] 제2조 제1의2호
[내용]
“가명처리”란 개인정보의 일부를 삭제하거나 일부 또는 전부를 대체하는 등의 방법으로 추가 정보가 없이는 특정 개인을 알아볼 수 없도록 처리하는 것을 말한다.

출력:
{
  "items": []
}

---

### Few-shot 예시 4 (한 조항에 여러 '이하' 정의가 있는 경우)

입력:
[조항] 제4조의2
[제목] 영리업무의 금지
[내용]
법 제7조제1항에 따른 개인정보 보호위원회(이하 “보호위원회”라 한다)의 위원은 법 제7조의6제1항에 따라 영리를 목적으로 다음 각 호의 어느 하나에 해당하는 업무에 종사해서는 안 된다.
1. 법 제7조의9제1항에 따라 보호위원회가 심의ㆍ의결하는 사항과 관련된 업무
2. 법 제40조제1항에 따른 개인정보 분쟁조정위원회(이하 “분쟁조정위원회”라 한다)가 조정하는 사항과 관련된 업무

출력:
{
  "items": [
    {
      "definition_term": "보호위원회",
      "full_term": "법 제7조제1항에 따른 개인정보 보호위원회"
    },
    {
      "definition_term": "분쟁조정위원회",
      "full_term": "법 제40조제1항에 따른 개인정보 분쟁조정위원회"
    }
  ]
}
"""

# 7) GPT 호출 및 결과 처리
results = []

def parse_items_maybe_array(payload: str):
    """json_object 강제에서 모델이 객체/배열 어느 쪽을 내더라도 안전 파싱"""
    try:
        obj = json.loads(payload)
    except Exception:
        return []
    # 예상 경로 1: {"items":[...]}
    if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], list):
        return obj["items"]
    # 예상 경로 2: 모델이 배열을 직접 낸 경우(방어)
    if isinstance(obj, list):
        return obj
    # 그 외는 빈 리스트 처리
    return []

# 조 단위 1회 호출
for article in tqdm([x for x in data if x.get("class") == "조"], desc="Extracting 이하-definitions"):
    merged_text = build_article_text(article)
    if not merged_text:
        continue

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": merged_text}
            ],
            response_format={"type": "json_object"}  # 객체 보장(배열은 items에 담도록 지시)
        )
        items = parse_items_maybe_array(resp.choices[0].message.content)

        # 정의가 없으면 추가하지 않음
        if not items:
            continue

        # source_id는 코드에서만 추가
        for it in items:
            # 방어: 필요한 키만 유지(혹시 모델이 더 보태면 제거)
            clean = {
                "definition_term": it.get("definition_term", "").strip(),
                "full_term": it.get("full_term", "").strip(),
                "source_id": article["id"]
            }
            # 빈 항목 방지
            if clean["definition_term"] and clean["full_term"]:
                results.append(clean)

    except Exception as e:
        print(f"[ERROR] {article.get('id','?')}: {e}")

# 8) 저장: 최종 결과는 '리스트'
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ 이하 정의 추출 완료: {len(results)}개 항목 → {output_path}")