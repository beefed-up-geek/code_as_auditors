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
input_path = "/Users/taeyoonkwack/Documents/code_as_auditors/dataset/PIPA/law/law.json"
output_path = input_path.replace(".json", "_definition.json")

# 3️⃣ 데이터 로드
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 4️⃣ parent → children 매핑
children_map = defaultdict(list)
for item in data:
    if "parent" in item:
        children_map[item["parent"]].append(item)

# 5️⃣ 시스템 프롬프트 (GPT는 definition_term, full_term만 반환)
SYSTEM_PROMPT = """너는 한국 법령 텍스트를 분석하는 전문가이다.
아래 입력은 한 조항과 그 하위 항목(항, 호, 목 등)을 모두 포함한다.
입력 텍스트에서 ‘(이하 “…이라 한다)’ 또는 ‘란 …을 말한다’ 형태의 정의 구문을 찾아,
아래 JSON 배열로 반환하라.

규칙:
- 각 정의는 {"definition_term": "약칭", "full_term": "정의 전체 문구"} 형식이다.
- full_term은 원문에서 정의된 개념 전체를 자연스럽게 포함해야 한다.
- 법 조항(예: '법 제XX조')이 있으면 반드시 그대로 포함한다.
- 여러 개가 있으면 리스트로 모두 반환하라.
- 정의가 없으면 빈 리스트([])를 반환하라.
- 오직 JSON 배열만 반환해야 하며, 추가 설명문은 절대 포함하지 마라.

---

### Few-shot 예시 1

입력:
[조항] 제49조
[내용]
이 법은 법 제49조에 따른 집단분쟁조정(이하 '집단분쟁조정'이라 한다) 절차를 규정한다.

출력:
[
  {
    "definition_term": "집단분쟁조정",
    "full_term": "법 제49조에 따른 집단분쟁조정"
  }
]

---

### Few-shot 예시 2

입력:
[조항] 제3조
[내용]
국무총리 소속으로 개인정보 보호위원회(이하 '보호위원회'라 한다)를 둔다.

출력:
[
  {
    "definition_term": "보호위원회",
    "full_term": "국무총리 소속 개인정보 보호위원회"
  }
]

---

### Few-shot 예시 3

입력:
[조항] 제2조 제1의2호
[내용]
“가명처리”란 개인정보의 일부를 삭제하거나 일부 또는 전부를 대체하는 등의 방법으로 추가 정보가 없이는 특정 개인을 알아볼 수 없도록 처리하는 것을 말한다.

출력:
[
  {
    "definition_term": "가명처리",
    "full_term": "개인정보의 일부를 삭제하거나 일부 또는 전부를 대체하는 등의 방법으로 추가 정보가 없이는 특정 개인을 알아볼 수 없도록 처리하는 것"
  }
]
"""

# 6️⃣ content 수집 함수 (재귀적으로 하위 포함)
def collect_all_content(item_id):
    contents = []
    for child in children_map.get(item_id, []):
        if child.get("content"):
            contents.append(f"{child['id']}: {child['content']}")
        contents.extend(collect_all_content(child["id"]))
    return contents

# 7️⃣ GPT 호출 및 결과 처리
results = []

for item in tqdm(data, desc="Extracting definitions"):
    if item.get("class") == "조":
        # ---- 조 및 하위 항목 묶기 ----
        lines = [f"[조항] {item['id']}"]
        if item.get("title"):
            lines.append(f"[제목] {item['title']}")
        if item.get("content"):
            lines.append(f"[내용]\n{item['content']}")

        sub_contents = collect_all_content(item["id"])
        if sub_contents:
            lines.append("[하위항목]")
            lines.extend(sub_contents)

        merged_text = "\n".join(lines)

        try:
            # ---- GPT-5 호출 ----
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                #reasoning_effort="low",  # ✅ 최소 reasoning
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": merged_text}
                ],
                response_format={"type": "json_object"}  # ✅ JSON 강제
            )

            # ---- 결과 파싱 ----
            output_json = response.choices[0].message.content
            parsed = json.loads(output_json)
            if isinstance(parsed, dict):
                parsed = [parsed]

            # ---- source_id 추가 ----
            for p in parsed:
                p["source_id"] = item["id"]
            results.extend(parsed)

        except Exception as e:
            print(f"[ERROR] {item.get('id', '?')}: {e}")

# 8️⃣ 결과 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ 정의 추출 완료: {len(results)}개 정의 → {output_path}")