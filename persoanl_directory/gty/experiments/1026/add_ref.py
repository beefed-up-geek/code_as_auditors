import os
import json
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# -------------------- 환경 설정 --------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------- 하이퍼파라미터 --------------------
# 현재 처리 중인 데이터의 법 종류 지정 ("개인정보보호법" 또는 "시행령" 등)
LAW_TYPE = "시행령"

# 입력 및 출력 경로
INPUT_PATH = "/Users/taeyoonkwack/Documents/code_as_auditors/dataset/PIPA/law/decree.json"
OUTPUT_PATH = INPUT_PATH.replace(".json", "_addref.json")

# -------------------- Few-shot 예시 --------------------
few_shots = [
    {
        "input": {
            "id": "제4조의2 제2호",
            "class": "호",
            "content": "법 제40조제1항에 따른 개인정보 분쟁조정위원회(이하 “분쟁조정위원회”라 한다)가 조정 하는 사항과 관련된 업무"
        },
        "output": [
            {"law": "개인정보보호법", "id": "제40조 제1항"}
        ]
    },
    {
        "input": {
            "id": "제5조 제2항",
            "class": "항",
            "content": "제1항에 따라 전문위원회를 두는 경우 각 전문위원회는 위원장 1명을 포함한 20명 이내의 위원으로 성별을 고려하여 구성하되, 전문위원회 위원은 다음 각 호의 사람 중에서 보호위원회 위원장이 임명하거나 위촉하고, 전문위원회 위원장은 보호위원회 위원장이 전문위원회 위원 중에서 지명한다."
        },
        "output": [
            {"law": LAW_TYPE, "id": "제5조 제1항"}
        ]
    },
    {
        "input": {
            "id": "제60조의4 제2항",
            "class": "항",
            "content": "보호위원회는 법 제64조의2제1항에 따른 과징금을 「행정기본법」 제29조 및 같은 법 시행령 제7조에 따라 분할 납부하게 하는 경우에는 각 분할된 납부기한 간의 간격은 6개월을초과할 수 없으며, 분할 횟수는 6회를 초과할 수 없다."
        },
        "output": [
            {"law": "행정기본법", "id": "제29조"},
            {"law": "시행령", "id": "제7조"}
        ]
    }
]

few_shots = [
    {
        "input": {
            "id": "제4조의2 제2호",
            "class": "호",
            "content": "법 제40조제1항에 따른 개인정보 분쟁조정위원회(이하 “분쟁조정위원회”라 한다)가 조정 하는 사항과 관련된 업무"
        },
        "output": [
            {"law": "개인정보보호법", "id": "제40조 제1항"}
        ]
    },
    {
        "input": {
            "id": "제5조 제2항",
            "class": "항",
            "content": "제1항에 따라 전문위원회를 두는 경우 각 전문위원회는 위원장 1명을 포함한 20명 이내의 위원으로 성별을 고려하여 구성하되, 전문위원회 위원은 다음 각 호의 사람 중에서 보호위원회 위원장이 임명하거나 위촉하고, 전문위원회 위원장은 보호위원회 위원장이 전문위원회 위원 중에서 지명한다."
        },
        "output": [
            {"law": "개인정보보호법", "id": "제5조 제1항"}
        ]
    },
    {
        "input": {
            "id": "제60조의4 제2항",
            "class": "항",
            "content": "보호위원회는 법 제64조의2제1항에 따른 과징금을 「행정기본법」 제29조 및 같은 법 시행령 제7조에 따라 분할 납부하게 하는 경우에는 각 분할된 납부기한 간의 간격은 6개월을초과할 수 없으며, 분할 횟수는 6회를 초과할 수 없다."
        },
        "output": [
            {"law": "행정기본법", "id": "제29조"},
            {"law": "시행령", "id": "제7조"}
        ]
    }
]

# -------------------- 프롬프트 생성 --------------------
def build_prompt(law_item):
    examples_text = ""
    for ex in few_shots:
        examples_text += (
            f"\n입력:\n{json.dumps(ex['input'], ensure_ascii=False, indent=2)}"
            f"\n출력:\n{json.dumps(ex['output'], ensure_ascii=False, indent=2)}\n"
        )

    prompt = f"""
너는 한국 법령 조항 내에서 타 법령 조항의 참조를 자동으로 식별하는 시스템이야.

주어진 content 안에서 다른 법 조항이 언급된 경우 이를 JSON 배열로 반환해.
각 항목은 다음 형식을 따라야 해:
[
  {{"law": "법명", "id": "조항 식별자"}}
]

규칙:
- "법"이라고만 쓰인 경우: "개인정보보호법"
- "시행령"이라고만 쓰인 경우: "시행령"
- 구체적인 법명이 명시된 경우(예: 「행정기본법」): 해당 법명을 그대로 사용
- 조, 항, 호, 목 등 상위 구조나 법명이 명시되지 않은 경우: 현재 처리 중인 데이터({LAW_TYPE}) 및 계층 구조를 따라 최대한 일관되게 해석
- 아무 참조가 없는 경우에는 빈 배열 [] 반환

few-shot 예시:
{examples_text}

이제 아래 항목에 대해 같은 방식으로 분석해줘.

입력:
{json.dumps(law_item, ensure_ascii=False, indent=2)}

출력(JSON 배열만 반환):
"""
    return prompt.strip()


# -------------------- GPT 호출 함수 --------------------
def extract_references(law_item):
    prompt = build_prompt(law_item)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "너는 한국 법령 조항에서 타 법령 참조를 추출하는 전문가야."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "text"}  # ✅ 수정된 부분
    )

    content = response.choices[0].message.content.strip()

    # 모델이 JSON 배열로 출력한다고 가정하고 파싱 시도
    try:
        refs = json.loads(content)
        if isinstance(refs, list):
            return refs
        else:
            return []
    except Exception:
        # 혹시 JSON 형식이 아닐 경우 자동 복구 시도
        try:
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1:
                return json.loads(content[start:end+1])
        except Exception:
            pass
        return []


# -------------------- 메인 루프 --------------------
def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in tqdm(data, desc="Adding references"):
        refs = extract_references(item)
        if refs:
            item["reference"] = refs

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
