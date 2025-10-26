import re, json

path = "/Users/taeyoonkwack/Documents/code_as_auditors/dataset/PIPA/law/original/개인정보 보호법 시행령(대통령령).txt"

with open(path, encoding="utf-8") as f:
    text = f.read()

# “이하 “○○”이라 한다” 또는 “이하 ‘○○’이라 한다” 형태 모두 대응
pattern = r'([^\n]*?)\(이하[ \n]*[“"](.*?)[”"’‘][ \n]*이라 한다\)'
matches = re.finditer(pattern, text)

result = []
for i, m in enumerate(matches, 1):
    full_term = m.group(1).strip().replace("(", "").replace(")", "")
    defined_term = m.group(2).strip()
    result.append({"id": i, "defined_term": defined_term, "full_term": full_term})

with open("defined_terms.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(len(result), "개 추출 완료")