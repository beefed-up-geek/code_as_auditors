import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

CURRENT_DIR = Path(__file__).resolve().parent
METHOD_DIR = CURRENT_DIR.parent
PROJECT_ROOT = METHOD_DIR.parent

if __package__ is None or __package__ == "":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from method.utils.llm_interface import llm_response

CASE_PATH = PROJECT_ROOT / "dataset/PIPA/cases/cases.jsonl"
CODE_DIR_NAME = "20251104_184338_5_5_5"
CODE_DIR = METHOD_DIR / "outputs" / "legal_code_output" / CODE_DIR_NAME
OUTPUT_DIR = METHOD_DIR / "outputs" / "case_code_output"
ANSWER_MODEL = "gpt-5"
RANDOM_SEED = 42
NUM_TEST_DATA = 20
DEBUG = True

def tick_checklist(case: str, checklist: str):
    sys_prompt = '''You are a business expert. 
    You will be given a business case text and a question.
    Understand the text and answer the question.
    If the given text does not include the answer to the question, answer true or false in the direction of legality.
    Answer only True or False according to the JSON format below.
    {
        "answer": True or False
    }'''

    usr_prompt = f'''[business case text]
    {case}

    [question]
    {checklist}

    [Answer in JSON format]'''

    result = llm_response(ANSWER_MODEL, sys_prompt, usr_prompt)
    return result["answer"]

def get_case_data(num: Optional[int] = None):
    if num is None:
        num = NUM_TEST_DATA
    with CASE_PATH.open("r", encoding="utf-8") as case_file:
        cases = [json.loads(line) for line in case_file if line.strip()]

    rng = random.Random(RANDOM_SEED)
    if num >= len(cases):
        return cases
    return rng.sample(cases, num)

def generate_case_specific_codes(case, code_dir: Optional[Path] = None):
    """
    Populate the legal code scaffold with case-specific answers and write it to
    the case evaluation output directory.
    """
    template_dir = Path(code_dir) if code_dir is not None else Path(CODE_DIR)
    if not template_dir.is_absolute():
        template_dir = METHOD_DIR / template_dir

    template_path = template_dir / "code.py"
    if not template_path.exists():
        raise FileNotFoundError(f"Template code.py not found in {template_dir}")

    template_text = template_path.read_text(encoding="utf-8")
    original_has_trailing_newline = template_text.endswith("\n")

    case_id = str(case.get("case_id", "UnknownCase"))
    case_id_literal = json.dumps(case_id, ensure_ascii=False)
    template_text = re.sub(
        r'CASE_ID\s*=\s*(".*?"|\'.*?\')',
        f"CASE_ID = {case_id_literal}",
        template_text,
        count=1,
    )

    case_text_parts = [
        f"case_id: {case_id}",
        f"business: {case.get('business', '')}",
        f"violated_articles: {json.dumps(case.get('violated_articles', []), ensure_ascii=False)}",
        f"source_path: {case.get('source_path', '')}",
        f"content:\n{case.get('content', '')}",
    ]
    checklist_context = "\n".join(case_text_parts)

    checklist_block_pattern = re.compile(
        r'(# --- Checklist variables start ---\r?\n)(?P<body>.*?)(# --- Checklist variables end ---)',
        re.DOTALL,
    )
    block_match = checklist_block_pattern.search(template_text)
    if not block_match:
        raise ValueError("Checklist block not found in template.")

    checklist_body = block_match.group("body")
    var_pattern = re.compile(
        r'^(?P<indent>\s*)(?P<var>[A-Z0-9_]+)\s*=\s*(True|False)'
        r'(?P<suffix>\s*(#\s*(?P<question>.*))?)$',
        re.MULTILINE,
    )

    matches = [m for m in var_pattern.finditer(checklist_body) if m.group("question")]
    total_checklist_vars = len(matches)
    if DEBUG:
        print(f"\t↳ Checklist variables to process: {total_checklist_vars}")

    checklist_cache = {}
    processed_count = 0

    def replace_var(match: re.Match) -> str:
        nonlocal processed_count
        question = match.group("question")
        if not question:
            return match.group(0)

        processed_count += 1
        var_name = match.group("var")

        question_key = question.strip()
        if question_key not in checklist_cache:
            try:
                answer = tick_checklist(checklist_context, question_key)
                if DEBUG:
                    print(f"\t↳ Processing checklist variable {processed_count}/{total_checklist_vars}: {var_name} = {answer}")
            except Exception as exc:  # pragma: no cover - defensive fallback
                if DEBUG:
                    print(f"\t↳[warning] tick_checklist failed for {var_name}: {exc}")
                answer = False
            if isinstance(answer, str):
                normalized = answer.strip().lower()
                answer_bool = normalized == "true"
            else:
                answer_bool = bool(answer)
            checklist_cache[question_key] = answer_bool

        answer_bool = checklist_cache[question_key]
        bool_literal = "True" if answer_bool else "False"
        return f"{match.group('indent')}{var_name} = {bool_literal}{match.group('suffix') or ''}"

    updated_body = var_pattern.sub(replace_var, checklist_body)
    updated_text = (
        template_text[:block_match.start("body")]
        + updated_body
        + template_text[block_match.end("body"):]
    )
    if original_has_trailing_newline and not updated_text.endswith("\n"):
        updated_text += "\n"

    output_root = OUTPUT_DIR / template_dir.name
    output_root.mkdir(parents=True, exist_ok=True)

    safe_case_id = case_id.replace("/", "_").replace("\\", "_")
    output_path = output_root / f"{safe_case_id}.py"
    output_path.write_text(updated_text, encoding="utf-8")
    if DEBUG:
        print(f"\t↳ Saved case-specific code to {output_path}")
    return output_path

# python case_specific_code_builder.py --code-dir /Users/taeyoonkwack/Documents/code_as_auditors/method/outputs/legal_code_output/20251105_222407_5_5mini_10
def main():
    global ANSWER_MODEL, NUM_TEST_DATA, CODE_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Generate case-specific code from legal templates.")
    parser.add_argument(
        "--answer-model",
        default=ANSWER_MODEL,
        help="LLM model name to answer checklist questions.",
    )
    parser.add_argument(
        "--num-test-data",
        type=int,
        default=NUM_TEST_DATA,
        help="Number of cases to sample from the dataset.",
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=CODE_DIR,
        help="Directory containing the legal code template.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where generated case-specific code will be written.",
    )
    args = parser.parse_args()

    ANSWER_MODEL = args.answer_model
    NUM_TEST_DATA = args.num_test_data

    code_dir_path = args.code_dir
    if not code_dir_path.is_absolute():
        code_dir_path = METHOD_DIR / code_dir_path
    CODE_DIR = code_dir_path

    output_dir_path = args.output_dir
    if not output_dir_path.is_absolute():
        output_dir_path = METHOD_DIR / output_dir_path
    OUTPUT_DIR = output_dir_path

    cases = get_case_data()
    print(f"Loaded {len(cases)} cases from {CASE_PATH}")
    print(f"Loaded {len(cases)} cases from {CASE_PATH}")
    for i in range(len(cases)):
        print(f"Building Code for {i}/{len(cases)}")
        generate_case_specific_codes(cases[i])

if __name__ == "__main__":
    main()

