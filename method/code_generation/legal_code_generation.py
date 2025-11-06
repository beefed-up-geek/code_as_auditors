import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from method.utils.law_parser import (
    article_to_json_list,
    id_to_full_article_text,
    id_to_formated_string,
    id_to_ref_string,
)
from method.utils.llm_interface import llm_response

DEBUG = True
DEFAULT_GENERATION_LLM = "gpt-5-mini"
DEFAULT_FEEDBACK_LLM = "gpt-5"
DEFAULT_MAX_FEEDBACK_LOOP = 15
DEFAULT_OUTPUT_DIR = (Path(__file__).resolve().parent / "../outputs/legal_code_output").resolve()

GENERATION_LLM = DEFAULT_GENERATION_LLM
FEEDBACK_LLM = DEFAULT_FEEDBACK_LLM
MAX_FEEDBACK_LOOP = DEFAULT_MAX_FEEDBACK_LOOP
ARTICLES = ["제21조", "제24조", "제24조의2", "제26조", "제29조", "제34조", "제39조의4"] #["제29조", "제26조"] # ["제29조", "제26조", "제34조", "제21조", "제24조의2", "제39조의4", "제24조"]
base_variables = [
        {
            "variable": "BUSINESS_USES_PERSONAL_INFORMATION",
            "question": "귀사는 고객 또는 이용자의 개인정보를 처리하거나 보유합니까?"
        },
        {
            "variable": "BUSINESS_OUTSOURCES_PROCESSING",
            "question": "귀사는 개인정보 처리 업무를 제3자에게 위탁합니까?"
        }
    ]
fewshot_examples = """
  제16조 제1항: 개인정보처리자는 제15조 제1항 각 호의 어느 하나에 해당하여 개인정보를 수집하는 경우에는 그 목적에 필요한 최소한의 개인정보를 수집하여야 한다.
  {
    "pseudocode": {
      "condition_pseudocode": "RECIEVE_CONSENT and (LAW_A15_P1_S1['condition'] or LAW_A15_P1_S2['condition'] or LAW_A15_P1_S3['condition'] or LAW_A15_P1_S4['condition'] or LAW_A15_P1_S5['condition'] or LAW_A15_P1_S6['condition'] or LAW_A15_P1_S7['condition'])",
      "legal_pseudocode": "BUSINESS_COLLECTS_MINIMUM_ONLY",
      "action_pseudocode": ""
    },
    "added_variables": [
      {
        "variable": "BUSINESS_COLLECTS_MINIMUM_ONLY",
        "question": "귀사는 고객의 개인정보를 수집할 때 서비스 제공에 반드시 필요한 최소한의 항목만을 수집합니까? 예를 들어 불필요한 생년월일, 주소, 직업, 가족정보 등을 요구하지 않습니까?"
      }
    ]
  }

  제26조 (업무위탁에 따른 개인정보의 처리 제한)
  {
  "pseudocode": {
    "condition_pseudocode": "BUSINESS_OUTSOURCES_PROCESSING",
    "legal_pseudocode": "(LAW_A26_P1['legal'] and LAW_A26_P2['legal'] and LAW_A26_P3['legal'] and LAW_A26_P4['legal'] and LAW_A26_P5['legal'] and LAW_A26_P6['legal'] and LAW_A26_P7['legal'])",
    "action_pseudocode": ""
  },
  "added_variables": [
  ]
  }

  제2조 제1호: 개인정보’란 살아 있는 개인에 관한 정보로서 성명, 주민등록번호 및 영상 등을 통하여 개인을 식별할 수 있는 정보를 말한다.
  {
    "pseudocode": {
      "condition_pseudocode": "BUSINESS_USES_PERSONAL_INFORMATION",
      "legal_pseudocode": "True",
      "action_pseudocode": ""
    },
    "added_variables": []
  }

  {
  "pseudocode": {
    "condition_pseudocode": "USE_WITHOUT_CONSENT and RELATED_TO_ORIGINAL_PURPOSE and SAFETY_MEASURES_TAKEN and NO_DISADVANTAGE_TO_SUBJECT",
    "legal_pseudocode": "True",
    "action_pseudocode": "LAW_A15_P1['legal'] = True"
  },
  "added_variables": [
    {
      "variable": "USE_WITHOUT_CONSENT",
      "question": "귀사는 정보주체의 추가 동의 없이 개인정보를 이용하는 경우가 있습니까?"
    },
    {
      "variable": "RELATED_TO_ORIGINAL_PURPOSE",
      "question": "해당 이용이 개인정보를 처음 수집한 목적과 합리적으로 관련된 범위 내에서 이루어집니까?"
    },
    {
      "variable": "SAFETY_MEASURES_TAKEN",
      "question": "추가 이용 시 암호화 등 안전성 확보에 필요한 조치를 취하고 있습니까?"
    },
    {
      "variable": "NO_DISADVANTAGE_TO_SUBJECT",
      "question": "추가 이용으로 인해 정보주체에게 불이익이 발생하지 않도록 보장하고 있습니까?"
    }
  ]
}
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate logical encodings for PIPA articles.")
    parser.add_argument(
        "--generation-llm",
        default=DEFAULT_GENERATION_LLM,
        help=f"LLM used for generation (default: {DEFAULT_GENERATION_LLM})",
    )
    parser.add_argument(
        "--feedback-llm",
        default=DEFAULT_FEEDBACK_LLM,
        help=f"LLM used for feedback evaluation (default: {DEFAULT_FEEDBACK_LLM})",
    )
    parser.add_argument(
        "--max-feedback-loop",
        type=int,
        default=DEFAULT_MAX_FEEDBACK_LOOP,
        help=f"Maximum number of feedback iterations (default: {DEFAULT_MAX_FEEDBACK_LOOP})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store generated outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()

def _resolve_output_dir(path_value):
    path = path_value if isinstance(path_value, Path) else Path(path_value)
    path = path.expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    return path

def initial_generation(law_data, variables):
    base_var_str = json.dumps(variables, ensure_ascii=False, indent=2)
    full_article = id_to_full_article_text(law_data["id"])
    test_article = id_to_formated_string(law_data["id"])
    ref_texts = id_to_ref_string(law_data["id"])
    sys_prompt = f"""
You are a model that logically encodes each article of the Personal Information Protection Act (PIPA).
Analyze the given article and output it as a JSON in the following structure.

Each article must include the following fields: 
- pseudocode 
    - condition_pseudocode: applicability conditions (based on other articles’ conditions or business variables) 
    - legal_pseudocode: logical rule for determining legality (False means violation) 
    - action_pseudocode: actions to be taken when condition_pseudocode is True 
- added_variables: newly added variables only when strictly necessary for logical reasoning 
    - variable: a Boolean variable (True/False) representing a specific characteristic of a business 
    - question: a concrete question corresponding to the variable, allowing one to determine whether a business violates the law.  
      It must **never** quote or describe the legal text itself. 
 
Rules: 
- Every Chapter, Section, Article, Paragraph, and Subparagraph of the law must have two logical variables: 
   - condition: whether the article is applicable (True/False) 
   - legal: True if the article is complied with, False if violated 
- All variable names must follow this rule: 
   LAW_[ArticleIdentifier]['condition' or 'legal'] 
   Example: LAW_A29_P1['condition'], LAW_A29_P1['legal'] 
- When referencing another article, you must use these variable names. 
- If the legal_pseudocode of an article is False, it is considered a “violation”; if True, it is considered “compliance.” 
- If condition_pseudocode is False, that article is excluded from evaluation (not applicable). 
- Only add variables under added_variables when absolutely necessary.  
  The question must be a **specific question about the actual business activity**, never a quotation or explanation of the law. 
- Use logical operators and, or, and not in pseudocode, combining other articles’ condition/legal variables and added variables. 
- Example variable names: 
   - LAW_A15_P1 : Article 15(1) of the Personal Information Protection Act 
   - LAW_A15_P1_S2 : Article 15(1)(ii) of the Personal Information Protection Act 
- action_pseudocode is only used when state of other articles has to be changed. 
- Variable names must be written in uppercase English, with up to four words separated by underscores. 
- Questions must not contain any content from the law itself, but must instead be **specific, concrete questions** about business practices. 

Below are examples of how PIPA articles are encoded in JSON.
In these examples, if legal_pseudocode is False, it is regarded as a “violation.” (True → compliance)

[Examples]
{fewshot_examples}
"""

    usr_prompt = f"""
Below is the list of basic variables used for business evaluation.  
These variables should be prioritized when constructing logic, and new variables should only be added when absolutely necessary.

[Variable List]
{base_var_str}

[Full Article]
{full_article}

[Target Article to be Processed]
{test_article}

[References]
{ref_texts}

output only the JSON structure for {law_data["id"]}
"""
    pseudocodes = llm_response(GENERATION_LLM, sys_prompt, usr_prompt)
    return pseudocodes

def feedback(law_data, pseudocodes, variables):
    base_var_str = json.dumps(variables, ensure_ascii=False, indent=2)
    full_article = id_to_full_article_text(law_data["id"])
    target_article = id_to_formated_string(law_data["id"])
    ref_texts = id_to_ref_string(law_data["id"])
    pseudocodes_str = json.dumps(pseudocodes, ensure_ascii=False, indent=2)
    sys_prompt = f'''You are a legal logic evaluator assessing JSON results that logically encode articles of the Korean Personal Information Protection Act (PIPA).  
The evaluation target is the given article and its generated pseudocode result.  

Each generated JSON follows a standardized schema designed to encode the logical structure of a PIPA article.  
It consists of two main components:

1. **pseudocode** – defines the logical representation of the article in three parts:  
   - **condition_pseudocode**: the logical condition describing when the article applies (True/False expression).  
   - **legal_pseudocode**: the logic determining compliance; if False, it indicates a violation.  
   - **action_pseudocode**: optional logic describing inter-article actions or effects when the condition is True.  

2. **added_variables** – defines new Boolean variables used in the pseudocode, each accompanied by:  
   - **variable**: an uppercase English identifier with intuitive names(≤4 words, separated by underscores).  
   - **question**: a concrete Yes/No question about real business behavior, written without quoting or paraphrasing the legal text.  

[Examples of generated JSON]
{fewshot_examples}

Your response must be returned strictly as a valid JSON object.

Evaluation Guidelines:
Evaluate the generated result according to the following four criteria.

1. Necessity  
   Determine whether newly added variables were truly necessary.  
   - Could the new variable have been replaced by an existing base variable?  
   - Was the new variable essential to express the logic of the legal article?

   **Example:**
     - Incorrect: A new variable `GATHERS_USER_DATA` was created even though `COLLECTS_PERSONAL_INFO` already exists.  
     - Fix: Reuse the existing variable and remove the redundant definition.

2. Specificity & Clarity  
   Each variable’s question must be specific, must not quote or paraphrase the law, and must clearly assess an actual business activity.  
   - Does the question avoid vague words like “procedure,” “legality,” “reasonable,” or “related measures”? 
   - Can the meaning of each variable be easily understood from its name?
   - Does it focus on business conduct rather than legal text?  
   - Can it be answered with a clear Yes/No?

   **Example:**
     - Incorrect: “Has your company complied with the data collection procedure?” → vague (“procedure” undefined)  
     - Fix: “When collecting personal information, does your company clearly notify the data subject of the purpose and items collected?”

3. Logical Completeness
   - Assess whether each pseudocode faithfully reflects the logical structure of the article and maintains consistency without logical errors.  
   - Consider whether the pseudocode could fail to identify a violation in a hypothetical case, and specify how to improve it.
  **Example**
    - Incorrect: "legal_pseudocode": "LAW_A26_P1['condition'] and LAW_A26_P2['condition']" → wrong logic (checks applicability instead of compliance). Article 26 requires ensuring safe management clauses in outsourcing, so referencing ['condition'] can mark violations as compliant.
    - Fix: "legal_pseudocode": "LAW_A26_P1['legal'] and LAW_A26_P2['legal']" → correctly checks whether each clause was actually complied with.

4. Code Completeness  
   - Check that the pseudocode and JSON structure are syntactically and structurally complete.  
   - Ensure that pseudocode does not reference undefined variables.  
   - Confirm that each variable name in added_variables follows the rule: uppercase English words (≤4) separated by underscores.  
   - Verify that article identifiers follow the proper format.

   **Example:**
     - Incorrect: "legal_pseudocode":(not OTHER_LAW_REQUIRES_RETENTION) implies DATA_DESTROYED_WITHOUT_DELAY → "implies" does not exist in python. 
     - Fix: Change to "legal_pseudocode":OTHER_LAW_REQUIRES_RETENTION or DATA_DESTROYED_WITHOUT_DELAY

     - Incorrect: A new variable "OUTSOURCING_CONTRACT_INCLUDES_OTHER_SAFE_MANAGEMENT_ITEMS" has more than 4 words
     - Fix: Change variable name "OUTSOURCING_CONTRACT_INCLUDES_OTHER_SAFE_MANAGEMENT_ITEMS" to "OUTSOURCING_CONTRACT_SAFE_ITEMS" to ensure added_variables are less than 5 words.

   **Article Identifier Rules**
   - Base format: `LAW_A[ArticleNumber]_P[ParagraphNumber]`  
   - For subparagraphs (e.g., item (1), (2), (3)), append `_S[SubparagraphNumber]`.  
   - Examples:  
       - Article 15(1) → `LAW_A15_P1`  
       - Article 15(1)(ii) → `LAW_A15_P1_S2`  
   - All identifiers must be uppercase and, when referenced, must include either `['condition']` or `['legal']`.  
       - Example: `LAW_A15_P1['condition']`, `LAW_A29_P2_S1['legal']`
'''
    usr_prompt = f'''Assess a following pseudocode JSON representation of a Korean privacy law article.
[Existing Base Variable List]
{base_var_str}

[Full Article]
{full_article}

[Target Article Processed]
{target_article}

[References of Target Article]
{ref_texts}

[JSON to Feedback]
{pseudocodes_str}

You must return your evaluation **strictly as a valid JSON object** following the schema below.
OUTPUT SCHEMA:
{{
  "summary": "string",
  "scores": {{
    "necessity": int,
    "specificity": int,
    "logic": int
    "code": int,
  }},
  "issues": {{
    "necessity": [ "list of issue strings (why + fix)" ],
    "specificity": [ "..." ],
    "logic": [ "..." ]
    "code": [ "..." ],
  }},
  "recommendations": [
    "Each item must describe an explicit fix in one line: which variable, what to change, and the corrected example. Example: Change variable 'RECIEVE_CONSENT' question to '귀사는 고객의 개인정보를 해외로 이전하기 전에 명시적으로 동의를 받았습니까?'"
  ]
}}

Rules:
- scores should be 0~5 (5 as best)
- If no major issues exist, or if only minor **Specificity & Clarity** issues exist, summarize only as "No major issues found" and set all scores to 5 with empty issue lists.
- Recommendations must not be vague. Each one must correspond to single issues in JSON representation of Target Article
- Keep feedback concise, factual, and actionable.

Return only the output schema.
'''
    feedback_json = llm_response(FEEDBACK_LLM, sys_prompt, usr_prompt)
    return feedback_json

def regeneration_with_feedback(law_data, pseudocodes, feedback, variables):
    base_var_str = json.dumps(variables, ensure_ascii=False, indent=2)
    full_article = id_to_full_article_text(law_data["id"])
    target_article = id_to_formated_string(law_data["id"])
    ref_texts = id_to_ref_string(law_data["id"])
    pseudocodes_str = json.dumps(pseudocodes, ensure_ascii=False, indent=2)
    feedback_str = json.dumps(feedback, ensure_ascii=False, indent=2)
    sys_prompt = f"""
You are a model that logically encodes each article of the Korean Personal Information Protection Act (PIPA).  
Your task is to **regenerate an improved JSON schema** based on the previously generated pseudocode JSON and the provided evaluation feedback.
---
[Objective]
- Review the existing pseudocode JSON and the feedback identifying issues and recommendations.  
- Generate an improved, logically consistent, and syntactically valid JSON structure.  
- The output must strictly follow the schema format below and must incorporate all valid corrections suggested in the feedback.  
- You must output **only a valid JSON object** — no explanations, markdown, or comments.
---

[Output JSON Schema Format]
{{
  "pseudocode": {{
    "condition_pseudocode": "Logical condition describing when the article applies (True/False expression)",
    "legal_pseudocode": "Logical rule determining compliance (False means violation)",
    "action_pseudocode": "Optional — actions or inter-article effects executed when condition_pseudocode is True"
  }},
  "added_variables": [
    {{
      "variable": "Variable name written in uppercase English with up to four words separated by underscores",
      "question": "A concrete Yes/No question about real business behavior — must not quote or describe legal text"
    }}
  ]
}}

---
[Rules]
1. The JSON must always include both fields: `pseudocode` and `added_variables`.  
2. All pseudocode fields (`condition_pseudocode`, `legal_pseudocode`, `action_pseudocode`) must be valid Python-like logical expressions written as strings.  
3. All variable names must be written in uppercase English, limited to four words separated by underscores.  
4. Each question must describe an **actual business practice**, not the legal text itself, and must be answerable with Yes/No.  
5. `action_pseudocode` is only used when encoding interactions or dependencies between articles.  
6. When referencing other articles, follow the naming convention below:
   - Base format: `LAW_A[ArticleNumber]_P[ParagraphNumber]`
   - For subparagraphs: append `_S[SubparagraphNumber]`
   - Every reference must include either `['condition']` or `['legal']`
     - Example: `LAW_A15_P1['condition']`
7. Incorporate all feedback regarding “incorrect variable names,” “ambiguous questions,” “redundant variables,” and “logical inconsistencies.”  
8. Maintain the logical intent of the original pseudocode while correcting missing conditions, inconsistent logic, or invalid expressions.  
9. The output must consist of **only one valid JSON object**, without any markdown, explanations, or extra text.

---

[Output Goal]
- Return the **final improved JSON schema** that fully reflects the corrections indicated in the feedback.  
- The result must be logically consistent, specific, complete, and syntactically correct.
"""
    usr_prompt = f"""
[Existing Base Variable List]
{base_var_str}

[Full Article]
{full_article}

[Target Article Processed]
{target_article}

[References of Target Article]
{ref_texts}

[Previously Generated JSON Representation of Target Article]
{pseudocodes_str}

[Evaluation Feedback Identifying Issues and Recommendations]
{feedback_str}

Strictly follow the Feedback Recommendations!
Return only the improved JSON based on the Evaluation Feedback. 
"""

    new_pseudocodes = llm_response(GENERATION_LLM, sys_prompt, usr_prompt)
    return new_pseudocodes

def generate_single(law_data, variables):
    log_messages = []

    def record(message):
        log_messages.append(message)
        if DEBUG:
            print(message)

    current_result = initial_generation(law_data, variables)
    evaluated_results = []
    record("\t↳ initial generation")
    for i in range(MAX_FEEDBACK_LOOP):
        current_feedback = feedback(law_data, current_result, variables)
        evaluated_results.append(
            {
                "result": current_result,
                "feedback": current_feedback,
            }
        )
        scores = {}
        if isinstance(current_feedback, dict):
            scores = current_feedback.get("scores") or {}

        def _format_score(value):
            return "-" if value is None else str(value)

        score_text = ", ".join(
            [
                f"code: {_format_score(scores.get('code'))}",
                f"logic: {_format_score(scores.get('logic'))}",
                f"necs: {_format_score(scores.get('necessity'))}",
                f"spec: {_format_score(scores.get('specificity'))}",
            ]
        )
        record(f"\t↳ {i}th feedback ({score_text})")
        summary_text = ""
        if isinstance(current_feedback, dict):
            summary_text = current_feedback.get("summary", "") or ""

        if "No major issues found" in summary_text:
            record("\t↳ No Major Issues Found")
            return current_result, log_messages

        current_result = regeneration_with_feedback(
            law_data, current_result, current_feedback, variables
        )
        record(f"\t↳ {i}th regeneration")

    record(f"\t↳ Max iteration ({MAX_FEEDBACK_LOOP}) finished")
    if not evaluated_results:
        return current_result, log_messages

    def score_tuple(entry):
        scores = {}
        if isinstance(entry.get("feedback"), dict):
            scores = entry["feedback"].get("scores", {}) or {}
        return (
            scores.get("code", -1),
            scores.get("logic", -1),
            scores.get("necessity", -1),
            scores.get("specificity", -1),
        )
    
    best_entry = max(evaluated_results, key=score_tuple)
    return best_entry["result"], log_messages

def generate_article_list(article_list=ARTICLES):
    variables = base_variables
    datalist = [
        entry
        for article_id in article_list
        for entry in article_to_json_list(article_id)
    ]
    total_count = len(datalist)
    processed_laws = []
    existing_variables = {
        entry.get("variable")
        for entry in variables
        if isinstance(entry, dict) and entry.get("variable")
    }
    log_entries = []

    def record(message):
        log_entries.append(message)
        if DEBUG:
            print(message)

    for index, law_data in enumerate(datalist, start=1):
        law_id = law_data.get("id")
        record(f"[{index}/{total_count}] Processing {law_id}")
        generation_result, single_logs = generate_single(law_data, variables)
        log_entries.extend(single_logs)

        pseudocode = generation_result.get("pseudocode") if isinstance(generation_result, dict) else None
        added_variables = generation_result.get("added_variables") if isinstance(generation_result, dict) else []

        law_with_pseudocode = dict(law_data)
        if pseudocode is not None:
            law_with_pseudocode["pseudocode"] = pseudocode
        processed_laws.append(law_with_pseudocode)

        if isinstance(added_variables, list):
            for variable_entry in added_variables:
                if (
                    isinstance(variable_entry, dict)
                    and variable_entry.get("variable")
                    and variable_entry["variable"] not in existing_variables
                ):
                    variables.append(variable_entry)
                    existing_variables.add(variable_entry["variable"])

    return processed_laws, variables, log_entries

def main(
    article_list=ARTICLES,
    *,
    generation_llm=None,
    feedback_llm=None,
    max_feedback_loop=None,
    output_dir=None,
):
    global GENERATION_LLM, FEEDBACK_LLM, MAX_FEEDBACK_LOOP

    if generation_llm is not None:
        GENERATION_LLM = generation_llm
    if feedback_llm is not None:
        FEEDBACK_LLM = feedback_llm
    if max_feedback_loop is not None:
        if max_feedback_loop < 1:
            raise ValueError("max_feedback_loop must be a positive integer.")
        MAX_FEEDBACK_LOOP = max_feedback_loop

    processed_laws, variables, log_entries = generate_article_list(article_list=article_list)
    output_dir_path = _resolve_output_dir(output_dir or DEFAULT_OUTPUT_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = output_dir_path / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)

    law_path = target_dir / "law_code.json"
    variables_path = target_dir / "variables.json"
    log_path = target_dir / "log.txt"
    metadata_path = target_dir / "code_gen_metadata.json"

    with law_path.open("w", encoding="utf-8") as law_file:
        json.dump(processed_laws, law_file, ensure_ascii=False, indent=2)

    with variables_path.open("w", encoding="utf-8") as variables_file:
        json.dump(variables, variables_file, ensure_ascii=False, indent=2)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("\n".join(log_entries))

    metadata_payload = {
        "GENERATION_LLM": GENERATION_LLM,
        "FEEDBACK_LLM": FEEDBACK_LLM,
        "MAX_FEEDBACK_LOOP": MAX_FEEDBACK_LOOP,
        "ARTICLES": article_list,
        "OUTPUT_DIR": str(output_dir_path),
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata_payload, metadata_file, ensure_ascii=False, indent=2)

    if DEBUG:
        print(f"pseudocode saved in {law_path}")
        print(f"variables saved in {variables_path}")
        print(f"logs saved in {log_path}")
        print(f"metadata saved in {metadata_path}")

if __name__ == "__main__":
    args = parse_arguments()
    main(
        generation_llm=args.generation_llm,
        feedback_llm=args.feedback_llm,
        max_feedback_loop=args.max_feedback_loop,
        output_dir=args.output_dir,
    )
