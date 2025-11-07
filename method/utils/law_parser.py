from __future__ import annotations

import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set

LAW_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "dataset" / "PIPA" / "law" / "law_LRX.json"
)

_ARTICLE_ID_PATTERN = re.compile(r"(제\d+조(?:의\d+)?)")


def _extract_article_id(raw_id: str) -> str:
    if not raw_id:
        return ""
    match = _ARTICLE_ID_PATTERN.search(raw_id)
    return match.group(1) if match else raw_id.strip()

@lru_cache(maxsize=1)
def _load_law_entries() -> List[Dict[str, object]]:
    with LAW_DATASET_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)

@lru_cache(maxsize=1)
def _build_indexes() -> Dict[str, Dict[str, object]]:
    entries = _load_law_entries()
    id_index: Dict[str, Dict[str, object]] = {}
    children_index: Dict[str, List[str]] = defaultdict(list)

    for entry in entries:
        entry_id = entry["id"]
        id_index[entry_id] = entry

    for entry in entries:
        parent_id = entry.get("parent")
        if parent_id:
            children_index[parent_id].append(entry["id"])

    return {"entries": entries, "id_index": id_index, "children_index": children_index}

def _collect_subtree(root_id: str) -> List[Dict[str, object]]:
    indexes = _build_indexes()
    id_index = indexes["id_index"]
    children_index = indexes["children_index"]

    if root_id not in id_index:
        raise KeyError(f"Law id '{root_id}' not found in dataset.")

    result: List[Dict[str, object]] = []

    def _visit(node_id: str) -> None:
        node = id_index[node_id]
        result.append(node)
        for child_id in children_index.get(node_id, []):
            _visit(child_id)

    _visit(root_id)
    return [{**node} for node in result]

def article_to_json_list(article_id: str) -> List[Dict[str, object]]:
    indexes = _build_indexes()
    id_index = indexes["id_index"]

    if article_id not in id_index:
        raise KeyError(f"Article '{article_id}' not found.")

    article = id_index[article_id]
    if article.get("class") != "조":
        raise ValueError(f"'{article_id}' is not classified as '조'.")

    return _collect_subtree(article_id)

def id_to_formated_string(law_id: str) -> str:
    indexes = _build_indexes()
    id_index = indexes["id_index"]
    if law_id not in id_index:
        raise KeyError(f"Law id '{law_id}' not found.")
    node = id_index[law_id]
    entry_id = node["id"]
    var_name = (node.get("var_name") or "").strip()
    title = (node.get("title") or "").strip()
    content = (node.get("content") or "").strip()
    parts: List[str] = [f"{entry_id} [{var_name}]:"]
    if title:
        parts.append(f"({title})")
    if content:
        parts.append(content)
    return " ".join(parts)

def id_to_full_article_text(law_id: str) -> str:
    indexes = _build_indexes()
    id_index = indexes["id_index"]

    if law_id not in id_index:
        raise KeyError(f"Law id '{law_id}' not found.")

    current = id_index[law_id]
    while current.get("class") != "조":
        parent_id = current.get("parent")
        if not parent_id:
            raise ValueError(
                f"Unable to locate parent article ('조') for law id '{law_id}'."
            )
        if parent_id not in id_index:
            raise KeyError(f"Parent id '{parent_id}' not found in dataset.")
        current = id_index[parent_id]

    article_nodes = _collect_subtree(current["id"])
    lines = [id_to_formated_string(node["id"]) for node in article_nodes]
    return "\n".join(lines)

def id_to_ref_string(law_id: str) -> str:
    indexes = _build_indexes()
    id_index = indexes["id_index"]

    if law_id not in id_index:
        raise KeyError(f"Law id '{law_id}' not found.")

    entry = id_index[law_id]
    references = entry.get("reference") or []

    normalized_ids: List[str] = []
    seen: Set[str] = set()
    for reference in references:
        if not isinstance(reference, dict):
            continue
        if reference.get("law") != "개인정보보호법":
            continue
        ref_id = _extract_article_id(str(reference.get("id", "")).strip())
        if not ref_id or ref_id in seen:
            continue
        seen.add(ref_id)
        normalized_ids.append(ref_id)

    base_id = _extract_article_id(law_id)
    filtered_ids = [ref_id for ref_id in normalized_ids if ref_id and ref_id != base_id]

    lines: List[str] = []
    for ref_id in filtered_ids:
        try:
            ref_text = id_to_full_article_text(ref_id)
        except KeyError:
            continue
        if ref_text:
            lines.extend(ref_text.splitlines())

    return "\n".join(lines)

def full_law_json() -> List[Dict[str, object]]:
    """Return a shallow copy of every law entry as a JSON-like list."""
    entries = _load_law_entries()
    # Return copies so callers cannot mutate the cached dataset in-place.
    return [{**entry} for entry in entries]

def main() -> None:
    sample_article_id = "제26조"
    sample_law_id = "제26조 제7항"
    expected_formatted_string = (
        "제26조 제7항 [LAW_A26_P7]: 수탁자가 위탁받은 업무와 관련하여 개인정보를 처리하는 과정에서 이 법을 위반하여 발생한 손해배상책임에 대하여는 수탁자를 개인정보처리자의 소속 직원으로 본다."
    )

    article_entries = article_to_json_list(sample_article_id)
    print(f"{sample_article_id} subtree entry count: {len(article_entries)}")
    print(json.dumps(article_entries, ensure_ascii=False, indent=2))

    print("\n---\n")
    formatted_string = id_to_formated_string(sample_law_id)
    assert (
        formatted_string == expected_formatted_string
    ), f"id_to_formated_string('{sample_law_id}') returned unexpected value."
    print(formatted_string)

    print("\n---\n")
    print(id_to_full_article_text(sample_law_id))

    print("\n---\n")
    ref_sample_id = "제7조의9 제1항 제2호"
    print(f"{ref_sample_id} references:\n{id_to_ref_string(ref_sample_id)}")

if __name__ == "__main__":
    main()
