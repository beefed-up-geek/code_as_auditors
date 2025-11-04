import json
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from method.utils.law_parser import full_law_json

INPUT_PSEUDOCODE_DIR = "./code_output/20251104_053331_gpt5"
DEBUG = True


@dataclass
class LawNode:
    node_id: str
    data: Dict[str, Any]
    pseudocode: Dict[str, Any] = field(default_factory=dict)
    children: List["LawNode"] = field(default_factory=list)
    parent: Optional["LawNode"] = field(default=None, repr=False, compare=False)

    def add_child(self, child: "LawNode") -> None:
        child.parent = self
        self.children.append(child)

    @property
    def condition_pseudocode(self) -> Optional[str]:
        return self.pseudocode.get("condition_pseudocode")

    @property
    def legal_pseudocode(self) -> Optional[str]:
        return self.pseudocode.get("legal_pseudocode")

    @property
    def action_pseudocode(self) -> Optional[str]:
        return self.pseudocode.get("action_pseudocode")


def resolve_pseudocode_dir(path: Optional[Path] = None) -> Path:
    default_dir = Path(__file__).resolve().parent / INPUT_PSEUDOCODE_DIR
    target_dir = default_dir if path is None else Path(path)
    if not target_dir.exists():
        raise FileNotFoundError(f"Could not find pseudocode directory at {target_dir}")
    return target_dir


def load_law_code(path: Optional[Path] = None, filename: str = "law_code.json") -> List[Dict[str, Any]]:
    base_path = resolve_pseudocode_dir(path)
    filepath = base_path / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find law code file at {filepath}")
    with filepath.open(encoding="utf-8") as json_file:
        payload = json.load(json_file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {filepath}, received {type(payload).__name__}")
    return payload


def load_variables(path: Optional[Path] = None, filename: str = "variables.json") -> List[Dict[str, Any]]:
    base_path = resolve_pseudocode_dir(path)
    filepath = base_path / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find variables file at {filepath}")
    with filepath.open(encoding="utf-8") as json_file:
        payload = json.load(json_file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {filepath}, received {type(payload).__name__}")
    return payload


def _normalize_question(question: str) -> str:
    return " ".join(question.split())


def build_variable_declarations(variable_entries: List[Dict[str, Any]]) -> List[str]:
    declarations: List[str] = []
    for entry in variable_entries:
        var_name = (entry.get("variable") or "").strip()
        if not var_name:
            continue
        question = entry.get("question") or ""
        question_text = _normalize_question(str(question))
        comment = f"  # {question_text}" if question_text else ""
        declarations.append(f"{var_name} = False{comment}")
    return declarations


def generate_code_source(path: Optional[Path] = None) -> str:
    variables = load_variables(path)
    declarations = build_variable_declarations(variables)

    lines: List[str] = []
    lines.append("# Auto-generated code scaffold")
    lines.append("from method.utils.law_parser import full_law_json")
    lines.append("")
    lines.extend(declarations)
    if declarations:
        lines.append("")
    lines.append("def build_law_variables():")
    lines.append("    law_variables = {}")
    lines.append("    for entry in full_law_json():")
    lines.append("        var_name = entry.get('var_name')")
    lines.append("        if not var_name:")
    lines.append("            continue")
    lines.append("        law_variables[var_name] = {'condition': False, 'legal': True}")
    lines.append("    return law_variables")
    lines.append("")
    lines.append("LAW_VARIABLES = build_law_variables()")
    lines.append("")
    return "\n".join(lines)


def write_generated_code(path: Optional[Path] = None, filename: str = "code.py") -> Path:
    base_path = resolve_pseudocode_dir(path)
    code_source = generate_code_source(base_path)
    target_path = base_path / filename
    target_path.write_text(code_source, encoding="utf-8")
    return target_path


def build_law_tree(path: Optional[Path] = None) -> LawNode:
    records = load_law_code(path)
    node_index: Dict[str, LawNode] = {}
    for record in records:
        node_id = record.get("id")
        if not node_id:
            raise ValueError("Encountered record without an 'id' field.")
        if node_id in node_index:
            raise ValueError(f"Duplicate node id detected: {node_id}")
        pseudocode_payload = record.get("pseudocode") or {}
        if not isinstance(pseudocode_payload, dict):
            pseudocode_payload = {}
        node_index[node_id] = LawNode(
            node_id=node_id,
            data=record,
            pseudocode=dict(pseudocode_payload),
        )

    root = LawNode(node_id="ROOT", data={"id": "ROOT", "class": "root"})
    for record in records:
        node = node_index[record["id"]]
        if record.get("class") == "조":
            root.add_child(node)
            continue

        parent_id = record.get("parent")
        parent = node_index.get(parent_id)
        if parent is None:
            parent = root
            if DEBUG:
                print(f"[build_law_tree] Missing parent '{parent_id}' for node '{node.node_id}', attached to root.")
        parent.add_child(node)

    return root


def visualize_tree(root: LawNode) -> str:
    lines: List[str] = []
    lines.append("Tree layout:")

    def render(node: LawNode, prefix: str = "", is_last: bool = True) -> None:
        connector = "└─ " if is_last else "├─ "
        label = node.node_id
        if prefix:
            lines.append(f"{prefix}{connector}{label}")
        else:
            lines.append(label)
        new_prefix = f"{prefix}{'   ' if is_last else '│  '}"
        for idx, child in enumerate(node.children):
            render(child, new_prefix, idx == len(node.children) - 1)

    render(root)

    return "\n".join(lines)

def main():
    tree_root = build_law_tree()
    visualization = visualize_tree(tree_root)
    print(visualization)
    generated_path = write_generated_code()
    print(f"\nGenerated code stub written to: {generated_path}")
    return

if __name__ == "__main__":
    main()
