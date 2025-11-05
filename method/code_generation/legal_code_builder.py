import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from method.utils.law_parser import full_law_json

INPUT_PSEUDOCODE_DIR = "../outputs/legal_code_output/20251105_010712_5_5mini_15"
DEBUG = False


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


def generate_user_variable_lines(path: Optional[Path] = None) -> List[str]:
    variables = load_variables(path)
    return build_variable_declarations(variables)


def generate_law_variable_lines() -> List[str]:
    declarations: List[str] = []
    seen: set[str] = set()
    for entry in full_law_json():
        var_name = (entry.get("var_name") or "").strip()
        if not var_name or var_name in seen:
            continue
        seen.add(var_name)
        declarations.append(f"{var_name} = {{'condition': False, 'legal': True}}")
    return declarations


def _function_name_for(node: LawNode) -> str:
    var_name = node.data.get("var_name", "").strip()
    if not var_name:
        raise ValueError(f"Node '{node.node_id}' lacks 'var_name', cannot generate function name.")
    return f"visit_{var_name}"


def _should_emit_action(action: Optional[str]) -> bool:
    return bool(action and action.strip())


def generate_traversal_code(root: LawNode, starting_line: int = 0) -> Tuple[List[str], List[Dict[str, Optional[int]]]]:
    lines: List[str] = []
    metadata_entries: List[Dict[str, Optional[int]]] = []
    current_line = starting_line

    def add_line(text: str) -> int:
        nonlocal current_line
        current_line += 1
        lines.append(text)
        return current_line

    def emit_node(node: LawNode, add_leading_blank: bool = True) -> None:
        function_name = _function_name_for(node)
        var_name = node.data["var_name"]
        metadata_entry: Dict[str, Optional[int]] = {
            "id": node.node_id,
            "condition_pseudocode": None,
            "action_pseudocode": None,
            "legal_pseudocode": None,
        }
        metadata_entries.append(metadata_entry)

        if add_leading_blank:
            add_line("")

        add_line(f"def {function_name}():")
        indent = "    "
        try_indent = indent
        body_indent = indent * 2
        nested_indent = indent * 3
        deepest_indent = indent * 4

        condition_expr = (node.condition_pseudocode or "").strip()
        action_expr = node.action_pseudocode or ""
        legal_expr = (node.legal_pseudocode or "").strip()

        var_literal = repr(var_name)

        add_line(f"{try_indent}try:")
        has_body = False

        if condition_expr:
            metadata_entry["condition_pseudocode"] = add_line(f"{body_indent}if {condition_expr}:")
            has_body = True
            add_line(f"{nested_indent}{var_name}['condition'] = True")
            for child in node.children:
                child_var_name = child.data.get("var_name")
                if not child_var_name:
                    continue
                add_line(f"{nested_indent}{_function_name_for(child)}()")

            if _should_emit_action(action_expr):
                metadata_entry["action_pseudocode"] = None
                for idx, action_line in enumerate(action_expr.splitlines()):
                    if not action_line.strip():
                        continue
                    action_line_no = add_line(f"{nested_indent}{action_line.rstrip()}")
                    if metadata_entry["action_pseudocode"] is None:
                        metadata_entry["action_pseudocode"] = action_line_no

            if legal_expr:
                metadata_entry["legal_pseudocode"] = add_line(f"{nested_indent}if not ({legal_expr}):")
                add_line(f"{deepest_indent}{var_name}['legal'] = False")
        else:
            # No condition expression; still recurse into children for completeness
            for child in node.children:
                child_var_name = child.data.get("var_name")
                if not child_var_name:
                    continue
                add_line(f"{body_indent}{_function_name_for(child)}()")
                has_body = True

        if not has_body:
            add_line(f"{body_indent}pass")

        add_line(f"{try_indent}except Exception:")
        add_line(f"{body_indent}record_error({var_literal})")

        for child in node.children:
            emit_node(child)

    for idx, child in enumerate(root.children):
        emit_node(child, add_leading_blank=idx != 0)

    add_line("")
    add_line("def main():")
    indent = "    "
    if root.children:
        for child in root.children:
            add_line(f"{indent}{_function_name_for(child)}()")
    else:
        add_line(f"{indent}run_law_tree()")
    add_line(f"{indent}illegal_law_vars = sorted(")
    add_line(f"{indent}    name")
    add_line(f"{indent}    for name, value in globals().items()")
    add_line(f'{indent}    if name.startswith("LAW_") and isinstance(value, dict) and value.get("legal") is False')
    add_line(f"{indent})")
    add_line(f"{indent}if illegal_law_vars:")
    add_line(f'{indent}    record_result("Non-compliant law variables: " + ", ".join(illegal_law_vars))')
    add_line(f"{indent}else:")
    add_line(f'{indent}    record_result("No non-compliant law variables detected.")')
    add_line(f"{indent}return illegal_law_vars")

    add_line("")
    add_line('if __name__ == "__main__":')
    add_line("    try:")
    add_line("        main()")
    add_line("    except Exception:")
    add_line("        record_error('MAIN')")
    add_line("    finally:")
    add_line("        write_output()")

    return lines, metadata_entries


def generate_code_source(root: LawNode, path: Optional[Path] = None) -> Tuple[str, List[Dict[str, Optional[int]]]]:
    user_variable_lines = generate_user_variable_lines(path)
    law_variable_lines = generate_law_variable_lines()
    lines: List[str] = []
    lines.append("# Auto-generated code scaffold")
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("")
    lines.append('CASE_ID = "Default"')
    lines.append("RESULT_LINES = []")
    lines.append("ERROR_LINES = []")
    lines.append("")
    lines.append("def record_result(message):")
    lines.append("    RESULT_LINES.append(str(message))")
    lines.append("")
    lines.append("def record_error(law_var):")
    lines.append('    ERROR_LINES.append(f"{law_var}에서 시행 오류가 발생했다")')
    lines.append("")
    lines.append("def write_output():")
    lines.append("    result_path = Path(__file__).resolve().with_name(f'{CASE_ID}.txt')")
    lines.append('    lines = [f"CASE_ID: {CASE_ID}"]')
    lines.append('    lines.append("")')
    lines.append('    lines.append("Results:")')
    lines.append("    if RESULT_LINES:")
    lines.append("        lines.extend(RESULT_LINES)")
    lines.append("    else:")
    lines.append('        lines.append("None")')
    lines.append('    lines.append("")')
    lines.append('    lines.append("Errors:")')
    lines.append("    if ERROR_LINES:")
    lines.append("        lines.extend(ERROR_LINES)")
    lines.append("    else:")
    lines.append('        lines.append("None")')
    lines.append('    with result_path.open("w", encoding="utf-8") as handle:')
    lines.append('        handle.write("\\n".join(lines) + "\\n")')
    lines.append("")
    if user_variable_lines:
        lines.append("# --- Checklist variables start ---")
        lines.extend(user_variable_lines)
        lines.append("# --- Checklist variables end ---")
        lines.append("")
    lines.extend(law_variable_lines)
    if law_variable_lines:
        lines.append("")

    traversal_lines, metadata_entries = generate_traversal_code(root, starting_line=len(lines))
    lines.extend(traversal_lines)

    return "\n".join(lines), metadata_entries


def write_generated_code(root: LawNode, path: Optional[Path] = None, filename: str = "code.py") -> Tuple[Path, List[Dict[str, Optional[int]]]]:
    base_path = resolve_pseudocode_dir(path)
    code_source, metadata_entries = generate_code_source(root, base_path)
    target_path = base_path / filename
    target_path.write_text(code_source, encoding="utf-8")
    return target_path, metadata_entries


def write_metadata(metadata_entries: List[Dict[str, Optional[int]]], path: Optional[Path] = None, filename: str = "metadata.json") -> Path:
    base_path = resolve_pseudocode_dir(path)
    target_path = base_path / filename
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_entries, handle, ensure_ascii=False, indent=2)
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

def parse_cli_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate executable code from legal pseudocode artifacts.")
    parser.add_argument(
        "-d",
        "--directory",
        dest="directory",
        metavar="PATH",
        help="Directory containing law_code.json, variables.json, etc. Generated code will be written there.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_cli_arguments(argv)
    target_dir = Path(args.directory).expanduser().resolve() if args.directory else None

    tree_root = build_law_tree(target_dir)
    visualization = visualize_tree(tree_root)
    if DEBUG:
        print(visualization)
    generated_path, metadata_entries = write_generated_code(tree_root, target_dir)
    metadata_path = write_metadata(metadata_entries, target_dir)
    print(f"\nGenerated code written to: {generated_path}")
    print(f"Metadata written to: {metadata_path}")
    return

if __name__ == "__main__":
    main()
