import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

INPUT_PSEUDOCODE_DIR = Path("pseudocode_output/20251104_053331_gpt5")
DEBUG = True


@dataclass
class LawNode:
    """Lightweight tree node for law entries."""

    law_id: str
    data: Dict[str, Any]
    children: List["LawNode"] = field(default_factory=list)
    parent: Optional["LawNode"] = None

    def add_child(self, child: "LawNode") -> None:
        if child is self:
            raise ValueError("A node cannot be a child of itself.")
        if child.parent is self:
            return
        if child.parent is not None and child in child.parent.children:
            child.parent.children.remove(child)
        child.parent = self
        self.children.append(child)

    def iter_subtree(self) -> Iterable["LawNode"]:
        yield self
        for child in self.children:
            yield from child.iter_subtree()


def _resolve_input_dir(input_dir: Optional[Path]) -> Path:
    if input_dir is None:
        input_dir = INPUT_PSEUDOCODE_DIR
    if not input_dir.is_absolute():
        input_dir = Path(__file__).resolve().parent / input_dir
    return input_dir


def load_law_entries(input_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load law entries from `law_json.json` under the given directory."""

    base_dir = _resolve_input_dir(Path(input_dir) if input_dir else None)
    law_json_path = base_dir / "law_json.json"
    if not law_json_path.exists():
        raise FileNotFoundError(f"law_json.json not found at: {law_json_path}")

    with law_json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("Expected law_json.json to contain a list of entries.")

    if DEBUG:
        print(f"Loaded {len(payload)} law entries from {law_json_path}.")

    return payload


def build_law_tree(entries: Iterable[Dict[str, Any]]) -> LawNode:
    """Build and return a tree of `LawNode` objects from raw law entries."""

    root = LawNode("ROOT", {"id": "ROOT", "class": "root", "title": "ROOT"})
    nodes_by_id: Dict[str, LawNode] = {}
    ordered_ids: List[str] = []

    for entry in entries:
        law_id = str(entry.get("id", "")).strip()
        if not law_id:
            raise ValueError("Encountered law entry without an 'id'.")
        if law_id in nodes_by_id:
            raise ValueError(f"Duplicate law id detected: {law_id}")
        node = LawNode(law_id, dict(entry))
        nodes_by_id[law_id] = node
        ordered_ids.append(law_id)

    for law_id in ordered_ids:
        node = nodes_by_id[law_id]
        entry = node.data
        if entry.get("class") == "조":
            parent_node = root
        else:
            parent_id = str(entry.get("parent", "")).strip()
            parent_node = nodes_by_id.get(parent_id, root)
        parent_node.add_child(node)

    if DEBUG:
        print(
            "Tree construction complete. Root children count:",
            len(root.children),
        )

    return root


def visualize_tree(
    root: LawNode,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Visualize the tree and return the node ids and edges used."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for visualize_tree; please install it."
        ) from exc

    nodes: List[str] = []
    edges: List[Tuple[str, str]] = []
    positions: Dict[str, Tuple[int, int]] = {}

    y_counter = 0

    def _assign_positions(node: LawNode, depth: int) -> None:
        nonlocal y_counter
        positions[node.law_id] = (depth, y_counter)
        nodes.append(node.law_id)
        y_counter += 1
        for child in node.children:
            edges.append((node.law_id, child.law_id))
            _assign_positions(child, depth + 1)

    _assign_positions(root, 0)

    x_vals = [coord[0] for coord in positions.values()]
    y_vals = [coord[1] for coord in positions.values()]

    plt.figure(figsize=figsize)
    ax = plt.gca()

    for parent_id, child_id in edges:
        x0, y0 = positions[parent_id]
        x1, y1 = positions[child_id]
        ax.plot([x0, x1], [y0, y1], color="gray", linewidth=1.0, zorder=1)

    ax.scatter(x_vals, y_vals, s=200, color="#4F6D7A", zorder=2)

    for node_id, (x_pos, y_pos) in positions.items():
        ax.text(
            x_pos,
            y_pos,
            node_id,
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            zorder=3,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim(min(x_vals) - 1, max(x_vals) + 1)
    ax.set_ylim(min(y_vals) - 1, max(y_vals) + 1)
    ax.set_title("Law Tree Visualization", pad=20)
    ax.set_axis_off()
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        if DEBUG:
            print(f"Tree visualization saved to {output_path}.")
    else:  # pragma: no cover - interactive fallback
        plt.show()

    plt.close()

    return nodes, edges


def main() -> None:
    try:
        entries = load_law_entries()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Failed to load law entries: {exc}")
        return

    root = build_law_tree(entries)
    print(f"Root has {len(root.children)} direct children.")

    try:
        output_file = Path(__file__).resolve().parent / "tree_visualization.png"
        nodes, edges = visualize_tree(root, output_path=output_file)
        print(
            "Visualization complete:",
            f"{len(nodes)} nodes, {len(edges)} edges (saved to {output_file}).",
        )
    except ImportError as exc:
        print(f"Visualization skipped: {exc}")

if __name__ == "__main__":
    main()
