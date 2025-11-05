import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_BUILDER = PROJECT_ROOT / "method" / "code_generation" / "code_builder.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "method" / "outputs" / "legal_code_output"


def discover_target_dirs(root: Path) -> Iterable[Path]:
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        law_code = candidate / "law_code.json"
        variables = candidate / "variables.json"
        if law_code.exists() and variables.exists():
            yield candidate


def run_code_builder(target_dir: Path, verbose: bool = False) -> None:
    if verbose:
        print(f"[code_builder] Generating code in {target_dir}")
    subprocess.run(
        [sys.executable, str(CODE_BUILDER), "-d", str(target_dir)],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run code_builder.py for every generated legal code directory.",
    )
    parser.add_argument(
        "-r",
        "--root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory containing legal code outputs (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress information during execution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.root.expanduser().resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"Output root not found: {output_root}")
    if not CODE_BUILDER.exists():
        raise FileNotFoundError(f"code_builder.py not found at {CODE_BUILDER}")

    targets = list(discover_target_dirs(output_root))
    if not targets:
        print(f"No valid target directories found under {output_root}")
        return

    for directory in targets:
        run_code_builder(directory, verbose=args.verbose)

    print(f"Completed code generation for {len(targets)} directories under {output_root}")


if __name__ == "__main__":
    main()
