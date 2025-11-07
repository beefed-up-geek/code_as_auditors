import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

CASE_DIR = PROJECT_ROOT / "dataset" / "PIPA" / "cases"
_VAR_NAME_PATTERN = re.compile(r"^(LAW_[A-Z0-9]+(?:_[A-Z0-9]+)?)")
CASE_CODE_PATH_EXAMPLE = "/Users/taeyoonkwack/Documents/code_as_auditors/method/outputs/case_code_output/20251104_184338_5_5_5"
DEBUG = True

def _iter_case_records(case_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    jsonl_files = sorted(case_dir.glob("*.jsonl"))
    if not jsonl_files:
        return records

    for jsonl_path in jsonl_files:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {jsonl_path}"
                    ) from exc
                records.append(record)
    return records


def _normalise_var_name(raw_var_name: str) -> str:
    stripped = raw_var_name.strip()
    if not stripped:
        return ""
    bracket_index = stripped.find("[")
    if bracket_index != -1:
        stripped = stripped[:bracket_index]
    stripped = stripped.strip()
    match = _VAR_NAME_PATTERN.match(stripped)
    if match:
        return match.group(1)
    tokens = stripped.split("_")
    normalized_parts: List[str] = []
    for idx, token in enumerate(tokens):
        token_upper = token.upper()
        if idx == 0:
            normalized_parts.append(token_upper)
            continue
        if token_upper.startswith("P") or token_upper.startswith("S"):
            break
        normalized_parts.append(token_upper)
    if not normalized_parts:
        return stripped.upper()
    return "_".join(normalized_parts)


def get_cases(case_dir: Path = CASE_DIR) -> List[Dict[str, Any]]:
    """Load case metadata from CASE_DIR and normalise violated article var_names."""
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    case_records = _iter_case_records(case_dir)
    cases: List[Dict[str, Any]] = []

    for record in case_records:
        case_id = str(record.get("case_id", "")).strip()
        if not case_id:
            continue

        violated_articles = record.get("violated_articles") or []
        seen: Set[str] = set()
        var_names: List[str] = []

        for article in violated_articles:
            if not isinstance(article, dict):
                continue
            var_name_raw = article.get("var_name")
            if not var_name_raw:
                continue
            var_name = _normalise_var_name(str(var_name_raw))
            if not var_name or var_name in seen:
                continue
            seen.add(var_name)
            var_names.append(var_name)

        cases.append({"case_id": case_id, "violated_articles": var_names})

    return cases


def _parse_non_compliant_variables(lines: List[str]) -> List[str]:
    target_prefix = "Non-compliant law variables:"
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped.startswith(target_prefix):
            continue
        payload = stripped[len(target_prefix) :].strip()
        if not payload or payload.lower().startswith("no "):
            return []
        items = [item.strip() for item in payload.split(",") if item.strip()]
        normalized: List[str] = []
        seen: Set[str] = set()
        for item in items:
            norm = _normalise_var_name(item)
            if norm and norm not in seen:
                seen.add(norm)
                normalized.append(norm)
        return normalized
    return []


def _strip_existing_evaluation_section(lines: List[str]) -> List[str]:
    marker = "Evaluation Summary:"
    for index, line in enumerate(lines):
        if line.strip().lower() == marker.lower():
            return lines[:index]
    return lines


def _format_article_list(values: List[str]) -> str:
    return ", ".join(values) if values else "-"


def run_single_case_specific_code(
    single_case_code_path: Path,
    ground_truth: Dict[str, Set[str]],
    known_articles: Set[str],
) -> Dict[str, Any]:
    single_case_code_path = single_case_code_path.resolve()
    if not single_case_code_path.exists():
        raise FileNotFoundError(f"Case code not found: {single_case_code_path}")

    result = subprocess.run(
        [sys.executable, str(single_case_code_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr_tail = (result.stderr or "").strip().splitlines()[-1:] if result.stderr else []
        raise RuntimeError(
            f"Execution failed for {single_case_code_path.name} (exit {result.returncode}). "
            + (" ".join(stderr_tail) if stderr_tail else "")
        )

    result_path = single_case_code_path.with_suffix(".txt")
    if not result_path.exists():
        raise FileNotFoundError(f"Expected result file not found: {result_path}")

    file_lines = result_path.read_text(encoding="utf-8").splitlines()
    predicted_list = _parse_non_compliant_variables(file_lines)
    predicted_set = set(predicted_list)

    case_id = single_case_code_path.stem
    actual_set = set(ground_truth.get(case_id, set()))

    candidate_articles = set(known_articles)
    candidate_articles.update(predicted_set)

    tp_set = predicted_set & actual_set
    fp_set = predicted_set - actual_set
    fn_set = actual_set - predicted_set
    tn_set = candidate_articles - actual_set - predicted_set

    tp_list = sorted(tp_set)
    fp_list = sorted(fp_set)
    fn_list = sorted(fn_set)
    tn_list = sorted(tn_set)

    precision = len(tp_set) / (len(tp_set) + len(fp_set)) if (tp_set or fp_set) else 0.0
    recall = len(tp_set) / (len(tp_set) + len(fn_set)) if (tp_set or fn_set) else 0.0
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    base_lines = _strip_existing_evaluation_section(file_lines)
    while base_lines and not base_lines[-1].strip():
        base_lines.pop()

    evaluation_lines = [
        "Evaluation Summary:",
        "| Category | Articles |",
        "| --- | --- |",
        f"| TP | {_format_article_list(tp_list)} |",
        f"| FP | {_format_article_list(fp_list)} |",
        f"| TN | {_format_article_list(tn_list)} |",
        f"| FN | {_format_article_list(fn_list)} |",
        "",
        "Metrics:",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"F1 Score: {f1:.4f}",
    ]

    output_lines = list(base_lines)
    if output_lines and output_lines[-1] != "":
        output_lines.append("")
    output_lines.extend(evaluation_lines)

    result_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    return {
        "case_id": case_id,
        "predicted": sorted(predicted_set),
        "actual": sorted(actual_set),
        "tp": tp_list,
        "fp": fp_list,
        "tn": tn_list,
        "fn": fn_list,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "result_path": str(result_path),
    }

def case_code_evaluation_result(
    case_code_path: Path = Path(CASE_CODE_PATH_EXAMPLE),
) -> List[Path]:
    base_path = Path(case_code_path).expanduser().resolve()
    if not base_path.exists() or not base_path.is_dir():
        raise FileNotFoundError(f"Case code directory not found: {base_path}")

    cases = get_cases()
    ground_truth_map: Dict[str, Set[str]] = {
        case["case_id"]: set(case["violated_articles"]) for case in cases
    }
    known_articles: Set[str] = {
        article for case in cases for article in case["violated_articles"]
    }

    target_dirs = sorted(path for path in base_path.iterdir() if path.is_dir())
    if not target_dirs:
        target_dirs = [base_path]

    result_paths: List[Path] = []
    run_summaries: List[Dict[str, Any]] = []

    for target_dir in target_dirs:
        results: List[Dict[str, Any]] = []
        failures: List[str] = []

        code_files = sorted(file for file in target_dir.glob("*.py") if file.is_file())
        for code_file in code_files:
            try:
                result = run_single_case_specific_code(
                    code_file, ground_truth_map, known_articles
                )
            except Exception as exc:
                failures.append(f"{code_file}: {exc}")
                if DEBUG:
                    print(f"[FAIL] {code_file}: {exc}")
                continue
            results.append(result)

        output_lines: List[str] = []

        if results:
            precision_values = [item["precision"] for item in results]
            recall_values = [item["recall"] for item in results]
            f1_values = [item["f1"] for item in results]

            macro_metrics = {
                "precision": mean(precision_values),
                "recall": mean(recall_values),
                "f1": mean(f1_values),
            }
            case_level_stds = {
                "precision": stdev(precision_values) if len(precision_values) > 1 else 0.0,
                "recall": stdev(recall_values) if len(recall_values) > 1 else 0.0,
                "f1": stdev(f1_values) if len(f1_values) > 1 else 0.0,
            }

            run_summaries.append(
                {
                    "run_dir": target_dir,
                    "evaluated_cases": len(results),
                    "macro": macro_metrics,
                }
            )

            output_lines.extend(
                [
                    f"Evaluated cases: {len(results)}",
                    "Macro metrics (mean across cases):",
                    f"* Precision: {macro_metrics['precision']:.4f}",
                    f"* Recall: {macro_metrics['recall']:.4f}",
                    f"* F1 Score: {macro_metrics['f1']:.4f}",
                    "",
                    "Case-level variation (std across cases):",
                    f"* Precision std: {case_level_stds['precision']:.4f}",
                    f"* Recall std: {case_level_stds['recall']:.4f}",
                    f"* F1 Score std: {case_level_stds['f1']:.4f}",
                ]
            )
        else:
            output_lines.append("Evaluated cases: 0")
            output_lines.append(
                "No successful evaluations. Precision/Recall/F1 statistics unavailable."
            )

        if failures:
            output_lines.append("")
            output_lines.append(f"Failed evaluations: {len(failures)}")
            output_lines.extend(failures)

        result_path = target_dir / "_result.txt"
        result_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
        result_paths.append(result_path)

        if DEBUG:
            print(f"[RESULT] {target_dir}")
            for line in output_lines:
                print(line)

    aggregate_lines: List[str] = []
    successful_runs = len(run_summaries)
    total_runs = len(target_dirs)

    def _mean_std(values: List[float]) -> Dict[str, float]:
        avg = mean(values)
        std_dev = stdev(values) if len(values) > 1 else 0.0
        return {"mean": avg, "std": std_dev}

    if run_summaries:
        precision_means = [summary["macro"]["precision"] for summary in run_summaries]
        recall_means = [summary["macro"]["recall"] for summary in run_summaries]
        f1_means = [summary["macro"]["f1"] for summary in run_summaries]

        precision_stats = _mean_std(precision_means)
        recall_stats = _mean_std(recall_means)
        f1_stats = _mean_std(f1_means)

        aggregate_lines.extend(
            [
                "Aggregated run metrics (macro means across cases):",
                f"Runs aggregated: {successful_runs} / {total_runs}",
                f"* Precision mean: {precision_stats['mean']:.4f}, std over runs: {precision_stats['std']:.4f}",
                f"* Recall mean: {recall_stats['mean']:.4f}, std over runs: {recall_stats['std']:.4f}",
                f"* F1 Score mean: {f1_stats['mean']:.4f}, std over runs: {f1_stats['std']:.4f}",
            ]
        )
    else:
        aggregate_lines.append("No successful runs to aggregate.")

    aggregate_result_path = base_path / "_aggregate_result.txt"
    aggregate_result_path.write_text("\n".join(aggregate_lines) + "\n", encoding="utf-8")
    result_paths.append(aggregate_result_path)

    if DEBUG:
        print("[RESULT] Aggregated metrics")
        for line in aggregate_lines:
            print(line)

    return result_paths

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated case-specific code and aggregate metrics."
    )
    parser.add_argument(
        "case_code_path",
        type=Path,
        help="Directory containing folders of case-specific Python code.",
    )
    args = parser.parse_args(argv)

    case_code_evaluation_result(args.case_code_path)


if __name__ == "__main__":
    main()
