from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from sphinx_corpus.io import atomic_json, load_json, now_utc

_CONDITION_ID = re.compile(r"^0x[a-f0-9]{64}$")


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def audit_fast_ledger(data_dir: Path, namespace: str) -> dict[str, Any]:
    summary_path = data_dir / "receipts" / f"{namespace}.json"
    group_root = data_dir / "receipts" / namespace
    summary = load_json(summary_path)
    paths = sorted(group_root.glob("*.json"))
    problems: list[str] = []
    condition_ids: set[str] = set()
    duplicate_condition_ids = 0
    rows = 0
    zero_row_groups = 0
    groups_with_gaps = 0
    incomplete_groups = 0

    for path in paths:
        receipt = _object(path)
        scope_id = str(receipt.get("scope_id") or "")
        if scope_id != path.stem and len(problems) < 100:
            problems.append(f"scope mismatch: {path.name} != {scope_id}")
        complete = receipt.get("complete") is True
        gaps = int(receipt.get("gaps") or 0)
        row_count = int(receipt.get("rows") or 0)
        incomplete_groups += int(not complete)
        groups_with_gaps += int(gaps > 0)
        zero_row_groups += int(row_count == 0)
        rows += row_count
        values = receipt.get("condition_ids")
        if not isinstance(values, list):
            if len(problems) < 100:
                problems.append(f"condition_ids is not a list: {path.name}")
            continue
        for value in values:
            condition_id = str(value).lower()
            if not _CONDITION_ID.fullmatch(condition_id):
                if len(problems) < 100:
                    problems.append(f"invalid condition id in {path.name}: {condition_id}")
                continue
            if condition_id in condition_ids:
                duplicate_condition_ids += 1
            condition_ids.add(condition_id)

    expected_groups = int(summary.get("groups_selected") or 0)
    expected_markets = int(summary.get("markets_selected") or 0)
    structural_checks = {
        "summary_complete": summary.get("complete") is True,
        "summary_has_no_gaps": int(summary.get("gaps") or 0) == 0,
        "summary_has_no_incomplete_groups": (int(summary.get("groups_incomplete") or 0) == 0),
        "request_budget_not_reached": (summary.get("request_budget_reached") is False),
        "group_count_matches": len(paths) == expected_groups,
        "market_count_matches": len(condition_ids) == expected_markets,
        "all_group_receipts_complete": incomplete_groups == 0,
        "all_group_receipts_gap_free": groups_with_gaps == 0,
        "condition_ids_are_unique": duplicate_condition_ids == 0,
        "receipt_shapes_are_valid": not problems,
    }
    temporary_files = sorted(
        path.relative_to(data_dir).as_posix() for path in data_dir.rglob("*.tmp")
    )
    result = {
        "schema_version": "1.0.0",
        "generated_at": now_utc(),
        "namespace": namespace,
        "valid": all(structural_checks.values()),
        "checks": structural_checks,
        "groups": len(paths),
        "markets": len(condition_ids),
        "rows_from_group_receipts": rows,
        "zero_row_groups": zero_row_groups,
        "incomplete_groups": incomplete_groups,
        "groups_with_gaps": groups_with_gaps,
        "duplicate_condition_ids": duplicate_condition_ids,
        "temporary_files": temporary_files,
        "problems": problems,
        "source_summary": summary,
    }
    output_path = data_dir / "receipts" / f"{namespace}-audit.json"
    atomic_json(output_path, result)
    if not result["valid"]:
        raise RuntimeError(f"Fast Ledger audit failed; see {output_path}")
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--data-dir", type=Path, required=True)
    root.add_argument("--namespace", default="ledger-api-cash-min-25")
    return root


def main() -> None:
    args = parser().parse_args()
    result = audit_fast_ledger(args.data_dir.resolve(), str(args.namespace))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
