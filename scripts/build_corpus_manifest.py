from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sphinx_corpus.config import CorpusConfig
from sphinx_corpus.io import atomic_json, build_manifest, load_json, now_utc, sha256_file


def build_and_qualify(config_path: Path, data_dir: Path, workers: int) -> dict[str, Any]:
    config = CorpusConfig.load(config_path, data_dir)
    namespace = str(config.payload["sources"]["ledger"]["primary"]["storage_namespace"])
    audit_path = data_dir / "receipts" / f"{namespace}-audit.json"
    audit = load_json(audit_path)
    if audit.get("valid") is not True:
        raise RuntimeError(f"A valid receipt audit is required: {audit_path}")

    manifest = build_manifest(
        data_dir,
        corpus_id=config.id,
        version=config.version,
        research_id=config.research_id,
        source_config=config.payload,
        workers=workers,
    )
    prefix = f"normalized/{namespace}/"
    ledger_entries = [entry for entry in manifest["files"] if str(entry["path"]).startswith(prefix)]
    ledger_rows = sum(int(entry.get("rows", 0)) for entry in ledger_entries)
    receipt_rows = int(audit["rows_from_group_receipts"])
    checks = {
        "receipt_audit_valid": audit.get("valid") is True,
        "ledger_rows_match_receipts": ledger_rows == receipt_rows,
        "manifest_has_files": int(manifest["file_count"]) > 0,
        "manifest_has_bytes": int(manifest["total_bytes"]) > 0,
    }
    manifest_path = data_dir / "manifest.json"
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_at": now_utc(),
        "valid": all(checks.values()),
        "checks": checks,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "manifest_file_count": int(manifest["file_count"]),
        "manifest_total_bytes": int(manifest["total_bytes"]),
        "manifest_row_count": int(manifest["row_count"]),
        "ledger_file_count": len(ledger_entries),
        "ledger_rows": ledger_rows,
        "receipt_rows": receipt_rows,
        "workers": workers,
    }
    qualification_path = data_dir / "receipts" / f"{namespace}-qualification.json"
    atomic_json(qualification_path, result)
    if not result["valid"]:
        raise RuntimeError(f"Corpus qualification failed; see {qualification_path}")
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument(
        "--config",
        type=Path,
        default=Path("configs/corpus/sphinx_corpus_s0_fast_v1.json"),
    )
    root.add_argument("--data-dir", type=Path, required=True)
    root.add_argument("--workers", type=int, default=16)
    return root


def main() -> None:
    args = parser().parse_args()
    result = build_and_qualify(args.config, args.data_dir.resolve(), args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
