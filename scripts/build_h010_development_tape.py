"""Build the reusable validation/calibration liquidity tape for H010."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sphinx_trace.development_tape import build_development_tape


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--pack-dir", type=Path, required=True)
    value.add_argument("--chronicle-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    result = build_development_tape(
        args.pack_dir.resolve(),
        args.chronicle_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
