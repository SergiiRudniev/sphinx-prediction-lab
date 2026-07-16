"""Command-line entrypoint for contract inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

from sphinx_trace import __version__
from sphinx_trace.config import load_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sphinx-trace")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--validate-config", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.version:
        print(f"Sphinx Trace {__version__}")
        return
    if args.validate_config is not None:
        config = load_json(args.validate_config)
        print(f"valid: {config.get('id', args.validate_config.name)}")
        return
    build_parser().print_help()


if __name__ == "__main__":
    main()
