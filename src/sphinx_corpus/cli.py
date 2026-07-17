from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from sphinx_corpus.atlas import AtlasBackfill
from sphinx_corpus.config import CorpusConfig
from sphinx_corpus.depth import DepthBackfill
from sphinx_corpus.io import build_manifest
from sphinx_corpus.ledger import LedgerBackfill
from sphinx_corpus.rpc import PolygonRPC
from sphinx_corpus.trade_api import TradeAPIBackfill


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sphinx-corpus")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/corpus/sphinx_corpus_v1.json"),
    )
    parser.add_argument("--data-dir", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    atlas = subparsers.add_parser("atlas")
    atlas.add_argument("--max-pages", type=int)

    ledger = subparsers.add_parser("ledger")
    ledger.add_argument("--market", action="append", dest="markets")
    ledger.add_argument("--max-markets", type=int)
    ledger.add_argument("--max-requests", type=int)
    ledger.add_argument("--workers", type=int)
    ledger.add_argument("--requests-per-second", type=float)

    chain_ledger = subparsers.add_parser("chain-ledger")
    chain_ledger.add_argument("--rpc-url")
    chain_ledger.add_argument("--exchange", action="append", dest="exchanges")
    chain_ledger.add_argument("--max-chunks", type=int)
    chain_ledger.add_argument("--allow-full-chain", action="store_true")

    probe = subparsers.add_parser("probe")
    probe.add_argument("--rpc-url")
    probe.add_argument("--exchange", action="append", dest="exchanges")
    probe.add_argument("--samples", type=int, default=8)
    probe.add_argument("--sample-blocks", type=int, default=200)

    depth = subparsers.add_parser("depth")
    depth.add_argument("--max-tokens", type=int)
    depth.add_argument("--max-windows", type=int)

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--max-pages", type=int)
    all_parser.add_argument("--max-markets", type=int)
    all_parser.add_argument("--max-requests", type=int)
    all_parser.add_argument("--workers", type=int)
    all_parser.add_argument("--requests-per-second", type=float)
    all_parser.add_argument("--max-tokens", type=int)
    all_parser.add_argument("--max-windows", type=int)

    subparsers.add_parser("manifest")
    return parser


def _rpc_url(args: argparse.Namespace, config: CorpusConfig) -> str:
    value = getattr(args, "rpc_url", None) or os.environ.get(config.rpc_env)
    if not value:
        raise RuntimeError(f"Polygon RPC is required; pass --rpc-url or set {config.rpc_env}")
    return str(value)


def _atlas(config: CorpusConfig, max_pages: int | None) -> dict[str, Any]:
    with AtlasBackfill(config) as collector:
        return collector.collect(max_pages=max_pages)


def _chain_ledger(
    config: CorpusConfig,
    args: argparse.Namespace,
    exchange_ids: set[str] | None = None,
) -> dict[str, Any]:
    if getattr(args, "max_chunks", None) is None and not args.allow_full_chain:
        raise RuntimeError(
            "Full chain Ledger collection requires --allow-full-chain; "
            "use --max-chunks for verification samples"
        )
    with PolygonRPC(_rpc_url(args, config)) as rpc:
        return LedgerBackfill(config, rpc).collect(
            exchange_ids=exchange_ids,
            max_chunks=getattr(args, "max_chunks", None),
        )


def _trade_ledger(config: CorpusConfig, args: argparse.Namespace) -> dict[str, Any]:
    markets = set(args.markets) if getattr(args, "markets", None) else None
    with TradeAPIBackfill(
        config,
        max_requests=getattr(args, "max_requests", None),
        workers=getattr(args, "workers", None),
        requests_per_second=getattr(args, "requests_per_second", None),
    ) as collector:
        return collector.collect(
            market_ids=markets,
            max_markets=getattr(args, "max_markets", None),
        )


def _depth(config: CorpusConfig, args: argparse.Namespace) -> dict[str, Any]:
    with DepthBackfill(config) as collector:
        return collector.collect(
            max_tokens=getattr(args, "max_tokens", None),
            max_windows=getattr(args, "max_windows", None),
        )


def _manifest(config: CorpusConfig) -> dict[str, Any]:
    return build_manifest(
        config.data_dir,
        corpus_id=config.id,
        version=config.version,
        research_id=config.research_id,
        source_config=config.payload,
    )


def main() -> None:
    args = _parser().parse_args()
    config = CorpusConfig.load(args.config, args.data_dir)
    result: dict[str, Any]
    if args.command == "atlas":
        result = _atlas(config, args.max_pages)
    elif args.command == "ledger":
        result = _trade_ledger(config, args)
    elif args.command == "chain-ledger":
        exchanges = set(args.exchanges) if args.exchanges else None
        result = _chain_ledger(config, args, exchanges)
    elif args.command == "depth":
        result = _depth(config, args)
    elif args.command == "probe":
        exchanges = set(args.exchanges) if args.exchanges else None
        with PolygonRPC(_rpc_url(args, config)) as rpc:
            result = LedgerBackfill(config, rpc).probe(
                exchange_ids=exchanges,
                samples=args.samples,
                sample_blocks=args.sample_blocks,
            )
    elif args.command == "all":
        result = {
            "atlas": _atlas(config, args.max_pages),
            "ledger": _trade_ledger(config, args),
            "depth": _depth(config, args),
        }
    elif args.command == "manifest":
        result = _manifest(config)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
