"""Train one preregistered H021 price-aware economic-veto variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.train_h018_conservative_residual_policy import train
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from train_h018_conservative_residual_policy import (  # type: ignore[import-not-found,no-redef]
        train,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "trace"
    / "sphinx_trace_s0_h021_price_aware_economic_veto_v1.json"
)
DEFAULT_POLICY_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h021_policy_v1.json"
)
DEFAULT_INITIAL_POLICY_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h012_selective_policy_v1.json"
)
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json"
DEFAULT_RESIDUAL_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h013_market_residual_v1.json"
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    value.add_argument(
        "--initial-policy-config", type=Path, default=DEFAULT_INITIAL_POLICY_CONFIG
    )
    value.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    value.add_argument("--residual-config", type=Path, default=DEFAULT_RESIDUAL_CONFIG)
    value.add_argument("--state-dir", type=Path, required=True)
    value.add_argument("--encoding-dir", type=Path, required=True)
    value.add_argument("--pack-dir", type=Path, required=True)
    value.add_argument("--initial-policy-dir", type=Path, required=True)
    value.add_argument("--outcome-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument(
        "--variant",
        choices=("economic_only", "price_curriculum_080"),
        required=True,
    )
    value.add_argument("--seed", type=int, default=17)
    return value


def main() -> None:
    args = parser().parse_args()
    result = train(
        args.config.resolve(),
        args.policy_config.resolve(),
        args.initial_policy_config.resolve(),
        args.model_config.resolve(),
        args.residual_config.resolve(),
        args.state_dir.resolve(),
        args.encoding_dir.resolve(),
        args.pack_dir.resolve(),
        args.initial_policy_dir.resolve(),
        args.outcome_dir.resolve(),
        args.output_dir.resolve(),
        seed=args.seed,
        variant_id=args.variant,
    )
    print(
        json.dumps(
            {
                "status": result.get("status", "complete"),
                "variant_id": result.get("variant_id"),
                "best_epoch": result.get("best_epoch"),
                "selection": result.get("selection"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
