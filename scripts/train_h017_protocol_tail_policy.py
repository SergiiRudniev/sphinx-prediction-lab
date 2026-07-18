"""Train H017 on protocol-exact action value and learned lower-tail utility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from scripts.train_h015_portfolio_advantage import (
        DEFAULT_MODEL_CONFIG,
        DEFAULT_POLICY_CONFIG,
        DEFAULT_RESIDUAL_CONFIG,
        ROOT,
        train,
    )
else:
    from train_h015_portfolio_advantage import (  # type: ignore[import-not-found,no-redef]
        DEFAULT_MODEL_CONFIG,
        DEFAULT_POLICY_CONFIG,
        DEFAULT_RESIDUAL_CONFIG,
        ROOT,
        train,
    )

DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "trace"
    / "sphinx_trace_s0_h017_protocol_tail_utility_v1.json"
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    value.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    value.add_argument("--residual-config", type=Path, default=DEFAULT_RESIDUAL_CONFIG)
    value.add_argument("--state-dir", type=Path, required=True)
    value.add_argument("--encoding-dir", type=Path, required=True)
    value.add_argument("--pack-dir", type=Path, required=True)
    value.add_argument("--initial-policy-dir", type=Path, required=True)
    value.add_argument("--outcome-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--seed", type=int, default=17)
    return value


def main() -> None:
    args = parser().parse_args()
    result = train(
        args.config.resolve(),
        args.policy_config.resolve(),
        args.model_config.resolve(),
        args.residual_config.resolve(),
        args.state_dir.resolve(),
        args.encoding_dir.resolve(),
        args.pack_dir.resolve(),
        args.initial_policy_dir.resolve(),
        args.outcome_dir.resolve(),
        args.output_dir.resolve(),
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": result.get("status", "complete"),
                "best_epoch": result.get("best_epoch"),
                "selection": result.get("selection"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
