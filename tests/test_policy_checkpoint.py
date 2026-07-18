from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import torch

from sphinx_corpus.io import sha256_file
from sphinx_trace.config import load_json
from sphinx_trace.model_h011 import SphinxTraceS0H011
from sphinx_trace.model_h012 import SphinxTraceS0H012
from sphinx_trace.policy_checkpoint import load_outcome_checkpoint, load_policy_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def _configs() -> tuple[dict, dict, dict]:
    model = deepcopy(load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json"))
    model["architecture"]["candidates"].append(
        {"id": "load", "width": 64, "heads": 4, "layers": 1, "ffn_width": 128}
    )
    residual = deepcopy(
        load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h013_market_residual_v1.json")
    )
    policy = deepcopy(
        load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h012_selective_policy_v1.json")
    )
    return model, residual, policy


def test_loads_hash_bound_outcome_and_policy_checkpoints(tmp_path: Path) -> None:
    model_config, residual_config, policy_config = _configs()
    outcome_dir = tmp_path / "outcome"
    policy_dir = tmp_path / "policy"
    outcome_dir.mkdir()
    policy_dir.mkdir()
    outcome_model = SphinxTraceS0H011(model_config, candidate_id="load")
    torch.save({"model": outcome_model.state_dict()}, outcome_dir / "best-model.pt")
    (outcome_dir / "result.json").write_text(
        json.dumps(
            {
                "valid": True,
                "record_type": "h011_outcome_training_result",
                "candidate_id": "load",
                "variant_id": "h011_market_only",
                "best_model_sha256": sha256_file(outcome_dir / "best-model.pt"),
                "test_rows_consumed": 0,
                "test_labels_opened": False,
            }
        ),
        encoding="utf-8",
    )
    policy_model = SphinxTraceS0H012(outcome_model, policy_config)
    torch.save(
        {"model": policy_model.state_dict(), "contract_sha256": "contract"},
        policy_dir / "best-policy.pt",
    )
    (policy_dir / "result.json").write_text(
        json.dumps(
            {
                "valid": True,
                "contract_sha256": "contract",
                "best_model_sha256": sha256_file(policy_dir / "best-policy.pt"),
                "outcome_result_sha256": sha256_file(outcome_dir / "result.json"),
                "test_rows_consumed": 0,
                "test_labels_opened": False,
            }
        ),
        encoding="utf-8",
    )

    outcome = load_outcome_checkpoint(
        outcome_dir, model_config, residual_config, torch.device("cpu")
    )
    policy = load_policy_checkpoint(
        policy_dir,
        outcome_dir,
        model_config,
        residual_config,
        policy_config,
        torch.device("cpu"),
    )

    assert outcome.feature_mask.shape == (128,)
    assert policy.group_mask.shape == (6,)
    assert policy.result["valid"] is True
