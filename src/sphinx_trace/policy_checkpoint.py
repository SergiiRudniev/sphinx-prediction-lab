"""Load source-bound H011/H013 outcomes and H012 policy checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from sphinx_corpus.io import sha256_file
from sphinx_trace.model_h011 import (
    SphinxTraceS0H011,
    h011_variant_feature_mask,
    h011_variant_group_mask,
)
from sphinx_trace.model_h012 import SphinxTraceS0H012
from sphinx_trace.model_h013 import (
    SphinxTraceS0H013,
    h013_variant_feature_mask,
    h013_variant_group_mask,
)


@dataclass(frozen=True, slots=True)
class OutcomeCheckpoint:
    model: SphinxTraceS0H011 | SphinxTraceS0H013
    feature_mask: Tensor
    group_mask: Tensor
    result: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PolicyCheckpoint:
    model: SphinxTraceS0H012
    feature_mask: Tensor
    group_mask: Tensor
    result: dict[str, Any]
    outcome_result: dict[str, Any]


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def load_outcome_checkpoint(
    outcome_dir: Path,
    model_config: dict[str, Any],
    residual_config: dict[str, Any],
    device: torch.device,
) -> OutcomeCheckpoint:
    result = _load_object(outcome_dir / "result.json")
    best_path = outcome_dir / "best-model.pt"
    if (
        result.get("valid") is not True
        or result.get("test_labels_opened") is not False
        or int(result.get("test_rows_consumed", -1)) != 0
        or result.get("best_model_sha256") != sha256_file(best_path)
    ):
        raise RuntimeError("H012 requires a valid source-bound closed-test outcome model")
    candidate = str(result["candidate_id"])
    variant = str(result["variant_id"])
    direct = SphinxTraceS0H011(model_config, candidate_id=candidate)
    record_type = str(result.get("record_type"))
    if record_type == "h013_market_residual_training_result":
        architecture = residual_config["architecture"]
        outcome: SphinxTraceS0H011 | SphinxTraceS0H013 = SphinxTraceS0H013(
            direct,
            minimum_probability=float(architecture["minimum_anchor_probability"]),
            maximum_probability=float(architecture["maximum_anchor_probability"]),
        )
        feature_mask = h013_variant_feature_mask(variant, device=device)
        group_mask = h013_variant_group_mask(variant, device=device)
    elif record_type == "h011_outcome_training_result":
        outcome = direct
        feature_mask = h011_variant_feature_mask(variant, device=device)
        group_mask = h011_variant_group_mask(variant, device=device)
    else:
        raise RuntimeError(f"Unsupported H012 outcome model result: {record_type}")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    outcome.load_state_dict(best["model"])
    return OutcomeCheckpoint(outcome.to(device), feature_mask, group_mask, result)


def load_policy_checkpoint(
    policy_dir: Path,
    outcome_dir: Path,
    model_config: dict[str, Any],
    residual_config: dict[str, Any],
    policy_config: dict[str, Any],
    device: torch.device,
) -> PolicyCheckpoint:
    result = _load_object(policy_dir / "result.json")
    best_path = policy_dir / "best-policy.pt"
    outcome_result_path = outcome_dir / "result.json"
    if (
        result.get("valid") is not True
        or result.get("test_labels_opened") is not False
        or int(result.get("test_rows_consumed", -1)) != 0
        or result.get("best_model_sha256") != sha256_file(best_path)
        or result.get("outcome_result_sha256") != sha256_file(outcome_result_path)
    ):
        raise RuntimeError("H010 replay requires a valid source-bound closed-test policy")
    outcome = load_outcome_checkpoint(outcome_dir, model_config, residual_config, device)
    model = SphinxTraceS0H012(outcome.model, policy_config).to(device)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if best.get("contract_sha256") != result.get("contract_sha256"):
        raise RuntimeError("H012 best checkpoint contract changed")
    model.load_state_dict(best["model"])
    return PolicyCheckpoint(
        model=model,
        feature_mask=outcome.feature_mask,
        group_mask=outcome.group_mask,
        result=result,
        outcome_result=outcome.result,
    )
