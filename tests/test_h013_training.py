from __future__ import annotations

from pathlib import Path

import torch
from scripts.train_h013_residual import _contract_digest, residual_loss


def test_h013_residual_loss_penalizes_only_learned_correction() -> None:
    output = {
        "terminal_outcome_logit": torch.tensor([0.0, 0.0]),
        "terminal_outcome_residual_logit": torch.tensor([1.0, 1.0]),
        "expected_net_edge": torch.tensor([0.0, 0.0]),
        "uncertainty_log_scale": torch.tensor([0.0, 0.0]),
    }
    training_config = {
        "training": {
            "loss": {
                "binary_cross_entropy": 1.0,
                "brier": 0.0,
                "expected_edge_smooth_l1": 0.0,
                "heteroscedastic_binary_cross_entropy": 0.0,
                "uncertainty_scale_penalty": 0.0,
            }
        }
    }
    residual_config = {"training": {"residual_l2_weight": 0.5}}
    loss, metrics = residual_loss(
        output,
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.5, 0.5]),
        torch.ones(2),
        training_config,
        residual_config,
    )
    assert torch.isfinite(loss)
    assert metrics["residual_l2"] == 1.0
    assert loss.item() > 0.5


def test_h013_training_contract_binds_every_config_and_implementation(tmp_path: Path) -> None:
    training = tmp_path / "training.json"
    model = tmp_path / "model.json"
    residual = tmp_path / "residual.json"
    training.write_text("{}", encoding="utf-8")
    model.write_text("{}", encoding="utf-8")
    residual.write_text("{}", encoding="utf-8")
    first = _contract_digest(training, model, residual, "ab" * 32)
    residual.write_text('{"changed":true}', encoding="utf-8")
    second = _contract_digest(training, model, residual, "ab" * 32)
    assert first != second
