"""Receipt-bound H022 ensemble inference over one frozen-H021 CALL candidate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import lightgbm as lgb
import numpy as np
import torch
from numpy.typing import NDArray

from sphinx_corpus.io import sha256_file
from sphinx_trace.h022_features import (
    H022_TREE_FEATURE_NAMES,
    H022_TREE_FEATURE_WIDTH,
    assemble_tree_features,
)
from sphinx_trace.h022_training import predict_weighted_ridge
from sphinx_trace.model_h022 import H022_GROUP_IDS, SphinxTraceS0H022NeuralMember

H022_BREAK_EVEN_INDEX = H022_TREE_FEATURE_NAMES.index(
    "candidate.break_even_probability"
)


class TreePredictor(Protocol):
    def predict(
        self,
        data: NDArray[np.float32],
        *,
        pred_contrib: bool = False,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class H022DecisionDebug:
    keep_base_call: bool
    candidate_action_id: int
    neural_mean_net_return: float
    neural_return_quantiles: tuple[float, float, float]
    neural_fill_probability: float
    neural_calibrated_probability0: float
    neural_calibrated_candidate_edge: float
    tree_net_return: float
    ensemble_net_return: float
    stacker_intercept: float
    stacker_contributions: tuple[float, ...]
    neural_group_attention: tuple[float, ...]
    tree_group_contributions: tuple[float, ...]
    tree_price_context_contribution: float
    tree_wallet_contribution: float
    tree_event_contribution: float


class H022EnsembleRuntime:
    """Load and score the selected H022 neural/tree/stacker artifact."""

    def __init__(
        self,
        neural: SphinxTraceS0H022NeuralMember,
        tree: TreePredictor,
        statistics: dict[str, NDArray[np.float32]],
        stacker: dict[str, Any],
        device: torch.device,
        *,
        policy_sha256: str,
    ) -> None:
        required = {"latent_mean", "latent_scale", "tree_mean", "tree_scale"}
        if set(statistics) != required or len(policy_sha256) != 64:
            raise ValueError("H022 runtime artifact contract is incomplete")
        if (
            statistics["latent_mean"].shape != (512,)
            or statistics["latent_scale"].shape != (512,)
            or statistics["tree_mean"].shape != (H022_TREE_FEATURE_WIDTH,)
            or statistics["tree_scale"].shape != (H022_TREE_FEATURE_WIDTH,)
        ):
            raise ValueError("H022 runtime normalizer shape changed")
        self.neural = neural.to(device).eval()
        self.tree = tree
        self.statistics = statistics
        self.stacker = stacker
        self.device = device
        self.policy_sha256 = policy_sha256
        self.price_indices = np.asarray(
            [
                index
                for index, name in enumerate(H022_TREE_FEATURE_NAMES)
                if any(
                    token in name
                    for token in (
                        "price",
                        "payout",
                        "break_even",
                        "probability",
                        "terminal_edge",
                        "market_edge",
                    )
                )
            ],
            dtype=np.int64,
        )

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: Path,
        summary_path: Path,
        device: torch.device,
    ) -> H022EnsembleRuntime:
        summary = _load_object(summary_path)
        result_path = artifact_dir / "result.json"
        neural_path = artifact_dir / "final" / "neural" / "best-neural.pt"
        tree_path = artifact_dir / "final" / "tree.txt"
        stacker_path = artifact_dir / "final" / "stacker.json"
        artifacts = summary.get("artifacts")
        if not isinstance(artifacts, dict):
            raise RuntimeError("H022 summary has no artifact receipts")
        if (
            summary.get("research_id") != "SPH-T-H022"
            or int(summary.get("selected_seed", -1)) != 17
            or summary.get("test_labels_opened") is not False
            or int(summary.get("test_rows_consumed", -1)) != 0
            or summary.get("promotion_allowed") is not False
            or sha256_file(result_path) != artifacts.get("seed17_result_sha256")
            or sha256_file(neural_path) != artifacts.get("seed17_neural_sha256")
            or sha256_file(tree_path) != artifacts.get("seed17_tree_sha256")
            or sha256_file(stacker_path) != artifacts.get("seed17_stacker_sha256")
        ):
            raise RuntimeError("H022 selected runtime artifact receipt changed")
        result = _load_object(result_path)
        if (
            result.get("valid") is not True
            or int(result.get("seed", -1)) != 17
            or result.get("selection_breadth_eligible") is not True
            or result.get("test_labels_opened") is not False
            or int(result.get("test_rows_consumed", -1)) != 0
        ):
            raise RuntimeError("H022 runtime requires the breadth-qualified seed 17")
        checkpoint = torch.load(neural_path, map_location="cpu", weights_only=False)
        if checkpoint.get("record_type") != "h022_neural_member":
            raise RuntimeError("H022 neural artifact type changed")
        neural_config = checkpoint.get("config")
        raw_statistics = checkpoint.get("statistics")
        if not isinstance(neural_config, dict) or not isinstance(raw_statistics, dict):
            raise RuntimeError("H022 neural artifact metadata is incomplete")
        neural = SphinxTraceS0H022NeuralMember(neural_config)
        neural.load_state_dict(checkpoint["model"])
        statistics = {
            str(name): np.asarray(value, dtype=np.float32)
            for name, value in raw_statistics.items()
        }
        stacker = _load_object(stacker_path)
        if tuple(stacker.get("predictor_names", ())) != (
            "neural_mean_net_return",
            "neural_q10",
            "neural_q50",
            "neural_q90",
            "neural_fill_probability",
            "neural_calibrated_edge",
            "tree_net_return",
        ):
            raise RuntimeError("H022 stacker feature contract changed")
        policy_payload = {
            "summary_sha256": sha256_file(summary_path),
            "result_sha256": sha256_file(result_path),
            "neural_sha256": sha256_file(neural_path),
            "tree_sha256": sha256_file(tree_path),
            "stacker_sha256": sha256_file(stacker_path),
        }
        policy_sha256 = hashlib.sha256(
            json.dumps(policy_payload, sort_keys=True).encode()
        ).hexdigest()
        return cls(
            neural,
            cast(TreePredictor, lgb.Booster(model_file=str(tree_path))),
            statistics,
            stacker,
            device,
            policy_sha256=policy_sha256,
        )

    @torch.inference_mode()
    def score(
        self,
        market_latent: NDArray[np.float32],
        normalized_market_features: NDArray[np.float32],
        terminal_outcome_logit: float,
        uncertainty_log_scale: float,
        portfolio_features: tuple[float, ...],
        prediction_memory_features: tuple[float, ...],
        base_action_logits: tuple[float, float, float],
        protocol_action_values: tuple[float, float, float],
        execution_context: tuple[float, ...],
        candidate_action_id: int,
    ) -> H022DecisionDebug:
        if (
            market_latent.shape != (512,)
            or normalized_market_features.shape != (128,)
            or len(portfolio_features) != 9
            or len(prediction_memory_features) != 7
            or len(execution_context) != 6
            or candidate_action_id not in (0, 1)
        ):
            raise ValueError("H022 runtime candidate inputs are invalid")
        tree_features = assemble_tree_features(
            normalized_market_features[None, :],
            np.asarray([terminal_outcome_logit], dtype=np.float32),
            np.asarray([uncertainty_log_scale], dtype=np.float32),
            np.asarray([portfolio_features], dtype=np.float32),
            np.asarray([prediction_memory_features], dtype=np.float32),
            np.asarray([base_action_logits], dtype=np.float32),
            np.asarray([protocol_action_values], dtype=np.float32),
            np.asarray([execution_context], dtype=np.float32),
            np.asarray([candidate_action_id], dtype=np.int64),
        )
        latent = (market_latent - self.statistics["latent_mean"]) / self.statistics[
            "latent_scale"
        ]
        normalized_tree = (
            tree_features - self.statistics["tree_mean"]
        ) / self.statistics["tree_scale"]
        neural_inputs = (
            torch.from_numpy(latent[None, :]).to(self.device),
            torch.from_numpy(normalized_tree).to(self.device),
            torch.tensor(
                [terminal_outcome_logit], dtype=torch.float32, device=self.device
            ),
            torch.tensor(
                [candidate_action_id], dtype=torch.long, device=self.device
            ),
            torch.tensor(
                [tree_features[0, H022_BREAK_EVEN_INDEX]],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        output = self.neural(*neural_inputs, return_debug=True)
        mean = float(output["net_return_mean"][0].float())
        quantiles = tuple(
            float(value)
            for value in output["net_return_quantiles"][0].float().cpu().tolist()
        )
        if len(quantiles) != 3:
            raise RuntimeError("H022 neural quantile width changed")
        fill = float(output["fill_probability"][0].float())
        probability0 = float(output["calibrated_outcome_probability0"][0].float())
        edge = float(output["calibrated_candidate_edge"][0].float())
        attention = tuple(
            float(value)
            for value in output["debug_group_attention"][0].float().cpu().tolist()
        )
        tree_score = float(np.asarray(self.tree.predict(tree_features)).reshape(-1)[0])
        predictors = np.asarray(
            [[mean, *quantiles, fill, edge, tree_score]], dtype=np.float64
        )
        ensemble = float(predict_weighted_ridge(predictors, self.stacker)[0])
        contribution_raw = np.asarray(
            self.tree.predict(tree_features, pred_contrib=True), dtype=np.float64
        ).reshape(-1)
        if contribution_raw.shape != (H022_TREE_FEATURE_WIDTH + 1,):
            raise RuntimeError("H022 tree contribution width changed")
        group_ranges = ((0, 8), (8, 48), (48, 72), (72, 116), (116, 128), (128, 170))
        tree_groups = (
            0.0,
            *(float(contribution_raw[start:stop].sum()) for start, stop in group_ranges),
        )
        stacker_mean = np.asarray(self.stacker["feature_mean"], dtype=np.float64)
        stacker_scale = np.asarray(self.stacker["feature_scale"], dtype=np.float64)
        stacker_coefficients = np.asarray(
            self.stacker["coefficients"], dtype=np.float64
        )
        stacker_contributions = tuple(
            float(value)
            for value in (
                (predictors[0] - stacker_mean)
                / stacker_scale
                * stacker_coefficients
            ).tolist()
        )
        return H022DecisionDebug(
            keep_base_call=ensemble > 0.0,
            candidate_action_id=candidate_action_id,
            neural_mean_net_return=mean,
            neural_return_quantiles=(quantiles[0], quantiles[1], quantiles[2]),
            neural_fill_probability=fill,
            neural_calibrated_probability0=probability0,
            neural_calibrated_candidate_edge=edge,
            tree_net_return=tree_score,
            ensemble_net_return=ensemble,
            stacker_intercept=float(self.stacker["intercept"]),
            stacker_contributions=stacker_contributions,
            neural_group_attention=attention,
            tree_group_contributions=tree_groups,
            tree_price_context_contribution=float(
                contribution_raw[self.price_indices].sum()
            ),
            tree_wallet_contribution=float(contribution_raw[72:116].sum()),
            tree_event_contribution=float(contribution_raw[48:72].sum()),
        )


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def h022_debug_payload(debug: H022DecisionDebug) -> dict[str, Any]:
    """Convert one decision debug record to stable audit JSON."""

    return {
        "keep_base_call": debug.keep_base_call,
        "candidate_action_id": debug.candidate_action_id,
        "member_scores": {
            "neural_mean_net_return": debug.neural_mean_net_return,
            "tree_net_return": debug.tree_net_return,
            "ensemble_net_return": debug.ensemble_net_return,
        },
        "calibrated_probability0": debug.neural_calibrated_probability0,
        "calibrated_candidate_edge": debug.neural_calibrated_candidate_edge,
        "net_return_quantiles": list(debug.neural_return_quantiles),
        "fill_probability": debug.neural_fill_probability,
        "stacker": {
            "intercept": debug.stacker_intercept,
            "contributions": list(debug.stacker_contributions),
        },
        "attribution": {
            "group_ids": list(H022_GROUP_IDS),
            "neural_group_attention": list(debug.neural_group_attention),
            "tree_group_contributions": list(debug.tree_group_contributions),
            "tree_price_context": debug.tree_price_context_contribution,
            "tree_wallet": debug.tree_wallet_contribution,
            "tree_event": debug.tree_event_contribution,
        },
    }
