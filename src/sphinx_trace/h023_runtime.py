"""Receipt-bound H023 fill-realized veto inference and decision debugging."""

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
from sphinx_trace.h022_runtime import H022DecisionDebug
from sphinx_trace.h022_training import predict_weighted_ridge
from sphinx_trace.model_h023 import (
    H023_AUX_FEATURE_WIDTH,
    H023_GROUP_IDS,
    SphinxTraceS0H023NeuralMember,
)

H023_BREAK_EVEN_INDEX = H022_TREE_FEATURE_NAMES.index(
    "candidate.break_even_probability"
)
H023_PREDICTOR_NAMES = (
    "neural_realized_contribution",
    "neural_conditional_return_mean",
    "neural_conditional_return_q10",
    "neural_conditional_return_q50",
    "neural_conditional_return_q90",
    "neural_fill_probability",
    "neural_positive_probability",
    "neural_keep_logit",
    "h022_neural_mean",
    "h022_tree",
    "h022_ensemble",
    "h023_tree_realized_contribution",
)


class TreePredictor(Protocol):
    def predict(
        self,
        data: NDArray[np.float32],
        *,
        pred_contrib: bool = False,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class H023FeatureAttribution:
    feature: str
    value: float
    contribution: float


@dataclass(frozen=True, slots=True)
class H023DecisionDebug:
    keep_base_call: bool
    gate_reason: str
    candidate_action_id: int
    entry_price: float
    break_even_probability: float
    neural_realized_contribution: float
    neural_conditional_return_mean: float
    neural_conditional_return_quantiles: tuple[float, float, float]
    neural_fill_probability: float
    neural_positive_probability: float
    neural_keep_logit: float
    tree_realized_contribution: float
    ensemble_realized_contribution: float
    stacker_intercept: float
    stacker_contributions: tuple[float, ...]
    neural_group_attention: tuple[float, ...]
    tree_group_contributions: tuple[float, ...]
    top_tree_features: tuple[H023FeatureAttribution, ...]
    h022_ensemble_net_return: float


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


class H023EnsembleRuntime:
    """Load and score the selected full-calibration H023 runtime artifact."""

    def __init__(
        self,
        neural: SphinxTraceS0H023NeuralMember,
        tree: TreePredictor,
        statistics: dict[str, NDArray[np.float32]],
        stacker: dict[str, Any],
        device: torch.device,
        *,
        policy_sha256: str,
    ) -> None:
        required = {
            "latent_mean",
            "latent_scale",
            "tree_mean",
            "tree_scale",
            "aux_mean",
            "aux_scale",
        }
        if set(statistics) != required or len(policy_sha256) != 64:
            raise ValueError("H023 runtime artifact contract is incomplete")
        if (
            statistics["latent_mean"].shape != (512,)
            or statistics["latent_scale"].shape != (512,)
            or statistics["tree_mean"].shape != (H022_TREE_FEATURE_WIDTH,)
            or statistics["tree_scale"].shape != (H022_TREE_FEATURE_WIDTH,)
            or statistics["aux_mean"].shape != (H023_AUX_FEATURE_WIDTH,)
            or statistics["aux_scale"].shape != (H023_AUX_FEATURE_WIDTH,)
        ):
            raise ValueError("H023 runtime normalizer shape changed")
        if tuple(stacker.get("predictor_names", ())) != H023_PREDICTOR_NAMES:
            raise ValueError("H023 runtime stacker feature contract changed")
        self.neural = neural.to(device).eval()
        self.tree = tree
        self.statistics = statistics
        self.stacker = stacker
        self.device = device
        self.policy_sha256 = policy_sha256

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: Path,
        summary_path: Path,
        device: torch.device,
    ) -> H023EnsembleRuntime:
        summary = _load_object(summary_path)
        result_path = artifact_dir / "result.json"
        neural_path = artifact_dir / "neural" / "best-neural.pt"
        tree_path = artifact_dir / "tree.txt"
        stacker_path = artifact_dir / "stacker.json"
        analog_path = artifact_dir / "realized-training-analogs.npz"
        if (
            summary.get("research_id") != "SPH-T-H023"
            or summary.get("valid") is not True
            or summary.get("test_labels_opened") is not False
            or int(summary.get("test_rows_consumed", -1)) != 0
            or summary.get("promotion_allowed") is not False
            or sha256_file(result_path)
            != summary.get("selected_final_result_sha256")
            or sha256_file(neural_path)
            != summary.get("selected_final_neural_sha256")
            or sha256_file(tree_path) != summary.get("selected_final_tree_sha256")
            or sha256_file(stacker_path)
            != summary.get("selected_final_stacker_sha256")
            or sha256_file(analog_path)
            != summary.get("selected_final_realized_training_analogs_sha256")
        ):
            raise RuntimeError("H023 selected runtime artifact receipt changed")
        result = _load_object(result_path)
        if (
            result.get("record_type") != "h023_selected_full_calibration_runtime"
            or result.get("valid") is not True
            or result.get("test_labels_opened") is not False
            or int(result.get("test_rows_consumed", -1)) != 0
            or result.get("validation_labels_opened_for_H023") is not False
            or result.get("promotion_allowed") is not False
        ):
            raise RuntimeError("H023 runtime requires a closed-validation artifact")
        checkpoint = torch.load(neural_path, map_location="cpu", weights_only=False)
        if checkpoint.get("record_type") != "h023_neural_member":
            raise RuntimeError("H023 neural artifact type changed")
        neural_config = checkpoint.get("config")
        raw_statistics = checkpoint.get("statistics")
        if not isinstance(neural_config, dict) or not isinstance(raw_statistics, dict):
            raise RuntimeError("H023 neural artifact metadata is incomplete")
        neural = SphinxTraceS0H023NeuralMember(neural_config)
        neural.load_state_dict(checkpoint["model"])
        statistics = {
            str(name): np.asarray(value, dtype=np.float32)
            for name, value in raw_statistics.items()
        }
        stacker = _load_object(stacker_path)
        policy_payload = {
            "summary_sha256": sha256_file(summary_path),
            "result_sha256": sha256_file(result_path),
            "neural_sha256": sha256_file(neural_path),
            "tree_sha256": sha256_file(tree_path),
            "stacker_sha256": sha256_file(stacker_path),
            "realized_training_analogs_sha256": sha256_file(analog_path),
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
        size_fraction: float,
        h022: H022DecisionDebug,
    ) -> H023DecisionDebug:
        if (
            market_latent.shape != (512,)
            or normalized_market_features.shape != (128,)
            or len(portfolio_features) != 9
            or len(prediction_memory_features) != 7
            or len(execution_context) != 6
            or candidate_action_id not in (0, 1)
            or not np.isfinite(size_fraction)
            or not 0.0 <= size_fraction <= 1.0
            or h022.candidate_action_id != candidate_action_id
        ):
            raise ValueError("H023 runtime candidate inputs are invalid")
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
        auxiliary = np.asarray(
            [
                h022.neural_mean_net_return,
                *h022.neural_return_quantiles,
                h022.neural_fill_probability,
                h022.neural_calibrated_probability0,
                h022.neural_calibrated_candidate_edge,
                h022.tree_net_return,
                h022.ensemble_net_return,
                float(h022.keep_base_call),
                size_fraction,
            ],
            dtype=np.float32,
        )
        if auxiliary.shape != (H023_AUX_FEATURE_WIDTH,):
            raise RuntimeError("H023 auxiliary feature width changed")
        normalized_latent = (
            market_latent - self.statistics["latent_mean"]
        ) / self.statistics["latent_scale"]
        normalized_tree = (
            tree_features - self.statistics["tree_mean"]
        ) / self.statistics["tree_scale"]
        normalized_aux = (
            auxiliary - self.statistics["aux_mean"]
        ) / self.statistics["aux_scale"]
        output = self.neural(
            torch.from_numpy(normalized_latent[None, :]).to(self.device),
            torch.from_numpy(normalized_tree).to(self.device),
            torch.tensor(
                [terminal_outcome_logit], dtype=torch.float32, device=self.device
            ),
            torch.tensor(
                [candidate_action_id], dtype=torch.long, device=self.device
            ),
            torch.tensor(
                [tree_features[0, H023_BREAK_EVEN_INDEX]],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.from_numpy(normalized_aux[None, :]).to(self.device),
            return_debug=True,
        )
        contribution = float(output["realized_net_contribution_mean"][0].float())
        return_mean = float(output["conditional_realized_return_mean"][0].float())
        quantiles = tuple(
            float(value)
            for value in output["conditional_realized_return_quantiles"][0]
            .float()
            .cpu()
            .tolist()
        )
        if len(quantiles) != 3:
            raise RuntimeError("H023 neural quantile width changed")
        fill = float(output["fill_probability"][0].float())
        positive = float(
            output["probability_realized_contribution_positive"][0].float()
        )
        keep_logit = float(output["keep_base_call_logit"][0].float())
        attention = tuple(
            float(value)
            for value in output["debug_group_attention"][0].float().cpu().tolist()
        )
        tree_score = float(np.asarray(self.tree.predict(tree_features)).reshape(-1)[0])
        predictors = np.asarray(
            [
                [
                    contribution,
                    return_mean,
                    *quantiles,
                    fill,
                    positive,
                    keep_logit,
                    h022.neural_mean_net_return,
                    h022.tree_net_return,
                    h022.ensemble_net_return,
                    tree_score,
                ]
            ],
            dtype=np.float64,
        )
        ensemble = float(predict_weighted_ridge(predictors, self.stacker)[0])
        contribution_raw = np.asarray(
            self.tree.predict(tree_features, pred_contrib=True), dtype=np.float64
        ).reshape(-1)
        if contribution_raw.shape != (H022_TREE_FEATURE_WIDTH + 1,):
            raise RuntimeError("H023 tree contribution width changed")
        group_ranges = ((0, 8), (8, 48), (48, 72), (72, 116), (116, 128), (128, 170))
        tree_groups = (
            0.0,
            *(float(contribution_raw[start:stop].sum()) for start, stop in group_ranges),
        )
        top_indices = np.argsort(np.abs(contribution_raw[:-1]))[::-1][:16]
        top_features = tuple(
            H023FeatureAttribution(
                H022_TREE_FEATURE_NAMES[int(index)],
                float(tree_features[0, index]),
                float(contribution_raw[index]),
            )
            for index in top_indices
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
        keep = ensemble > 0.0
        return H023DecisionDebug(
            keep_base_call=keep,
            gate_reason=(
                "positive_expected_realized_net_contribution"
                if keep
                else "nonpositive_expected_realized_net_contribution"
            ),
            candidate_action_id=candidate_action_id,
            entry_price=float(execution_context[candidate_action_id]),
            break_even_probability=float(
                tree_features[0, H023_BREAK_EVEN_INDEX]
            ),
            neural_realized_contribution=contribution,
            neural_conditional_return_mean=return_mean,
            neural_conditional_return_quantiles=(
                quantiles[0],
                quantiles[1],
                quantiles[2],
            ),
            neural_fill_probability=fill,
            neural_positive_probability=positive,
            neural_keep_logit=keep_logit,
            tree_realized_contribution=tree_score,
            ensemble_realized_contribution=ensemble,
            stacker_intercept=float(self.stacker["intercept"]),
            stacker_contributions=stacker_contributions,
            neural_group_attention=attention,
            tree_group_contributions=tree_groups,
            top_tree_features=top_features,
            h022_ensemble_net_return=h022.ensemble_net_return,
        )


def h023_debug_payload(debug: H023DecisionDebug) -> dict[str, Any]:
    """Convert one H023 decision debug record to stable audit JSON."""

    return {
        "keep_base_call": debug.keep_base_call,
        "gate_reason": debug.gate_reason,
        "candidate_action_id": debug.candidate_action_id,
        "entry_price": debug.entry_price,
        "break_even_probability": debug.break_even_probability,
        "member_scores": {
            "neural_realized_contribution": debug.neural_realized_contribution,
            "tree_realized_contribution": debug.tree_realized_contribution,
            "ensemble_realized_contribution": debug.ensemble_realized_contribution,
            "h022_ensemble_net_return": debug.h022_ensemble_net_return,
        },
        "conditional_realized_return_mean": debug.neural_conditional_return_mean,
        "conditional_realized_return_quantiles": list(
            debug.neural_conditional_return_quantiles
        ),
        "fill_probability": debug.neural_fill_probability,
        "probability_realized_contribution_positive": (
            debug.neural_positive_probability
        ),
        "neural_keep_logit": debug.neural_keep_logit,
        "stacker": {
            "intercept": debug.stacker_intercept,
            "predictor_names": list(H023_PREDICTOR_NAMES),
            "contributions": list(debug.stacker_contributions),
        },
        "attribution": {
            "group_ids": list(H023_GROUP_IDS),
            "neural_group_attention": list(debug.neural_group_attention),
            "tree_group_contributions": list(debug.tree_group_contributions),
            "top_tree_features": [
                {
                    "feature": value.feature,
                    "value": value.value,
                    "contribution": value.contribution,
                }
                for value in debug.top_tree_features
            ],
        },
    }
