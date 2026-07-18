"""Calibrate H011 outcome logits and measure component-block uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sphinx_corpus.io import atomic_json, now_utc, sha256_file
from sphinx_trace.config import load_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_calibration_v1.json"


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _sigmoid(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(logits, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _log_loss_rows(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.float64],
) -> NDArray[np.float64]:
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    return -(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))


def _objective(
    logits: NDArray[np.float64],
    labels: NDArray[np.float64],
    slope: float,
    intercept: float,
    ridge: float,
) -> float:
    loss = float(_log_loss_rows(_sigmoid(slope * logits + intercept), labels).mean())
    return loss + 0.5 * ridge * ((slope - 1.0) ** 2 + intercept**2)


def fit_platt(
    logits: NDArray[np.float64],
    labels: NDArray[np.float64],
    config: dict[str, Any],
) -> tuple[float, float, list[float]]:
    """Fit a positive affine logit transform with deterministic Newton steps."""

    if logits.shape != labels.shape or logits.ndim != 1 or not len(logits):
        raise ValueError("Platt fitting requires aligned non-empty vectors")
    if not np.isfinite(logits).all() or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("Platt fitting received invalid logits or labels")
    slope = float(config["initial_slope"])
    intercept = float(config["initial_intercept"])
    minimum = float(config["minimum_slope"])
    maximum = float(config["maximum_slope"])
    ridge = float(config["ridge"])
    tolerance = float(config["tolerance"])
    history = [_objective(logits, labels, slope, intercept, ridge)]
    for _ in range(int(config["maximum_iterations"])):
        probabilities = _sigmoid(slope * logits + intercept)
        residual = probabilities - labels
        weights = probabilities * (1.0 - probabilities)
        gradient = np.array(
            [
                float(np.mean(residual * logits)) + ridge * (slope - 1.0),
                float(np.mean(residual)) + ridge * intercept,
            ],
            dtype=np.float64,
        )
        hessian = np.array(
            [
                [
                    float(np.mean(weights * logits * logits)) + ridge,
                    float(np.mean(weights * logits)),
                ],
                [float(np.mean(weights * logits)), float(np.mean(weights)) + ridge],
            ],
            dtype=np.float64,
        )
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        accepted = False
        for line_step in range(int(config["line_search_steps"])):
            scale = 0.5**line_step
            candidate_slope = min(maximum, max(minimum, slope - scale * float(step[0])))
            candidate_intercept = intercept - scale * float(step[1])
            candidate_loss = _objective(
                logits,
                labels,
                candidate_slope,
                candidate_intercept,
                ridge,
            )
            if candidate_loss <= history[-1]:
                slope = candidate_slope
                intercept = candidate_intercept
                history.append(candidate_loss)
                accepted = True
                break
        if not accepted or float(np.max(np.abs(step))) < tolerance:
            break
    return slope, intercept, history


def _metrics(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.float64],
) -> dict[str, float]:
    losses = _log_loss_rows(probabilities, labels)
    brier = float(np.mean((probabilities - labels) ** 2))
    ece = 0.0
    for lower in np.linspace(0.0, 0.95, 20):
        selected = (probabilities >= lower) & (probabilities < lower + 0.05)
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(probabilities[selected].mean()) - float(labels[selected].mean())
            )
    return {
        "rows": float(len(labels)),
        "log_loss": float(losses.mean()),
        "brier": brier,
        "accuracy": float(np.mean((probabilities >= 0.5) == labels)),
        "expected_calibration_error": ece,
    }


def component_bootstrap(
    model_probabilities: NDArray[np.float64],
    baseline_probabilities: NDArray[np.float64],
    labels: NDArray[np.float64],
    component_ids: NDArray[np.int64],
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, float | int]:
    """Bootstrap equal-weight connected-component mean paired loss deltas."""

    if not (
        model_probabilities.shape
        == baseline_probabilities.shape
        == labels.shape
        == component_ids.shape
    ):
        raise ValueError("Component bootstrap inputs must align")
    if replicates <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Component bootstrap settings are invalid")
    row_delta = _log_loss_rows(model_probabilities, labels) - _log_loss_rows(
        baseline_probabilities, labels
    )
    _, inverse = np.unique(component_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=row_delta)
    counts = np.bincount(inverse)
    component_delta = sums / counts
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    component_count = len(component_delta)
    batch_size = max(1, min(64, 2_000_000 // max(component_count, 1)))
    for offset in range(0, replicates, batch_size):
        count = min(batch_size, replicates - offset)
        indices = rng.integers(0, component_count, size=(count, component_count))
        bootstrap[offset : offset + count] = component_delta[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "components": component_count,
        "replicates": replicates,
        "mean_delta": float(component_delta.mean()),
        "lower": float(np.quantile(bootstrap, tail)),
        "upper": float(np.quantile(bootstrap, 1.0 - tail)),
    }


def calibrate(config_path: Path, model_dir: Path, output_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    result = _load_object(model_dir / "result.json")
    if (
        result.get("valid") is not True
        or result.get("test_labels_opened") is not False
        or int(result.get("test_rows_consumed", -1)) != 0
    ):
        raise RuntimeError("H011 calibration requires a valid closed-test model result")
    predictions_path = model_dir / "predictions.npz"
    if result.get("predictions_sha256") != sha256_file(predictions_path):
        raise RuntimeError("H011 prediction artifact digest changed")
    with np.load(predictions_path, allow_pickle=False) as archive:
        if any(key.startswith("test_") for key in archive.files):
            raise RuntimeError("H011 calibration encountered forbidden test arrays")
        validation_logits = np.asarray(archive["validation_logits"], dtype=np.float64)
        validation_labels = np.asarray(archive["validation_labels"], dtype=np.float64)
        validation_baselines = np.asarray(archive["validation_baselines"], dtype=np.float64)
        validation_components = np.asarray(archive["validation_component_ids"], dtype=np.int64)
        calibration_logits = np.asarray(archive["calibration_logits"], dtype=np.float64)
        calibration_labels = np.asarray(archive["calibration_labels"], dtype=np.float64)
        calibration_baselines = np.asarray(archive["calibration_baselines"], dtype=np.float64)
        calibration_components = np.asarray(archive["calibration_component_ids"], dtype=np.int64)
    slope, intercept, history = fit_platt(
        validation_logits,
        validation_labels,
        dict(config["method"]),
    )
    bootstrap_config = config["component_bootstrap"]

    def evaluate(
        logits: NDArray[np.float64],
        labels: NDArray[np.float64],
        baselines: NDArray[np.float64],
        components: NDArray[np.int64],
    ) -> dict[str, Any]:
        raw = _sigmoid(logits)
        calibrated = _sigmoid(slope * logits + intercept)
        calibrated_metrics = _metrics(calibrated, labels)
        baseline_metrics = _metrics(baselines, labels)
        return {
            "raw": _metrics(raw, labels),
            "calibrated": calibrated_metrics,
            "market_baseline": baseline_metrics,
            "calibrated_log_loss_delta_vs_market": (
                calibrated_metrics["log_loss"] - baseline_metrics["log_loss"]
            ),
            "component_bootstrap_delta_vs_market": component_bootstrap(
                calibrated,
                baselines,
                labels,
                components,
                replicates=int(bootstrap_config["replicates"]),
                seed=int(bootstrap_config["seed"]),
                confidence=float(bootstrap_config["confidence"]),
            ),
        }

    implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "h011_outcome_calibration_result",
        "generated_at": now_utc(),
        "research_id": str(config["research_id"]),
        "config_sha256": sha256_file(config_path),
        "model_result_sha256": sha256_file(model_dir / "result.json"),
        "predictions_sha256": sha256_file(predictions_path),
        "implementation_sha256": implementation_sha256,
        "valid": True,
        "candidate_id": result["candidate_id"],
        "variant_id": result["variant_id"],
        "platt_slope": slope,
        "platt_intercept": intercept,
        "fit_objective_history": history,
        "validation": evaluate(
            validation_logits,
            validation_labels,
            validation_baselines,
            validation_components,
        ),
        "calibration": evaluate(
            calibration_logits,
            calibration_labels,
            calibration_baselines,
            calibration_components,
        ),
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "evidence_boundary": str(config["evidence_boundary"]),
    }
    atomic_json(output_path, output)
    return output


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--model-dir", type=Path, required=True)
    value.add_argument("--output", type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    model_dir = args.model_dir.resolve()
    output_path = args.output.resolve() if args.output else model_dir / "calibration.json"
    result = calibrate(args.config.resolve(), model_dir, output_path)
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "variant_id": result["variant_id"],
                "platt_slope": result["platt_slope"],
                "platt_intercept": result["platt_intercept"],
                "calibration": result["calibration"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
