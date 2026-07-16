from pathlib import Path

from sphinx_trace.config import load_json

ROOT = Path(__file__).resolve().parents[1]


def test_trace_policy_has_design_status() -> None:
    config = load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_policy.json")
    assert config["id"] == "sphinx-trace-s0"
    assert config["evidence"]["status"] == "design"
    assert config["execution"]["live_trading_enabled"] is False


def test_corpus_contract_is_point_in_time() -> None:
    config = load_json(ROOT / "configs" / "corpus" / "sphinx_chronicle_v1.json")
    assert config["point_in_time"]["future_features_allowed"] is False
    assert config["point_in_time"]["future_wallet_performance_allowed"] is False
