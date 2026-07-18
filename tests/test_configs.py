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
    assert config["status"] == "registered"
    assert config["research_id"] == "SPH-T-H005"
    assert config["point_in_time"]["future_features_allowed"] is False
    assert config["point_in_time"]["future_wallet_performance_allowed"] is False
    assert config["split"]["test_labels_opened"] is False


def test_trial_t0_contract_keeps_test_labels_closed() -> None:
    config = load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_trial_t0.json")
    assert config["research_id"] == "SPH-T-H005"
    assert config["split"]["test_labels_opened"] is False
    assert config["split"]["development_builder_emits_test_rows"] is False
    assert config["targets"]["markout_executable"] is False
    assert config["targets"]["net_edge_proxy"]["executable_evidence"] is False


def test_trial_t0_learning_receipt_cannot_promote_checkpoint() -> None:
    config = load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_trial_t0_train.json")
    result = load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_trial_t0_result.json")

    assert config["research_id"] == "SPH-T-H006"
    assert config["dataset"]["test_labels_opened"] is False
    assert config["acceptance"]["model_promotion_allowed"] is False
    assert result["decision"] == "diagnostic_only_no_promotion"
    assert result["test_labels_opened"] is False
    assert result["feature_pack"]["test_rows"] == 0
    assert result["metrics"]["validation"]["continuous_heads_better_than_zero_baseline"] == 0


def test_h007_keeps_price_as_input_and_wallet_result_inconclusive() -> None:
    config = load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h007_ablation.json")
    result = load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h007_result.json")

    assert config["research_id"] == "SPH-T-H007"
    assert {tuple(variant["outputs"]) for variant in config["variants"]} == {("resolved_yes",)}
    assert config["comparison"]["position_rule"] == "hold_to_resolution"
    assert config["controlled_constants"]["test_labels_opened"] is False
    assert result["wallet_signal_supported"] is False
    assert result["test_labels_opened"] is False
    assert result["decision"] == "diagnostic_only_no_promotion"


def test_h008_prioritizes_model_quality_and_stateful_full_universe() -> None:
    config = load_json(ROOT / "configs" / "trace" / "sphinx_trace_research_mandate_v1.json")

    assert config["research_id"] == "SPH-T-H008"
    assert config["universe"]["categories"] == "all"
    assert config["universe"]["include_multi_outcome"] is True
    assert config["universe"]["include_neg_risk"] is True
    assert config["information"]["hard_wallet_count_cap"] is None
    assert config["information"]["polygon_funding_and_transfer_graph_required"] is True
    assert config["outputs"]["learned_skip"] is True
    assert config["outputs"]["learned_position_size_from_current_balance_and_state"] is True
    assert config["prediction_memory"]["retain_full_prediction_trajectory"] is True
    assert config["objective"]["compute_economy_is_an_objective"] is False
    assert config["training"]["graceful_pause_and_exact_resume_required"] is True
    assert config["evaluation"]["primary_metric"] == "net_profit_after_costs_in_full_simulator"


def test_h009_chronicle_is_full_universe_causal_and_resumable() -> None:
    config = load_json(ROOT / "configs" / "corpus" / "sphinx_chronicle_h009_v1.json")

    assert config["research_id"] == "SPH-T-H009"
    assert config["sources"]["ledger"]["rows"] == 176_119_673
    assert config["sources"]["ledger"]["hard_wallet_count_cap"] is None
    assert config["sources"]["ledger"]["all_valid_rows_preserved"] is True
    assert config["sources"]["polygon"]["required"] is True
    assert config["sources"]["polygon"]["full_qualification_requires_complete_backfill"] is True
    assert config["episode"]["unit"] == "connected_event_component"
    assert config["episode"]["test_terminal_fields_accessed"] is False
    assert config["split"]["development_builder_emits_test_terminal_labels"] is False
    assert config["resume"]["checkpoint_maximum_interval_seconds"] <= 900
    assert config["acceptance"]["ledger_rows_preserved_exactly"] == 176_119_673


def test_h010_simulator_keeps_strategy_learned_and_execution_physical() -> None:
    config = load_json(ROOT / "configs" / "trace" / "sphinx_trace_simulator_h010_v1.json")
    assert config["research_id"] == "SPH-T-H010"
    assert config["portfolio"]["strategy_position_count_limit"] is None
    assert config["execution"]["allow_partial_fills"] is True
    assert config["execution"]["historical_trade_tape_proxy"]["qualification_evidence"] is False
    assert (
        config["execution"]["historical_orderbook"]["required_for_simulator_qualification"] is True
    )
    assert config["causality"]["test_labels_opened"] is False


def test_h011_campaign_is_uncapped_and_cannot_open_test() -> None:
    config = load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_campaign_v1.json")
    assert config["research_id"] == "SPH-T-H011"
    assert config["data"]["hard_trade_cap"] is None
    assert config["data"]["hard_wallet_cap"] is None
    assert config["data"]["all_h009_ledger_rows_processed"] is True
    assert (
        config["causal_state"]["resolution_update_rule"]
        == "wallet_performance_changes_only_after_registered_public_resolution_time"
    )
    assert config["architecture"]["first_candidate_target"] == 50_000_000
    assert config["architecture"]["capacity_selection_ignores_compute_cost"] is True
    assert config["selection"]["test_opening_allowed"] is False


def test_h011_pack_streams_all_rows_without_identity_features() -> None:
    config = load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_pack_v1.json")
    assert config["research_id"] == "SPH-T-H011"
    assert config["features"]["width"] == 128
    assert config["features"]["identity_embedding"] is False
    assert config["features"]["hard_wallet_cap"] is None
    assert config["features"]["hard_market_cap"] is None
    assert config["acceptance"]["full_stream_rows"] == 176_119_673
    assert config["input"]["test_labels_opened"] is False
    assert config["storage"]["checkpoint_maximum_interval_seconds"] <= 900


def test_h011_resolution_context_is_causal_and_keeps_test_closed() -> None:
    config = load_json(ROOT / "configs" / "trace" / "sphinx_trace_h011_resolution_context_v1.json")
    assert config["research_id"] == "SPH-T-H011"
    assert config["input"]["test_terminal_fields_accessed"] is False
    assert config["input"]["same_second_application"] == "strictly_after_resolution_second"
    assert config["proxy"]["rows_after_resolution_used"] is False
    assert config["acceptance"]["hard_wallet_cap"] is None


def test_h011_model_and_training_are_resumable_and_multi_head() -> None:
    model = load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json")
    training = load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_train_v1.json")
    assert [candidate["id"] for candidate in model["architecture"]["candidates"]] == [
        "50m",
        "100m",
        "150m",
    ]
    assert "call_sufficiency_logit" in model["heads"]
    assert "position_size_beta_alpha" in model["heads"]
    assert training["data"]["test_rows_consumed"] == 0
    assert training["evaluation"]["test_opening_allowed"] is False
    assert training["training"]["checkpoint_maximum_interval_seconds"] <= 900


def test_h014_replay_state_training_keeps_selection_and_test_closed() -> None:
    result = load_json(
        ROOT / "configs" / "trace" / "sphinx_trace_s0_h012_v2_exact_validation_result.json"
    )
    config = load_json(
        ROOT / "configs" / "trace" / "sphinx_trace_s0_h014_replay_state_distillation_v1.json"
    )

    assert result["promotion_allowed"] is False
    assert result["gates"]["all_pass"] is False
    assert config["research_id"] == "SPH-T-H014"
    assert config["corpus"]["rows"] == 809_614
    assert config["corpus"]["teacher_action_is_input_or_target"] is False
    assert config["architecture"]["fixed_confidence_threshold"] is None
    assert config["architecture"]["fixed_call_frequency"] is None
    assert config["partition"]["calibration_used_for_training_or_selection"] is False
    assert config["corpus"]["test_labels_opened"] is False
    assert config["training"]["checkpoint_maximum_interval_seconds"] <= 900


def test_h014_exact_result_rejects_promotion_and_h015_is_pre_registered() -> None:
    h014 = load_json(
        ROOT / "configs" / "trace" / "sphinx_trace_s0_h014_exact_validation_result.json"
    )
    h015 = load_json(
        ROOT
        / "configs"
        / "trace"
        / "sphinx_trace_s0_h015_on_policy_portfolio_advantage_v1.json"
    )

    assert h014["research_id"] == "SPH-T-H014"
    assert h014["promotion_allowed"] is False
    assert h014["gates"]["all_pass"] is False
    assert h014["test_labels_opened"] is False

    assert h015["research_id"] == "SPH-T-H015"
    assert h015["corpus"]["rows_expected"] == 1_619_228
    assert h015["corpus"]["behavior_policies"] == 2
    assert h015["weighting"]["primary"] == "equal_market_total_weight_within_behavior_policy"
    assert h015["weighting"]["fixed_call_frequency_target"] is None
    assert h015["partition"]["partition_sha256"] == (
        "d3c2c16b9addfcf2a1f4a50beaab2eab48fb3d12e92f820047eb424c5608f3bf"
    )
    assert h015["partition"]["calibration_used_for_training_or_selection"] is False
    assert h015["corpus"]["calibration_rows_consumed"] == 0
    assert h015["corpus"]["test_rows_consumed"] == 0
    assert h015["corpus"]["test_labels_opened"] is False
    assert h015["architecture"]["fixed_confidence_threshold"] is None
    assert h015["architecture"]["fixed_edge_threshold"] is None
    assert h015["architecture"]["fixed_bet_size"] is None
    assert h015["architecture"]["fixed_portfolio_limit"] is None

    pack = load_json(
        ROOT / "configs" / "trace" / "sphinx_trace_s0_h015_on_policy_pack_v1_result.json"
    )
    assert pack["valid"] is True
    assert pack["rows"] == 1_619_228
    assert pack["fit_rows"] == 1_204_402
    assert pack["selection_rows"] == 414_826
    assert pack["source_reproduction"]["H012_net_profit_matches_exact_replay"] is True
    assert pack["source_reproduction"]["H014_net_profit_matches_exact_replay"] is True
    assert pack["source_reproduction"]["teacher_action_used_as_imitation_label"] is False
    assert pack["source_reproduction"]["test_labels_opened"] is False


def test_corpus_v1_covers_both_clob_protocols() -> None:
    config = load_json(ROOT / "configs" / "corpus" / "sphinx_corpus_v1.json")
    contracts = config["sources"]["ledger"]["contracts"]
    assert {item["protocol"] for item in contracts} == {"clob-v1", "clob-v2"}
    assert {item["market_type"] for item in contracts} == {"standard", "neg_risk"}
    assert config["window"]["end_exclusive"] == "2026-07-16T00:00:00Z"
