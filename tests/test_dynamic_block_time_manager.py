"""
Tests for dynamic block time manager critical runtime paths.
"""

import time

import pytest

from dessin.runtime.config import ConsensusConfig
from dessin.consensus.dynamic_block_time import (
    BlockTimeAdjustment,
    BlockTimeAdjustmentProposal,
    DynamicBlockTimeManager,
    VerificationTiming,
)
from dessin.consensus.two_phase_verification import VerificationPhase


def _mk_manager() -> DynamicBlockTimeManager:
    config = ConsensusConfig()
    config.enable_dynamic_block_time = True
    config.verification_timeout_threshold = 0.5
    config.block_time_adjustment_seconds = 10
    config.training_block_time_minutes = 60.0
    return DynamicBlockTimeManager(
        config,
        use_adaptive_timing=True,
        use_logarithmic_scaling=True,
        max_scaling_factor=1000.0,
        min_block_time_seconds=40.0,
    )


def _timing(
    phase: VerificationPhase,
    allocated: float,
    actual: float,
    threshold: float = 0.5,
) -> VerificationTiming:
    start = time.time() - actual
    end = time.time()
    vt = VerificationTiming(
        phase=phase,
        allocated_time_seconds=allocated,
        actual_time_seconds=actual,
        start_time=start,
        end_time=end,
        timeout_threshold=threshold,
    )
    vt.is_timeout = vt.is_exceeding_threshold()
    return vt


def test_start_and_end_verification_timing_sets_fields():
    manager = _mk_manager()
    block_hash = "block_timing_1"

    manager.current_block_time_seconds = 100.0
    manager.start_verification_timing(block_hash, VerificationPhase.TRAINING)
    manager.start_verification_timing(block_hash, VerificationPhase.PHASE1)

    assert len(manager.verification_timings[block_hash]) == 2
    assert manager.verification_timings[block_hash][0].allocated_time_seconds == 100.0
    assert manager.verification_timings[block_hash][1].allocated_time_seconds == 30.0

    training_result = manager.end_verification_timing(block_hash, VerificationPhase.TRAINING)
    assert training_result is not None
    assert training_result.actual_time_seconds >= 0.0
    assert training_result.end_time > 0.0


def test_end_verification_timing_returns_none_when_block_or_phase_missing():
    manager = _mk_manager()
    assert manager.end_verification_timing("missing", VerificationPhase.PHASE1) is None

    block_hash = "block_timing_2"
    manager.start_verification_timing(block_hash, VerificationPhase.TRAINING)
    assert manager.end_verification_timing(block_hash, VerificationPhase.PHASE2) is None


def test_analyze_verification_performance_returns_none_when_disabled_or_empty():
    manager = _mk_manager()
    block_hash = "analyze_none_1"

    manager.config.enable_dynamic_block_time = False
    assert manager.analyze_verification_performance(block_hash) is None

    manager.config.enable_dynamic_block_time = True
    assert manager.analyze_verification_performance(block_hash) is None


def test_analyze_verification_performance_increase_for_high_timeout_ratio():
    manager = _mk_manager()
    block_hash = "analyze_increase_1"
    manager.current_block_time_seconds = 100.0
    manager.verification_timings[block_hash] = [
        _timing(VerificationPhase.TRAINING, 100.0, 80.0),  # timeout
        _timing(VerificationPhase.PHASE1, 30.0, 16.0),  # timeout
        _timing(VerificationPhase.PHASE2, 30.0, 10.0),  # no-timeout
    ]

    proposal = manager.analyze_verification_performance(block_hash)
    assert proposal is not None
    assert proposal.adjustment_type == BlockTimeAdjustment.INCREASE
    assert proposal.proposed_block_time_seconds > proposal.current_block_time_seconds


def test_analyze_verification_performance_decrease_for_low_timeout_ratio():
    manager = _mk_manager()
    block_hash = "analyze_decrease_1"
    manager.current_block_time_seconds = 200.0
    manager.verification_timings[block_hash] = [
        _timing(VerificationPhase.TRAINING, 200.0, 20.0),  # no-timeout
        _timing(VerificationPhase.PHASE1, 60.0, 10.0),  # no-timeout
    ]

    proposal = manager.analyze_verification_performance(block_hash)
    assert proposal is not None
    assert proposal.adjustment_type == BlockTimeAdjustment.DECREASE
    assert proposal.proposed_block_time_seconds < proposal.current_block_time_seconds


def test_analyze_verification_performance_maintain_for_mid_timeout_ratio():
    manager = _mk_manager()
    block_hash = "analyze_maintain_1"
    manager.current_block_time_seconds = 100.0
    manager.verification_timings[block_hash] = [
        _timing(VerificationPhase.TRAINING, 100.0, 80.0),  # timeout
        _timing(VerificationPhase.PHASE1, 30.0, 10.0),  # no-timeout
        _timing(VerificationPhase.PHASE2, 30.0, 10.0),  # no-timeout
    ]

    proposal = manager.analyze_verification_performance(block_hash)
    assert proposal is not None
    assert proposal.adjustment_type == BlockTimeAdjustment.MAINTAIN
    assert proposal.proposed_block_time_seconds == proposal.current_block_time_seconds


def test_propose_vote_apply_adjustment_flow():
    manager = _mk_manager()
    block_hash = "governance_flow_1"
    manager.current_block_time_seconds = 100.0
    manager.verification_timings[block_hash] = [
        _timing(VerificationPhase.TRAINING, 100.0, 80.0),  # timeout
        _timing(VerificationPhase.PHASE1, 30.0, 20.0),  # timeout
    ]

    proposal = manager.propose_block_time_adjustment(block_hash, proposer="validator_a")
    assert proposal is not None
    assert proposal.proposer == "validator_a"
    assert block_hash in manager.pending_adjustments
    assert len(manager.pending_adjustments[block_hash]) == 1

    assert manager.vote_on_adjustment(block_hash, voter="validator_b", approve=True) is True
    assert block_hash in manager.approved_adjustments

    old_time = manager.current_block_time_seconds
    assert manager.apply_approved_adjustment(block_hash) is True
    assert manager.current_block_time_seconds != old_time
    assert block_hash not in manager.pending_adjustments
    assert block_hash not in manager.approved_adjustments
    assert len(manager.recent_adjustments) == 1


def test_vote_and_apply_fail_for_missing_state():
    manager = _mk_manager()
    assert manager.vote_on_adjustment("missing", voter="v1", approve=True) is False
    assert manager.apply_approved_adjustment("missing") is False


def test_get_performance_summary_empty_and_populated():
    manager = _mk_manager()
    summary = manager.get_performance_summary()
    assert "error" in summary

    block_hash = "summary_1"
    manager.verification_timings[block_hash] = [
        _timing(VerificationPhase.TRAINING, 100.0, 80.0),  # timeout
        _timing(VerificationPhase.PHASE1, 30.0, 10.0),  # no-timeout
    ]
    summary = manager.get_performance_summary()
    assert summary["total_blocks_analyzed"] == 1
    assert summary["total_verification_phases"] == 2
    assert summary["total_timeouts"] == 1
    assert summary["timeout_ratio"] == pytest.approx(0.5)


def test_cleanup_old_data_does_not_crash_and_preserves_recent_adjustments():
    manager = _mk_manager()
    block_hash = "cleanup_1"
    manager.current_block_time_seconds = 100.0
    manager.verification_timings[block_hash] = [
        _timing(VerificationPhase.TRAINING, 100.0, 80.0),  # timeout
        _timing(VerificationPhase.PHASE1, 30.0, 20.0),  # timeout
    ]
    proposal = manager.propose_block_time_adjustment(block_hash, proposer="validator_x")
    assert proposal is not None
    manager.vote_on_adjustment(block_hash, voter="validator_y", approve=True)
    manager.apply_approved_adjustment(block_hash)
    assert len(manager.recent_adjustments) == 1

    manager.cleanup_old_data(current_block_index=1000, retention_blocks=100)
    assert len(manager.recent_adjustments) == 1


def test_verification_timing_progress_ratio_when_allocated_zero():
    vt = VerificationTiming(
        phase=VerificationPhase.TRAINING,
        allocated_time_seconds=0.0,
        actual_time_seconds=5.0,
        start_time=0.0,
        end_time=5.0,
        timeout_threshold=0.5,
    )
    assert vt.get_progress_ratio() == 0.0
    assert vt.is_exceeding_threshold() is False


def test_block_time_adjustment_proposal_is_valid_on_bounds():
    cfg = ConsensusConfig()
    cfg.min_block_time_seconds = 30
    cfg.max_block_time_seconds = 600
    ts = time.time()
    proposal_at_min = BlockTimeAdjustmentProposal(
        adjustment_type=BlockTimeAdjustment.MAINTAIN,
        current_block_time_seconds=100.0,
        proposed_block_time_seconds=30.0,
        reason="at min",
        verification_timings=[],
        timestamp=ts,
        proposer="t",
    )
    proposal_at_max = BlockTimeAdjustmentProposal(
        adjustment_type=BlockTimeAdjustment.MAINTAIN,
        current_block_time_seconds=100.0,
        proposed_block_time_seconds=600.0,
        reason="at max",
        verification_timings=[],
        timestamp=ts,
        proposer="t",
    )
    assert proposal_at_min.is_valid(cfg)
    assert proposal_at_max.is_valid(cfg)

    below_min = BlockTimeAdjustmentProposal(
        adjustment_type=BlockTimeAdjustment.DECREASE,
        current_block_time_seconds=100.0,
        proposed_block_time_seconds=29.0,
        reason="below",
        verification_timings=[],
        timestamp=ts,
        proposer="t",
    )
    above_max = BlockTimeAdjustmentProposal(
        adjustment_type=BlockTimeAdjustment.INCREASE,
        current_block_time_seconds=100.0,
        proposed_block_time_seconds=600.1,
        reason="above",
        verification_timings=[],
        timestamp=ts,
        proposer="t",
    )
    assert not below_min.is_valid(cfg)
    assert not above_max.is_valid(cfg)
