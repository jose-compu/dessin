"""Pre-flight affordability check on EnhancedPogoConsensus.

The miner must skip training when the model owner cannot pay the resulting
``EconomicSystem.process_training_payment`` (escrow + liquid; during bootstrap,
liquid plus staked rewards when the payment path allows stake debit). The
gate avoids burning gradient steps for nothing, and the consensus-side check
in ``create_enhanced_training_block`` is a defense-in-depth backstop for the
node-level filter in ``DessinNode.mine_block``.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from dessin.runtime.config import ConsensusConfig
from dessin.consensus.enhanced_pogo_consensus import EnhancedPogoConsensus
from dessin.models.model_manager import ModelInfo, ModelManager


_MID = "test_model"
_OWNER = "owner_addr_aaaa"


def _build_consensus() -> EnhancedPogoConsensus:
    cfg = ConsensusConfig()
    mm = Mock(spec=ModelManager)
    mm.get_model_info.return_value = ModelInfo(
        model_id=_MID,
        name="test",
        size_gb=0.1,
        format="pytorch",
        quantization="32bit",
        parameters=1000,
        ipfs_hash="hash",
        owner=_OWNER,
        upload_block=0,
        storage_expires=10_000,
        model_hash="h",
    )
    consensus = EnhancedPogoConsensus(cfg, mm, miner_address="miner_addr_zzzz")
    consensus.register_model_owner(_MID, _OWNER)
    return consensus


def test_estimate_training_cost_uses_catalog_size_when_no_chain_history():
    consensus = _build_consensus()
    cost = consensus.estimate_training_cost(_MID, training_steps=10)
    per_step = consensus.economic_system.training_payment_per_step
    expected_factor = 1.0 + 0.1 * 0.1
    assert cost == pytest.approx(10 * per_step * expected_factor)


def test_owner_with_zero_balance_cannot_afford_training():
    consensus = _build_consensus()
    ok, reason, cost, funds = consensus.can_owner_afford_training(_MID, training_steps=20)
    assert not ok
    assert "insufficient" in reason
    assert cost > 0.0
    assert funds == pytest.approx(0.0)


def test_escrow_alone_can_cover_training_cost():
    consensus = _build_consensus()
    cost = consensus.estimate_training_cost(_MID, training_steps=20)
    consensus.economic_system.balances["fund_src"] = cost * 2
    assert consensus.economic_system.deposit_model_training_escrow(
        "fund_src", _MID, cost * 1.5, block_index=0
    )
    ok, reason, est_cost, funds = consensus.can_owner_afford_training(_MID, training_steps=20)
    assert ok
    assert "escrow covers" in reason
    assert est_cost == pytest.approx(cost)
    assert funds == pytest.approx(cost * 1.5)


def test_owner_liquid_balance_covers_remainder():
    consensus = _build_consensus()
    cost = consensus.estimate_training_cost(_MID, training_steps=20)
    consensus.economic_system.balances[_OWNER] = cost * 2
    ok, reason, _est, funds = consensus.can_owner_afford_training(_MID, training_steps=20)
    assert ok
    assert "owner liquid" in reason
    assert funds >= cost


def test_bootstrap_stake_only_covers_remainder_when_liquid_zero():
    consensus = _build_consensus()
    cost = consensus.estimate_training_cost(_MID, training_steps=20)
    consensus.economic_system.staked_amounts[_OWNER] = cost * 2.0
    ok, reason, _est, funds = consensus.can_owner_afford_training(
        _MID, training_steps=20, block_index=1
    )
    assert ok
    assert "bootstrap stake" in reason
    assert funds >= cost


def test_post_bootstrap_height_does_not_count_stake_for_affordability():
    consensus = _build_consensus()
    cost = consensus.estimate_training_cost(_MID, training_steps=20)
    consensus.economic_system.staked_amounts[_OWNER] = cost * 2.0
    cfg = consensus.config
    cap = int(
        cfg.consensus.bootstrap_period_blocks
        if hasattr(cfg, "consensus")
        else cfg.bootstrap_period_blocks
    )
    ok, reason, _, _ = consensus.can_owner_afford_training(
        _MID, training_steps=20, block_index=max(cap, 1),
    )
    assert not ok
    assert "insufficient" in reason


def test_create_enhanced_training_block_skips_when_owner_cannot_pay():
    consensus = _build_consensus()
    consensus.model_manager.is_model_available.return_value = True
    block = consensus.create_enhanced_training_block(_MID, training_steps=20)
    assert block is None
    assert _MID not in consensus._frozen_training_models
    assert consensus._training_failure_strikes.get(_MID, 0) == 0


def test_system_owned_models_are_always_affordable():
    consensus = _build_consensus()
    consensus.model_owners["sys_only_model"] = "system"
    ok, reason, cost, funds = consensus.can_owner_afford_training(
        "sys_only_model", training_steps=20
    )
    assert ok
    assert "system-owned" in reason
    assert cost == 0.0
    assert funds == 0.0
