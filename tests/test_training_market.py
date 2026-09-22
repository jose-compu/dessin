"""
Unit tests for the TrainingMarket fee-market scheduler.

Covers:
- task submission and validation
- effective-tip ordering and urgency growth
- VRF-based tie-breaking (deterministic + order-sensitive)
- cooldown enforcement
- EIP-1559 base-fee adjustment
- greedy slot packing (max_tasks_per_block)
- task expiry
- owner cancellation
- state snapshot
- PostTrainingTaskTransaction / CancelTrainingTaskTransaction round-trip
"""

import pytest
from dessin.consensus.training_market import (
    TrainingMarket,
    TrainingTaskSpec,
    ScheduledTrainingSlot,
)
from dessin.runtime.config import TrainingMarketConfig
from dessin.consensus.transactions import (
    PostTrainingTaskTransaction,
    CancelTrainingTaskTransaction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**kw) -> TrainingMarketConfig:
    cfg = TrainingMarketConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _task(
    model_id: str = "model_A",
    owner: str = "owner_1",
    submitted_block: int = 0,
    tip: float = 1.0,
    base_fee: float = 0.01,
    cooldown_blocks: int = 5,
    expiry_block: int = 100,
    task_id: str = None,
) -> TrainingTaskSpec:
    tid = task_id or TrainingTaskSpec.make_task_id(model_id, owner, submitted_block)
    return TrainingTaskSpec(
        task_id=tid,
        model_id=model_id,
        owner=owner,
        deposit=base_fee + tip,
        base_fee=base_fee,
        tip=tip,
        cooldown_blocks=cooldown_blocks,
        max_slots=1,
        submitted_block=submitted_block,
        expiry_block=expiry_block,
    )


VRF_SEED = b"test_vrf_seed_32byteslong_______"


# ---------------------------------------------------------------------------
# Submission validation
# ---------------------------------------------------------------------------

class TestSubmit:
    def test_valid_task_accepted(self):
        market = TrainingMarket(_cfg())
        t = _task()
        ok, msg = market.submit_task(t, current_block=0)
        assert ok, msg
        assert market.pending_count == 1

    def test_base_fee_too_low_rejected(self):
        market = TrainingMarket(_cfg(initial_base_fee=1.0))
        t = _task(base_fee=0.001)
        ok, msg = market.submit_task(t, current_block=0)
        assert not ok
        assert "base_fee" in msg

    def test_tip_below_min_rejected(self):
        market = TrainingMarket(_cfg(min_tip=0.5))
        t = _task(tip=0.1)
        ok, msg = market.submit_task(t, current_block=0)
        assert not ok
        assert "min_tip" in msg

    def test_deposit_too_low_rejected(self):
        market = TrainingMarket(_cfg())
        t = _task(base_fee=0.01, tip=1.0)
        t.deposit = 0.005  # deliberately too small
        ok, msg = market.submit_task(t, current_block=0)
        assert not ok
        assert "deposit" in msg

    def test_duplicate_task_id_rejected(self):
        market = TrainingMarket(_cfg())
        t = _task(task_id="fixed_id")
        market.submit_task(t, current_block=0)
        t2 = _task(model_id="model_B", task_id="fixed_id")
        ok, msg = market.submit_task(t2, current_block=0)
        assert not ok
        assert "duplicate" in msg


# ---------------------------------------------------------------------------
# Urgency multiplier
# ---------------------------------------------------------------------------

class TestUrgency:
    def test_urgency_at_submission_is_one(self):
        market = TrainingMarket(_cfg(urgency_rate=0.1))
        t = _task(submitted_block=10)
        assert market.urgency_multiplier(t, current_block=10) == 1.0

    def test_urgency_grows_linearly(self):
        market = TrainingMarket(_cfg(urgency_rate=0.1, urgency_cap=5.0))
        t = _task(submitted_block=0)
        assert market.urgency_multiplier(t, current_block=5) == pytest.approx(1.5)
        assert market.urgency_multiplier(t, current_block=10) == pytest.approx(2.0)

    def test_urgency_capped(self):
        market = TrainingMarket(_cfg(urgency_rate=0.5, urgency_cap=3.0))
        t = _task(submitted_block=0)
        # after 100 blocks, raw = 1 + 50 = 51, but cap is 3
        assert market.urgency_multiplier(t, current_block=100) == pytest.approx(3.0)

    def test_effective_tip_uses_urgency(self):
        market = TrainingMarket(_cfg(urgency_rate=0.1))
        t = _task(tip=1.0, submitted_block=0)
        assert market.effective_tip(t, current_block=10) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Task selection: ordering, VRF tie-break, slot limit
# ---------------------------------------------------------------------------

class TestSelection:
    def test_higher_tip_selected_first(self):
        market = TrainingMarket(_cfg(max_tasks_per_block=2))
        t_low = _task(model_id="low", tip=0.5, task_id="low")
        t_high = _task(model_id="high", tip=5.0, task_id="high")
        market.submit_task(t_low, current_block=0)
        market.submit_task(t_high, current_block=0)
        slots = market.select_tasks_for_block(0, VRF_SEED)
        assert slots[0].model_id == "high"

    def test_slot_budget_respected(self):
        market = TrainingMarket(_cfg(max_tasks_per_block=2))
        for i in range(5):
            market.submit_task(
                _task(model_id=f"m{i}", tip=float(i), task_id=f"t{i}"),
                current_block=0,
            )
        slots = market.select_tasks_for_block(0, VRF_SEED)
        assert len(slots) == 2

    def test_empty_queue_returns_empty(self):
        market = TrainingMarket(_cfg())
        slots = market.select_tasks_for_block(0, VRF_SEED)
        assert slots == []

    def test_one_slot_per_model_per_block(self):
        """Two tasks for the same model → only the higher-tip one is taken."""
        market = TrainingMarket(_cfg(max_tasks_per_block=4))
        t1 = _task(model_id="same", tip=2.0, task_id="t1")
        t2 = _task(model_id="same", tip=1.0, task_id="t2")
        market.submit_task(t1, current_block=0)
        market.submit_task(t2, current_block=0)
        slots = market.select_tasks_for_block(0, VRF_SEED)
        assert len(slots) == 1
        assert slots[0].task_id == "t1"

    def test_vrf_tiebreak_is_deterministic(self):
        market = TrainingMarket(_cfg(max_tasks_per_block=1))
        t1 = _task(model_id="m1", tip=1.0, task_id="aaa")
        t2 = _task(model_id="m2", tip=1.0, task_id="bbb")
        market.submit_task(t1, current_block=0)
        market.submit_task(t2, current_block=0)
        slots_a = market.select_tasks_for_block(0, VRF_SEED)
        slots_b = market.select_tasks_for_block(0, VRF_SEED)
        assert slots_a[0].task_id == slots_b[0].task_id

    def test_different_vrf_seed_may_change_tiebreak(self):
        """With equal tips, different VRF seeds should (statistically) differ."""
        results = set()
        for seed_byte in range(20):
            market = TrainingMarket(_cfg(max_tasks_per_block=1))
            t1 = _task(model_id="m1", tip=1.0, task_id="aaa")
            t2 = _task(model_id="m2", tip=1.0, task_id="bbb")
            market.submit_task(t1, current_block=0)
            market.submit_task(t2, current_block=0)
            seed = bytes([seed_byte] * 32)
            slots = market.select_tasks_for_block(0, seed)
            results.add(slots[0].task_id)
        # Over 20 different seeds we must see both tasks selected at least once
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_model_in_cooldown_not_selectable(self):
        market = TrainingMarket(_cfg(default_cooldown_blocks=5))
        t = _task(model_id="m1", task_id="t1")
        market.submit_task(t, current_block=0)
        slots = market.select_tasks_for_block(0, VRF_SEED)
        assert len(slots) == 1
        market.finalize_block(slots, current_block=0)
        # Re-queue same model immediately
        t2 = _task(model_id="m1", task_id="t2")
        ok, msg = market.submit_task(t2, current_block=1)
        assert not ok
        assert "cooldown" in msg

    def test_model_selectable_after_cooldown(self):
        market = TrainingMarket(_cfg(default_cooldown_blocks=3))
        t = _task(model_id="m1", task_id="t1")
        market.submit_task(t, current_block=0)
        slots = market.select_tasks_for_block(0, VRF_SEED)
        market.finalize_block(slots, current_block=0)
        # cooldown ends at block 3; block 4 is safe
        t2 = _task(model_id="m1", task_id="t2")
        ok, msg = market.submit_task(t2, current_block=4)
        assert ok, msg

    def test_cooldown_remaining_helper(self):
        market = TrainingMarket(_cfg(default_cooldown_blocks=10))
        t = _task(model_id="m1", task_id="t1")
        market.submit_task(t, current_block=0)
        slots = market.select_tasks_for_block(0, VRF_SEED)
        market.finalize_block(slots, current_block=0)
        assert market.model_cooldown_remaining("m1", current_block=5) == 5
        assert market.model_cooldown_remaining("m1", current_block=10) == 0


# ---------------------------------------------------------------------------
# Base-fee adjustment
# ---------------------------------------------------------------------------

class TestBaseFee:
    def test_base_fee_rises_when_full(self):
        market = TrainingMarket(_cfg(
            max_tasks_per_block=2,
            initial_base_fee=1.0,
            base_fee_change_rate=0.125,
            target_utilization=0.5,
        ))
        before = market.current_base_fee
        # Simulate full block (2/2 = 100% utilization, target 50%)
        market._adjust_base_fee(slots_used=2, max_slots=2)
        assert market.current_base_fee > before

    def test_base_fee_falls_when_empty(self):
        market = TrainingMarket(_cfg(
            max_tasks_per_block=2,
            initial_base_fee=1.0,
            base_fee_change_rate=0.125,
            target_utilization=0.5,
        ))
        before = market.current_base_fee
        market._adjust_base_fee(slots_used=0, max_slots=2)
        assert market.current_base_fee < before

    def test_base_fee_floor(self):
        market = TrainingMarket(_cfg(initial_base_fee=0.001, min_base_fee=0.001))
        for _ in range(50):
            market._adjust_base_fee(slots_used=0, max_slots=4)
        assert market.current_base_fee >= 0.001

    def test_base_fee_ceiling(self):
        market = TrainingMarket(_cfg(initial_base_fee=999.0, max_base_fee=1000.0))
        for _ in range(50):
            market._adjust_base_fee(slots_used=4, max_slots=4)
        assert market.current_base_fee <= 1000.0

    def test_base_fee_stable_at_target(self):
        """At exactly 50% utilization the fee should barely move."""
        market = TrainingMarket(_cfg(
            max_tasks_per_block=4,
            initial_base_fee=1.0,
            target_utilization=0.5,
        ))
        before = market.current_base_fee
        market._adjust_base_fee(slots_used=2, max_slots=4)
        # deviation = 0; change_factor = 0 → fee unchanged
        assert market.current_base_fee == pytest.approx(before, rel=1e-6)


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------

class TestExpiry:
    def test_expired_tasks_removed(self):
        market = TrainingMarket(_cfg())
        t = _task(task_id="exp", expiry_block=5)
        market.submit_task(t, current_block=0)
        expired = market.expire_stale_tasks(current_block=5)
        assert len(expired) == 1
        assert market.pending_count == 0

    def test_non_expired_tasks_kept(self):
        market = TrainingMarket(_cfg())
        t = _task(task_id="alive", expiry_block=10)
        market.submit_task(t, current_block=0)
        expired = market.expire_stale_tasks(current_block=4)
        assert len(expired) == 0
        assert market.pending_count == 1

    def test_expired_tasks_not_selected(self):
        market = TrainingMarket(_cfg())
        t = _task(task_id="exp2", expiry_block=3)
        market.submit_task(t, current_block=0)
        # at block 4 the task has expired (4 >= 3), so it shouldn't be selected
        slots = market.select_tasks_for_block(current_block=4, vrf_seed=VRF_SEED)
        assert len(slots) == 0


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

class TestCancellation:
    def test_owner_can_cancel(self):
        market = TrainingMarket(_cfg())
        t = _task(owner="alice", task_id="del_me")
        market.submit_task(t, current_block=0)
        ok, msg = market.cancel_task("del_me", owner="alice")
        assert ok
        assert market.pending_count == 0

    def test_non_owner_cannot_cancel(self):
        market = TrainingMarket(_cfg())
        t = _task(owner="alice", task_id="guarded")
        market.submit_task(t, current_block=0)
        ok, msg = market.cancel_task("guarded", owner="bob")
        assert not ok
        assert "owner" in msg

    def test_cancel_nonexistent_task(self):
        market = TrainingMarket(_cfg())
        ok, msg = market.cancel_task("ghost_id", owner="alice")
        assert not ok
        assert "not found" in msg


# ---------------------------------------------------------------------------
# Finalize + urgency interaction over multiple blocks
# ---------------------------------------------------------------------------

class TestMultiBlock:
    def test_waiting_task_promoted_by_urgency(self):
        """A low-tip task that has waited many blocks beats a fresh high-tip task."""
        market = TrainingMarket(_cfg(
            max_tasks_per_block=1,
            urgency_rate=0.5,   # aggressive urgency
            urgency_cap=10.0,
        ))
        old_task = _task(model_id="old", tip=1.0, submitted_block=0, task_id="old_t")
        new_task = _task(model_id="new", tip=3.0, submitted_block=8, task_id="new_t")
        market.submit_task(old_task, current_block=0)
        market.submit_task(new_task, current_block=8)
        # At block 8: old_task urgency = 1 + 8*0.5 = 5.0 → eff_tip = 5.0
        # new_task urgency = 1 + 0*0.5 = 1.0 → eff_tip = 3.0
        slots = market.select_tasks_for_block(current_block=8, vrf_seed=VRF_SEED)
        assert slots[0].task_id == "old_t"

    def test_finalize_removes_scheduled_tasks(self):
        market = TrainingMarket(_cfg(max_tasks_per_block=2))
        t1 = _task(model_id="m1", task_id="t1")
        t2 = _task(model_id="m2", task_id="t2")
        market.submit_task(t1, current_block=0)
        market.submit_task(t2, current_block=0)
        slots = market.select_tasks_for_block(0, VRF_SEED)
        market.finalize_block(slots, current_block=0)
        assert market.pending_count == 0


# ---------------------------------------------------------------------------
# State snapshot
# ---------------------------------------------------------------------------

class TestStateSnapshot:
    def test_snapshot_reflects_state(self):
        market = TrainingMarket(_cfg(max_tasks_per_block=2))
        t = _task(task_id="snap_t")
        market.submit_task(t, current_block=5)
        snap = market.state_snapshot(current_block=5)
        assert snap.pending_count == 1
        assert snap.base_fee == pytest.approx(market.current_base_fee)


# ---------------------------------------------------------------------------
# Transaction round-trip
# ---------------------------------------------------------------------------

class TestTransactions:
    def test_post_training_task_transaction_fields(self):
        tx = PostTrainingTaskTransaction(
            sender="alice",
            fee=0.001,
            timestamp=1.0,
            public_key="pk_alice",
            signature="sig",
            tx_id="txid_001",
            model_id="model_A",
            deposit=1.01,
            base_fee=0.01,
            tip=1.0,
            cooldown_blocks=5,
            max_slots=1,
            expiry_block=200,
        )
        data = tx.get_transaction_data()
        assert data["type"] == "post_training_task"
        assert data["model_id"] == "model_A"
        assert data["tip"] == 1.0
        assert data["cooldown_blocks"] == 5

    def test_cancel_training_task_transaction_fields(self):
        tx = CancelTrainingTaskTransaction(
            sender="alice",
            fee=0.001,
            timestamp=2.0,
            public_key="pk_alice",
            signature="sig",
            tx_id="txid_002",
            task_id="some_task_id",
        )
        data = tx.get_transaction_data()
        assert data["type"] == "cancel_training_task"
        assert data["task_id"] == "some_task_id"

    def test_post_tx_registered_in_factory(self):
        from dessin.consensus.transactions import TransactionFactory
        assert "post_training_task" in TransactionFactory.TRANSACTION_TYPES
        assert "cancel_training_task" in TransactionFactory.TRANSACTION_TYPES
