"""
Tests for StorageLeaseManager — creation, renewal, expiry, deletion, cancellation,
grace period, and event audit trail.
"""

import pytest
from dessin.economics.storage_lease import (
    StorageLeaseManager,
    StorageLeaseConfig,
    LeaseStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mgr(grace: int = 10, warning: int = 5) -> StorageLeaseManager:
    return StorageLeaseManager(StorageLeaseConfig(
        grace_period_blocks=grace,
        warning_blocks=warning,
    ))


def _create(mgr: StorageLeaseManager, file_id: str = "f1", current_block: int = 0,
            initial_blocks: int = 100, size_mb: float = 64.0, owner: str = "alice") -> None:
    ok, msg = mgr.create_lease(
        file_id=file_id,
        owner=owner,
        size_mb=size_mb,
        initial_blocks=initial_blocks,
        payment=1.0,
        current_block=current_block,
    )
    assert ok, msg


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

class TestCreate:
    def test_creates_active_lease(self):
        mgr = _mgr()
        _create(mgr)
        lease = mgr.get_lease("f1")
        assert lease is not None
        assert lease.status == LeaseStatus.ACTIVE

    def test_paid_through_block_set_correctly(self):
        mgr = _mgr()
        _create(mgr, current_block=50, initial_blocks=100)
        assert mgr.get_lease("f1").paid_through_block == 150

    def test_duplicate_rejected(self):
        mgr = _mgr()
        _create(mgr)
        ok, msg = mgr.create_lease(
            file_id="f1", owner="alice", size_mb=64,
            initial_blocks=50, payment=1.0, current_block=0
        )
        assert not ok
        assert "already active" in msg

    def test_zero_initial_blocks_rejected(self):
        mgr = StorageLeaseManager(StorageLeaseConfig(min_renewal_blocks=1))
        ok, msg = mgr.create_lease(
            file_id="f0", owner="alice", size_mb=10,
            initial_blocks=0, payment=0.1, current_block=0,
        )
        assert not ok


# ---------------------------------------------------------------------------
# Renewal
# ---------------------------------------------------------------------------

class TestRenewal:
    def test_renewal_extends_lease(self):
        mgr = _mgr()
        _create(mgr, initial_blocks=100)
        ok, msg = mgr.renew_lease(
            file_id="f1", payer="alice", additional_blocks=50,
            payment=0.5, current_block=0
        )
        assert ok
        assert mgr.get_lease("f1").paid_through_block == 150

    def test_renewal_by_non_owner_allowed(self):
        mgr = _mgr()
        _create(mgr, owner="alice")
        ok, msg = mgr.renew_lease(
            file_id="f1", payer="community_dao", additional_blocks=30,
            payment=0.3, current_block=0
        )
        assert ok

    def test_renewal_tracks_total_paid(self):
        mgr = _mgr()
        _create(mgr, current_block=0)
        mgr.renew_lease(file_id="f1", payer="alice", additional_blocks=50,
                        payment=2.0, current_block=10)
        assert mgr.get_lease("f1").total_paid == pytest.approx(3.0)

    def test_renewal_increments_count(self):
        mgr = _mgr()
        _create(mgr)
        mgr.renew_lease(file_id="f1", payer="alice", additional_blocks=10, payment=0.1, current_block=0)
        mgr.renew_lease(file_id="f1", payer="alice", additional_blocks=10, payment=0.1, current_block=0)
        assert mgr.get_lease("f1").renewal_count == 2

    def test_renewal_nonexistent_fails(self):
        mgr = _mgr()
        ok, msg = mgr.renew_lease(
            file_id="ghost", payer="alice", additional_blocks=10,
            payment=0.1, current_block=0,
        )
        assert not ok

    def test_renewal_exceeds_max_fails(self):
        mgr = StorageLeaseManager(StorageLeaseConfig(max_renewal_blocks=100))
        _create(mgr)
        ok, msg = mgr.renew_lease(
            file_id="f1", payer="alice", additional_blocks=101,
            payment=1.0, current_block=0,
        )
        assert not ok
        assert "max" in msg


# ---------------------------------------------------------------------------
# State machine (warning → expired → deletion_eligible)
# ---------------------------------------------------------------------------

class TestStateMachine:
    def test_warning_state_triggered(self):
        """Lease paid_through=100, warning_blocks=5 → warning at block 95."""
        mgr = _mgr(grace=10, warning=5)
        _create(mgr, initial_blocks=100, current_block=0)
        # Block 95: inside warning window (95 = 100 - 5)
        mgr.check_block(95)
        assert mgr.get_lease("f1").status == LeaseStatus.WARNING

    def test_expired_state_triggered(self):
        mgr = _mgr(grace=10, warning=5)
        _create(mgr, initial_blocks=10, current_block=0)
        expired, _ = mgr.check_block(10)
        assert "f1" in expired
        assert mgr.get_lease("f1").status == LeaseStatus.EXPIRED

    def test_deletion_eligible_after_grace(self):
        mgr = _mgr(grace=5, warning=2)
        _create(mgr, initial_blocks=10, current_block=0)
        mgr.check_block(10)   # expires
        _, deletion_ready = mgr.check_block(15)   # 10 + 5 grace
        assert "f1" in deletion_ready
        assert mgr.get_lease("f1").status == LeaseStatus.DELETION_ELIGIBLE

    def test_mark_deleted(self):
        mgr = _mgr(grace=5, warning=2)
        _create(mgr, initial_blocks=10, current_block=0)
        mgr.check_block(10)
        mgr.check_block(15)
        mgr.mark_deleted("f1", current_block=15)
        assert mgr.get_lease("f1").status == LeaseStatus.DELETED

    def test_renewal_during_grace_reactivates(self):
        mgr = _mgr(grace=10, warning=5)
        _create(mgr, initial_blocks=10, current_block=0)
        mgr.check_block(10)   # → EXPIRED
        ok, _ = mgr.renew_lease(
            file_id="f1", payer="alice", additional_blocks=100,
            payment=1.0, current_block=12,
        )
        assert ok
        assert mgr.get_lease("f1").status == LeaseStatus.ACTIVE

    def test_renewal_after_deletion_fails(self):
        mgr = _mgr(grace=5, warning=2)
        _create(mgr, initial_blocks=10, current_block=0)
        mgr.check_block(10)
        mgr.check_block(15)
        mgr.mark_deleted("f1", current_block=15)
        ok, msg = mgr.renew_lease(
            file_id="f1", payer="alice", additional_blocks=50,
            payment=1.0, current_block=16
        )
        assert not ok

    def test_no_premature_warning(self):
        """Before the warning window, lease should stay ACTIVE."""
        mgr = _mgr(grace=10, warning=10)
        _create(mgr, initial_blocks=100, current_block=0)
        mgr.check_block(80)   # 100 - 80 = 20 blocks left, window is 10
        assert mgr.get_lease("f1").status == LeaseStatus.ACTIVE


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

class TestCancellation:
    def test_owner_can_cancel(self):
        mgr = _mgr()
        _create(mgr, owner="alice")
        ok, msg = mgr.cancel_lease(file_id="f1", owner="alice", current_block=5)
        assert ok
        assert mgr.get_lease("f1").status == LeaseStatus.CANCELLED

    def test_non_owner_cannot_cancel(self):
        mgr = _mgr()
        _create(mgr, owner="alice")
        ok, msg = mgr.cancel_lease(file_id="f1", owner="bob", current_block=5)
        assert not ok

    def test_cancelled_lease_renewal_rejected(self):
        mgr = _mgr()
        _create(mgr, owner="alice")
        mgr.cancel_lease(file_id="f1", owner="alice", current_block=0)
        ok, _ = mgr.renew_lease(
            file_id="f1", payer="alice", additional_blocks=50,
            payment=1.0, current_block=1
        )
        assert not ok


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

class TestQueries:
    def test_active_leases(self):
        mgr = _mgr()
        _create(mgr, file_id="f1")
        _create(mgr, file_id="f2")
        assert len(mgr.active_leases()) == 2

    def test_expiring_soon(self):
        mgr = _mgr(grace=10, warning=5)
        _create(mgr, file_id="f1", initial_blocks=20, current_block=0)
        _create(mgr, file_id="f2", initial_blocks=200, current_block=0)
        # At block 5, f1 expires in 15 blocks (not soon), f2 in 195 (not soon)
        soon = mgr.expiring_soon(current_block=5, within_blocks=20)
        assert any(l.file_id == "f1" for l in soon)
        assert not any(l.file_id == "f2" for l in soon)

    def test_cooldown_remaining_helper(self):
        mgr = _mgr()
        _create(mgr, initial_blocks=50)
        lease = mgr.get_lease("f1")
        assert lease.paid_through_block - 10 == 40


# ---------------------------------------------------------------------------
# Audit events
# ---------------------------------------------------------------------------

class TestAuditEvents:
    def test_create_emits_event(self):
        mgr = _mgr()
        _create(mgr)
        events = mgr.events_for("f1")
        assert any(e.kind == "created" for e in events)

    def test_renewal_emits_event(self):
        mgr = _mgr()
        _create(mgr)
        mgr.renew_lease(file_id="f1", payer="alice", additional_blocks=20,
                        payment=0.2, current_block=1)
        events = mgr.events_for("f1")
        assert any(e.kind == "renewed" for e in events)

    def test_expiry_emits_event(self):
        mgr = _mgr(grace=10, warning=5)
        _create(mgr, initial_blocks=5, current_block=0)
        mgr.check_block(5)
        events = mgr.events_for("f1")
        assert any(e.kind == "expired" for e in events)

    def test_deletion_eligible_emits_event(self):
        mgr = _mgr(grace=5, warning=2)
        _create(mgr, initial_blocks=5, current_block=0)
        mgr.check_block(5)
        mgr.check_block(10)
        events = mgr.events_for("f1")
        assert any(e.kind == "deletion_eligible" for e in events)
