"""
Storage Lease System
====================

A **storage lease** is a paid commitment that tells every seeding node:
*"keep this data available until at least block N."*

If the owner does not renew (or top up) before expiry, nodes are permitted to
delete the data after a configurable grace period.  This mirrors decentralised
storage networks (Filecoin, Storj, Arweave) while staying on-chain.

Lifecycle
---------
::

    [upload]  →  lease created (paid_through_block = current + initial_blocks)
    [renewal] →  paid_through_block extended  (RenewStorageLeaseTransaction)
    [warning] →  emitted when current_block > paid_through_block − warning_blocks
    [expired] →  current_block >= paid_through_block
    [deleted] →  current_block >= paid_through_block + grace_period_blocks
                 → node may discard data; ``deletion_eligible`` returns True

Design notes
------------
- One ``StorageLease`` per file (identified by ``file_id``).
- Renewal extends ``paid_through_block``; it never resets it.
- Top-ups from non-owners are accepted (``payer`` != ``owner``) so communities
  can collectively fund model data they care about.
- The :class:`StorageLeaseManager` is *stateful but deterministic*: two nodes
  running the same block history reach identical state.
- ``check_block`` drives the state machine; call it once per block in the
  consensus loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lease status
# ---------------------------------------------------------------------------

class LeaseStatus(str, Enum):
    ACTIVE = "active"              # paid and within expiry
    WARNING = "warning"            # within warning window before expiry
    EXPIRED = "expired"            # past paid_through_block, within grace
    DELETION_ELIGIBLE = "deletion_eligible"  # past grace period → safe to delete
    DELETED = "deleted"            # node has deleted the data
    CANCELLED = "cancelled"        # owner voluntarily cancelled


# ---------------------------------------------------------------------------
# Lease record
# ---------------------------------------------------------------------------

@dataclass
class StorageLease:
    """
    On-chain record of a storage commitment.

    Fields
    ------
    file_id:
        Unique identifier matching ``UploadTrainingDataTransaction.file_id``.
    owner:
        Address that originally uploaded the data.
    size_mb:
        File size at upload time (immutable; drives renewal pricing).
    paid_through_block:
        Data is guaranteed available until this block.
    grace_period_blocks:
        Blocks after expiry before deletion is permitted.
    warning_blocks:
        Issue a warning this many blocks before expiry.
    status:
        Current lifecycle state (see :class:`LeaseStatus`).
    total_paid:
        Cumulative DESSIN paid across all renewals.
    renewal_count:
        How many times the lease has been extended.
    """

    file_id: str
    owner: str
    size_mb: float
    paid_through_block: int
    grace_period_blocks: int = 50
    warning_blocks: int = 10
    status: LeaseStatus = LeaseStatus.ACTIVE
    total_paid: float = 0.0
    renewal_count: int = 0

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def deletion_eligible_at(self) -> int:
        """Block height at which a node may delete the data."""
        return self.paid_through_block + self.grace_period_blocks

    def warning_at(self) -> int:
        """Block height at which a warning should be emitted."""
        return self.paid_through_block - self.warning_blocks

    def is_deletable(self, current_block: int) -> bool:
        return current_block >= self.deletion_eligible_at()

    def is_expired(self, current_block: int) -> bool:
        return current_block >= self.paid_through_block

    def is_in_warning(self, current_block: int) -> bool:
        return (
            self.warning_at() <= current_block < self.paid_through_block
        )

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "StorageLease":
        d = dict(d)
        d["status"] = LeaseStatus(d.get("status", LeaseStatus.ACTIVE.value))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Lease events (for audit / economic ledger)
# ---------------------------------------------------------------------------

@dataclass
class LeaseEvent:
    """Immutable audit record emitted by :class:`StorageLeaseManager`."""

    file_id: str
    block: int
    kind: str          # "created", "renewed", "warning", "expired", "deleted", "cancelled"
    payer: str = ""
    amount: float = 0.0
    new_paid_through: int = 0
    note: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

@dataclass
class StorageLeaseConfig:
    """Tuneable parameters for the lease subsystem."""

    grace_period_blocks: int = 50       # after expiry, before deletion
    warning_blocks: int = 10            # blocks before expiry to warn
    min_renewal_blocks: int = 1         # smallest valid extension
    max_renewal_blocks: int = 100_000   # anti-griefing ceiling


class StorageLeaseManager:
    """
    Tracks storage leases and drives expiry / deletion state transitions.

    Usage in consensus loop
    -----------------------
    ::

        lease_mgr = StorageLeaseManager()

        # --- on UploadTrainingDataTransaction ---
        lease_mgr.create_lease(
            file_id=tx.file_id,
            owner=tx.sender,
            size_mb=tx.size_mb,
            initial_blocks=tx.storage_blocks,
            payment=tx.storage_payment,
            current_block=block_height,
        )

        # --- on RenewStorageLeaseTransaction ---
        lease_mgr.renew_lease(
            file_id=tx.file_id,
            payer=tx.sender,
            additional_blocks=tx.additional_blocks,
            payment=tx.payment,
            current_block=block_height,
        )

        # --- once per block (in block processing) ---
        expired, deletion_ready = lease_mgr.check_block(current_block)
        for file_id in deletion_ready:
            node_storage.delete(file_id)   # application code
    """

    def __init__(self, config: Optional[StorageLeaseConfig] = None) -> None:
        self.config = config or StorageLeaseConfig()
        self._leases: Dict[str, StorageLease] = {}
        self._events: List[LeaseEvent] = []

    # ------------------------------------------------------------------
    # Lease creation
    # ------------------------------------------------------------------

    def create_lease(
        self,
        *,
        file_id: str,
        owner: str,
        size_mb: float,
        initial_blocks: int,
        payment: float,
        current_block: int,
    ) -> Tuple[bool, str]:
        """
        Register a new lease.  Returns ``(ok, message)``.

        Fails if a non-deleted lease already exists for ``file_id`` (use
        :meth:`renew_lease` to extend an existing one).
        """
        if file_id in self._leases:
            existing = self._leases[file_id]
            if existing.status not in (LeaseStatus.DELETED, LeaseStatus.CANCELLED):
                return False, f"lease for {file_id!r} already active"

        if initial_blocks < self.config.min_renewal_blocks:
            return False, (
                f"initial_blocks {initial_blocks} < min {self.config.min_renewal_blocks}"
            )

        lease = StorageLease(
            file_id=file_id,
            owner=owner,
            size_mb=size_mb,
            paid_through_block=current_block + initial_blocks,
            grace_period_blocks=self.config.grace_period_blocks,
            warning_blocks=self.config.warning_blocks,
            status=LeaseStatus.ACTIVE,
            total_paid=payment,
            renewal_count=0,
        )
        self._leases[file_id] = lease
        self._emit(LeaseEvent(
            file_id=file_id,
            block=current_block,
            kind="created",
            payer=owner,
            amount=payment,
            new_paid_through=lease.paid_through_block,
        ))
        _log.debug(
            "StorageLease: created %s owner=%s paid_through=%d",
            file_id, owner, lease.paid_through_block,
        )
        return True, "created"

    # ------------------------------------------------------------------
    # Renewal
    # ------------------------------------------------------------------

    def renew_lease(
        self,
        *,
        file_id: str,
        payer: str,
        additional_blocks: int,
        payment: float,
        current_block: int,
    ) -> Tuple[bool, str]:
        """
        Extend an existing lease.

        Anyone can pay (``payer`` need not be the owner).  Renewal is
        rejected if the lease is already in ``DELETED`` or ``CANCELLED``
        state.
        """
        lease = self._leases.get(file_id)
        if lease is None:
            return False, "lease not found"
        if lease.status in (LeaseStatus.DELETED, LeaseStatus.CANCELLED):
            return False, f"lease is {lease.status.value}; cannot renew"
        if additional_blocks < self.config.min_renewal_blocks:
            return False, (
                f"additional_blocks {additional_blocks} < min "
                f"{self.config.min_renewal_blocks}"
            )
        if additional_blocks > self.config.max_renewal_blocks:
            return False, (
                f"additional_blocks {additional_blocks} > max "
                f"{self.config.max_renewal_blocks}"
            )

        lease.paid_through_block += additional_blocks
        lease.total_paid += payment
        lease.renewal_count += 1
        # Re-activate if it was in a warning/expired state
        if lease.status in (LeaseStatus.WARNING, LeaseStatus.EXPIRED,
                             LeaseStatus.DELETION_ELIGIBLE):
            lease.status = LeaseStatus.ACTIVE

        self._emit(LeaseEvent(
            file_id=file_id,
            block=current_block,
            kind="renewed",
            payer=payer,
            amount=payment,
            new_paid_through=lease.paid_through_block,
        ))
        _log.debug(
            "StorageLease: renewed %s +%d blocks, now paid_through=%d",
            file_id, additional_blocks, lease.paid_through_block,
        )
        return True, "renewed"

    # ------------------------------------------------------------------
    # Owner cancellation
    # ------------------------------------------------------------------

    def cancel_lease(self, *, file_id: str, owner: str, current_block: int) -> Tuple[bool, str]:
        """Owner may cancel; refund logic lives in the economic system."""
        lease = self._leases.get(file_id)
        if lease is None:
            return False, "lease not found"
        if lease.owner != owner:
            return False, "not the lease owner"
        if lease.status in (LeaseStatus.DELETED, LeaseStatus.CANCELLED):
            return False, "already cancelled/deleted"
        lease.status = LeaseStatus.CANCELLED
        self._emit(LeaseEvent(file_id=file_id, block=current_block, kind="cancelled",
                              payer=owner))
        return True, "cancelled"

    # ------------------------------------------------------------------
    # Block tick – state machine driver
    # ------------------------------------------------------------------

    def check_block(
        self, current_block: int
    ) -> Tuple[List[str], List[str]]:
        """
        Advance the state machine for every lease.

        Must be called exactly once per block (idempotent within a block).

        Returns
        -------
        expired : list[str]
            ``file_id`` values that just entered ``EXPIRED`` state this block.
        deletion_ready : list[str]
            ``file_id`` values that just entered ``DELETION_ELIGIBLE`` —
            the node application should delete these files.
        """
        just_expired: List[str] = []
        just_deletion_ready: List[str] = []

        for file_id, lease in list(self._leases.items()):
            if lease.status in (LeaseStatus.DELETED, LeaseStatus.CANCELLED):
                continue

            # Warning window
            if lease.status == LeaseStatus.ACTIVE and lease.is_in_warning(current_block):
                lease.status = LeaseStatus.WARNING
                _log.warning(
                    "StorageLease WARNING: %s expires at block %d (now %d)",
                    file_id, lease.paid_through_block, current_block,
                )
                self._emit(LeaseEvent(
                    file_id=file_id, block=current_block, kind="warning",
                    note=f"expires at {lease.paid_through_block}",
                ))

            # Expiry
            if lease.status in (LeaseStatus.ACTIVE, LeaseStatus.WARNING):
                if lease.is_expired(current_block):
                    lease.status = LeaseStatus.EXPIRED
                    just_expired.append(file_id)
                    _log.warning(
                        "StorageLease EXPIRED: %s at block %d (grace until %d)",
                        file_id, current_block, lease.deletion_eligible_at(),
                    )
                    self._emit(LeaseEvent(
                        file_id=file_id, block=current_block, kind="expired",
                        note=f"grace period ends at {lease.deletion_eligible_at()}",
                    ))

            # Deletion eligible
            if lease.status == LeaseStatus.EXPIRED:
                if lease.is_deletable(current_block):
                    lease.status = LeaseStatus.DELETION_ELIGIBLE
                    just_deletion_ready.append(file_id)
                    _log.warning(
                        "StorageLease DELETION_ELIGIBLE: %s (block %d)",
                        file_id, current_block,
                    )
                    self._emit(LeaseEvent(
                        file_id=file_id, block=current_block, kind="deletion_eligible",
                    ))

        return just_expired, just_deletion_ready

    def mark_deleted(self, file_id: str, current_block: int) -> None:
        """
        Called by the node after it has physically deleted the data.
        Transitions lease to ``DELETED``.
        """
        lease = self._leases.get(file_id)
        if lease is None:
            return
        lease.status = LeaseStatus.DELETED
        self._emit(LeaseEvent(file_id=file_id, block=current_block, kind="deleted"))
        _log.info("StorageLease DELETED: %s", file_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_lease(self, file_id: str) -> Optional[StorageLease]:
        return self._leases.get(file_id)

    def active_leases(self) -> List[StorageLease]:
        return [
            l for l in self._leases.values()
            if l.status in (LeaseStatus.ACTIVE, LeaseStatus.WARNING)
        ]

    def expiring_soon(self, current_block: int, within_blocks: int = 20) -> List[StorageLease]:
        """Return leases expiring within the next ``within_blocks`` blocks."""
        return [
            l for l in self._leases.values()
            if l.status in (LeaseStatus.ACTIVE, LeaseStatus.WARNING)
            and 0 <= l.paid_through_block - current_block <= within_blocks
        ]

    def events_for(self, file_id: str) -> List[LeaseEvent]:
        return [e for e in self._events if e.file_id == file_id]

    def all_events(self) -> List[LeaseEvent]:
        return list(self._events)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, event: LeaseEvent) -> None:
        self._events.append(event)
