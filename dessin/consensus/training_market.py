"""
Training-task fee market for multi-model-per-block scheduling.

Design
------
Model owners post *training tasks* to the mempool by submitting a
``PostTrainingTaskTransaction``.  Each task carries:

  - a **base_fee deposit** covering the minimum protocol gas, and
  - an optional **tip** – extra DESSIN paid on top of base fee, used by the
    miner leader as a priority signal (Ethereum-style).

The block leader then calls :meth:`TrainingMarket.select_tasks_for_block` to
greedy-pack up to ``max_tasks_per_block`` tasks sorted by **effective tip**:

    effective_tip(task) = task.tip × urgency_multiplier(task, current_block)

The **urgency multiplier** grows linearly (capped) with the number of blocks a
task has been waiting, so low-tip tasks cannot be starved indefinitely.

A **VRF seed** derived from the block randomness breaks ties in a
tamper-resistant way: the leader commits to the VRF output *before* knowing
which tasks will be selected, so they cannot cherry-pick after the draw.

The **base fee** auto-adjusts EIP-1559-style every block:
  - If ``slots_used / max_tasks > target_utilization``:  raise by up to 12.5 %
  - Otherwise: lower by up to 12.5 %

**Cooldowns** prevent the same model from consuming a slot every block.
After being trained, a model enters a ``default_cooldown_blocks`` cooldown
window before it can be queued again.
"""

from __future__ import annotations

import hashlib
import logging
import struct
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from ..runtime.config import TrainingMarketConfig

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task data model
# ---------------------------------------------------------------------------

@dataclass
class TrainingTaskSpec:
    """
    A pending training request from a model owner.

    Fields
    ------
    task_id:
        SHA-256 of ``(model_id, owner, submitted_block)`` – stable identifier.
    model_id:
        The catalog model to train during the scheduled block.
    owner:
        Address of the model owner who posted the deposit.
    deposit:
        Total DESSIN escrowed: ``base_fee + tip``.  The protocol burns
        ``base_fee`` and pays ``tip`` to the block leader.
    base_fee:
        Minimum protocol gas the owner is willing to pay.  Must be ≥ the
        market's current ``base_fee`` at submission time.
    tip:
        Extra DESSIN paid to the miner who includes this task.
    cooldown_blocks:
        Minimum gap in blocks before the same model can be re-queued.
        Defaults to ``TrainingMarketConfig.default_cooldown_blocks``.
    max_slots:
        Maximum compute-unit slots this task may consume.  Currently every
        task costs 1 slot; reserved for future heterogeneous slot sizes.
    submitted_block:
        Block height at submission (used to compute urgency).
    expiry_block:
        If the task is not scheduled by this height it is cancelled and the
        deposit refunded minus a small expiry fee.
    """

    task_id: str
    model_id: str
    owner: str
    deposit: float          # base_fee + tip
    base_fee: float
    tip: float
    cooldown_blocks: int
    max_slots: int
    submitted_block: int
    expiry_block: int

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TrainingTaskSpec":
        return cls(**d)

    @staticmethod
    def make_task_id(model_id: str, owner: str, submitted_block: int) -> str:
        raw = f"{model_id}:{owner}:{submitted_block}:{time.time_ns()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Scheduled slot (what goes into a block header)
# ---------------------------------------------------------------------------

@dataclass
class ScheduledTrainingSlot:
    """
    One training slot committed by the block leader.

    Both ``task_id`` and ``model_id`` are stored so verifiers can audit the
    selection without looking up the pending queue.
    """

    task_id: str
    model_id: str
    owner: str
    tip_paid: float       # tip collected by the miner
    base_fee_burned: float

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Market state
# ---------------------------------------------------------------------------

@dataclass
class TrainingMarketState:
    """Serialisable snapshot of the market (for block/state export)."""

    base_fee: float
    pending_count: int
    cooldowns: Dict[str, int]          # model_id → unblock_at_block
    last_block_slots_used: int
    last_block_height: int


# ---------------------------------------------------------------------------
# Core market logic
# ---------------------------------------------------------------------------

class TrainingMarket:
    """
    EIP-1559-style mempool for model training scheduling.

    Lifecycle
    ---------
    1. **Owner** calls :meth:`submit_task` with a signed
       ``PostTrainingTaskTransaction``; the deposit is debited immediately.
    2. **Block leader** calls :meth:`select_tasks_for_block` after constructing
       the block header (has VRF seed); receives an ordered list of
       ``ScheduledTrainingSlot`` objects.
    3. After the block is finalised, **call** :meth:`finalize_block` to remove
       spent tasks, set model cooldowns, and adjust the base fee.
    4. If a task was not selected, it stays in the queue until
       ``expiry_block`` when :meth:`expire_stale_tasks` removes it.
    """

    def __init__(self, config: TrainingMarketConfig) -> None:
        self.config = config
        self.base_fee: float = config.initial_base_fee
        # model_id → expiry block height (while in cooldown)
        self._cooldowns: Dict[str, int] = {}
        # task_id → TrainingTaskSpec
        self._pending: Dict[str, TrainingTaskSpec] = {}
        self._last_block_slots_used: int = 0
        self._last_block_height: int = 0

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit_task(
        self,
        task: TrainingTaskSpec,
        current_block: int,
    ) -> Tuple[bool, str]:
        """
        Add a task to the pending queue.

        Returns ``(accepted, reason)``; ``accepted=False`` if the task does not
        pass validation.
        """
        if task.base_fee < self.base_fee:
            return False, (
                f"base_fee {task.base_fee} < market base_fee {self.base_fee:.6f}"
            )
        if task.tip < self.config.min_tip:
            return False, f"tip {task.tip} < min_tip {self.config.min_tip}"
        if task.deposit < task.base_fee + task.tip:
            return False, "deposit < base_fee + tip"
        if task.task_id in self._pending:
            return False, "duplicate task_id"
        if self._model_in_cooldown(task.model_id, current_block):
            unblock = self._cooldowns[task.model_id]
            return False, (
                f"model {task.model_id!r} in cooldown until block {unblock}"
            )
        self._pending[task.task_id] = task
        _log.debug(
            "TrainingMarket: queued task %s model=%s tip=%.4f eff_tip=%.4f",
            task.task_id, task.model_id, task.tip,
            self._effective_tip(task, current_block),
        )
        return True, "accepted"

    def cancel_task(self, task_id: str, owner: str) -> Tuple[bool, str]:
        """Owner-initiated cancellation; deposit refunded minus base_fee."""
        task = self._pending.get(task_id)
        if task is None:
            return False, "task not found"
        if task.owner != owner:
            return False, "not the task owner"
        del self._pending[task_id]
        return True, "cancelled"

    # ------------------------------------------------------------------
    # Block leader: selection
    # ------------------------------------------------------------------

    def select_tasks_for_block(
        self,
        current_block: int,
        vrf_seed: bytes,
        *,
        max_slots: Optional[int] = None,
    ) -> List[ScheduledTrainingSlot]:
        """
        Return an ordered list of ``ScheduledTrainingSlot`` for the next block.

        Algorithm
        ---------
        1. Filter out expired tasks and models still in cooldown.
        2. Score each task: ``effective_tip = tip × urgency_multiplier``.
        3. Sort descending by score; break ties with a VRF-derived integer so
           the leader cannot bias selection after seeing the draw.
        4. Greedy-pack up to ``max_slots`` (defaults to
           ``config.max_tasks_per_block``).
        """
        if max_slots is None:
            max_slots = self.config.max_tasks_per_block

        eligible = [
            t for t in self._pending.values()
            if not self._model_in_cooldown(t.model_id, current_block)
            and current_block < t.expiry_block
        ]

        if not eligible:
            return []

        def _sort_key(t: TrainingTaskSpec) -> Tuple[float, int]:
            eff = self._effective_tip(t, current_block)
            tiebreak = self._vrf_tiebreak(vrf_seed, t.task_id)
            return (eff, tiebreak)

        eligible.sort(key=_sort_key, reverse=True)

        slots_used = 0
        selected: List[ScheduledTrainingSlot] = []
        seen_models: set = set()  # one slot per model per block
        for task in eligible:
            if slots_used >= max_slots:
                break
            if task.model_id in seen_models:
                continue
            seen_models.add(task.model_id)
            selected.append(
                ScheduledTrainingSlot(
                    task_id=task.task_id,
                    model_id=task.model_id,
                    owner=task.owner,
                    tip_paid=task.tip,
                    base_fee_burned=task.base_fee,
                )
            )
            slots_used += 1

        _log.debug(
            "TrainingMarket: block %d selected %d/%d eligible tasks",
            current_block, len(selected), len(eligible),
        )
        return selected

    # ------------------------------------------------------------------
    # Post-block finalisation
    # ------------------------------------------------------------------

    def finalize_block(
        self,
        scheduled: List[ScheduledTrainingSlot],
        current_block: int,
    ) -> None:
        """
        Apply side-effects after a block is committed:
        - remove scheduled tasks from the queue,
        - set per-model cooldowns,
        - adjust the base fee (EIP-1559).
        """
        for slot in scheduled:
            self._pending.pop(slot.task_id, None)
            cooldown_end = current_block + max(
                1, min(self.config.default_cooldown_blocks,
                       self.config.max_cooldown_blocks)
            )
            self._cooldowns[slot.model_id] = cooldown_end

        self._adjust_base_fee(len(scheduled), self.config.max_tasks_per_block)
        self._last_block_slots_used = len(scheduled)
        self._last_block_height = current_block

    def expire_stale_tasks(self, current_block: int) -> List[TrainingTaskSpec]:
        """Remove and return tasks that have passed their ``expiry_block``."""
        expired = [
            t for t in list(self._pending.values())
            if current_block >= t.expiry_block
        ]
        for t in expired:
            del self._pending[t.task_id]
        if expired:
            _log.debug(
                "TrainingMarket: expired %d stale task(s) at block %d",
                len(expired), current_block,
            )
        return expired

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def urgency_multiplier(self, task: TrainingTaskSpec, current_block: int) -> float:
        """Linear urgency growth, capped at ``config.urgency_cap``."""
        blocks_waiting = max(0, current_block - task.submitted_block)
        return min(
            1.0 + blocks_waiting * self.config.urgency_rate,
            self.config.urgency_cap,
        )

    def effective_tip(self, task: TrainingTaskSpec, current_block: int) -> float:
        """Public alias for tests / analytics."""
        return self._effective_tip(task, current_block)

    @property
    def current_base_fee(self) -> float:
        return self.base_fee

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def pending_tasks(self) -> List[TrainingTaskSpec]:
        return list(self._pending.values())

    def get_task(self, task_id: str) -> Optional[TrainingTaskSpec]:
        return self._pending.get(task_id)

    def model_cooldown_remaining(self, model_id: str, current_block: int) -> int:
        """Blocks until model can be re-queued (0 if not in cooldown)."""
        unblock = self._cooldowns.get(model_id, 0)
        return max(0, unblock - current_block)

    def state_snapshot(self, current_block: int) -> TrainingMarketState:
        return TrainingMarketState(
            base_fee=self.base_fee,
            pending_count=len(self._pending),
            cooldowns=dict(self._cooldowns),
            last_block_slots_used=self._last_block_slots_used,
            last_block_height=self._last_block_height,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _effective_tip(self, task: TrainingTaskSpec, current_block: int) -> float:
        return task.tip * self.urgency_multiplier(task, current_block)

    def _model_in_cooldown(self, model_id: str, current_block: int) -> bool:
        unblock = self._cooldowns.get(model_id, 0)
        return current_block < unblock

    @staticmethod
    def _vrf_tiebreak(vrf_seed: bytes, task_id: str) -> int:
        """Deterministic tie-break integer derived from VRF seed + task_id."""
        h = hashlib.sha256(vrf_seed + task_id.encode()).digest()
        return struct.unpack(">Q", h[:8])[0]

    def _adjust_base_fee(self, slots_used: int, max_slots: int) -> None:
        """
        EIP-1559 base-fee update.

        If the block was fuller than ``target_utilization``, raise the base fee
        up to ``base_fee_change_rate``; if emptier, lower it symmetrically.
        The change is proportional to the deviation from the target so lightly
        under-used blocks do not crash the fee.
        """
        if max_slots == 0:
            return
        utilization = slots_used / max_slots
        target = self.config.target_utilization
        rate = self.config.base_fee_change_rate
        deviation = (utilization - target) / target if target > 0 else 0.0
        # clamp deviation to [-1, 1] then scale by rate
        change_factor = max(-rate, min(rate, deviation * rate))
        new_fee = self.base_fee * (1.0 + change_factor)
        self.base_fee = max(
            self.config.min_base_fee,
            min(self.config.max_base_fee, new_fee),
        )
        _log.debug(
            "TrainingMarket: base_fee %.6f → %.6f (utilization=%.2f)",
            self.base_fee / (1.0 + change_factor), self.base_fee, utilization,
        )
