"""
Compute Fee Schedule
====================

Heuristic fee pricing for model training and data upload on DeSSIN.

Design principles
-----------------
1. **Per-MB base rate** – the dominant driver is the weight-tensor footprint
   that every seeding node must store and every verifier must checksum.
2. **Training-phase multiplier** – different training recipes have very
   different compute costs per MB.  Full fine-tuning touches every parameter;
   LoRA/QLoRA trains only adapter ranks; pretraining runs many more forward
   passes per epoch than SFT.
3. **Quantisation discount** – quantised models are cheaper to run, so nodes
   charge less per MB for INT4/INT8 workloads.
4. **Step multiplier** – long jobs (many gradient steps) cost more.  Base
   rates assume a "standard" 1 000-step run; steps are billed linearly.
5. **Upload fee** – uploading training data costs a flat per-MB fee that
   covers initial storage rental.

All rates are expressed in DESSIN per megabyte (or per megabyte per step) so
they scale naturally as the ecosystem grows.  Governance can update the rate
table by posting an on-chain parameter change transaction.

Usage
-----
::

    schedule = ComputeFeeSchedule()
    fee = schedule.estimate_training_fee(
        size_mb=4096,           # model weight size
        phase=TrainingPhase.LORA,
        steps=2000,
        is_quantized=True,
    )
    upload_cost = schedule.estimate_upload_fee(size_mb=512)   # dataset
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


STANDARD_STEPS = 1_000   # reference step count for base rates


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

class TrainingPhase(str, Enum):
    """
    What kind of work the miners will actually do.

    Rates (relative to FULL_FINETUNE = 1.0):

    ==============================  ==========  ==============================
    Phase                           Multiplier  Notes
    ==============================  ==========  ==============================
    PRETRAINING                     3.0×        Many epochs, full forward pass
    FULL_FINETUNE                   1.0×        Baseline; all params updated
    SFT (supervised fine-tune)      0.8×        Shorter dataset, fewer steps
    DPO                             1.2×        Preference pairs, dual forward
    RLHF                            2.0×        Reward model + PPO loop
    LORA                            0.35×       Only adapter ranks trained
    QLORA                           0.25×       LoRA on quantised backbone
    ADAPTER                         0.30×       Similar adapter-only cost
    PROMPT_TUNING                   0.15×       Fewest params, soft prompts
    PREFIX_TUNING                   0.18×       Slightly more than prompt
    INFERENCE_ONLY                  0.05×       Just model-serving escrow
    ==============================  ==========  ==============================
    """

    PRETRAINING = "pretraining"
    FULL_FINETUNE = "full_finetune"
    SFT = "sft"
    DPO = "dpo"
    RLHF = "rlhf"
    LORA = "lora"
    QLORA = "qlora"
    ADAPTER = "adapter"
    PROMPT_TUNING = "prompt_tuning"
    PREFIX_TUNING = "prefix_tuning"
    INFERENCE_ONLY = "inference_only"


# ---------------------------------------------------------------------------
# Rate table
# ---------------------------------------------------------------------------

#: Base cost in DESSIN per MB of model weights per STANDARD_STEPS (1 000 steps).
#: Governance can replace this entire dict via a parameter-change transaction.
DEFAULT_PHASE_RATE_PER_MB: Dict[str, float] = {
    TrainingPhase.PRETRAINING.value:     0.0030,
    TrainingPhase.FULL_FINETUNE.value:   0.0010,
    TrainingPhase.SFT.value:             0.0008,
    TrainingPhase.DPO.value:             0.0012,
    TrainingPhase.RLHF.value:            0.0020,
    TrainingPhase.LORA.value:            0.00035,
    TrainingPhase.QLORA.value:           0.00025,
    TrainingPhase.ADAPTER.value:         0.00030,
    TrainingPhase.PROMPT_TUNING.value:   0.00015,
    TrainingPhase.PREFIX_TUNING.value:   0.00018,
    TrainingPhase.INFERENCE_ONLY.value:  0.00005,
}

#: Multiplier applied when the model is quantised (< full precision).
#: Quantised inference uses less memory/compute so miners earn less per block.
QUANTISATION_DISCOUNT = 0.7    # 30 % cheaper for INT4/INT8

#: Flat fee per MB of training data uploaded (covers first storage block).
DATA_UPLOAD_RATE_PER_MB = 0.0002   # DESSIN per MB

#: Flat fee per MB per storage block for data at rest after the first block.
DATA_STORAGE_RATE_PER_MB_PER_BLOCK = 0.00005   # DESSIN per MB per block

#: Minimum fee floor so dust submissions don't flood the queue.
MIN_FEE = 0.001


# ---------------------------------------------------------------------------
# Config dataclass (governable)
# ---------------------------------------------------------------------------

@dataclass
class ComputeFeeSchedule:
    """
    Holds the current fee rates and exposes estimation helpers.

    All rates can be updated by governance without changing this class.

    Parameters
    ----------
    phase_rate_per_mb:
        DESSIN cost per MB per :data:`STANDARD_STEPS` steps for each
        :class:`TrainingPhase`.
    quantisation_discount:
        Fractional multiplier applied to quantised models (< 1.0 = cheaper).
    data_upload_rate_per_mb:
        Flat fee per MB of training data on upload.
    data_storage_rate_per_mb_per_block:
        Recurring cost per MB per storage block for data at rest.
    min_fee:
        Absolute fee floor; no estimate will be lower than this.
    """

    phase_rate_per_mb: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PHASE_RATE_PER_MB)
    )
    quantisation_discount: float = QUANTISATION_DISCOUNT
    data_upload_rate_per_mb: float = DATA_UPLOAD_RATE_PER_MB
    data_storage_rate_per_mb_per_block: float = DATA_STORAGE_RATE_PER_MB_PER_BLOCK
    min_fee: float = MIN_FEE

    # ------------------------------------------------------------------
    # Primary estimators
    # ------------------------------------------------------------------

    def estimate_training_fee(
        self,
        *,
        size_mb: float,
        phase: TrainingPhase,
        steps: int = STANDARD_STEPS,
        is_quantized: bool = False,
        tip: float = 0.0,
    ) -> float:
        """
        Estimate the total DESSIN cost for a training task.

        The deposit a user must lock in a ``PostTrainingTaskTransaction``:

            ``deposit = base_fee + tip``

        where ``base_fee`` is the value returned here (before adding any tip).

        Parameters
        ----------
        size_mb:
            Weight footprint of the model in megabytes.
        phase:
            Training recipe (see :class:`TrainingPhase`).
        steps:
            Number of gradient steps to run.  Billed linearly relative to
            :data:`STANDARD_STEPS`.
        is_quantized:
            When ``True``, applies :attr:`quantisation_discount`.
        tip:
            Optional extra included in the returned total (for convenience).
        """
        rate = self.phase_rate_per_mb.get(
            phase.value, self.phase_rate_per_mb[TrainingPhase.FULL_FINETUNE.value]
        )
        step_multiplier = max(1, steps) / STANDARD_STEPS
        base = size_mb * rate * step_multiplier
        if is_quantized:
            base *= self.quantisation_discount
        return max(self.min_fee, base) + tip

    def estimate_upload_fee(self, *, size_mb: float) -> float:
        """
        Flat fee to upload a training-data file.

        Covers the initial storage block; the owner must then pay
        :meth:`estimate_storage_renewal_fee` each renewal cycle.
        """
        return max(self.min_fee, size_mb * self.data_upload_rate_per_mb)

    def estimate_storage_renewal_fee(
        self, *, size_mb: float, blocks: int
    ) -> float:
        """
        Cost to extend a storage lease by ``blocks`` additional blocks.

        If the owner lets this lapse, nodes are permitted to delete the data
        after the grace period (see :class:`~dessin.economics.storage_lease.StorageLeaseManager`).
        """
        return max(
            self.min_fee,
            size_mb * self.data_storage_rate_per_mb_per_block * blocks,
        )

    def estimate_pipeline_fee(
        self,
        *,
        size_mb: float,
        stages: list,           # list of dicts with "method" and "max_steps"
        is_quantized: bool = False,
    ) -> float:
        """
        Sum fees across all stages of a :class:`~dessin.models.model_format_registry.TrainingPipelineSpec`.

        ``stages`` is a list of dicts with at least ``"method"`` and
        ``"max_steps"`` keys (matching :class:`~dessin.models.model_format_registry.PipelineStageSpec`).
        Unknown method strings fall back to ``FULL_FINETUNE`` rate.
        """
        # Map fine_tuning method names → TrainingPhase values
        _method_to_phase: Dict[str, TrainingPhase] = {
            "full": TrainingPhase.FULL_FINETUNE,
            "lora": TrainingPhase.LORA,
            "qlora": TrainingPhase.QLORA,
            "prompt_tuning": TrainingPhase.PROMPT_TUNING,
            "prefix_tuning": TrainingPhase.PREFIX_TUNING,
            "adapter": TrainingPhase.ADAPTER,
            "p_tuning_v2": TrainingPhase.PROMPT_TUNING,
            "ia3": TrainingPhase.ADAPTER,
            "sft": TrainingPhase.SFT,
            "dpo": TrainingPhase.DPO,
            "rlhf": TrainingPhase.RLHF,
            "pretraining": TrainingPhase.PRETRAINING,
        }
        total = 0.0
        for stage in stages:
            method = str(stage.get("method", "full")).lower()
            phase = _method_to_phase.get(method, TrainingPhase.FULL_FINETUNE)
            steps = int(stage.get("max_steps", STANDARD_STEPS))
            total += self.estimate_training_fee(
                size_mb=size_mb,
                phase=phase,
                steps=steps,
                is_quantized=is_quantized,
            )
        return max(self.min_fee, total)

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    def rate_card(self) -> Dict[str, float]:
        """Return per-MB rates for all phases at STANDARD_STEPS (for display)."""
        return {
            phase: self.phase_rate_per_mb.get(phase, 0.0)
            for phase in [p.value for p in TrainingPhase]
        }

    def cheapest_phase(self) -> TrainingPhase:
        mn = min(self.phase_rate_per_mb, key=self.phase_rate_per_mb.get)
        return TrainingPhase(mn)

    def most_expensive_phase(self) -> TrainingPhase:
        mx = max(self.phase_rate_per_mb, key=self.phase_rate_per_mb.get)
        return TrainingPhase(mx)
