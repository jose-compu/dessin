"""Consensus subpackage for PoGO protocol implementation."""

import time  # noqa: F401 — tests patch ``dessin.consensus.time`` (monkeypatch)

# Import basic modules first (no heavy deps)
from .transactions import (
    BaseTransaction,
    ModelUploadTransaction,
    ModelQueryTransaction,
    ConditionalTransferTransaction,
    AttestationTransaction,
    PostTrainingTaskTransaction,
    CancelTrainingTaskTransaction,
)
from .training_market import (
    TrainingMarket,
    TrainingTaskSpec,
    ScheduledTrainingSlot,
    TrainingMarketState,
)
from .two_phase_verification import (
    TwoPhaseVerificationSystem,
    VerificationPhase,
    Phase1VerificationData,
    Phase2VerificationData,
    PhaseTiming,
)
from .spot_check_verification import (
    ChainStrikeState,
    ReplayOutcome,
    SpotCheckResult,
    StatisticalThresholds,
    TrainingTrace,
    TrainingTracePrivate,
    accumulate_strikes,
    compute_grad_hash_chain,
    derive_challenge_step_indices,
    verify_spot_replay,
    verify_statistical,
    verify_structural,
)
from .slashing_mechanism import SlashingMechanism, SlashingReason
from .dynamic_block_time import (
    DynamicBlockTimeManager,
    BlockTimeAdjustment,
    BlockTimeAdjustmentProposal,
    VerificationTiming,
)

# Heavy imports wrapped in try/except
try:
    from .consensus import PogoConsensus, PogoBlock, VerificationData, AttestationType
    _CONSENSUS_AVAILABLE = True
except ImportError:
    _CONSENSUS_AVAILABLE = False
    PogoConsensus = None  # type: ignore[misc,assignment]
    PogoBlock = None  # type: ignore[misc,assignment]
    VerificationData = None  # type: ignore[misc,assignment]
    AttestationType = None  # type: ignore[misc,assignment]

try:
    from .enhanced_pogo_consensus import (
        EnhancedPogoConsensus,
        MultiMerkleProof,
        AttestationAggregation,
        AttestationResult,
        NOOP_TRAINING_MODEL_ID,
    )
    _ENHANCED_AVAILABLE = True
except ImportError:
    _ENHANCED_AVAILABLE = False
    EnhancedPogoConsensus = None  # type: ignore[misc,assignment]
    MultiMerkleProof = None  # type: ignore[misc,assignment]
    AttestationAggregation = None  # type: ignore[misc,assignment]
    AttestationResult = None  # type: ignore[misc,assignment]
    NOOP_TRAINING_MODEL_ID = "noop_training"

__all__ = [
    # Transactions
    "BaseTransaction",
    "ModelUploadTransaction",
    "ModelQueryTransaction",
    "ConditionalTransferTransaction",
    "AttestationTransaction",
    "PostTrainingTaskTransaction",
    "CancelTrainingTaskTransaction",
    # Training market
    "TrainingMarket",
    "TrainingTaskSpec",
    "ScheduledTrainingSlot",
    "TrainingMarketState",
    # Two-phase verification
    "TwoPhaseVerificationSystem",
    "VerificationPhase",
    "Phase1VerificationData",
    "Phase2VerificationData",
    "PhaseTiming",
    # Spot check
    "ChainStrikeState",
    "ReplayOutcome",
    "SpotCheckResult",
    "StatisticalThresholds",
    "TrainingTrace",
    "TrainingTracePrivate",
    "accumulate_strikes",
    "compute_grad_hash_chain",
    "derive_challenge_step_indices",
    "verify_spot_replay",
    "verify_statistical",
    "verify_structural",
    # Slashing
    "SlashingMechanism",
    "SlashingReason",
    # Dynamic block time
    "DynamicBlockTimeManager",
    "BlockTimeAdjustment",
    "BlockTimeAdjustmentProposal",
    "VerificationTiming",
]

if _CONSENSUS_AVAILABLE:
    __all__.extend([
        "PogoConsensus",
        "PogoBlock",
        "VerificationData",
        "AttestationType",
    ])

if _ENHANCED_AVAILABLE:
    __all__.extend([
        "EnhancedPogoConsensus",
        "MultiMerkleProof",
        "AttestationAggregation",
        "AttestationResult",
        "NOOP_TRAINING_MODEL_ID",
    ])
