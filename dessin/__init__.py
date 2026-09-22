"""
DeSSIN: Decentralized Secure Super Intelligence Network

A blockchain implementation based on PoGO (Proof of Gradient Optimization) consensus
for training and verifying machine learning models.
"""

__version__ = "0.1.0"
__author__ = "DeSSIN Team"
__license__ = "AGPL-3.0-or-later"

# Import core modules that don't require heavy dependencies
from .runtime.pretty_console import pretty_print, pretty_style_enabled

# Optional imports that may require transformers/torch
try:
    from .models.llm_model_spec import (
        ARCHITECTURE_SCHEMA_VERSION,
        DecoderOnlyTransformerSpec,
        LLMVariant,
        NeuralArchitectureFamily,
        TokenizerKind,
        TokenizerProfile,
        parse_architecture_dict,
        resolve_char_lm_tokenizer_options,
    )
    _LLM_SPEC_AVAILABLE = True
except ImportError:
    _LLM_SPEC_AVAILABLE = False

try:
    from .models.model_tokenization_ops import ChangeTokenizationResult, change_model_tokenization_stub
    _TOKENIZATION_OPS_AVAILABLE = True
except ImportError:
    _TOKENIZATION_OPS_AVAILABLE = False

# Optional imports that may require heavy dependencies
try:
    from .consensus.spot_check_verification import (
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
    _SPOT_CHECK_AVAILABLE = True
except ImportError:
    _SPOT_CHECK_AVAILABLE = False

try:
    from .models.model_manager import ModelManager
except ImportError:
    ModelManager = None  # type: ignore[misc, assignment]

try:
    from .distribution.bittorrent_distributor import BitTorrentDistributor
except ImportError:
    BitTorrentDistributor = None  # type: ignore[misc, assignment]

try:
    from .runtime.progress_tracker import ProgressTracker
except ImportError:
    ProgressTracker = None  # type: ignore[misc, assignment]

# Import hybrid distributor with error handling
try:
    from .distribution.hybrid_distributor import HybridDistributor
    _HYBRID_AVAILABLE = True
except (ImportError, SyntaxError):
    _HYBRID_AVAILABLE = False

# Optional imports that may require chaincraft
try:
    from .runtime.node import DessinNode
    from .consensus import PogoConsensus
    from .consensus.transactions import ModelQueryTransaction, ConditionalTransferTransaction
    _FULL_IMPORTS = True
except ImportError:
    _FULL_IMPORTS = False

__all__ = [
    "pretty_print",
    "pretty_style_enabled",
]

if ProgressTracker is not None:
    __all__.append("ProgressTracker")

if _SPOT_CHECK_AVAILABLE:
    __all__.extend([
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
    ])

if _LLM_SPEC_AVAILABLE:
    __all__.extend([
        "ARCHITECTURE_SCHEMA_VERSION",
        "DecoderOnlyTransformerSpec",
        "LLMVariant",
        "NeuralArchitectureFamily",
        "TokenizerKind",
        "TokenizerProfile",
        "parse_architecture_dict",
        "resolve_char_lm_tokenizer_options",
    ])

if _TOKENIZATION_OPS_AVAILABLE:
    __all__.extend([
        "ChangeTokenizationResult",
        "change_model_tokenization_stub",
    ])

if ModelManager is not None:
    __all__.append("ModelManager")
if BitTorrentDistributor is not None:
    __all__.append("BitTorrentDistributor")

if _HYBRID_AVAILABLE:
    __all__.append("HybridDistributor")

if _FULL_IMPORTS:
    __all__.extend([
        "DessinNode",
        "PogoConsensus",
        "ModelQueryTransaction",
        "ConditionalTransferTransaction"
    ])
