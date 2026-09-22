"""Models subpackage for DeSSIN model management and lifecycle."""

# Light imports (no heavy deps)
from .model_file_manager import (
    ModelFileManager,
    ModelFile,
    FileBundle,
)
from .model_lifecycle_transactions import (
    ModelLifecycleManager,
)
from .model_training_scheduler import (
    ModelTrainingScheduler,
    ModelTrainingSlot,
    ModelPriority,
)
from .model_tokenization_ops import (
    change_model_tokenization_stub,
    ChangeTokenizationResult,
)
from .file_upload_transactions import (
    UploadFileTransaction,
    CreateFileBundleTransaction,
    DeleteFileTransaction,
    FileUploadTransactionProcessor,
)
from .model_format_registry import (
    ModelFormat,
    ModelFamily,
    ModelPrecision,
    ModelFormatSpec,
    HardwareRequirements,
    GGUFMetadata,
    TrainingPipelineSpec,
    PipelineStageSpec,
    ModelSubmissionSpec,
    dessin_native_preset,
    gguf_preset,
    safetensors_preset,
)
from .model_submission_transactions import (
    RegisterModelTransaction,
    UploadTrainingDataTransaction,
    RenewStorageLeaseTransaction,
)

# Heavy imports wrapped in try/except
try:
    from .model_manager import (
        ModelManager,
        ModelInfo,
        ModelQueryResult,
    )
    _MODEL_MANAGER_AVAILABLE = True
except ImportError:
    _MODEL_MANAGER_AVAILABLE = False
    ModelManager = None  # type: ignore[misc,assignment]
    ModelInfo = None  # type: ignore[misc,assignment]
    ModelQueryResult = None  # type: ignore[misc,assignment]

try:
    from .llm_model_spec import (
        DecoderOnlyTransformerSpec,
        TokenizerKind,
        TokenizerProfile,
        LLMVariant,
        NeuralArchitectureFamily,
        ARCHITECTURE_SCHEMA_VERSION,
        parse_architecture_dict,
        resolve_char_lm_tokenizer_options,
    )
    _LLM_SPEC_AVAILABLE = True
except ImportError:
    _LLM_SPEC_AVAILABLE = False
    DecoderOnlyTransformerSpec = None  # type: ignore[misc,assignment]
    TokenizerKind = None  # type: ignore[misc,assignment]
    TokenizerProfile = None  # type: ignore[misc,assignment]
    LLMVariant = None  # type: ignore[misc,assignment]
    NeuralArchitectureFamily = None  # type: ignore[misc,assignment]
    ARCHITECTURE_SCHEMA_VERSION = None
    parse_architecture_dict = None  # type: ignore[misc,assignment]
    resolve_char_lm_tokenizer_options = None  # type: ignore[misc,assignment]

__all__ = [
    # Model file manager
    "ModelFileManager",
    "ModelFile",
    "FileBundle",
    # Model lifecycle
    "ModelLifecycleManager",
    # Training scheduler
    "ModelTrainingScheduler",
    "ModelTrainingSlot",
    "ModelPriority",
    # Tokenization
    "change_model_tokenization_stub",
    "ChangeTokenizationResult",
    # File upload (legacy)
    "UploadFileTransaction",
    "CreateFileBundleTransaction",
    "DeleteFileTransaction",
    "FileUploadTransactionProcessor",
    # Model format registry
    "ModelFormat",
    "ModelFamily",
    "ModelPrecision",
    "ModelFormatSpec",
    "HardwareRequirements",
    "GGUFMetadata",
    "TrainingPipelineSpec",
    "PipelineStageSpec",
    "ModelSubmissionSpec",
    "dessin_native_preset",
    "gguf_preset",
    "safetensors_preset",
    # Model submission transactions
    "RegisterModelTransaction",
    "UploadTrainingDataTransaction",
    "RenewStorageLeaseTransaction",
]

if _MODEL_MANAGER_AVAILABLE:
    __all__.extend([
        "ModelManager",
        "ModelInfo",
        "ModelQueryResult",
    ])

if _LLM_SPEC_AVAILABLE:
    __all__.extend([
        "DecoderOnlyTransformerSpec",
        "TokenizerKind",
        "TokenizerProfile",
        "LLMVariant",
        "NeuralArchitectureFamily",
        "ARCHITECTURE_SCHEMA_VERSION",
        "parse_architecture_dict",
        "resolve_char_lm_tokenizer_options",
    ])
