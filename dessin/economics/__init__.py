"""Economics subpackage for DeSSIN tokenomics and payments."""

from .dynamic_parameters import (
    DynamicParameterManager,
    ParameterAdjustment,
    ParameterType,
)
from .storage_topup import (
    StorageTopUpManager,
    StorageTopUp,
)
from .compute_fee_schedule import (
    ComputeFeeSchedule,
    TrainingPhase,
    DEFAULT_PHASE_RATE_PER_MB,
    DATA_UPLOAD_RATE_PER_MB,
    DATA_STORAGE_RATE_PER_MB_PER_BLOCK,
)
from .storage_lease import (
    StorageLease,
    StorageLeaseManager,
    StorageLeaseConfig,
    LeaseStatus,
    LeaseEvent,
)

# Heavy import — depends on chaincraft
try:
    from .economic_system import (
        EconomicSystem,
        EconomicEvent,
        EconomicEventType,
    )
    _ECON_AVAILABLE = True
except ImportError:
    _ECON_AVAILABLE = False
    EconomicSystem = None  # type: ignore[misc,assignment]
    EconomicEvent = None   # type: ignore[misc,assignment]
    EconomicEventType = None  # type: ignore[misc,assignment]

__all__ = [
    # Dynamic parameters
    "DynamicParameterManager",
    "ParameterAdjustment",
    "ParameterType",
    # Storage top-up (legacy)
    "StorageTopUpManager",
    "StorageTopUp",
    # Compute fee schedule
    "ComputeFeeSchedule",
    "TrainingPhase",
    "DEFAULT_PHASE_RATE_PER_MB",
    "DATA_UPLOAD_RATE_PER_MB",
    "DATA_STORAGE_RATE_PER_MB_PER_BLOCK",
    # Storage lease
    "StorageLease",
    "StorageLeaseManager",
    "StorageLeaseConfig",
    "LeaseStatus",
    "LeaseEvent",
]

if _ECON_AVAILABLE:
    __all__.extend([
        "EconomicSystem",
        "EconomicEvent",
        "EconomicEventType",
    ])
