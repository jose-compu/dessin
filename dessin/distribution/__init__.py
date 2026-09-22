"""Distribution subpackage for DeSSIN model distribution via BitTorrent and HuggingFace."""

from .bittorrent_distributor import (
    BitTorrentDistributor,
    TorrentInfo,
)
from .enhanced_torrent_distributor import (
    EnhancedTorrentDistributor,
    TorrentPeer,
)

# Optional imports that require libtorrent
try:
    from .real_torrent_distributor import (
        RealTorrentDistributor,
    )
    _REAL_TORRENT_AVAILABLE = True
except ImportError:
    _REAL_TORRENT_AVAILABLE = False
    RealTorrentDistributor = None  # type: ignore[misc,assignment]

from .huggingface_distributor import (
    HuggingFaceDistributor,
)
from .hybrid_distributor import (
    HybridDistributor,
    DistributionMethod,
)
from .hf_integration import (
    HuggingFaceIntegration,
)
from .hf_mock import (
    create_hf_integration,
)
from .hf_updater import (
    HuggingFaceUpdater,
    ModelVersion,
    ModelUpdateRecord,
)

__all__ = [
    # BitTorrent
    "BitTorrentDistributor",
    "TorrentInfo",
    # Enhanced torrent
    "EnhancedTorrentDistributor",
    "TorrentPeer",
    # HuggingFace
    "HuggingFaceDistributor",
    # Hybrid
    "HybridDistributor",
    "DistributionMethod",
    # HF integration
    "HuggingFaceIntegration",
    # HF mock
    "create_hf_integration",
    # HF updater
    "HuggingFaceUpdater",
    "ModelVersion",
    "ModelUpdateRecord",
]

if _REAL_TORRENT_AVAILABLE:
    __all__.append("RealTorrentDistributor")
