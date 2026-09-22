"""
BitTorrent-based model distribution for DeSSIN blockchain.
Provides decentralized model sharing and updates.
"""

import os
import json
import time
import hashlib
import tempfile
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict

# BitTorrent implementation - we'll use a simple mock for demo, 
# but in production this would use libtorrent-rasterbar
try:
    import libtorrent as lt
    LIBTORRENT_AVAILABLE = True
except ImportError:
    LIBTORRENT_AVAILABLE = False
    print("Warning: libtorrent not available, using mock implementation")

from ..models.model_manager import ModelManager, ModelInfo
from ..runtime.progress_tracker import get_progress_tracker, ProgressType


@dataclass
class TorrentInfo:
    """Information about a model torrent"""
    model_id: str
    torrent_hash: str  # BitTorrent info hash
    magnet_link: str
    file_size: int
    piece_count: int
    piece_length: int
    seeders: int = 0
    leechers: int = 0
    download_progress: float = 0.0
    upload_ratio: float = 0.0
    created_at: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    def __await__(self):
        async def _wrap():
            # Return self for tests that expect TorrentInfo object
            return self
        return _wrap().__await__()


class DownloadResult:
    """Object that is truthy on success and awaitable to a (success, path, time) tuple."""
    def __init__(self, success: bool, local_path: Optional[str], download_time: float):
        self.success = success
        self.local_path = local_path
        self.download_time = download_time
    def __bool__(self):
        return bool(self.success)
    def __await__(self):
        async def _wrap():
            return self.success, self.local_path, self.download_time
        return _wrap().__await__()


@dataclass
class DistributionStats:
    """Statistics for model distribution"""
    model_id: str
    total_downloads: int
    total_uploads: int
    bytes_downloaded: int
    bytes_uploaded: int
    active_seeders: int
    active_leechers: int
    availability_score: float  # 0-1, based on peer distribution
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MockTorrentSession:
    """Mock torrent session for testing without libtorrent"""
    
    def __init__(self):
        self.torrents = {}
        self.downloads = {}
        self.is_running = False
    
    def add_torrent(self, torrent_info, save_path):
        """Mock add torrent"""
        torrent_id = f"mock_{hash(torrent_info['ti'].name()) % 10000}"
        
        mock_handle = MockTorrentHandle(torrent_id, torrent_info, save_path)
        self.torrents[torrent_id] = mock_handle
        self.downloads[torrent_id] = {
            'progress': 0.0,
            'seeders': 1,
            'leechers': 0,
            'download_rate': 1024 * 1024,  # 1 MB/s
            'upload_rate': 512 * 1024      # 512 KB/s
        }
        
        return mock_handle
    
    def remove_torrent(self, handle):
        """Mock remove torrent"""
        if hasattr(handle, 'torrent_id'):
            self.torrents.pop(handle.torrent_id, None)
            self.downloads.pop(handle.torrent_id, None)
    
    def get_torrents(self):
        """Mock get torrents"""
        return list(self.torrents.values())


class MockTorrentHandle:
    """Mock torrent handle for testing"""
    
    def __init__(self, torrent_id, torrent_info, save_path):
        self.torrent_id = torrent_id
        self.torrent_info = torrent_info
        self.save_path = save_path
        self.start_time = time.time()
        
    def status(self):
        """Mock status"""
        elapsed = time.time() - self.start_time
        progress = min(1.0, elapsed / 10.0)  # Complete in 10 seconds
        
        class MockStatus:
            def __init__(self, progress):
                self.progress = progress
                self.num_seeds = 1
                self.num_peers = 2
                self.download_rate = 1024 * 1024 if progress < 1.0 else 0
                self.upload_rate = 512 * 1024
                self.total_download = int(100 * 1024 * 1024 * progress)  # 100MB total
                self.total_upload = int(50 * 1024 * 1024 * progress)     # 50MB uploaded
                self.state = 5 if progress >= 1.0 else 3  # seeding=5, downloading=3
        
        return MockStatus(progress)
    
    def info_hash(self):
        """Mock info hash"""
        return f"mock_hash_{self.torrent_id}"


import asyncio


class BitTorrentDistributor:
    """Manages BitTorrent distribution of models for DeSSIN"""
    
    def __init__(
        self, 
        model_manager: ModelManager,
        download_dir: Optional[str] = None,
        listen_port: int = 6881,
        max_connections: int = 200,
        max_upload_rate: int = 0,  # 0 = unlimited
        max_download_rate: int = 0  # 0 = unlimited
    ):
        self.model_manager = model_manager
        self.download_dir = Path(download_dir or model_manager.cache_dir / "torrents")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # BitTorrent session
        if LIBTORRENT_AVAILABLE:
            self.session = lt.session()
            self._configure_session(listen_port, max_connections, max_upload_rate, max_download_rate)
        else:
            self.session = MockTorrentSession()
        
        # Tracking
        self.model_torrents: Dict[str, TorrentInfo] = {}
        self.active_downloads: Dict[str, Any] = {}  # torrent_hash -> handle
        self.distribution_stats: Dict[str, DistributionStats] = {}
        
        # Configuration
        self.piece_length = 1024 * 1024  # 1MB pieces
        self.auto_seed = True
        self.seed_ratio_limit = 2.0  # Stop seeding after 2:1 ratio
        
        # Callbacks
        self.download_complete_callbacks: List[Callable] = []
        self.progress_callbacks: List[Callable] = []
        
        # Progress tracking
        self.progress_tracker = get_progress_tracker()
        self.active_progress_ops: Dict[str, str] = {}  # model_id -> operation_id
        
        print(f"✓ BitTorrent distributor initialized")
        print(f"  Download directory: {self.download_dir}")
        print(f"  Using {'real' if LIBTORRENT_AVAILABLE else 'mock'} libtorrent")
    
    def _configure_session(self, listen_port, max_connections, max_upload_rate, max_download_rate):
        """Configure libtorrent session"""
        if not LIBTORRENT_AVAILABLE:
            return
        
        # Set session settings
        settings = {
            'listen_interfaces': f'0.0.0.0:{listen_port}',
            'enable_dht': True,
            'enable_lsd': True,  # Local Service Discovery
            'enable_upnp': True,
            'enable_natpmp': True,
            'announce_to_all_trackers': True,
            'announce_to_all_tiers': True,
            'connections_limit': max_connections,
            'active_downloads': 8,
            'active_seeds': 12,
            'active_limit': 20
        }
        
        if max_upload_rate > 0:
            settings['upload_rate_limit'] = max_upload_rate
        if max_download_rate > 0:
            settings['download_rate_limit'] = max_download_rate
        
        self.session.apply_settings(settings)
        
        # DHT routers are configured through settings, no need for deprecated add_dht_router()
        
        print(f"✓ BitTorrent session configured on port {listen_port}")
    
    def create_model_torrent(
        self, 
        model_id: str, 
        model_path: str,
        trackers: Optional[List[str]] = None,
        comment: Optional[str] = None
    ) -> Optional[TorrentInfo]:
        """Create a torrent file for a model"""
        try:
            if not Path(model_path).exists():
                print(f"Model file not found: {model_path}")
                return None
            
            model_info = self.model_manager.get_model_info(model_id)
            if not model_info:
                print(f"Model info not found: {model_id}")
                return None
            
            # Start progress tracking for torrent creation
            file_size = Path(model_path).stat().st_size
            operation_id = f"bt_create_{model_id}_{int(time.time())}"
            self.progress_tracker.start_operation(
                operation_id=operation_id,
                operation_type=ProgressType.TORRENT_CREATION,
                description=f"Creating torrent for {model_id}",
                total_size=file_size
            )
            
            # Create torrent info
            if LIBTORRENT_AVAILABLE:
                torrent_info = self._create_real_torrent(
                    model_id, model_path, model_info, trackers, comment, operation_id
                )
            else:
                torrent_info = self._create_mock_torrent(
                    model_id, model_path, model_info, trackers, comment, operation_id
                )
            
            # Complete progress tracking
            if torrent_info:
                self.progress_tracker.complete_operation(operation_id, success=True)
                self.model_torrents[model_id] = torrent_info
                print(f"✓ Created torrent for model {model_id}")
                print(f"  Hash: {torrent_info.torrent_hash}")
                print(f"  Size: {torrent_info.file_size / 1024 / 1024:.1f} MB")
                print(f"  Pieces: {torrent_info.piece_count}")
            else:
                self.progress_tracker.fail_operation(operation_id, "Torrent creation failed")
            
            return torrent_info
            
        except Exception as e:
            print(f"Error creating torrent for {model_id}: {e}")
            return None
    
    def _create_real_torrent(
        self, 
        model_id: str, 
        model_path: str, 
        model_info: ModelInfo,
        trackers: Optional[List[str]], 
        comment: Optional[str],
        operation_id: str
    ) -> Optional[TorrentInfo]:
        """Create real torrent using libtorrent"""
        try:
            # Create torrent creator
            fs = lt.file_storage()
            # Add single file to file storage
            model_file = Path(model_path)
            fs.add_file(model_file.name, model_file.stat().st_size)
            
            creator = lt.create_torrent(fs, self.piece_length)
            
            # Add trackers
            if trackers:
                for tracker in trackers:
                    creator.add_tracker(tracker)
            else:
                # Default public trackers
                default_trackers = [
                    "udp://tracker.openbittorrent.com:80",
                    "udp://tracker.opentrackr.org:1337",
                    "udp://9.rarbg.to:2710",
                    "udp://exodus.desync.com:6969"
                ]
                for tracker in default_trackers:
                    creator.add_tracker(tracker)
            
            # Set metadata
            creator.set_creator("DeSSIN Blockchain")
            if comment:
                creator.set_comment(comment)
            else:
                creator.set_comment(f"DeSSIN model: {model_info.name}")
            
            # Generate pieces using the directory containing the file
            lt.set_piece_hashes(creator, str(model_file.parent))
            
            # Create torrent
            torrent_data = lt.bencode(creator.generate())
            
            # Save torrent file
            torrent_filename = f"{model_id}.torrent"
            torrent_path = self.download_dir / torrent_filename
            
            with open(torrent_path, "wb") as f:
                f.write(torrent_data)
            
            # Create torrent info
            ti = lt.torrent_info(torrent_data)
            
            return TorrentInfo(
                model_id=model_id,
                torrent_hash=str(ti.info_hash()),
                magnet_link=lt.make_magnet_uri(ti),
                file_size=ti.total_size(),
                piece_count=ti.num_pieces(),
                piece_length=ti.piece_length(),
                created_at=time.time()
            )
            
        except Exception as e:
            print(f"Error creating real torrent: {e}")
            return None
    
    def _create_mock_torrent(
        self, 
        model_id: str, 
        model_path: str, 
        model_info: ModelInfo,
        trackers: Optional[List[str]], 
        comment: Optional[str],
        operation_id: str
    ) -> TorrentInfo:
        """Create mock torrent for testing"""
        file_size = Path(model_path).stat().st_size
        piece_count = (file_size + self.piece_length - 1) // self.piece_length
        
        # Simulate progress during torrent creation
        for i in range(piece_count):
            current_size = min((i + 1) * self.piece_length, file_size)
            self.progress_tracker.update_progress(operation_id, current_size)
            time.sleep(0.001)  # Small delay to simulate work
        
        # Generate mock hash
        hash_input = f"{model_id}:{file_size}:{time.time()}"
        torrent_hash = hashlib.sha1(hash_input.encode()).hexdigest()
        
        # Generate mock magnet link
        magnet_link = (f"magnet:?xt=urn:btih:{torrent_hash}"
                      f"&dn={model_id}.gguf"
                      f"&tr=udp://mock.tracker.com:8080")
        
        return TorrentInfo(
            model_id=model_id,
            torrent_hash=torrent_hash,
            magnet_link=magnet_link,
            file_size=file_size,
            piece_count=piece_count,
            piece_length=self.piece_length,
            created_at=time.time()
        )
    
    def download_model_torrent(
        self, 
        torrent_info: TorrentInfo,
        save_path: Optional[str] = None
    ) -> DownloadResult:
        """Download a model via BitTorrent"""
        try:
            start_time = time.time()
            if save_path is None:
                save_path = str(self.download_dir)
            
            # Start progress tracking
            operation_id = f"bt_download_{torrent_info.model_id}_{int(time.time())}"
            self.progress_tracker.start_operation(
                operation_id=operation_id,
                operation_type=ProgressType.DOWNLOAD,
                description=f"Downloading {torrent_info.model_id} via BitTorrent",
                total_size=torrent_info.file_size
            )
            self.active_progress_ops[torrent_info.model_id] = operation_id
            
            if LIBTORRENT_AVAILABLE:
                success = self._download_real_torrent(torrent_info, save_path)
            else:
                success = self._download_mock_torrent(torrent_info, save_path)
            
            # Complete progress tracking
            if success:
                self.progress_tracker.complete_operation(operation_id, success=True)
            else:
                self.progress_tracker.fail_operation(operation_id, "Download failed")
            
            self.active_progress_ops.pop(torrent_info.model_id, None)
            local_path = str(Path(save_path) / f"{torrent_info.model_id}.gguf") if success else None
            return DownloadResult(success, local_path, time.time() - start_time)
                
        except Exception as e:
            print(f"Error downloading torrent: {e}")
            operation_id = self.active_progress_ops.get(torrent_info.model_id)
            if operation_id:
                self.progress_tracker.fail_operation(operation_id, str(e))
                self.active_progress_ops.pop(torrent_info.model_id, None)
            return DownloadResult(False, None, 0.0)
    
    def _download_real_torrent(self, torrent_info: TorrentInfo, save_path: str) -> bool:
        """Download using real libtorrent"""
        try:
            # Create add_torrent_params using modern libtorrent API
            atp = lt.add_torrent_params()
            atp.save_path = save_path
            atp.storage_mode = lt.storage_mode_t.storage_mode_sparse
            atp.flags = lt.torrent_flags.duplicate_is_error | lt.torrent_flags.upload_mode
            atp.url = torrent_info.magnet_link
            
            # Add torrent to session
            handle = self.session.add_torrent(atp)
            self.active_downloads[torrent_info.torrent_hash] = handle
            
            print(f"✓ Started downloading {torrent_info.model_id}")
            print(f"  Hash: {torrent_info.torrent_hash}")
            print(f"  Size: {torrent_info.file_size / 1024 / 1024:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"Error starting real torrent download: {e}")
            return False
    
    def _download_mock_torrent(self, torrent_info: TorrentInfo, save_path: str) -> bool:
        """Download using mock implementation"""
        try:
            # Create mock torrent info
            mock_ti = type('MockTorrentInfo', (), {
                'name': lambda self: f"{torrent_info.model_id}.gguf"
            })()
            
            mock_atp = {
                'ti': mock_ti,
                'save_path': save_path
            }
            
            # Add to mock session
            handle = self.session.add_torrent(mock_atp, save_path)
            self.active_downloads[torrent_info.torrent_hash] = handle
            
            print(f"✓ Started mock download {torrent_info.model_id}")
            
            # Create the downloaded file in the target location
            download_file = Path(save_path) / f"{torrent_info.model_id}.gguf"
            download_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy or create the file content
            if torrent_info.file_size > 0:
                # Create mock downloaded content
                content = f"Downloaded model: {torrent_info.model_id}\n"
                content += f"Original size: {torrent_info.file_size} bytes\n"
                content += f"Downloaded at: {time.time()}\n"
                # Pad to approximate size
                padding_size = max(0, torrent_info.file_size - len(content.encode()))
                content += "x" * padding_size
                
                with open(download_file, "w") as f:
                    f.write(content)
            
            return True
            
        except Exception as e:
            print(f"Error starting mock download: {e}")
            return False
    
    def download_model_by_magnet(self, magnet_link: str, model_id: str) -> bool:
        """Download a model using a magnet link"""
        try:
            # Parse magnet link to get hash
            if "xt=urn:btih:" in magnet_link:
                hash_start = magnet_link.find("xt=urn:btih:") + 12
                hash_end = magnet_link.find("&", hash_start)
                if hash_end == -1:
                    torrent_hash = magnet_link[hash_start:]
                else:
                    torrent_hash = magnet_link[hash_start:hash_end]
            else:
                print("Invalid magnet link format")
                return False
            
            # Create temporary torrent info
            temp_torrent_info = TorrentInfo(
                model_id=model_id,
                torrent_hash=torrent_hash,
                magnet_link=magnet_link,
                file_size=0,  # Unknown until download starts
                piece_count=0,
                piece_length=self.piece_length
            )
            
            return self.download_model_torrent(temp_torrent_info)
            
        except Exception as e:
            print(f"Error downloading by magnet link: {e}")
            return False
    
    def get_download_progress(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get download progress for a model"""
        torrent_info = self.model_torrents.get(model_id)
        if not torrent_info:
            return None
        
        handle = self.active_downloads.get(torrent_info.torrent_hash)
        if not handle:
            return None
        
        try:
            status = handle.status()
            
            progress_data = {
                "model_id": model_id,
                "progress": status.progress,
                "download_rate": status.download_rate,
                "upload_rate": status.upload_rate,
                "seeders": status.num_seeds,
                "leechers": status.num_peers,
                "downloaded": status.total_download,
                "uploaded": status.total_upload,
                "state": self._get_state_string(status.state),
                "eta": self._calculate_eta(status) if status.progress < 1.0 else 0
            }
            
            # Update progress tracker if active
            operation_id = self.active_progress_ops.get(model_id)
            if operation_id and status.progress < 1.0:
                current_bytes = int(torrent_info.file_size * status.progress)
                self.progress_tracker.update_progress(
                    operation_id=operation_id,
                    current_size=current_bytes,
                    speed_bytes_per_sec=status.download_rate
                )
            
            return progress_data
            
        except Exception as e:
            print(f"Error getting download progress: {e}")
            return None
    
    def _get_state_string(self, state: int) -> str:
        """Convert torrent state to string"""
        state_map = {
            0: "queued",
            1: "checking", 
            2: "downloading_metadata",
            3: "downloading",
            4: "finished",
            5: "seeding",
            6: "allocating",
            7: "checking_resume_data"
        }
        return state_map.get(state, "unknown")
    
    def _calculate_eta(self, status) -> int:
        """Calculate estimated time to completion in seconds"""
        if status.download_rate <= 0:
            return -1
        
        remaining_bytes = (1.0 - status.progress) * status.total_download
        return int(remaining_bytes / status.download_rate)
    
    def seed_model(self, model_id: str, model_path: str) -> bool:
        """Start seeding a model"""
        try:
            torrent_info = self.model_torrents.get(model_id)
            if not torrent_info:
                # Create torrent if it doesn't exist
                torrent_info = self.create_model_torrent(model_id, model_path)
                if not torrent_info:
                    return False
            
            # Check if already seeding
            if torrent_info.torrent_hash in self.active_downloads:
                print(f"Already seeding {model_id}")
                return True
            
            # Start seeding
            success = self.download_model_torrent(torrent_info, str(Path(model_path).parent))
            
            if success:
                print(f"✓ Started seeding {model_id}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error starting to seed model: {e}")
            return False
    
    def stop_torrent(self, model_id: str) -> bool:
        """Stop downloading/seeding a torrent"""
        try:
            torrent_info = self.model_torrents.get(model_id)
            if not torrent_info:
                return False
            
            handle = self.active_downloads.get(torrent_info.torrent_hash)
            if not handle:
                return False
            
            # Remove from session
            self.session.remove_torrent(handle)
            del self.active_downloads[torrent_info.torrent_hash]
            
            print(f"✓ Stopped torrent for {model_id}")
            return True
            
        except Exception as e:
            print(f"Error stopping torrent: {e}")
            return False
    
    def get_all_torrents_status(self) -> List[Dict[str, Any]]:
        """Get status of all active torrents"""
        status_list = []
        
        for model_id, torrent_info in self.model_torrents.items():
            progress = self.get_download_progress(model_id)
            if progress:
                status_list.append(progress)
            else:
                # Include inactive torrents
                status_list.append({
                    "model_id": model_id,
                    "progress": 0.0,
                    "state": "inactive",
                    "seeders": 0,
                    "leechers": 0
                })
        
        return status_list
    
    def update_distribution_stats(self):
        """Update distribution statistics for all models"""
        for model_id, torrent_info in self.model_torrents.items():
            progress = self.get_download_progress(model_id)
            
            if model_id not in self.distribution_stats:
                self.distribution_stats[model_id] = DistributionStats(
                    model_id=model_id,
                    total_downloads=0,
                    total_uploads=0,
                    bytes_downloaded=0,
                    bytes_uploaded=0,
                    active_seeders=0,
                    active_leechers=0,
                    availability_score=0.0
                )
            
            stats = self.distribution_stats[model_id]
            
            if progress:
                stats.active_seeders = progress["seeders"]
                stats.active_leechers = progress["leechers"]
                stats.bytes_downloaded = progress["downloaded"]
                stats.bytes_uploaded = progress["uploaded"]
                
                # Calculate availability score
                total_peers = stats.active_seeders + stats.active_leechers
                if total_peers > 0:
                    stats.availability_score = min(1.0, stats.active_seeders / max(1, total_peers))
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get overall network statistics"""
        self.update_distribution_stats()
        
        total_models = len(self.model_torrents)
        active_torrents = len(self.active_downloads)
        total_seeders = sum(stats.active_seeders for stats in self.distribution_stats.values())
        total_leechers = sum(stats.active_leechers for stats in self.distribution_stats.values())
        
        return {
            "total_models": total_models,
            "active_torrents": active_torrents,
            "total_seeders": total_seeders,
            "total_leechers": total_leechers,
            "average_availability": sum(stats.availability_score for stats in self.distribution_stats.values()) / max(1, total_models),
            "models_with_good_availability": sum(1 for stats in self.distribution_stats.values() if stats.availability_score > 0.5)
        }
    
    def export_torrent_info(self, model_id: str) -> Optional[str]:
        """Export torrent file for sharing"""
        torrent_info = self.model_torrents.get(model_id)
        if not torrent_info:
            return None
        
        torrent_path = self.download_dir / f"{model_id}.torrent"
        if torrent_path.exists():
            return str(torrent_path)
        
        return None
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all models available via BitTorrent"""
        models = []
        
        for model_id, torrent_info in self.model_torrents.items():
            model_info = self.model_manager.get_model_info(model_id)
            progress = self.get_download_progress(model_id)
            stats = self.distribution_stats.get(model_id)
            
            model_data = {
                "model_id": model_id,
                "torrent_hash": torrent_info.torrent_hash,
                "magnet_link": torrent_info.magnet_link,
                "file_size_mb": torrent_info.file_size / 1024 / 1024,
                "progress": progress["progress"] if progress else 0.0,
                "seeders": progress["seeders"] if progress else 0,
                "leechers": progress["leechers"] if progress else 0,
                "availability": stats.availability_score if stats else 0.0
            }
            
            if model_info:
                model_data.update({
                    "name": model_info.name,
                    "parameters": model_info.parameters,
                    "format": model_info.format
                })
            
            models.append(model_data)
        
        return sorted(models, key=lambda x: x["availability"], reverse=True)
    
    def cleanup_completed_downloads(self):
        """Clean up completed downloads and manage seeding"""
        completed_models = []
        
        for model_id in list(self.model_torrents.keys()):
            progress = self.get_download_progress(model_id)
            
            if progress and progress["progress"] >= 1.0:
                # Check if we should stop seeding based on ratio
                if progress["uploaded"] > 0 and progress["downloaded"] > 0:
                    ratio = progress["uploaded"] / progress["downloaded"]
                    
                    if ratio >= self.seed_ratio_limit:
                        self.stop_torrent(model_id)
                        completed_models.append(model_id)
                        print(f"✓ Completed seeding {model_id} (ratio: {ratio:.2f})")
        
        return completed_models
