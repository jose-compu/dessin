"""
Hybrid model distribution system for DeSSIN.
Manages multiple distribution channels with BitTorrent as primary.
"""

import os
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .bittorrent_distributor import BitTorrentDistributor, TorrentInfo
from ..models.model_manager import ModelManager, ModelInfo
from .huggingface_distributor import HuggingFaceDistributor


class DistributionMethod(Enum):
    BITTORRENT = "bittorrent"
    IPFS = "ipfs"
    HUGGINGFACE = "huggingface"
    HTTPS = "https"


@dataclass
class DistributionResult:
    """Result of a model distribution operation"""
    success: bool
    method_used: DistributionMethod
    local_path: Optional[str] = None
    download_time: float = 0.0
    download_speed_mbps: float = 0.0
    error_message: Optional[str] = None
    fallback_attempts: List[DistributionMethod] = None
    
    def __post_init__(self):
        if self.fallback_attempts is None:
            self.fallback_attempts = []


@dataclass
class DistributionConfig:
    """Configuration for hybrid distribution"""
    # Method priorities (higher number = higher priority)
    method_priorities: Dict[DistributionMethod, int]
    
    # Timeout settings (seconds)
    bittorrent_timeout: int = 300  # 5 minutes
    ipfs_timeout: int = 180        # 3 minutes  
    huggingface_timeout: int = 120  # 2 minutes
    https_timeout: int = 60        # 1 minute
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 5.0
    
    # Performance thresholds
    min_download_speed_mbps: float = 0.5  # Switch to fallback if slower
    max_concurrent_downloads: int = 3
    
    # BitTorrent specific
    bt_min_seeders: int = 1        # Minimum seeders required
    bt_max_wait_time: int = 60     # Max time to wait for seeders
    
    @classmethod
    def default(cls) -> 'DistributionConfig':
        """Create default configuration"""
        return cls(
            method_priorities={
                DistributionMethod.BITTORRENT: 4,    # Highest priority
                DistributionMethod.IPFS: 3,
                DistributionMethod.HUGGINGFACE: 2,
                DistributionMethod.HTTPS: 1          # Lowest priority (fallback)
            }
        )


class HybridDistributor:
    """
    Hybrid model distribution system that intelligently selects
    the best distribution method for each scenario.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        config: Optional[DistributionConfig] = None,
        enable_bittorrent: bool = True,
        enable_ipfs: bool = False,  # Not implemented yet
        enable_huggingface: bool = True,
        enable_https: bool = True
    ):
        self.model_manager = model_manager
        self.config = config or DistributionConfig.default()
        
        # Initialize distributors
        self.distributors = {}
        
        if enable_bittorrent:
            self.bt_distributor = BitTorrentDistributor(model_manager)
            self.distributors[DistributionMethod.BITTORRENT] = self.bt_distributor
        
        if enable_ipfs:
            # TODO: Implement IPFS distributor
            print("⚠️  IPFS distributor not yet implemented")
        
        if enable_huggingface:
            # Use mock by default in tests
            self.hf_distributor = HuggingFaceDistributor(model_manager, force_mock=True)
            self.distributors[DistributionMethod.HUGGINGFACE] = self.hf_distributor
        
        if enable_https:
            # TODO: Implement HTTPS distributor
            print("⚠️  HTTPS distributor not yet implemented")
        
        # Tracking
        self.active_downloads: Dict[str, DistributionMethod] = {}
        self.distribution_history: List[DistributionResult] = []
        self.method_performance: Dict[DistributionMethod, Dict[str, float]] = {
            method: {"total_time": 0.0, "total_bytes": 0, "success_count": 0, "failure_count": 0}
            for method in DistributionMethod
        }
        
        print(f"✓ Hybrid distributor initialized")
        print(f"  Enabled methods: {list(self.distributors.keys())}")
    
    async def distribute_model(
        self, 
        model_id: str,
        model_path: str,
        preferred_method: Optional[DistributionMethod] = None,
        create_torrent: bool = True
    ) -> DistributionResult:
        """
        Distribute a model using the best available method.
        Creates torrents for seeding and announces to network.
        """
        start_time = time.time()
        
        try:
            # Validate model exists
            if not Path(model_path).exists():
                return DistributionResult(
                    success=False,
                    method_used=DistributionMethod.BITTORRENT,
                    error_message=f"Model file not found: {model_path}"
                )
            
            # Get model info
            model_info = self.model_manager.get_model_info(model_id)
            if not model_info:
                return DistributionResult(
                    success=False,
                    method_used=DistributionMethod.BITTORRENT,
                    error_message=f"Model info not found: {model_id}"
                )
            
            # Create BitTorrent torrent if enabled and requested
            torrent_created = False
            if create_torrent and DistributionMethod.BITTORRENT in self.distributors:
                torrent_info = self.bt_distributor.create_model_torrent(model_id, model_path)
                if torrent_info:
                    # Start seeding
                    seed_success = self.bt_distributor.seed_model(model_id, model_path)
                    torrent_created = seed_success
                    print(f"✓ Model {model_id} is now being seeded")
                    print(f"  Magnet: {torrent_info.magnet_link}")
            
            # Determine distribution method
            method = preferred_method or self._select_best_method(model_id)
            
            # Record distribution
            result = DistributionResult(
                success=torrent_created,
                method_used=method,
                local_path=model_path,
                download_time=time.time() - start_time
            )
            
            self.distribution_history.append(result)
            self._update_method_performance(method, result, Path(model_path).stat().st_size)
            
            return result
            
        except Exception as e:
            return DistributionResult(
                success=False,
                method_used=preferred_method or DistributionMethod.BITTORRENT,
                error_message=f"Distribution error: {e}",
                download_time=time.time() - start_time
            )
    
    async def download_model(
        self, 
        model_id: str, 
        magnet_link: Optional[str] = None,
        torrent_info: Optional[TorrentInfo] = None,
        preferred_method: Optional[DistributionMethod] = None,
        fallback_on_failure: bool = True,
        save_path: Optional[str] = None
    ) -> DistributionResult:
        """
        Download a model using the hybrid distribution system.
        Tries multiple methods in order of preference/performance.
        """
        start_time = time.time()
        
        if model_id in self.active_downloads:
            return DistributionResult(
                success=False,
                method_used=DistributionMethod.BITTORRENT,
                error_message=f"Download already in progress for {model_id}"
            )
        
        # Determine method order
        if preferred_method:
            methods_to_try = [preferred_method]
            if fallback_on_failure:
                methods_to_try.extend(self._get_fallback_methods(preferred_method))
        else:
            methods_to_try = self._get_ordered_methods(model_id)
        
        fallback_attempts = []
        last_error = None
        
        for method in methods_to_try:
            if method not in self.distributors:
                continue
            
            try:
                self.active_downloads[model_id] = method
                print(f"🔄 Attempting download of {model_id} via {method.value}")
                
                # Try the method
                success, local_path, download_time = await self._download_via_method(
                    method, model_id, magnet_link, torrent_info, save_path
                )
                
                if success:
                    # Calculate performance metrics
                    file_size = Path(local_path).stat().st_size if local_path else 0
                    speed_mbps = (file_size / 1024 / 1024) / max(download_time, 0.1) if download_time > 0 else 0
                    
                    result = DistributionResult(
                        success=True,
                        method_used=method,
                        local_path=local_path,
                        download_time=download_time,
                        download_speed_mbps=speed_mbps,
                        fallback_attempts=fallback_attempts
                    )
                    
                    self.distribution_history.append(result)
                    self._update_method_performance(method, result, file_size)
                    
                    print(f"✓ Successfully downloaded {model_id} via {method.value}")
                    print(f"  Speed: {speed_mbps:.1f} MB/s")
                    print(f"  Time: {download_time:.1f}s")
                    
                    return result
                else:
                    fallback_attempts.append(method)
                    print(f"❌ Failed to download {model_id} via {method.value}")
            
            except Exception as e:
                fallback_attempts.append(method)
                last_error = str(e)
                print(f"❌ Error downloading {model_id} via {method.value}: {e}")
            
            finally:
                self.active_downloads.pop(model_id, None)
        
        # All methods failed
        result = DistributionResult(
            success=False,
            method_used=methods_to_try[0] if methods_to_try else DistributionMethod.BITTORRENT,
            error_message=last_error or "All download methods failed",
            download_time=time.time() - start_time,
            fallback_attempts=fallback_attempts
        )
        
        self.distribution_history.append(result)
        return result
    
    async def _download_via_method(
        self,
        method: DistributionMethod,
        model_id: str,
        magnet_link: Optional[str],
        torrent_info: Optional[TorrentInfo],
        save_path: Optional[str]
    ) -> Tuple[bool, Optional[str], float]:
        """Download using specific method"""
        start_time = time.time()
        
        if method == DistributionMethod.BITTORRENT:
            return await self._download_via_bittorrent(model_id, magnet_link, torrent_info, save_path)
        elif method == DistributionMethod.IPFS:
            return await self._download_via_ipfs(model_id, save_path)
        elif method == DistributionMethod.HUGGINGFACE:
            return await self._download_via_huggingface(model_id, save_path)
        elif method == DistributionMethod.HTTPS:
            return await self._download_via_https(model_id, save_path)
        else:
            return False, None, time.time() - start_time
    
    async def _download_via_bittorrent(
        self, 
        model_id: str,
        magnet_link: Optional[str],
        torrent_info: Optional[TorrentInfo],
        save_path: Optional[str]
    ) -> Tuple[bool, Optional[str], float]:
        """Download via BitTorrent"""
        start_time = time.time()
        
        try:
            # Use provided torrent info or magnet link
            if torrent_info:
                success = self.bt_distributor.download_model_torrent(torrent_info, save_path)
            elif magnet_link:
                success = self.bt_distributor.download_model_by_magnet(magnet_link, model_id)
            else:
                # Try to find existing torrent
                if model_id in self.bt_distributor.model_torrents:
                    torrent_info = self.bt_distributor.model_torrents[model_id]
                    success = self.bt_distributor.download_model_torrent(torrent_info, save_path)
                else:
                    return False, None, time.time() - start_time
            
            if not success:
                return False, None, time.time() - start_time
            
            # Wait for download to complete
            timeout = self.config.bittorrent_timeout
            check_interval = 2.0
            elapsed = 0.0
            
            while elapsed < timeout:
                await asyncio.sleep(check_interval)
                elapsed += check_interval
                
                progress = self.bt_distributor.get_download_progress(model_id)
                if progress and progress["progress"] >= 1.0:
                    # Download complete, find the file
                    download_dir = self.bt_distributor.download_dir
                    model_files = list(download_dir.glob(f"{model_id}*"))
                    if model_files:
                        return True, str(model_files[0]), time.time() - start_time
                
                # Check if we have minimum seeders
                if progress and progress["seeders"] < self.config.bt_min_seeders:
                    if elapsed > self.config.bt_max_wait_time:
                        print(f"⚠️  Insufficient seeders for {model_id} ({progress['seeders']} < {self.config.bt_min_seeders})")
                        return False, None, time.time() - start_time
            
            print(f"⚠️  BitTorrent download timeout for {model_id}")
            return False, None, time.time() - start_time
            
        except Exception as e:
            print(f"BitTorrent download error: {e}")
            return False, None, time.time() - start_time
    
    async def _download_via_ipfs(self, model_id: str, save_path: Optional[str]) -> Tuple[bool, Optional[str], float]:
        """Download via IPFS (placeholder)"""
        # TODO: Implement IPFS download
        print("⚠️  IPFS download not yet implemented")
        return False, None, 0.0
    
    async def _download_via_huggingface(self, model_id: str, save_path: Optional[str]) -> Tuple[bool, Optional[str], float]:
        """Download via HuggingFace"""
        try:
            if DistributionMethod.HUGGINGFACE not in self.distributors:
                return False, None, 0.0
            
            # Extract repo_id from model_id if it contains it
            if ":" in model_id:
                repo_id = model_id.split(":", 1)[1]
            else:
                # For testing, assume model_id is the repo_id
                repo_id = model_id
            
            start_time = time.time()
            success, local_path, download_time = await self.hf_distributor.download_model(
                model_id=model_id
            )
            
            return success, local_path, download_time
                
        except Exception as e:
            print(f"HuggingFace download error: {e}")
            return False, None, time.time() - start_time
    
    async def _download_via_https(self, model_id: str, save_path: Optional[str]) -> Tuple[bool, Optional[str], float]:
        """Download via HTTPS (placeholder)"""
        # TODO: Implement HTTPS download
        print("⚠️  HTTPS download not yet implemented")
        return False, None, 0.0
    
    def _select_best_method(self, model_id: str) -> DistributionMethod:
        """Select the best distribution method for a model"""
        # For now, prefer BitTorrent for large models
        model_info = self.model_manager.get_model_info(model_id)
        
        if model_info and hasattr(model_info, 'file_size'):
            # Use BitTorrent for files > 100MB
            if model_info.file_size > 100 * 1024 * 1024:
                if DistributionMethod.BITTORRENT in self.distributors:
                    return DistributionMethod.BITTORRENT
        
        # Fallback to highest priority available method
        available_methods = list(self.distributors.keys())
        if not available_methods:
            return DistributionMethod.BITTORRENT  # Default
        
        return max(available_methods, key=lambda m: self.config.method_priorities.get(m, 0))
    
    def _get_ordered_methods(self, model_id: str) -> List[DistributionMethod]:
        """Get methods ordered by preference and performance"""
        available_methods = list(self.distributors.keys())
        
        # Sort by priority and performance
        def method_score(method: DistributionMethod) -> float:
            priority = self.config.method_priorities.get(method, 0)
            
            # Factor in historical performance
            perf = self.method_performance.get(method, {})
            success_rate = 0.5  # Default
            if perf.get("success_count", 0) + perf.get("failure_count", 0) > 0:
                total_attempts = perf["success_count"] + perf["failure_count"]
                success_rate = perf["success_count"] / total_attempts
            
            return priority * success_rate
        
        return sorted(available_methods, key=method_score, reverse=True)
    
    def _get_fallback_methods(self, exclude_method: DistributionMethod) -> List[DistributionMethod]:
        """Get fallback methods excluding the specified one"""
        all_methods = self._get_ordered_methods("")
        return [m for m in all_methods if m != exclude_method]
    
    def _update_method_performance(
        self, 
        method: DistributionMethod,
        result: DistributionResult,
        file_size: int
    ):
        """Update performance statistics for a method"""
        if method not in self.method_performance:
            self.method_performance[method] = {
                "total_time": 0.0,
                "total_bytes": 0,
                "success_count": 0,
                "failure_count": 0
            }
        
        perf = self.method_performance[method]
        perf["total_time"] += result.download_time
        perf["total_bytes"] += file_size
        
        if result.success:
            perf["success_count"] += 1
        else:
            perf["failure_count"] += 1
    
    def get_torrent_info(self, model_id: str) -> Optional[TorrentInfo]:
        """Get torrent info for a model"""
        if DistributionMethod.BITTORRENT in self.distributors:
            return self.bt_distributor.model_torrents.get(model_id)
        return None
    
    def get_magnet_link(self, model_id: str) -> Optional[str]:
        """Get magnet link for a model"""
        torrent_info = self.get_torrent_info(model_id)
        return torrent_info.magnet_link if torrent_info else None
    
    def list_distributed_models(self) -> List[Dict[str, Any]]:
        """List all models available for distribution"""
        models = []
        
        # Add BitTorrent models
        if DistributionMethod.BITTORRENT in self.distributors:
            bt_models = self.bt_distributor.list_available_models()
            for model in bt_models:
                model["available_methods"] = [DistributionMethod.BITTORRENT.value]
                models.append(model)
        
        # TODO: Add other distribution methods
        
        return models
    
    def get_distribution_stats(self) -> Dict[str, Any]:
        """Get overall distribution statistics"""
        stats = {
            "total_distributions": len(self.distribution_history),
            "success_rate": 0.0,
            "method_performance": {},
            "active_downloads": len(self.active_downloads)
        }
        
        # Calculate success rate
        if self.distribution_history:
            successful = sum(1 for r in self.distribution_history if r.success)
            stats["success_rate"] = successful / len(self.distribution_history)
        
        # Method performance
        for method, perf in self.method_performance.items():
            total_attempts = perf["success_count"] + perf["failure_count"]
            avg_speed = 0.0
            
            if perf["total_time"] > 0 and perf["total_bytes"] > 0:
                avg_speed = (perf["total_bytes"] / 1024 / 1024) / perf["total_time"]
            
            stats["method_performance"][method.value] = {
                "success_rate": perf["success_count"] / max(total_attempts, 1),
                "total_attempts": total_attempts,
                "avg_speed_mbps": avg_speed
            }
        
        # Add BitTorrent network stats
        if DistributionMethod.BITTORRENT in self.distributors:
            stats["bittorrent_network"] = self.bt_distributor.get_network_statistics()
        
        return stats
    
    def stop_download(self, model_id: str) -> bool:
        """Stop an active download"""
        if model_id not in self.active_downloads:
            return False
        
        method = self.active_downloads[model_id]
        success = False
        
        if method == DistributionMethod.BITTORRENT and DistributionMethod.BITTORRENT in self.distributors:
            success = self.bt_distributor.stop_torrent(model_id)
        
        if success:
            del self.active_downloads[model_id]
        
        return success
    
    def cleanup(self):
        """Clean up completed downloads and manage resources"""
        # Clean up BitTorrent
        if DistributionMethod.BITTORRENT in self.distributors:
            completed = self.bt_distributor.cleanup_completed_downloads()
            if completed:
                print(f"✓ Cleaned up {len(completed)} completed BitTorrent downloads")
        
        # Remove old distribution history (keep last 1000)
        if len(self.distribution_history) > 1000:
            self.distribution_history = self.distribution_history[-1000:]
    
    def is_initial_download(self, model_id: str) -> bool:
        """Check if this is an initial download (vs update)"""
        # Check if we have any local version of this model
        model_info = self.model_manager.get_model_info(model_id)
        return model_info is None
    
    def enforce_bittorrent_for_updates(self, model_id: str) -> bool:
        """Enforce that model updates must use BitTorrent"""
        if self.is_initial_download(model_id):
            return False  # Initial downloads can use any method
        
        # This is an update - must use BitTorrent
        return True
    
    async def download_initial_model(
        self,
        model_id: str,
        repo_id: str,
        filename: Optional[str] = None,
        preferred_method: Optional[DistributionMethod] = None
    ) -> DistributionResult:
        """
        Download an initial model from HuggingFace or other sources.
        This method is for first-time model acquisition.
        """
        if not self.is_initial_download(model_id):
            return DistributionResult(
                success=False,
                method_used=DistributionMethod.HUGGINGFACE,
                error_message=f"Model {model_id} already exists. Use download_model() for updates."
            )
        
        # For initial downloads, prefer HuggingFace
        method = preferred_method or DistributionMethod.HUGGINGFACE
        
        if method == DistributionMethod.HUGGINGFACE and DistributionMethod.HUGGINGFACE in self.distributors:
            start_time = time.time()
            
            try:
                success, local_path, download_time = self.hf_distributor.download_model(
                    model_id=model_id,
                    repo_id=repo_id,
                    filename=filename
                )
                
                if success:
                    file_size = Path(local_path).stat().st_size if local_path else 0
                    speed_mbps = (file_size / 1024 / 1024) / max(download_time, 0.1)
                    
                    result = DistributionResult(
                        success=True,
                        method_used=method,
                        local_path=local_path,
                        download_time=download_time,
                        download_speed_mbps=speed_mbps
                    )
                    
                    print(f"✓ Downloaded initial model {model_id} from HuggingFace")
                    print(f"  Repo: {repo_id}")
                    print(f"  Speed: {speed_mbps:.1f} MB/s")
                    
                    self.distribution_history.append(result)
                    self._update_method_performance(method, result, file_size)
                    
                    return result
                else:
                    return DistributionResult(
                        success=False,
                        method_used=method,
                        error_message="HuggingFace download failed",
                        download_time=download_time
                    )
                    
            except Exception as e:
                return DistributionResult(
                    success=False,
                    method_used=method,
                    error_message=f"HuggingFace download error: {e}",
                    download_time=time.time() - start_time
                )
        else:
            return DistributionResult(
                success=False,
                method_used=method,
                error_message=f"Method {method.value} not available for initial download"
            )