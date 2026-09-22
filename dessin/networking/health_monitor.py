"""
Health monitoring and peer availability tracking for DeSSIN BitTorrent distribution.
"""

import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from ..distribution.bittorrent_distributor import BitTorrentDistributor, TorrentInfo
from ..distribution.hybrid_distributor import HybridDistributor, DistributionMethod


@dataclass
class PeerHealth:
    """Health information for a peer"""
    peer_id: str
    last_seen: float
    upload_rate: float  # bytes/sec
    download_rate: float  # bytes/sec
    connection_count: int
    ratio: float  # upload/download ratio
    reliability_score: float  # 0-1 based on historical performance
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelHealth:
    """Health information for a model's distribution"""
    model_id: str
    total_seeders: int
    total_leechers: int
    availability_score: float  # 0-1, based on peer distribution
    average_download_speed: float  # MB/s
    success_rate: float  # successful downloads / total attempts
    last_successful_download: float
    geographic_distribution: int  # number of different regions/ASNs
    network_score: float  # overall network health 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NetworkHealth:
    """Overall network health metrics"""
    total_active_peers: int
    total_models: int
    models_with_good_health: int  # >0.7 network score
    average_availability: float
    average_download_speed: float
    total_bandwidth_utilization: float  # MB/s
    network_congestion_score: float  # 0-1, lower is better
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HealthMonitor:
    """
    Monitors health of BitTorrent distribution network and individual models.
    Provides insights into peer availability, download performance, and network congestion.
    """
    
    def __init__(
        self,
        distributor: HybridDistributor,
        check_interval: float = 30.0,  # seconds
        history_retention: int = 1000   # number of snapshots to keep
    ):
        self.distributor = distributor
        self.check_interval = check_interval
        self.history_retention = history_retention
        
        # Health tracking
        self.peer_health: Dict[str, PeerHealth] = {}
        self.model_health: Dict[str, ModelHealth] = {}
        self.network_history: List[NetworkHealth] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Callbacks for health events
        self.health_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # Thresholds for alerts
        self.min_seeders_threshold = 2
        self.min_availability_threshold = 0.3
        self.max_congestion_threshold = 0.8
        
        print("✓ Health monitor initialized")
        print(f"  Check interval: {check_interval}s")
        print(f"  History retention: {history_retention} snapshots")
    
    def start_monitoring(self):
        """Start background health monitoring"""
        if self.monitoring_active:
            print("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print("✓ Health monitoring started")
    
    def stop_monitoring(self):
        """Stop background health monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        print("✓ Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._update_health_metrics()
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in health monitoring: {e}")
                time.sleep(self.check_interval)
    
    def _update_health_metrics(self):
        """Update all health metrics"""
        try:
            # Update model health
            self._update_model_health()
            
            # Update peer health  
            self._update_peer_health()
            
            # Update network health
            network_health = self._calculate_network_health()
            self.network_history.append(network_health)
            
            # Trim history
            if len(self.network_history) > self.history_retention:
                self.network_history = self.network_history[-self.history_retention:]
            
            # Check for alerts
            self._check_health_alerts(network_health)
            
            # Notify callbacks
            for callback in self.health_callbacks:
                try:
                    callback(network_health)
                except Exception as e:
                    print(f"Error in health callback: {e}")
                    
        except Exception as e:
            print(f"Error updating health metrics: {e}")
    
    def _update_model_health(self):
        """Update health metrics for all models"""
        if not hasattr(self.distributor, 'bt_distributor') or not self.distributor.bt_distributor:
            return
        
        bt_distributor = self.distributor.bt_distributor
        
        for model_id, torrent_info in bt_distributor.model_torrents.items():
            progress = bt_distributor.get_download_progress(model_id)
            
            if not progress:
                continue
            
            # Calculate metrics
            seeders = progress.get("seeders", 0)
            leechers = progress.get("leechers", 0)
            total_peers = seeders + leechers
            
            # Availability score based on peer count and distribution
            availability_score = min(1.0, total_peers / max(5, 1))  # Scale based on 5+ peers = full availability
            
            # Success rate (simplified - would track actual download attempts in production)
            success_rate = 0.9 if seeders > 0 else 0.1
            
            # Network score - combination of availability and peer quality
            network_score = (availability_score + success_rate) / 2
            
            self.model_health[model_id] = ModelHealth(
                model_id=model_id,
                total_seeders=seeders,
                total_leechers=leechers,
                availability_score=availability_score,
                average_download_speed=progress.get("download_rate", 0) / 1024 / 1024,  # MB/s
                success_rate=success_rate,
                last_successful_download=time.time() if progress.get("progress", 0) > 0 else 0,
                geographic_distribution=min(total_peers, 3),  # Simplified geographic estimate
                network_score=network_score
            )
    
    def _update_peer_health(self):
        """Update health metrics for peers"""
        if not hasattr(self.distributor, 'bt_distributor') or not self.distributor.bt_distributor:
            return
        
        bt_distributor = self.distributor.bt_distributor
        
        # Get peer information from active torrents
        for model_id, torrent_info in bt_distributor.model_torrents.items():
            progress = bt_distributor.get_download_progress(model_id)
            
            if progress:
                # Simplified peer tracking - in production this would track individual peers
                peer_id = f"peer_{model_id}_{progress.get('seeders', 0)}"
                
                if peer_id not in self.peer_health:
                    self.peer_health[peer_id] = PeerHealth(
                        peer_id=peer_id,
                        last_seen=time.time(),
                        upload_rate=progress.get("upload_rate", 0),
                        download_rate=progress.get("download_rate", 0),
                        connection_count=progress.get("seeders", 0) + progress.get("leechers", 0),
                        ratio=1.0,  # Default good ratio
                        reliability_score=0.8  # Default good reliability
                    )
                else:
                    # Update existing peer
                    peer = self.peer_health[peer_id]
                    peer.last_seen = time.time()
                    peer.upload_rate = progress.get("upload_rate", 0)
                    peer.download_rate = progress.get("download_rate", 0)
                    peer.connection_count = progress.get("seeders", 0) + progress.get("leechers", 0)
    
    def _calculate_network_health(self) -> NetworkHealth:
        """Calculate overall network health"""
        timestamp = time.time()
        
        # Count active peers
        active_peers = len([p for p in self.peer_health.values() 
                           if timestamp - p.last_seen < 300])  # 5 minutes
        
        # Model statistics
        total_models = len(self.model_health)
        models_with_good_health = len([m for m in self.model_health.values() 
                                     if m.network_score > 0.7])
        
        # Average metrics
        if self.model_health:
            avg_availability = sum(m.availability_score for m in self.model_health.values()) / total_models
            avg_download_speed = sum(m.average_download_speed for m in self.model_health.values()) / total_models
        else:
            avg_availability = 0.0
            avg_download_speed = 0.0
        
        # Bandwidth utilization (sum of all active transfers)
        total_bandwidth = sum(p.upload_rate + p.download_rate for p in self.peer_health.values()) / 1024 / 1024
        
        # Network congestion (simplified metric)
        # In production, this would consider factors like:
        # - Average download times vs expected
        # - Connection failure rates
        # - Peer connection success rates
        congestion_score = min(1.0, total_bandwidth / max(100, 1))  # Assume 100 MB/s is congested
        
        return NetworkHealth(
            total_active_peers=active_peers,
            total_models=total_models,
            models_with_good_health=models_with_good_health,
            average_availability=avg_availability,
            average_download_speed=avg_download_speed,
            total_bandwidth_utilization=total_bandwidth,
            network_congestion_score=congestion_score,
            timestamp=timestamp
        )
    
    def _check_health_alerts(self, network_health: NetworkHealth):
        """Check for health alerts and notify callbacks"""
        alerts = []
        
        # Check for models with low availability
        for model_id, health in self.model_health.items():
            if health.total_seeders < self.min_seeders_threshold:
                alerts.append({
                    "type": "low_seeders",
                    "model_id": model_id,
                    "seeders": health.total_seeders,
                    "threshold": self.min_seeders_threshold,
                    "severity": "warning"
                })
            
            if health.availability_score < self.min_availability_threshold:
                alerts.append({
                    "type": "low_availability",
                    "model_id": model_id,
                    "availability": health.availability_score,
                    "threshold": self.min_availability_threshold,
                    "severity": "critical"
                })
        
        # Check network congestion
        if network_health.network_congestion_score > self.max_congestion_threshold:
            alerts.append({
                "type": "network_congestion",
                "congestion_score": network_health.network_congestion_score,
                "threshold": self.max_congestion_threshold,
                "severity": "warning"
            })
        
        # Notify alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"Error in alert callback: {e}")
    
    def get_model_health(self, model_id: str) -> Optional[ModelHealth]:
        """Get health information for a specific model"""
        return self.model_health.get(model_id)
    
    def get_all_model_health(self) -> Dict[str, ModelHealth]:
        """Get health information for all models"""
        return self.model_health.copy()
    
    def get_peer_health(self, peer_id: str) -> Optional[PeerHealth]:
        """Get health information for a specific peer"""
        return self.peer_health.get(peer_id)
    
    def get_all_peer_health(self) -> Dict[str, PeerHealth]:
        """Get health information for all peers"""
        return self.peer_health.copy()
    
    def get_current_network_health(self) -> Optional[NetworkHealth]:
        """Get current network health snapshot"""
        return self.network_history[-1] if self.network_history else None
    
    def get_network_health_history(self, hours: float = 24.0) -> List[NetworkHealth]:
        """Get network health history for specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        return [h for h in self.network_history if h.timestamp >= cutoff_time]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        current_health = self.get_current_network_health()
        
        summary = {
            "timestamp": time.time(),
            "monitoring_active": self.monitoring_active,
            "network_health": current_health.to_dict() if current_health else None,
            "model_health_summary": {
                "total_models": len(self.model_health),
                "healthy_models": len([m for m in self.model_health.values() if m.network_score > 0.7]),
                "warning_models": len([m for m in self.model_health.values() if 0.3 <= m.network_score <= 0.7]),
                "critical_models": len([m for m in self.model_health.values() if m.network_score < 0.3])
            },
            "peer_health_summary": {
                "total_peers": len(self.peer_health),
                "active_peers": len([p for p in self.peer_health.values() 
                                   if time.time() - p.last_seen < 300]),
                "reliable_peers": len([p for p in self.peer_health.values() if p.reliability_score > 0.8])
            }
        }
        
        return summary
    
    def add_health_callback(self, callback: Callable[[NetworkHealth], None]):
        """Add callback for health updates"""
        self.health_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for health alerts"""
        self.alert_callbacks.append(callback)
    
    def export_health_data(self, file_path: str):
        """Export health data to JSON file"""
        try:
            data = {
                "timestamp": time.time(),
                "model_health": {k: v.to_dict() for k, v in self.model_health.items()},
                "peer_health": {k: v.to_dict() for k, v in self.peer_health.items()},
                "network_history": [h.to_dict() for h in self.network_history[-100:]],  # Last 100 snapshots
                "summary": self.get_health_summary()
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"✓ Health data exported to {file_path}")
            
        except Exception as e:
            print(f"Error exporting health data: {e}")
    
    def print_health_report(self):
        """Print a comprehensive health report"""
        summary = self.get_health_summary()
        current_health = self.get_current_network_health()
        
        print("\n" + "="*60)
        print("DESSIN NETWORK HEALTH REPORT")
        print("="*60)
        
        if current_health:
            print(f"\n📊 Network Overview:")
            print(f"  Active Peers: {current_health.total_active_peers}")
            print(f"  Total Models: {current_health.total_models}")
            print(f"  Healthy Models: {current_health.models_with_good_health}")
            print(f"  Average Availability: {current_health.average_availability:.1%}")
            print(f"  Average Download Speed: {current_health.average_download_speed:.1f} MB/s")
            print(f"  Network Utilization: {current_health.total_bandwidth_utilization:.1f} MB/s")
            print(f"  Congestion Score: {current_health.network_congestion_score:.1%}")
        
        model_summary = summary["model_health_summary"]
        print(f"\n🧠 Model Health:")
        print(f"  Total: {model_summary['total_models']}")
        print(f"  Healthy: {model_summary['healthy_models']} 🟢")
        print(f"  Warning: {model_summary['warning_models']} 🟡")
        print(f"  Critical: {model_summary['critical_models']} 🔴")
        
        peer_summary = summary["peer_health_summary"]
        print(f"\n👥 Peer Health:")
        print(f"  Total: {peer_summary['total_peers']}")
        print(f"  Active: {peer_summary['active_peers']}")
        print(f"  Reliable: {peer_summary['reliable_peers']}")
        
        # Top models by health
        if self.model_health:
            print(f"\n🏆 Top Models by Health:")
            sorted_models = sorted(self.model_health.values(), 
                                 key=lambda m: m.network_score, reverse=True)
            for i, model in enumerate(sorted_models[:5]):
                status = "🟢" if model.network_score > 0.7 else "🟡" if model.network_score > 0.3 else "🔴"
                print(f"  {i+1}. {model.model_id}: {model.network_score:.1%} {status}")
                print(f"     Seeders: {model.total_seeders}, Availability: {model.availability_score:.1%}")
        
        print("="*60)
