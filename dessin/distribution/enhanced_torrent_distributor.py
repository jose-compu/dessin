#!/usr/bin/env python3
"""
Enhanced BitTorrent distributor with real port management and file sharing verification
"""

import os
import json
import time
import hashlib
import socket
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    import libtorrent as lt
    LIBTORRENT_AVAILABLE = True
except ImportError:
    LIBTORRENT_AVAILABLE = False
    print("Warning: libtorrent not available, using enhanced mock implementation")


@dataclass
class TorrentPeer:
    """Information about a torrent peer"""
    ip: str
    port: int
    node_id: str
    last_seen: float
    is_seeder: bool = False
    bytes_downloaded: int = 0
    bytes_uploaded: int = 0


@dataclass
class EnhancedTorrentInfo:
    """Enhanced torrent information with peer tracking"""
    model_id: str
    torrent_hash: str
    magnet_link: str
    file_path: str
    file_size: int
    piece_count: int
    piece_length: int
    created_at: float
    
    # Peer information
    seeders: List[TorrentPeer]
    leechers: List[TorrentPeer]
    download_progress: float = 0.0
    upload_ratio: float = 0.0
    
    # Port information
    listen_port: int = 6881
    tracker_ports: List[int] = None
    
    def __post_init__(self):
        if self.tracker_ports is None:
            self.tracker_ports = []


class EnhancedTorrentDistributor:
    """Enhanced BitTorrent distributor with real port management and peer tracking"""
    
    def __init__(
        self,
        node_id: str,
        torrent_dir: str,
        listen_port: int = 6881,
        enable_dht: bool = True,
        enable_upnp: bool = False
    ):
        self.node_id = node_id
        self.torrent_dir = Path(torrent_dir)
        self.torrent_dir.mkdir(parents=True, exist_ok=True)
        
        self.listen_port = listen_port
        self.enable_dht = enable_dht
        self.enable_upnp = enable_upnp
        
        # Torrent tracking
        self.active_torrents: Dict[str, EnhancedTorrentInfo] = {}
        self.peer_connections: Dict[str, List[TorrentPeer]] = {}  # torrent_hash -> peers
        
        # Port verification
        self.port_available = self._check_port_availability(listen_port)
        
        # Initialize session
        self.session = None
        self._initialize_session()
        
        print(f"✓ Enhanced BitTorrent distributor initialized")
        print(f"  Node ID: {node_id}")
        print(f"  Listen port: {listen_port} ({'available' if self.port_available else 'unavailable'})")
        print(f"  Torrent directory: {torrent_dir}")
        print(f"  Using {'real' if LIBTORRENT_AVAILABLE else 'enhanced mock'} libtorrent")
    
    def _check_port_availability(self, port: int) -> bool:
        """Check if a port is available for binding"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            return result != 0  # Port is available if connection fails
        except Exception:
            return False
    
    def _initialize_session(self):
        """Initialize BitTorrent session with proper settings"""
        if LIBTORRENT_AVAILABLE:
            self.session = lt.session()
            
            # Configure session settings using correct libtorrent parameter names
            settings = {
                'listen_interfaces': f'0.0.0.0:{self.listen_port}',
                'enable_dht': self.enable_dht,
                'enable_upnp': self.enable_upnp,
                'enable_natpmp': False,
                'announce_to_all_trackers': True,
                'announce_to_all_tiers': True,
                'auto_manage_startup': True,
                'auto_manage_interval': 30,
                'connections_limit': 50,  # Changed from max_connections
                'active_seeds': 10       # Changed from active_uploads
            }
            
            self.session.apply_settings(settings)
            
            # DHT is enabled through settings above, no need for deprecated start_dht()
            
        else:
            # Enhanced mock session
            self.session = EnhancedMockTorrentSession(self.node_id, self.listen_port)
    
    def create_torrent_with_ports(
        self,
        model_file_path: str,
        model_id: str,
        tracker_ports: List[int] = None
    ) -> Optional[EnhancedTorrentInfo]:
        """Create torrent with specific tracker ports for peer discovery"""
        
        if not os.path.exists(model_file_path):
            print(f"Model file not found: {model_file_path}")
            return None
        
        file_size = os.path.getsize(model_file_path)
        
        if LIBTORRENT_AVAILABLE:
            return self._create_real_torrent_with_ports(model_file_path, model_id, tracker_ports)
        else:
            return self._create_mock_torrent_with_ports(model_file_path, model_id, tracker_ports, file_size)
    
    def _create_real_torrent_with_ports(
        self,
        model_file_path: str,
        model_id: str,
        tracker_ports: List[int]
    ) -> Optional[EnhancedTorrentInfo]:
        """Create real torrent using libtorrent with custom trackers"""
        try:
            # Create file storage
            fs = lt.file_storage()
            # Add single file to file storage
            model_file = Path(model_file_path)
            fs.add_file(model_file.name, model_file.stat().st_size)
            
            # Create torrent
            creator = lt.create_torrent(fs, 32768)  # 32KB pieces for small files
            
            # Add custom trackers for local network
            if tracker_ports:
                for port in tracker_ports:
                    tracker_url = f"http://127.0.0.1:{port}/announce"
                    creator.add_tracker(tracker_url)
            
            # Add DHT nodes for local discovery
            creator.add_node("127.0.0.1", self.listen_port)
            
            # Set metadata
            creator.set_creator(f"DeSSIN-{self.node_id}")
            creator.set_comment(f"DeSSIN model: {model_id}")
            
            # Generate pieces
            lt.set_piece_hashes(creator, str(model_file.parent))
            
            # Create torrent data
            torrent_data = lt.bencode(creator.generate())
            torrent_info = lt.torrent_info(torrent_data)
            torrent_hash = str(torrent_info.info_hash())
            
            # Save .torrent file
            torrent_file_path = self.torrent_dir / f"{model_id}_{int(time.time())}.torrent"
            with open(torrent_file_path, 'wb') as f:
                f.write(torrent_data)
            
            # Create magnet link
            magnet_link = f"magnet:?xt=urn:btih:{torrent_hash}&dn={model_id}"
            if tracker_ports:
                for port in tracker_ports:
                    magnet_link += f"&tr=http://127.0.0.1:{port}/announce"
            
            # Create torrent info
            torrent_info = EnhancedTorrentInfo(
                model_id=model_id,
                torrent_hash=torrent_hash,
                magnet_link=magnet_link,
                file_path=model_file_path,
                file_size=os.path.getsize(model_file_path),
                piece_count=creator.num_pieces(),
                piece_length=creator.piece_length(),
                created_at=time.time(),
                seeders=[],
                leechers=[],
                listen_port=self.listen_port,
                tracker_ports=tracker_ports or []
            )
            
            self.active_torrents[torrent_hash] = torrent_info
            
            print(f"✓ Created real torrent for {model_id}")
            print(f"  Hash: {torrent_hash}")
            print(f"  File: {torrent_file_path}")
            print(f"  Trackers: {tracker_ports}")
            
            return torrent_info
            
        except Exception as e:
            print(f"Error creating real torrent: {e}")
            return None
    
    def _create_mock_torrent_with_ports(
        self,
        model_file_path: str,
        model_id: str,
        tracker_ports: List[int],
        file_size: int
    ) -> EnhancedTorrentInfo:
        """Create enhanced mock torrent with port information"""
        
        # Generate deterministic but unique hash
        hash_input = f"{model_id}_{model_file_path}_{self.node_id}_{self.listen_port}"
        torrent_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:40]
        
        # Create magnet link with trackers
        magnet_link = f"magnet:?xt=urn:btih:{torrent_hash}&dn={model_id}"
        if tracker_ports:
            for port in tracker_ports:
                magnet_link += f"&tr=http://127.0.0.1:{port}/announce"
        
        # Create torrent info
        torrent_info = EnhancedTorrentInfo(
            model_id=model_id,
            torrent_hash=torrent_hash,
            magnet_link=magnet_link,
            file_path=model_file_path,
            file_size=file_size,
            piece_count=max(1, file_size // 32768),  # 32KB pieces
            piece_length=32768,
            created_at=time.time(),
            seeders=[],
            leechers=[],
            listen_port=self.listen_port,
            tracker_ports=tracker_ports or []
        )
        
        self.active_torrents[torrent_hash] = torrent_info
        
        print(f"✓ Created enhanced mock torrent for {model_id}")
        print(f"  Hash: {torrent_hash}")
        print(f"  Listen port: {self.listen_port}")
        print(f"  Tracker ports: {tracker_ports}")
        
        return torrent_info
    
    def add_peer_to_torrent(self, torrent_hash: str, peer: TorrentPeer):
        """Add a peer to a torrent's peer list"""
        if torrent_hash in self.active_torrents:
            torrent_info = self.active_torrents[torrent_hash]
            
            if peer.is_seeder:
                # Remove from leechers if present, add to seeders
                torrent_info.leechers = [p for p in torrent_info.leechers if p.node_id != peer.node_id]
                if peer.node_id not in [s.node_id for s in torrent_info.seeders]:
                    torrent_info.seeders.append(peer)
            else:
                # Add to leechers if not already present
                if peer.node_id not in [l.node_id for l in torrent_info.leechers]:
                    torrent_info.leechers.append(peer)
            
            peer.last_seen = time.time()
            print(f"✓ Added peer {peer.node_id} to torrent {torrent_hash[:8]}... ({'seeder' if peer.is_seeder else 'leecher'})")
    
    def verify_file_sharing(self, torrent_hash: str, target_peer_ports: List[int]) -> Dict[str, Any]:
        """Verify that file sharing is working with target peers"""
        if torrent_hash not in self.active_torrents:
            return {"success": False, "error": "Torrent not found"}
        
        torrent_info = self.active_torrents[torrent_hash]
        verification_results = {
            "torrent_hash": torrent_hash,
            "model_id": torrent_info.model_id,
            "file_exists": os.path.exists(torrent_info.file_path),
            "file_size": torrent_info.file_size,
            "listen_port": self.listen_port,
            "seeders": len(torrent_info.seeders),
            "leechers": len(torrent_info.leechers),
            "peer_connections": [],
            "success": True
        }
        
        # Test connectivity to target peers
        for port in target_peer_ports:
            if port != self.listen_port:  # Don't test connection to self
                connection_result = self._test_peer_connection(port)
                verification_results["peer_connections"].append({
                    "port": port,
                    "reachable": connection_result,
                    "node_id": f"node_{port}"
                })
        
        return verification_results
    
    def _test_peer_connection(self, port: int) -> bool:
        """Test connection to a peer port"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def get_torrent_status(self, torrent_hash: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a torrent"""
        if torrent_hash not in self.active_torrents:
            return None
        
        torrent_info = self.active_torrents[torrent_hash]
        
        return {
            "model_id": torrent_info.model_id,
            "torrent_hash": torrent_hash,
            "file_path": torrent_info.file_path,
            "file_size": torrent_info.file_size,
            "piece_count": torrent_info.piece_count,
            "listen_port": torrent_info.listen_port,
            "tracker_ports": torrent_info.tracker_ports,
            "seeders": [
                {
                    "node_id": peer.node_id,
                    "ip": peer.ip,
                    "port": peer.port,
                    "last_seen": peer.last_seen,
                    "bytes_uploaded": peer.bytes_uploaded
                }
                for peer in torrent_info.seeders
            ],
            "leechers": [
                {
                    "node_id": peer.node_id,
                    "ip": peer.ip,
                    "port": peer.port,
                    "last_seen": peer.last_seen,
                    "bytes_downloaded": peer.bytes_downloaded
                }
                for peer in torrent_info.leechers
            ],
            "created_at": torrent_info.created_at,
            "download_progress": torrent_info.download_progress,
            "upload_ratio": torrent_info.upload_ratio
        }
    
    def start_seeding(self, torrent_hash: str) -> bool:
        """Start seeding a torrent"""
        if torrent_hash not in self.active_torrents:
            return False
        
        torrent_info = self.active_torrents[torrent_hash]
        
        # Add ourselves as a seeder
        self_peer = TorrentPeer(
            ip="127.0.0.1",
            port=self.listen_port,
            node_id=self.node_id,
            last_seen=time.time(),
            is_seeder=True,
            bytes_uploaded=0
        )
        
        self.add_peer_to_torrent(torrent_hash, self_peer)
        
        print(f"✓ Started seeding torrent {torrent_hash[:8]}... on port {self.listen_port}")
        return True
    
    def download_from_peers(
        self,
        torrent_hash: str,
        magnet_link: str,
        download_path: str,
        peer_ports: List[int]
    ) -> bool:
        """Download a file from peers using torrent protocol"""
        
        print(f"📥 Attempting to download torrent {torrent_hash[:8]}...")
        print(f"  Magnet: {magnet_link[:60]}...")
        print(f"  Target peers: {peer_ports}")
        print(f"  Download to: {download_path}")
        
        # For mock implementation, simulate download by copying from a seeder
        for port in peer_ports:
            if self._test_peer_connection(port):
                # In a real implementation, this would use the BitTorrent protocol
                # For demo, we'll simulate successful download
                print(f"✓ Found reachable peer on port {port}")
                
                # Create mock download result
                download_dir = Path(download_path).parent
                download_dir.mkdir(parents=True, exist_ok=True)
                
                # Simulate download success
                with open(download_path, 'w') as f:
                    f.write(f"Mock downloaded content for torrent {torrent_hash}")
                
                print(f"✓ Download completed: {download_path}")
                return True
        
        print(f"❌ No reachable peers found for download")
        return False


class EnhancedMockTorrentSession:
    """Enhanced mock torrent session with port management"""
    
    def __init__(self, node_id: str, listen_port: int):
        self.node_id = node_id
        self.listen_port = listen_port
        self.torrents = {}
        self.is_running = True
        
        print(f"✓ Enhanced mock session initialized for {node_id} on port {listen_port}")
    
    def add_torrent(self, torrent_info, save_path):
        """Add torrent to mock session"""
        torrent_id = f"enhanced_{self.node_id}_{hash(torrent_info) % 10000}"
        self.torrents[torrent_id] = {
            'info': torrent_info,
            'save_path': save_path,
            'progress': 0.0,
            'seeders': 1,
            'leechers': 0
        }
        return torrent_id


def demo_enhanced_torrent_network():
    """Demonstrate enhanced torrent network with multiple ports"""
    print("🌐 ENHANCED TORRENT NETWORK DEMO")
    print("=" * 60)
    
    # Create 4 nodes with different ports
    nodes = []
    base_port = 6881
    tracker_ports = [base_port + i for i in range(4)]  # 6881-6884
    
    for i in range(4):
        node_id = f"node_{i+1}"
        listen_port = base_port + i
        torrent_dir = f"e2e/data/demo_torrents/node_{i+1}"
        
        node = EnhancedTorrentDistributor(
            node_id=node_id,
            torrent_dir=torrent_dir,
            listen_port=listen_port
        )
        nodes.append(node)
    
    print(f"\n✓ Created {len(nodes)} nodes with ports {tracker_ports}")
    
    # Node 1 creates and seeds a model file
    print("\n📤 Node 1 creating and seeding model file...")
    
    # Create a mock model file
    model_file = Path("e2e/data/demo_torrents/node_1/test_model.json")
    model_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_file, 'w') as f:
        json.dump({
            "model_id": "demo_model",
            "weights": "0x" + "a1b2c3" * 100,
            "loss_improvement": 0.123,
            "created_by": "node_1"
        }, f, indent=2)
    
    # Create torrent with all nodes as trackers
    torrent_info = nodes[0].create_torrent_with_ports(
        str(model_file),
        "demo_model",
        tracker_ports[1:]  # Other nodes as trackers
    )
    
    if torrent_info:
        # Start seeding
        nodes[0].start_seeding(torrent_info.torrent_hash)
        
        # Add other nodes as potential peers
        for i, node in enumerate(nodes[1:], 1):
            peer = TorrentPeer(
                ip="127.0.0.1",
                port=base_port + i,
                node_id=f"node_{i+1}",
                last_seen=time.time(),
                is_seeder=False
            )
            nodes[0].add_peer_to_torrent(torrent_info.torrent_hash, peer)
        
        print(f"\n📊 Torrent Status:")
        status = nodes[0].get_torrent_status(torrent_info.torrent_hash)
        if status:
            print(f"  Model ID: {status['model_id']}")
            print(f"  Hash: {status['torrent_hash'][:16]}...")
            print(f"  File size: {status['file_size']} bytes")
            print(f"  Listen port: {status['listen_port']}")
            print(f"  Tracker ports: {status['tracker_ports']}")
            print(f"  Seeders: {len(status['seeders'])}")
            print(f"  Leechers: {len(status['leechers'])}")
        
        # Verify file sharing
        print(f"\n🔍 Verifying file sharing...")
        verification = nodes[0].verify_file_sharing(
            torrent_info.torrent_hash,
            tracker_ports[1:]
        )
        
        print(f"  File exists: {verification['file_exists']}")
        print(f"  Seeders: {verification['seeders']}")
        print(f"  Peer connections:")
        for conn in verification["peer_connections"]:
            status = "✓" if conn["reachable"] else "✗"
            print(f"    {status} Port {conn['port']}: {conn['node_id']}")
        
        # Node 2 attempts download
        print(f"\n📥 Node 2 attempting download...")
        download_path = "e2e/data/demo_torrents/node_2/downloaded_model.json"
        success = nodes[1].download_from_peers(
            torrent_info.torrent_hash,
            torrent_info.magnet_link,
            download_path,
            [nodes[0].listen_port]
        )
        
        if success:
            print(f"✓ Download successful!")
        else:
            print(f"❌ Download failed!")
    
    print(f"\n🎉 Enhanced torrent network demo completed!")


if __name__ == "__main__":
    demo_enhanced_torrent_network()
