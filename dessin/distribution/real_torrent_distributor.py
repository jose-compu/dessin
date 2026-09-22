#!/usr/bin/env python3
"""
Real BitTorrent distributor using libtorrent for actual torrent functionality
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

import libtorrent as lt


@dataclass
class RealTorrentInfo:
    """Real torrent information with libtorrent integration"""
    model_id: str
    torrent_hash: str
    magnet_link: str
    torrent_file_path: str
    model_file_path: str
    file_size: int
    piece_count: int
    piece_length: int
    created_at: float
    
    # Session tracking
    torrent_handle: Optional[Any] = None
    is_seeding: bool = False
    download_progress: float = 0.0
    num_peers: int = 0
    upload_rate: int = 0
    download_rate: int = 0


class RealTorrentDistributor:
    """Real BitTorrent distributor using libtorrent"""
    
    def __init__(
        self,
        node_id: str,
        torrent_dir: str,
        listen_port: int = 6881,
        enable_dht: bool = True
    ):
        self.node_id = node_id
        self.torrent_dir = Path(torrent_dir)
        self.torrent_dir.mkdir(parents=True, exist_ok=True)
        
        self.listen_port = listen_port
        self.enable_dht = enable_dht
        
        # Create libtorrent session
        self.session = lt.session()
        self._configure_session()
        
        # Tracking
        self.active_torrents: Dict[str, RealTorrentInfo] = {}
        self.torrent_handles: Dict[str, Any] = {}
        
        # Status monitoring
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_torrents, daemon=True)
        self._monitor_thread.start()
        
        print(f"✓ Real BitTorrent distributor initialized")
        print(f"  Node ID: {node_id}")
        print(f"  Listen port: {listen_port}")
        print(f"  Torrent directory: {torrent_dir}")
        print(f"  Using libtorrent {lt.version}")
    
    def _configure_session(self):
        """Configure libtorrent session settings"""
        try:
            # Modern libtorrent configuration using settings
            settings = {
                'listen_interfaces': f'0.0.0.0:{self.listen_port}',
                'enable_dht': self.enable_dht,
                'enable_lsd': True,  # Local Service Discovery
                'enable_upnp': False,
                'enable_natpmp': False
            }
            
            self.session.apply_settings(settings)
            
            if self.enable_dht:
                print(f"  DHT enabled on port {self.listen_port}")
            
            print(f"  Local service discovery enabled")
            
        except Exception as e:
            print(f"Warning: Could not configure all session settings: {e}")
            print(f"  Session will use default settings")
    
    def create_real_torrent(
        self,
        model_file_path: str,
        model_id: str,
        trackers: List[str] = None
    ) -> Optional[RealTorrentInfo]:
        """Create a real torrent file using libtorrent"""
        
        if not os.path.exists(model_file_path):
            print(f"Model file not found: {model_file_path}")
            return None
        
        try:
            # Create file storage
            fs = lt.file_storage()
            # Add single file to file storage
            model_file = Path(model_file_path)
            fs.add_file(model_file.name, model_file.stat().st_size)
            
            # Create torrent creator
            piece_size = 32 * 1024  # 32KB pieces for small files
            creator = lt.create_torrent(fs, piece_size)
            
            # Add trackers
            if trackers:
                for tracker in trackers:
                    creator.add_tracker(tracker)
            else:
                # Add local tracker URLs for other nodes
                for port in [6881, 6882, 6883, 6884]:
                    if port != self.listen_port:
                        creator.add_tracker(f"http://127.0.0.1:{port}/announce")
            
            # Set metadata
            creator.set_creator(f"DeSSIN-{self.node_id}")
            creator.set_comment(f"DeSSIN model: {model_id}")
            
            # Generate piece hashes
            lt.set_piece_hashes(creator, str(model_file.parent))
            
            # Create torrent data
            torrent_data = lt.bencode(creator.generate())
            
            # Get torrent info
            torrent_info = lt.torrent_info(torrent_data)
            torrent_hash = str(torrent_info.info_hash())
            
            # Save .torrent file
            torrent_file_path = self.torrent_dir / f"{model_id}_{int(time.time())}.torrent"
            with open(torrent_file_path, 'wb') as f:
                f.write(torrent_data)
            
            # Create magnet link
            magnet_link = f"magnet:?xt=urn:btih:{torrent_hash}&dn={model_id}"
            if trackers:
                for tracker in trackers:
                    magnet_link += f"&tr={tracker}"
            
            # Create torrent info object
            real_torrent_info = RealTorrentInfo(
                model_id=model_id,
                torrent_hash=torrent_hash,
                magnet_link=magnet_link,
                torrent_file_path=str(torrent_file_path),
                model_file_path=model_file_path,
                file_size=os.path.getsize(model_file_path),
                piece_count=torrent_info.num_pieces(),
                piece_length=torrent_info.piece_length(),
                created_at=time.time()
            )
            
            self.active_torrents[torrent_hash] = real_torrent_info
            
            print(f"✓ Created real torrent for {model_id}")
            print(f"  Hash: {torrent_hash}")
            print(f"  File: {torrent_file_path}")
            print(f"  Pieces: {real_torrent_info.piece_count}")
            print(f"  Size: {real_torrent_info.file_size} bytes")
            
            return real_torrent_info
            
        except Exception as e:
            print(f"Error creating real torrent: {e}")
            return None
    
    def start_seeding(self, torrent_hash: str) -> bool:
        """Start seeding a torrent"""
        if torrent_hash not in self.active_torrents:
            print(f"Torrent not found: {torrent_hash}")
            return False
        
        torrent_info = self.active_torrents[torrent_hash]
        
        try:
            # Create add_torrent_params
            params = lt.add_torrent_params()
            params.ti = lt.torrent_info(torrent_info.torrent_file_path)
            params.save_path = os.path.dirname(torrent_info.model_file_path)
            params.seed_mode = True  # We have the complete file
            params.flags |= lt.torrent_flags.duplicate_is_error
            
            # Add torrent to session
            handle = self.session.add_torrent(params)
            
            # Store handle
            self.torrent_handles[torrent_hash] = handle
            torrent_info.torrent_handle = handle
            torrent_info.is_seeding = True
            
            print(f"✓ Started seeding torrent {torrent_hash[:8]}... on port {self.listen_port}")
            return True
            
        except Exception as e:
            print(f"Error starting seeding: {e}")
            return False
    
    def download_torrent(
        self,
        torrent_hash: str,
        magnet_link: str,
        download_dir: str
    ) -> bool:
        """Download a torrent from peers"""
        
        try:
            # Create download directory
            download_path = Path(download_dir)
            download_path.mkdir(parents=True, exist_ok=True)
            
            # Create add_torrent_params
            params = lt.add_torrent_params()
            params.url = magnet_link
            params.save_path = str(download_path)
            params.flags |= lt.torrent_flags.duplicate_is_error
            
            # Add torrent to session
            handle = self.session.add_torrent(params)
            
            # Store handle
            self.torrent_handles[torrent_hash] = handle
            
            print(f"✓ Started downloading torrent {torrent_hash[:8]}...")
            print(f"  Magnet: {magnet_link[:60]}...")
            print(f"  Download to: {download_path}")
            
            # Wait for download to complete (with timeout)
            timeout = 60  # 60 seconds timeout
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                status = handle.status()
                
                if status.is_seeding:
                    print(f"✓ Download completed: {torrent_hash[:8]}...")
                    return True
                
                if status.error:
                    print(f"❌ Download error: {status.error}")
                    return False
                
                # Show progress
                if status.total_wanted > 0:
                    progress = status.progress * 100
                    print(f"  Download progress: {progress:.1f}% ({status.num_peers} peers)")
                
                time.sleep(2)
            
            print(f"❌ Download timeout for {torrent_hash[:8]}...")
            return False
            
        except Exception as e:
            print(f"Error downloading torrent: {e}")
            return False
    
    def _monitor_torrents(self):
        """Monitor torrent status in background thread"""
        while self._running:
            try:
                for torrent_hash, torrent_info in self.active_torrents.items():
                    if torrent_info.torrent_handle:
                        status = torrent_info.torrent_handle.status()
                        
                        # Update status
                        torrent_info.download_progress = status.progress
                        torrent_info.num_peers = status.num_peers
                        torrent_info.upload_rate = status.upload_rate
                        torrent_info.download_rate = status.download_rate
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                print(f"Error monitoring torrents: {e}")
                time.sleep(10)
    
    def get_torrent_status(self, torrent_hash: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a torrent"""
        if torrent_hash not in self.active_torrents:
            return None
        
        torrent_info = self.active_torrents[torrent_hash]
        
        status_dict = {
            "model_id": torrent_info.model_id,
            "torrent_hash": torrent_hash,
            "file_size": torrent_info.file_size,
            "piece_count": torrent_info.piece_count,
            "piece_length": torrent_info.piece_length,
            "is_seeding": torrent_info.is_seeding,
            "download_progress": torrent_info.download_progress,
            "num_peers": torrent_info.num_peers,
            "upload_rate": torrent_info.upload_rate,
            "download_rate": torrent_info.download_rate,
            "created_at": torrent_info.created_at
        }
        
        # Add libtorrent status if available
        if torrent_info.torrent_handle:
            try:
                lt_status = torrent_info.torrent_handle.status()
                status_dict.update({
                    "state": str(lt_status.state),
                    "total_download": lt_status.total_download,
                    "total_upload": lt_status.total_upload,
                    "num_seeds": lt_status.num_seeds,
                    "num_complete": lt_status.num_complete,
                    "num_incomplete": lt_status.num_incomplete
                })
            except Exception as e:
                status_dict["lt_error"] = str(e)
        
        return status_dict
    
    def list_active_torrents(self) -> List[Dict[str, Any]]:
        """List all active torrents with their status"""
        torrents = []
        
        for torrent_hash in self.active_torrents:
            status = self.get_torrent_status(torrent_hash)
            if status:
                torrents.append(status)
        
        return torrents
    
    def stop_torrent(self, torrent_hash: str):
        """Stop a torrent (seeding or downloading)"""
        if torrent_hash in self.torrent_handles:
            try:
                handle = self.torrent_handles[torrent_hash]
                self.session.remove_torrent(handle)
                del self.torrent_handles[torrent_hash]
                
                if torrent_hash in self.active_torrents:
                    self.active_torrents[torrent_hash].is_seeding = False
                    self.active_torrents[torrent_hash].torrent_handle = None
                
                print(f"✓ Stopped torrent {torrent_hash[:8]}...")
                
            except Exception as e:
                print(f"Error stopping torrent: {e}")
    
    def close(self):
        """Close the BitTorrent session"""
        self._running = False
        
        # Stop all torrents
        for torrent_hash in list(self.torrent_handles.keys()):
            self.stop_torrent(torrent_hash)
        
        # Close session
        if hasattr(self.session, 'pause'):
            self.session.pause()
        
        print(f"✓ Real BitTorrent distributor closed")


def demo_real_torrent_network():
    """Demonstrate real torrent functionality with multiple nodes"""
    print("🌐 REAL TORRENT NETWORK DEMO")
    print("=" * 60)
    
    # Create 2 nodes for demo
    node1_dir = "demo_real_torrents/node1"
    node2_dir = "demo_real_torrents/node2"
    
    # Create directories
    Path(node1_dir).mkdir(parents=True, exist_ok=True)
    Path(node2_dir).mkdir(parents=True, exist_ok=True)
    
    # Create distributors
    distributor1 = RealTorrentDistributor("node1", node1_dir, 6881)
    time.sleep(2)  # Let first node start
    
    distributor2 = RealTorrentDistributor("node2", node2_dir, 6882)
    time.sleep(2)  # Let second node start
    
    try:
        # Node 1 creates and seeds a model file
        print("\n📤 Node 1 creating and seeding model file...")
        
        model_file = Path(node1_dir) / "test_model.json"
        with open(model_file, 'w') as f:
            json.dump({
                "model_id": "real_demo_model",
                "weights": "0x" + "deadbeef" * 100,
                "loss_improvement": 0.234,
                "created_by": "node1",
                "timestamp": time.time()
            }, f, indent=2)
        
        # Create real torrent
        torrent_info = distributor1.create_real_torrent(
            str(model_file),
            "real_demo_model"
        )
        
        if torrent_info:
            # Start seeding
            distributor1.start_seeding(torrent_info.torrent_hash)
            
            print(f"\n📊 Torrent Status (Node 1):")
            status = distributor1.get_torrent_status(torrent_info.torrent_hash)
            if status:
                print(f"  Model ID: {status['model_id']}")
                print(f"  Hash: {status['torrent_hash'][:16]}...")
                print(f"  File size: {status['file_size']} bytes")
                print(f"  Pieces: {status['piece_count']}")
                print(f"  Seeding: {status['is_seeding']}")
            
            # Wait a moment for seeding to start
            time.sleep(5)
            
            # Node 2 attempts download
            print(f"\n📥 Node 2 attempting download...")
            download_dir = Path(node2_dir) / "downloads"
            
            success = distributor2.download_torrent(
                torrent_info.torrent_hash,
                torrent_info.magnet_link,
                str(download_dir)
            )
            
            if success:
                print(f"✓ Download successful!")
                
                # Verify downloaded file
                downloaded_files = list(download_dir.glob("*.json"))
                if downloaded_files:
                    print(f"✓ Found downloaded file: {downloaded_files[0]}")
                    
                    # Compare file contents
                    with open(downloaded_files[0], 'r') as f:
                        downloaded_data = json.load(f)
                    
                    with open(model_file, 'r') as f:
                        original_data = json.load(f)
                    
                    if downloaded_data == original_data:
                        print(f"✓ File integrity verified!")
                    else:
                        print(f"❌ File integrity check failed")
                else:
                    print(f"❌ No downloaded files found")
            else:
                print(f"❌ Download failed!")
            
            # Show final status
            print(f"\n📊 Final Network Status:")
            
            print(f"Node 1 torrents:")
            for torrent in distributor1.list_active_torrents():
                print(f"  {torrent['torrent_hash'][:8]}... - {torrent['model_id']} ({torrent['num_peers']} peers)")
            
            print(f"Node 2 torrents:")
            for torrent in distributor2.list_active_torrents():
                print(f"  {torrent['torrent_hash'][:8]}... - {torrent['model_id']} ({torrent['num_peers']} peers)")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        
    except Exception as e:
        print(f"\nDemo error: {e}")
        
    finally:
        # Clean up
        print(f"\n🧹 Cleaning up...")
        distributor1.close()
        distributor2.close()
        time.sleep(2)
        
        print(f"🎉 Real torrent network demo completed!")


if __name__ == "__main__":
    demo_real_torrent_network()
