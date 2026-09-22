#!/usr/bin/env python3
"""
Enhanced 4-node network with BitTorrent model sharing
Demonstrates full PoGO protocol with torrent distribution
"""

import sys
import os
import time
import threading
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dessin.runtime.node import DessinNode
from dessin.runtime.config import DessinConfig
from dessin.llm.torrent_model_trainer import TorrentModelTrainer


def create_torrent_node(port: int, mining: bool = True, node_name: str = "Node") -> DessinNode:
    """Create a DessinNode configured for torrent sharing"""
    config = DessinConfig.default()
    config.network.port = port
    config.network.max_peers = 20  # Increased for better connectivity
    config.consensus.block_time_minutes = 15 / 60.0  # 15 seconds for testing
    
    # Create unique torrent directory for each node
    torrent_dir = f"model_cache/node_{port}_torrents"
    config.model.model_cache_dir = torrent_dir
    
    # Create and start node
    node = DessinNode(config)
    success = node.start()
    
    if not success:
        print(f"❌ Failed to start {node_name}")
        return None
    
    print(f"✓ {node_name} started successfully")
    print(f"  Address: {node.address}")
    print(f"  Port: {port}")
    print(f"  Mining: {'Yes' if mining else 'No'}")
    print(f"  Torrent dir: {torrent_dir}")
    
    # Create test model for mining nodes (decoder-LM variant, picked from
    # ``DESSIN_LLM_VARIANT``; default ``micro_gpt_char``). The legacy MLP
    # ``simple_neural_network`` catalog has been removed.
    if mining:
        node._create_dummy_model()
        registered_mid = next(iter(node.model_manager.models.keys()), "<unknown>")
        print(f"  Created test model: {registered_mid}")
    
    return node


def connect_nodes_full_mesh(nodes: list):
    """Connect all nodes to each other in a full mesh"""
    print("\nConnecting nodes to each other...")
    
    for i, node in enumerate(nodes):
        print(f"  Connecting Node {i+1} to other nodes...")
        for j, other_node in enumerate(nodes):
            if i != j:  # Don't connect to self
                try:
                    # Connect to other nodes
                    result = node.chaincraft_node.connect_to_peer('127.0.0.1', other_node.chaincraft_node.port)
                    # connect_to_peer returns None on success, so we check for None
                    if result is None:
                        print(f"    ✓ Node {i+1} -> Node {j+1}: Connected")
                    else:
                        print(f"    ✗ Node {i+1} -> Node {j+1}: Failed")
                    time.sleep(0.5)  # Small delay between connections
                except Exception as e:
                    print(f"    ✗ Node {i+1} -> Node {j+1}: Error - {e}")


def monitor_torrent_activity(nodes: list):
    """Monitor torrent activity across all nodes"""
    print("\n" + "="*60)
    print("TORRENT ACTIVITY MONITORING")
    print("="*60)
    
    torrent_files_seen = set()
    
    for i, node in enumerate(nodes):
        try:
            # Check torrent directory for this node
            torrent_dir = Path(node.config.model.model_cache_dir)
            if torrent_dir.exists():
                model_files = list(torrent_dir.glob("*.json"))
                
                print(f"\nNode {i+1} Torrent Directory:")
                print(f"  Directory: {torrent_dir}")
                print(f"  Model files: {len(model_files)}")
                
                for model_file in model_files:
                    file_size = model_file.stat().st_size
                    print(f"    - {model_file.name} ({file_size} bytes)")
                    torrent_files_seen.add(model_file.name)
                
                # Show recent blocks with torrent info
                if len(node.consensus.chain) > 1:
                    latest_block = node.consensus.chain[-1]
                    if hasattr(latest_block, 'torrent_hash') and latest_block.torrent_hash:
                        print(f"  Latest block torrent: {latest_block.torrent_hash[:16]}...")
                        print(f"  Model file size: {latest_block.model_size_bytes} bytes")
        
        except Exception as e:
            print(f"  Error checking Node {i+1}: {e}")
    
    print(f"\nTotal unique model files across network: {len(torrent_files_seen)}")
    return torrent_files_seen


def demonstrate_torrent_download(nodes: list):
    """Demonstrate downloading model files from other nodes"""
    print("\n" + "="*60)
    print("TORRENT DOWNLOAD DEMONSTRATION")
    print("="*60)
    
    # Find a node with model files (miner) and a node without (validator)
    miner_node = None
    validator_node = None
    
    for i, node in enumerate(nodes):
        torrent_dir = Path(node.config.model.model_cache_dir)
        if torrent_dir.exists():
            model_files = list(torrent_dir.glob("*.json"))
            if model_files and miner_node is None:
                miner_node = (i, node, model_files[0])
            elif not model_files and validator_node is None:
                validator_node = (i, node)
    
    if miner_node and validator_node:
        miner_idx, miner, model_file = miner_node
        validator_idx, validator = validator_node
        
        print(f"Attempting model download:")
        print(f"  From: Node {miner_idx + 1} (miner)")
        print(f"  To: Node {validator_idx + 1} (validator)")
        print(f"  Model file: {model_file.name}")
        
        # Create trainer for validator node
        validator_trainer = TorrentModelTrainer(validator.config.model.model_cache_dir)
        
        # Get model info from miner
        miner_trainer = TorrentModelTrainer(miner.config.model.model_cache_dir)
        model_info = miner_trainer.get_model_file_info(str(model_file))
        
        if model_info:
            print(f"  Model info: {model_info['architecture']}")
            print(f"  Loss improvement: {model_info['improvement']:.6f}")
            print(f"  File size: {model_info['file_size']} bytes")
            
            # In a real torrent implementation, we would:
            # 1. Use the torrent hash from the blockchain
            # 2. Download via BitTorrent protocol
            # 3. Verify file integrity
            
            # For demo, we'll simulate by copying the file
            try:
                import shutil
                validator_dir = Path(validator.config.model.model_cache_dir)
                validator_dir.mkdir(parents=True, exist_ok=True)
                
                dest_file = validator_dir / model_file.name
                shutil.copy2(model_file, dest_file)
                
                print(f"  ✓ Model downloaded successfully!")
                print(f"  Destination: {dest_file}")
                
                # Verify integrity
                if validator_trainer.verify_model_file_integrity(str(dest_file)):
                    print(f"  ✓ File integrity verified")
                    
                    # Load model
                    if validator_trainer.load_model_from_torrent_file(str(dest_file)):
                        print(f"  ✓ Model loaded successfully")
                
            except Exception as e:
                print(f"  ❌ Download failed: {e}")
        
    else:
        print("No suitable nodes found for download demonstration")


def main():
    """Main function to run the 4-node torrent network"""
    print("🌐 DeSSIN 4-Node Torrent Network Setup")
    print("=" * 60)
    print("This demo shows BitTorrent model distribution between nodes")
    print("=" * 60)
    
    nodes = []
    
    try:
        # Start all 4 nodes
        print("Starting nodes...")
        
        nodes.append(create_torrent_node(8001, mining=True, node_name="Node 1 (Bootstrap Miner)"))
        time.sleep(2)
        
        nodes.append(create_torrent_node(8002, mining=True, node_name="Node 2 (Miner)"))
        time.sleep(2)
        
        nodes.append(create_torrent_node(8003, mining=True, node_name="Node 3 (Miner)"))
        time.sleep(2)
        
        nodes.append(create_torrent_node(8004, mining=False, node_name="Node 4 (Validator)"))
        time.sleep(2)
        
        # Remove any None nodes (failed to start)
        nodes = [node for node in nodes if node is not None]
        
        if len(nodes) < 2:
            print("❌ Not enough nodes started successfully")
            return 1
        
        # Connect nodes
        connect_nodes_full_mesh(nodes)
        
        print("\nWaiting for connections to establish...")
        time.sleep(5)
        
        print(f"\n✓ Network running with {len(nodes)} nodes")
        
        print("\nNetwork Status:")
        print("-" * 40)
        for i, node in enumerate(nodes):
            mining_status = "Mining" if node.is_mining else "Validating"
            print(f"Node {i+1} ({mining_status}):")
            print(f"  Address: {node.address}")
            print(f"  Port: {node.chaincraft_node.port}")
            print(f"  Torrent dir: {node.config.model.model_cache_dir}")
            print()
        
        print("Network is running...")
        print("Nodes will automatically mine blocks and create torrents")
        print("Press Ctrl+C to stop all nodes")
        print("-" * 40)
        
        # Monitor network activity
        start_time = time.time()
        monitoring_interval = 20  # seconds
        
        while True:
            time.sleep(monitoring_interval)
            
            # Print periodic status
            elapsed = time.time() - start_time
            print(f"\n[Network Status - {elapsed:.0f}s elapsed]")
            
            for i, node in enumerate(nodes):
                try:
                    # Get basic node info
                    info = node.get_node_info()
                    chain_info = node.get_chain_info()
                    
                    print(f"Node {i+1}: Chain height {chain_info.get('height', 0)}, "
                          f"Peers: {len(node.chaincraft_node.peers) if hasattr(node.chaincraft_node, 'peers') else 'Unknown'}")
                    
                except Exception as e:
                    print(f"Node {i+1}: Error getting status - {e}")
            
            # Monitor torrent activity
            torrent_files = monitor_torrent_activity(nodes)
            
            # Demonstrate torrent download if we have files
            if torrent_files and elapsed > 30:  # After 30 seconds
                demonstrate_torrent_download(nodes)
            
            print("-" * 40)
            
    except KeyboardInterrupt:
        print("\n\nShutting down torrent network...")
        
        # Stop all nodes
        for i, node in enumerate(nodes):
            try:
                print(f"Stopping Node {i+1}...")
                node.stop()
            except Exception as e:
                print(f"Error stopping Node {i+1}: {e}")
        
        print("✓ Torrent network stopped successfully")
        return 0
    
    except Exception as e:
        print(f"Error running torrent network: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
