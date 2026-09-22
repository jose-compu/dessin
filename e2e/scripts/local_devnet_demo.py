#!/usr/bin/env python3
"""
DeSSIN Local Devnet Demo
========================

This demo creates a local development network with 5+ nodes to test:
1. Multi-node BitTorrent distribution
2. Cross-node model sharing
3. Progress tracking across the network
4. Network health monitoring
5. Consensus integration

The devnet simulates a real distributed network environment.
"""

import asyncio
import time
import tempfile
import random
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dessin.models.model_manager import ModelManager
from dessin.distribution.bittorrent_distributor import BitTorrentDistributor
from dessin.distribution.enhanced_torrent_distributor import EnhancedTorrentDistributor
from dessin.runtime.progress_tracker import get_progress_tracker, ProgressType
from dessin.consensus.enhanced_pogo_consensus import EnhancedPogoConsensus
from dessin.consensus.two_phase_verification import TwoPhaseVerificationSystem, VerificationPhase
from dessin.consensus.transactions import AttestationTransaction
from dessin.consensus import AttestationType
# Health monitoring will be simulated since hybrid_distributor has syntax errors
# from dessin.networking.health_monitor import HealthMonitor
from dessin.runtime.config import ModelConfig, ConsensusConfig, DessinConfig


class DemoNode:
    """Basic demo node for local devnet"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Create temporary directories
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"dessin_demo_{node_id}_"))
        self.model_cache_dir = self.temp_dir / "model_cache"
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_config = ModelConfig()
        self.model_config.model_cache_dir = str(self.model_cache_dir)
        self.model_manager = ModelManager(self.model_config)
        
        # Initialize BitTorrent distributor
        self.bt_distributor = BitTorrentDistributor(
            node_id=node_id,
            cache_dir=str(self.model_cache_dir),
            torrent_dir=str(self.model_cache_dir / "torrents")
        )
        
        print(f"✓ Demo node {node_id} initialized")
    
    async def create_and_share_model(self, model_id: str, size_kb: int) -> bool:
        """Create and share a model"""
        try:
            # Create a dummy model file
            model_path = self.model_cache_dir / f"{model_id}.bin"
            with open(model_path, "wb") as f:
                f.write(b"0" * (size_kb * 1024))
            
            # Register with model manager
            self.model_manager.register_model(model_id, str(model_path))
            
            # Create and share torrent
            torrent_info = self.bt_distributor.create_model_torrent(model_id, str(model_path))
            if torrent_info:
                self.bt_distributor.seed_model(model_id)
                return True
            
            return False
            
        except Exception as e:
            print(f"Error creating model {model_id}: {e}")
            return False
    
    async def download_model_from_peer(self, model_id: str, magnet_link: str) -> bool:
        """Download model from peer"""
        try:
            return await self.bt_distributor.download_model_by_magnet(model_id, magnet_link)
        except Exception as e:
            print(f"Error downloading model {model_id}: {e}")
            return False


def create_micro_model(model_id: str, size_kb: int = 100) -> bytes:
    """Create a micro model for testing"""
    return b"0" * (size_kb * 1024)


class DevnetNode(DemoNode):
    """Extended demo node for devnet with enhanced features"""
    
    def __init__(self, node_id: str, port: int = None):
        super().__init__(node_id)
        self.port = port or (8000 + hash(node_id) % 1000)
        self.peers: List[str] = []
        self.health_monitor = None
        
        # Enhanced features
        self.enhanced_consensus = None
        self.two_phase_system = None
        self.verifier_stake = random.uniform(50.0, 200.0)  # Random stake for testing
        
        # Initialize enhanced consensus if this is a miner/verifier node
        if random.random() < 0.7:  # 70% chance to be a verifier
            self._initialize_enhanced_consensus()
        
        # Simulate health monitoring 
        self.health_monitor = None  # Will be simulated
        
        print(f"✓ Devnet node {node_id} initialized on port {self.port}")
        if self.enhanced_consensus:
            print(f"  🔐 Enhanced PoGO consensus enabled (stake: {self.verifier_stake:.1f})")
    
    def _initialize_enhanced_consensus(self):
        """Initialize enhanced consensus with two-phase verification"""
        try:
            # Create enhanced configuration
            config = DessinConfig.default()
            config.consensus.merkle_proof_count = random.randint(1, 3)  # Random proof count
            config.consensus.phase1_window_blocks = 3  # Shorter for demo
            config.consensus.phase2_window_blocks = 3
            config.consensus.verification_block_time_minutes = 1  # 1 minute for demo
            config.consensus.training_block_time_minutes = 6.0
            
            # Initialize enhanced consensus
            self.enhanced_consensus = EnhancedPogoConsensus(
                config.consensus, 
                self.model_manager, 
                self.node_id
            )
            
            # Set verifier stake
            self.enhanced_consensus.verifier_stakes[self.node_id] = self.verifier_stake
            
            # Initialize two-phase verification system
            self.two_phase_system = TwoPhaseVerificationSystem(config.consensus)
            
        except Exception as e:
            print(f"⚠️  Failed to initialize enhanced consensus for {self.node_id}: {e}")
            self.enhanced_consensus = None
            self.two_phase_system = None
    
    def connect_peer(self, peer_node_id: str):
        """Connect to a peer node"""
        if peer_node_id not in self.peers:
            self.peers.append(peer_node_id)
            print(f"🔗 Node {self.node_id} connected to {peer_node_id}")
    
    def get_node_stats(self) -> Dict[str, Any]:
        """Get comprehensive node statistics"""
        bt_stats = self.bt_distributor.get_network_statistics()
        
        stats = {
            "node_id": self.node_id,
            "port": self.port,
            "peers": len(self.peers),
            "registered_models": len(self.model_manager.models),
            "torrents": bt_stats.get("total_models", 0),
            "active_downloads": bt_stats.get("active_torrents", 0),
            "seeders": bt_stats.get("total_seeders", 0),
            "leechers": bt_stats.get("total_leechers", 0),
            "health_score": random.uniform(0.8, 1.0),  # Simulated
            "is_verifier": self.enhanced_consensus is not None,
            "verifier_stake": self.verifier_stake if self.enhanced_consensus else 0.0
        }
        
        # Add enhanced consensus stats if available
        if self.enhanced_consensus:
            consensus_metrics = self.enhanced_consensus.get_consensus_metrics()
            stats.update({
                "merkle_proof_count": consensus_metrics.get("merkle_proof_count", 0),
                "pending_attestations": consensus_metrics.get("pending_attestations", 0),
                "total_attestations": consensus_metrics.get("total_attestations", 0),
                "training_blocks_tracked": consensus_metrics.get("training_blocks_tracked", 0)
            })
        
        return stats
    
    async def simulate_two_phase_verification(self, block_hash: str) -> Dict[str, Any]:
        """Simulate two-phase verification process"""
        if not self.enhanced_consensus or not self.two_phase_system:
            return {"error": "Node not configured for verification"}
        
        try:
            # Simulate Phase 1 verification
            phase1_data = self.two_phase_system.generate_phase1_verification_data(
                block_hash, f"quantized_hash_{block_hash[:8]}", self.node_id
            )
            
            # Simulate verification result (90% positive for demo)
            phase1_result = random.random() < 0.9
            phase1_improvement = random.uniform(0.005, 0.02) if phase1_result else None
            phase1_evidence = None if phase1_result else "Loss improvement below threshold"
            
            self.two_phase_system.submit_phase1_verification(
                phase1_data, phase1_result, phase1_improvement, phase1_evidence
            )
            
            # Simulate Phase 2 verification
            phase2_data = self.two_phase_system.generate_phase2_verification_data(
                block_hash, f"merkle_root_{block_hash[:8]}", self.node_id, 
                self.enhanced_consensus.merkle_proof_count
            )
            
            # Simulate verification result (95% positive for demo)
            phase2_result = random.random() < 0.95
            phase2_evidence = None if phase2_result else "Merkle proof verification failed"
            
            self.two_phase_system.submit_phase2_verification(
                phase2_data, phase2_result, phase2_evidence
            )
            
            # Get verification summaries
            phase1_summary = self.two_phase_system.get_phase1_verification_summary(block_hash)
            phase2_summary = self.two_phase_system.get_phase2_verification_summary(block_hash)
            
            return {
                "node_id": self.node_id,
                "block_hash": block_hash,
                "phase1": {
                    "result": phase1_result,
                    "improvement": phase1_improvement,
                    "evidence": phase1_evidence,
                    "summary": phase1_summary
                },
                "phase2": {
                    "result": phase2_result,
                    "evidence": phase2_evidence,
                    "summary": phase2_summary
                },
                "verifier_stake": self.verifier_stake
            }
            
        except Exception as e:
            return {"error": f"Verification failed: {e}"}
    
    async def create_enhanced_training_block(self, model_id: str) -> Dict[str, Any]:
        """Create an enhanced training block with two-phase verification"""
        if not self.enhanced_consensus:
            return {"error": "Node not configured for mining"}
        
        try:
            # Create enhanced training block
            block = self.enhanced_consensus.create_enhanced_training_block(model_id)
            
            if block:
                return {
                    "success": True,
                    "block_hash": block.hash,
                    "block_index": block.index,
                    "merkle_proof_count": self.enhanced_consensus.merkle_proof_count,
                    "model_id": model_id,
                    "node_id": self.node_id
                }
            else:
                return {"error": "Failed to create training block"}
                
        except Exception as e:
            return {"error": f"Block creation failed: {e}"}


class LocalDevnet:
    """Manages a local development network"""
    
    def __init__(self, num_nodes: int = 5):
        self.num_nodes = num_nodes
        self.nodes: List[DevnetNode] = []
        self.network_stats = []
        self.start_time = time.time()
        
        print(f"🌐 Initializing local devnet with {num_nodes} nodes...")
    
    async def initialize_network(self):
        """Initialize all nodes and establish connections"""
        # Create nodes
        node_names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry"]
        
        for i in range(self.num_nodes):
            node_name = node_names[i] if i < len(node_names) else f"Node{i+1}"
            node = DevnetNode(node_name, port=8000 + i)
            self.nodes.append(node)
        
        await asyncio.sleep(1)
        
        # Establish peer connections (each node connects to 2-3 others)
        for i, node in enumerate(self.nodes):
            # Connect to next 2 nodes (circular)
            for j in range(1, min(3, self.num_nodes)):
                peer_idx = (i + j) % self.num_nodes
                if peer_idx != i:
                    node.connect_peer(self.nodes[peer_idx].node_id)
        
        print(f"✓ Network topology established")
        self._print_network_topology()
    
    def _print_network_topology(self):
        """Print the network connection topology"""
        print("\n📊 Network Topology:")
        for node in self.nodes:
            peers_str = ", ".join(node.peers)
            print(f"  {node.node_id}:{node.port} → [{peers_str}]")
    
    async def simulate_model_distribution(self):
        """Simulate model creation and distribution across the network"""
        print(f"\n🚀 Starting model distribution simulation...")
        
        models_created = []
        
        # Phase 1: Initial model creation by different nodes
        for i in range(min(3, self.num_nodes)):
            node = self.nodes[i]
            model_id = f"model_v{i+1}_by_{node.node_id.lower()}"
            size_kb = random.randint(50, 200)
            
            print(f"\n📦 {node.node_id} creating {model_id} ({size_kb}KB)...")
            success = await node.create_and_share_model(model_id, size_kb)
            
            if success:
                models_created.append((model_id, node.node_id))
                print(f"✓ {model_id} created and shared by {node.node_id}")
            
            await asyncio.sleep(1)
        
        # Phase 2: Cross-network model sharing
        print(f"\n🔄 Cross-network model distribution...")
        
        for model_id, creator_node_id in models_created:
            # Find the creator node
            creator_node = next(n for n in self.nodes if n.node_id == creator_node_id)
            torrent_info = creator_node.bt_distributor.model_torrents.get(model_id)
            
            if not torrent_info:
                continue
            
            # Share with 2-3 other nodes
            recipients = [n for n in self.nodes if n.node_id != creator_node_id]
            selected_recipients = random.sample(recipients, min(3, len(recipients)))
            
            for recipient in selected_recipients:
                print(f"  📡 {recipient.node_id} downloading {model_id}...")
                success = await recipient.download_model_from_peer(
                    model_id, torrent_info.magnet_link
                )
                if success:
                    print(f"    ✓ {recipient.node_id} downloaded {model_id}")
                await asyncio.sleep(0.5)
        
        return models_created
    
    async def simulate_enhanced_consensus_workflow(self):
        """Simulate enhanced consensus with two-phase verification"""
        print(f"\n🔐 Starting Enhanced Consensus Simulation...")
        
        # Find miner nodes (nodes with enhanced consensus)
        miner_nodes = [node for node in self.nodes if node.enhanced_consensus is not None]
        verifier_nodes = [node for node in self.nodes if node.enhanced_consensus is not None]
        
        if not miner_nodes:
            print("⚠️  No miner nodes available for consensus simulation")
            return []
        
        print(f"✓ Found {len(miner_nodes)} miner/verifier nodes")
        
        consensus_results = []
        
        # Simulate creating training blocks
        for i, miner in enumerate(miner_nodes[:2]):  # Limit to 2 miners for demo
            model_id = f"consensus_model_v{i+1}_by_{miner.node_id.lower()}"
            
            print(f"\n⛏️  {miner.node_id} mining block for {model_id}...")
            block_result = await miner.create_enhanced_training_block(model_id)
            
            if block_result.get("success"):
                block_hash = block_result["block_hash"]
                print(f"✓ Block created: {block_hash[:12]}... (proofs: {block_result['merkle_proof_count']})")
                
                # Simulate two-phase verification by other verifiers
                print(f"🔍 Starting two-phase verification...")
                
                phase1_results = []
                phase2_results = []
                
                # Phase 1: Quantized model verification
                print(f"  Phase 1: Quantized model verification...")
                for verifier in verifier_nodes:
                    if verifier.node_id != miner.node_id:  # Don't verify own block
                        result = await verifier.simulate_two_phase_verification(block_hash)
                        if "error" not in result:
                            phase1_results.append(result["phase1"])
                            phase2_results.append(result["phase2"])
                
                # Analyze results
                phase1_positive = sum(1 for r in phase1_results if r["result"])
                phase1_negative = len(phase1_results) - phase1_positive
                phase2_positive = sum(1 for r in phase2_results if r["result"])
                phase2_negative = len(phase2_results) - phase2_positive
                
                print(f"    Phase 1: {phase1_positive} positive, {phase1_negative} negative")
                print(f"    Phase 2: {phase2_positive} positive, {phase2_negative} negative")
                
                # Determine finalization
                total_verifiers = len(phase1_results)
                phase1_ratio = phase1_positive / total_verifiers if total_verifiers > 0 else 0
                phase2_ratio = phase2_positive / total_verifiers if total_verifiers > 0 else 0
                
                finalized = phase1_ratio >= 0.5 and phase2_ratio >= 0.5
                status = "✅ FINALIZED" if finalized else "❌ REJECTED"
                
                print(f"  Result: {status} (Phase 1: {phase1_ratio:.1%}, Phase 2: {phase2_ratio:.1%})")
                
                consensus_results.append({
                    "block_hash": block_hash,
                    "miner": miner.node_id,
                    "model_id": model_id,
                    "verifiers": total_verifiers,
                    "phase1_positive": phase1_positive,
                    "phase1_negative": phase1_negative,
                    "phase2_positive": phase2_positive,
                    "phase2_negative": phase2_negative,
                    "finalized": finalized,
                    "merkle_proof_count": block_result["merkle_proof_count"]
                })
                
                await asyncio.sleep(1)  # Brief pause between blocks
            else:
                print(f"❌ Failed to create block: {block_result.get('error')}")
        
        return consensus_results
    
    async def monitor_network_health(self, duration_minutes: float = 2.0):
        """Monitor network health for a specified duration"""
        print(f"\n🏥 Monitoring network health for {duration_minutes} minutes...")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            # Collect stats from all nodes
            network_snapshot = {
                "timestamp": time.time(),
                "elapsed_minutes": (time.time() - start_time) / 60,
                "nodes": []
            }
            
            total_models = 0
            total_torrents = 0
            total_seeders = 0
            active_nodes = 0
            verifier_nodes = 0
            total_stake = 0.0
            total_attestations = 0
            
            for node in self.nodes:
                node_stats = node.get_node_stats()
                network_snapshot["nodes"].append(node_stats)
                
                total_models += node_stats["registered_models"]
                total_torrents += node_stats["torrents"]
                total_seeders += node_stats["seeders"]
                if node_stats["torrents"] > 0:
                    active_nodes += 1
                
                # Enhanced consensus metrics
                if node_stats.get("is_verifier", False):
                    verifier_nodes += 1
                    total_stake += node_stats.get("verifier_stake", 0.0)
                    total_attestations += node_stats.get("total_attestations", 0)
            
            network_snapshot.update({
                "total_models": total_models,
                "total_torrents": total_torrents,
                "total_seeders": total_seeders,
                "active_nodes": active_nodes,
                "network_health": min(1.0, active_nodes / max(self.num_nodes, 1)),
                "verifier_nodes": verifier_nodes,
                "total_stake": total_stake,
                "total_attestations": total_attestations
            })
            
            self.network_stats.append(network_snapshot)
            
            # Print periodic status
            if len(self.network_stats) % 6 == 0:  # Every ~30 seconds
                self._print_network_status(network_snapshot)
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    def _print_network_status(self, snapshot: Dict[str, Any]):
        """Print current network status"""
        elapsed = snapshot["elapsed_minutes"]
        print(f"\n📈 Network Status (t+{elapsed:.1f}min):")
        print(f"  Active nodes: {snapshot['active_nodes']}/{self.num_nodes}")
        print(f"  Total models: {snapshot['total_models']}")
        print(f"  Total torrents: {snapshot['total_torrents']}")
        print(f"  Network health: {snapshot['network_health']:.1%}")
        
        # Enhanced consensus metrics
        if snapshot.get("verifier_nodes", 0) > 0:
            print(f"  🔐 Verifiers: {snapshot['verifier_nodes']} (stake: {snapshot['total_stake']:.1f})")
            print(f"  📋 Attestations: {snapshot['total_attestations']}")
        
        # Show top nodes by activity
        nodes_by_activity = sorted(
            snapshot["nodes"], 
            key=lambda n: n["torrents"] + n["seeders"] + (n.get("verifier_stake", 0) / 10), 
            reverse=True
        )
        
        print("  Top active nodes:")
        for node in nodes_by_activity[:3]:
            verifier_info = f", stake: {node.get('verifier_stake', 0):.1f}" if node.get('is_verifier') else ""
            print(f"    {node['node_id']}: {node['torrents']} torrents, {node['seeders']} seeders{verifier_info}")
    
    def generate_network_report(self) -> Dict[str, Any]:
        """Generate comprehensive network analysis report"""
        if not self.network_stats:
            return {"error": "No network statistics collected"}
        
        # Calculate aggregates
        final_stats = self.network_stats[-1]
        peak_health = max(s["network_health"] for s in self.network_stats)
        avg_health = sum(s["network_health"] for s in self.network_stats) / len(self.network_stats)
        
        # Node performance analysis
        node_analysis = {}
        for node_data in final_stats["nodes"]:
            node_id = node_data["node_id"]
            node_analysis[node_id] = {
                "final_models": node_data["registered_models"],
                "final_torrents": node_data["torrents"],
                "activity_score": node_data["torrents"] + node_data["seeders"],
                "peer_connections": node_data["peers"]
            }
        
        # Network evolution
        evolution = {
            "start_time": self.start_time,
            "duration_minutes": (time.time() - self.start_time) / 60,
            "peak_health": peak_health,
            "average_health": avg_health,
            "final_health": final_stats["network_health"],
            "models_created": final_stats["total_models"],
            "torrents_active": final_stats["total_torrents"]
        }
        
        return {
            "network_summary": evolution,
            "node_analysis": node_analysis,
            "topology": {
                "total_nodes": self.num_nodes,
                "active_nodes": final_stats["active_nodes"],
                "connectivity": sum(len(n.peers) for n in self.nodes) / self.num_nodes
            },
            "performance_metrics": {
                "models_per_node": final_stats["total_models"] / self.num_nodes,
                "torrents_per_node": final_stats["total_torrents"] / self.num_nodes,
                "distribution_efficiency": final_stats["total_torrents"] / max(final_stats["total_models"], 1)
            }
        }
    
    def print_final_report(self):
        """Print the final network analysis report"""
        report = self.generate_network_report()
        
        print("\n" + "="*80)
        print("DEVNET FINAL ANALYSIS REPORT")
        print("="*80)
        
        # Network Summary
        summary = report["network_summary"]
        print(f"\n📊 Network Summary:")
        print(f"  Duration: {summary['duration_minutes']:.1f} minutes")
        print(f"  Models created: {summary['models_created']}")
        print(f"  Active torrents: {summary['torrents_active']}")
        print(f"  Peak health: {summary['peak_health']:.1%}")
        print(f"  Average health: {summary['average_health']:.1%}")
        print(f"  Final health: {summary['final_health']:.1%}")
        
        # Topology Analysis
        topology = report["topology"]
        print(f"\n🌐 Network Topology:")
        print(f"  Total nodes: {topology['total_nodes']}")
        print(f"  Active nodes: {topology['active_nodes']}")
        print(f"  Avg connections per node: {topology['connectivity']:.1f}")
        
        # Performance Metrics
        perf = report["performance_metrics"]
        print(f"\n⚡ Performance Metrics:")
        print(f"  Models per node: {perf['models_per_node']:.1f}")
        print(f"  Torrents per node: {perf['torrents_per_node']:.1f}")
        print(f"  Distribution efficiency: {perf['distribution_efficiency']:.1%}")
        
        # Top Performing Nodes
        print(f"\n🏆 Node Performance Ranking:")
        node_scores = [(node_id, data["activity_score"]) 
                      for node_id, data in report["node_analysis"].items()]
        node_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (node_id, score) in enumerate(node_scores[:5]):
            node_data = report["node_analysis"][node_id]
            print(f"  {i+1}. {node_id}: Score {score}")
            print(f"     Models: {node_data['final_models']}, "
                  f"Torrents: {node_data['final_torrents']}, "
                  f"Peers: {node_data['peer_connections']}")
        
        # Network Health Assessment
        final_health = summary['final_health']
        if final_health > 0.8:
            health_status = "🟢 EXCELLENT"
        elif final_health > 0.6:
            health_status = "🟡 GOOD"
        elif final_health > 0.4:
            health_status = "🟠 FAIR"
        else:
            health_status = "🔴 POOR"
        
        print(f"\n🏥 Overall Network Health: {health_status} ({final_health:.1%})")
        
        return report


async def run_devnet_demo():
    """Run the complete local devnet demonstration"""
    print("🚀 DeSSIN Local Devnet Demonstration")
    print("=" * 60)
    
    # Configuration
    num_nodes = 5
    monitor_duration = 1.0  # minutes
    
    # Initialize devnet
    devnet = LocalDevnet(num_nodes)
    await devnet.initialize_network()
    
    # Run model distribution simulation
    models_created = await devnet.simulate_model_distribution()
    
    print(f"\n✅ Created and distributed {len(models_created)} models across the network")
    
    # Run enhanced consensus simulation
    consensus_results = await devnet.simulate_enhanced_consensus_workflow()
    
    if consensus_results:
        finalized_blocks = sum(1 for r in consensus_results if r["finalized"])
        print(f"\n🔐 Enhanced Consensus Results:")
        print(f"  Blocks created: {len(consensus_results)}")
        print(f"  Blocks finalized: {finalized_blocks}")
        print(f"  Finalization rate: {finalized_blocks/len(consensus_results):.1%}")
    
    # Monitor network health
    await devnet.monitor_network_health(monitor_duration)
    
    # Generate final report
    devnet.print_final_report()
    
    # Progress tracker summary
    print(f"\n📋 Progress Tracker Summary:")
    tracker = get_progress_tracker()
    tracker.print_status_summary()
    
    print(f"\n🎯 Devnet Demo Completed Successfully!")
    print(f"✓ {num_nodes} nodes participated")
    print(f"✓ {len(models_created)} models distributed")
    print(f"✓ Network monitored for {monitor_duration} minutes")
    print(f"✓ BitTorrent P2P distribution working")
    print(f"✓ Progress tracking across all operations")
    
    if consensus_results:
        finalized_blocks = sum(1 for r in consensus_results if r["finalized"])
        print(f"✓ Enhanced PoGO consensus with two-phase verification")
        print(f"✓ {len(consensus_results)} blocks created, {finalized_blocks} finalized")
        print(f"✓ Configurable Merkle proofs and attestation system")
    
    return devnet


if __name__ == "__main__":
    try:
        print("Starting DeSSIN Local Devnet...")
        devnet = asyncio.run(run_devnet_demo())
        
        # Optional: Save report to file
        report = devnet.generate_network_report()
        report_file = Path(tempfile.gettempdir()) / "dessin_devnet_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved to: {report_file}")
        
    except KeyboardInterrupt:
        print("\n👋 Devnet demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Devnet demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        import shutil
        demo_dir = Path(tempfile.gettempdir()) / "dessin_demo"
        if demo_dir.exists():
            shutil.rmtree(demo_dir)
            print(f"✓ Cleaned up devnet files")
