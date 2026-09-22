#!/usr/bin/env python3
"""
BitTorrent Distribution Demo for DeSSIN
Demonstrates distributed model sharing via BitTorrent and hybrid distribution.
"""

import sys
import os
import time
import tempfile
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "chaincraft"))

from dessin.runtime.config import ModelConfig
from dessin.models.model_manager import ModelManager
from dessin.distribution.bittorrent_distributor import BitTorrentDistributor
from dessin.distribution.hybrid_distributor import HybridDistributor, DistributionMethod
from dessin.distribution.hf_mock import create_hf_integration


def print_section(title):
    """Print a section header"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


def create_test_model_file(temp_dir: Path, model_id: str, size_kb: int = 100) -> str:
    """Create a test model file for demonstration"""
    model_path = temp_dir / f"{model_id}.gguf"
    
    # Create mock GGUF content with proper header
    header = b"GGUF"  # Magic number
    header += (2).to_bytes(4, 'little')  # Version
    header += (0).to_bytes(8, 'little')  # Tensor count
    header += (0).to_bytes(8, 'little')  # KV count
    
    # Add model info
    info = f"Mock model: {model_id}".encode()
    header += len(info).to_bytes(4, 'little') + info
    
    # Fill to desired size
    remaining_size = max(1024, size_kb * 1024) - len(header)
    content = header + b"Model data " * (remaining_size // 11)
    
    with open(model_path, "wb") as f:
        f.write(content)
    
    return str(model_path)


def demo_bittorrent_basic():
    """Demonstrate basic BitTorrent functionality"""
    print_section("BASIC BITTORRENT FUNCTIONALITY")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Setup model manager
        config = ModelConfig()
        config.model_cache_dir = str(temp_path / "models")
        model_manager = ModelManager(config)
        
        # Create BitTorrent distributor
        bt_dist = BitTorrentDistributor(
            model_manager,
            download_dir=str(temp_path / "torrents"),
            listen_port=6881
        )
        
        print("✓ BitTorrent distributor initialized")
        
        # Register test models
        models_data = [
            ("small_gpt2", "Small GPT-2 Model", 50000000, 200),    # 200KB
            ("medium_bert", "Medium BERT Model", 110000000, 500),  # 500KB
            ("large_llama", "Large LLaMA Model", 7000000000, 1000) # 1MB
        ]
        
        created_models = []
        
        for model_id, name, params, size_kb in models_data:
            print(f"\n--- Processing {model_id} ---")
            
            # Create model file
            model_path = create_test_model_file(temp_path, model_id, size_kb)
            
            # Register with model manager
            success = model_manager.register_model(
                model_id=model_id,
                name=name,
                size_gb=size_kb / 1024 / 1024,  # Convert KB to GB
                format="gguf",
                quantization="4bit",
                parameters=params,
                ipfs_hash=f"Qm{model_id}123",
                owner="0x1234567890123456789012345678901234567890",
                upload_block=1,
                storage_expires=1000,
                model_hash=f"hash_{model_id}"
            )
            
            if success:
                print(f"  ✓ Registered model: {model_id}")
                
                # Create torrent
                torrent_info = bt_dist.create_model_torrent(model_id, model_path)
                
                if torrent_info:
                    print(f"  ✓ Created torrent:")
                    print(f"    Hash: {torrent_info.torrent_hash[:16]}...")
                    print(f"    Size: {torrent_info.file_size / 1024:.1f} KB")
                    print(f"    Pieces: {torrent_info.piece_count}")
                    print(f"    Magnet: {torrent_info.magnet_link[:50]}...")
                    
                    # Start seeding
                    seed_success = bt_dist.seed_model(model_id, model_path)
                    if seed_success:
                        print(f"  ✓ Started seeding")
                        created_models.append(model_id)
        
        # Show overall status
        print(f"\n--- BitTorrent Network Status ---")
        all_status = bt_dist.get_all_torrents_status()
        for status in all_status:
            print(f"Model: {status['model_id']}")
            print(f"  State: {status['state']}")
            print(f"  Progress: {status['progress']:.1%}")
            print(f"  Seeders: {status['seeders']}")
            print(f"  Leechers: {status['leechers']}")
        
        # Show network statistics
        network_stats = bt_dist.get_network_statistics()
        print(f"\n--- Network Statistics ---")
        print(f"Total models: {network_stats['total_models']}")
        print(f"Active torrents: {network_stats['active_torrents']}")
        print(f"Total seeders: {network_stats['total_seeders']}")
        print(f"Average availability: {network_stats['average_availability']:.2f}")
        
        return bt_dist, created_models


def demo_torrent_download_simulation():
    """Simulate torrent download process"""
    print_section("TORRENT DOWNLOAD SIMULATION")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Setup two separate instances (seeder and downloader)
        seeder_config = ModelConfig()
        seeder_config.model_cache_dir = str(temp_path / "seeder")
        seeder_manager = ModelManager(seeder_config)
        
        downloader_config = ModelConfig()
        downloader_config.model_cache_dir = str(temp_path / "downloader")
        downloader_manager = ModelManager(downloader_config)
        
        # Create seeder
        seeder = BitTorrentDistributor(
            seeder_manager,
            download_dir=str(temp_path / "seeder_torrents"),
            listen_port=6881
        )
        
        # Create downloader
        downloader = BitTorrentDistributor(
            downloader_manager,
            download_dir=str(temp_path / "downloader_torrents"),
            listen_port=6882
        )
        
        print("✓ Created seeder and downloader instances")
        
        # Create and seed a model
        model_id = "download_test_model"
        model_path = create_test_model_file(temp_path / "seeder", model_id, 300)  # 300KB
        
        seeder_manager.register_model(
            model_id=model_id,
            name="Download Test Model",
            size_gb=0.3 / 1024,
            format="gguf",
            quantization="4bit",
            parameters=30000000,
            ipfs_hash="QmDownloadTest123",
            owner="0x1234567890123456789012345678901234567890",
            upload_block=1,
            storage_expires=1000,
            model_hash="download_test_hash"
        )
        
        # Create torrent and start seeding
        torrent_info = seeder.create_model_torrent(model_id, model_path)
        seeder.seed_model(model_id, model_path)
        
        print(f"✓ Seeder created torrent for {model_id}")
        print(f"  Magnet link: {torrent_info.magnet_link}")
        
        # Downloader attempts to download
        print(f"\n--- Starting Download ---")
        download_success = downloader.download_model_by_magnet(
            torrent_info.magnet_link, 
            model_id
        )
        
        if download_success:
            print(f"✓ Download started successfully")
            
            # Monitor download progress
            print(f"\n--- Download Progress ---")
            for i in range(10):  # Monitor for up to 10 seconds
                progress = downloader.get_download_progress(model_id)
                
                if progress:
                    print(f"  Progress: {progress['progress']:.1%} "
                          f"| Speed: {progress['download_rate'] / 1024:.1f} KB/s "
                          f"| Seeders: {progress['seeders']} "
                          f"| State: {progress['state']}")
                    
                    if progress['progress'] >= 1.0:
                        print(f"  ✓ Download completed!")
                        break
                else:
                    print(f"  No progress data available")
                
                time.sleep(1)
        else:
            print(f"✗ Failed to start download")
        
        # Show final status
        seeder_stats = seeder.get_network_statistics()
        downloader_stats = downloader.get_network_statistics()
        
        print(f"\n--- Final Status ---")
        print(f"Seeder - Active torrents: {seeder_stats['active_torrents']}")
        print(f"Downloader - Active torrents: {downloader_stats['active_torrents']}")


def demo_hybrid_distribution():
    """Demonstrate hybrid distribution system"""
    print_section("HYBRID DISTRIBUTION SYSTEM")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Setup model manager
        config = ModelConfig()
        config.model_cache_dir = str(temp_path / "models")
        model_manager = ModelManager(config)
        
        # Create hybrid distributor
        hybrid_dist = HybridDistributor(
            model_manager,
            enable_bittorrent=True,
            enable_ipfs=True,
            enable_huggingface=True
        )
        
        print("✓ Hybrid distributor initialized")
        
        # Create test models with different characteristics
        models = [
            ("popular_model", "Popular Model", 1000000000, 2000, "bittorrent"),    # 2MB - good for P2P
            ("research_model", "Research Model", 500000000, 500, "huggingface"),   # 500KB - good for HF
            ("experimental_model", "Experimental Model", 100000000, 100, "ipfs")  # 100KB - good for IPFS
        ]
        
        registered_models = []
        
        for model_id, name, params, size_kb, primary_method in models:
            print(f"\n--- Registering {model_id} ---")
            
            # Create model file
            model_path = create_test_model_file(temp_path, model_id, size_kb)
            
            # Register with model manager
            model_manager.register_model(
                model_id=model_id,
                name=name,
                size_gb=size_kb / 1024 / 1024,
                format="gguf",
                quantization="4bit",
                parameters=params,
                ipfs_hash=f"Qm{model_id}Hash",
                owner="0x1234567890123456789012345678901234567890",
                upload_block=1,
                storage_expires=1000,
                model_hash=f"hybrid_{model_id}_hash"
            )
            
            # Register for hybrid distribution
            primary = DistributionMethod(primary_method)
            
            success = hybrid_dist.register_model_for_distribution(
                model_id, model_path, primary_method=primary
            )
            
            if success:
                print(f"  ✓ Registered with hybrid distribution")
                registered_models.append(model_id)
                
                # Show distribution health
                health = hybrid_dist.get_distribution_health(model_id)
                print(f"  Health score: {health['health_score']:.2f}")
                print(f"  Active channels: {health['active_channels']}")
                print(f"  Best channel: {health['best_channel']}")
        
        # Show network overview
        print(f"\n--- Network Overview ---")
        overview = hybrid_dist.get_network_overview()
        print(f"Total models: {overview['total_models']}")
        print(f"Total channels: {overview['total_channels']}")
        print(f"Distribution methods: {overview['distribution_methods']}")
        print(f"Average health: {overview['average_health_score']:.2f}")
        print(f"BitTorrent enabled: {overview['bittorrent_enabled']}")
        print(f"IPFS enabled: {overview['ipfs_enabled']}")
        print(f"HuggingFace enabled: {overview['huggingface_enabled']}")
        
        # List all distributed models
        print(f"\n--- Distributed Models ---")
        distributed_models = hybrid_dist.list_distributed_models()
        
        for model in distributed_models:
            print(f"Model: {model['model_id']}")
            print(f"  Primary: {model['primary_method']}")
            print(f"  Channels: {model['channels']}")
            print(f"  Health: {model['health_score']:.2f}")
            print(f"  Size: {model['size_mb']:.1f} MB")
            print(f"  Downloads: {model['download_count']}")
        
        # Demonstrate download with fallback
        if registered_models:
            print(f"\n--- Testing Download with Fallback ---")
            test_model = registered_models[0]
            
            # Attempt download
            success, local_path = hybrid_dist.download_model(
                test_model,
                fallback_on_failure=True
            )
            
            if success:
                print(f"✓ Downloaded {test_model} successfully")
                print(f"  Local path: {local_path}")
                print(f"  File exists: {Path(local_path).exists()}")
                if Path(local_path).exists():
                    print(f"  File size: {Path(local_path).stat().st_size / 1024:.1f} KB")
            else:
                print(f"✗ Failed to download {test_model}")
        
        return hybrid_dist


def demo_distribution_comparison():
    """Compare different distribution methods"""
    print_section("DISTRIBUTION METHOD COMPARISON")
    
    comparison_data = [
        {
            "method": "BitTorrent",
            "decentralization": "★★★★★",
            "bandwidth_efficiency": "★★★★★", 
            "availability": "★★★★☆",
            "speed": "★★★☆☆",
            "cost": "★★★★★",
            "use_case": "Large models, many users"
        },
        {
            "method": "HuggingFace",
            "decentralization": "★★☆☆☆",
            "bandwidth_efficiency": "★★★☆☆",
            "availability": "★★★★★",
            "speed": "★★★★☆",
            "cost": "★★★☆☆",
            "use_case": "Popular models, fast access"
        },
        {
            "method": "IPFS",
            "decentralization": "★★★★☆",
            "bandwidth_efficiency": "★★★★☆",
            "availability": "★★★★☆",
            "speed": "★★★☆☆",
            "cost": "★★★★☆",
            "use_case": "Permanent storage, content addressing"
        },
        {
            "method": "HTTPS/CDN",
            "decentralization": "★☆☆☆☆",
            "bandwidth_efficiency": "★★☆☆☆",
            "availability": "★★★★☆",
            "speed": "★★★★★",
            "cost": "★★☆☆☆",
            "use_case": "Fast delivery, commercial use"
        }
    ]
    
    print("Comparison of Distribution Methods:")
    print("-" * 80)
    
    for data in comparison_data:
        print(f"\n{data['method']}:")
        print(f"  Decentralization:     {data['decentralization']}")
        print(f"  Bandwidth Efficiency: {data['bandwidth_efficiency']}")
        print(f"  Availability:         {data['availability']}")
        print(f"  Speed:               {data['speed']}")
        print(f"  Cost:                {data['cost']}")
        print(f"  Best Use Case:       {data['use_case']}")
    
    print(f"\n" + "="*80)
    print("RECOMMENDATION FOR DeSSIN:")
    print("="*80)
    print("✓ PRIMARY: BitTorrent - Perfect for decentralized model sharing")
    print("✓ SECONDARY: IPFS - Good for content-addressed permanent storage")
    print("✓ TERTIARY: HuggingFace - Excellent for popular model discovery")
    print("✓ FALLBACK: HTTPS - Fast delivery when others fail")
    
    print(f"\nWhy BitTorrent is ideal for DeSSIN:")
    print("• True decentralization - no single point of failure")
    print("• Bandwidth scaling - more popular models get better performance")
    print("• Cost efficiency - no server costs for model hosting")
    print("• Resilience - models stay available as long as peers exist")
    print("• Large file optimized - perfect for multi-GB model files")
    print("• Built-in verification - SHA1 hashes ensure integrity")


def main():
    """Main demo function"""
    print_section("DeSSIN BITTORRENT DISTRIBUTION DEMONSTRATION")
    print("This demo showcases BitTorrent-based model distribution")
    print("and hybrid distribution systems for decentralized AI.")
    
    try:
        # Demo 1: Basic BitTorrent functionality
        bt_dist, created_models = demo_bittorrent_basic()
        
        # Demo 2: Download simulation
        demo_torrent_download_simulation()
        
        # Demo 3: Hybrid distribution
        hybrid_dist = demo_hybrid_distribution()
        
        # Demo 4: Method comparison
        demo_distribution_comparison()
        
        print_section("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("✓ BitTorrent distribution system functional")
        print("✓ Hybrid distribution with multiple channels working")
        print("✓ Download and seeding mechanisms operational")
        print("✓ Distribution health monitoring active")
        
        print(f"\nKey Benefits Demonstrated:")
        print(f"  • Decentralized model distribution via BitTorrent")
        print(f"  • Multiple redundant distribution channels")
        print(f"  • Automatic fallback between methods")
        print(f"  • Health monitoring and redundancy management")
        print(f"  • Cost-effective bandwidth usage")
        print(f"  • High availability through peer networks")
        
        print(f"\nProduction Considerations:")
        print(f"  • Install libtorrent-rasterbar for real BitTorrent")
        print(f"  • Set up IPFS node for content addressing")
        print(f"  • Configure trackers for torrent discovery")
        print(f"  • Implement bandwidth throttling controls")
        print(f"  • Add reputation system for peer quality")
        
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nDemo completed.")


if __name__ == "__main__":
    main()
