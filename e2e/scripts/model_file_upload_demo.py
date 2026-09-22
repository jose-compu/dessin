#!/usr/bin/env python3
"""
Model File Upload Demo
======================

Demonstrates flexible file upload system for model owners.
Supports uploading datasets, checkpoints, configs, and any other files
needed for pretraining and fine-tuning phases.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dessin.runtime.node import DessinNode
from dessin.runtime.config import DessinConfig
from dessin.nanochat.nanochat_integration import TrainingDevice
from dessin.models.model_file_manager import (
    ModelFileManager,
    FileType,
    TrainingPhase
)
from dessin.models.file_upload_transactions import (
    UploadFileTransaction,
    CreateFileBundleTransaction,
    FileUploadTransactionProcessor
)
import tempfile
import os
import time
import json


def main():
    print("=" * 80)
    print("Model File Upload Demo")
    print("=" * 80)
    print()
    print("Demonstrates flexible file upload for model owners via BitTorrent")
    print()
    
    # Initialize node
    print("Step 1: Initialize Node")
    print("-" * 80)
    config = DessinConfig.default()
    config.network.port = 9001
    config.network.bootstrap = True
    
    node = DessinNode(config)
    node.start()
    print(f"✓ Node started")
    print()
    
    # Create model owner
    owner_address = "model_owner_uploader"
    if hasattr(node.consensus, 'economic_system'):
        node.consensus.economic_system.balances[owner_address] = 1000.0
    
    # Create a model
    print("Step 2: Create Model")
    print("-" * 80)
    model_id = node.nanochat_integration.create_model(
        name="uploadable-model",
        owner_address=owner_address,
        depth=10,
        device_batch_size=4,
        dataset_name="InitialData",
        storage_blocks=100,
        storage_payment=1.0,
        training_payment_per_iteration=0.1,
        training_device=TrainingDevice.CPU
    )
    
    print(f"✓ Created model: {model_id[:30]}...")
    print()
    
    # Initialize file manager
    file_manager = ModelFileManager(
        storage_dir="model_files_demo",
        torrent_distributor=None  # Would use real distributor in production
    )
    
    tx_processor = FileUploadTransactionProcessor(file_manager, node.model_manager)
    
    # Step 3: Upload Pretraining Files
    print("Step 3: Upload Pretraining Files")
    print("-" * 80)
    print()
    
    # Create dummy pretraining dataset
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(100):
            f.write(json.dumps({"text": f"Pretraining example {i}"}) + '\n')
        pretrain_dataset_path = f.name
    
    # Upload dataset
    dataset_file = file_manager.upload_file(
        model_id=model_id,
        owner=owner_address,
        file_path=pretrain_dataset_path,
        file_type=FileType.DATASET,
        phase=TrainingPhase.PRETRAINING,
        description="Main pretraining dataset",
        tags=["pretraining", "baseline", "jsonl"],
        create_torrent=True
    )
    
    print()
    
    # Create config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_data = {
            "learning_rate": 3e-4,
            "batch_size": 32,
            "epochs": 3,
            "optimizer": "adamw"
        }
        json.dump(config_data, f, indent=2)
        config_path = f.name
    
    config_file = file_manager.upload_file(
        model_id=model_id,
        owner=owner_address,
        file_path=config_path,
        file_type=FileType.CONFIG,
        phase=TrainingPhase.PRETRAINING,
        description="Pretraining configuration",
        tags=["config", "pretraining"],
        create_torrent=True
    )
    
    print()
    
    # Step 4: Upload Fine-Tuning Files
    print("Step 4: Upload Fine-Tuning Files")
    print("-" * 80)
    print()
    
    # Create multiple fine-tuning datasets
    ft_files = []
    for i, dataset_name in enumerate(["domain_specific", "task_specific", "instruction"]):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for j in range(50):
                f.write(json.dumps({"text": f"{dataset_name} example {j}"}) + '\n')
            ft_dataset_path = f.name
        
        ft_file = file_manager.upload_file(
            model_id=model_id,
            owner=owner_address,
            file_path=ft_dataset_path,
            file_type=FileType.DATASET,
            phase=TrainingPhase.FINE_TUNING,
            description=f"Fine-tuning dataset: {dataset_name}",
            tags=["fine-tuning", dataset_name],
            create_torrent=True
        )
        
        if ft_file:
            ft_files.append(ft_file.file_id)
        
        # Clean up temp file
        os.unlink(ft_dataset_path)
    
    print()
    
    # Create checkpoint file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pt', delete=False) as f:
        # Simulate checkpoint data
        checkpoint_data = b"mock_checkpoint_data" * 100
        f.write(checkpoint_data)
        checkpoint_path = f.name
    
    checkpoint_file = file_manager.upload_file(
        model_id=model_id,
        owner=owner_address,
        file_path=checkpoint_path,
        file_type=FileType.CHECKPOINT,
        phase=TrainingPhase.FINE_TUNING,
        description="Checkpoint after layer 1 fine-tuning",
        tags=["checkpoint", "layer1"],
        create_torrent=True
    )
    
    print()
    os.unlink(checkpoint_path)
    
    # Step 5: Create File Bundle
    print("Step 5: Create File Bundle")
    print("-" * 80)
    print()
    
    bundle = file_manager.create_bundle(
        model_id=model_id,
        owner=owner_address,
        name="fine_tuning_complete",
        phase=TrainingPhase.FINE_TUNING,
        file_ids=ft_files,
        description="Complete fine-tuning dataset bundle",
        create_bundle_torrent=True
    )
    
    print()
    
    # Step 6: Upload RLHF Files
    print("Step 6: Upload RLHF Files")
    print("-" * 80)
    print()
    
    # Create preference dataset
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(30):
            f.write(json.dumps({
                "prompt": f"Question {i}",
                "chosen": f"Good response {i}",
                "rejected": f"Bad response {i}"
            }) + '\n')
        rlhf_dataset_path = f.name
    
    rlhf_file = file_manager.upload_file(
        model_id=model_id,
        owner=owner_address,
        file_path=rlhf_dataset_path,
        file_type=FileType.DATASET,
        phase=TrainingPhase.RLHF,
        description="Human preference dataset for RLHF",
        tags=["rlhf", "preferences", "human-feedback"],
        create_torrent=True
    )
    
    print()
    os.unlink(rlhf_dataset_path)
    
    # Step 7: Upload Custom Files
    print("Step 7: Upload Custom Files")
    print("-" * 80)
    print()
    
    # Tokenizer file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tokenizer_data = {
            "vocab_size": 50000,
            "model_max_length": 2048,
            "special_tokens": ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        }
        json.dump(tokenizer_data, f, indent=2)
        tokenizer_path = f.name
    
    tokenizer_file = file_manager.upload_file(
        model_id=model_id,
        owner=owner_address,
        file_path=tokenizer_path,
        file_type=FileType.TOKENIZER,
        phase=TrainingPhase.PRETRAINING,
        description="Custom tokenizer for model",
        tags=["tokenizer", "vocab"],
        create_torrent=True
    )
    
    print()
    os.unlink(tokenizer_path)
    
    # Step 8: File Summary
    print("Step 8: File Summary for Model")
    print("-" * 80)
    print()
    
    summary = file_manager.list_model_files_summary(model_id)
    
    print(f"Model: {model_id[:30]}...")
    print(f"Total files: {summary['total_files']}")
    print(f"Total bundles: {summary['total_bundles']}")
    print(f"Total size: {summary['total_size_mb']:.2f} MB")
    print(f"Seeding files: {summary['seeding_files']}")
    print()
    
    print("Files by Phase:")
    for phase, stats in summary['phase_stats'].items():
        print(f"  {phase}:")
        print(f"    Files: {stats['file_count']}")
        print(f"    Size: {stats['total_size_mb']:.2f} MB")
    print()
    
    print("Files by Type:")
    for file_type, count in summary['type_stats'].items():
        print(f"  {file_type}: {count}")
    print()
    
    # Step 9: List All Files
    print("Step 9: Detailed File Listing")
    print("-" * 80)
    print()
    
    all_files = file_manager.get_model_files(model_id)
    
    for phase in TrainingPhase:
        phase_files = [f for f in all_files if f.phase == phase]
        if phase_files:
            print(f"{phase.value.upper()}:")
            for file in phase_files:
                print(f"  • {file.file_name}")
                print(f"      Type: {file.file_type.value}")
                print(f"      Size: {file.file_size / 1024:.2f} KB")
                print(f"      Hash: {file.file_hash[:16]}...")
                if file.torrent_hash:
                    print(f"      Torrent: {file.torrent_hash[:16]}...")
                if file.tags:
                    print(f"      Tags: {', '.join(file.tags)}")
            print()
    
    # Step 10: Blockchain Transactions
    print("Step 10: Blockchain Transactions for Files")
    print("-" * 80)
    print()
    
    # Simulate file upload transaction
    if dataset_file:
        upload_tx = UploadFileTransaction(
            sender=owner_address,
            model_id=model_id,
            file_id=dataset_file.file_id,
            file_name=dataset_file.file_name,
            file_hash=dataset_file.file_hash,
            file_size=dataset_file.file_size,
            file_type=dataset_file.file_type.value,
            phase=dataset_file.phase.value,
            torrent_hash=dataset_file.torrent_hash,
            magnet_link=dataset_file.magnet_link,
            description=dataset_file.description,
            tags=dataset_file.tags,
            fee=0.01,
            timestamp=time.time(),
            public_key="pk_owner",
            signature="sig_upload",
            tx_id="tx_upload"
        )
        
        tx_processor.process_transaction(upload_tx)
        print()
    
    # Simulate bundle transaction
    if bundle:
        bundle_tx = CreateFileBundleTransaction(
            sender=owner_address,
            model_id=model_id,
            bundle_id=bundle.bundle_id,
            bundle_name=bundle.name,
            phase=bundle.phase.value,
            file_ids=bundle.files,
            bundle_torrent_hash=bundle.bundle_torrent_hash,
            bundle_magnet_link=bundle.bundle_magnet_link,
            description=bundle.description,
            fee=0.01,
            timestamp=time.time(),
            public_key="pk_owner",
            signature="sig_bundle",
            tx_id="tx_bundle"
        )
        
        tx_processor.process_transaction(bundle_tx)
        print()
    
    # Clean up temp files
    os.unlink(pretrain_dataset_path)
    os.unlink(config_path)
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("✓ Model owners can upload any files needed for training")
    print("✓ Supported file types:")
    print("   • Datasets (pretraining, fine-tuning, RLHF)")
    print("   • Checkpoints (resume training)")
    print("   • Configurations (training args)")
    print("   • Tokenizers & Vocabularies")
    print("   • Model weights & optimizer states")
    print("   • Custom files")
    print()
    print("✓ Organized by training phases:")
    print("   • Pretraining")
    print("   • Fine-tuning")
    print("   • Instruction Tuning")
    print("   • RLHF / DPO")
    print("   • Custom phases")
    print()
    print("✓ All files distributed via BitTorrent:")
    print("   • Individual file torrents")
    print("   • Multi-file bundles")
    print("   • Decentralized sharing")
    print("   • On-chain metadata")
    print()
    print("✓ Blockchain transactions for file management")
    print("✓ Flexible and extensible system")
    print()
    print("Model owners have full control over their training files!")
    print()
    
    # Cleanup
    print("Stopping node...")
    node.stop()
    print("✓ Node stopped")


if __name__ == "__main__":
    main()
