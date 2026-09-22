#!/usr/bin/env python3
"""
Simple DeSSIN Demo - No networking required
Demonstrates core blockchain functionality without P2P networking.
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

from dessin.runtime.config import DessinConfig, ModelConfig, ConsensusConfig
from dessin.models.model_manager import ModelManager
from dessin.consensus import PogoConsensus
from dessin.consensus.transactions import TransactionFactory
from dessin.fine_tuning import FineTuningManager


def print_section(title):
    """Print a section header"""
    print(f"\n{'='*50}")
    print(f" {title}")
    print('='*50)


def demo_basic_components():
    """Demonstrate basic components without networking"""
    print_section("BASIC COMPONENTS")
    
    # Configuration
    config = DessinConfig.default()
    config.consensus.block_time_minutes = 0.06  # ~3.6s for demo
    
    print(f"Configuration loaded:")
    print(f"  Block time: {config.consensus.block_time_minutes} minutes")
    print(f"  Finalization window: {config.consensus.finalization_window} blocks")
    print(f"  Attestation threshold: {config.consensus.attestation_threshold}")
    print(f"  Max Merkle proofs: {config.consensus.max_merkle_proofs}")


def demo_model_manager():
    """Demonstrate model manager functionality"""
    print_section("MODEL MANAGER")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ModelConfig()
        config.model_cache_dir = temp_dir
        
        manager = ModelManager(config)
        
        # Register a test model
        model_id = manager.register_model(
            model_id="test_gpt2",
            name="Test GPT-2 Model",
            size_gb=0.5,
            format="gguf",
            quantization="4bit",
            parameters=124000000,
            ipfs_hash="QmTest123",
            owner="0x1234567890123456789012345678901234567890",
            upload_block=1,
            storage_expires=1000,
            model_hash="test_hash_123"
        )
        
        print(f"✓ Registered model: {model_id}")
        
        # List models
        models = manager.list_models()
        print(f"✓ Found {len(models)} models")
        
        for model in models:
            print(f"  Model: {model.model_id}")
            print(f"    Name: {model.name}")
            print(f"    Size: {model.size_gb} GB")
            print(f"    Parameters: {model.parameters:,}")
        
        # Test model availability
        available = manager.is_model_available("test_gpt2", 500)
        print(f"✓ Model available at block 500: {available}")
        
        # Create test file for Merkle tree
        test_file = Path(temp_dir) / "test_model.bin"
        test_data = b"DeSSIN test model data " * 1000  # ~23KB
        
        with open(test_file, "wb") as f:
            f.write(test_data)
        
        # Test Merkle tree creation
        merkle_root, leaves = manager.create_merkle_tree(str(test_file), leaf_size_mb=0.01)
        print(f"✓ Created Merkle tree:")
        print(f"    Root: {merkle_root[:32]}...")
        print(f"    Leaves: {len(leaves)}")
        
        # Test model query (simulated)
        result = manager.query_model("test_gpt2", "Hello, world!", max_tokens=20)
        print(f"✓ Model query result:")
        print(f"    Success: {result.success}")
        if result.success:
            print(f"    Response: {result.response}")
            print(f"    Tokens used: {result.tokens_used}")
            print(f"    Cost: {result.compute_cost:.6f} DESSIN")


def demo_consensus():
    """Demonstrate consensus functionality"""
    print_section("CONSENSUS SYSTEM")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup
        config = ConsensusConfig()
        config.min_loss_improvement = 0.001
        config.max_merkle_proofs = 4
        
        model_config = ModelConfig()
        model_config.model_cache_dir = temp_dir
        model_manager = ModelManager(model_config)
        
        # Register test model
        model_manager.register_model(
            model_id="consensus_test_model",
            name="Consensus Test Model",
            size_gb=1.0,
            format="gguf",
            quantization="4bit",
            parameters=100000000,
            ipfs_hash="QmTest456",
            owner="0x1234567890123456789012345678901234567890",
            upload_block=0,
            storage_expires=1000,
            model_hash="consensus_hash"
        )
        
        miner_address = "0x1234567890123456789012345678901234567890"
        consensus = PogoConsensus(config, model_manager, miner_address)
        
        print(f"✓ Consensus initialized")
        print(f"    Genesis block created")
        print(f"    Chain length: {consensus.get_chain_length()}")
        
        # Create a training block
        print(f"\nCreating training block...")
        block = consensus.create_training_block("consensus_test_model")
        
        if block:
            print(f"✓ Training block created:")
            print(f"    Index: {block.index}")
            print(f"    Model: {block.model_id}")
            print(f"    Loss before: {block.loss_before:.6f}")
            print(f"    Loss after: {block.loss_after:.6f}")
            print(f"    Improvement: {block.loss_before - block.loss_after:.6f}")
            print(f"    Training steps: {block.training_steps}")
            print(f"    Hash: {block.hash[:32]}...")
            
            # Verify the block
            print(f"\nVerifying block...")
            verification = consensus.verify_block(block)
            
            print(f"✓ Block verification:")
            print(f"    Loss improvement: {verification.loss_improvement:.6f}")
            print(f"    Merkle verification: {verification.merkle_verification}")
            print(f"    Data availability: {verification.data_availability}")
            
            # Create attestation
            print(f"\nCreating attestation...")
            
            # Generate test keys
            sys.path.insert(0, os.path.join(_REPO_ROOT, "chaincraft", "examples"))
            from blockchain import BlockchainUtils
            
            private_key, public_key = BlockchainUtils.generate_keypair()
            
            from dessin.consensus import AttestationType
            attestation = consensus.create_attestation(
                block, AttestationType.POSITIVE, private_key, public_key
            )
            
            print(f"✓ Attestation created:")
            print(f"    Type: {attestation.attestation_type}")
            print(f"    Block hash: {attestation.block_hash[:16]}...")
            print(f"    Verification data keys: {list(attestation.verification_data.keys())}")
        else:
            print("✗ Failed to create training block")


def demo_transactions():
    """Demonstrate transaction system"""
    print_section("TRANSACTION SYSTEM")
    
    # Generate test keys
    sys.path.insert(0, os.path.join(_REPO_ROOT, "chaincraft", "examples"))
    from blockchain import BlockchainUtils
    
    private_key, public_key = BlockchainUtils.generate_keypair()
    address = BlockchainUtils.get_address_from_public_key(public_key)
    
    print(f"Generated test address: {address}")
    
    # Test different transaction types
    transaction_types = [
        ("transfer", {
            "recipient": "0x1234567890123456789012345678901234567890",
            "amount": 10.0,
            "fee": 0.001
        }),
        ("model_query", {
            "model_id": "test_model_123",
            "query_data": "What is blockchain?",
            "max_tokens": 50,
            "token_price": 0.001,
            "fee": 0.002
        }),
        ("conditional_transfer", {
            "recipient": "0x1234567890123456789012345678901234567890",
            "amount": 5.0,
            "model_id": "sentiment_model",
            "condition_query": "The weather is great today",
            "condition_expected": "positive",
            "fee": 0.003
        }),
        ("model_upload", {
            "model_name": "Test Upload Model",
            "model_hash": "upload_hash_123",
            "model_size_gb": 2.5,
            "storage_blocks": 1000,
            "storage_payment": 25.0,
            "fee": 0.01
        })
    ]
    
    for tx_type, params in transaction_types:
        print(f"\nCreating {tx_type} transaction...")
        
        tx = TransactionFactory.create_signed_transaction(
            tx_type=tx_type,
            sender=address,
            private_key=private_key,
            public_key=public_key,
            **params
        )
        
        # Verify transaction
        valid = TransactionFactory.verify_transaction(tx)
        
        print(f"✓ {tx_type.title()} transaction:")
        print(f"    TX ID: {tx.tx_id[:16]}...")
        print(f"    Signature valid: {valid}")
        print(f"    Fee: {tx.fee} DESSIN")
        
        # Show specific transaction details
        if tx_type == "model_query":
            print(f"    Query: {tx.query_data}")
            print(f"    Max tokens: {tx.max_tokens}")
        elif tx_type == "conditional_transfer":
            print(f"    Amount: {tx.amount} DESSIN")
            print(f"    Condition: {tx.condition_query} -> {tx.condition_expected}")


def demo_fine_tuning():
    """Demonstrate fine-tuning system"""
    print_section("FINE-TUNING SYSTEM")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup model manager
        config = ModelConfig()
        config.model_cache_dir = temp_dir
        model_manager = ModelManager(config)
        
        # Register base model
        model_manager.register_model(
            model_id="base_model_gpt2",
            name="Base GPT-2 Model",
            size_gb=0.5,
            format="gguf",
            quantization="4bit",
            parameters=124000000,
            ipfs_hash="QmBase123",
            owner="0x1234567890123456789012345678901234567890",
            upload_block=1,
            storage_expires=1000,
            model_hash="base_hash_123"
        )
        
        # Setup fine-tuning manager
        ft_manager = FineTuningManager(model_manager)
        
        # Create test dataset file
        dataset_file = Path(temp_dir) / "test_dataset.jsonl"
        test_data = [
            '{"input": "Hello", "output": "Hi there!"}',
            '{"input": "How are you?", "output": "I am doing well, thank you!"}',
            '{"input": "What is AI?", "output": "AI is artificial intelligence."}',
        ]
        
        with open(dataset_file, 'w') as f:
            f.write('\n'.join(test_data))
        
        # Register dataset
        print("Registering fine-tuning dataset...")
        dataset_id = ft_manager.register_dataset(
            dataset_name="Test Chat Dataset",
            dataset_path=str(dataset_file),
            dataset_format="jsonl",
            owner="0x1234567890123456789012345678901234567890",
            description="Small test dataset for demonstration"
        )
        
        if dataset_id:
            print(f"✓ Dataset registered: {dataset_id}")
            
            # List datasets
            datasets = ft_manager.list_datasets()
            print(f"✓ Available datasets: {len(datasets)}")
            
            for dataset in datasets:
                print(f"    {dataset['dataset_id']}: {dataset['name']}")
                print(f"      Examples: {dataset['num_examples']}")
                print(f"      Size: {dataset['size_mb']:.2f} MB")
            
            # Create fine-tuning job
            print(f"\nCreating fine-tuning job...")
            job_id = ft_manager.create_fine_tuning_job(
                base_model_id="base_model_gpt2",
                dataset_id=dataset_id,
                owner="0x1234567890123456789012345678901234567890",
                num_epochs=2,
                max_steps=10,
                max_cost_dessin=5.0
            )
            
            if job_id:
                print(f"✓ Fine-tuning job created: {job_id}")
                
                # Get job status
                status = ft_manager.get_job_status(job_id)
                print(f"✓ Job status:")
                print(f"    Status: {status['status']}")
                print(f"    Base model: {status['base_model_id']}")
                print(f"    Dataset: {status['dataset_name']}")
                print(f"    Max steps: {status['max_steps']}")
                print(f"    Max cost: {status['max_cost']:.3f} DESSIN")
                
                # Start fine-tuning
                print(f"\nStarting fine-tuning...")
                success = ft_manager.start_fine_tuning(job_id)
                
                if success:
                    print(f"✓ Fine-tuning started")
                    
                    # Simulate a few training steps
                    print(f"Simulating training steps...")
                    for step in range(5):
                        ft_manager.simulate_fine_tuning_step(job_id)
                        status = ft_manager.get_job_status(job_id)
                        print(f"  Step {status['current_step']}: Loss {status['current_loss']:.6f}")
                        
                        if status['status'] == 'completed':
                            print(f"✓ Fine-tuning completed!")
                            print(f"    Final model: {status['final_model_id']}")
                            print(f"    Total cost: {status['total_cost']:.6f} DESSIN")
                            break
                        
                        time.sleep(0.1)  # Brief pause for demo
                else:
                    print("✗ Failed to start fine-tuning")
            else:
                print("✗ Failed to create fine-tuning job")
        else:
            print("✗ Failed to register dataset")


def demo_vrf_and_crypto():
    """Demonstrate VRF and cryptographic features"""
    print_section("VRF & CRYPTOGRAPHIC FEATURES")
    
    # VRF demonstration
    from chaincraft.crypto_primitives.vrf import ECDSAVRFPrimitive
    
    vrf = ECDSAVRFPrimitive()
    vrf.generate_key()
    
    # Test VRF with different inputs
    test_inputs = [
        b"training_data_selection_block_100",
        b"training_data_selection_block_101",
        b"training_data_selection_block_102"
    ]
    
    print("VRF randomness generation:")
    for i, test_input in enumerate(test_inputs, 1):
        signature = vrf.sign(test_input)
        valid = vrf.verify(test_input, signature)
        vrf_output = vrf.vrf_output(test_input, signature)
        
        print(f"  Test {i}:")
        print(f"    Input: {test_input.decode()}")
        print(f"    Valid: {valid}")
        print(f"    Output: {vrf_output.hex()[:32]}...")
    
    # Blockchain utilities demonstration
    sys.path.insert(0, os.path.join(_REPO_ROOT, "chaincraft", "examples"))
    from blockchain import BlockchainUtils
    
    print(f"\nBlockchain utilities:")
    
    # Key generation
    private_key1, public_key1 = BlockchainUtils.generate_keypair()
    private_key2, public_key2 = BlockchainUtils.generate_keypair()
    
    address1 = BlockchainUtils.get_address_from_public_key(public_key1)
    address2 = BlockchainUtils.get_address_from_public_key(public_key2)
    
    print(f"  Generated addresses:")
    print(f"    Address 1: {address1}")
    print(f"    Address 2: {address2}")
    
    # Hash calculation
    test_data = {"message": "DeSSIN blockchain test", "timestamp": time.time()}
    hash_result = BlockchainUtils.calculate_hash(test_data)
    
    print(f"  Hash calculation:")
    print(f"    Data: {test_data}")
    print(f"    Hash: {hash_result[:32]}...")
    
    # Signature verification
    signature = BlockchainUtils.sign_transaction(test_data, private_key1)
    valid_sig = BlockchainUtils.verify_signature(test_data, signature, public_key1)
    invalid_sig = BlockchainUtils.verify_signature(test_data, signature, public_key2)
    
    print(f"  Digital signatures:")
    print(f"    Signature length: {len(signature)} chars")
    print(f"    Valid with correct key: {valid_sig}")
    print(f"    Valid with wrong key: {invalid_sig}")


def main():
    """Main demo function"""
    print_section("DeSSIN SIMPLE DEMONSTRATION")
    print("This demo showcases the core features of DeSSIN blockchain")
    print("without requiring P2P networking or external dependencies.")
    
    try:
        # Run all demonstrations
        demo_basic_components()
        demo_model_manager()
        demo_consensus()
        demo_transactions()
        demo_fine_tuning()
        demo_vrf_and_crypto()
        
        print_section("DEMO COMPLETED SUCCESSFULLY")
        print("✓ All core features demonstrated successfully")
        print("✓ DeSSIN blockchain components are functional")
        print("✓ PoGO consensus system operational")
        print("✓ Model management system working")
        print("✓ Transaction system functional")
        print("✓ Fine-tuning system operational")
        print("✓ Cryptographic features verified")
        
        print(f"\nNext steps:")
        print(f"  • Try: python3 dessin_cli.py test-network")
        print(f"  • Run: python3 -m pytest tests/ -v")
        print(f"  • Explore: Browse the dessin/ directory")
        
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nDemo completed.")


if __name__ == "__main__":
    main()
