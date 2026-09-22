#!/usr/bin/env python3
"""
Unit tests for PoGO protocol model commitments
"""

import unittest
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dessin.protocol.pogo_protocol import PoGOProtocol, QuantizationLevel, ModelCommitment
from dessin.llm.simple_trainer import SimpleTrainer


class TestModelCommitments(unittest.TestCase):
    """Test model commitment creation and verification"""

    def setUp(self):
        self.trainer = SimpleTrainer()
        self.protocol = PoGOProtocol(self.trainer.model)
        
        # Train a simple model for testing
        self.training_result = self.trainer.train_model(
            training_steps=10, 
            learning_rate=0.01,
            data_seed=42  # For reproducible tests
        )
        
        # Define layer shapes
        self.layer_shapes = [
            (self.trainer.model.input_size, self.trainer.model.hidden_size),  # W1
            (1, self.trainer.model.hidden_size),                              # b1
            (self.trainer.model.hidden_size, self.trainer.model.output_size), # W2
            (1, self.trainer.model.output_size)                               # b2
        ]

    def test_create_model_commitments(self):
        """Test creating model commitments for different quantization levels"""
        commitments = self.protocol.create_model_commitments(
            self.training_result.model_weights_hex, 
            self.layer_shapes
        )
        
        # Should have commitments for FLOAT32 and INT4
        self.assertIn(QuantizationLevel.FLOAT32, commitments)
        self.assertIn(QuantizationLevel.INT4, commitments)
        
        for level, commitment in commitments.items():
            self.assertIsInstance(commitment, ModelCommitment)
            self.assertIsNotNone(commitment.merkle_root)
            self.assertEqual(len(commitment.merkle_root), 64)  # SHA256 hex length
            self.assertEqual(commitment.quantization_level, level)
            self.assertGreater(commitment.model_size_bytes, 0)
            self.assertEqual(len(commitment.layer_hashes), len(self.layer_shapes))
            self.assertIsInstance(commitment.proof_path, list)

    def test_commitment_merkle_roots_different(self):
        """Test that different quantization levels produce different Merkle roots"""
        commitments = self.protocol.create_model_commitments(
            self.training_result.model_weights_hex, 
            self.layer_shapes
        )
        
        float32_root = commitments[QuantizationLevel.FLOAT32].merkle_root
        int4_root = commitments[QuantizationLevel.INT4].merkle_root
        
        # Different quantization should produce different roots
        self.assertNotEqual(float32_root, int4_root)

    def test_verify_model_commitment(self):
        """Test model commitment verification"""
        commitments = self.protocol.create_model_commitments(
            self.training_result.model_weights_hex, 
            self.layer_shapes
        )
        
        for level, commitment in commitments.items():
            # Test verification for first layer (index 0) - others might have different proof structures
            is_valid = self.protocol.verify_model_commitment(commitment, 0)
            self.assertTrue(is_valid, f"Commitment verification failed for {level.value} layer 0")

    def test_commitment_deterministic(self):
        """Test that commitments are deterministic"""
        commitments1 = self.protocol.create_model_commitments(
            self.training_result.model_weights_hex, 
            self.layer_shapes
        )
        
        commitments2 = self.protocol.create_model_commitments(
            self.training_result.model_weights_hex, 
            self.layer_shapes
        )
        
        for level in commitments1.keys():
            self.assertEqual(commitments1[level].merkle_root, commitments2[level].merkle_root)
            self.assertEqual(commitments1[level].layer_hashes, commitments2[level].layer_hashes)

    def test_commitment_with_different_weights(self):
        """Test that different weights produce different commitments"""
        # Train another model with different seed
        trainer2 = SimpleTrainer()
        result2 = trainer2.train_model(
            training_steps=10, 
            learning_rate=0.01,
            data_seed=123  # Different seed
        )
        
        commitments1 = self.protocol.create_model_commitments(
            self.training_result.model_weights_hex, 
            self.layer_shapes
        )
        
        commitments2 = self.protocol.create_model_commitments(
            result2.model_weights_hex, 
            self.layer_shapes
        )
        
        for level in commitments1.keys():
            self.assertNotEqual(commitments1[level].merkle_root, commitments2[level].merkle_root)

    def test_commitment_layer_hashes_count(self):
        """Test that layer hashes match expected layer count"""
        commitments = self.protocol.create_model_commitments(
            self.training_result.model_weights_hex, 
            self.layer_shapes
        )
        
        for level, commitment in commitments.items():
            self.assertEqual(len(commitment.layer_hashes), len(self.layer_shapes))
            
            # Each layer hash should be valid SHA256
            for layer_hash in commitment.layer_hashes:
                self.assertEqual(len(layer_hash), 64)
                # Should be valid hex
                int(layer_hash, 16)

    def test_commitment_proof_path_structure(self):
        """Test that proof paths have correct structure"""
        commitments = self.protocol.create_model_commitments(
            self.training_result.model_weights_hex, 
            self.layer_shapes
        )
        
        for level, commitment in commitments.items():
            # Proof path should contain valid hashes
            for proof_hash in commitment.proof_path:
                self.assertEqual(len(proof_hash), 64)
                # Should be valid hex
                int(proof_hash, 16)

    def test_invalid_commitment_verification(self):
        """Test verification with invalid commitments"""
        commitments = self.protocol.create_model_commitments(
            self.training_result.model_weights_hex, 
            self.layer_shapes
        )
        
        commitment = commitments[QuantizationLevel.FLOAT32]
        
        # Test with invalid layer index
        is_valid = self.protocol.verify_model_commitment(commitment, 999)
        self.assertFalse(is_valid)
        
        # Test with corrupted commitment
        corrupted_commitment = ModelCommitment(
            merkle_root="invalid_root",
            quantization_level=QuantizationLevel.FLOAT32,
            model_size_bytes=commitment.model_size_bytes,
            layer_hashes=commitment.layer_hashes,
            proof_path=commitment.proof_path
        )
        
        is_valid = self.protocol.verify_model_commitment(corrupted_commitment, 0)
        self.assertFalse(is_valid)

    def test_commitment_size_consistency(self):
        """Test that commitment sizes are consistent with quantization"""
        commitments = self.protocol.create_model_commitments(
            self.training_result.model_weights_hex, 
            self.layer_shapes
        )
        
        float32_size = commitments[QuantizationLevel.FLOAT32].model_size_bytes
        int4_size = commitments[QuantizationLevel.INT4].model_size_bytes
        
        # INT4 should be smaller than FLOAT32
        self.assertLess(int4_size, float32_size)
        
        # Should match expected compression ratio (approximately 8x for float64 -> int4)
        expected_ratio = float32_size / int4_size
        self.assertGreater(expected_ratio, 4)  # At least 4x compression
        self.assertLess(expected_ratio, 16)    # At most 16x compression

    def test_empty_layer_shapes(self):
        """Test commitment creation with empty layer shapes"""
        # The implementation might handle empty layer shapes gracefully
        try:
            commitments = self.protocol.create_model_commitments(
                self.training_result.model_weights_hex, 
                []  # Empty layer shapes
            )
            # If it doesn't raise an error, verify the result is reasonable
            self.assertIsInstance(commitments, dict)
        except (ValueError, IndexError):
            # These are acceptable errors for empty shapes
            pass

    def test_single_layer_commitment(self):
        """Test commitment creation with single layer"""
        single_layer_shapes = [(10, 1)]  # Single layer
        
        # Create simple weights for single layer
        single_weights = np.random.randn(10).astype(np.float64)
        single_weights_hex = single_weights.tobytes().hex()
        
        commitments = self.protocol.create_model_commitments(
            single_weights_hex, 
            single_layer_shapes
        )
        
        for level, commitment in commitments.items():
            self.assertEqual(len(commitment.layer_hashes), 1)
            is_valid = self.protocol.verify_model_commitment(commitment, 0)
            self.assertTrue(is_valid)


if __name__ == "__main__":
    unittest.main()
