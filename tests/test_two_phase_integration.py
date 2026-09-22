"""
Integration test for Two-Phase Verification System with Enhanced PoGO Consensus.
Demonstrates the complete two-phase verification workflow.
"""

import unittest
import time
from unittest.mock import Mock, patch

from dessin.consensus.enhanced_pogo_consensus import EnhancedPogoConsensus
from dessin.consensus.two_phase_verification import VerificationPhase
from dessin.runtime.config import ConsensusConfig, ModelConfig
from dessin.models.model_manager import ModelManager


class TestTwoPhaseIntegration(unittest.TestCase):
    """Integration tests for two-phase verification system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ConsensusConfig(
            phase1_window_blocks=2,  # Shorter for testing
            phase2_window_blocks=2,
            verification_block_time_minutes=1,  # 1 minute for testing
            training_block_time_minutes=6.0,  # 6 minutes for testing
            merkle_proof_count=2  # Test with 2 Merkle proofs
        )
        
        self.model_config = ModelConfig()
        self.model_manager = Mock(spec=ModelManager)
        self.miner_address = "miner_test"
        
        self.consensus = EnhancedPogoConsensus(
            self.config, 
            self.model_manager, 
            self.miner_address
        )
        
        # Set up mock verifier stakes
        self.consensus.verifier_stakes = {
            "verifier_1": 100.0,
            "verifier_2": 150.0,
            "verifier_3": 200.0
        }
        self.consensus.total_stake = 450.0
    
    def test_complete_two_phase_workflow(self):
        """Test the complete two-phase verification workflow"""
        # Step 1: Create training block
        with patch.object(self.consensus, 'create_training_block') as mock_create:
            mock_block = Mock()
            mock_block.hash = "1234567890abcdef1234567890abcdef12345678"
            mock_block.hash_full_model_32 = "merkle_root_123"
            mock_block.hash_quant_4 = "quantized_hash_123"
            mock_block.index = 100
            mock_create.return_value = mock_block
            
            # Create enhanced training block
            result = self.consensus.create_enhanced_training_block("test_model")
            
            self.assertIsNotNone(result)
            self.assertEqual(result, mock_block)
            
            # Verify training block is tracked
            self.assertIn(mock_block.hash, self.consensus.training_blocks)
            self.assertEqual(self.consensus.training_blocks[mock_block.hash], 100)
        
        # Step 2: Check current phase (should be TRAINING)
        # Mock the chain length to simulate the training block being at index 100
        with patch.object(self.consensus, 'chain', [None] * 100):  # Simulate 100 blocks in chain
            current_phase = self.consensus.get_current_verification_phase(mock_block.hash)
            self.assertEqual(current_phase, VerificationPhase.TRAINING)
        
        # Step 3: Simulate moving to Phase 1
        # In practice, this would happen when the next block is created
        with patch.object(self.consensus, 'get_block_by_hash', return_value=mock_block):
            # Start Phase 1 verification
            phase1_data = self.consensus.start_phase1_verification(
                mock_block.hash, "verifier_1"
            )
            
            self.assertIsNotNone(phase1_data)
            self.assertEqual(phase1_data.block_hash, mock_block.hash)
            self.assertEqual(phase1_data.quantized_model_hash, mock_block.hash_quant_4)
            self.assertEqual(phase1_data.verifier_address, "verifier_1")
            
            # Submit Phase 1 verification results
            self.consensus.submit_phase1_verification_result(
                phase1_data, True, 0.01, "Loss improved as expected"
            )
            
            # Check Phase 1 summary
            phase1_summary = self.consensus.two_phase_system.get_phase1_verification_summary(mock_block.hash)
            self.assertEqual(phase1_summary["total_verifications"], 1)
            self.assertEqual(phase1_summary["positive_count"], 1)
            self.assertEqual(phase1_summary["positive_ratio"], 1.0)
        
        # Step 4: Simulate moving to Phase 2
        with patch.object(self.consensus, 'get_block_by_hash', return_value=mock_block):
            # Start Phase 2 verification
            phase2_data = self.consensus.start_phase2_verification(
                mock_block.hash, "verifier_2"
            )
            
            self.assertIsNotNone(phase2_data)
            self.assertEqual(phase2_data.block_hash, mock_block.hash)
            self.assertEqual(phase2_data.merkle_root, mock_block.hash_full_model_32)
            self.assertEqual(phase2_data.verifier_address, "verifier_2")
            self.assertEqual(len(phase2_data.leaf_indices), 2)  # Should have 2 Merkle proofs
            
            # Submit Phase 2 verification results
            self.consensus.submit_phase2_verification_result(
                phase2_data, True, "Merkle proofs verified successfully"
            )
            
            # Check Phase 2 summary
            phase2_summary = self.consensus.two_phase_system.get_phase2_verification_summary(mock_block.hash)
            self.assertEqual(phase2_summary["total_verifications"], 1)
            self.assertEqual(phase2_summary["positive_count"], 1)
            self.assertEqual(phase2_summary["positive_ratio"], 1.0)
        
        # Step 5: Check if ready for finalization
        should_finalize = self.consensus.should_proceed_to_finalization(mock_block.hash)
        self.assertTrue(should_finalize)
        
        # Step 6: Get comprehensive verification summary
        summary = self.consensus.get_phase_verification_summary(mock_block.hash)
        
        self.assertEqual(summary["block_hash"], mock_block.hash)
        self.assertEqual(summary["training_block_index"], 100)
        self.assertIn("current_phase", summary)
        self.assertIn("phase1", summary)
        self.assertIn("phase2", summary)
        
        # Verify both phases have positive results
        self.assertEqual(summary["phase1"]["positive_ratio"], 1.0)
        self.assertEqual(summary["phase2"]["positive_ratio"], 1.0)
    
    def test_verification_timeline(self):
        """Test verification timeline generation"""
        # Create a training block
        with patch.object(self.consensus, 'create_training_block') as mock_create:
            mock_block = Mock()
            mock_block.hash = "timeline_test_block"
            mock_block.index = 200
            mock_block.hash_full_model_32 = "merkle_root_timeline"
            mock_block.hash_quant_4 = "quantized_hash_timeline"
            mock_create.return_value = mock_block
            
            result = self.consensus.create_enhanced_training_block("timeline_model")
            self.assertIsNotNone(result)
        
        # Get verification timeline
        # Mock the chain length to simulate the training block being at index 200
        with patch.object(self.consensus, 'chain', [None] * 200):  # Simulate 200 blocks in chain
            timeline = self.consensus.get_verification_timeline(mock_block.hash)
            
            self.assertIsInstance(timeline, list)
            self.assertGreater(len(timeline), 0)
        
        # Check timeline structure
        for entry in timeline:
            self.assertIn("block_index", entry)
            self.assertIn("phase", entry)
            self.assertIn("block_time_seconds", entry)
            self.assertIn("is_active", entry)
            self.assertIn("should_start", entry)
    
    def test_negative_verification_workflow(self):
        """Test workflow with negative verification results"""
        # Create training block
        with patch.object(self.consensus, 'create_training_block') as mock_create:
            mock_block = Mock()
            mock_block.hash = "negative_test_block"
            mock_block.hash_full_model_32 = "merkle_root_negative"
            mock_block.hash_quant_4 = "quantized_hash_negative"
            mock_block.index = 300
            mock_create.return_value = mock_block
            
            self.consensus.create_enhanced_training_block("negative_model")
        
        # Phase 1: Submit negative verification
        with patch.object(self.consensus, 'get_block_by_hash', return_value=mock_block):
            phase1_data = self.consensus.start_phase1_verification(
                mock_block.hash, "verifier_1"
            )
            
            # Submit negative result
            self.consensus.submit_phase1_verification_result(
                phase1_data, False, None, "Loss did not improve as expected"
            )
            
            # Check summary
            phase1_summary = self.consensus.two_phase_system.get_phase1_verification_summary(mock_block.hash)
            self.assertEqual(phase1_summary["positive_count"], 0)
            self.assertEqual(phase1_summary["negative_count"], 1)
            self.assertEqual(phase1_summary["positive_ratio"], 0.0)
        
        # Phase 2: Submit negative verification
        with patch.object(self.consensus, 'get_block_by_hash', return_value=mock_block):
            phase2_data = self.consensus.start_phase2_verification(
                mock_block.hash, "verifier_2"
            )
            
            # Submit negative result
            self.consensus.submit_phase2_verification_result(
                phase2_data, False, "Merkle proof verification failed"
            )
            
            # Check summary
            phase2_summary = self.consensus.two_phase_system.get_phase2_verification_summary(mock_block.hash)
            self.assertEqual(phase2_summary["positive_count"], 0)
            self.assertEqual(phase2_summary["negative_count"], 1)
            self.assertEqual(phase2_summary["positive_ratio"], 0.0)
        
        # Should not proceed to finalization with negative results
        should_finalize = self.consensus.should_proceed_to_finalization(mock_block.hash)
        self.assertFalse(should_finalize)
    
    def test_multiple_verifiers_workflow(self):
        """Test workflow with multiple verifiers"""
        # Create training block
        with patch.object(self.consensus, 'create_training_block') as mock_create:
            mock_block = Mock()
            mock_block.hash = "multi_verifier_block"
            mock_block.hash_full_model_32 = "merkle_root_multi"
            mock_block.hash_quant_4 = "quantized_hash_multi"
            mock_block.index = 400
            mock_create.return_value = mock_block
            
            self.consensus.create_enhanced_training_block("multi_verifier_model")
        
        # Phase 1: Multiple verifiers
        with patch.object(self.consensus, 'get_block_by_hash', return_value=mock_block):
            verifiers = ["verifier_1", "verifier_2", "verifier_3"]
            results = [True, True, False]  # 2 positive, 1 negative
            
            for verifier, result in zip(verifiers, results):
                phase1_data = self.consensus.start_phase1_verification(
                    mock_block.hash, verifier
                )
                
                self.consensus.submit_phase1_verification_result(
                    phase1_data, result, 0.01 if result else None
                )
            
            # Check summary
            phase1_summary = self.consensus.two_phase_system.get_phase1_verification_summary(mock_block.hash)
            self.assertEqual(phase1_summary["total_verifications"], 3)
            self.assertEqual(phase1_summary["positive_count"], 2)
            self.assertEqual(phase1_summary["negative_count"], 1)
            self.assertAlmostEqual(phase1_summary["positive_ratio"], 2/3, places=2)
        
        # Phase 2: Multiple verifiers
        with patch.object(self.consensus, 'get_block_by_hash', return_value=mock_block):
            for verifier, result in zip(verifiers, results):
                phase2_data = self.consensus.start_phase2_verification(
                    mock_block.hash, verifier
                )
                
                self.consensus.submit_phase2_verification_result(
                    phase2_data, result, "Verification result"
                )
            
            # Check summary
            phase2_summary = self.consensus.two_phase_system.get_phase2_verification_summary(mock_block.hash)
            self.assertEqual(phase2_summary["total_verifications"], 3)
            self.assertEqual(phase2_summary["positive_count"], 2)
            self.assertEqual(phase2_summary["negative_count"], 1)
            self.assertAlmostEqual(phase2_summary["positive_ratio"], 2/3, places=2)
        
        # Should proceed to finalization (majority positive)
        should_finalize = self.consensus.should_proceed_to_finalization(mock_block.hash)
        self.assertTrue(should_finalize)
    
    def test_consensus_metrics_with_two_phase(self):
        """Test consensus metrics include two-phase verification data"""
        metrics = self.consensus.get_consensus_metrics()
        
        # Check existing metrics
        self.assertIn("merkle_proof_count", metrics)
        self.assertIn("attestation_window_blocks", metrics)
        self.assertIn("total_stake", metrics)
        
        # Check new two-phase metrics
        self.assertIn("training_blocks_tracked", metrics)
        self.assertIn("phase1_verifications", metrics)
        self.assertIn("phase2_verifications", metrics)
        self.assertIn("active_phase_timings", metrics)
        
        # Verify values
        self.assertEqual(metrics["merkle_proof_count"], 2)
        self.assertEqual(metrics["total_stake"], 450.0)
        self.assertEqual(metrics["verifier_count"], 3)

    def test_model_locked_during_full_verification_window(self):
        """Same model locked from training height through ``finalization_window`` (committed tip)."""
        cfg = ConsensusConfig(
            phase1_window_blocks=2,
            phase2_window_blocks=2,
            finalization_window=6,
            merkle_proof_count=2,
            verification_block_time_minutes=1,
            training_block_time_minutes=6.0,
        )
        mm = Mock(spec=ModelManager)
        c = EnhancedPogoConsensus(cfg, mm, "miner_test")
        mid = "locked_model"
        bh = "cd" + "0" * 38
        tbi = 10
        blk = Mock()
        blk.hash = bh
        blk.index = tbi
        blk.model_id = mid
        blk.training_steps = 20
        c.training_blocks[bh] = tbi

        def _gbh(h):
            return blk if h == bh else None

        fin = int(cfg.finalization_window)
        with patch.object(c, "get_block_by_hash", side_effect=_gbh):
            c.chain = [None] * (tbi + 1)
            self.assertTrue(c.is_model_training_locked(mid))

            c.chain = [None] * (tbi + 2 + 1)
            self.assertTrue(c.is_model_training_locked(mid))

            c.chain = [None] * (tbi + 4)
            self.assertTrue(c.is_model_training_locked(mid))

            c.chain = [None] * (tbi + fin + 2)
            self.assertFalse(c.is_model_training_locked(mid))

            self.assertFalse(c.is_model_training_locked("other_model"))


if __name__ == "__main__":
    unittest.main()
