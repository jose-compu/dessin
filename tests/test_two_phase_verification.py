"""
Test suite for Two-Phase Verification System.
"""

import math
import unittest
import time
from unittest.mock import Mock, patch

from dessin.consensus.two_phase_verification import (
    TwoPhaseVerificationSystem,
    VerificationPhase,
    Phase1VerificationData,
    Phase2VerificationData,
    PhaseTiming
)
from dessin.runtime.config import ConsensusConfig


class TestTwoPhaseVerificationSystem(unittest.TestCase):
    """Test cases for Two-Phase Verification System"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ConsensusConfig(
            phase1_window_blocks=3,
            phase2_window_blocks=3,
            verification_block_time_minutes=5,
            training_block_time_minutes=60.0,
            finalization_window=10
        )
        
        self.verification_system = TwoPhaseVerificationSystem(self.config)
    
    def test_phase_determination(self):
        """Test phase determination logic"""
        training_block_index = 100
        
        # Training phase
        phase = self.verification_system.get_current_phase(100, training_block_index)
        self.assertEqual(phase, VerificationPhase.TRAINING)
        
        # Phase 1 (blocks 101-103)
        phase = self.verification_system.get_current_phase(101, training_block_index)
        self.assertEqual(phase, VerificationPhase.PHASE1)
        
        phase = self.verification_system.get_current_phase(103, training_block_index)
        self.assertEqual(phase, VerificationPhase.PHASE1)
        
        # Phase 2 (blocks 104-106)
        phase = self.verification_system.get_current_phase(104, training_block_index)
        self.assertEqual(phase, VerificationPhase.PHASE2)
        
        phase = self.verification_system.get_current_phase(106, training_block_index)
        self.assertEqual(phase, VerificationPhase.PHASE2)
        
        # Finalization (blocks 107+)
        phase = self.verification_system.get_current_phase(107, training_block_index)
        self.assertEqual(phase, VerificationPhase.FINALIZATION)
    
    def test_block_time_for_phase(self):
        """Test block time calculation for different phases"""
        # Training phase should use training block time
        training_time = self.verification_system.get_block_time_for_phase(VerificationPhase.TRAINING)
        expected_training_time = float(
            math.ceil(float(self.config.training_block_time_minutes) * 60.0 - 1e-9)
        )
        self.assertEqual(training_time, expected_training_time)

        # Other phases should use verification block time (wall-clock mode)
        verification_time = self.verification_system.get_block_time_for_phase(VerificationPhase.PHASE1)
        expected_verification_time = self.config.verification_block_time_minutes * 60
        self.assertEqual(verification_time, expected_verification_time)

        verification_time = self.verification_system.get_block_time_for_phase(VerificationPhase.PHASE2)
        self.assertEqual(verification_time, expected_verification_time)

    def test_block_time_for_phase_matches_training_when_block_completion(self):
        """Verification slot duration tracks training cadence when using block-based completion."""
        cfg = ConsensusConfig(
            phase1_window_blocks=3,
            phase2_window_blocks=3,
            verification_block_time_minutes=30,
            training_block_time_minutes=1.0,
            verification_phase_completion_in_blocks=True,
            finalization_window=10,
        )
        vs = TwoPhaseVerificationSystem(cfg)
        slot = float(math.ceil(float(cfg.training_block_time_minutes) * 60.0 - 1e-9))
        self.assertEqual(vs.get_block_time_for_phase(VerificationPhase.PHASE1), slot)
        self.assertEqual(vs.get_block_time_for_phase(VerificationPhase.PHASE2), slot)
    
    def test_phase_timing(self):
        """Test phase timing functionality"""
        block_hash = "test_block_123"
        training_block_index = 97
        current_block_index = 100
        phase = VerificationPhase.PHASE1

        timing = self.verification_system.start_phase_timing(
            block_hash,
            phase,
            training_block_index=training_block_index,
            current_block_index=current_block_index,
        )

        self.assertIsInstance(timing, PhaseTiming)
        self.assertEqual(timing.block_index, current_block_index)
        self.assertEqual(timing.phase, phase)
        self.assertTrue(timing.is_active)
        self.assertGreater(timing.end_time, timing.start_time)

        progress = self.verification_system.get_phase_progress(block_hash)
        self.assertGreaterEqual(progress, 0.0)
        self.assertLessEqual(progress, 1.0)

    def test_phase_completion_in_blocks(self):
        """Phase completes when chain tip crosses the exclusive boundary, not wall-clock duration."""
        cfg = ConsensusConfig(
            phase1_window_blocks=3,
            phase2_window_blocks=3,
            verification_phase_completion_in_blocks=True,
            training_block_time_minutes=1.0,
            verification_block_time_minutes=30,
            finalization_window=10,
        )
        vs = TwoPhaseVerificationSystem(cfg)
        block_hash = "train_blk"
        vs.start_phase_timing(
            block_hash,
            VerificationPhase.PHASE1,
            training_block_index=100,
            current_block_index=101,
        )
        boundary = 100 + cfg.phase1_window_blocks + 1
        self.assertFalse(vs.is_phase_complete(block_hash, chain_tip_index=boundary - 1))
        self.assertTrue(vs.is_phase_complete(block_hash, chain_tip_index=boundary))
    
    def test_phase1_verification_data_generation(self):
        """Test Phase 1 verification data generation"""
        block_hash = "test_block_456"
        quantized_model_hash = "quantized_hash_123"
        verifier_address = "verifier_1"
        
        verification_data = self.verification_system.generate_phase1_verification_data(
            block_hash, quantized_model_hash, verifier_address
        )
        
        self.assertIsInstance(verification_data, Phase1VerificationData)
        self.assertEqual(verification_data.block_hash, block_hash)
        self.assertEqual(verification_data.quantized_model_hash, quantized_model_hash)
        self.assertEqual(verification_data.verifier_address, verifier_address)
        self.assertIsNotNone(verification_data.verification_dataset_hash)
        self.assertEqual(verification_data.expected_loss_improvement, self.config.min_loss_improvement)
        self.assertIsNone(verification_data.verification_result)
    
    def test_phase2_verification_data_generation(self):
        """Test Phase 2 verification data generation"""
        block_hash = "test_block_789"
        merkle_root = "merkle_root_123"
        verifier_address = "verifier_2"
        num_proofs = 3
        
        verification_data = self.verification_system.generate_phase2_verification_data(
            block_hash, merkle_root, verifier_address, num_proofs
        )
        
        self.assertIsInstance(verification_data, Phase2VerificationData)
        self.assertEqual(verification_data.block_hash, block_hash)
        self.assertEqual(verification_data.merkle_root, merkle_root)
        self.assertEqual(verification_data.verifier_address, verifier_address)
        self.assertEqual(len(verification_data.leaf_indices), num_proofs)
        self.assertEqual(len(verification_data.leaf_data), num_proofs)
        self.assertEqual(len(verification_data.proof_hashes), num_proofs)
        self.assertIsNotNone(verification_data.random_seed)
        self.assertIsNone(verification_data.verification_result)
    
    def test_phase1_verification_submission(self):
        """Test Phase 1 verification submission"""
        block_hash = "test_block_phase1"
        verification_data = self.verification_system.generate_phase1_verification_data(
            block_hash, "quantized_hash", "verifier_1"
        )
        
        # Submit positive verification
        self.verification_system.submit_phase1_verification(
            verification_data, True, 0.01, "Loss improved as expected"
        )
        
        self.assertTrue(verification_data.verification_result)
        self.assertEqual(verification_data.actual_loss_improvement, 0.01)
        self.assertEqual(verification_data.evidence, "Loss improved as expected")
        
        # Check summary
        summary = self.verification_system.get_phase1_verification_summary(block_hash)
        self.assertEqual(summary["total_verifications"], 1)
        self.assertEqual(summary["positive_count"], 1)
        self.assertEqual(summary["negative_count"], 0)
        self.assertEqual(summary["positive_ratio"], 1.0)
    
    def test_phase2_verification_submission(self):
        """Test Phase 2 verification submission"""
        block_hash = "test_block_phase2"
        verification_data = self.verification_system.generate_phase2_verification_data(
            block_hash, "merkle_root", "verifier_2", 2
        )
        
        # Submit negative verification
        self.verification_system.submit_phase2_verification(
            verification_data, False, "Merkle proof mismatch detected"
        )
        
        self.assertFalse(verification_data.verification_result)
        self.assertEqual(verification_data.evidence, "Merkle proof mismatch detected")
        
        # Check summary
        summary = self.verification_system.get_phase2_verification_summary(block_hash)
        self.assertEqual(summary["total_verifications"], 1)
        self.assertEqual(summary["positive_count"], 0)
        self.assertEqual(summary["negative_count"], 1)
        self.assertEqual(summary["positive_ratio"], 0.0)
    
    def test_multiple_verifications(self):
        """Test multiple verifications for the same block"""
        block_hash = "test_block_multiple"
        
        # Submit multiple Phase 1 verifications
        for i in range(3):
            verification_data = self.verification_system.generate_phase1_verification_data(
                block_hash, "quantized_hash", f"verifier_{i}"
            )
            
            result = i < 2  # First 2 positive, last negative
            self.verification_system.submit_phase1_verification(
                verification_data, result, 0.01 if result else None
            )
        
        summary = self.verification_system.get_phase1_verification_summary(block_hash)
        self.assertEqual(summary["total_verifications"], 3)
        self.assertEqual(summary["positive_count"], 2)
        self.assertEqual(summary["negative_count"], 1)
        self.assertAlmostEqual(summary["positive_ratio"], 2/3, places=2)
    
    def test_verification_timeline(self):
        """Test verification timeline generation"""
        training_block_index = 100
        current_block_index = 105
        
        timeline = self.verification_system.get_verification_timeline(
            training_block_index, current_block_index
        )
        
        self.assertEqual(len(timeline), 6)  # Blocks 100-105
        
        # Check training block
        self.assertEqual(timeline[0]["block_index"], 100)
        self.assertEqual(timeline[0]["phase"], "training")
        
        # Check Phase 1 blocks
        for i in range(1, 4):  # Blocks 101-103
            self.assertEqual(timeline[i]["phase"], "phase1")
        
        # Check Phase 2 blocks
        for i in range(4, 6):  # Blocks 104-105
            self.assertEqual(timeline[i]["phase"], "phase2")
    
    def test_should_proceed_to_next_phase(self):
        """Test phase progression logic"""
        block_hash = "test_block_progression"
        
        # Initially should not proceed (no verifications)
        should_proceed = self.verification_system.should_proceed_to_next_phase(
            block_hash, VerificationPhase.PHASE1
        )
        self.assertFalse(should_proceed)
        
        # Add positive verifications
        for i in range(3):
            verification_data = self.verification_system.generate_phase1_verification_data(
                block_hash, "quantized_hash", f"verifier_{i}"
            )
            self.verification_system.submit_phase1_verification(verification_data, True)
        
        # Should proceed with sufficient positive verifications
        should_proceed = self.verification_system.should_proceed_to_next_phase(
            block_hash, VerificationPhase.PHASE1
        )
        self.assertTrue(should_proceed)
    
    def test_cleanup_old_verifications(self):
        """Test cleanup of old verification data"""
        # Add some old verification data
        old_block_hash = "old_block_123"
        verification_data = self.verification_system.generate_phase1_verification_data(
            old_block_hash, "quantized_hash", "verifier_1"
        )
        self.verification_system.submit_phase1_verification(verification_data, True)
        
        # Verify data exists
        summary = self.verification_system.get_phase1_verification_summary(old_block_hash)
        self.assertEqual(summary["total_verifications"], 1)
        
        # Cleanup (simplified test - in practice would use actual block indices)
        self.verification_system.cleanup_old_verifications(1000, 100)
        
        # Data should still exist (cleanup is simplified in this implementation)
        summary = self.verification_system.get_phase1_verification_summary(old_block_hash)
        self.assertEqual(summary["total_verifications"], 1)


class TestPhaseTiming(unittest.TestCase):
    """Test cases for PhaseTiming class"""
    
    def test_phase_timing_creation(self):
        """Test PhaseTiming object creation"""
        timing = PhaseTiming(
            block_index=100,
            phase=VerificationPhase.PHASE1,
            start_time=time.time(),
            end_time=time.time() + 300,  # 5 minutes
            block_time_seconds=300,
            is_active=True
        )
        
        self.assertEqual(timing.block_index, 100)
        self.assertEqual(timing.phase, VerificationPhase.PHASE1)
        self.assertTrue(timing.is_active)
        self.assertGreater(timing.end_time, timing.start_time)
        self.assertEqual(timing.block_time_seconds, 300)


if __name__ == "__main__":
    unittest.main()
