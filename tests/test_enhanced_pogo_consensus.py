"""
Test suite for Enhanced PoGO Consensus with configurable Merkle proofs and attestation system.
"""

import unittest
import time
import tempfile
import os
from dataclasses import replace
from unittest.mock import Mock, patch

from dessin.consensus.enhanced_pogo_consensus import (
    EnhancedPogoConsensus, 
    MultiMerkleProof, 
    AttestationAggregation,
    AttestationResult
)
from dessin.consensus import PogoBlock
from dessin.runtime.config import ConsensusConfig, ModelConfig
from dessin.consensus.two_phase_verification import VerificationPhase
from dessin.models.model_manager import ModelManager
from dessin.consensus.transactions import AttestationTransaction


class TestEnhancedPogoConsensus(unittest.TestCase):
    """Test cases for Enhanced PoGO Consensus"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ConsensusConfig(
            merkle_proof_count=3,  # Test with 3 Merkle proofs
            attestation_window_blocks=5,
            slashing_threshold=0.33,
            attestation_threshold=0.67
        )
        
        self.model_config = ModelConfig()
        self.model_manager = Mock(spec=ModelManager)
        self.miner_address = "miner_123"
        
        self.consensus = EnhancedPogoConsensus(
            self.config, 
            self.model_manager, 
            self.miner_address
        )
        
        # Set up mock verifier stakes
        self.consensus.verifier_stakes = {
            "verifier_1": 100.0,
            "verifier_2": 150.0,
            "verifier_3": 200.0,
            "verifier_4": 50.0
        }
        self.consensus.total_stake = 500.0
    
    def test_merkle_proof_count_configuration(self):
        """Test that Merkle proof count is configurable"""
        self.assertEqual(self.consensus.merkle_proof_count, 3)
        
        # Test updating Merkle proof count
        self.assertTrue(self.consensus.update_merkle_proof_count(5))
        self.assertEqual(self.consensus.merkle_proof_count, 5)
        
        # Test invalid values
        self.assertFalse(self.consensus.update_merkle_proof_count(0))
        self.assertFalse(self.consensus.update_merkle_proof_count(10))
    
    def test_multiple_merkle_proof_creation(self):
        """Test creation of multiple Merkle proofs"""
        # Mock a block with proper hex hash
        block = Mock()
        block.hash = "1234567890abcdef1234567890abcdef12345678"  # Proper hex hash
        block.hash_full_model_32 = "test_merkle_root"
        
        # Create multiple Merkle proofs
        multi_proof = self.consensus._create_multiple_merkle_proofs(block)
        
        self.assertEqual(multi_proof.num_proofs, 3)
        self.assertEqual(len(multi_proof.leaf_indices), 3)
        self.assertEqual(len(multi_proof.leaf_data), 3)
        self.assertEqual(len(multi_proof.proof_hashes), 3)
        self.assertEqual(multi_proof.root_hash, "test_merkle_root")
        self.assertEqual(multi_proof.block_hash, "1234567890abcdef1234567890abcdef12345678")
        
        # Verify all proofs are unique
        self.assertEqual(len(set(multi_proof.leaf_indices)), 3)
    
    def test_attestation_submission(self):
        """Test attestation submission and tracking"""
        block_hash = "test_block_123"
        verifier_address = "verifier_1"
        
        verification_data = {
            "merkle_proof_verified": True,
            "quantized_model_verified": True,
            "data_availability_verified": True
        }
        
        # Submit positive attestation
        attestation = self.consensus.submit_attestation(
            block_hash, verifier_address, "positive", verification_data
        )
        
        self.assertIsInstance(attestation, AttestationTransaction)
        self.assertTrue(attestation.is_positive())
        self.assertEqual(attestation.block_hash, block_hash)
        self.assertEqual(attestation.sender, verifier_address)
        
        # Check that attestation was added to pending
        self.assertIn(block_hash, self.consensus.pending_attestations)
        self.assertEqual(len(self.consensus.pending_attestations[block_hash]), 1)
    
    def test_attestation_aggregation(self):
        """Test attestation aggregation logic"""
        block_hash = "test_block_456"
        
        # Submit multiple attestations
        verification_data = {"merkle_proof_verified": True}
        
        # Positive attestations (should total 250.0 stake)
        self.consensus.submit_attestation(block_hash, "verifier_1", "positive", verification_data)
        self.consensus.submit_attestation(block_hash, "verifier_2", "positive", verification_data)
        
        # Negative attestation (50.0 stake)
        self.consensus.submit_attestation(block_hash, "verifier_4", "negative", verification_data)
        
        # Aggregate attestations
        aggregation = self.consensus.aggregate_attestations(block_hash)
        
        self.assertEqual(len(aggregation.positive_attestations), 2)
        self.assertEqual(len(aggregation.negative_attestations), 1)
        self.assertEqual(aggregation.total_stake_positive, 250.0)
        self.assertEqual(aggregation.total_stake_negative, 50.0)
        self.assertEqual(aggregation.get_positive_ratio(), 0.5)  # 250/500
        self.assertEqual(aggregation.get_negative_ratio(), 0.1)  # 50/500
        
        # Should be pending (not enough positive attestations)
        self.assertEqual(aggregation.result, AttestationResult.PENDING)
    
    def test_finalization_threshold(self):
        """Test finalization threshold logic"""
        block_hash = "test_block_789"
        
        verification_data = {"merkle_proof_verified": True}
        
        # Submit enough positive attestations to meet threshold (67%)
        # Need at least 335.0 stake out of 500.0 total
        self.consensus.submit_attestation(block_hash, "verifier_1", "positive", verification_data)  # 100.0
        self.consensus.submit_attestation(block_hash, "verifier_2", "positive", verification_data)  # 150.0
        self.consensus.submit_attestation(block_hash, "verifier_3", "positive", verification_data)  # 200.0
        # Total: 450.0 stake (90% positive)
        
        aggregation = self.consensus.aggregate_attestations(block_hash)
        
        self.assertTrue(aggregation.should_finalize(0.67))
        self.assertEqual(aggregation.result, AttestationResult.FINALIZED)
    
    def test_slashing_threshold(self):
        """Test slashing threshold logic"""
        block_hash = "test_block_slash"
        
        verification_data = {"merkle_proof_verified": False, "evidence": "Merkle proof mismatch"}
        
        # Submit enough negative attestations to trigger slashing (33%)
        # Need at least 165.0 stake out of 500.0 total
        self.consensus.submit_attestation(block_hash, "verifier_2", "negative", verification_data)  # 150.0
        self.consensus.submit_attestation(block_hash, "verifier_4", "negative", verification_data)  # 50.0
        # Total: 200.0 stake (40% negative)
        
        aggregation = self.consensus.aggregate_attestations(block_hash)
        
        self.assertTrue(aggregation.should_slash(0.33))
        self.assertEqual(aggregation.result, AttestationResult.SLASHED)
    
    def test_merkle_proof_verification(self):
        """Test multiple Merkle proof verification"""
        block_hash = "test_merkle_verification"
        
        # Create mock multi-proof
        multi_proof = MultiMerkleProof(
            num_proofs=3,
            leaf_indices=[0, 1, 2],
            leaf_data=[b"leaf_0", b"leaf_1", b"leaf_2"],
            proof_hashes=[["proof_0"], ["proof_1"], ["proof_2"]],
            root_hash="test_root",
            block_hash=block_hash
        )
        
        # Add to cache
        self.consensus.merkle_proof_cache[block_hash] = multi_proof
        
        # Test verification
        result = self.consensus.verify_multiple_merkle_proofs(block_hash)
        self.assertTrue(result)
        
        # Test with non-existent block
        result = self.consensus.verify_multiple_merkle_proofs("non_existent")
        self.assertFalse(result)
    
    def test_attestation_summary(self):
        """Test attestation summary generation"""
        block_hash = "test_summary"
        
        verification_data = {"merkle_proof_verified": True}
        
        # Submit some attestations
        self.consensus.submit_attestation(block_hash, "verifier_1", "positive", verification_data)
        self.consensus.submit_attestation(block_hash, "verifier_2", "negative", verification_data)
        
        # Aggregate to create summary
        self.consensus.aggregate_attestations(block_hash)
        
        summary = self.consensus.get_attestation_summary(block_hash)
        
        self.assertEqual(summary["block_hash"], block_hash)
        self.assertEqual(summary["positive_count"], 1)
        self.assertEqual(summary["negative_count"], 1)
        self.assertEqual(summary["positive_stake"], 100.0)
        self.assertEqual(summary["negative_stake"], 150.0)
        self.assertIn("result", summary)
        self.assertIn("finalization_eligible", summary)
    
    def test_consensus_metrics(self):
        """Test consensus metrics collection"""
        metrics = self.consensus.get_consensus_metrics()
        
        self.assertEqual(metrics["merkle_proof_count"], 3)
        self.assertEqual(metrics["attestation_window_blocks"], 5)
        self.assertEqual(metrics["slashing_threshold"], 0.33)
        self.assertEqual(metrics["total_stake"], 500.0)
        self.assertEqual(metrics["verifier_count"], 4)
        self.assertIn("pending_attestations", metrics)
        self.assertIn("total_attestations", metrics)
    
    def test_enhanced_block_creation(self):
        """Test enhanced block creation with multiple Merkle proofs"""
        # Mock the parent create_training_block method
        with patch.object(self.consensus, 'create_training_block') as mock_create:
            mock_block = Mock()
            mock_block.hash = "1234567890abcdef1234567890abcdef12345678"  # Proper hex hash
            mock_block.hash_full_model_32 = "enhanced_merkle_root"
            mock_create.return_value = mock_block
            
            # Create enhanced block
            result = self.consensus.create_enhanced_training_block("test_model")
            
            self.assertIsNotNone(result)
            self.assertEqual(result, mock_block)
            
            # Check that Merkle proof cache was populated
            self.assertIn("1234567890abcdef1234567890abcdef12345678", self.consensus.merkle_proof_cache)
            
            # Check that block was updated with Merkle proof info
            self.assertIsNotNone(result.merkle_proof_full)

    def test_equivocation_triggers_slashing_when_staked(self):
        evil = "evil_equiv_miner"
        self.consensus.economic_system.balances[evil] = 1000.0
        self.assertTrue(self.consensus.economic_system.stake_tokens(evil, 100.0))

        genesis = self.consensus.chain[0]
        incumbent = _minimal_follower_block(genesis, miner=evil)
        incumbent.hash = "equiv_incumbent_hash_" + "a" * 40
        conflicting = replace(incumbent, hash="equiv_conflicting_hash_" + "b" * 40)

        self.consensus._handle_potential_equivocation(incumbent, conflicting)
        self.assertTrue(self.consensus.slashing_mechanism.is_address_banned(evil))

    def test_equivocation_ban_without_stake(self):
        evil = "evil_no_stake_equiv"
        genesis = self.consensus.chain[0]
        incumbent = _minimal_follower_block(genesis, miner=evil)
        incumbent.hash = "inc_" + "c" * 40
        conflicting = replace(incumbent, hash="alt_" + "d" * 40)

        self.consensus._handle_potential_equivocation(incumbent, conflicting)
        self.assertTrue(self.consensus.slashing_mechanism.is_address_banned(evil))


class TestMultiMerkleProof(unittest.TestCase):
    """Test cases for MultiMerkleProof class"""
    
    def test_merkle_proof_verification(self):
        """Test Merkle proof verification logic"""
        # Create a simple multi-proof
        multi_proof = MultiMerkleProof(
            num_proofs=2,
            leaf_indices=[0, 1],
            leaf_data=[b"test_leaf_0", b"test_leaf_1"],
            proof_hashes=[["proof_0"], ["proof_1"]],
            root_hash="test_root",
            block_hash="test_block"
        )
        
        # Test verification (simplified - in practice would verify actual hashes)
        result = multi_proof.verify_all()
        self.assertTrue(result)  # Should pass with our simplified implementation


class TestAttestationAggregation(unittest.TestCase):
    """Test cases for AttestationAggregation class"""
    
    def test_ratio_calculations(self):
        """Test ratio calculation methods"""
        # Create mock attestations
        positive_attestations = [Mock(), Mock()]
        negative_attestations = [Mock()]
        
        aggregation = AttestationAggregation(
            block_hash="test",
            positive_attestations=positive_attestations,
            negative_attestations=negative_attestations,
            total_stake_positive=200.0,
            total_stake_negative=100.0,
            total_stake=500.0,
            result=AttestationResult.PENDING,
            finalization_timestamp=time.time()
        )
        
        self.assertEqual(aggregation.get_positive_ratio(), 0.4)  # 200/500
        self.assertEqual(aggregation.get_negative_ratio(), 0.2)  # 100/500
    
    def test_finalization_logic(self):
        """Test finalization and slashing logic"""
        aggregation = AttestationAggregation(
            block_hash="test",
            positive_attestations=[],
            negative_attestations=[],
            total_stake_positive=400.0,  # 80% positive
            total_stake_negative=50.0,   # 10% negative
            total_stake=500.0,
            result=AttestationResult.PENDING,
            finalization_timestamp=time.time()
        )
        
        # Should finalize with 80% positive (above 67% threshold)
        self.assertTrue(aggregation.should_finalize(0.67))
        
        # Should not slash with 10% negative (below 33% threshold)
        self.assertFalse(aggregation.should_slash(0.33))


_FINALIZATION_PADDING = 20


def _minimal_follower_block(previous: PogoBlock, miner: str = "miner_123") -> PogoBlock:
    idx = previous.index + 1
    return PogoBlock(
        index=idx,
        timestamp=time.time(),
        previous_hash=previous.hash,
        miner=miner,
        model_id="t_model",
        training_data_hash="00" * 16,
        loss_before=1.0,
        loss_after=0.95,
        hash_full_model_32="aa" * 32,
        hash_quant_4="bb" * 32,
        vrf_proof=b"\x07" * 32,
        training_steps=2,
        learning_rate=0.01,
        batch_size=16,
        finalization_block=idx + _FINALIZATION_PADDING,
    )


class TestEnhancedPogoDynamicBlockTime(unittest.TestCase):
    """Enhanced consensus wiring to DynamicBlockTimeManager."""

    def setUp(self):
        self.config = ConsensusConfig(
            merkle_proof_count=1,
            attestation_window_blocks=5,
            slashing_threshold=0.33,
            attestation_threshold=0.67,
            block_time_adjustment_seconds=10,
            min_block_time_seconds=20,
            max_block_time_seconds=3600,
        )
        self.model_manager = Mock(spec=ModelManager)
        self.consensus = EnhancedPogoConsensus(
            self.config, self.model_manager, "miner_123"
        )
        self.model_manager.configure_mock(**{"is_model_available.return_value": True})

    def test_timing_helpers_delegate_to_dynamic_block_manager(self):
        block_hash = "timing_delegate_block"
        mgr = self.consensus.dynamic_block_time_manager
        mgr.current_block_time_seconds = 120.0

        self.consensus.start_verification_timing(block_hash, VerificationPhase.TRAINING)
        self.assertIn(block_hash, mgr.verification_timings)
        ended = self.consensus.end_verification_timing(block_hash, VerificationPhase.TRAINING)
        self.assertIsNotNone(ended)
        self.assertGreaterEqual(ended.actual_time_seconds, 0.0)

    def test_get_current_block_time_seconds_delegates(self):
        self.consensus.dynamic_block_time_manager.current_block_time_seconds = 333.0
        self.assertEqual(self.consensus.get_current_block_time_seconds(), 333.0)

    def test_get_block_time_performance_summary_empty(self):
        c = EnhancedPogoConsensus(
            ConsensusConfig(), self.model_manager, "m"
        )
        c.dynamic_block_time_manager.verification_timings.clear()
        summary = c.get_block_time_performance_summary()
        self.assertIn("error", summary)

    def test_apply_block_time_increase(self):
        genesis = self.consensus.chain[-1]
        block = _minimal_follower_block(genesis)
        block.block_time_adjustment_flag = "increase"
        block.current_block_time_seconds = 400.0
        block.adjustment_reason = "unit test increase"
        self.consensus.chain.append(block)

        self.assertTrue(self.consensus.apply_block_time_adjustment(block.hash))
        self.assertEqual(
            self.consensus.dynamic_block_time_manager.current_block_time_seconds,
            410.0,
        )

    def test_apply_block_time_decrease(self):
        genesis = self.consensus.chain[-1]
        block = _minimal_follower_block(genesis)
        block.block_time_adjustment_flag = "decrease"
        block.current_block_time_seconds = 55.0
        self.consensus.chain.append(block)

        self.assertTrue(self.consensus.apply_block_time_adjustment(block.hash))
        self.assertEqual(
            self.consensus.dynamic_block_time_manager.current_block_time_seconds,
            45.0,
        )

    def test_apply_block_time_decrease_clamped_to_minimum(self):
        genesis = self.consensus.chain[-1]
        block = _minimal_follower_block(genesis)
        block.block_time_adjustment_flag = "decrease"
        block.current_block_time_seconds = 28.0
        self.consensus.chain.append(block)

        self.assertTrue(self.consensus.apply_block_time_adjustment(block.hash))
        self.assertEqual(
            self.consensus.dynamic_block_time_manager.current_block_time_seconds,
            self.config.min_block_time_seconds,
        )

    def test_apply_block_time_maintain(self):
        genesis = self.consensus.chain[-1]
        block = _minimal_follower_block(genesis)
        block.block_time_adjustment_flag = "maintain"
        block.current_block_time_seconds = 300.0
        self.consensus.chain.append(block)

        self.assertTrue(self.consensus.apply_block_time_adjustment(block.hash))
        self.assertEqual(
            self.consensus.dynamic_block_time_manager.current_block_time_seconds,
            300.0,
        )

    def test_apply_block_time_returns_false_unknown_hash(self):
        self.assertFalse(self.consensus.apply_block_time_adjustment("no_such_hash"))


if __name__ == "__main__":
    unittest.main()
