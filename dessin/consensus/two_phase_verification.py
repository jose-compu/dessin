"""
Two-Phase Verification System for PoGO Protocol
Implements the whitepaper's two-phase verification with proper timing.
"""

import math
import time
import random
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..runtime.config import ConsensusConfig


class VerificationPhase(Enum):
    """Verification phases in the two-phase system"""
    TRAINING = "training"           # Block N: Training and commitment
    PHASE1 = "phase1"              # Block N + w/2: Quantized model verification
    PHASE2 = "phase2"              # Block N + w/2: Random leaf challenge
    FINALIZATION = "finalization"  # Block N + w: Attestation aggregation


@dataclass
class PhaseTiming:
    """Timing information for verification phases"""
    block_index: int
    phase: VerificationPhase
    start_time: float
    end_time: float
    block_time_seconds: float
    is_active: bool = False
    # When using block-based completion: chain height at/after which this phase is done
    training_block_index: Optional[int] = None
    completion_block_exclusive: Optional[int] = None
    phase_start_block_inclusive: Optional[int] = None


@dataclass
class Phase1VerificationData:
    """Data for Phase 1 verification (quantized model)"""
    block_hash: str
    quantized_model_hash: str
    verification_dataset_hash: str
    expected_loss_improvement: float
    verification_timestamp: float
    verifier_address: str
    verification_result: Optional[bool] = None
    actual_loss_improvement: Optional[float] = None
    evidence: Optional[str] = None


@dataclass
class Phase2VerificationData:
    """Data for Phase 2 verification (random leaf challenge)"""
    block_hash: str
    merkle_root: str
    random_seed: str
    leaf_indices: List[int]
    leaf_data: List[bytes]
    proof_hashes: List[List[str]]
    verification_timestamp: float
    verifier_address: str
    verification_result: Optional[bool] = None
    evidence: Optional[str] = None


class TwoPhaseVerificationSystem:
    """Two-phase verification system for PoGO protocol"""
    
    def __init__(self, config: ConsensusConfig):
        self.config = config
        
        # Phase timing tracking
        self.phase_timings: Dict[str, PhaseTiming] = {}
        
        # Verification data storage
        self.phase1_verifications: Dict[str, List[Phase1VerificationData]] = {}
        self.phase2_verifications: Dict[str, List[Phase2VerificationData]] = {}
        
        # Block timing cache
        self.block_timing_cache: Dict[int, float] = {}
    
    def get_current_phase(self, block_index: int, training_block_index: int) -> VerificationPhase:
        """Determine the current verification phase for a block"""
        blocks_since_training = block_index - training_block_index
        
        if blocks_since_training == 0:
            return VerificationPhase.TRAINING
        
        phase1_end = self.config.phase1_window_blocks
        phase2_end = phase1_end + self.config.phase2_window_blocks
        finalization_end = self.config.finalization_window
        
        if blocks_since_training <= phase1_end:
            return VerificationPhase.PHASE1
        elif blocks_since_training <= phase2_end:
            return VerificationPhase.PHASE2
        elif blocks_since_training <= finalization_end:
            return VerificationPhase.FINALIZATION
        else:
            return VerificationPhase.FINALIZATION  # Default to finalization
    
    def get_block_time_for_phase(self, phase: VerificationPhase) -> float:
        """Seconds per slot for this phase: training uses training cadence; verification uses the same
        when ``verification_phase_completion_in_blocks`` (one slot, proportional to chain time)."""
        train_sec = float(
            math.ceil(float(self.config.training_block_time_minutes) * 60.0 - 1e-9)
        )
        if phase == VerificationPhase.TRAINING:
            return train_sec
        if getattr(self.config, "verification_phase_completion_in_blocks", False):
            return train_sec
        return float(
            math.ceil(float(self.config.verification_block_time_minutes) * 60.0 - 1e-9)
        )

    def _completion_block_exclusive(
        self, phase: VerificationPhase, training_block_index: int
    ) -> int:
        c = self.config
        if phase == VerificationPhase.TRAINING:
            return training_block_index + 1
        if phase == VerificationPhase.PHASE1:
            return training_block_index + c.phase1_window_blocks + 1
        if phase == VerificationPhase.PHASE2:
            return (
                training_block_index
                + c.phase1_window_blocks
                + c.phase2_window_blocks
                + 1
            )
        if phase == VerificationPhase.FINALIZATION:
            return training_block_index + c.finalization_window + 1
        return training_block_index + 1

    def _phase_start_block_inclusive(
        self, phase: VerificationPhase, training_block_index: int
    ) -> int:
        c = self.config
        if phase == VerificationPhase.TRAINING:
            return training_block_index
        if phase == VerificationPhase.PHASE1:
            return training_block_index + 1
        if phase == VerificationPhase.PHASE2:
            return training_block_index + c.phase1_window_blocks + 1
        if phase == VerificationPhase.FINALIZATION:
            return training_block_index + c.phase1_window_blocks + c.phase2_window_blocks + 1
        return training_block_index
    
    def should_start_phase(self, block_index: int, training_block_index: int) -> bool:
        """Check if a verification phase should start at the current block"""
        phase = self.get_current_phase(block_index, training_block_index)
        blocks_since_training = block_index - training_block_index
        
        if phase == VerificationPhase.PHASE1:
            return blocks_since_training == 1  # Start Phase 1 immediately after training
        elif phase == VerificationPhase.PHASE2:
            return blocks_since_training == self.config.phase1_window_blocks + 1
        elif phase == VerificationPhase.FINALIZATION:
            return blocks_since_training == (self.config.phase1_window_blocks + 
                                           self.config.phase2_window_blocks + 1)
        
        return False
    
    def start_phase_timing(
        self,
        block_hash: str,
        phase: VerificationPhase,
        *,
        training_block_index: int,
        current_block_index: int,
    ) -> PhaseTiming:
        """Start timing for a verification phase (wall clock and/or block-index completion)."""
        block_time = self.get_block_time_for_phase(phase)
        current_time = time.time()
        use_blocks = getattr(self.config, "verification_phase_completion_in_blocks", False)

        completion_exclusive: Optional[int] = None
        phase_start_inc: Optional[int] = None
        if use_blocks:
            completion_exclusive = self._completion_block_exclusive(phase, training_block_index)
            phase_start_inc = self._phase_start_block_inclusive(phase, training_block_index)

        timing = PhaseTiming(
            block_index=current_block_index,
            phase=phase,
            start_time=current_time,
            end_time=current_time + block_time,
            block_time_seconds=block_time,
            is_active=True,
            training_block_index=training_block_index,
            completion_block_exclusive=completion_exclusive,
            phase_start_block_inclusive=phase_start_inc,
        )

        self.phase_timings[block_hash] = timing
        return timing

    def is_phase_complete(
        self, block_hash: str, chain_tip_index: Optional[int] = None
    ) -> bool:
        """Complete when wall clock passes end_time, or (block mode) when ``chain_tip_index`` reaches
        the configured exclusive boundary."""
        if block_hash not in self.phase_timings:
            return False

        timing = self.phase_timings[block_hash]
        use_blocks = getattr(self.config, "verification_phase_completion_in_blocks", False)
        if (
            use_blocks
            and timing.completion_block_exclusive is not None
            and chain_tip_index is not None
        ):
            return int(chain_tip_index) >= int(timing.completion_block_exclusive)
        return time.time() >= timing.end_time

    def get_phase_progress(
        self, block_hash: str, chain_tip_index: Optional[int] = None
    ) -> float:
        """Progress 0..1: wall-clock fraction, or block fraction within the phase window when block mode."""
        if block_hash not in self.phase_timings:
            return 0.0

        timing = self.phase_timings[block_hash]
        if not timing.is_active:
            return 1.0

        use_blocks = getattr(self.config, "verification_phase_completion_in_blocks", False)
        if (
            use_blocks
            and timing.completion_block_exclusive is not None
            and timing.phase_start_block_inclusive is not None
            and chain_tip_index is not None
        ):
            span = int(timing.completion_block_exclusive) - int(
                timing.phase_start_block_inclusive
            )
            if span <= 0:
                return 1.0
            done = int(chain_tip_index) - int(timing.phase_start_block_inclusive)
            return max(0.0, min(1.0, float(done) / float(span)))

        elapsed = time.time() - timing.start_time
        return min(1.0, elapsed / timing.block_time_seconds)
    
    def generate_phase1_verification_data(
        self, 
        block_hash: str, 
        quantized_model_hash: str,
        verifier_address: str
    ) -> Phase1VerificationData:
        """Generate Phase 1 verification data (quantized model verification)"""
        # Generate deterministic verification dataset hash
        verification_seed = hashlib.sha256(f"{block_hash}_phase1".encode()).hexdigest()
        
        return Phase1VerificationData(
            block_hash=block_hash,
            quantized_model_hash=quantized_model_hash,
            verification_dataset_hash=verification_seed,
            expected_loss_improvement=self.config.min_loss_improvement,
            verification_timestamp=time.time(),
            verifier_address=verifier_address
        )
    
    def generate_phase2_verification_data(
        self, 
        block_hash: str, 
        merkle_root: str,
        verifier_address: str,
        num_proofs: int = 1
    ) -> Phase2VerificationData:
        """Generate Phase 2 verification data (random leaf challenge)"""
        # Generate deterministic random seed for leaf selection
        random_seed = hashlib.sha256(f"{block_hash}_phase2_{block_hash[:8]}".encode()).hexdigest()
        
        # Use seed for deterministic randomness
        random.seed(int(random_seed[:8], 16))
        
        # Generate random leaf indices
        leaf_indices = []
        leaf_data = []
        proof_hashes = []
        
        for i in range(num_proofs):
            # Simulate random leaf selection (in practice, would use actual Merkle tree)
            leaf_index = random.randint(0, 99)  # Assume 100 leaves max
            while leaf_index in leaf_indices:
                leaf_index = random.randint(0, 99)
            
            leaf_indices.append(leaf_index)
            
            # Simulate leaf data and proof path
            leaf_data.append(f"leaf_{leaf_index}_{block_hash}".encode())
            proof_hashes.append([f"proof_hash_{j}_{leaf_index}" for j in range(3)])
        
        return Phase2VerificationData(
            block_hash=block_hash,
            merkle_root=merkle_root,
            random_seed=random_seed,
            leaf_indices=leaf_indices,
            leaf_data=leaf_data,
            proof_hashes=proof_hashes,
            verification_timestamp=time.time(),
            verifier_address=verifier_address
        )
    
    def submit_phase1_verification(
        self, 
        verification_data: Phase1VerificationData,
        verification_result: bool,
        actual_loss_improvement: Optional[float] = None,
        evidence: Optional[str] = None
    ) -> None:
        """Submit Phase 1 verification result"""
        verification_data.verification_result = verification_result
        verification_data.actual_loss_improvement = actual_loss_improvement
        verification_data.evidence = evidence
        
        if verification_data.block_hash not in self.phase1_verifications:
            self.phase1_verifications[verification_data.block_hash] = []
        
        self.phase1_verifications[verification_data.block_hash].append(verification_data)
    
    def submit_phase2_verification(
        self, 
        verification_data: Phase2VerificationData,
        verification_result: bool,
        evidence: Optional[str] = None
    ) -> None:
        """Submit Phase 2 verification result"""
        verification_data.verification_result = verification_result
        verification_data.evidence = evidence
        
        if verification_data.block_hash not in self.phase2_verifications:
            self.phase2_verifications[verification_data.block_hash] = []
        
        self.phase2_verifications[verification_data.block_hash].append(verification_data)
    
    def get_phase1_verification_summary(
        self, block_hash: str, chain_tip_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get summary of Phase 1 verifications for a block"""
        verifications = self.phase1_verifications.get(block_hash, [])

        if not verifications:
            return {"status": "no_verifications"}

        positive_count = sum(1 for v in verifications if v.verification_result is True)
        negative_count = sum(1 for v in verifications if v.verification_result is False)
        pending_count = sum(1 for v in verifications if v.verification_result is None)

        return {
            "block_hash": block_hash,
            "total_verifications": len(verifications),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "pending_count": pending_count,
            "positive_ratio": positive_count / len(verifications) if verifications else 0.0,
            "phase_complete": self.is_phase_complete(block_hash, chain_tip_index),
            "phase_progress": self.get_phase_progress(block_hash, chain_tip_index),
        }

    def get_phase2_verification_summary(
        self, block_hash: str, chain_tip_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get summary of Phase 2 verifications for a block"""
        verifications = self.phase2_verifications.get(block_hash, [])

        if not verifications:
            return {"status": "no_verifications"}

        positive_count = sum(1 for v in verifications if v.verification_result is True)
        negative_count = sum(1 for v in verifications if v.verification_result is False)
        pending_count = sum(1 for v in verifications if v.verification_result is None)

        return {
            "block_hash": block_hash,
            "total_verifications": len(verifications),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "pending_count": pending_count,
            "positive_ratio": positive_count / len(verifications) if verifications else 0.0,
            "phase_complete": self.is_phase_complete(block_hash, chain_tip_index),
            "phase_progress": self.get_phase_progress(block_hash, chain_tip_index),
        }

    def should_proceed_to_next_phase(
        self,
        block_hash: str,
        phase: VerificationPhase,
        chain_tip_index: Optional[int] = None,
    ) -> bool:
        """Check if verification should proceed to the next phase"""
        if phase == VerificationPhase.PHASE1:
            summary = self.get_phase1_verification_summary(block_hash, chain_tip_index)
            return summary.get("positive_ratio", 0.0) >= 0.5 or summary.get(
                "phase_complete", False
            )

        if phase == VerificationPhase.PHASE2:
            summary = self.get_phase2_verification_summary(block_hash, chain_tip_index)
            return summary.get("positive_ratio", 0.0) >= 0.5 or summary.get(
                "phase_complete", False
            )

        return True
    
    def get_verification_timeline(self, training_block_index: int, current_block_index: int) -> List[Dict[str, Any]]:
        """Get the verification timeline for a training block"""
        timeline = []
        
        for block_offset in range(current_block_index - training_block_index + 1):
            block_index = training_block_index + block_offset
            phase = self.get_current_phase(block_index, training_block_index)
            
            timeline.append({
                "block_index": block_index,
                "phase": phase.value,
                "block_time_seconds": self.get_block_time_for_phase(phase),
                "is_active": block_offset == (current_block_index - training_block_index),
                "should_start": self.should_start_phase(block_index, training_block_index)
            })
        
        return timeline
    
    def cleanup_old_verifications(self, current_block_index: int, retention_blocks: int = 100) -> None:
        """Clean up old verification data to prevent memory leaks"""
        cutoff_block = current_block_index - retention_blocks
        
        # Clean up phase timings
        to_remove = []
        for block_hash, timing in self.phase_timings.items():
            if timing.block_index < cutoff_block:
                to_remove.append(block_hash)
        
        for block_hash in to_remove:
            del self.phase_timings[block_hash]
        
        # Clean up verification data
        for block_hash in list(self.phase1_verifications.keys()):
            # This is a simplified cleanup - in practice would need block index mapping
            if len(self.phase1_verifications[block_hash]) > 0:
                # Keep recent verifications
                pass
        
        for block_hash in list(self.phase2_verifications.keys()):
            # This is a simplified cleanup - in practice would need block index mapping
            if len(self.phase2_verifications[block_hash]) > 0:
                # Keep recent verifications
                pass
