"""
Dynamic Block Time Management System
===================================

Adjusts inter-block time from verification/utilization signals. Asymmetric by default:
shorten the interval gently when validation has slack (``block_time_ramp_up_capacity_pct``,
default 5% per step—stable under flaky nodes/links); lengthen the interval aggressively
under stress (``block_time_ramp_down_capacity_pct``, default 15% per step) so heavy load
quickly gets a longer slot.
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..runtime.config import ConsensusConfig
from .two_phase_verification import VerificationPhase
from ..runtime.pretty_console import pretty_print, format_interval_sec, interval_change_kind, adjustment_flag_kind


class BlockTimeAdjustment(Enum):
    """Types of block time adjustments"""
    INCREASE = "increase"  # Increase block time by 10 seconds
    DECREASE = "decrease"  # Decrease block time by 10 seconds
    MAINTAIN = "maintain"  # Keep current block time


@dataclass
class VerificationTiming:
    """Timing information for verification phases"""
    phase: VerificationPhase
    allocated_time_seconds: float
    actual_time_seconds: float
    start_time: float
    end_time: float
    timeout_threshold: float
    is_timeout: bool = False
    
    def get_progress_ratio(self) -> float:
        """Get the progress ratio (0.0 to 1.0+)"""
        if self.allocated_time_seconds == 0:
            return 0.0
        return self.actual_time_seconds / self.allocated_time_seconds
    
    def is_exceeding_threshold(self) -> bool:
        """Check if verification is exceeding the timeout threshold"""
        return self.get_progress_ratio() >= self.timeout_threshold


@dataclass
class BlockTimeAdjustmentProposal:
    """Proposal for block time adjustment"""
    adjustment_type: BlockTimeAdjustment
    current_block_time_seconds: float
    proposed_block_time_seconds: float
    reason: str
    verification_timings: List[VerificationTiming]
    timestamp: float
    proposer: str
    
    def is_valid(self, config: ConsensusConfig) -> bool:
        """Check if the proposed adjustment is within valid bounds"""
        return (config.min_block_time_seconds <= 
                self.proposed_block_time_seconds <= 
                config.max_block_time_seconds)


class DynamicBlockTimeManager:
    """Manages dynamic block time adjustments based on verification performance"""
    
    def __init__(
        self, 
        config: ConsensusConfig,
        use_adaptive_timing: bool = True,
        use_logarithmic_scaling: bool = False,
        max_scaling_factor: float = 1000.0,
        min_block_time_seconds: float = 40.0
    ):
        self.config = config
        self.current_block_time_seconds = self._get_initial_block_time()
        
        # Adaptive timing configuration (simpler approach)
        self.use_adaptive_timing = use_adaptive_timing
        self.min_block_time_seconds = min_block_time_seconds
        self.increase_threshold = 0.8  # Utilization above this → lengthen interval (stress path)
        self.decrease_threshold = 0.4  # Utilization below this → shorten interval (slack path)
        self.ramp_up_capacity_pct = float(
            getattr(config, "block_time_ramp_up_capacity_pct", 0.05)
        )
        self.ramp_down_capacity_pct = float(
            getattr(config, "block_time_ramp_down_capacity_pct", 0.15)
        )
        self.slack_step_multiplier = float(
            getattr(config, "block_time_slack_step_multiplier", 1.45)
        )
        self.stress_step_multiplier = float(
            getattr(config, "block_time_stress_step_multiplier", 0.55)
        )
        
        # Legacy scaling configuration (complex approach)
        self.use_logarithmic_scaling = use_logarithmic_scaling
        self.max_scaling_factor = max_scaling_factor
        
        # Tracking verification timings
        self.verification_timings: Dict[str, List[VerificationTiming]] = {}  # block_hash -> timings
        self.recent_adjustments: List[BlockTimeAdjustmentProposal] = []
        
        # Network consensus tracking
        self.pending_adjustments: Dict[str, List[BlockTimeAdjustmentProposal]] = {}  # block_hash -> proposals
        self.approved_adjustments: Dict[str, BlockTimeAdjustmentProposal] = {}  # block_hash -> approved proposal
        
        self.current_block_time_seconds = self._clamp_block_time_seconds(
            self.current_block_time_seconds
        )
        pretty_print(
            f"Dynamic block time ready │ target slot {format_interval_sec(self.current_block_time_seconds)} "
            f"│ consensus floor {format_interval_sec(self.config.min_block_time_seconds)} … "
            f"ceiling {format_interval_sec(self.config.max_block_time_seconds)} "
            f"│ adaptive floor {format_interval_sec(self.min_block_time_seconds)}",
            kind="clock",
        )
        if use_adaptive_timing:
            pretty_print(
                f"Mode: adaptive │ 🐢 lengthen if validation > {self.increase_threshold*100:.0f}% "
                f"(+{self.ramp_down_capacity_pct*100:.1f}%/step) │ "
                f"⚡ shorten if < {self.decrease_threshold*100:.0f}% "
                f"(-{self.ramp_up_capacity_pct*100:.1f}%/step)",
                kind="info",
            )
        else:
            pretty_print(
                f"Mode: model-based ({'logarithmic' if use_logarithmic_scaling else 'step-based'}) │ "
                f"step ±{self.config.block_time_adjustment_seconds}s",
                kind="info",
            )
    
    def _bound_clamp_seconds(self, seconds: float) -> float:
        """Keep block time within consensus min/max (float, before whole-second ceil)."""
        lo = float(self.config.min_block_time_seconds)
        hi = float(self.config.max_block_time_seconds)
        if hi < lo:
            hi = lo
        return max(lo, min(float(seconds), hi))

    def _whole_seconds_ceil(self, seconds: float) -> float:
        """Smallest integer-second duration not less than ``seconds`` (for sleeps and slots)."""
        return float(math.ceil(float(seconds) - 1e-9))

    def _clamp_block_time_seconds(self, seconds: float) -> float:
        """Bounds + whole-second ceiling (public path for proposals and current time)."""
        return self._whole_seconds_ceil(self._bound_clamp_seconds(seconds))
    
    def _get_initial_block_time(self) -> float:
        """Target block time from training schedule (minutes→seconds, ceiled), then clamped."""
        if hasattr(self.config, "training_block_time_minutes"):
            raw = math.ceil(float(self.config.training_block_time_minutes) * 60.0 - 1e-9)
        else:
            raw = 30.0
        return self._clamp_block_time_seconds(float(raw))
    
    def adjust_block_time_adaptive(
        self,
        validation_time_seconds: float,
        current_block_time_seconds: Optional[float] = None
    ) -> Tuple[float, str]:
        """
        Simple adaptive block time adjustment based on validation utilization.

        Asymmetric:
        - Stress (slow validation): lengthen interval by ``ramp_down_capacity_pct`` (default 15%/step).
        - Slack (fast validation): shorten interval by ``ramp_up_capacity_pct`` (default 5%/step).
        
        Otherwise maintain current block time (between thresholds).
        
        Args:
            validation_time_seconds: Time taken to validate the block
            current_block_time_seconds: Current block time (uses self.current_block_time_seconds if None)
        
        Returns:
            Tuple of (new_block_time_seconds, reason)
        """
        if current_block_time_seconds is None:
            current_block_time_seconds = self.current_block_time_seconds
        
        # Calculate utilization ratio
        utilization = validation_time_seconds / current_block_time_seconds
        
        # Determine adjustment (asymmetric: gentle shorten on slack, strong lengthen under stress)
        if utilization > self.increase_threshold:
            new_block_time = current_block_time_seconds * (
                1 + self.ramp_down_capacity_pct
            )
            reason = f"Validation took {utilization*100:.1f}% of block time (> {self.increase_threshold*100:.0f}%)"
            action = "increased (stress)"
        
        elif utilization < self.decrease_threshold:
            adaptive_floor = max(
                float(self.config.min_block_time_seconds),
                float(self.min_block_time_seconds),
            )
            new_block_time = max(
                adaptive_floor,
                current_block_time_seconds * (1 - self.ramp_up_capacity_pct),
            )
            
            if new_block_time <= adaptive_floor + 1e-6:
                reason = f"Block time at minimum ({adaptive_floor:.1f}s)"
                action = "maintained (at minimum)"
            else:
                reason = f"Validation took {utilization*100:.1f}% of block time (< {self.decrease_threshold*100:.0f}%)"
                action = "decreased"
        
        else:
            # Validation time is within acceptable range - maintain current
            new_block_time = current_block_time_seconds
            reason = f"Validation took {utilization*100:.1f}% of block time (optimal range)"
            action = "maintained"
        
        bound = self._bound_clamp_seconds(new_block_time)
        final = self._whole_seconds_ceil(bound)
        if abs(new_block_time - bound) > 1e-3:
            reason = f"{reason} (clamped {new_block_time:.1f}s → {bound:.1f}s to [min,max])"
            action = f"{action} [bounds]"
        new_block_time = final
        
        # Update current block time
        if new_block_time != current_block_time_seconds:
            kind = interval_change_kind(current_block_time_seconds, new_block_time)
            pretty_print(
                f"Slot {action}: {format_interval_sec(current_block_time_seconds)} → {format_interval_sec(new_block_time)}",
                kind=kind,
            )
            pretty_print(f"Reason: {reason}", kind="detail")
            pretty_print(
                f"Validation sample {format_interval_sec(validation_time_seconds)} "
                f"(util {utilization * 100:.1f}% of slot)",
                kind="detail",
            )
        
        self.current_block_time_seconds = new_block_time
        
        return new_block_time, reason
    
    def calculate_model_size_factor(
        self, 
        model_size_bytes: int,
        use_logarithmic: bool = True,
        max_factor: Optional[float] = None
    ) -> float:
        """
        Calculate a scaling factor based on model size for block time adjustment
        
        This ensures models from 1MB to 10TB+ get appropriate block times
        for training and verification. Uses logarithmic scaling for smooth
        transitions across the entire size spectrum.
        
        Args:
            model_size_bytes: Model size in bytes
            use_logarithmic: Use logarithmic scaling (smoother, recommended)
        
        Returns:
            Scaling factor (1.0 for tiny models, scales up to 1000+ for massive models)
        
        Examples:
            1 MB    -> 1.0x
            10 MB   -> 1.0x
            100 MB  -> 1.2x
            1 GB    -> 2.0x
            10 GB   -> 4.0x
            100 GB  -> 10.0x
            1 TB    -> 25.0x
            10 TB   -> 63.0x
        """
        if model_size_bytes <= 0:
            return 1.0
        
        # Define size units
        MB = 1024 * 1024
        GB = 1024 * MB
        TB = 1024 * GB
        
        if use_logarithmic:
            # Logarithmic scaling - smooth across all sizes
            # Base size: 10MB (baseline factor = 1.0)
            # Each 10x increase adds to the factor
            
            base_size = 10 * MB  # 10MB baseline
            size_ratio = model_size_bytes / base_size
            
            if size_ratio < 1.0:
                # Smaller than baseline: minimal factor
                return 1.0
            
            # Logarithmic formula: factor = 1 + log10(size_ratio) * multiplier
            # This gives smooth scaling:
            # 10 MB   (1x)   -> 1.0
            # 100 MB  (10x)  -> 1.2
            # 1 GB    (100x) -> 2.0
            # 10 GB   (1000x)-> 4.0
            # 100 GB  (10000x)-> 10.0
            # 1 TB    (100K) -> 25.0
            # 10 TB   (1M)   -> 63.0
            
            import math
            log_ratio = math.log10(size_ratio)
            
            # Scaling curve: scales more aggressively for larger models
            # Using power of 1.8 for good scaling across full range
            # Multiplier of 2.5 gives good spread from 1GB to 100TB
            factor = 1.0 + (log_ratio ** 1.8) * 2.5
            
            # Apply cap (use provided max_factor or default)
            cap = max_factor if max_factor is not None else 1000.0
            return min(factor, cap)
        
        else:
            # Step-based scaling - discrete jumps
            # Kept for backward compatibility
            if model_size_bytes < 10 * MB:  # < 10MB
                return 1.0
            elif model_size_bytes < 100 * MB:  # 10-100MB
                return 1.2
            elif model_size_bytes < 500 * MB:  # 100-500MB
                return 1.5
            elif model_size_bytes < 1 * GB:  # 500MB-1GB
                return 2.0
            elif model_size_bytes < 5 * GB:  # 1-5GB
                return 3.0
            elif model_size_bytes < 10 * GB:  # 5-10GB
                return 4.0
            elif model_size_bytes < 50 * GB:  # 10-50GB
                return 8.0
            elif model_size_bytes < 100 * GB:  # 50-100GB
                return 15.0
            elif model_size_bytes < 500 * GB:  # 100-500GB
                return 30.0
            elif model_size_bytes < 1 * TB:  # 500GB-1TB
                return 50.0
            elif model_size_bytes < 10 * TB:  # 1-10TB
                return 100.0
            else:  # >= 10TB
                # For massive models, scale logarithmically from here
                import math
                tb_size = model_size_bytes / TB
                return 100.0 * math.log10(tb_size + 1)
    
    def start_verification_timing(self, block_hash: str, phase: VerificationPhase) -> float:
        """Start timing a verification phase"""
        if block_hash not in self.verification_timings:
            self.verification_timings[block_hash] = []
        
        start_time = time.time()
        
        # Calculate allocated time for this phase
        if phase == VerificationPhase.TRAINING:
            allocated_time = self.current_block_time_seconds
        else:
            # Verification phases get a portion of the block time
            allocated_time = self.current_block_time_seconds * 0.3  # 30% of block time
        
        timing = VerificationTiming(
            phase=phase,
            allocated_time_seconds=allocated_time,
            actual_time_seconds=0.0,
            start_time=start_time,
            end_time=0.0,
            timeout_threshold=self.config.verification_timeout_threshold
        )
        
        self.verification_timings[block_hash].append(timing)
        return start_time
    
    def end_verification_timing(self, block_hash: str, phase: VerificationPhase) -> Optional[VerificationTiming]:
        """End timing a verification phase and return timing info"""
        if block_hash not in self.verification_timings:
            return None
        
        end_time = time.time()
        
        # Find the most recent timing for this phase
        for timing in reversed(self.verification_timings[block_hash]):
            if timing.phase == phase and timing.end_time == 0.0:
                timing.end_time = end_time
                timing.actual_time_seconds = end_time - timing.start_time
                timing.is_timeout = timing.is_exceeding_threshold()
                return timing
        
        return None
    
    def analyze_verification_performance(
        self,
        block_hash: str,
        model_size_bytes: Optional[int] = None
    ) -> Optional[BlockTimeAdjustmentProposal]:
        """
        Analyze verification performance and propose block time adjustment
        
        Args:
            block_hash: Block hash
            model_size_bytes: Optional model size in bytes for scaling adjustments
        
        Returns:
            Adjustment proposal or None
        """
        if not self.config.enable_dynamic_block_time:
            return None
        
        if block_hash not in self.verification_timings:
            return None
        
        timings = self.verification_timings[block_hash]
        if not timings:
            return None
        
        # Analyze timing performance
        timeout_count = sum(1 for timing in timings if timing.is_timeout)
        total_phases = len(timings)
        timeout_ratio = timeout_count / total_phases if total_phases > 0 else 0.0
        
        # Calculate model size factor for scaling adjustments
        size_factor = 1.0
        if model_size_bytes:
            size_factor = self.calculate_model_size_factor(
                model_size_bytes,
                use_logarithmic=self.use_logarithmic_scaling
            )
            # Apply max scaling factor cap
            size_factor = min(size_factor, self.max_scaling_factor)
        
        # Adjust thresholds based on model size
        # Larger models get more lenient timeout thresholds
        high_timeout_threshold = 0.5 / size_factor  # Lower threshold for larger models
        low_timeout_threshold = 0.1 / size_factor
        
        # Base step from model size; asymmetric multipliers (stress=slow lengthen, slack=fast shorten)
        base_step = self.config.block_time_adjustment_seconds * size_factor
        
        # Determine adjustment based on performance
        if timeout_ratio >= high_timeout_threshold:
            adjustment_type = BlockTimeAdjustment.INCREASE
            delta = base_step * self.stress_step_multiplier
            new_block_time = self.current_block_time_seconds + delta
            reason = f"High timeout ratio: {timeout_ratio:.1%} ({timeout_count}/{total_phases} phases), model size factor: {size_factor:.1f}x"
        elif timeout_ratio <= low_timeout_threshold and self.current_block_time_seconds > self.config.min_block_time_seconds:
            adjustment_type = BlockTimeAdjustment.DECREASE
            delta = base_step * self.slack_step_multiplier
            new_block_time = max(
                self.config.min_block_time_seconds,
                self.current_block_time_seconds - delta,
            )
            reason = f"Low timeout ratio: {timeout_ratio:.1%} ({timeout_count}/{total_phases} phases), model size factor: {size_factor:.1f}x"
        else:
            adjustment_type = BlockTimeAdjustment.MAINTAIN
            new_block_time = self.current_block_time_seconds
            reason = f"Optimal performance: {timeout_ratio:.1%} ({timeout_count}/{total_phases} phases)"
        
        raw_proposed = new_block_time
        bound_clamped = self._bound_clamp_seconds(raw_proposed)
        if abs(raw_proposed - bound_clamped) > 1e-3:
            reason = f"{reason} [clamped {raw_proposed:.1f}s → {bound_clamped:.1f}s to consensus bounds]"
            if raw_proposed > bound_clamped and adjustment_type == BlockTimeAdjustment.INCREASE:
                adjustment_type = BlockTimeAdjustment.MAINTAIN
            elif raw_proposed < bound_clamped and adjustment_type == BlockTimeAdjustment.DECREASE:
                adjustment_type = BlockTimeAdjustment.MAINTAIN
        new_block_time = self._whole_seconds_ceil(bound_clamped)

        # Create adjustment proposal
        proposal = BlockTimeAdjustmentProposal(
            adjustment_type=adjustment_type,
            current_block_time_seconds=self.current_block_time_seconds,
            proposed_block_time_seconds=new_block_time,
            reason=reason,
            verification_timings=timings.copy(),
            timestamp=time.time(),
            proposer="dynamic_block_time_manager"
        )
        
        if not proposal.is_valid(self.config):
            # Defensive: should not occur after clamping
            new_block_time = self._clamp_block_time_seconds(proposal.proposed_block_time_seconds)
            proposal = BlockTimeAdjustmentProposal(
                adjustment_type=BlockTimeAdjustment.MAINTAIN,
                current_block_time_seconds=self.current_block_time_seconds,
                proposed_block_time_seconds=new_block_time,
                reason=f"{reason} (forced maintain; bounds safety)",
                verification_timings=timings.copy(),
                timestamp=time.time(),
                proposer="dynamic_block_time_manager",
            )
        
        return proposal
    
    def propose_block_time_adjustment(self, block_hash: str, proposer: str) -> Optional[BlockTimeAdjustmentProposal]:
        """Propose a block time adjustment for a specific block"""
        proposal = self.analyze_verification_performance(block_hash)
        if not proposal:
            return None
        
        # Update proposer
        proposal.proposer = proposer
        
        # Store proposal
        if block_hash not in self.pending_adjustments:
            self.pending_adjustments[block_hash] = []
        self.pending_adjustments[block_hash].append(proposal)
        
        adj = proposal.adjustment_type.value
        pretty_print(
            f"Proposal block {block_hash[:12]}… │ {adj}: "
            f"{format_interval_sec(proposal.current_block_time_seconds)} → "
            f"{format_interval_sec(proposal.proposed_block_time_seconds)}",
            kind=adjustment_flag_kind(adj),
        )
        pretty_print(f"Reason: {proposal.reason}", kind="detail")
        
        return proposal
    
    def vote_on_adjustment(self, block_hash: str, voter: str, approve: bool) -> bool:
        """Vote on a block time adjustment proposal"""
        if block_hash not in self.pending_adjustments:
            return False
        
        proposals = self.pending_adjustments[block_hash]
        if not proposals:
            return False
        
        # For simplicity, we'll use the most recent proposal
        proposal = proposals[-1]
        
        # In a real implementation, this would track votes and require consensus
        # For now, we'll auto-approve if the proposal is reasonable
        if approve and proposal.is_valid(self.config):
            self.approved_adjustments[block_hash] = proposal
            pretty_print(f"Block time adjustment approved by {voter}", kind="ok")
            return True
        
        pretty_print(f"Block time adjustment rejected by {voter}", kind="bad")
        return False
    
    def apply_approved_adjustment(self, block_hash: str) -> bool:
        """Apply an approved block time adjustment"""
        if block_hash not in self.approved_adjustments:
            return False
        
        proposal = self.approved_adjustments[block_hash]
        
        # Apply the adjustment
        old_block_time = self.current_block_time_seconds
        self.current_block_time_seconds = self._clamp_block_time_seconds(
            proposal.proposed_block_time_seconds
        )
        
        # Record the adjustment
        self.recent_adjustments.append(proposal)
        
        # Clean up
        del self.approved_adjustments[block_hash]
        if block_hash in self.pending_adjustments:
            del self.pending_adjustments[block_hash]
        
        kind = interval_change_kind(old_block_time, self.current_block_time_seconds)
        pretty_print(
            f"Applied approved slot: {format_interval_sec(old_block_time)} → "
            f"{format_interval_sec(self.current_block_time_seconds)}",
            kind=kind,
        )
        pretty_print(f"Reason: {proposal.reason}", kind="detail")
        
        return True
    
    def get_current_block_time_seconds(self) -> float:
        """Get the current block time in seconds"""
        return self.current_block_time_seconds
    
    def get_block_time_for_phase(self, phase: VerificationPhase) -> float:
        """Get the appropriate block time for a verification phase"""
        if phase == VerificationPhase.TRAINING:
            return self.current_block_time_seconds
        else:
            # Verification phases get a portion of the block time (whole seconds)
            return float(math.ceil(self.current_block_time_seconds * 0.3 - 1e-9))
    
    def get_verification_timings(self, block_hash: str) -> List[VerificationTiming]:
        """Get verification timings for a specific block"""
        return self.verification_timings.get(block_hash, [])
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for recent blocks"""
        total_blocks = len(self.verification_timings)
        if total_blocks == 0:
            return {"error": "No verification data available"}
        
        # Calculate statistics
        total_timeouts = 0
        total_phases = 0
        total_verification_time = 0.0
        
        for block_hash, timings in self.verification_timings.items():
            for timing in timings:
                total_phases += 1
                if timing.is_timeout:
                    total_timeouts += 1
                total_verification_time += timing.actual_time_seconds
        
        timeout_ratio = total_timeouts / total_phases if total_phases > 0 else 0.0
        avg_verification_time = total_verification_time / total_phases if total_phases > 0 else 0.0
        
        return {
            "total_blocks_analyzed": total_blocks,
            "total_verification_phases": total_phases,
            "total_timeouts": total_timeouts,
            "timeout_ratio": timeout_ratio,
            "average_verification_time_seconds": avg_verification_time,
            "current_block_time_seconds": self.current_block_time_seconds,
            "recent_adjustments_count": len(self.recent_adjustments),
            "pending_adjustments_count": sum(len(proposals) for proposals in self.pending_adjustments.values()),
            "approved_adjustments_count": len(self.approved_adjustments)
        }
    
    def cleanup_old_data(self, current_block_index: int, retention_blocks: int = 100):
        """Clean up old verification data to prevent memory leaks"""
        cutoff_block = current_block_index - retention_blocks
        
        # Clean up old verification timings
        to_remove = []
        for block_hash in self.verification_timings.keys():
            # This is simplified - in practice would need block index mapping
            if len(self.verification_timings[block_hash]) > 0:
                # Keep recent data
                pass
        
        # Clean up old adjustments
        self.recent_adjustments = [
            adj for adj in self.recent_adjustments 
            if current_block_index - adj.timestamp < retention_blocks
        ]
