"""
Slashing Mechanism for DeSSIN Blockchain
=======================================

Implements economic penalties for dishonest behavior in the PoGO consensus.
Provides stake slashing, temporary bans, and reputation tracking.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

from .consensus import PogoBlock
from .transactions import AttestationTransaction
from ..runtime.config import ConsensusConfig
from ..economics.economic_system import EconomicSystem, EconomicEventType


class SlashingReason(Enum):
    """Reasons for slashing"""
    INVALID_BLOCK = "invalid_block"  # Miner produced invalid block
    DOUBLE_SIGNING = "double_signing"  # Miner signed conflicting blocks
    NEGATIVE_ATTESTATIONS = "negative_attestations"  # Too many negative attestations
    VERIFICATION_FAILURE = "verification_failure"  # Failed verification checks
    MALICIOUS_BEHAVIOR = "malicious_behavior"  # Other malicious activities
    UNAVAILABILITY = "unavailability"  # Miner unavailable for extended periods


class SlashingSeverity(Enum):
    """Severity levels for slashing"""
    MINOR = "minor"  # Small penalty, temporary restriction
    MODERATE = "moderate"  # Medium penalty, longer restriction
    MAJOR = "major"  # Large penalty, extended ban
    CRITICAL = "critical"  # Maximum penalty, permanent ban


@dataclass
class SlashingEvent:
    """Represents a slashing event"""
    event_id: str
    timestamp: float
    block_index: int
    block_hash: str
    slashed_address: str
    reason: SlashingReason
    severity: SlashingSeverity
    amount_slashed: float
    stake_before: float
    stake_after: float
    ban_duration_seconds: int
    evidence: Dict[str, Any]
    reporter: str
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class SlashingPenalty:
    """Penalty configuration for different slashing reasons"""
    reason: SlashingReason
    severity: SlashingSeverity
    stake_penalty_percentage: float  # Percentage of stake to slash
    ban_duration_seconds: int  # Duration of ban in seconds
    reputation_penalty: float  # Reputation score penalty
    economic_penalty: float  # Additional economic penalty in DESSIN


class SlashingMechanism:
    """Manages slashing penalties and enforcement"""
    
    def __init__(self, config: ConsensusConfig, economic_system: EconomicSystem):
        self.config = config
        self.economic_system = economic_system
        
        # Slashing configuration
        self.slashing_threshold = config.slashing_threshold  # Default 33%
        self.attestation_window_blocks = config.attestation_window_blocks
        
        # Slashing penalties configuration
        self.penalties = self._initialize_penalties()
        
        # State tracking
        self.slashing_events: List[SlashingEvent] = []
        self.banned_addresses: Dict[str, float] = {}  # address -> ban_end_timestamp
        self.reputation_scores: Dict[str, float] = {}  # address -> reputation_score
        self.negative_attestation_counts: Dict[str, int] = {}  # address -> count
        self.slashing_proposals: Dict[str, List[SlashingEvent]] = {}  # block_hash -> proposals
        
        # Initialize default reputation scores
        self._initialize_reputation_scores()
        
        print(f"✓ Slashing mechanism initialized")
        print(f"  Slashing threshold: {self.slashing_threshold:.1%}")
        print(f"  Attestation window: {self.attestation_window_blocks} blocks")
        print(f"  Penalty types: {len(self.penalties)}")
    
    def _initialize_penalties(self) -> Dict[SlashingReason, SlashingPenalty]:
        """Initialize slashing penalties for different violations"""
        penalties = {}
        
        # Invalid block penalties
        penalties[SlashingReason.INVALID_BLOCK] = SlashingPenalty(
            reason=SlashingReason.INVALID_BLOCK,
            severity=SlashingSeverity.MAJOR,
            stake_penalty_percentage=0.20,  # 20% of stake
            ban_duration_seconds=3600,  # 1 hour ban
            reputation_penalty=50.0,
            economic_penalty=100.0
        )
        
        # Double signing penalties
        penalties[SlashingReason.DOUBLE_SIGNING] = SlashingPenalty(
            reason=SlashingReason.DOUBLE_SIGNING,
            severity=SlashingSeverity.CRITICAL,
            stake_penalty_percentage=0.50,  # 50% of stake
            ban_duration_seconds=86400,  # 24 hour ban
            reputation_penalty=100.0,
            economic_penalty=500.0
        )
        
        # Negative attestations penalties
        penalties[SlashingReason.NEGATIVE_ATTESTATIONS] = SlashingPenalty(
            reason=SlashingReason.NEGATIVE_ATTESTATIONS,
            severity=SlashingSeverity.MODERATE,
            stake_penalty_percentage=0.10,  # 10% of stake
            ban_duration_seconds=1800,  # 30 minute ban
            reputation_penalty=25.0,
            economic_penalty=50.0
        )
        
        # Verification failure penalties
        penalties[SlashingReason.VERIFICATION_FAILURE] = SlashingPenalty(
            reason=SlashingReason.VERIFICATION_FAILURE,
            severity=SlashingSeverity.MINOR,
            stake_penalty_percentage=0.05,  # 5% of stake
            ban_duration_seconds=900,  # 15 minute ban
            reputation_penalty=10.0,
            economic_penalty=25.0
        )
        
        # Malicious behavior penalties
        penalties[SlashingReason.MALICIOUS_BEHAVIOR] = SlashingPenalty(
            reason=SlashingReason.MALICIOUS_BEHAVIOR,
            severity=SlashingSeverity.MAJOR,
            stake_penalty_percentage=0.30,  # 30% of stake
            ban_duration_seconds=7200,  # 2 hour ban
            reputation_penalty=75.0,
            economic_penalty=200.0
        )
        
        # Unavailability penalties
        penalties[SlashingReason.UNAVAILABILITY] = SlashingPenalty(
            reason=SlashingReason.UNAVAILABILITY,
            severity=SlashingSeverity.MINOR,
            stake_penalty_percentage=0.02,  # 2% of stake
            ban_duration_seconds=600,  # 10 minute ban
            reputation_penalty=5.0,
            economic_penalty=10.0
        )
        
        return penalties
    
    def _initialize_reputation_scores(self):
        """Initialize reputation scores for all addresses"""
        # In a real system, this would be loaded from persistent storage
        # For now, we'll initialize with default scores
        pass
    
    def get_reputation_score(self, address: str) -> float:
        """Get reputation score for an address (0-100, higher is better)"""
        return self.reputation_scores.get(address, 100.0)  # Default to perfect score
    
    def update_reputation_score(self, address: str, penalty: float):
        """Update reputation score after slashing"""
        current_score = self.get_reputation_score(address)
        new_score = max(0.0, current_score - penalty)
        self.reputation_scores[address] = new_score
        
        print(f"📉 Reputation updated for {address}: {current_score:.1f} → {new_score:.1f}")
    
    def is_address_banned(self, address: str) -> bool:
        """Check if an address is currently banned"""
        if address not in self.banned_addresses:
            return False
        
        ban_end_time = self.banned_addresses[address]
        if time.time() > ban_end_time:
            # Ban has expired, remove it
            del self.banned_addresses[address]
            return False
        
        return True
    
    def get_ban_remaining_time(self, address: str) -> int:
        """Get remaining ban time in seconds"""
        if not self.is_address_banned(address):
            return 0
        
        ban_end_time = self.banned_addresses[address]
        return max(0, int(ban_end_time - time.time()))
    
    def propose_slashing(
        self, 
        block_hash: str, 
        slashed_address: str, 
        reason: SlashingReason, 
        evidence: Dict[str, Any],
        reporter: str,
        description: str = ""
    ) -> Optional[SlashingEvent]:
        """Propose a slashing event"""
        try:
            # Check if address is already banned
            if self.is_address_banned(slashed_address):
                print(f"⚠️  Address {slashed_address} is already banned")
                return None
            
            # Get penalty configuration
            penalty = self.penalties.get(reason)
            if not penalty:
                print(f"⚠️  No penalty configuration for reason: {reason}")
                return None
            
            # Calculate slashing amount
            current_stake = self.economic_system.get_staked_amount(slashed_address)
            if current_stake <= 0:
                print(f"⚠️  Address {slashed_address} has no stake to slash")
                return None
            
            amount_slashed = current_stake * penalty.stake_penalty_percentage
            new_stake = current_stake - amount_slashed
            
            # Create slashing event
            event = SlashingEvent(
                event_id=f"slash_{int(time.time())}_{slashed_address[:8]}",
                timestamp=time.time(),
                block_index=0,  # Will be updated when applied
                block_hash=block_hash,
                slashed_address=slashed_address,
                reason=reason,
                severity=penalty.severity,
                amount_slashed=amount_slashed,
                stake_before=current_stake,
                stake_after=new_stake,
                ban_duration_seconds=penalty.ban_duration_seconds,
                evidence=evidence,
                reporter=reporter,
                description=description or f"Slashing for {reason.value}"
            )
            
            # Store proposal
            if block_hash not in self.slashing_proposals:
                self.slashing_proposals[block_hash] = []
            self.slashing_proposals[block_hash].append(event)
            
            print(f"🔨 Slashing proposed for {slashed_address}")
            print(f"  Reason: {reason.value}")
            print(f"  Severity: {penalty.severity.value}")
            print(f"  Amount: {amount_slashed:.4f} DESSIN ({penalty.stake_penalty_percentage:.1%} of stake)")
            print(f"  Ban duration: {penalty.ban_duration_seconds} seconds")
            print(f"  Reporter: {reporter}")
            
            return event
            
        except Exception as e:
            print(f"Error proposing slashing: {e}")
            return None
    
    def apply_slashing(self, event: SlashingEvent, block_index: int) -> bool:
        """Apply a slashing event"""
        try:
            # Update block index
            event.block_index = block_index
            
            # Apply stake slashing
            success = self.economic_system.process_slashing_penalty(
                event.slashed_address, 
                event.amount_slashed, 
                event.block_hash, 
                block_index
            )
            
            if not success:
                print(f"❌ Failed to apply economic penalty for {event.slashed_address}")
                return False
            
            # Apply ban
            ban_end_time = time.time() + event.ban_duration_seconds
            self.banned_addresses[event.slashed_address] = ban_end_time
            
            # Update reputation score
            penalty = self.penalties.get(event.reason)
            if penalty:
                self.update_reputation_score(event.slashed_address, penalty.reputation_penalty)
            
            # Record slashing event
            self.slashing_events.append(event)
            
            print(f"✅ Slashing applied for {event.slashed_address}")
            print(f"  Event ID: {event.event_id}")
            print(f"  Amount slashed: {event.amount_slashed:.4f} DESSIN")
            print(f"  Ban until: {time.ctime(ban_end_time)}")
            print(f"  New reputation: {self.get_reputation_score(event.slashed_address):.1f}")
            
            return True
            
        except Exception as e:
            print(f"Error applying slashing: {e}")
            return False
    
    def check_negative_attestations(self, block_hash: str, attestations: List[AttestationTransaction]) -> List[SlashingEvent]:
        """Check for excessive negative attestations and propose slashing"""
        slashing_events = []
        
        # Count negative attestations by address
        negative_counts = {}
        total_attestations = {}
        
        for attestation in attestations:
            address = attestation.sender
            total_attestations[address] = total_attestations.get(address, 0) + 1
            
            if attestation.is_negative():
                negative_counts[address] = negative_counts.get(address, 0) + 1
        
        # Check each address for excessive negative attestations
        for address, negative_count in negative_counts.items():
            total_count = total_attestations.get(address, 0)
            if total_count == 0:
                continue
            
            negative_ratio = negative_count / total_count
            
            # If negative ratio exceeds threshold, propose slashing
            if negative_ratio >= self.slashing_threshold:
                evidence = {
                    "negative_attestations": negative_count,
                    "total_attestations": total_count,
                    "negative_ratio": negative_ratio,
                    "threshold": self.slashing_threshold
                }
                
                slashing_event = self.propose_slashing(
                    block_hash=block_hash,
                    slashed_address=address,
                    reason=SlashingReason.NEGATIVE_ATTESTATIONS,
                    evidence=evidence,
                    reporter="consensus_mechanism",
                    description=f"Excessive negative attestations: {negative_ratio:.1%} >= {self.slashing_threshold:.1%}"
                )
                
                if slashing_event:
                    slashing_events.append(slashing_event)
        
        return slashing_events
    
    def check_block_validity(self, block: PogoBlock) -> List[SlashingEvent]:
        """Check block validity and propose slashing for invalid blocks"""
        slashing_events = []
        
        # Basic block validation checks
        if not self._validate_block_basic(block):
            evidence = {
                "validation_failure": "basic_validation_failed",
                "block_index": block.index,
                "block_hash": block.hash
            }
            
            slashing_event = self.propose_slashing(
                block_hash=block.hash,
                slashed_address=block.miner,
                reason=SlashingReason.INVALID_BLOCK,
                evidence=evidence,
                reporter="consensus_mechanism",
                description="Block failed basic validation checks"
            )
            
            if slashing_event:
                slashing_events.append(slashing_event)
        
        # Check for double signing (simplified)
        if self._check_double_signing(block):
            evidence = {
                "double_signing_detected": True,
                "block_index": block.index,
                "block_hash": block.hash
            }
            
            slashing_event = self.propose_slashing(
                block_hash=block.hash,
                slashed_address=block.miner,
                reason=SlashingReason.DOUBLE_SIGNING,
                evidence=evidence,
                reporter="consensus_mechanism",
                description="Double signing detected"
            )
            
            if slashing_event:
                slashing_events.append(slashing_event)
        
        return slashing_events
    
    def _validate_block_basic(self, block: PogoBlock) -> bool:
        """Basic block validation checks"""
        try:
            # Check required fields
            if not block.hash or not block.miner or block.index < 0:
                return False
            
            # Check loss improvement is reasonable
            if hasattr(block, 'loss_before') and hasattr(block, 'loss_after'):
                if block.loss_after > block.loss_before * 1.1:  # Allow 10% tolerance
                    return False
            
            # Check training steps are reasonable
            if hasattr(block, 'training_steps'):
                if block.training_steps <= 0 or block.training_steps > 1000:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _check_double_signing(self, block: PogoBlock) -> bool:
        """Reserved for block-local checks; gossip-path equivocation uses consensus hooks."""
        return False
    
    def get_slashing_summary(self) -> Dict[str, Any]:
        """Get comprehensive slashing summary"""
        total_events = len(self.slashing_events)
        total_slashed = sum(event.amount_slashed for event in self.slashing_events)
        currently_banned = len(self.banned_addresses)
        
        # Count by reason
        reason_counts = {}
        severity_counts = {}
        
        for event in self.slashing_events:
            reason = event.reason.value
            severity = event.severity.value
            
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_slashing_events": total_events,
            "total_amount_slashed": total_slashed,
            "currently_banned_addresses": currently_banned,
            "banned_addresses": list(self.banned_addresses.keys()),
            "reason_breakdown": reason_counts,
            "severity_breakdown": severity_counts,
            "reputation_scores": self.reputation_scores.copy(),
            "pending_proposals": sum(len(proposals) for proposals in self.slashing_proposals.values())
        }
    
    def get_address_slashing_history(self, address: str) -> List[SlashingEvent]:
        """Get slashing history for a specific address"""
        return [event for event in self.slashing_events if event.slashed_address == address]
    
    def cleanup_expired_bans(self):
        """Clean up expired bans"""
        current_time = time.time()
        expired_addresses = []
        
        for address, ban_end_time in self.banned_addresses.items():
            if current_time > ban_end_time:
                expired_addresses.append(address)
        
        for address in expired_addresses:
            del self.banned_addresses[address]
            print(f"🔓 Ban expired for {address}")
    
    def export_slashing_data(self) -> Dict[str, Any]:
        """Export all slashing data for analysis"""
        return {
            "slashing_events": [event.to_dict() for event in self.slashing_events],
            "banned_addresses": self.banned_addresses.copy(),
            "reputation_scores": self.reputation_scores.copy(),
            "negative_attestation_counts": self.negative_attestation_counts.copy(),
            "summary": self.get_slashing_summary()
        }
