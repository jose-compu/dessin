"""
Economic System for DeSSIN Blockchain
====================================

Handles block rewards, training payments, and economic incentives for the PoGO consensus.
"""

import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from ..consensus.transactions import BaseTransaction, TransactionFactory
from ..consensus.consensus import PogoBlock
from ..runtime.pretty_console import pretty_print


class EconomicEventType(Enum):
    """Types of economic events"""
    BLOCK_REWARD = "block_reward"
    TRAINING_PAYMENT = "training_payment"
    TRAINING_ITERATION_PAYMENT = "training_iteration_payment"
    ATTESTATION_REWARD = "attestation_reward"
    SLASHING_PENALTY = "slashing_penalty"
    QUERY_PAYMENT = "query_payment"
    STORAGE_FEE = "storage_fee"
    STORAGE_RENTAL = "storage_rental"
    STORAGE_EXTENSION = "storage_extension"
    TRANSACTION_FEE = "transaction_fee"
    TRAINING_ESCROW_DEPOSIT = "training_escrow_deposit"
    BOOTSTRAP_MATERIALIZATION = "bootstrap_materialization"
    UNSTAKE_REQUEST = "unstake_request"


@dataclass
class EconomicEvent:
    """Represents an economic event in the system"""
    event_type: EconomicEventType
    timestamp: float
    block_index: int
    from_address: str
    to_address: str
    amount: float
    description: str
    transaction_hash: Optional[str] = None
    block_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class AccountBalance:
    """Account balance information"""
    address: str
    balance: float
    staked_amount: float
    total_earned: float
    total_spent: float
    last_updated: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class EconomicSystem:
    """Manages economic transactions and balances"""
    
    def __init__(self, initial_supply: float = 1000000.0, mining_reward: float = 10.0):
        self.initial_supply = initial_supply
        self.mining_reward = mining_reward
        
        # Economic parameters
        self.training_payment_per_step = 0.1  # DESSIN per training step
        self.training_iteration_payment = 0.001  # DESSIN per training iteration
        self.attestation_reward = 0.5  # DESSIN per attestation
        self.query_fee_per_token = 0.001  # DESSIN per token
        self.storage_fee_per_gb_per_block = 0.001  # DESSIN per GB per block (base rate)
        
        # Dynamic pricing for storage (can be adjusted by nodes)
        self.storage_price_multiplier = 1.0  # Default multiplier
        self.min_storage_price_multiplier = 0.1  # Minimum 10% of base rate
        self.max_storage_price_multiplier = 10.0  # Maximum 10x base rate
        
        # State tracking
        self.balances: Dict[str, float] = {}
        self.staked_amounts: Dict[str, float] = {}
        self.economic_events: List[EconomicEvent] = []
        self.total_supply = initial_supply
        
        # Model storage tracking
        self.model_storage_payments: Dict[str, float] = {}  # model_id -> total paid
        self.model_training_payments: Dict[str, float] = {}  # model_id -> total paid
        # Per-model escrow reserved by owners for miner training payouts (see deposit_model_training_escrow)
        self.model_training_escrow: Dict[str, float] = {}
        self.bootstrap_virtual_locked: Dict[str, float] = {}
        self.total_tokens_emitted: float = 0.0
        # Queued exits from stake: (amount, unlock_at_block_index inclusive boundary)
        self.unbonding_queue: Dict[str, List[Tuple[float, int]]] = {}
        
        # Initialize genesis account
        self._initialize_genesis()
    
    def genesis_address(self) -> str:
        return "genesis_0000000000000000000000000000000000000000"

    def _initialize_genesis(self):
        """Initialize genesis account with initial supply"""
        genesis_address = self.genesis_address()
        self.balances[genesis_address] = self.initial_supply
        self.staked_amounts[genesis_address] = 0.0
        
        # Record genesis event
        genesis_event = EconomicEvent(
            event_type=EconomicEventType.BLOCK_REWARD,
            timestamp=time.time(),
            block_index=0,
            from_address="system",
            to_address=genesis_address,
            amount=self.initial_supply,
            description="Genesis block - initial supply",
            block_hash="genesis"
        )
        self.economic_events.append(genesis_event)
    
    def get_balance(self, address: str) -> float:
        """Get account balance"""
        return self.balances.get(address, 0.0)
    
    def get_staked_amount(self, address: str) -> float:
        """Get staked amount for an address"""
        return self.staked_amounts.get(address, 0.0)
    
    def get_total_balance(self, address: str) -> float:
        """Get total balance (liquid + staked)"""
        return self.get_balance(address) + self.get_staked_amount(address)

    def total_protocol_rewards_received(self, address: str) -> float:
        """Sum of protocol rewards credited to ``address`` (block + attestation + bootstrap materialization mints)."""
        types = (
            EconomicEventType.BLOCK_REWARD,
            EconomicEventType.ATTESTATION_REWARD,
            EconomicEventType.BOOTSTRAP_MATERIALIZATION,
        )
        return sum(
            e.amount
            for e in self.economic_events
            if e.to_address == address and e.event_type in types
        )
    
    def _transfer(self, from_address: str, to_address: str, amount: float, 
                  description: str, event_type: EconomicEventType, 
                  block_index: int = 0, block_hash: str = None) -> bool:
        """Internal transfer method"""
        if from_address != "system":
            self.finalize_matured_unstakes(from_address, block_index)
        if from_address != "system" and self.get_balance(from_address) < amount:
            print(f"Insufficient balance for {from_address}: {self.get_balance(from_address)} < {amount}")
            return False
        
        # Update balances
        if from_address != "system":
            self.balances[from_address] -= amount
        self.balances[to_address] = self.balances.get(to_address, 0.0) + amount
        
        # Record economic event
        event = EconomicEvent(
            event_type=event_type,
            timestamp=time.time(),
            block_index=block_index,
            from_address=from_address,
            to_address=to_address,
            amount=amount,
            description=description,
            block_hash=block_hash
        )
        self.economic_events.append(event)
        
        return True
    
    def distribute_block_reward(
        self,
        miner_address: str,
        block: PogoBlock,
        *,
        annual_emission_rate: Optional[float] = None,
        auto_stake_reward: bool = False,
        blocks_per_year: float = 4380.0,
    ) -> bool:
        """Distribute block reward; optionally emission-based APY and auto-stake (bootstrap)."""
        training_bonus = block.training_steps * 0.01  # 0.01 DESSIN per step

        if annual_emission_rate is not None:
            base_reward = self.initial_supply * float(annual_emission_rate) / max(1.0, blocks_per_year)
        else:
            base_reward = self.mining_reward

        total_reward = base_reward + training_bonus

        if auto_stake_reward:
            success = self._mint_from_genesis_to_stake(
                miner_address,
                total_reward,
                block,
                description=(
                    f"Auto-staked block reward block {block.index} "
                    f"(training steps: {block.training_steps})"
                ),
            )
        else:
            success = self._mint_from_genesis_to_liquid(
                miner_address,
                total_reward,
                block,
                description=f"Block reward for block {block.index} (training steps: {block.training_steps})",
            )

        if success:
            pretty_print(
                f"💰 Block reward distributed: {total_reward:.4f} DESSIN to {miner_address}",
                kind="ok",
            )
            pretty_print(
                f"Base reward: {base_reward:.4f} DESSIN; training bonus: {training_bonus:.4f} DESSIN",
                kind="detail",
            )
            if auto_stake_reward:
                pretty_print("(auto-staked; unstake to transfer)", kind="detail")

        return success

    def _mint_from_genesis(
        self,
        miner_address: str,
        amount: float,
        block: PogoBlock,
        *,
        to_stake: bool,
        description: str,
    ) -> bool:
        if amount <= 0:
            return True
        g = self.genesis_address()
        avail = self.balances.get(g, 0.0)
        pay = min(amount, avail)
        if pay <= 0:
            return False
        self.balances[g] -= pay
        if to_stake:
            self.staked_amounts[miner_address] = self.staked_amounts.get(miner_address, 0.0) + pay
        else:
            self.balances[miner_address] = self.balances.get(miner_address, 0.0) + pay
        self.total_tokens_emitted += pay
        self.economic_events.append(
            EconomicEvent(
                event_type=EconomicEventType.BLOCK_REWARD,
                timestamp=time.time(),
                block_index=block.index,
                from_address=g,
                to_address=miner_address,
                amount=pay,
                description=description,
                block_hash=block.hash,
            )
        )
        return True

    def _mint_from_genesis_to_stake(
        self, miner_address: str, amount: float, block: PogoBlock, description: str
    ) -> bool:
        return self._mint_from_genesis(
            miner_address, amount, block, to_stake=True, description=description
        )

    def _mint_from_genesis_to_liquid(
        self, miner_address: str, amount: float, block: PogoBlock, description: str
    ) -> bool:
        return self._mint_from_genesis(
            miner_address, amount, block, to_stake=False, description=description
        )

    def grant_bootstrap_virtual_stake(self, address: str, amount: float) -> None:
        """Non-liquid stake used for ranking only until bootstrap materialization or slash."""
        self.bootstrap_virtual_locked[address] = self.bootstrap_virtual_locked.get(address, 0.0) + amount

    def effective_bootstrap_stake(self, address: str) -> float:
        """Ranking stake during bootstrap: locked virtual + auto-staked rewards."""
        return self.bootstrap_virtual_locked.get(address, 0.0) + self.staked_amounts.get(address, 0.0)

    def _unbonding_total(self, address: str) -> float:
        return sum(a for a, _ in self.unbonding_queue.get(address, ()))

    def get_queued_unbonding_principal(self, address: str) -> float:
        """Principal still in the unbonding queue (not yet liquid)."""
        return self._unbonding_total(address)

    def total_holdings_for_eligibility(self, address: str) -> float:
        """Liquid + reward stake + virtual + principal still in unbonding queue."""
        return (
            self.balances.get(address, 0.0)
            + self.staked_amounts.get(address, 0.0)
            + self.bootstrap_virtual_locked.get(address, 0.0)
            + self._unbonding_total(address)
        )

    def preview_transferable_liquid(self, address: str, at_block: int) -> float:
        """Liquid balance plus matured-but-not-yet-flushed unbonding (no state change; for display)."""
        b = float(self.balances.get(address, 0.0))
        matured = sum(
            float(amt)
            for amt, unlock in self.unbonding_queue.get(address, ())
            if int(at_block) >= int(unlock)
        )
        return b + matured

    def finalize_matured_unstakes(self, address: str, current_block_index: int) -> float:
        """Credit liquid balance for matured unbonding entries (call before debiting sender)."""
        q = self.unbonding_queue.get(address)
        if not q:
            return 0.0
        credited = 0.0
        pending: List[Tuple[float, int]] = []
        tip = int(current_block_index)
        for amt, unlock in q:
            if tip >= int(unlock):
                credited += float(amt)
            else:
                pending.append((float(amt), int(unlock)))
        if credited > 0:
            self.balances[address] = self.balances.get(address, 0.0) + credited
        if pending:
            self.unbonding_queue[address] = pending
        else:
            self.unbonding_queue.pop(address, None)
        return credited

    def request_unstake(self, address: str, amount: float, current_block_index: int, cooldown_blocks: int) -> bool:
        """Move stake into unbonding; becomes liquid only after cooldown (credited on spend paths)."""
        if amount <= 0:
            return False
        if cooldown_blocks < 0:
            return False
        if cooldown_blocks == 0:
            return self._unstake_immediate_to_liquid(address, amount)
        if self.get_staked_amount(address) + 1e-12 < amount:
            return False
        self.staked_amounts[address] = self.get_staked_amount(address) - amount
        unlock_at = int(current_block_index) + int(cooldown_blocks)
        self.unbonding_queue.setdefault(address, []).append((float(amount), unlock_at))
        self.economic_events.append(
            EconomicEvent(
                event_type=EconomicEventType.UNSTAKE_REQUEST,
                timestamp=time.time(),
                block_index=int(current_block_index),
                from_address="system",
                to_address=address,
                amount=float(amount),
                description=f"Unstake request → unbonding, unlock at block {unlock_at}",
                block_hash=None,
            )
        )
        print(f"⏳ Unstake queued: {amount:.4f} DESSIN for {address[:16]}… unlock block {unlock_at}")
        return True

    def _unstake_immediate_to_liquid(self, address: str, amount: float) -> bool:
        """Legacy/dev: move stake to liquid with no cooldown."""
        if self.get_staked_amount(address) < amount:
            return False
        self.staked_amounts[address] = self.get_staked_amount(address) - amount
        self.balances[address] = self.balances.get(address, 0.0) + amount
        print(f"🔓 Unstaked (immediate): {amount:.4f} DESSIN by {address}")
        return True

    def slash_bootstrap_participant(self, address: str, block_index: int = 0) -> None:
        """Remove virtual stake and reward stake for inactive bootstrap registrants."""
        self.bootstrap_virtual_locked.pop(address, None)
        lost = self.staked_amounts.pop(address, 0.0)
        if lost > 0:
            print(f"⚡ Bootstrap slash {address[:16]}… staked rewards removed ({lost:.4f})")

    def materialize_bootstrap_survivor(
        self,
        address: str,
        materialization_grant: float,
        block_index: int = 0,
    ) -> None:
        """Convert virtual stake + bootstrap auto-staked rewards into real staked DESSIN; mint grant + virtual claim from genesis into stake (not liquid)."""
        virt = float(self.bootstrap_virtual_locked.pop(address, 0.0))
        rew = float(self.staked_amounts.pop(address, 0.0))
        grant = float(materialization_grant)
        need_from_genesis = virt + grant
        minted = 0.0
        if need_from_genesis > 0:
            g = self.genesis_address()
            avail = self.balances.get(g, 0.0)
            minted = min(need_from_genesis, avail)
            if minted > 0:
                self.balances[g] -= minted
                self.total_tokens_emitted += minted
                self.economic_events.append(
                    EconomicEvent(
                        event_type=EconomicEventType.BOOTSTRAP_MATERIALIZATION,
                        timestamp=time.time(),
                        block_index=block_index,
                        from_address=g,
                        to_address=address,
                        amount=minted,
                        description=(
                            f"Bootstrap materialization → stake: virtual {virt:.4f} + grant {grant:.4f} "
                            f"(minted {minted:.4f}; carried reward stake {rew:.4f})"
                        ),
                        block_hash=None,
                    )
                )
            if minted < need_from_genesis - 1e-9:
                print(
                    f"⚠️  Bootstrap materialization under-funded from genesis: "
                    f"need {need_from_genesis:.4f}, minted {minted:.4f}"
                )
        staked_total = rew + minted
        if staked_total <= 0:
            return
        self.staked_amounts[address] = self.staked_amounts.get(address, 0.0) + staked_total
        print(
            f"✅ Bootstrap materialized {address[:16]}… +{staked_total:.4f} DESSIN staked "
            f"(reward stake carried {rew:.4f} + minted {minted:.4f} from virtual+grant)"
        )
    
    def deposit_model_training_escrow(
        self,
        payer_address: str,
        model_id: str,
        amount: float,
        block_index: int = 0,
        description: str = "",
    ) -> bool:
        """Move DESSIN from payer balance into per-model training escrow."""
        if amount < 0:
            return False
        if amount == 0:
            return True
        self.finalize_matured_unstakes(payer_address, block_index)
        if self.get_balance(payer_address) < amount:
            return False
        self.balances[payer_address] -= amount
        self.model_training_escrow[model_id] = self.model_training_escrow.get(model_id, 0.0) + amount
        desc = description or f"Training escrow deposit for model {model_id}"
        self.economic_events.append(
            EconomicEvent(
                event_type=EconomicEventType.TRAINING_ESCROW_DEPOSIT,
                timestamp=time.time(),
                block_index=block_index,
                from_address=payer_address,
                to_address=f"escrow:{model_id}",
                amount=amount,
                description=desc,
                block_hash=None,
            )
        )
        print(f"📥 Training escrow +{amount:.4f} DESSIN for model {model_id} (from {payer_address[:16]}…)")
        return True

    def collect_transaction_fee(
        self, payer_address: str, fee: float, block_index: int = 0, note: str = ""
    ) -> bool:
        """Collect a transaction fee to the system account."""
        if fee < 0:
            return False
        if fee == 0:
            return True
        return self._transfer(
            from_address=payer_address,
            to_address="system",
            amount=fee,
            description=note or "Transaction fee",
            event_type=EconomicEventType.TRANSACTION_FEE,
            block_index=block_index,
            block_hash=None,
        )

    def process_training_payment(
        self,
        model_owner: str,
        miner_address: str,
        block: PogoBlock,
        *,
        allow_debit_bootstrap_stake: bool = False,
    ) -> bool:
        """Pay miner for training: use per-model escrow first, then model owner liquid balance.

        When ``allow_debit_bootstrap_stake`` is True (bootstrap blocks only), if liquid is short
        but the owner has staked DESSIN (e.g. auto-staked block rewards), unstake immediately
        up to the shortfall so ``_transfer`` can complete — mirrors the enhanced affordability
        check that counts stake during bootstrap.
        """
        training_cost = block.training_steps * self.training_payment_per_step
        model_complexity_factor = 1.0 + (block.model_size_full / (1024 * 1024 * 1024)) * 0.1
        total_cost = training_cost * model_complexity_factor
        model_id = block.model_id

        escrow_avail = self.model_training_escrow.get(model_id, 0.0)
        from_escrow = min(escrow_avail, total_cost)
        remainder = total_cost - from_escrow

        if remainder > 0 and model_owner != "system":
            self.finalize_matured_unstakes(model_owner, block.index)
            if self.get_balance(model_owner) < remainder and allow_debit_bootstrap_stake:
                short = float(remainder) - float(self.get_balance(model_owner))
                st = float(self.get_staked_amount(model_owner))
                take = min(max(0.0, short), st)
                if take > 1e-12:
                    self._unstake_immediate_to_liquid(model_owner, take)
            if self.get_balance(model_owner) < remainder:
                print(
                    f"Insufficient balance for training remainder {remainder:.4f} "
                    f"(owner {model_owner[:16]}…)"
                )
                return False

        if from_escrow > 0:
            self.model_training_escrow[model_id] = escrow_avail - from_escrow
            self.balances[miner_address] = self.balances.get(miner_address, 0.0) + from_escrow
            self.economic_events.append(
                EconomicEvent(
                    event_type=EconomicEventType.TRAINING_PAYMENT,
                    timestamp=time.time(),
                    block_index=block.index,
                    from_address=f"escrow:{model_id}",
                    to_address=miner_address,
                    amount=from_escrow,
                    description=(
                        f"Training payment (escrow) for {block.training_steps} steps on {model_id}"
                    ),
                    block_hash=block.hash,
                )
            )
            self.model_training_payments[model_id] = (
                self.model_training_payments.get(model_id, 0.0) + from_escrow
            )

        if remainder > 0:
            success = self._transfer(
                from_address=model_owner,
                to_address=miner_address,
                amount=remainder,
                description=(
                    f"Training payment for {block.training_steps} steps on model {model_id}"
                ),
                event_type=EconomicEventType.TRAINING_PAYMENT,
                block_index=block.index,
                block_hash=block.hash,
            )
            if not success:
                return False
            self.model_training_payments[model_id] = (
                self.model_training_payments.get(model_id, 0.0) + remainder
            )

        if total_cost > 0:
            print(f"💳 Training payment: {total_cost:.4f} DESSIN to {miner_address}")
            print(f"   Base cost: {training_cost:.4f} DESSIN; complexity ×{model_complexity_factor:.2f}")
            if from_escrow > 0:
                print(f"   From escrow: {from_escrow:.4f} DESSIN")
            if remainder > 0:
                print(f"   From owner: {remainder:.4f} DESSIN")

        return True
    
    def distribute_attestation_rewards(
        self,
        verifiers: List[str],
        block: PogoBlock,
        *,
        auto_stake: bool = False,
    ) -> Dict[str, float]:
        """Distribute attestation rewards to verifiers (liquid or auto-staked like block rewards)."""
        rewards: Dict[str, float] = {}
        block_hash = block.hash
        block_index = block.index

        for verifier in verifiers:
            reward_amount = self.attestation_reward
            if auto_stake:
                success = self._mint_from_genesis_to_stake(
                    verifier,
                    reward_amount,
                    block,
                    description=f"Auto-staked attestation reward block {block_index}",
                )
            else:
                success = self._transfer(
                    from_address="system",
                    to_address=verifier,
                    amount=reward_amount,
                    description=f"Attestation reward for block {block_index}",
                    event_type=EconomicEventType.ATTESTATION_REWARD,
                    block_index=block_index,
                    block_hash=block_hash,
                )

            if success:
                rewards[verifier] = reward_amount
                tag = "auto-staked" if auto_stake else "liquid"
                print(f"🏆 Attestation reward ({tag}): {reward_amount:.4f} DESSIN to {verifier}")

        return rewards
    
    def process_slashing_penalty(self, miner_address: str, penalty_amount: float, 
                                block_hash: str, block_index: int) -> bool:
        """Process slashing penalty for invalid block"""
        # Slash from staked amount first, then from balance
        staked_amount = self.get_staked_amount(miner_address)
        balance_amount = self.get_balance(miner_address)
        
        if staked_amount + balance_amount < penalty_amount:
            penalty_amount = staked_amount + balance_amount
        
        # Slash from stake first
        if staked_amount > 0:
            slash_from_stake = min(penalty_amount, staked_amount)
            self.staked_amounts[miner_address] -= slash_from_stake
            penalty_amount -= slash_from_stake
        
        # Slash from balance if needed
        if penalty_amount > 0:
            self.balances[miner_address] -= penalty_amount
        
        # Record slashing event
        event = EconomicEvent(
            event_type=EconomicEventType.SLASHING_PENALTY,
            timestamp=time.time(),
            block_index=block_index,
            from_address=miner_address,
            to_address="system",
            amount=penalty_amount,
            description=f"Slashing penalty for invalid block {block_index}",
            block_hash=block_hash
        )
        self.economic_events.append(event)
        
        print(f"🔨 Slashing penalty: {penalty_amount:.4f} DESSIN from {miner_address}")
        return True
    
    def process_query_payment(self, user_address: str, model_owner: str, 
                            tokens_used: int, block_index: int = 0) -> bool:
        """Process payment for model query"""
        query_cost = tokens_used * self.query_fee_per_token
        
        success = self._transfer(
            from_address=user_address,
            to_address=model_owner,
            amount=query_cost,
            description=f"Query payment for {tokens_used} tokens",
            event_type=EconomicEventType.QUERY_PAYMENT,
            block_index=block_index
        )
        
        if success:
            print(f"🔍 Query payment: {query_cost:.4f} DESSIN from {user_address} to {model_owner}")
        
        return success
    
    def process_storage_rental(
        self,
        model_owner: str,
        model_id: str,
        model_size_gb: float,
        storage_blocks: int,
        block_index: int = 0
    ) -> bool:
        """
        Process upfront storage rental payment
        
        Args:
            model_owner: Owner's address
            model_id: Model ID
            model_size_gb: Model size in GB
            storage_blocks: Number of blocks to pay for
            block_index: Current block index
        
        Returns:
            True if successful
        """
        # Calculate storage cost with dynamic pricing
        base_cost = model_size_gb * storage_blocks * self.storage_fee_per_gb_per_block
        storage_cost = base_cost * self.storage_price_multiplier
        
        success = self._transfer(
            from_address=model_owner,
            to_address="system",
            amount=storage_cost,
            description=f"Storage rental for model {model_id}: {model_size_gb:.2f}GB for {storage_blocks} blocks",
            event_type=EconomicEventType.STORAGE_RENTAL,
            block_index=block_index
        )
        
        if success:
            # Track storage payment
            self.model_storage_payments[model_id] = self.model_storage_payments.get(model_id, 0.0) + storage_cost
            
            print(f"💾 Storage rental: {storage_cost:.4f} DESSIN from {model_owner}")
            print(f"   Model: {model_id}")
            print(f"   Size: {model_size_gb:.2f} GB")
            print(f"   Duration: {storage_blocks} blocks")
            print(f"   Price multiplier: {self.storage_price_multiplier:.2f}x")
        
        return success
    
    def process_storage_extension(
        self,
        model_owner: str,
        model_id: str,
        model_size_gb: float,
        additional_blocks: int,
        block_index: int = 0
    ) -> bool:
        """
        Process storage extension payment
        
        Args:
            model_owner: Owner's address
            model_id: Model ID
            model_size_gb: Model size in GB
            additional_blocks: Number of additional blocks
            block_index: Current block index
        
        Returns:
            True if successful
        """
        # Calculate extension cost with dynamic pricing
        base_cost = model_size_gb * additional_blocks * self.storage_fee_per_gb_per_block
        extension_cost = base_cost * self.storage_price_multiplier
        
        success = self._transfer(
            from_address=model_owner,
            to_address="system",
            amount=extension_cost,
            description=f"Storage extension for model {model_id}: {additional_blocks} additional blocks",
            event_type=EconomicEventType.STORAGE_EXTENSION,
            block_index=block_index
        )
        
        if success:
            # Track storage payment
            self.model_storage_payments[model_id] = self.model_storage_payments.get(model_id, 0.0) + extension_cost
            
            print(f"💾 Storage extension: {extension_cost:.4f} DESSIN from {model_owner}")
            print(f"   Model: {model_id}")
            print(f"   Additional blocks: {additional_blocks}")
        
        return success
    
    def process_training_iteration_payment(
        self,
        model_owner: str,
        miner_address: str,
        model_id: str,
        iterations: int = 1,
        block_index: int = 0,
        block_hash: str = None
    ) -> bool:
        """
        Process payment for training iterations
        
        Args:
            model_owner: Model owner's address
            miner_address: Miner's address
            model_id: Model ID
            iterations: Number of iterations
            block_index: Current block index
            block_hash: Block hash
        
        Returns:
            True if successful
        """
        training_cost = iterations * self.training_iteration_payment
        
        success = self._transfer(
            from_address=model_owner,
            to_address=miner_address,
            amount=training_cost,
            description=f"Training payment for {iterations} iterations on model {model_id}",
            event_type=EconomicEventType.TRAINING_ITERATION_PAYMENT,
            block_index=block_index,
            block_hash=block_hash
        )
        
        if success:
            # Track training payment
            self.model_training_payments[model_id] = self.model_training_payments.get(model_id, 0.0) + training_cost
            
            print(f"🔨 Training iteration payment: {training_cost:.4f} DESSIN from {model_owner} to {miner_address}")
            print(f"   Model: {model_id}")
            print(f"   Iterations: {iterations}")
        
        return success
    
    def adjust_storage_price(self, new_multiplier: float) -> bool:
        """
        Adjust storage price multiplier (dynamic pricing by nodes)
        
        Args:
            new_multiplier: New price multiplier
        
        Returns:
            True if successful
        """
        if not (self.min_storage_price_multiplier <= new_multiplier <= self.max_storage_price_multiplier):
            print(f"⚠️  Invalid storage price multiplier: {new_multiplier}")
            print(f"   Must be between {self.min_storage_price_multiplier} and {self.max_storage_price_multiplier}")
            return False
        
        old_multiplier = self.storage_price_multiplier
        self.storage_price_multiplier = new_multiplier
        
        print(f"📊 Storage price adjusted: {old_multiplier:.2f}x → {new_multiplier:.2f}x")
        
        return True
    
    def get_current_storage_price_per_gb_per_block(self) -> float:
        """Get current storage price per GB per block"""
        return self.storage_fee_per_gb_per_block * self.storage_price_multiplier
    
    def get_model_economics(self, model_id: str) -> Dict[str, Any]:
        """Get economic summary for a specific model"""
        return {
            "model_id": model_id,
            "total_storage_paid": self.model_storage_payments.get(model_id, 0.0),
            "total_training_paid": self.model_training_payments.get(model_id, 0.0),
            "training_escrow_balance": self.model_training_escrow.get(model_id, 0.0),
            "total_cost": (
                self.model_storage_payments.get(model_id, 0.0) +
                self.model_training_payments.get(model_id, 0.0)
            ),
        }
    
    def stake_tokens(self, address: str, amount: float, *, block_index: int = 0) -> bool:
        """Stake tokens for verification"""
        self.finalize_matured_unstakes(address, block_index)
        if self.get_balance(address) < amount:
            return False

        self.balances[address] -= amount
        self.staked_amounts[address] = self.staked_amounts.get(address, 0.0) + amount

        print(f"🔒 Staked: {amount:.4f} DESSIN by {address}")
        return True

    def unstake_tokens(
        self,
        address: str,
        amount: float,
        *,
        current_block_index: int = 0,
        cooldown_blocks: int = 0,
    ) -> bool:
        """Unstake: ``cooldown_blocks`` > 0 queues unbonding; 0 moves stake to liquid immediately (legacy)."""
        return self.request_unstake(address, amount, current_block_index, cooldown_blocks)
    
    def get_account_info(self, address: str) -> AccountBalance:
        """Get comprehensive account information"""
        return AccountBalance(
            address=address,
            balance=self.get_balance(address),
            staked_amount=self.get_staked_amount(address),
            total_earned=self._calculate_total_earned(address),
            total_spent=self._calculate_total_spent(address),
            last_updated=time.time()
        )
    
    def _calculate_total_earned(self, address: str) -> float:
        """Calculate total earned by an address"""
        total = 0.0
        for event in self.economic_events:
            if event.to_address == address and event.event_type in [
                EconomicEventType.BLOCK_REWARD,
                EconomicEventType.ATTESTATION_REWARD,
                EconomicEventType.TRAINING_PAYMENT,
                EconomicEventType.QUERY_PAYMENT
            ]:
                total += event.amount
        return total
    
    def _calculate_total_spent(self, address: str) -> float:
        """Calculate total spent by an address"""
        total = 0.0
        for event in self.economic_events:
            if event.from_address == address and event.event_type in [
                EconomicEventType.TRAINING_PAYMENT,
                EconomicEventType.QUERY_PAYMENT,
                EconomicEventType.STORAGE_FEE,
                EconomicEventType.SLASHING_PENALTY
            ]:
                total += event.amount
        return total
    
    def get_economic_summary(self) -> Dict[str, Any]:
        """Get comprehensive economic summary"""
        total_circulating = sum(self.balances.values())
        total_staked = sum(self.staked_amounts.values())
        
        # Calculate recent activity (last 100 events)
        recent_events = self.economic_events[-100:] if len(self.economic_events) > 100 else self.economic_events
        
        event_counts = {}
        total_volume = 0.0
        
        for event in recent_events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            total_volume += event.amount
        
        return {
            "total_supply": self.total_supply,
            "total_circulating": total_circulating,
            "total_staked": total_staked,
            "active_accounts": len([addr for addr, bal in self.balances.items() if bal > 0]),
            "staked_accounts": len([addr for addr, staked in self.staked_amounts.items() if staked > 0]),
            "recent_activity": {
                "event_counts": event_counts,
                "total_volume": total_volume,
                "events_analyzed": len(recent_events)
            },
            "economic_parameters": {
                "mining_reward": self.mining_reward,
                "training_payment_per_step": self.training_payment_per_step,
                "attestation_reward": self.attestation_reward,
                "query_fee_per_token": self.query_fee_per_token
            }
        }
    
    def get_top_accounts(self, limit: int = 10) -> List[AccountBalance]:
        """Get top accounts by total balance"""
        accounts = []
        for address in set(list(self.balances.keys()) + list(self.staked_amounts.keys())):
            if address != "system":
                account_info = self.get_account_info(address)
                accounts.append(account_info)
        
        # Sort by total balance (liquid + staked)
        accounts.sort(key=lambda x: x.balance + x.staked_amount, reverse=True)
        return accounts[:limit]
    
    def export_economic_data(self) -> Dict[str, Any]:
        """Export all economic data for analysis"""
        return {
            "balances": self.balances.copy(),
            "staked_amounts": self.staked_amounts.copy(),
            "unbonding_queue": {
                k: [[a, u] for a, u in v] for k, v in self.unbonding_queue.items()
            },
            "economic_events": [event.to_dict() for event in self.economic_events],
            "summary": self.get_economic_summary(),
            "top_accounts": [account.to_dict() for account in self.get_top_accounts()]
        }
