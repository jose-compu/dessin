"""
Main DeSSIN blockchain node implementation.
"""

import math
import os
import time
import threading
import hashlib
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..networking.chaincraft_peer_debug import LoggingChaincraftNode
from chaincraft.shared_message import SharedMessage

from .config import DessinConfig
from ..consensus import PogoConsensus, PogoBlock, AttestationType
from ..consensus.enhanced_pogo_consensus import EnhancedPogoConsensus, NOOP_TRAINING_MODEL_ID
from ..models.model_manager import ModelManager
from ..consensus.transactions import (
    TransactionFactory, BaseTransaction, ModelQueryTransaction, 
    ConditionalTransferTransaction, ModelUploadTransaction
)
from ..nanochat.nanochat_integration import NanochatIntegration, TrainingDevice
from ..nanochat.nanochat_web import NanochatWebServer
from ..economics.dynamic_parameters import DynamicParameterManager, ParameterType
from ..economics.storage_topup import StorageTopUpManager
from ..networking.node_operator_api import (
    NodeOperatorAPI, 
    PricingParameter, 
    PricingStrategy,
    PricingTarget
)
from .pretty_console import (
    pretty_print,
    format_interval_sec,
    bind_pretty_log_node,
    resolve_log_node_label_from_config,
)


class DessinNode:
    """Main DeSSIN blockchain node"""
    
    def __init__(
        self, 
        config: Optional[DessinConfig] = None,
        private_key: Optional[str] = None,
        public_key: Optional[str] = None
    ):
        self.config = config or DessinConfig.default()
        self._log_node_label = resolve_log_node_label_from_config(self.config)

        # Generate keys if not provided
        from chaincraft.crypto_primitives.sign import ECDSASignaturePrimitive
        from cryptography.hazmat.primitives import serialization
        
        if private_key is None or public_key is None:
            # Generate new keypair using chaincraft
            crypto = ECDSASignaturePrimitive()
            crypto.generate_key()
            self.private_key = crypto.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode('utf-8')
            self.public_key = crypto.get_public_pem()
            # Generate address from public key (simplified)
            import hashlib
            self.address = "0x" + hashlib.sha256(self.public_key.encode()).hexdigest()[:40]
        else:
            self.private_key = private_key
            self.public_key = public_key
            # Generate address from public key (simplified)
            import hashlib
            self.address = "0x" + hashlib.sha256(self.public_key.encode()).hexdigest()[:40]
        
        # Initialize components
        self.model_manager = ModelManager(self.config.model)
        
        # Initialize dynamic parameter manager (network-wide)
        self.dynamic_params = DynamicParameterManager()
        
        # Initialize storage top-up manager
        self.storage_topup = StorageTopUpManager()
        
        # Initialize node operator API (per-node pricing)
        self.operator_api = NodeOperatorAPI()
        
        # Initialize nanochat integration
        self.nanochat_integration = NanochatIntegration(self.model_manager)
        self.model_manager.nanochat_integration = self.nanochat_integration
        
        # Web server (will be initialized on demand)
        self.web_server: Optional[NanochatWebServer] = None
        
        # Use EnhancedPogoConsensus which includes economic system
        self.consensus = EnhancedPogoConsensus(
            self.config,  # Pass full config so consensus can access model_cache_dir
            self.model_manager,
            self.address
        )
        
        # Initialize chaincraft networking (logging subclass; ban/strike logs if DESSIN_DEBUG_PEER_BANS)
        self.chaincraft_node = LoggingChaincraftNode(
            max_peers=self.config.network.max_peers,
            port=self.config.network.port,
            use_compression=self.config.network.use_compression,
            local_discovery=self.config.network.local_discovery,
        )
        
        # Add consensus as shared object
        self.chaincraft_node.add_shared_object(self.consensus)
        
        # Token balances (simplified)
        self.balances: Dict[str, float] = {
            self.address: 1000.0  # Initial balance
        }
        
        # Transaction pool
        self.pending_transactions: Dict[str, BaseTransaction] = {}
        
        # Model pricing
        self.token_prices: Dict[str, float] = {}  # model_id -> price per token
        
        # Mining state
        self.is_mining = False
        self.mining_thread: Optional[threading.Thread] = None

        with bind_pretty_log_node(self._log_node_label):
            pretty_print(f"DeSSIN node initialized with address: {self.address}", kind="setup")
    
    def start(self) -> bool:
        """Start the DeSSIN node"""
        try:
            with bind_pretty_log_node(self._log_node_label):
                # Start chaincraft networking
                pretty_print(
                    f"Starting chaincraft node on port {self.config.network.port}…",
                    kind="wait",
                )
                self.chaincraft_node.start()  # ChaincraftNode.start() returns None, not boolean

                pretty_print(
                    f"DeSSIN node started on port {self.chaincraft_node.port}",
                    kind="ok",
                )

                # Start mining if configured
                if self.config.consensus.block_time_minutes > 0:
                    self.start_mining()

            return True

        except Exception as e:
            pretty_print(f"Error starting node: {e}", kind="bad")
            return False
    
    def stop(self):
        """Stop the DeSSIN node"""
        try:
            with bind_pretty_log_node(self._log_node_label):
                self.stop_mining()

                # Stop web server if running
                if self.web_server:
                    self.web_server.stop()
                    self.web_server = None

                # Use close() method instead of stop() for ChaincraftNode
                if hasattr(self.chaincraft_node, 'close'):
                    self.chaincraft_node.close()
                elif hasattr(self.chaincraft_node, 'stop'):
                    self.chaincraft_node.stop()
                pretty_print("DeSSIN node stopped", kind="ok")

        except Exception as e:
            pretty_print(f"Error stopping node: {e}", kind="bad")
    
    def start_mining(self):
        """Start mining PoGO blocks"""
        if self.is_mining:
            return
        
        self.is_mining = True
        if hasattr(self.consensus, "register_bootstrap_participant"):
            self.consensus.register_bootstrap_participant(self.address)
        self.mining_thread = threading.Thread(target=self._mining_loop, daemon=True)
        self.mining_thread.start()
        with bind_pretty_log_node(self._log_node_label):
            pretty_print("Started mining", kind="ok")
            if hasattr(self.consensus, "get_current_block_time_seconds"):
                slot = float(self.consensus.get_current_block_time_seconds())
                cadence_sec = float(
                    max(0, math.ceil(float(self.config.consensus.block_time_minutes) * 60.0 - 1e-9))
                )
                pretty_print(
                    f"Initial timer │ target slot {format_interval_sec(slot)} │ "
                    f"miner cadence {format_interval_sec(cadence_sec)} "
                    f"({self.config.consensus.block_time_minutes:.4f} min)",
                    kind="clock",
                )
    
    def stop_mining(self):
        """Stop mining"""
        self.is_mining = False
        if self.mining_thread:
            self.mining_thread.join(timeout=2.0)
            self.mining_thread = None
        with bind_pretty_log_node(self._log_node_label):
            pretty_print("Stopped mining", kind="info")

    def _interruptible_sleep(self, seconds: float, chunk_seconds: float = 0.2) -> None:
        """Sleep up to seconds but return early when is_mining becomes False."""
        if seconds <= 0:
            return
        deadline = time.monotonic() + seconds
        while self.is_mining:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(chunk_seconds, remaining))
    
    def _mining_loop(self):
        """Main mining loop"""
        with bind_pretty_log_node(self._log_node_label):
            while self.is_mining:
                try:
                    # Sleep for block time (whole seconds; ceil so fractional minutes never undersleep)
                    sleep_time = float(
                        max(0, math.ceil(float(self.config.consensus.block_time_minutes) * 60.0 - 1e-9))
                    )
                    # For testing, use much shorter intervals
                    if sleep_time > 60:  # If > 1 minute, use test interval
                        sleep_time = 10.0  # 10 seconds for testing

                    if hasattr(self.consensus, "get_current_block_time_seconds"):
                        slot = float(self.consensus.get_current_block_time_seconds())
                        pretty_print(
                            f"Mining round │ sleep {format_interval_sec(sleep_time)} next │ "
                            f"consensus target slot {format_interval_sec(slot)} │ "
                            f"cadence {self.config.consensus.block_time_minutes:.4f} min",
                            kind="clock",
                        )
                    self._interruptible_sleep(sleep_time)

                    # Attempt to mine a block
                    self.mine_block()

                except Exception as e:
                    pretty_print(f"Mining error: {e}", kind="bad")
                    self._interruptible_sleep(5)  # Brief pause on error (stops promptly if shutting down)
    
    def mine_block(self) -> Optional[str]:
        """Mine a new PoGO block"""
        with bind_pretty_log_node(self._log_node_label):
            try:
                import random

                # Select a model to train (simplified random selection)
                available_models = [
                    model_id
                    for model_id, model_info in self.model_manager.models.items()
                    if self.model_manager.is_model_available(model_id, self.consensus.get_chain_length())
                ]

                # For testing, create a dummy model if none exist
                if not available_models:
                    pretty_print(
                        "No models available for training, creating dummy model…",
                        kind="warn",
                    )
                    self._create_dummy_model()
                    available_models = list(self.model_manager.models.keys())

                if not available_models:
                    pretty_print(
                        "Still no models available after creating dummy — will try no-op block if supported",
                        kind="warn",
                    )
                    trainable: list[str] = []
                else:
                    trainable = [
                        mid
                        for mid in available_models
                        if not self.consensus.is_model_training_frozen(mid)
                        and not self.consensus.is_model_training_locked(mid)
                    ]

                if available_models and not trainable:
                    pretty_print(
                        "No trainable models (all frozen, locked, or unavailable); may fall back to no-op block",
                        kind="warn",
                    )

                if hasattr(self.consensus, "register_bootstrap_participant"):
                    self.consensus.register_bootstrap_participant(self.address)
                if hasattr(self.consensus, "is_eligible_to_mine") and not self.consensus.is_eligible_to_mine(
                    self.address
                ):
                    pretty_print(
                        "Mining skipped: address not eligible (bootstrap active set or post-bootstrap 0.1% stake)",
                        kind="warn",
                    )
                    return None

                # Proposer lottery before affordability: only the selected miner should
                # evaluate fundable models (avoids duplicate warnings on other nodes).
                if hasattr(self.consensus, "expected_training_leader"):
                    expect = self.consensus.expected_training_leader()
                    if expect is not None and expect != self.address:
                        pretty_print(
                            "Mining skipped: not the selected training proposer for this height "
                            f"(selected={expect[:20]}…; lottery binds tip miner + prior block hash — "
                            "see EnhancedPogoConsensus.expected_training_leader)",
                            kind="detail",
                        )
                        return None

                next_idx = len(self.consensus.chain)
                pretty_print(
                    f"⛏️ 🪙 🧱 🧮 Mining Block #{next_idx} (next chain index) · selected proposer round",
                    kind="training",
                )

                train_targets: list[str] = []
                if trainable and hasattr(self.consensus, "can_owner_afford_training"):
                    steps_for_estimate = 20
                    fundable: list[str] = []
                    skipped: list[tuple[str, str]] = []
                    for mid in trainable:
                        try:
                            ok, why, _cost, _funds = self.consensus.can_owner_afford_training(
                                mid, steps_for_estimate
                            )
                        except Exception as exc:
                            pretty_print(
                                f"Affordability check error for {mid}: {exc!r} — keeping in pool",
                                kind="warn",
                            )
                            fundable.append(mid)
                            continue
                        if ok:
                            fundable.append(mid)
                        else:
                            skipped.append((mid, why))
                    for mid, why in skipped:
                        pretty_print(
                            f"Affordability filter: skipping {mid} — {why}",
                            kind="warn",
                        )
                    if not fundable:
                        pretty_print(
                            "No fundable models for training this round — publishing no-op block "
                            "so height and base reward still advance (training payment skipped).",
                            kind="warn",
                        )
                    else:
                        train_targets = fundable
                elif trainable:
                    train_targets = trainable

                block = None
                if train_targets:
                    tip = self.consensus.get_latest_block().hash
                    rng = random.Random(hash((tip, tuple(sorted(train_targets)))))
                    model_order = train_targets[:]
                    rng.shuffle(model_order)
                    for model_id in model_order:
                        if hasattr(self.consensus, "create_enhanced_training_block"):
                            block = self.consensus.create_enhanced_training_block(model_id)
                        else:
                            block = self.consensus.create_training_block(model_id)
                        if block:
                            break

                if block is None and hasattr(self.consensus, "create_empty_round_block"):
                    block = self.consensus.create_empty_round_block()

                if not block:
                    pretty_print(
                        "No block produced (training failed and no-op blocks not available on this consensus)",
                        kind="bad",
                    )
                    return None

                # Add block to local chain first (miner should add their own block)
                self.consensus.chain.append(block)
                self.consensus.observe_committed_training_block(block)
                pretty_print(f"Added block {block.index} to local chain", kind="block")
                if hasattr(self.consensus, "log_miner_bootstrap_snapshot"):
                    self.consensus.log_miner_bootstrap_snapshot(self.address, block.index)
                if hasattr(self.consensus, "get_current_block_time_seconds"):
                    slot = float(self.consensus.get_current_block_time_seconds())
                    cadence_sec = float(
                        max(0, math.ceil(float(self.config.consensus.block_time_minutes) * 60.0 - 1e-9))
                    )
                    pretty_print(
                        f"After block {block.index} │ target slot {format_interval_sec(slot)} │ "
                        f"miner cadence {format_interval_sec(cadence_sec)}",
                        kind="clock",
                    )
            
                # Broadcast the block - Tendermint style with simple message structure
                block_dict = block.to_dict()
            
                # Ensure all values are JSON serializable
                import json
                try:
                    json.dumps(block_dict)  # Test serialization
                except Exception as e:
                    pretty_print(f"Block serialization error: {e}", kind="bad")
                    pretty_print(f"Problematic block data: {block_dict}", kind="warn")
                    return block.hash  # Return without broadcasting if serialization fails
            
                message_data = {
                    "message_type": "POGO_BLOCK",  # Use message_type like Tendermint
                    "height": block.index,
                    "block_data": block_dict  # All simple types only
                }
            
                # Test message serialization and create simplified message for Chaincraft
                try:
                    json.dumps(message_data)

                    simplified_message = {
                        "message_type": "POGO_BLOCK",
                        "height": block.index,
                        "block_hash": block.hash,
                        "miner": block.miner,
                        "model_id": block.model_id,
                        "torrent_hash": block.torrent_hash,
                    }

                    # Gossip full block so peers can extend their chain (hash-only left listeners stuck).
                    try:
                        _, _msg = self.chaincraft_node.create_shared_message(message_data)
                    except Exception as exc_full:
                        pretty_print(
                            f"Full block gossip failed ({exc_full!r}); sending hash-only fallback",
                            kind="warn",
                        )
                        _, _msg = self.chaincraft_node.create_shared_message(simplified_message)
                    pretty_print(
                        f"Gossiped block notification: {block.hash[:8]}",
                        kind="gossip",
                    )

                except Exception as e:
                    pretty_print(
                        f"Warning: Could not broadcast block notification: {e}",
                        kind="warn",
                    )
                if block.model_id == NOOP_TRAINING_MODEL_ID:
                    pretty_print(
                        f"Mined no-op block {block.index} (base reward only), hash: {block.hash[:8]}",
                        kind="block",
                    )
                else:
                    pretty_print(
                        f"Mined block {block.index} for model {block.model_id}, hash: {block.hash[:8]}",
                        kind="block",
                    )
            
                return block.hash
            
            except Exception as e:
                pretty_print(f"Error mining block: {e}", kind="bad")
                return None
    
    def _create_dummy_model(
        self,
        model_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Register a minimal catalog entry so mining can target ``model_id`` (testing / demos).

        Default model id is the configured **decoder-LM variant** (``DESSIN_LLM_VARIANT``,
        default ``micro_gpt_char``). The legacy ``simple_neural_network`` MLP catalog
        is gone: every code path that calls this helper now produces an LM catalog
        entry consistent with the trainer that actually runs in ``mine_block``.
        Pass an explicit ``model_id`` only when registering a non-default model.
        """
        variant = os.environ.get("DESSIN_LLM_VARIANT", "micro_gpt_char").strip().lower()
        if variant in ("gpt2_nano", "gpt2", "nanogpt", "gpt2-nano"):
            default_mid = "gpt2_nano"
            default_name = "GPT-2 nano (decoder LM)"
        else:
            default_mid = "micro_gpt_char"
            default_name = "Micro GPT (char)"
        mid = (model_id or default_mid).strip().lower()
        display_name = name or default_name
        raw = os.environ.get("DESSIN_CANONICAL_MODEL_OWNER", "").strip()
        economic_owner = raw if raw else self.address
        # Catalog metadata only; on-chain size comes from real LM training
        # (~855 KiB float64 payload for the micro preset).
        self.model_manager.register_model(
            model_id=mid,
            name=display_name,
            size_gb=0.001,
            format="pytorch",
            quantization="32bit",
            parameters=110_000,
            ipfs_hash="QmDecoderLmStub",
            owner=economic_owner,
            upload_block=0,
            storage_expires=1000,
            model_hash=f"{mid}_catalog_stub",
        )
        self.consensus.register_model_owner(mid, economic_owner)
    
    def submit_transaction(self, transaction: BaseTransaction) -> bool:
        """Submit a transaction to the network"""
        try:
            # Validate transaction
            if not TransactionFactory.verify_transaction(transaction):
                print("Invalid transaction signature")
                return False
            
            # Check if sender has sufficient balance for fees
            sender_balance = self.balances.get(transaction.sender, 0.0)
            if sender_balance < transaction.fee:
                print("Insufficient balance for transaction fee")
                return False
            
            # Add to pending transactions
            self.pending_transactions[transaction.tx_id] = transaction
            
            # Broadcast transaction - Tendermint style
            message_data = {
                "message_type": "POGO_TRANSACTION",
                "transaction_data": transaction.to_dict()
            }
            
            self.chaincraft_node.create_shared_message(message_data)
            print(f"Submitted transaction {transaction.tx_id[:8]}")
            
            return True
            
        except Exception as e:
            print(f"Error submitting transaction: {e}")
            return False
    
    def query_model(
        self, 
        model_id: str, 
        query: str, 
        max_tokens: int = 100
    ) -> Optional[str]:
        """Query a model and create payment transaction"""
        try:
            # Check if model exists
            if model_id not in self.model_manager.models:
                print(f"Model {model_id} not found")
                return None
            
            # Get token price for this model
            token_price = self.token_prices.get(model_id, 0.001)  # Default price
            
            # Calculate maximum cost
            max_cost = max_tokens * token_price
            
            # Check balance
            if self.balances.get(self.address, 0) < max_cost:
                print("Insufficient balance for model query")
                return None
            
            # Query the model
            result = self.model_manager.query_model(model_id, query, max_tokens)
            
            if not result.success:
                print(f"Model query failed: {result.error_message}")
                return None
            
            # Calculate actual cost
            actual_cost = result.tokens_used * token_price
            
            # Create payment transaction
            query_tx = TransactionFactory.create_signed_transaction(
                tx_type="model_query",
                sender=self.address,
                private_key=self.private_key,
                public_key=self.public_key,
                model_id=model_id,
                query_data=query,
                max_tokens=max_tokens,
                token_price=token_price,
                fee=0.001
            )
            
            # Submit transaction
            self.submit_transaction(query_tx)
            
            # Deduct cost from balance (simplified)
            self.balances[self.address] -= actual_cost + query_tx.fee
            
            print(f"Model query cost: {actual_cost:.4f} DESSIN for {result.tokens_used} tokens")
            
            return result.response
            
        except Exception as e:
            print(f"Error querying model: {e}")
            return None
    
    def upload_model(
        self, 
        model_name: str,
        model_path: str,
        storage_blocks: int = 1000
    ) -> Optional[str]:
        """Upload a new model to the network"""
        try:
            # Validate model file
            if not Path(model_path).exists():
                print(f"Model file not found: {model_path}")
                return None
            
            # Calculate model details
            model_hash = self.model_manager.get_model_hash(model_path)
            model_size = self.model_manager.estimate_model_size(model_path)
            
            # Calculate storage cost
            storage_cost = (model_size * 
                          self.config.token.base_storage_price * 
                          storage_blocks)
            
            # Check balance
            if self.balances.get(self.address, 0) < storage_cost:
                print(f"Insufficient balance for storage: need {storage_cost:.4f} DESSIN")
                return None
            
            # Create upload transaction
            upload_tx = TransactionFactory.create_signed_transaction(
                tx_type="model_upload",
                sender=self.address,
                private_key=self.private_key,
                public_key=self.public_key,
                model_name=model_name,
                model_hash=model_hash,
                model_size_gb=model_size,
                storage_blocks=storage_blocks,
                storage_payment=storage_cost,
                fee=0.01
            )
            
            # Submit transaction
            success = self.submit_transaction(upload_tx)
            if not success:
                return None
            
            # Register model locally (in practice, would upload to IPFS first)
            model_id = f"model_{model_hash[:8]}"
            self.model_manager.register_model(
                model_id=model_id,
                name=model_name,
                size_gb=model_size,
                format="gguf",
                quantization="4bit", 
                parameters=int(model_size * 1e9 / 4),  # Rough estimate
                ipfs_hash=f"Qm{model_hash[:44]}",  # Simulated IPFS hash
                owner=self.address,
                upload_block=self.consensus.get_chain_length(),
                storage_expires=self.consensus.get_chain_length() + storage_blocks,
                model_hash=model_hash
            )
            
            # Set default token price
            self.token_prices[model_id] = 0.001  # 0.001 DESSIN per token
            
            # Deduct storage cost
            self.balances[self.address] -= storage_cost + upload_tx.fee
            
            print(f"Uploaded model {model_id} with {storage_blocks} blocks storage")
            
            return model_id
            
        except Exception as e:
            print(f"Error uploading model: {e}")
            return None
    
    def create_conditional_transfer(
        self,
        recipient: str,
        amount: float,
        model_id: str,
        condition_query: str,
        condition_expected: str,
        fallback_recipient: Optional[str] = None
    ) -> bool:
        """Create a conditional transfer based on model output"""
        try:
            # Create conditional transfer transaction
            tx = TransactionFactory.create_signed_transaction(
                tx_type="conditional_transfer",
                sender=self.address,
                private_key=self.private_key,
                public_key=self.public_key,
                recipient=recipient,
                amount=amount,
                model_id=model_id,
                condition_query=condition_query,
                condition_expected=condition_expected,
                fallback_recipient=fallback_recipient,
                fee=0.001
            )
            
            return self.submit_transaction(tx)
            
        except Exception as e:
            print(f"Error creating conditional transfer: {e}")
            return False
    
    def get_balance(self, address: Optional[str] = None) -> float:
        """Get balance for an address"""
        if address is None:
            address = self.address
        return self.balances.get(address, 0.0)

    def request_unstake(self, amount: float) -> bool:
        """Queue withdrawal from protocol stake; liquid after ``unstake_cooldown_blocks`` (credited on spend)."""
        if amount <= 0:
            return False
        return self.consensus.submit_unstake_request(self.address, amount)

    def get_node_info(self) -> Dict[str, Any]:
        """Get general node information"""
        return {
            "address": self.address,
            "balance": self.get_balance(),
            "chain_length": self.consensus.get_chain_length(),
            "latest_block_hash": self.consensus.get_latest_block_hash(),
            "models_count": len(self.model_manager.models),
            "pending_transactions": len(self.pending_transactions),
            "is_mining": self.is_mining,
            "peers": len(self.chaincraft_node.peers),
            "config": {
                "block_time_minutes": self.config.consensus.block_time_minutes,
                "finalization_window": self.config.consensus.finalization_window,
                "max_peers": self.config.network.max_peers
            }
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        models = []
        current_block = self.consensus.get_chain_length()
        
        for model_id, model_info in self.model_manager.models.items():
            available = self.model_manager.is_model_available(model_id, current_block)
            price = self.token_prices.get(model_id, 0.001)
            
            models.append({
                "model_id": model_id,
                "name": model_info.name,
                "size_gb": model_info.size_gb,
                "parameters": model_info.parameters,
                "owner": model_info.owner,
                "available": available,
                "token_price": price,
                "expires_block": model_info.storage_expires
            })
        
        return models
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get blockchain information"""
        latest_block = self.consensus.get_latest_block()
        
        return {
            "chain_length": len(self.consensus.chain),
            "latest_block": {
                "index": latest_block.index,
                "hash": latest_block.hash,
                "miner": latest_block.miner,
                "timestamp": latest_block.timestamp,
                "model_id": latest_block.model_id,
                "loss_improvement": latest_block.loss_before - latest_block.loss_after
            },
            "pending_blocks": len(self.consensus.pending_blocks),
            "total_attestations": sum(len(atts) for atts in self.consensus.attestations.values())
        }
    
    def create_nanochat_model(
        self,
        name: str,
        depth: int,
        device_batch_size: int = 32,
        dataset_name: str = "fineweb",
        storage_blocks: int = 1000,
        storage_payment: float = 10.0,
        training_device: TrainingDevice = TrainingDevice.AUTO,
        **kwargs
    ) -> Optional[str]:
        """
        Create a new nanochat model (owner only operation)
        
        Args:
            name: Model name
            depth: Number of transformer layers (e.g., 20, 26, 32)
            device_batch_size: Batch size per device
            dataset_name: Training dataset name (owner only)
            storage_blocks: Number of blocks to pay for storage (minimum enforced)
            storage_payment: Upfront storage payment
            training_device: Training device preference (CPU/GPU/AUTO)
            **kwargs: Additional model configuration
        
        Returns:
            model_id if successful, None otherwise
        """
        mid = self.nanochat_integration.create_model(
            name=name,
            depth=depth,
            owner_address=self.address,
            device_batch_size=device_batch_size,
            dataset_name=dataset_name,
            storage_blocks=storage_blocks,
            storage_payment=storage_payment,
            training_device=training_device,
            **kwargs
        )
        if mid:
            self.consensus.register_model_owner(mid, self.address)
        return mid
    
    def refresh_model_training_data(
        self,
        model_id: str,
        training_data_manifest_hash: str,
        training_escrow_topup: float = 0.0,
        training_data_uri: Optional[str] = None,
        dataset_name: Optional[str] = None,
        fee: Optional[float] = None,
    ) -> bool:
        """
        Owner-only: commit to new training data, pay tx fee, optionally fund training escrow,
        and clear strike-based freeze so miners can train again.
        """
        cfg = self.config.consensus
        fee_eff = fee if fee is not None else cfg.training_refresh_min_fee
        total = fee_eff + training_escrow_topup
        es = self.consensus.economic_system
        tip = self.consensus.get_chain_length()
        es.finalize_matured_unstakes(self.address, tip)
        if es.get_balance(self.address) < total:
            short = total - es.get_balance(self.address)
            if self.balances.get(self.address, 0.0) < short:
                with bind_pretty_log_node(self._log_node_label):
                    pretty_print(
                        "Insufficient balance for fee + escrow (economic + local wallet)",
                        kind="bad",
                    )
                return False
            es.balances[self.address] = es.get_balance(self.address) + short
            self.balances[self.address] -= short
        return self.consensus.apply_model_training_data_refresh(
            owner_address=self.address,
            model_id=model_id,
            training_data_manifest_hash=training_data_manifest_hash,
            fee=fee_eff,
            training_escrow_topup=training_escrow_topup,
            training_data_uri=training_data_uri,
            dataset_name=dataset_name,
            block_index=self.consensus.get_chain_length(),
        )

    def topup_model_storage(
        self,
        model_id: str,
        additional_blocks: int,
        payment: float
    ) -> bool:
        """
        Top-up storage for any model (anyone can do this, not just owner)
        
        Args:
            model_id: Model ID to top-up
            additional_blocks: Number of additional blocks
            payment: Payment amount
        
        Returns:
            True if successful
        """
        try:
            # Get model info
            model_info = self.model_manager.get_model_info(model_id)
            if not model_info:
                print(f"Model not found: {model_id}")
                return False
            
            # Get current storage price
            storage_price = self.dynamic_params.storage_price
            
            # Process top-up
            topup = self.storage_topup.topup_storage(
                model_id=model_id,
                model_owner=model_info.owner,
                payer_address=self.address,
                model_size_gb=model_info.size_gb,
                additional_blocks=additional_blocks,
                payment=payment,
                current_expiration=model_info.storage_expires,
                storage_price_per_gb_per_block=storage_price,
                tx_hash=hashlib.sha256(f"{model_id}_{self.address}_{time.time()}".encode()).hexdigest()
            )
            
            if not topup:
                return False
            
            # Update model info
            model_info.storage_expires = topup.new_expiration_block
            model_info.storage_payment += payment
            
            # Deduct payment from payer
            if self.address in self.balances:
                self.balances[self.address] -= payment
            
            print(f"✓ Storage topped up successfully")
            return True
            
        except Exception as e:
            print(f"Error topping up storage: {e}")
            return False
    
    def adjust_network_parameter(
        self,
        parameter_type: ParameterType,
        increase: bool,
        reason: str = ""
    ) -> bool:
        """
        Adjust a network parameter (block leader only)
        
        Args:
            parameter_type: Type of parameter to adjust
            increase: True to increase, False to decrease
            reason: Reason for adjustment
        
        Returns:
            True if successful
        """
        try:
            # Get current block index
            block_index = self.consensus.get_chain_length()
            
            # Adjust parameter
            if parameter_type == ParameterType.BLOCK_REWARD:
                adjustment = self.dynamic_params.adjust_block_reward(
                    block_index=block_index,
                    block_leader=self.address,
                    increase=increase,
                    reason=reason
                )
            elif parameter_type == ParameterType.STORAGE_PRICE:
                adjustment = self.dynamic_params.adjust_storage_price(
                    block_index=block_index,
                    block_leader=self.address,
                    increase=increase,
                    reason=reason
                )
            elif parameter_type == ParameterType.TRAINING_PAYMENT:
                adjustment = self.dynamic_params.adjust_training_payment(
                    block_index=block_index,
                    block_leader=self.address,
                    increase=increase,
                    reason=reason
                )
            else:
                return False
            
            return adjustment is not None
            
        except Exception as e:
            print(f"Error adjusting parameter: {e}")
            return False
    
    def start_web_interface(self, port: int = 8000, host: str = "0.0.0.0") -> Optional[str]:
        """
        Start the nanochat web interface
        
        Args:
            port: Port to serve on
            host: Host to bind to
        
        Returns:
            URL if successful, None otherwise
        """
        if self.web_server:
            print("Web interface already running")
            return self.web_server.get_url()
        
        self.web_server = NanochatWebServer(
            model_manager=self.model_manager,
            nanochat_integration=self.nanochat_integration,
            node=self,
            port=port,
            host=host
        )
        
        if self.web_server.start():
            return self.web_server.get_url()
        else:
            self.web_server = None
            return None
    
    def stop_web_interface(self):
        """Stop the web interface"""
        if self.web_server:
            self.web_server.stop()
            self.web_server = None
    
    def set_storage_price(self, new_price: float, reason: str = "") -> bool:
        """
        Set this node's storage price (on-the-fly)
        
        Args:
            new_price: New storage price in DESSIN per GB per block
            reason: Reason for change
        
        Returns:
            True if successful
        """
        return self.operator_api.set_storage_price(new_price, reason)
    
    def set_block_reward_target(self, new_target: float, reason: str = "") -> bool:
        """
        Set this node's target block reward (on-the-fly)
        
        Args:
            new_target: Target block reward in DESSIN
            reason: Reason for change
        
        Returns:
            True if successful
        """
        return self.operator_api.set_block_reward_target(new_target, reason)
    
    def set_training_price(self, new_price: float, reason: str = "") -> bool:
        """
        Set this node's training price (on-the-fly)
        
        Args:
            new_price: New training price in DESSIN per iteration
            reason: Reason for change
        
        Returns:
            True if successful
        """
        return self.operator_api.set_training_price(new_price, reason)
    
    def set_query_price(self, new_price: float, reason: str = "") -> bool:
        """
        Set this node's query price (on-the-fly)
        
        Args:
            new_price: New query price in DESSIN per token
            reason: Reason for change
        
        Returns:
            True if successful
        """
        return self.operator_api.set_query_price(new_price, reason)
    
    def get_operator_pricing(self) -> Dict[str, Any]:
        """Get this node's pricing configuration"""
        return self.operator_api.get_pricing_summary()
    
    def bulk_update_pricing(
        self,
        updates: Dict[str, float],
        reason: str = ""
    ) -> Dict[str, bool]:
        """
        Update multiple prices at once
        
        Args:
            updates: Dict with keys like 'storage', 'training', 'query', 'block_reward'
            reason: Reason for changes
        
        Returns:
            Dict mapping parameter names to success status
        """
        return self.operator_api.bulk_update_prices(updates, reason)
    
    # ========================================================================
    # Block Leader Pricing Targets
    # ========================================================================
    
    def set_storage_price_target(
        self,
        target: Optional[float] = None,
        increase_percent: Optional[float] = None,
        decrease_percent: Optional[float] = None,
        duration_blocks: Optional[int] = None
    ) -> bool:
        """
        Set storage price target (applies when selected as block leader)
        
        Args:
            target: Absolute target price
            increase_percent: Percentage to increase (e.g., 0.5 for 50%)
            decrease_percent: Percentage to decrease (e.g., 0.5 for 50%)
            duration_blocks: Apply for N blocks (temporary)
        
        Returns:
            True if successful
        
        Examples:
            # Set target price
            node.set_storage_price_target(target=0.002)
            
            # Increase by 50%
            node.set_storage_price_target(increase_percent=0.5)
            
            # Decrease by 20% for 100 blocks
            node.set_storage_price_target(decrease_percent=0.2, duration_blocks=100)
            
            # Remove target
            node.set_storage_price_target()
        """
        if target is not None:
            return self.operator_api.set_pricing_target(
                PricingParameter.STORAGE_PRICE,
                PricingStrategy.ABSOLUTE_TARGET,
                target_value=target
            )
        
        elif increase_percent is not None:
            strategy = (PricingStrategy.TEMPORARY_ADJUSTMENT if duration_blocks 
                       else PricingStrategy.RELATIVE_INCREASE)
            return self.operator_api.set_pricing_target(
                PricingParameter.STORAGE_PRICE,
                strategy,
                percentage_change=abs(increase_percent),
                duration_blocks=duration_blocks
            )
        
        elif decrease_percent is not None:
            strategy = (PricingStrategy.TEMPORARY_ADJUSTMENT if duration_blocks 
                       else PricingStrategy.RELATIVE_DECREASE)
            return self.operator_api.set_pricing_target(
                PricingParameter.STORAGE_PRICE,
                strategy,
                percentage_change=-abs(decrease_percent),
                duration_blocks=duration_blocks
            )
        
        else:
            # Remove target
            return self.operator_api.set_pricing_target(
                PricingParameter.STORAGE_PRICE,
                PricingStrategy.NO_TARGET
            )
    
    def set_block_reward_target(
        self,
        target: Optional[float] = None,
        increase_percent: Optional[float] = None,
        decrease_percent: Optional[float] = None,
        duration_blocks: Optional[int] = None
    ) -> bool:
        """
        Set block reward target (applies when selected as block leader)
        
        Args:
            target: Absolute target reward
            increase_percent: Percentage to increase
            decrease_percent: Percentage to decrease  
            duration_blocks: Apply for N blocks (temporary)
        
        Returns:
            True if successful
        """
        if target is not None:
            return self.operator_api.set_pricing_target(
                PricingParameter.BLOCK_REWARD_TARGET,
                PricingStrategy.ABSOLUTE_TARGET,
                target_value=target
            )
        
        elif increase_percent is not None:
            strategy = (PricingStrategy.TEMPORARY_ADJUSTMENT if duration_blocks 
                       else PricingStrategy.RELATIVE_INCREASE)
            return self.operator_api.set_pricing_target(
                PricingParameter.BLOCK_REWARD_TARGET,
                strategy,
                percentage_change=abs(increase_percent),
                duration_blocks=duration_blocks
            )
        
        elif decrease_percent is not None:
            strategy = (PricingStrategy.TEMPORARY_ADJUSTMENT if duration_blocks 
                       else PricingStrategy.RELATIVE_DECREASE)
            return self.operator_api.set_pricing_target(
                PricingParameter.BLOCK_REWARD_TARGET,
                strategy,
                percentage_change=-abs(decrease_percent),
                duration_blocks=duration_blocks
            )
        
        else:
            # Remove target
            return self.operator_api.set_pricing_target(
                PricingParameter.BLOCK_REWARD_TARGET,
                PricingStrategy.NO_TARGET
            )
    
    def on_selected_as_block_leader(self):
        """
        Called when this node is selected as block leader
        
        Applies configured pricing targets to adjust network parameters
        """
        # Get current network prices
        current_prices = {
            'storage_price': self.dynamic_params.storage_price,
            'block_reward': self.dynamic_params.block_reward,
            'training_price': self.dynamic_params.training_payment
        }
        
        # Apply targets
        adjustments = self.operator_api.apply_as_block_leader(current_prices)
        
        # Execute adjustments
        for adj in adjustments:
            param = adj['parameter']
            increase = adj['increase']
            reason = adj['reason']
            
            # Map to dynamic parameter type
            if param == PricingParameter.STORAGE_PRICE:
                self.adjust_network_parameter(
                    ParameterType.STORAGE_PRICE,
                    increase=increase,
                    reason=reason
                )
            
            elif param == PricingParameter.BLOCK_REWARD_TARGET:
                self.adjust_network_parameter(
                    ParameterType.BLOCK_REWARD,
                    increase=increase,
                    reason=reason
                )
            
            elif param == PricingParameter.TRAINING_PRICE:
                self.adjust_network_parameter(
                    ParameterType.TRAINING_PAYMENT,
                    increase=increase,
                    reason=reason
                )
    
    def get_leader_targets(self) -> Dict[str, Any]:
        """
        Get configured block leader pricing targets
        
        Returns:
            Dict with target information
        """
        targets = self.operator_api.get_pricing_targets()
        stats = self.operator_api.get_leader_statistics()
        
        return {
            "targets": {
                param.value: {
                    "strategy": target.strategy.value,
                    "target_value": target.target_value,
                    "percentage_change": target.percentage_change,
                    "duration_blocks": target.duration_blocks,
                    "blocks_remaining": target.blocks_remaining,
                    "applied_count": target.applied_count
                }
                for param, target in targets.items()
                if not target.is_expired()
            },
            "statistics": stats
        }
