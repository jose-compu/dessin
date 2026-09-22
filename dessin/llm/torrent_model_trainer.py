#!/usr/bin/env python3
"""
Enhanced model trainer that creates actual model files for torrent sharing
"""

import os
import json
import time
import hashlib
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    from .simple_trainer import SimpleTrainer, TrainingResult
    from ..distribution.bittorrent_distributor import BitTorrentDistributor, TorrentInfo
    from ..runtime.pretty_console import pretty_print
except ImportError:
    # For direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from simple_trainer import SimpleTrainer, TrainingResult
    from bittorrent_distributor import BitTorrentDistributor, TorrentInfo
    from pretty_console import pretty_print


@dataclass
class TorrentTrainingResult(TrainingResult):
    """Training result with torrent information"""

    model_file_path: str = ""
    torrent_hash: str = ""
    magnet_link: str = ""
    model_file_size: int = 0


def _use_micro_gpt_training() -> bool:
    return os.environ.get("DESSIN_MICRO_GPT_TRAINING", "").strip().lower() in ("1", "true", "yes")


def _llm_variant_name() -> str:
    """Which decoder-only preset to run when LM training is enabled (see docs/E2E_NETWORK_TESTS.md)."""
    return os.environ.get("DESSIN_LLM_VARIANT", "micro_gpt_char").strip().lower()


def _is_gpt2_nano_variant(name: str) -> bool:
    return name in ("gpt2_nano", "gpt2", "nanogpt", "gpt2-nano")


class TorrentModelTrainer(SimpleTrainer):
    """Model trainer that creates torrent-shareable model files"""
    
    def __init__(self, torrent_dir: str = None):
        super().__init__()
        
        # Directory for storing model files and torrents
        self.torrent_dir = Path(torrent_dir) if torrent_dir else Path("model_cache/torrents")
        self.torrent_dir.mkdir(parents=True, exist_ok=True)
        
        pretty_print(f"TorrentModelTrainer using directory: {self.torrent_dir}", kind="info")
        
        # Real BitTorrent distributor for creating torrents
        self.real_bt_distributor = None
        try:
            from .real_torrent_distributor import RealTorrentDistributor
            # Use unique port and node ID to avoid conflicts
            unique_id = int(time.time()) % 10000  # Larger range for uniqueness
            unique_port = 6881 + (unique_id % 4)  # Cycle through 6881-6884
            
            self.real_bt_distributor = RealTorrentDistributor(
                node_id=f"trainer_{unique_id}",
                torrent_dir=str(self.torrent_dir),
                listen_port=unique_port
            )
            pretty_print(
                f"Real BitTorrent distributor using port {unique_port} and directory {self.torrent_dir}",
                kind="mesh",
            )
            pretty_print(
                "Real BitTorrent distributor initialized for model training",
                kind="ok",
            )
        except Exception as e:
            pretty_print(
                f"Could not initialize real BitTorrent distributor: {e}",
                kind="warn",
            )
            pretty_print("Falling back to mock torrent creation", kind="info")

    def train_model(
        self,
        learning_rate: float = 0.01,
        training_steps: int = 10,
        batch_size: int = 32,
        data_seed: Optional[int] = None,
        checkpoint_weights_hex: Optional[str] = None,
    ) -> TrainingResult:
        """Train MLP by default; use decoder-only LM when DESSIN_MICRO_GPT_TRAINING is set.
        
        If checkpoint_weights_hex is provided, the LM trainers will load those weights
        before training to continue from a previous checkpoint.
        """
        if _use_micro_gpt_training():
            seed = 0 if data_seed is None else int(data_seed)
            steps = int(os.environ.get("DESSIN_MICRO_GPT_STEPS", str(training_steps)))
            bs = int(os.environ.get("DESSIN_MICRO_GPT_BATCH", str(batch_size)))
            lr = float(os.environ.get("DESSIN_MICRO_GPT_LR", str(learning_rate)))
            variant = _llm_variant_name()
            if _is_gpt2_nano_variant(variant):
                from .gpt2_nano_trainer import train_gpt2_nano

                dev = os.environ.get("DESSIN_GPT2_DEVICE") or os.environ.get(
                    "DESSIN_MICRO_GPT_DEVICE", "cpu"
                )
                pretty_print(
                    f"GPT-2 nano (decoder-only) training: steps={steps} batch={bs} lr={lr} "
                    f"device={dev} (char Shakespeare LM; DESSIN_GPT2_* overrides)",
                    kind="training",
                )
                return train_gpt2_nano(
                    data_seed=seed,
                    training_steps=steps,
                    batch_size=bs,
                    learning_rate=lr,
                    checkpoint_weights_hex=checkpoint_weights_hex,
                )
            from .micro_gpt_trainer import train_micro_gpt

            pretty_print(
                f"Micro-GPT training: steps={steps} batch={bs} lr={lr} device="
                f"{os.environ.get('DESSIN_MICRO_GPT_DEVICE', 'cpu')} (char Shakespeare LM)",
                kind="training",
            )
            return train_micro_gpt(
                data_seed=seed,
                training_steps=steps,
                batch_size=bs,
                learning_rate=lr,
                checkpoint_weights_hex=checkpoint_weights_hex,
            )
        self._last_data_seed = data_seed
        return super().train_model(
            learning_rate=learning_rate,
            training_steps=training_steps,
            batch_size=batch_size,
            data_seed=data_seed,
        )
    
    def train_and_create_torrent(
        self,
        model_id: str,
        learning_rate: float = 0.01,
        training_steps: int = 20,
        batch_size: int = 32,
        data_seed: Optional[int] = None,
        checkpoint_weights_hex: Optional[str] = None,
        previous_loss: Optional[float] = None,
    ) -> TorrentTrainingResult:
        """Train model and create torrent for sharing.
        
        If checkpoint_weights_hex is provided, loads those weights before training
        to continue from a previous block's checkpoint.
        """
        # Load checkpoint into SimpleTrainer MLP only; LM loads inside train_decoder_only_char_lm.
        if checkpoint_weights_hex and not _use_micro_gpt_training():
            try:
                self.load_model_from_hex(checkpoint_weights_hex)
                prev_s = (
                    f"{previous_loss:.6f}"
                    if previous_loss is not None
                    else "unknown"
                )
                pretty_print(
                    f"Loaded checkpoint weights for continuous training "
                    f"(previous loss: {prev_s})",
                    kind="ok",
                )
            except Exception as e:
                pretty_print(
                    f"Failed to load checkpoint weights: {e}. Starting fresh.",
                    kind="warn",
                )
        
        # First, do the regular training (pass checkpoint weights for continuous training)
        base_result = self.train_model(
            learning_rate=learning_rate,
            training_steps=training_steps,
            batch_size=batch_size,
            data_seed=data_seed,
            checkpoint_weights_hex=checkpoint_weights_hex,
        )
        
        # Create model file for torrent sharing
        model_file_path = self._create_model_file(model_id, base_result)
        
        # Create torrent and get hash/magnet link
        torrent_hash, magnet_link = self._create_torrent(model_id, model_file_path)
        
        # Get file size
        file_size = os.path.getsize(model_file_path) if os.path.exists(model_file_path) else 0
        
        return TorrentTrainingResult(
            loss_before=base_result.loss_before,
            loss_after=base_result.loss_after,
            model_weights_hex=base_result.model_weights_hex,
            model_size_bytes=base_result.model_size_bytes,
            training_steps=base_result.training_steps,
            learning_rate=base_result.learning_rate,
            batch_size=base_result.batch_size,
            training_data_hash=base_result.training_data_hash,
            pogo_layer_shapes=base_result.pogo_layer_shapes,
            model_architecture_json=base_result.model_architecture_json,
            training_data_json=base_result.training_data_json,
            model_file_path=model_file_path,
            torrent_hash=torrent_hash,
            magnet_link=magnet_link,
            model_file_size=file_size
        )

    def publish_training_result(
        self, model_id: str, base_result: TrainingResult
    ) -> TorrentTrainingResult:
        """Build torrent artifact from an already-computed training run (e.g. after salt retries)."""
        model_file_path = self._create_model_file(model_id, base_result)
        torrent_hash, magnet_link = self._create_torrent(model_id, model_file_path)
        file_size = (
            os.path.getsize(model_file_path) if os.path.exists(model_file_path) else 0
        )
        return TorrentTrainingResult(
            loss_before=base_result.loss_before,
            loss_after=base_result.loss_after,
            model_weights_hex=base_result.model_weights_hex,
            model_size_bytes=base_result.model_size_bytes,
            training_steps=base_result.training_steps,
            learning_rate=base_result.learning_rate,
            batch_size=base_result.batch_size,
            training_data_hash=base_result.training_data_hash,
            pogo_layer_shapes=base_result.pogo_layer_shapes,
            model_architecture_json=base_result.model_architecture_json,
            training_data_json=base_result.training_data_json,
            model_file_path=model_file_path,
            torrent_hash=torrent_hash,
            magnet_link=magnet_link,
            model_file_size=file_size,
        )
    
    def _create_model_file(self, model_id: str, training_result: TrainingResult) -> str:
        """Create a model file that can be shared via torrent"""
        
        # Create unique filename based on model_id and training result
        model_hash = hashlib.sha256(training_result.model_weights_hex.encode()).hexdigest()[:16]
        timestamp = int(time.time())
        filename = f"{model_id}_{timestamp}_{model_hash}.json"
        model_file_path = self.torrent_dir / filename
        
        arch_block: Dict[str, Any]
        if training_result.model_architecture_json:
            arch_block = training_result.model_architecture_json
        else:
            arch_block = {
                "input_size": self.model.input_size,
                "hidden_size": self.model.hidden_size,
                "output_size": self.model.output_size,
                "layer_shapes": [
                    [self.model.input_size, self.model.hidden_size],
                    [1, self.model.hidden_size],
                    [self.model.hidden_size, self.model.output_size],
                    [1, self.model.output_size]
                ]
            }
        train_block: Dict[str, Any]
        if training_result.training_data_json:
            train_block = dict(training_result.training_data_json)
            train_block.setdefault("data_hash", training_result.training_data_hash)
        else:
            train_block = {
                "seed_used": getattr(self, '_last_data_seed', None),
                "samples_count": 100,  # Default from SimpleTrainer
                "features_count": 4,
                "data_hash": training_result.training_data_hash
            }

        model_data = {
            "model_id": model_id,
            "timestamp": timestamp,
            "training_metadata": {
                "loss_before": training_result.loss_before,
                "loss_after": training_result.loss_after,
                "improvement": training_result.loss_before - training_result.loss_after,
                "training_data_hash": training_result.training_data_hash,
                "model_size_bytes": training_result.model_size_bytes
            },
            "model_architecture": arch_block,
            "model_weights": {
                "format": "hex_encoded_float64",
                "weights_hex": training_result.model_weights_hex,
                "weights_shape": len(bytes.fromhex(training_result.model_weights_hex)) // 8,  # float64 bytes / 8
                "checksum": hashlib.sha256(training_result.model_weights_hex.encode()).hexdigest()
            },
            "training_data": train_block,
        }
        
        # Write model file
        with open(model_file_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        pretty_print(
            f"Created model file: {model_file_path} ({os.path.getsize(model_file_path)} bytes)",
            kind="ok",
        )
        return str(model_file_path)
    
    def _create_torrent(self, model_id: str, model_file_path: str) -> Tuple[str, str]:
        """Create torrent for model file and return hash and magnet link"""
        
        # Try real torrent creation first
        if self.real_bt_distributor:
            try:
                torrent_info = self.real_bt_distributor.create_real_torrent(
                    model_file_path, 
                    model_id
                )
                
                if torrent_info:
                    # Start seeding immediately
                    self.real_bt_distributor.start_seeding(torrent_info.torrent_hash)
                    pretty_print(f"Created real torrent: {torrent_info.torrent_hash}", kind="ok")
                    return torrent_info.torrent_hash, torrent_info.magnet_link
                    
            except Exception as e:
                pretty_print(f"Real torrent creation failed: {e}", kind="warn")
        
        # Fallback to mock torrent info
        mock_hash = hashlib.sha256(f"{model_id}_{model_file_path}".encode()).hexdigest()[:40]
        mock_magnet = f"magnet:?xt=urn:btih:{mock_hash}&dn={model_id}_model"
        pretty_print(f"Created mock torrent: {mock_hash}", kind="info")
        return mock_hash, mock_magnet
    
    def load_model_from_torrent_file(self, model_file_path: str) -> bool:
        """Load model weights from a torrent-shared model file"""
        try:
            if not os.path.exists(model_file_path):
                pretty_print(f"Model file not found: {model_file_path}", kind="bad")
                return False
            
            with open(model_file_path, 'r') as f:
                model_data = json.load(f)
            
            # Verify checksum
            weights_hex = model_data["model_weights"]["weights_hex"]
            expected_checksum = model_data["model_weights"]["checksum"]
            actual_checksum = hashlib.sha256(weights_hex.encode()).hexdigest()
            
            if actual_checksum != expected_checksum:
                pretty_print(
                    f"Model file checksum mismatch: expected {expected_checksum}, got {actual_checksum}",
                    kind="bad",
                )
                return False
            
            # Load weights into model
            self.load_model_from_hex(weights_hex)
            
            pretty_print(f"Loaded model from torrent file: {model_file_path}", kind="ok")
            pretty_print(f"Model ID: {model_data['model_id']}", kind="detail")
            pretty_print(
                f"Loss improvement: {model_data['training_metadata']['improvement']:.6f}",
                kind="detail",
            )
            pretty_print(
                f"Architecture: {model_data['model_architecture']['input_size']}x"
                f"{model_data['model_architecture']['hidden_size']}x"
                f"{model_data['model_architecture']['output_size']}",
                kind="detail",
            )
            
            return True
            
        except Exception as e:
            pretty_print(f"Error loading model from torrent file: {e}", kind="bad")
            return False
    
    def verify_model_file_integrity(self, model_file_path: str) -> bool:
        """Verify integrity of a downloaded model file"""
        try:
            with open(model_file_path, 'r') as f:
                model_data = json.load(f)
            
            # Check required fields
            required_fields = [
                "model_id", "training_metadata", "model_architecture", 
                "model_weights", "training_data"
            ]
            
            for field in required_fields:
                if field not in model_data:
                    pretty_print(f"Missing required field: {field}", kind="bad")
                    return False
            
            # Verify weights checksum
            weights_hex = model_data["model_weights"]["weights_hex"]
            expected_checksum = model_data["model_weights"]["checksum"]
            actual_checksum = hashlib.sha256(weights_hex.encode()).hexdigest()
            
            if actual_checksum != expected_checksum:
                pretty_print("Weights checksum verification failed", kind="bad")
                return False
            
            # Verify training improvement
            improvement = model_data["training_metadata"]["improvement"]
            if improvement <= 0:
                pretty_print(f"No training improvement: {improvement}", kind="warn")
                return False
            
            pretty_print(f"Model file integrity verified: {model_file_path}", kind="ok")
            return True
            
        except Exception as e:
            pretty_print(f"Error verifying model file: {e}", kind="bad")
            return False
    
    def get_model_file_info(self, model_file_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a model file"""
        try:
            if not os.path.exists(model_file_path):
                return None
                
            with open(model_file_path, 'r') as f:
                model_data = json.load(f)
            
            return {
                "model_id": model_data["model_id"],
                "timestamp": model_data["timestamp"],
                "loss_before": model_data["training_metadata"]["loss_before"],
                "loss_after": model_data["training_metadata"]["loss_after"],
                "improvement": model_data["training_metadata"]["improvement"],
                "model_size_bytes": model_data["training_metadata"]["model_size_bytes"],
                "file_size": os.path.getsize(model_file_path),
                "architecture": f"{model_data['model_architecture']['input_size']}x{model_data['model_architecture']['hidden_size']}x{model_data['model_architecture']['output_size']}"
            }
            
        except Exception as e:
            pretty_print(f"Error getting model file info: {e}", kind="bad")
            return None


def demo_torrent_training():
    """Demonstrate torrent-based model training"""
    pretty_print("Torrent model training demo", kind="training")
    pretty_print("=" * 50, kind="detail")

    # Create trainer
    trainer = TorrentModelTrainer("e2e/data/demo_torrents")
    
    # Train and create torrent
    result = trainer.train_and_create_torrent(
        model_id="demo_model",
        training_steps=15,
        learning_rate=0.01
    )
    
    pretty_print("Training completed:", kind="ok")
    pretty_print(f"Loss: {result.loss_before:.6f} -> {result.loss_after:.6f}", kind="detail")
    pretty_print(f"Improvement: {result.loss_before - result.loss_after:.6f}", kind="detail")
    pretty_print(f"Model file: {result.model_file_path}", kind="detail")
    pretty_print(f"File size: {result.model_file_size} bytes", kind="detail")
    pretty_print(f"Torrent hash: {result.torrent_hash}", kind="detail")
    pretty_print(f"Magnet link: {result.magnet_link[:80]}...", kind="detail")

    # Verify file integrity
    if trainer.verify_model_file_integrity(result.model_file_path):
        pretty_print("Model file integrity verified", kind="ok")

    # Get file info
    info = trainer.get_model_file_info(result.model_file_path)
    if info:
        pretty_print(
            f"Model info: {info['architecture']}, improvement: {info['improvement']:.6f}",
            kind="ok",
        )

    pretty_print("Torrent training demo completed.", kind="ok")


if __name__ == "__main__":
    demo_torrent_training()
