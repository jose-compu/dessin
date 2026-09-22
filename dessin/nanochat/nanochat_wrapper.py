"""
Nanochat Wrapper for DeSSIN
============================

Wrapper around the real nanochat implementation for blockchain integration.
Uses actual nanochat code from https://github.com/karpathy/nanochat

Pinned upstream revision (kept inline for reproducibility):
- Repository: https://github.com/karpathy/nanochat
- Commit: 90442de35f860226ccec6d64ab1829bdc1fad55a
- Date: 2024-12-08
- Branch: master
- Commit message: fix bug where any rank has to be able to create
  checkpoint_dir if saving optim
"""

import logging
import os
import sys
import subprocess
import tempfile
from typing import Dict, Optional, Any, List
from pathlib import Path
import torch

_log = logging.getLogger(__name__)

# Upstream nanochat pin metadata.
NANOCHAT_REPOSITORY = "https://github.com/karpathy/nanochat"
NANOCHAT_COMMIT = "90442de35f860226ccec6d64ab1829bdc1fad55a"
NANOCHAT_BRANCH = "master"
NANOCHAT_COMMIT_DATE = "2024-12-08"
NANOCHAT_COMMIT_MESSAGE = (
    "fix bug where any rank has to be able to create checkpoint_dir if saving optim"
)

# Optional checkout: repo root external/nanochat (see scripts/setup_nanochat.sh), or legacy dessin/external/nanochat.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DESSIN_PKG_ROOT = Path(__file__).resolve().parents[1]
_NANOCHAT_CANDIDATES = (
    _REPO_ROOT / "external" / "nanochat",
    _DESSIN_PKG_ROOT / "external" / "nanochat",
)
NANOCHAT_PATH: Optional[Path] = next((p for p in _NANOCHAT_CANDIDATES if p.is_dir()), None)
NANOCHAT_EXPECTED = _REPO_ROOT / "external" / "nanochat"
if NANOCHAT_PATH is not None:
    sys.path.insert(0, str(NANOCHAT_PATH))
    NANOCHAT_AVAILABLE = True
else:
    NANOCHAT_AVAILABLE = False
    _log.debug(
        "nanochat optional checkout not found (simulation mode). Expected: %s — "
        "run from repo root: bash scripts/setup_nanochat.sh",
        NANOCHAT_EXPECTED,
    )


class NanochatWrapper:
    """Wrapper around real nanochat for DeSSIN integration"""
    
    def __init__(self, model_cache_dir: str = "model_cache"):
        """
        Initialize nanochat wrapper
        
        Args:
            model_cache_dir: Directory for model storage
        """
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.nanochat_available = NANOCHAT_AVAILABLE
        
        if self.nanochat_available:
            try:
                # Import nanochat modules
                from nanochat import gpt, engine, tokenizer, dataloader
                self.gpt = gpt
                self.engine = engine
                self.tokenizer = tokenizer
                self.dataloader = dataloader
                _log.debug("nanochat modules loaded from %s", NANOCHAT_PATH)
            except (ImportError, TypeError, SyntaxError) as e:
                _log.warning(
                    "failed to load nanochat from checkout (wrong Python version or broken tree): %s",
                    e,
                )
                self.nanochat_available = False
        else:
            _log.debug("nanochat simulation mode (no upstream checkout)")
    
    def create_model_config(
        self,
        depth: int = 20,
        vocab_size: int = 50257,
        context_length: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a nanochat model configuration
        
        Args:
            depth: Number of transformer layers
            vocab_size: Vocabulary size
            context_length: Context length (sequence length)
            **kwargs: Additional config options
        
        Returns:
            Model configuration dict
        """
        # Calculate heads and embedding size  - must satisfy n_embd % n_head == 0
        n_head = 6  # Use fixed number of heads for simplicity
        head_dim = 128  # Fixed head dimension
        n_embd = n_head * head_dim  # This guarantees divisibility: 6 * 128 = 768
        
        # Base config based on nanochat defaults
        config = {
            "depth": depth,
            "n_layer": depth,
            "n_head": n_head,
            "n_kv_head": n_head,  # GQA: use same as n_head for simplicity
            "n_embd": n_embd,
            "vocab_size": vocab_size,
            "sequence_len": context_length,  # nanochat uses 'sequence_len' not 'block_size'
            "dropout": 0.0,
            "bias": False,
        }
        
        # Update with custom kwargs
        config.update(kwargs)
        
        return config
    
    def initialize_model(
        self,
        config: Dict[str, Any],
        device: str = "cuda"
    ) -> Optional[Any]:
        """
        Initialize a nanochat model
        
        Args:
            config: Model configuration
            device: Device to use (cuda/cpu)
        
        Returns:
            Initialized model or None
        """
        if not self.nanochat_available:
            return None
        
        try:
            # Create proper GPTConfig dataclass
            model_config = self.gpt.GPTConfig(
                sequence_len=config.get('sequence_len', 1024),
                vocab_size=config.get('vocab_size', 50304),
                n_layer=config.get('n_layer', 12),
                n_head=config.get('n_head', 6),
                n_kv_head=config.get('n_kv_head', 6),
                n_embd=config.get('n_embd', 768)
            )
            
            # Create model using nanochat
            model = self.gpt.GPT(model_config)
            model = model.to(device)
            
            return model
            
        except Exception as e:
            print(f"Error initializing nanochat model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_model(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        dataset_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Train a nanochat model using REAL torch training
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            dataset_path: Path to training data
            checkpoint_path: Path to save checkpoints
        
        Returns:
            Training metrics dict
        """
        if not self.nanochat_available:
            # Fallback to simulation
            return self._simulate_training(model_config, training_config)
        
        try:
            # REAL TRAINING IMPLEMENTATION
            depth = model_config.get("depth", 20)
            device_batch_size = training_config.get("device_batch_size", 4)
            max_steps = training_config.get("max_steps", 1)
            learning_rate = training_config.get("learning_rate", 3e-4)
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Initialize model if not provided (first iteration)
            if model is None:
                model = self.initialize_model(model_config, device=device)
                if model is None:
                    return self._simulate_training(model_config, training_config)
            
            # Create optimizer if not provided (first iteration)
            if optimizer is None:
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=0.1
                )
            
            # Generate synthetic training data (random token sequences)
            batch_size = device_batch_size
            seq_len = model_config.get("sequence_len") or model_config.get("block_size", 128)
            vocab_size = model_config.get("vocab_size", 1024)
            
            # Record loss before
            model.eval()
            with torch.no_grad():
                x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                loss_before = model(x, y)  # nanochat GPT returns loss directly
            
            # Training step(s)
            model.train()
            losses = []
            for step in range(max_steps):
                # Generate random batch
                x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                
                # Forward pass
                loss = model(x, y)  # nanochat GPT returns loss directly
                losses.append(loss.item())
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            loss_after = losses[-1] if losses else loss_before.item()
            
            return {
                "loss_before": loss_before.item(),
                "loss_after": loss_after,
                "loss_improvement": loss_before.item() - loss_after,
                "steps": max_steps,
                "mode": "real",
                "model": model,  # Return model for persistence
                "optimizer": optimizer  # Return optimizer for persistence
            }
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return self._simulate_training(model_config, training_config)
    
    def _simulate_training(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate training (temporary fallback)
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
        
        Returns:
            Simulated training metrics
        """
        import random
        
        # Simulate loss improvement
        loss_before = random.uniform(8.0, 12.0)
        loss_after = loss_before - random.uniform(0.5, 1.5)
        
        return {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "loss_improvement": loss_before - loss_after,
            "steps": training_config.get("max_steps", 1000),
            "mode": "simulation"
        }
    
    def load_tokenizer(self) -> Optional[Any]:
        """
        Load nanochat tokenizer
        
        Returns:
            Tokenizer instance or None
        """
        if not self.nanochat_available:
            return None
        
        try:
            # Load nanochat's BPE tokenizer
            tok = self.tokenizer.Tokenizer()
            return tok
            
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            return None
    
    def inference(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        device: str = "cuda"
    ) -> str:
        """
        Run inference using nanochat engine
        
        Args:
            model: Nanochat model
            tokenizer: Nanochat tokenizer
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            device: Device to use
        
        Returns:
            Generated text
        """
        if not self.nanochat_available or model is None:
            # Fallback to simulation
            return f"[Simulated response to: {prompt[:50]}...]"
        
        try:
            # Use nanochat's inference engine
            # This would use the actual engine with KV cache
            # For now, basic implementation
            
            # Encode prompt
            tokens = tokenizer.encode(prompt, bos=True, eos=False)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            # Generate
            model.eval()
            with torch.no_grad():
                for _ in range(max_tokens):
                    logits = model(tokens)
                    logits = logits[:, -1, :] / temperature
                    
                    # Sample
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    tokens = torch.cat([tokens, next_token], dim=1)
                    
                    # Check for EOS
                    if next_token.item() == tokenizer.eos_id:
                        break
            
            # Decode
            response = tokenizer.decode(tokens[0].tolist())
            return response
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return f"[Error: {str(e)}]"
    
    def get_nanochat_info(self) -> Dict[str, Any]:
        """Get information about nanochat integration"""
        return {
            "available": self.nanochat_available,
            "path": str(NANOCHAT_PATH) if NANOCHAT_PATH is not None else None,
            "repository": NANOCHAT_REPOSITORY,
            "branch": NANOCHAT_BRANCH,
            "commit_date": NANOCHAT_COMMIT_DATE,
            "commit_message": NANOCHAT_COMMIT_MESSAGE,
            "commit": NANOCHAT_COMMIT,
            "modules": {
                "gpt": hasattr(self, 'gpt'),
                "engine": hasattr(self, 'engine'),
                "tokenizer": hasattr(self, 'tokenizer'),
                "dataloader": hasattr(self, 'dataloader'),
            }
        }


# Singleton instance
_wrapper_instance = None

def get_nanochat_wrapper(model_cache_dir: str = "model_cache") -> NanochatWrapper:
    """Get singleton nanochat wrapper instance"""
    global _wrapper_instance
    if _wrapper_instance is None:
        _wrapper_instance = NanochatWrapper(model_cache_dir)
    return _wrapper_instance
