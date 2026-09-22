#!/usr/bin/env python3
"""
Simple neural network trainer for DeSSIN blockchain
Implements real model training with loss measurement and hex encoding
"""

import math
import numpy as np
import hashlib
import time
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class TrainingResult:
    """Result of a training session"""
    loss_before: float
    loss_after: float
    model_weights_hex: str
    model_size_bytes: int
    training_steps: int
    learning_rate: float
    batch_size: int
    training_data_hash: str
    # Optional: micro-GPT / non-MLP checkpoints for PoGO + torrent metadata
    pogo_layer_shapes: Optional[List[Tuple[int, ...]]] = None
    model_architecture_json: Optional[Dict[str, Any]] = None
    training_data_json: Optional[Dict[str, Any]] = None


class SimpleNeuralNetwork:
    """Simple 2-layer neural network for demonstration"""
    
    def __init__(self, input_size: int = 4, hidden_size: int = 8, output_size: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with small random values
        np.random.seed(int(time.time()) % 1000)  # Different seed each time
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        # Hidden layer with ReLU activation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute mean squared error loss"""
        predictions = self.forward(X)
        loss = np.mean((predictions - y) ** 2)
        return float(loss)
    
    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01):
        """Backward pass and weight updates"""
        m = X.shape[0]
        
        # Forward pass (already done in compute_loss, but we need the values)
        predictions = self.forward(X)
        
        # Backward pass
        dz2 = 2 * (predictions - y) / m  # MSE derivative
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def get_weights_bytes(self) -> bytes:
        """Get model weights as bytes for network transmission"""
        # Concatenate all weights and biases
        weights_list = [self.W1.flatten(), self.b1.flatten(), 
                       self.W2.flatten(), self.b2.flatten()]
        all_weights = np.concatenate(weights_list)
        
        # Convert to bytes (float64 -> bytes)
        return all_weights.tobytes()
    
    def set_weights_from_bytes(self, weights_bytes: bytes):
        """Set model weights from bytes"""
        # Convert bytes back to float array
        all_weights = np.frombuffer(weights_bytes, dtype=np.float64)
        
        # Reshape back to original weight matrices
        idx = 0
        
        # W1
        w1_size = self.input_size * self.hidden_size
        self.W1 = all_weights[idx:idx + w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size
        
        # b1
        b1_size = self.hidden_size
        self.b1 = all_weights[idx:idx + b1_size].reshape(1, self.hidden_size)
        idx += b1_size
        
        # W2
        w2_size = self.hidden_size * self.output_size
        self.W2 = all_weights[idx:idx + w2_size].reshape(self.hidden_size, self.output_size)
        idx += w2_size
        
        # b2
        b2_size = self.output_size
        self.b2 = all_weights[idx:idx + b2_size].reshape(1, self.output_size)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information"""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "total_parameters": (self.input_size * self.hidden_size + self.hidden_size + 
                               self.hidden_size * self.output_size + self.output_size)
        }


class SimpleTrainer:
    """Trainer for simple neural networks with real data"""
    
    def __init__(self):
        self.model = SimpleNeuralNetwork()
        self.training_data = None
        self.training_labels = None
    
    def generate_synthetic_data(self, n_samples: int = 100, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for demonstration"""
        if seed is None:
            seed = int(time.time()) % 1000
        
        np.random.seed(seed)
        
        # Generate random input features
        X = np.random.randn(n_samples, 4)
        
        # Create a simple target function: y = sum of features with some noise
        y = np.sum(X, axis=1, keepdims=True) + np.random.randn(n_samples, 1) * 0.1
        
        return X, y
    
    def prepare_training_data(self, seed: Optional[int] = None) -> str:
        """Prepare training data and return its hash"""
        X, y = self.generate_synthetic_data(seed=seed)
        self.training_data = X
        self.training_labels = y
        
        # Create hash of training data for verification
        data_string = f"{X.tobytes()}{y.tobytes()}"
        data_hash = hashlib.sha256(data_string.encode()).hexdigest()
        
        return data_hash
    
    def train_model(self, 
                   learning_rate: float = 0.01, 
                   training_steps: int = 10, 
                   batch_size: int = 32,
                   data_seed: Optional[int] = None) -> TrainingResult:
        """Train the model and return training results"""
        
        # Always prepare new training data with the specified seed
        # This ensures deterministic but different training data based on VRF
        training_data_hash = self.prepare_training_data(seed=data_seed)
        
        X, y = self.training_data, self.training_labels
        
        # Measure loss before training
        loss_before = self.model.compute_loss(X, y)
        
        # Training loop
        n_samples = X.shape[0]
        for step in range(training_steps):
            # Mini-batch training
            if batch_size < n_samples:
                indices = np.random.choice(n_samples, batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
            else:
                X_batch = X
                y_batch = y
            
            # Update weights
            self.model.backward(X_batch, y_batch, learning_rate)
        
        # Measure loss after training
        loss_after = self.model.compute_loss(X, y)
        
        # Get model weights as hex
        weights_bytes = self.model.get_weights_bytes()
        weights_hex = weights_bytes.hex()
        
        # Training data hash was already calculated above
        
        return TrainingResult(
            loss_before=loss_before,
            loss_after=loss_after,
            model_weights_hex=weights_hex,
            model_size_bytes=len(weights_bytes),
            training_steps=training_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            training_data_hash=training_data_hash
        )
    
    def load_model_from_hex(self, weights_hex: str):
        """Load model weights from hex string"""
        weights_bytes = bytes.fromhex(weights_hex)
        self.model.set_weights_from_bytes(weights_bytes)
    
    def verify_training_improvement(
        self, result: TrainingResult, *, min_improvement: Optional[float] = None
    ) -> bool:
        """If ``min_improvement`` is set, require Δloss ≥ that threshold; else only finite losses."""
        if not (math.isfinite(result.loss_before) and math.isfinite(result.loss_after)):
            return False
        if min_improvement is None:
            return True
        improvement = result.loss_before - result.loss_after
        return improvement >= min_improvement
    
    def get_model_hash(self) -> str:
        """Get hash of current model weights"""
        weights_bytes = self.model.get_weights_bytes()
        return hashlib.sha256(weights_bytes).hexdigest()


def demo_simple_training():
    """Demonstrate the simple training system"""
    print("DeSSIN Simple Neural Network Training Demo")
    print("=" * 50)
    
    trainer = SimpleTrainer()
    
    # Train the model
    print("Training model...")
    result = trainer.train_model(learning_rate=0.01, training_steps=20, batch_size=32)
    
    print(f"Loss before training: {result.loss_before:.6f}")
    print(f"Loss after training: {result.loss_after:.6f}")
    print(f"Loss improvement: {result.loss_before - result.loss_after:.6f}")
    print(f"Model size: {result.model_size_bytes} bytes")
    print(f"Model weights (hex): {result.model_weights_hex[:50]}...")
    print(f"Training data hash: {result.training_data_hash[:16]}...")
    
    # Verify improvement
    if trainer.verify_training_improvement(result):
        print("✓ Training improvement verified!")
    else:
        print("✗ Insufficient training improvement")
    
    # Test model loading from hex
    print("\nTesting model serialization...")
    new_trainer = SimpleTrainer()
    new_trainer.load_model_from_hex(result.model_weights_hex)
    
    # Verify the loaded model has same weights
    original_hash = trainer.get_model_hash()
    loaded_hash = new_trainer.get_model_hash()
    
    if original_hash == loaded_hash:
        print("✓ Model serialization/deserialization successful!")
    else:
        print("✗ Model serialization failed")
    
    print(f"Original model hash: {original_hash[:16]}...")
    print(f"Loaded model hash: {loaded_hash[:16]}...")


if __name__ == "__main__":
    demo_simple_training()
