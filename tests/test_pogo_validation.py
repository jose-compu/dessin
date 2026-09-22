#!/usr/bin/env python3
"""
Unit tests for PoGO protocol validation methods
"""

import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dessin.protocol.pogo_protocol import PoGOProtocol, QuantizationLevel
from dessin.llm.simple_trainer import SimpleTrainer


class TestPoGOValidation(unittest.TestCase):
    """Test PoGO protocol validation methods"""

    def setUp(self):
        self.trainer = SimpleTrainer()
        self.protocol = PoGOProtocol(self.trainer.model)

    def test_validate_training_improvement_valid(self):
        """Test training improvement validation with valid improvement"""
        loss_before = 5.0
        loss_after = 4.0
        min_improvement = 0.5
        
        is_valid = self.protocol.validate_training_improvement(
            loss_before, loss_after, min_improvement
        )
        self.assertTrue(is_valid)

    def test_validate_training_improvement_insufficient(self):
        """Test training improvement validation with insufficient improvement"""
        loss_before = 5.0
        loss_after = 4.9
        min_improvement = 0.5
        
        is_valid = self.protocol.validate_training_improvement(
            loss_before, loss_after, min_improvement
        )
        self.assertFalse(is_valid)

    def test_validate_training_improvement_no_improvement(self):
        """Test training improvement validation with no improvement"""
        loss_before = 5.0
        loss_after = 5.0
        
        is_valid = self.protocol.validate_training_improvement(loss_before, loss_after)
        self.assertFalse(is_valid)

    def test_validate_training_improvement_regression(self):
        """Test training improvement validation with regression"""
        loss_before = 4.0
        loss_after = 5.0  # Worse performance
        
        is_valid = self.protocol.validate_training_improvement(loss_before, loss_after)
        self.assertFalse(is_valid)

    def test_validate_training_improvement_default_threshold(self):
        """Test training improvement validation with default threshold"""
        loss_before = 1.0
        loss_after = 0.9999  # Very small improvement (0.0001)
        
        # Should pass with default threshold (0.00001) since 0.0001 > 0.00001
        is_valid = self.protocol.validate_training_improvement(loss_before, loss_after)
        self.assertTrue(is_valid)
        
        # Should fail with even smaller improvement
        loss_after_tiny = 0.999999  # Tiny improvement (0.000001)
        is_valid = self.protocol.validate_training_improvement(loss_before, loss_after_tiny)
        self.assertFalse(is_valid)

    def test_validate_quantization_tolerance_valid(self):
        """Test quantization tolerance validation with acceptable degradation"""
        original_loss = 1.0
        quantized_loss = 1.0005  # Small degradation
        tolerance = 0.001
        
        is_valid = self.protocol.validate_quantization_tolerance(
            original_loss, quantized_loss, tolerance
        )
        self.assertTrue(is_valid)

    def test_validate_quantization_tolerance_excessive(self):
        """Test quantization tolerance validation with excessive degradation"""
        original_loss = 1.0
        quantized_loss = 1.1  # Large degradation
        tolerance = 0.001
        
        is_valid = self.protocol.validate_quantization_tolerance(
            original_loss, quantized_loss, tolerance
        )
        self.assertFalse(is_valid)

    def test_validate_quantization_tolerance_improvement(self):
        """Test quantization tolerance validation when quantization improves performance"""
        original_loss = 1.0
        quantized_loss = 0.9  # Improvement (unusual but possible)
        tolerance = 0.001
        
        # The method checks abs(degradation), so improvement (0.1) > tolerance (0.001) = False
        is_valid = self.protocol.validate_quantization_tolerance(
            original_loss, quantized_loss, tolerance
        )
        # This will fail because abs(0.9 - 1.0) = 0.1 > 0.001
        self.assertFalse(is_valid)
        
        # But with larger tolerance, it should pass
        is_valid = self.protocol.validate_quantization_tolerance(
            original_loss, quantized_loss, tolerance=0.2
        )
        self.assertTrue(is_valid)

    def test_validate_quantization_tolerance_zero_tolerance(self):
        """Test quantization tolerance validation with zero tolerance"""
        original_loss = 1.0
        quantized_loss = 1.0001
        tolerance = 0.0
        
        is_valid = self.protocol.validate_quantization_tolerance(
            original_loss, quantized_loss, tolerance
        )
        self.assertFalse(is_valid)
        
        # Should only pass if no degradation
        is_valid = self.protocol.validate_quantization_tolerance(
            original_loss, original_loss, tolerance
        )
        self.assertTrue(is_valid)

    def test_validate_with_real_training_data(self):
        """Test validation with real training results"""
        # Train a model
        result = self.trainer.train_model(
            training_steps=20, 
            learning_rate=0.01,
            data_seed=42
        )
        
        # Should validate training improvement
        is_valid = self.protocol.validate_training_improvement(
            result.loss_before, result.loss_after
        )
        self.assertTrue(is_valid, f"Training improvement validation failed: {result.loss_before} -> {result.loss_after}")
        
        # Test quantization tolerance with reasonable tolerance
        is_valid = self.protocol.validate_quantization_tolerance(
            result.loss_after, result.loss_after + 0.0005, tolerance=0.001
        )
        self.assertTrue(is_valid)

    def test_validate_edge_cases(self):
        """Test validation with edge case values"""
        # Test with very small losses
        is_valid = self.protocol.validate_training_improvement(0.001, 0.0005)
        self.assertTrue(is_valid)
        
        # Test with zero losses
        is_valid = self.protocol.validate_training_improvement(0.0, 0.0)
        self.assertFalse(is_valid)
        
        # Test with negative losses (shouldn't happen but test robustness)
        is_valid = self.protocol.validate_training_improvement(-1.0, -2.0)
        self.assertTrue(is_valid)  # Improvement from -1 to -2

    def test_validate_with_nan_values(self):
        """Test validation with NaN values"""
        import math
        
        # Training improvement with NaN should fail
        is_valid = self.protocol.validate_training_improvement(math.nan, 1.0)
        self.assertFalse(is_valid)
        
        is_valid = self.protocol.validate_training_improvement(1.0, math.nan)
        self.assertFalse(is_valid)
        
        # Quantization tolerance with NaN should fail
        is_valid = self.protocol.validate_quantization_tolerance(math.nan, 1.0)
        self.assertFalse(is_valid)
        
        is_valid = self.protocol.validate_quantization_tolerance(1.0, math.nan)
        self.assertFalse(is_valid)

    def test_validate_with_infinite_values(self):
        """Test validation with infinite values"""
        import math
        
        # The current implementation doesn't explicitly handle infinity
        # It just does arithmetic, so inf - 1.0 = inf, which is >= min_improvement
        is_valid = self.protocol.validate_training_improvement(math.inf, 1.0)
        self.assertTrue(is_valid)  # inf - 1.0 = inf >= 0.0001
        
        is_valid = self.protocol.validate_training_improvement(1.0, math.inf)
        self.assertFalse(is_valid)  # 1.0 - inf = -inf < 0.0001
        
        # For quantization tolerance, abs(inf - 1.0) = inf > tolerance
        is_valid = self.protocol.validate_quantization_tolerance(math.inf, 1.0)
        self.assertFalse(is_valid)
        
        is_valid = self.protocol.validate_quantization_tolerance(1.0, math.inf)
        self.assertFalse(is_valid)

    def test_validate_negative_thresholds(self):
        """Test validation with negative thresholds"""
        # Negative minimum improvement should still work
        is_valid = self.protocol.validate_training_improvement(
            1.0, 0.5, min_improvement=-0.1
        )
        self.assertTrue(is_valid)
        
        # Negative tolerance should still work
        is_valid = self.protocol.validate_quantization_tolerance(
            1.0, 1.1, tolerance=-0.1
        )
        self.assertFalse(is_valid)  # Still fails because degradation > tolerance


if __name__ == "__main__":
    unittest.main()
