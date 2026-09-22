#!/usr/bin/env python3
"""
Unit tests for PoGO protocol model quantization
"""

import unittest
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dessin.protocol.pogo_protocol import ModelQuantizer, QuantizationLevel


class TestModelQuantization(unittest.TestCase):
    """Test model quantization functionality"""

    def setUp(self):
        self.quantizer = ModelQuantizer()
        # Create test weights with known range
        np.random.seed(42)  # For reproducible tests
        self.test_weights = np.random.randn(100).astype(np.float64)

    def test_float32_quantization(self):
        """Test FLOAT32 quantization (should be no-op)"""
        quantized, error = self.quantizer.quantize_weights(self.test_weights, QuantizationLevel.FLOAT32)
        
        # Should be very close (float64 -> float32 conversion might introduce tiny errors)
        np.testing.assert_allclose(quantized, self.test_weights, rtol=1e-6)
        self.assertLess(error, 1e-10)  # Very small error acceptable

    def test_float16_quantization(self):
        """Test FLOAT16 quantization"""
        quantized, error = self.quantizer.quantize_weights(self.test_weights, QuantizationLevel.FLOAT16)
        
        # Should have some quantization error but small
        self.assertGreater(error, 0.0)
        self.assertLess(error, 0.01)  # Should be small error
        
        # Shape should be preserved
        self.assertEqual(quantized.shape, self.test_weights.shape)
        
        # Values should be close
        np.testing.assert_allclose(quantized, self.test_weights, rtol=1e-3)

    def test_int8_quantization(self):
        """Test INT8 quantization"""
        quantized, error = self.quantizer.quantize_weights(self.test_weights, QuantizationLevel.INT8)
        
        # Should have quantization error
        self.assertGreater(error, 0.0)
        
        # Shape should be preserved
        self.assertEqual(quantized.shape, self.test_weights.shape)
        
        # Check that values are within reasonable range after dequantization
        self.assertLess(np.max(np.abs(quantized - self.test_weights)), 1.0)

    def test_int4_quantization(self):
        """Test INT4 quantization"""
        quantized, error = self.quantizer.quantize_weights(self.test_weights, QuantizationLevel.INT4)
        
        # Should have higher quantization error than INT8
        self.assertGreater(error, 0.0)
        
        # Shape should be preserved
        self.assertEqual(quantized.shape, self.test_weights.shape)

    def test_quantization_error_ordering(self):
        """Test that quantization error increases with lower precision"""
        errors = {}
        
        for level in [QuantizationLevel.FLOAT32, QuantizationLevel.FLOAT16, 
                     QuantizationLevel.INT8, QuantizationLevel.INT4]:
            _, error = self.quantizer.quantize_weights(self.test_weights, level)
            errors[level] = error
        
        # Error should increase with lower precision
        self.assertLess(errors[QuantizationLevel.FLOAT32], 1e-10)  # Very small
        self.assertLessEqual(errors[QuantizationLevel.FLOAT16], errors[QuantizationLevel.INT8])
        self.assertLessEqual(errors[QuantizationLevel.INT8], errors[QuantizationLevel.INT4])

    def test_zero_weights(self):
        """Test quantization with zero weights"""
        zero_weights = np.zeros(10, dtype=np.float64)
        
        for level in QuantizationLevel:
            quantized, error = self.quantizer.quantize_weights(zero_weights, level)
            
            # Should remain zero
            np.testing.assert_array_equal(quantized, zero_weights)
            self.assertEqual(error, 0.0)

    def test_small_weights(self):
        """Test quantization with very small weights"""
        small_weights = np.array([1e-10, -1e-10, 1e-8, -1e-8], dtype=np.float64)
        
        for level in QuantizationLevel:
            quantized, error = self.quantizer.quantize_weights(small_weights, level)
            
            # Should handle small weights gracefully
            self.assertFalse(np.any(np.isnan(quantized)))
            self.assertFalse(np.any(np.isinf(quantized)))
            self.assertGreaterEqual(error, 0.0)

    def test_large_weights(self):
        """Test quantization with large weights"""
        large_weights = np.array([1000.0, -1000.0, 500.0, -500.0], dtype=np.float64)
        
        for level in QuantizationLevel:
            quantized, error = self.quantizer.quantize_weights(large_weights, level)
            
            # Should handle large weights gracefully
            self.assertFalse(np.any(np.isnan(quantized)))
            self.assertFalse(np.any(np.isinf(quantized)))
            self.assertGreaterEqual(error, 0.0)

    def test_single_weight(self):
        """Test quantization with single weight"""
        single_weight = np.array([0.5], dtype=np.float64)
        
        for level in QuantizationLevel:
            quantized, error = self.quantizer.quantize_weights(single_weight, level)
            
            self.assertEqual(len(quantized), 1)
            self.assertGreaterEqual(error, 0.0)

    def test_get_quantized_size(self):
        """Test quantized size calculation"""
        original_size = 1000  # bytes
        
        # FLOAT32 should be same size
        size_f32 = ModelQuantizer.get_quantized_size(original_size, QuantizationLevel.FLOAT32)
        self.assertEqual(size_f32, original_size)
        
        # FLOAT16 should be half size
        size_f16 = ModelQuantizer.get_quantized_size(original_size, QuantizationLevel.FLOAT16)
        self.assertEqual(size_f16, original_size // 2)
        
        # INT8 should be quarter size (from FLOAT32)
        size_i8 = ModelQuantizer.get_quantized_size(original_size, QuantizationLevel.INT8)
        self.assertEqual(size_i8, original_size // 4)
        
        # INT4 should be eighth size
        size_i4 = ModelQuantizer.get_quantized_size(original_size, QuantizationLevel.INT4)
        self.assertEqual(size_i4, original_size // 8)

    def test_empty_weights(self):
        """Test quantization with empty weights array"""
        empty_weights = np.array([], dtype=np.float64)
        
        for level in QuantizationLevel:
            quantized, error = self.quantizer.quantize_weights(empty_weights, level)
            self.assertEqual(len(quantized), 0)
            # Empty arrays should return 0.0 error
            self.assertEqual(error, 0.0)

    def test_deterministic_quantization(self):
        """Test that quantization is deterministic"""
        for level in QuantizationLevel:
            q1, e1 = self.quantizer.quantize_weights(self.test_weights, level)
            q2, e2 = self.quantizer.quantize_weights(self.test_weights, level)
            
            np.testing.assert_array_equal(q1, q2)
            self.assertEqual(e1, e2)

    def test_quantization_preserves_shape(self):
        """Test that quantization preserves tensor shape"""
        shapes = [(10,), (5, 4), (2, 3, 4), (2, 2, 2, 2)]
        
        for shape in shapes:
            weights = np.random.randn(*shape).astype(np.float64)
            
            for level in QuantizationLevel:
                quantized, _ = self.quantizer.quantize_weights(weights, level)
                self.assertEqual(quantized.shape, weights.shape)


if __name__ == "__main__":
    unittest.main()
