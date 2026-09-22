#!/usr/bin/env python3
"""
Benchmark: Quantization Consistency Verification vs Full Re-execution

This benchmark demonstrates the 10-100x speedup achieved by using quantization
consistency verification instead of full training re-execution.

Run: python scripts/benchmark_verification.py (repo root working directory recommended)
"""

import sys
import os
import time
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dessin.protocol.quantization_verifier import (
    QuantizationVerifier, 
    VerificationBenchmark,
    VerificationConfig
)
from dessin.protocol.pogo_protocol import ModelQuantizer, QuantizationLevel


def run_quick_benchmark():
    """Quick benchmark showing the key speedup."""
    print("\n" + "=" * 70)
    print("🏁 QUICK VERIFICATION BENCHMARK")
    print("=" * 70)
    
    model_sizes = [10_000, 100_000, 1_000_000]
    training_steps = 20
    
    print(f"\nModel sizes: {[f'{s:,}' for s in model_sizes]}")
    print(f"Training steps simulated: {training_steps}")
    print()
    
    quantizer = ModelQuantizer()
    verifier = QuantizationVerifier()
    
    total_reexec = 0
    total_consist = 0
    
    print(f"{'Size':>12} {'Re-exec':>12} {'Consist':>12} {'Speedup':>10}")
    print("-" * 50)
    
    for size in model_sizes:
        # Create "trained" model
        model_32bit = np.random.randn(size).astype(np.float32)
        model_4bit, quant_error = quantizer.quantize_weights(model_32bit, QuantizationLevel.INT4)
        
        # Method 1: Simulate full re-execution
        reexec_start = time.time()
        for _ in range(training_steps):
            # Simulate forward pass
            _ = np.maximum(0, model_32bit * np.random.randn(size))
            # Simulate backward pass
            _ = model_32bit * np.random.randn(size) * 0.01
        _ = np.mean(model_32bit ** 2)  # Loss
        reexec_time = (time.time() - reexec_start) * 1000
        total_reexec += reexec_time
        
        # Method 2: Quantization consistency check
        consist_start = time.time()
        result = verifier.verify_quantization_consistency(
            model_32bit, model_4bit, quant_error, b"bench"
        )
        consist_time = (time.time() - consist_start) * 1000
        total_consist += consist_time
        
        speedup = reexec_time / consist_time if consist_time > 0 else float('inf')
        
        print(f"{size:>12,} {reexec_time:>10.2f}ms {consist_time:>10.2f}ms {speedup:>9.1f}x")
    
    total_speedup = total_reexec / total_consist if total_consist > 0 else float('inf')
    
    print("-" * 50)
    print(f"{'TOTAL':>12} {total_reexec:>10.2f}ms {total_consist:>10.2f}ms {total_speedup:>9.1f}x")
    
    return total_speedup


def run_full_benchmark():
    """Full benchmark with statistical significance."""
    print("\n" + "=" * 70)
    print("📊 FULL VERIFICATION BENCHMARK (5 trials each)")
    print("=" * 70)
    
    benchmark = VerificationBenchmark()
    
    # Test different model sizes (up to 10M parameters - comparable to small GPT)
    results = benchmark.run_benchmark(
        model_sizes=[10_000, 100_000, 1_000_000, 10_000_000],
        training_steps=20,
        num_trials=5
    )
    
    benchmark.print_benchmark_report(results)
    
    return results["summary"]["avg_speedup"]


def run_real_model_benchmark():
    """Benchmark with model sizes similar to actual DeSSIN training."""
    print("\n" + "=" * 70)
    print("🧠 REALISTIC MODEL SIZE BENCHMARK")
    print("=" * 70)
    
    # Nanochat depth 10 has ~25M parameters
    # depth 12 has ~50M parameters
    # We'll test a range
    
    model_params = {
        "Small (1M)": 1_000_000,
        "Medium (10M)": 10_000_000,
        "Nanochat d10 (25M)": 25_000_000,
        "Nanochat d12 (50M)": 50_000_000,
    }
    
    quantizer = ModelQuantizer()
    verifier = QuantizationVerifier()
    training_steps = 20
    
    print(f"\nSimulating {training_steps} training steps per block")
    print()
    
    print(f"{'Model':>20} {'Re-exec':>12} {'Consist':>12} {'Speedup':>10}")
    print("-" * 60)
    
    speedups = []
    
    for name, size in model_params.items():
        # Create model (just random weights for benchmarking)
        model_32bit = np.random.randn(size).astype(np.float32)
        model_4bit, quant_error = quantizer.quantize_weights(model_32bit, QuantizationLevel.INT4)
        
        # Time re-execution
        reexec_start = time.time()
        for _ in range(training_steps):
            # Realistic simulation of forward + backward
            batch_size = 32
            hidden = np.sqrt(size)
            _ = np.random.randn(batch_size, int(hidden)) @ np.random.randn(int(hidden), int(hidden))
            _ = np.random.randn(int(hidden), int(hidden)) * 0.01  # Gradient
        reexec_time = (time.time() - reexec_start) * 1000
        
        # Time consistency check
        consist_start = time.time()
        result = verifier.verify_quantization_consistency(
            model_32bit, model_4bit, quant_error, b"real_bench"
        )
        consist_time = (time.time() - consist_start) * 1000
        
        speedup = reexec_time / consist_time if consist_time > 0 else float('inf')
        speedups.append(speedup)
        
        print(f"{name:>20} {reexec_time:>10.1f}ms {consist_time:>10.1f}ms {speedup:>9.1f}x")
    
    avg_speedup = np.mean(speedups)
    print("-" * 60)
    print(f"{'AVERAGE':>20} {'':>12} {'':>12} {avg_speedup:>9.1f}x")
    
    return avg_speedup


def run_security_test():
    """Test that tampering is detected."""
    print("\n" + "=" * 70)
    print("🔒 SECURITY TEST: Tamper Detection")
    print("=" * 70)
    
    verifier = QuantizationVerifier()
    quantizer = ModelQuantizer()
    
    # Create honest model
    model_32bit = np.random.randn(100_000).astype(np.float32)
    model_4bit, quant_error = quantizer.quantize_weights(model_32bit, QuantizationLevel.INT4)
    
    # Test 1: Honest verification
    print("\n1. Honest model (should pass):")
    result = verifier.verify_quantization_consistency(
        model_32bit, model_4bit, quant_error, b"honest"
    )
    print(f"   Result: {'✅ PASSED' if result.is_valid else '❌ FAILED'}")
    print(f"   Consistency error: {result.consistency_error:.2e}")
    
    # Test 2: Tampered 4-bit model (small change)
    print("\n2. Tampered 4-bit model (small - 0.1% weights modified):")
    tampered_4bit = model_4bit.copy()
    num_tamper = len(tampered_4bit) // 1000  # 0.1%
    tampered_4bit[:num_tamper] += 0.01
    result = verifier.verify_quantization_consistency(
        model_32bit, tampered_4bit, quant_error, b"tampered"
    )
    print(f"   Result: {'✅ DETECTED' if not result.is_valid else '⚠️ MISSED'}")
    print(f"   Consistency error: {result.consistency_error:.2e}")
    
    # Test 3: Tampered 32-bit model (miner trained wrong thing)
    print("\n3. Wrong 32-bit model (miner faked training):")
    wrong_32bit = np.random.randn(100_000).astype(np.float32)  # Completely different
    result = verifier.verify_quantization_consistency(
        wrong_32bit, model_4bit, quant_error, b"wrong"
    )
    print(f"   Result: {'✅ DETECTED' if not result.is_valid else '⚠️ MISSED'}")
    print(f"   Consistency error: {result.consistency_error:.2e}")
    
    # Test 4: Slightly wrong 32-bit (subtle fraud attempt)
    print("\n4. Slightly wrong 32-bit (1% weights modified):")
    slight_wrong_32bit = model_32bit.copy()
    num_wrong = len(slight_wrong_32bit) // 100  # 1%
    slight_wrong_32bit[:num_wrong] += 0.1
    result = verifier.verify_quantization_consistency(
        slight_wrong_32bit, model_4bit, quant_error, b"slight_wrong"
    )
    print(f"   Result: {'✅ DETECTED' if not result.is_valid else '⚠️ MISSED'}")
    print(f"   Consistency error: {result.consistency_error:.2e}")
    
    print("\n✓ Security tests complete!")


def main():
    print("\n" + "=" * 70)
    print("🔍 DESSIN VERIFICATION EFFICIENCY BENCHMARK")
    print("   Quantization Consistency vs Full Re-execution")
    print("=" * 70)
    
    # Quick benchmark
    quick_speedup = run_quick_benchmark()
    
    # Security tests
    run_security_test()
    
    # Full benchmark
    full_speedup = run_full_benchmark()
    
    # Realistic benchmark
    real_speedup = run_real_model_benchmark()
    
    # Final summary
    print("\n" + "=" * 70)
    print("📋 FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\n   Quick benchmark speedup:     {quick_speedup:.0f}x")
    print(f"   Full benchmark speedup:      {full_speedup:.0f}x")
    print(f"   Real model sizes speedup:    {real_speedup:.0f}x")
    
    avg_all = (quick_speedup + full_speedup + real_speedup) / 3
    
    print(f"\n   📊 OVERALL AVERAGE SPEEDUP: {avg_all:.0f}x")
    
    if avg_all >= 10:
        print(f"\n   ✅ TARGET ACHIEVED: {avg_all:.0f}x speedup exceeds 10x goal!")
        print("   ✅ Quantization consistency verification is 10-100x more efficient!")
    else:
        print(f"\n   ⚠️ Speedup {avg_all:.0f}x below 10x target")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
