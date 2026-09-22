#!/usr/bin/env python3
"""
Comprehensive test suite for the full PoGO protocol
Runs all PoGO-related unit tests and integration tests
"""

import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import all test modules
from .test_pogo_merkle_tree import TestMerkleTree
from .test_pogo_quantization import TestModelQuantization
from .test_pogo_commitments import TestModelCommitments
from .test_pogo_validation import TestPoGOValidation


def create_comprehensive_test_suite():
    """Create a comprehensive test suite for PoGO protocol"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestMerkleTree,
        TestModelQuantization,
        TestModelCommitments,
        TestPoGOValidation
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_comprehensive_tests():
    """Run all PoGO protocol tests"""
    print("🔍 COMPREHENSIVE PoGO PROTOCOL TEST SUITE")
    print("=" * 80)
    print("Running all unit tests and integration tests for PoGO protocol...")
    print("=" * 80)
    
    suite = create_comprehensive_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("🎉 ALL PoGO PROTOCOL TESTS PASSED!")
        print(f"✓ {result.testsRun} tests completed successfully")
        print("✓ Merkle tree implementation working correctly")
        print("✓ Model quantization working correctly")
        print("✓ Model commitments working correctly")
        print("✓ Protocol validation working correctly")
        print("✓ All protocol tests passing")
    else:
        print("💥 SOME TESTS FAILED!")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
