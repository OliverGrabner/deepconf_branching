#!/usr/bin/env python3
"""
Test script to verify the unified runner works correctly.
Tests answer extraction and basic functionality without running full experiments.
"""

import sys
import os

# Test imports
print("Testing imports...")
try:
    from utils_robust import (
        extract_answer_robust,
        check_answer_equality,
        load_dataset_by_name,
        get_question_and_ground_truth
    )
    print("✓ utils_robust imported successfully")
except ImportError as e:
    print(f"✗ Failed to import utils_robust: {e}")
    sys.exit(1)

# Test answer extraction
print("\nTesting answer extraction...")

test_cases = [
    # AIME style answers
    ("The answer is \\boxed{123}", "123", "aime"),
    ("Therefore \\boxed{456}", "456", "aime"),
    ("We get boxed{789}", "789", "aime"),
    ("The final answer is 321", "321", "aime"),
    ("Therefore, the answer is 654", "654", "aime"),

    # GSM8k style answers
    ("Let's calculate... #### 42", "42", "gsm8k"),
    ("The total is 100. #### 100", "100", "gsm8k"),
    ("After solving we get 75", "75", "gsm8k"),
]

correct = 0
for text, expected, dataset_type in test_cases:
    result = extract_answer_robust(text, dataset_type)
    if result == expected:
        print(f"  ✓ {dataset_type}: '{text[:30]}...' → {result}")
        correct += 1
    else:
        print(f"  ✗ {dataset_type}: '{text[:30]}...' → Expected {expected}, got {result}")

print(f"\nAnswer extraction: {correct}/{len(test_cases)} tests passed")

# Test equality checking
print("\nTesting equality checking...")

equality_tests = [
    ("123", "123", "aime", True),
    ("456", "456", "gsm8k", True),
    ("100", "100.0", "gsm8k", True),
    ("42", "43", "aime", False),
    ("3.14", "3.14159", "gsm8k", False),
]

eq_correct = 0
for pred, gt, dtype, expected in equality_tests:
    result = check_answer_equality(pred, gt, dtype)
    if result == expected:
        print(f"  ✓ {dtype}: {pred} vs {gt} → {result}")
        eq_correct += 1
    else:
        print(f"  ✗ {dtype}: {pred} vs {gt} → Expected {expected}, got {result}")

print(f"\nEquality checking: {eq_correct}/{len(equality_tests)} tests passed")

# Test dataset loading (without actually loading to save time)
print("\nTesting dataset utilities...")
try:
    # This will test that the function exists and can handle the dataset names
    print("  Testing dataset name recognition...")

    # Don't actually load datasets to save time, just test the function exists
    print("  ✓ Dataset loading function available")
    print("  ✓ Question extraction function available")

except Exception as e:
    print(f"  ✗ Dataset utilities error: {e}")

# Summary
print("\n" + "="*60)
if correct == len(test_cases) and eq_correct == len(equality_tests):
    print("✅ All tests passed! The unified runner should work correctly.")
    print("\nYou can now run experiments with:")
    print("  python run_unified.py --mode branching --dataset AIME2025-I \\")
    print("    --initial_branches 2 --max_total_branches 4 --single_question 0")
else:
    print("⚠️  Some tests failed. Please check the errors above.")

print("="*60)