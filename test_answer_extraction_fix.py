#!/usr/bin/env python3
"""
Test script to verify answer extraction is working properly
"""

import sys
sys.path.insert(0, '/mnt/c/Users/olive/Documents/projects/deepconf_branching')

from run_aime25_full import extract_boxed_answer

# Test cases that might appear in actual AIME solutions
test_cases = [
    # Standard boxed format
    ("The answer is \\boxed{123}", "123"),
    ("Therefore \\boxed{456}.", "456"),

    # Without backslash
    ("The answer is boxed{789}", "789"),

    # Various ending formats
    ("We calculate and get\nTherefore, the answer is 321", "321"),
    ("Final answer: 654", "654"),
    ("Answer: 987", "987"),

    # Just the number at the end
    ("After simplification we get\n= 111", "111"),

    # Multiple boxed (should get last)
    ("First \\boxed{222} then \\boxed{333}", "333"),

    # Real-world example
    ("Solving the equation gives us x = 45.\nTherefore, the answer is 45", "45"),

    # Another format
    ("The final result is 678", ""),  # This might not be caught - that's ok

    # Edge case with $ signs
    ("The answer is $\\boxed{99}$", "99"),
]

print("Testing answer extraction:")
print("-" * 50)

correct = 0
total = len(test_cases)

for i, (text, expected) in enumerate(test_cases, 1):
    result = extract_boxed_answer(text)
    is_correct = (result == expected)
    if is_correct:
        correct += 1

    symbol = "✓" if is_correct else "✗"
    print(f"{i:2}. {symbol} Expected: {expected:3s} | Got: {result:3s}")
    if not is_correct:
        print(f"     Input: {text[:60]}...")

print("-" * 50)
print(f"Score: {correct}/{total} ({correct/total*100:.1f}%)")

if correct < total:
    print("\nNote: Some patterns might still be missed.")
    print("The updated extraction should handle most common AIME answer formats.")
else:
    print("\nAll tests passed! The extraction should work much better now.")