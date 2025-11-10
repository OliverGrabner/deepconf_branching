#!/usr/bin/env python3
"""
Test script for multi-stage peak branching implementation

This script tests the new multi-stage peak branching with doubling strategy
on a single AIME question to verify the implementation works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepconf.peak_branching import PeakBranchingManager


def test_branching_stages():
    """Test the branching stage calculation"""
    print("Testing branching stage calculation...")

    # Test case 1: 8 initial, max 32
    manager = PeakBranchingManager(initial_traces=8, max_traces=32)
    print(f"\nInitial=8, Max=32:")
    print(f"  Stages: {manager.branching_stages}")
    print(f"  Total: {8 + sum(manager.branching_stages)}")
    assert manager.branching_stages == [8, 16], f"Expected [8, 16], got {manager.branching_stages}"

    # Test case 2: 8 initial, max 64
    manager = PeakBranchingManager(initial_traces=8, max_traces=64)
    print(f"\nInitial=8, Max=64:")
    print(f"  Stages: {manager.branching_stages}")
    print(f"  Total: {8 + sum(manager.branching_stages)}")
    assert manager.branching_stages == [8, 16, 32], f"Expected [8, 16, 32], got {manager.branching_stages}"

    # Test case 3: 4 initial, max 32
    manager = PeakBranchingManager(initial_traces=4, max_traces=32)
    print(f"\nInitial=4, Max=32:")
    print(f"  Stages: {manager.branching_stages}")
    print(f"  Total: {4 + sum(manager.branching_stages)}")
    assert manager.branching_stages == [4, 8, 16], f"Expected [4, 8, 16], got {manager.branching_stages}"

    # Test case 4: 16 initial, max 64
    manager = PeakBranchingManager(initial_traces=16, max_traces=64)
    print(f"\nInitial=16, Max=64:")
    print(f"  Stages: {manager.branching_stages}")
    print(f"  Total: {16 + sum(manager.branching_stages)}")
    assert manager.branching_stages == [16, 32], f"Expected [16, 32], got {manager.branching_stages}"

    print("\n✓ All branching stage calculations passed!")


def test_exclusion_zones():
    """Test exclusion zone functionality"""
    print("\nTesting exclusion zones...")

    manager = PeakBranchingManager(
        initial_traces=2,
        max_traces=8,
        exclusion_zone_size=200
    )

    # Mark some exclusion zones
    manager.mark_exclusion_zone(0, 1000)  # Trace 0, position 1000
    manager.mark_exclusion_zone(1, 2000)  # Trace 1, position 2000

    # Test positions
    assert manager.is_position_in_exclusion_zone(0, 1000) == True, "Position 1000 should be in zone"
    assert manager.is_position_in_exclusion_zone(0, 950) == True, "Position 950 should be in zone"
    assert manager.is_position_in_exclusion_zone(0, 1050) == True, "Position 1050 should be in zone"
    assert manager.is_position_in_exclusion_zone(0, 800) == False, "Position 800 should NOT be in zone"
    assert manager.is_position_in_exclusion_zone(1, 1000) == False, "Trace 1 position 1000 should NOT be in zone"

    print(f"  Created {len(manager.used_peak_zones)} exclusion zones")
    print("✓ Exclusion zone tests passed!")


def test_trace_creation():
    """Test trace creation with stages"""
    print("\nTesting trace creation...")

    manager = PeakBranchingManager(initial_traces=2, max_traces=8)

    # Create initial traces
    trace1 = manager.create_initial_trace(
        text="Initial trace 1",
        token_ids=list(range(100)),
        confs=[0.5] * 100,
        extracted_answer="42"
    )

    trace2 = manager.create_initial_trace(
        text="Initial trace 2",
        token_ids=list(range(150)),
        confs=[0.6] * 150,
        extracted_answer="37"
    )

    assert trace1.stage == 0, "Initial trace should be stage 0"
    assert trace1.parent_idx is None, "Initial trace should have no parent"
    assert len(manager.traces) == 2, "Should have 2 traces"

    # Create a branch trace
    branch1 = manager.create_branch_trace(
        parent_trace_idx=0,
        branch_point=50,
        stage=1,
        new_text="Branched from trace 0",
        new_token_ids=list(range(200)),
        new_confs=[0.7] * 200,
        extracted_answer="41"
    )

    assert branch1.stage == 1, "Branch should be stage 1"
    assert branch1.parent_idx == 0, "Branch should have parent 0"
    assert branch1.branch_point_tokens == 50, "Branch point should be 50"
    assert branch1.tokens_generated == 150, "Should have generated 150 new tokens (200-50)"
    assert len(manager.traces) == 3, "Should have 3 traces total"

    # Create a branch from a branch (stage 2)
    branch2 = manager.create_branch_trace(
        parent_trace_idx=2,  # The previous branch
        branch_point=100,
        stage=2,
        new_text="Branched from branch",
        new_token_ids=list(range(250)),
        new_confs=[0.8] * 250,
        extracted_answer="40"
    )

    assert branch2.stage == 2, "Second-level branch should be stage 2"
    assert branch2.parent_idx == 2, "Should branch from trace 2"
    assert branch2.tokens_generated == 150, "Should have generated 150 new tokens"

    print(f"  Created {len(manager.traces)} traces across stages")
    print(f"  Stages: {[t.stage for t in manager.traces]}")
    print("✓ Trace creation tests passed!")


def main():
    """Run all tests"""
    print("="*60)
    print("TESTING MULTI-STAGE PEAK BRANCHING")
    print("="*60)

    test_branching_stages()
    test_exclusion_zones()
    test_trace_creation()

    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    print("\nThe multi-stage peak branching implementation is working correctly.")
    print("Key features verified:")
    print("  - Doubling strategy for branching stages")
    print("  - Exclusion zones to prevent duplicate peaks")
    print("  - Multi-stage trace creation with proper genealogy")
    print("  - Token accounting with prefix subtraction")


if __name__ == "__main__":
    main()