#!/usr/bin/env python3
"""
Quick test script for comparison pipeline
Tests on first 3 problems to verify everything works
"""

import subprocess
import sys

def main():
    print("="*60)
    print("TESTING COMPARISON PIPELINE")
    print("Running on first 3 AIME2025-I problems")
    print("="*60)

    cmd = [
        sys.executable, "run_comparison_aime.py",
        "--dry_run"  # This limits to 3 problems
    ]

    print(f"\nCommand: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✅ Test completed successfully!")
        print("You can now run the full experiment with:")
        print("  python run_comparison_aime.py")
        print("or")
        print("  ./run_experiments.sh")
    else:
        print(f"\n❌ Test failed with return code {result.returncode}")

if __name__ == "__main__":
    main()