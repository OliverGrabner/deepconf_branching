#!/usr/bin/env python3
"""
Safe test run with conservative GPU settings
Runs on first problem only with reduced memory usage
"""

import subprocess
import sys
import os

# Force specific GPUs if needed
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use only first 2 GPUs

def run_safe_test():
    """Run with very conservative settings to avoid OOM"""

    print("="*60)
    print("SAFE TEST RUN - SINGLE PROBLEM")
    print("="*60)
    print("Using conservative settings to avoid GPU issues:")
    print("  - Single GPU (tensor_parallel_size=1)")
    print("  - 50% memory utilization")
    print("  - Reduced sequences and tokens")
    print("  - Single problem only")
    print("="*60)

    cmd = [
        sys.executable, "run_unified.py",
        "--mode", "branching",
        "--dataset", "AIME2025-I",
        "--model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "--single_question", "0",  # Just first problem
        "--initial_branches", "2",  # Very few branches
        "--max_total_branches", "4",  # Very few total
        "--confidence_threshold", "2.0",  # High threshold = less branching
        "--tensor_parallel_size", "1",  # Single GPU
        "--gpu_memory_utilization", "0.5",  # Only 50% of GPU memory
        "--max_num_seqs", "4",  # Very few sequences
        "--max_tokens", "4096",  # Shorter generation
        "--temperature", "0.6"
    ]

    print(f"\nCommand: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✅ Safe test completed successfully!")
        print("\nYou can gradually increase settings:")
        print("  1. Try tensor_parallel_size=2 (use 2 GPUs)")
        print("  2. Increase gpu_memory_utilization to 0.7")
        print("  3. Increase max_num_seqs to 32")
        print("  4. Increase branches to 8→32")
    else:
        print(f"\n❌ Test failed with return code {result.returncode}")
        print("\nTry debugging with:")
        print("  python debug_gpu_setup.py")

if __name__ == "__main__":
    # First check if GPUs are available
    import torch
    if torch.cuda.is_available():
        print(f"GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available!")
        sys.exit(1)

    run_safe_test()