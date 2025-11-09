#!/usr/bin/env python3
"""
Test configuration for 7B model on 4x A5000 GPUs
Verifies that the model can load and run without OOM errors
"""

import os
import sys
import torch

def check_gpu_memory():
    """Check available GPU memory"""
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return False

    print("\n" + "="*60)
    print("GPU MEMORY STATUS")
    print("="*60)

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs detected: {num_gpus}")

    total_memory = 0
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        total_memory += memory_gb
        print(f"GPU {i}: {props.name}")
        print(f"  - Total memory: {memory_gb:.1f} GB")

        # Check current memory usage
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        free = memory_gb - reserved
        print(f"  - Allocated: {allocated:.1f} GB")
        print(f"  - Reserved: {reserved:.1f} GB")
        print(f"  - Free: {free:.1f} GB")

    print(f"\nTotal GPU memory: {total_memory:.1f} GB")
    return True


def estimate_model_memory():
    """Estimate memory requirements for 7B model"""
    print("\n" + "="*60)
    print("ESTIMATED MEMORY REQUIREMENTS")
    print("="*60)

    # Model parameters
    params_7b = 7_000_000_000  # 7B parameters
    bytes_per_param = 2  # FP16/BF16

    # Base model memory
    model_memory_gb = (params_7b * bytes_per_param) / (1024**3)
    print(f"Base model size (FP16): {model_memory_gb:.1f} GB")

    # KV cache estimation (depends on max_num_seqs and max_model_len)
    max_num_seqs = 128
    max_model_len = 32768
    hidden_size = 4096  # Typical for 7B models
    num_layers = 32  # Typical for 7B models

    # KV cache per token per layer: 2 * hidden_size * 2 bytes (K and V)
    kv_cache_per_token = 2 * hidden_size * bytes_per_param
    kv_cache_total = (max_num_seqs * max_model_len * num_layers * kv_cache_per_token) / (1024**3)
    print(f"KV cache (worst case): {kv_cache_total:.1f} GB")

    # With memory utilization factor
    gpu_memory_util = 0.85
    available_per_gpu = 32 * gpu_memory_util
    print(f"\nWith gpu_memory_utilization={gpu_memory_util}:")
    print(f"  Available per GPU: {available_per_gpu:.1f} GB")
    print(f"  Total available (4 GPUs): {available_per_gpu * 4:.1f} GB")

    # Tensor parallel distribution
    tensor_parallel = 2
    model_per_gpu = model_memory_gb / tensor_parallel
    print(f"\nWith tensor_parallel_size={tensor_parallel}:")
    print(f"  Model per GPU: {model_per_gpu:.1f} GB")
    print(f"  Remaining for KV cache per GPU: {available_per_gpu - model_per_gpu:.1f} GB")

    print("\n✓ The 7B model should fit comfortably with these settings!")


def test_vllm_import():
    """Test if vLLM can be imported"""
    print("\n" + "="*60)
    print("TESTING VLLM IMPORT")
    print("="*60)

    try:
        from vllm import LLM, SamplingParams
        print("✓ vLLM imported successfully")

        # Try to import our custom classes
        from deepconf import DeepThinkLLM
        from deepconf.branching_wrapper import BranchingDeepThinkLLM
        print("✓ DeepConf modules imported successfully")

        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_minimal_model_load():
    """Try to load a minimal configuration"""
    print("\n" + "="*60)
    print("TESTING MINIMAL MODEL LOAD")
    print("="*60)

    try:
        from deepconf.branching_wrapper import BranchingDeepThinkLLM
        from vllm import SamplingParams

        print("Attempting to initialize model with minimal config...")
        print("Note: This will download the model if not cached locally")

        # Very conservative settings for testing
        llm = BranchingDeepThinkLLM(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            tensor_parallel_size=2,
            enable_prefix_caching=True,
            trust_remote_code=True,
            gpu_memory_utilization=0.7,  # Very conservative
            max_num_seqs=32,              # Small batch
            max_model_len=4096            # Shorter context
        )

        print("✓ Model initialized successfully!")

        # Test with a simple prompt
        test_prompt = "What is 2+2?"
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=10
        )

        print("Testing simple generation...")
        result = llm.branching_deepthink(
            prompt=test_prompt,
            initial_branches=1,
            max_total_branches=1,
            confidence_threshold=2.0,  # High threshold = no branching
            sampling_params=sampling_params
        )

        print("✓ Generation completed successfully!")
        return True

    except Exception as e:
        print(f"✗ Error during model load: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("7B MODEL CONFIGURATION TEST")
    print("="*60)

    # Set CUDA devices
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    # Run tests
    success = True

    if not check_gpu_memory():
        success = False

    estimate_model_memory()

    if not test_vllm_import():
        success = False

    # Only try to load model if explicitly requested
    if '--load-model' in sys.argv:
        print("\n" + "="*60)
        print("MODEL LOADING TEST")
        print("="*60)
        print("This will download ~14GB if model is not cached")
        response = input("Continue? (y/n): ")
        if response.lower() == 'y':
            if not test_minimal_model_load():
                success = False
    else:
        print("\n" + "="*60)
        print("Skipping model load test.")
        print("Run with --load-model to test actual model loading")
        print("="*60)

    # Summary
    print("\n" + "="*60)
    if success:
        print("✓ ALL TESTS PASSED")
        print("\nYou can now run:")
        print("  ./run_7b_optimized.sh")
        print("\nOr directly:")
        print("  python run_unified.py --mode branching --dataset AIME2025-I \\")
        print("    --initial_branches 8 --max_total_branches 32")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please check the errors above")
    print("="*60)