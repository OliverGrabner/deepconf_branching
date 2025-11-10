#!/usr/bin/env python3
"""
Debug GPU setup and test model loading
"""

import torch
import os
import sys

def check_gpu_status():
    """Check GPU availability and memory"""
    print("="*60)
    print("GPU STATUS CHECK")
    print("="*60)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return False

    num_gpus = torch.cuda.device_count()
    print(f"✓ Number of GPUs available: {num_gpus}")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)

        print(f"\nGPU {i}: {props.name}")
        print(f"  Total memory: {memory_gb:.1f} GB")
        print(f"  Allocated: {allocated:.1f} GB")
        print(f"  Reserved: {reserved:.1f} GB")
        print(f"  Free: {memory_gb - reserved:.1f} GB")

    return True

def test_minimal_load():
    """Try to load model with minimal settings"""
    print("\n" + "="*60)
    print("TESTING MINIMAL MODEL LOAD")
    print("="*60)

    try:
        # First check if model exists locally
        from huggingface_hub import snapshot_download
        import os

        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

        print(f"Checking for model: {model_name}")

        # Try to download/verify model exists
        try:
            local_dir = snapshot_download(
                repo_id=model_name,
                local_files_only=True,
                cache_dir=cache_dir
            )
            print(f"✓ Model found locally at: {local_dir}")
        except Exception as e:
            print(f"⚠ Model not found locally, will need to download (~14GB)")
            print(f"  This might be causing the connection reset")
            print(f"  Error: {e}")

            response = input("\nDo you want to download the model now? (y/n): ")
            if response.lower() == 'y':
                print("Downloading model... This may take a while...")
                local_dir = snapshot_download(
                    repo_id=model_name,
                    cache_dir=cache_dir
                )
                print(f"✓ Model downloaded to: {local_dir}")
            else:
                print("Skipping download. Model needs to be downloaded first.")
                return False

        # Now try to load with vLLM
        print("\n" + "-"*40)
        print("Testing vLLM initialization...")
        print("-"*40)

        from vllm import LLM, SamplingParams

        # Very conservative settings
        print("Using conservative settings:")
        print("  tensor_parallel_size: 1 (single GPU)")
        print("  gpu_memory_utilization: 0.5 (50%)")
        print("  max_model_len: 2048 (short context)")
        print("  max_num_seqs: 1 (single sequence)")

        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,  # Start with single GPU
            gpu_memory_utilization=0.5,  # Very conservative
            max_model_len=2048,  # Short context for testing
            max_num_seqs=1,  # Single sequence
            trust_remote_code=True,
            enable_prefix_caching=False,  # Disable for testing
        )

        print("✓ Model loaded successfully!")

        # Try a simple generation
        print("\nTesting generation...")
        sampling_params = SamplingParams(temperature=0.1, max_tokens=10)
        output = llm.generate("What is 2+2?", sampling_params)
        print(f"✓ Generation successful: {output[0].outputs[0].text[:50]}")

        del llm  # Clean up
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ Error during model load: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parallel_load():
    """Test with parallel GPUs"""
    print("\n" + "="*60)
    print("TESTING PARALLEL GPU LOAD")
    print("="*60)

    try:
        from vllm import LLM, SamplingParams

        print("Testing with 2 GPUs...")
        print("Settings:")
        print("  tensor_parallel_size: 2")
        print("  gpu_memory_utilization: 0.7")
        print("  max_model_len: 8192")
        print("  max_num_seqs: 32")

        llm = LLM(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.7,
            max_model_len=8192,
            max_num_seqs=32,
            trust_remote_code=True,
            enable_prefix_caching=True,
        )

        print("✓ 2-GPU model loaded successfully!")

        # Test generation
        sampling_params = SamplingParams(temperature=0.6, max_tokens=50)
        output = llm.generate("Explain chain of thought reasoning:", sampling_params)
        print(f"✓ Generation successful!")

        del llm
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"❌ Error with parallel GPUs: {e}")
        return False

def main():
    print("="*60)
    print("DEEPSEEK-7B GPU DEBUGGING")
    print("="*60)

    # Check environment variables
    print("\nEnvironment Variables:")
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (all GPUs visible)')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

    # Step 1: Check GPUs
    if not check_gpu_status():
        print("\n❌ GPU check failed. Please ensure CUDA is properly installed.")
        return

    # Step 2: Test minimal load
    print("\n" + "="*60)
    print("Step 1: Testing minimal single-GPU load")
    print("="*60)
    if test_minimal_load():
        print("✓ Single GPU test passed!")

        # Step 3: Test parallel load
        print("\n" + "="*60)
        print("Step 2: Testing multi-GPU load")
        print("="*60)
        if test_parallel_load():
            print("✓ Multi-GPU test passed!")

            print("\n" + "="*60)
            print("✅ ALL TESTS PASSED")
            print("="*60)
            print("\nRecommended settings for your system:")
            print("  --tensor_parallel_size 2")
            print("  --gpu_memory_utilization 0.75")
            print("  --max_num_seqs 64")
            print("  --max_model_len 16384")
            print("\nYou can now run:")
            print("  python run_comparison_aime.py --tensor_parallel_size 2")
        else:
            print("\n⚠ Multi-GPU test failed. Try using single GPU:")
            print("  python run_comparison_aime.py --tensor_parallel_size 1")
    else:
        print("\n❌ Single GPU test failed.")
        print("\nPossible issues:")
        print("1. Model not downloaded (14GB needed)")
        print("2. Insufficient GPU memory")
        print("3. CUDA/PyTorch version mismatch")
        print("4. vLLM installation issue")

        print("\nTry these fixes:")
        print("1. Download model first:")
        print("   python -c \"from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')\"")
        print("\n2. Reinstall vLLM:")
        print("   pip install --upgrade vllm")
        print("\n3. Check CUDA version:")
        print("   nvidia-smi")
        print("   python -c \"import torch; print(torch.version.cuda)\"")

if __name__ == "__main__":
    main()