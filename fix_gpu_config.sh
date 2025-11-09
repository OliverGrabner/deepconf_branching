#!/bin/bash
#################################################
# Fix GPU configuration issues
#################################################

echo "=========================================="
echo "GPU CONFIGURATION FIX"
echo "=========================================="

# Check current GPU status
echo "Current GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo ""

# Kill any stuck processes
echo "Checking for stuck vLLM processes..."
pkill -f "vllm"
pkill -f "ray::"
sleep 2

# Clear GPU memory
echo "Clearing GPU memory..."
for i in {0..3}; do
    echo "Resetting GPU $i..."
    nvidia-smi -i $i -r 2>/dev/null || true
done

# Set proper environment variables
echo ""
echo "Setting environment variables..."

# Reduce CPU threads to avoid contention
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Set CUDA settings
export CUDA_LAUNCH_BLOCKING=1  # Helps with debugging
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Limit to 2 GPUs initially for testing
export CUDA_VISIBLE_DEVICES="0,1"

echo "Environment configured:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo ""

# Download model if needed
echo "Checking if model needs to be downloaded..."
python3 -c "
from huggingface_hub import snapshot_download
import os

model = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
try:
    path = snapshot_download(model, local_files_only=True)
    print(f'✓ Model already downloaded: {path}')
except:
    print('⚠ Model not found locally')
    print('Downloading model (14GB)... This will take several minutes...')
    path = snapshot_download(model)
    print(f'✓ Model downloaded to: {path}')
"

echo ""
echo "=========================================="
echo "TESTING WITH SAFE CONFIGURATION"
echo "=========================================="
echo ""
echo "Running minimal test with:"
echo "  - 2 GPUs only (0,1)"
echo "  - 50% memory utilization"
echo "  - Single problem"
echo ""

# Run safe test
python3 run_safe_test.py

echo ""
echo "=========================================="
echo "If the test succeeded, you can:"
echo "1. Use all 4 GPUs: export CUDA_VISIBLE_DEVICES='0,1,2,3'"
echo "2. Run full comparison: python run_comparison_aime.py"
echo ""
echo "If it failed, try:"
echo "1. python debug_gpu_setup.py"
echo "2. Reduce to single GPU: export CUDA_VISIBLE_DEVICES='0'"
echo "=========================================="