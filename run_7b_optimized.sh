#!/bin/bash
#################################################
# Optimized runner for 7B model on 4x A5000 GPUs
#################################################

# Model: DeepSeek-R1-Distill-Qwen-7B
# Hardware: 4x NVIDIA A5000 Ada (32GB each, 128GB total)
# Optimized for 7B parameter model

echo "=========================================="
echo "Running with DeepSeek-R1-Distill-Qwen-7B"
echo "=========================================="

# Set GPU visibility (all 4 GPUs)
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Option 1: Run unified runner with branching (YOUR method)
echo ""
echo "Option 1: Unified runner with YOUR branching method"
echo "-----------------------------------------------------"
echo "Running AIME2025-I with confidence-based branching..."
echo ""

python run_unified.py \
    --mode branching \
    --dataset AIME2025-I \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --initial_branches 8 \
    --max_total_branches 32 \
    --confidence_threshold 1.5 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.85 \
    --max_num_seqs 128 \
    --max_tokens 32768 \
    --temperature 0.6 \
    --output_dir results/unified_7b

echo ""
echo "=========================================="
echo ""

# Option 2: Run original AIME runner with branching
echo "Option 2: Original AIME runner with branching"
echo "----------------------------------------------"
echo "Running full AIME (both I and II) with branching..."
echo ""

# Uncomment to run:
# python run_aime25_full.py \
#     --mode branching \
#     --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
#     --initial_branches 4 \
#     --max_total_branches 16 \
#     --tensor_parallel_size 2 \
#     --gpu_memory_utilization 0.85 \
#     --max_num_seqs 128 \
#     --output_dir results/aime25_7b \
#     --save_plots

echo ""
echo "=========================================="
echo ""

# Option 3: Test on small subset first
echo "Option 3: Quick test on first 3 problems"
echo "-----------------------------------------"
echo ""

# Uncomment to run:
# python run_unified.py \
#     --mode branching \
#     --dataset AIME2025-I \
#     --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
#     --initial_branches 4 \
#     --max_total_branches 8 \
#     --confidence_threshold 1.5 \
#     --tensor_parallel_size 2 \
#     --gpu_memory_utilization 0.85 \
#     --max_num_seqs 64 \
#     --max_tokens 16384 \
#     --end_idx 3 \
#     --output_dir results/test_7b

echo ""
echo "=========================================="
echo "MEMORY OPTIMIZATION TIPS:"
echo "=========================================="
echo "If you still encounter CUDA OOM errors:"
echo ""
echo "1. Reduce gpu_memory_utilization to 0.75 or 0.7"
echo "2. Reduce max_num_seqs to 64 or 32"
echo "3. Reduce max_tokens to 16384 or 8192 if needed"
echo "4. Reduce max_total_branches"
echo "5. Use only 2 GPUs instead of 4 (set tensor_parallel_size=2)"
echo ""
echo "The 7B model should fit comfortably with these settings."
echo "=========================================="