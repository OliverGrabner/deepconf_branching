#!/bin/bash

# Traditional Self-Consistency Experiments on AIME 2025
# This script runs multiple configurations for comprehensive evaluation

# Set output directory
OUTPUT_DIR="outputs_sc"
mkdir -p $OUTPUT_DIR

# Model configuration
MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MODEL_TYPE="deepseek"

echo "=========================================="
echo "Traditional Self-Consistency on AIME 2025"
echo "=========================================="
echo ""

# ============================================
# Experiment 1: Quick test with 8 traces
# ============================================
echo "Experiment 1: Quick test (8 traces, first 3 questions)"
echo "------------------------------------------"
python scripts/run_traditional_sc_aime25.py \
    --model $MODEL \
    --model_type $MODEL_TYPE \
    --num_traces 8 \
    --end_idx 3 \
    --output_dir $OUTPUT_DIR

echo ""
echo "Experiment 1 complete!"
echo ""

# ============================================
# Experiment 2: Standard SC (64 traces)
# ============================================
echo "Experiment 2: Standard Self-Consistency (64 traces, both datasets)"
echo "------------------------------------------"
python scripts/run_traditional_sc_aime25.py \
    --model $MODEL \
    --model_type $MODEL_TYPE \
    --num_traces 64 \
    --output_dir $OUTPUT_DIR

echo ""
echo "Experiment 2 complete!"
echo ""

# ============================================
# Experiment 3: High-budget SC (128 traces)
# ============================================
# Uncomment to run high-budget experiment
# echo "Experiment 3: High-budget Self-Consistency (128 traces, both datasets)"
# echo "------------------------------------------"
# python scripts/run_traditional_sc_aime25.py \
#     --model $MODEL \
#     --model_type $MODEL_TYPE \
#     --num_traces 128 \
#     --output_dir $OUTPUT_DIR
#
# echo ""
# echo "Experiment 3 complete!"
# echo ""

# ============================================
# Experiment 4: Low-budget SC (16 traces)
# ============================================
# Uncomment to run low-budget experiment
# echo "Experiment 4: Low-budget Self-Consistency (16 traces, both datasets)"
# echo "------------------------------------------"
# python scripts/run_traditional_sc_aime25.py \
#     --model $MODEL \
#     --model_type $MODEL_TYPE \
#     --num_traces 16 \
#     --output_dir $OUTPUT_DIR
#
# echo ""
# echo "Experiment 4 complete!"
# echo ""

echo "=========================================="
echo "All experiments complete!"
echo "Results saved in: $OUTPUT_DIR"
echo "=========================================="

# Display summary of output files
echo ""
echo "Output files:"
ls -lh $OUTPUT_DIR/*.json $OUTPUT_DIR/*.csv 2>/dev/null

echo ""
echo "To view CSV results:"
echo "  cat $OUTPUT_DIR/traditional_sc_aime25_summary_*.csv | column -t -s, | less -S"
echo ""
echo "To analyze JSON results:"
echo "  python scripts/analyze_sc_results.py $OUTPUT_DIR/traditional_sc_aime25_detailed_*.json"
