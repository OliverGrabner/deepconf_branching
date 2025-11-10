#!/bin/bash
#################################################
# Run AIME2025-I comparison experiments
# Compares branching vs standard self-consistency
#################################################

# Configuration
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TENSOR_PARALLEL=2
GPU_MEMORY=0.85
MAX_NUM_SEQS=128
TEMPERATURE=0.6
MAX_TOKENS=32768

# Branching config (8 initial, 32 total = 8 + 24 branches)
INITIAL_BRANCHES=8
MAX_TOTAL_BRANCHES=32
CONFIDENCE_THRESHOLD=1.5

# Standard config (32 traces for fair comparison)
NUM_TRACES=32

# Set GPU visibility (all 4 GPUs)
export CUDA_VISIBLE_DEVICES="0,1,2,3"

echo "============================================"
echo "AIME2025-I COMPARISON EXPERIMENT"
echo "============================================"
echo "Model: $MODEL"
echo "Temperature: $TEMPERATURE"
echo "Max tokens: $MAX_TOKENS"
echo ""
echo "Branching: $INITIAL_BRANCHES initial → $MAX_TOTAL_BRANCHES total"
echo "Standard: $NUM_TRACES traces"
echo "============================================"
echo ""

# Parse command line arguments
DRY_RUN=""
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN="--dry_run"
    echo "🔬 DRY RUN MODE: Testing on first 3 problems only"
    echo ""
fi

# Option 1: Run complete comparison (both methods)
echo "Option 1: Run complete comparison"
echo "----------------------------------"
echo "This will run both branching and standard SC, then create comparison charts"
echo ""
echo "Command:"
echo "python run_comparison_aime.py \\"
echo "    --model \"$MODEL\" \\"
echo "    --tensor_parallel_size $TENSOR_PARALLEL \\"
echo "    --gpu_memory_utilization $GPU_MEMORY \\"
echo "    --max_num_seqs $MAX_NUM_SEQS \\"
echo "    --temperature $TEMPERATURE \\"
echo "    --max_tokens $MAX_TOKENS \\"
echo "    --initial_branches $INITIAL_BRANCHES \\"
echo "    --max_total_branches $MAX_TOTAL_BRANCHES \\"
echo "    --confidence_threshold $CONFIDENCE_THRESHOLD \\"
echo "    --num_traces $NUM_TRACES $DRY_RUN"
echo ""
echo "Press Enter to run this, or Ctrl+C to cancel..."
read -r

python run_comparison_aime.py \
    --model "$MODEL" \
    --tensor_parallel_size $TENSOR_PARALLEL \
    --gpu_memory_utilization $GPU_MEMORY \
    --max_num_seqs $MAX_NUM_SEQS \
    --temperature $TEMPERATURE \
    --max_tokens $MAX_TOKENS \
    --initial_branches $INITIAL_BRANCHES \
    --max_total_branches $MAX_TOTAL_BRANCHES \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --num_traces $NUM_TRACES $DRY_RUN

echo ""
echo "============================================"
echo "EXPERIMENT COMPLETE"
echo "============================================"
echo ""
echo "Alternative: Run methods individually"
echo "--------------------------------------"
echo ""
echo "# Run branching only:"
echo "python run_unified.py --mode branching --dataset AIME2025-I \\"
echo "    --initial_branches $INITIAL_BRANCHES --max_total_branches $MAX_TOTAL_BRANCHES \\"
echo "    --confidence_threshold $CONFIDENCE_THRESHOLD \\"
echo "    --temperature $TEMPERATURE --max_tokens $MAX_TOKENS"
echo ""
echo "# Run standard SC only:"
echo "python run_unified.py --mode standard --dataset AIME2025-I \\"
echo "    --num_traces $NUM_TRACES \\"
echo "    --temperature $TEMPERATURE --max_tokens $MAX_TOKENS"
echo ""
echo "# Create charts from existing results:"
echo "python -c \"
import json
import sys
sys.path.append('.')
from run_comparison_aime import create_comparison_charts, extract_statistics

# Load results
with open('results/branching_[timestamp]/branching_AIME2025-I_[timestamp].json') as f:
    branching_results = json.load(f)
with open('results/standard_[timestamp]/standard_AIME2025-I_[timestamp].json') as f:
    standard_results = json.load(f)

# Extract stats and create charts
branching_stats = extract_statistics(branching_results)
standard_stats = extract_statistics(standard_results)
create_comparison_charts(branching_stats, standard_stats, 'results/comparison_charts')
\""