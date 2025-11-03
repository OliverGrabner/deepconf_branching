# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Run Experiments

### Test on One Question (~2 min)

```bash
python scripts/test_sc_single_question.py
```

### Full Experiment (~30-40 min)

```bash
# Both AIME25-I and AIME25-II with 64 traces
python scripts/run_traditional_sc_aime25.py --num_traces 64

# Single dataset only
python scripts/run_traditional_sc_aime25.py --dataset AIME2025-I --num_traces 64

# Quick test (first 5 questions)
python scripts/run_traditional_sc_aime25.py --end_idx 5 --num_traces 16
```

## Visualize & Analyze

### Trace Confidence Evolution

```bash
python scripts/visualize_trace_confidence.py --qid 0 --num_traces 16
```

### Analyze Results

```bash
python scripts/analyze_sc_results.py outputs_sc/traditional_sc_aime25_detailed_*.json
python scripts/visualize_sc_results.py outputs_sc/traditional_sc_aime25_detailed_*.json
```

## Output Files

Results saved to `outputs_sc/`:
- `*_detailed_*.json` - Full trace data
- `*_summary_*.csv` - Spreadsheet format
- `*_stats_*.json` - Statistics
- `trace_confidence_*.png` - Graphs

## Common Options

```bash
--num_traces 64          # Number of reasoning paths
--temperature 1.0        # Sampling temperature
--dataset AIME2025-I     # Which dataset
--qid 0                  # Specific question ID
--end_idx 5              # First N questions only
--tensor_parallel_size 4 # Number of GPUs
```

## Troubleshooting

**Out of memory:**
```bash
python scripts/run_traditional_sc_aime25.py --num_traces 32
```

**Slow generation:**
```bash
nvidia-smi  # Check GPU utilization
```

**Module not found:**
Make sure you run from project root, not `scripts/` directory.

## More Info

- **SC Details**: See `SELF_CONSISTENCY.md`
- **Visualization**: See `TRACE_VISUALIZATION.md`
