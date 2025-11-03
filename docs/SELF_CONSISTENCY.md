# Self-Consistency on AIME 2025

Traditional self-consistency (Wang et al., 2022) implementation on AIME 2025 datasets.

## What is Self-Consistency?

1. **Generate N reasoning paths** (with temperature > 0 for diversity)
2. **Extract answers** from each path
3. **Majority vote** to select final answer

No confidence weighting - pure majority voting.

## Usage

```bash
# Standard run (64 traces on both datasets)
python scripts/run_traditional_sc_aime25.py --num_traces 64

# Quick test
python scripts/test_sc_single_question.py
```

## Key Parameters

```bash
--model MODEL                    # Model to use (default: DeepSeek-R1)
--num_traces N                   # Number of reasoning paths (default: 64)
--temperature 1.0                # Sampling temperature (use 1.0 for diversity)
--dataset AIME2025-I|AIME2025-II # Which dataset (default: both)
--start_idx N --end_idx M        # Question range
--tensor_parallel_size 4         # Number of GPUs
```

## Output

Three files saved to `outputs_sc/`:

1. **Detailed JSON** - Full trace data, all reasoning paths
2. **Summary CSV** - Spreadsheet format, one row per question
3. **Stats JSON** - Aggregate statistics, accuracy metrics

## Understanding Results

### Console Output

```
Q0: ✓
  Ground Truth: 42
  Voted Answer: 42
  Valid Traces: 64/64
  Individual Accuracy: 78.1%
  Vote Distribution: {'42': 50, '43': 10, '41': 4}
  Tokens: 98,432
  Time: 45.32s
```

**Key metrics:**
- **Individual Accuracy**: % of single traces that were correct
- **Voted Answer**: Final answer after majority vote
- **Vote Distribution**: How traces voted

### Final Summary

```
Overall Results (AIME25-I + AIME25-II):
  Total Correct: 17/30 (56.7%)
  Total Tokens: 2,432,801
  Total Time: 1224.6s (20.4 minutes)
```

## Why It Works

Even if individual traces are only 45% accurate, majority voting can achieve 60%+ accuracy. Different reasoning paths make different mistakes, but correct answers tend to converge.

## Expected Performance

AIME is extremely challenging (math olympiad level):
- **Individual trace accuracy**: 30-50%
- **Self-consistency accuracy**: 40-70%
- **SC improvement**: +10-20% over single path

## Analysis

```bash
# Deep analysis
python scripts/analyze_sc_results.py outputs_sc/*.json

# Visualization
python scripts/visualize_sc_results.py outputs_sc/*.json
```

Shows:
- Vote consensus vs correctness
- Individual vs voting accuracy
- Answer diversity patterns
- Success/failure cases

## Troubleshooting

**Out of memory:**
```bash
python scripts/run_traditional_sc_aime25.py --num_traces 32 --max_tokens 65000
```

**Slow:**
```bash
nvidia-smi  # Check GPU usage
```

## Citation

```bibtex
@article{wang2022self,
  title={Self-consistency improves chain of thought reasoning in language models},
  author={Wang, Xuezhi and Wei, Jason and Schuurmans, Dale and Le, Quoc and Chi, Ed and Narang, Sharan and Chowdhery, Aakanksha and Zhou, Denny},
  journal={arXiv preprint arXiv:2203.11171},
  year={2022}
}
```
