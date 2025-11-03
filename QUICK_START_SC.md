# Quick Start: Traditional Self-Consistency on AIME 2025

## TL;DR - Run This

```bash
# 1. Install dependencies
pip install -r requirements_sc.txt

# 2. Test with a single question (8 traces, ~2-3 minutes)
python test_sc_single_question.py

# 3. Run full experiment (64 traces on both AIME25-I and AIME25-II)
python run_traditional_sc_aime25.py --num_traces 64
```

## What You'll Get

### Console Output (Real-time)

```
Processing AIME2025-I (15 questions)
================================================================================

============================================================
Question 1/15
============================================================
Q: Let $ABCD$ be a parallelogram with...

Q0: ✓
  Ground Truth: 42
  Voted Answer: 42
  Valid Traces: 64/64
  Individual Accuracy: 78.1%
  Vote Distribution: {'42': 50, '43': 10, '41': 4}
  Tokens: 98,432 (1,538.0 avg)
  Time: 45.32s
```

### Final Summary

```
================================================================================
TRADITIONAL SELF-CONSISTENCY - FINAL SUMMARY
================================================================================

Per-Dataset Results:
--------------------------------------------------------------------------------

AIME2025-I:
  Questions: 15
  Correct: 9/15 (60.0%)
  Avg Individual Trace Accuracy: 45.2%
  Total Tokens: 1,234,567
  Avg Tokens/Question: 82,304
  Total Time: 623.4s
  Avg Time/Question: 41.6s
  Throughput: 1,980.5 tokens/sec

AIME2025-II:
  Questions: 15
  Correct: 8/15 (53.3%)
  Avg Individual Trace Accuracy: 42.8%
  Total Tokens: 1,198,234
  Avg Tokens/Question: 79,882
  Total Time: 601.2s
  Avg Time/Question: 40.1s
  Throughput: 1,993.2 tokens/sec

--------------------------------------------------------------------------------
Overall Results (AIME25-I + AIME25-II):
--------------------------------------------------------------------------------
  Total Questions: 30
  Total Correct: 17/30 (56.7%)
  Total Tokens: 2,432,801
  Avg Tokens/Question: 81,093
  Total Time: 1224.6s (20.4 minutes)
  Avg Time/Question: 40.8s
  Overall Throughput: 1,986.8 tokens/sec
================================================================================
```

### Output Files (in `outputs_sc/`)

1. **`traditional_sc_aime25_detailed_TIMESTAMP.json`**
   - Complete trace data for every question
   - Full reasoning paths and extracted answers
   - Perfect for deep analysis

2. **`traditional_sc_aime25_summary_TIMESTAMP.csv`**
   - Spreadsheet-friendly format
   - Open in Excel/Google Sheets
   - One row per question

3. **`traditional_sc_aime25_stats_TIMESTAMP.json`**
   - Aggregate statistics
   - Per-dataset and overall metrics

## Understanding Key Metrics

### Individual Trace Accuracy vs Final Accuracy

```
Individual Accuracy: 45.2%  ← How often a single reasoning path is correct
Final Accuracy: 60.0%       ← How often majority voting is correct

Improvement from SC: +14.8% ← The benefit of self-consistency!
```

**This is the core value of self-consistency**: Even though individual reasoning paths are often wrong, aggregating multiple diverse paths gives you a better answer.

### Vote Distribution

```
Vote Distribution: {'42': 50, '43': 10, '41': 4}
```

Shows how the 64 traces voted:
- 50 traces said "42" ← Winner by majority
- 10 traces said "43"
- 4 traces said "41"

**Interpretation:**
- **High consensus** (e.g., 60/64 votes): Model is very confident
- **Split votes** (e.g., 30/34 split): Model is uncertain, answer might be wrong
- **Low valid traces**: Question may be unclear or very difficult

## Common Use Cases

### Quick Test (2-3 minutes)

```bash
python test_sc_single_question.py
```

### Standard Experiment (~30-40 minutes for both datasets)

```bash
python run_traditional_sc_aime25.py --num_traces 64
```

### Single Dataset Only (~15-20 minutes)

```bash
# Just AIME25-I
python run_traditional_sc_aime25.py --dataset AIME2025-I --num_traces 64

# Just AIME25-II
python run_traditional_sc_aime25.py --dataset AIME2025-II --num_traces 64
```

### First N Questions (For Testing)

```bash
# First 5 questions only
python run_traditional_sc_aime25.py --end_idx 5 --num_traces 32
```

### Different Budget Sizes

```bash
# Low budget (faster, less accurate)
python run_traditional_sc_aime25.py --num_traces 16

# Standard budget
python run_traditional_sc_aime25.py --num_traces 64

# High budget (slower, more accurate)
python run_traditional_sc_aime25.py --num_traces 128
```

### Multiple Experiments

```bash
# Run automated experiments with different configurations
./run_sc_experiments.sh
```

## Expected Results

### AIME Difficulty

AIME (American Invitational Mathematics Examination) is **extremely challenging**:
- Designed for top high school math students
- Qualification exam for US Mathematical Olympiad
- Questions require deep mathematical reasoning

### Typical Performance

| Model Type | Expected Accuracy | Notes |
|------------|------------------|-------|
| Small models (<7B) | 10-25% | Struggle with complexity |
| Medium models (7B-70B) | 20-50% | Reasonable performance |
| Large reasoning models (DeepSeek-R1, etc.) | 40-70% | Best current performance |

### Self-Consistency Improvement

Typical improvement from using SC vs single-path:
- **+5 to +20 percentage points** in accuracy
- Larger improvements when individual traces are diverse
- Diminishing returns above ~64-128 traces

## Analyzing Results

### View Summary CSV

```bash
# Pretty print CSV
cat outputs_sc/traditional_sc_aime25_summary_*.csv | column -t -s, | less -S
```

### View Statistics JSON

```bash
# Pretty print JSON
python -m json.tool outputs_sc/traditional_sc_aime25_stats_*.json
```

### Load in Python for Analysis

```python
import json
import pandas as pd

# Load detailed results
with open('outputs_sc/traditional_sc_aime25_detailed_TIMESTAMP.json', 'r') as f:
    data = json.load(f)

# Load as DataFrame
df = pd.read_csv('outputs_sc/traditional_sc_aime25_summary_TIMESTAMP.csv')

# Analyze
print(f"Overall accuracy: {df['is_correct'].mean():.1%}")
print(f"Avg individual trace accuracy: {df['individual_trace_accuracy'].mean():.1%}")

# By dataset
print(df.groupby('dataset')['is_correct'].mean())
```

## Troubleshooting

### "Out of memory"

```bash
# Reduce number of traces
python run_traditional_sc_aime25.py --num_traces 32

# Or reduce max tokens
python run_traditional_sc_aime25.py --num_traces 64 --max_tokens 65000
```

### "Dataset not found"

```bash
# Install datasets library
pip install datasets

# May need HuggingFace login
huggingface-cli login
```

### "CUDA out of memory" but you have 4 GPUs

```bash
# Make sure tensor_parallel_size is set correctly
python run_traditional_sc_aime25.py --tensor_parallel_size 4
```

### Script runs but very slow

Check GPU utilization:
```bash
# In another terminal
watch -n 1 nvidia-smi

# Should see all 4 GPUs at high utilization
```

## Next Steps

### Compare with Other Voting Methods

The DeepConf framework supports 7 voting methods. Compare traditional SC with confidence-weighted methods:

```bash
# Run with all voting methods
python examples/example_offline.py \
    --dataset your_aime_dataset.jsonl \
    --qid 0 \
    --budget 64
```

### Experiment with Different Models

```bash
# Try different models
python run_traditional_sc_aime25.py \
    --model "meta-llama/Llama-3.1-70B-Instruct" \
    --model_type gpt \
    --num_traces 64
```

### Analyze Failure Cases

Look at questions where SC failed:
1. Check vote distribution (was there consensus?)
2. Look at individual traces (were they all wrong?)
3. Check token length (was reasoning cut off?)

## Questions?

- **Implementation details**: See comments in [run_traditional_sc_aime25.py](run_traditional_sc_aime25.py)
- **Method explanation**: See [README_SC_AIME25.md](README_SC_AIME25.md)
- **DeepConf framework**: See main [README.md](README.md)
