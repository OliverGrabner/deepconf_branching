# run_branching_eval.py Usage Guide

## Overview
`run_branching_eval.py` is an enhanced evaluation script that provides:
- ✅ Detailed formatted output with section headers
- ✅ Answer evaluation against ground truth
- ✅ Accuracy analysis by depth
- ✅ **Confidence-accuracy correlation analysis**
- ✅ Branch statistics and token savings
- ✅ Comprehensive voting results

## Quick Start

### Basic Usage
```bash
python run_branching_eval.py \
    --question "What is 15% of 240?" \
    --ground_truth "36"
```

### With Custom Parameters
```bash
python run_branching_eval.py \
    --question "What is the sum of 25 and 37?" \
    --ground_truth "62" \
    --initial_branches 4 \
    --max_total_branches 12 \
    --confidence_threshold 1.5 \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
```

### On Server with Specific GPUs
```bash
python run_branching_eval.py \
    --question "Calculate 7! (7 factorial)" \
    --ground_truth "5040" \
    --gpus "0,1,2" \
    --initial_branches 8 \
    --max_total_branches 32 \
    --model "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
```

## Command-Line Arguments

### Required
- `--question`: The question to process
- `--ground_truth`: Correct answer for evaluation

### Model Configuration
- `--model`: Model path/name (default: DeepSeek-R1-Distill-Qwen-1.5B)
- `--gpus`: GPU IDs to use (e.g., "0,1,2")

### Branching Parameters
- `--initial_branches`: Number of initial traces (default: 2)
- `--max_total_branches`: Max total traces including branches (default: 6)
- `--confidence_threshold`: Threshold for branching (default: 1.5)
- `--window_size`: Sliding window size (default: 128)
- `--max_tokens`: Max tokens per generation (default: 4000)

### Output
- `--output_dir`: Directory for results (default: outputs)
- `--log_dir`: Directory for logs (default: logs)

## Output Sections

### 1. Configuration
Shows all experiment parameters

### 2. Initialization
Reports model and tokenizer loading

### 3. Generation
Shows progress during trace generation with branching details

### 4. Detailed Results
```
Generation Statistics
  Total traces generated: 6
  Total tokens: 8,450
  ...

Traces by Depth
  Initial: 2 traces
  Branched (depth 1): 4 traces

Branch Details
  Number of branches: 4
  Average prefix length: 650 tokens
  Total tokens saved via prefix caching: ~2,600
  ...

Confidence Analysis
  Average mean confidence: 1.523
  Average min confidence: 0.845
  ...

Branching Statistics
  total_branches: 4
  avg_branch_step: 650.5
  ...
```

### 5. Evaluation Results
```
Accuracy by Depth
  Initial traces:
    Total: 2
    Correct: 1
    Accuracy: 50.0%

  Depth 1 branches:
    Total: 4
    Correct: 3
    Accuracy: 75.0%

Overall Accuracy
  Correct traces: 4/6
  Overall accuracy: 66.7%

Confidence-Accuracy Correlation
  Traces analyzed: 6 (4 correct)

  Correlation coefficients (Pearson r):
    Mean confidence:       +0.6234
    Tail confidence:       +0.5891
    Bottom 10% confidence: +0.4567
    Min confidence:        +0.3421
    Max confidence:        +0.7123

  Average confidence by correctness:
    Correct traces:   1.6234
    Incorrect traces: 1.3456
    Difference:       +0.2778

  ✓ Correct traces have higher confidence on average

  Interpretation:
    Moderate correlation between confidence and accuracy
    Higher confidence → higher likelihood of correctness

Voting Results
  Method                         Answer    Correct
  --------------------------------------------------------
  majority                       36        ✓ YES
  mean_confidence_weighted       36        ✓ YES
  tail_confidence_weighted       36        ✓ YES
  ...

  ✓ 5/7 voting methods got correct answer

Answer Distribution
  ✓ 36: 4 traces (66.7%)
    35: 1 trace (16.7%)
    38: 1 trace (16.7%)
```

### 6. Summary
```
SUMMARY
  Generated: 6 traces (8,450 tokens)
  Time: 45.2s
  Overall accuracy: 66.7%
  Initial trace accuracy: 50.0%
  Average branch accuracy: 75.0%
  Improvement: +25.0%

  ✓ Branching provided significant improvement!
  ✓ Voting successfully found correct answer (5 methods)
```

## Output Files

### Pickle File (full data)
`outputs/branching_eval_TIMESTAMP.pkl`
- Contains complete result object
- All traces with confidences
- Full metadata

### JSON File (summary)
`outputs/branching_eval_TIMESTAMP.json`
- Human-readable summary
- Voting results
- Configuration

### Log File
`logs/branching_eval_TIMESTAMP.log`
- Detailed execution log
- Console output mirrored

## Understanding Correlation

### Correlation Coefficients (r):
- **r > 0.7**: Strong positive correlation
- **r = 0.3 to 0.7**: Moderate positive correlation
- **r < 0.3**: Weak correlation
- **r < 0**: Negative correlation (problematic!)

### What to Look For:

**Good Signs:**
- ✓ Positive correlation (r > 0.4)
- ✓ Correct traces have higher average confidence
- ✓ Confidence difference > 0.1

**Warning Signs:**
- ⚠ Negative correlation (r < 0)
- ⚠ Incorrect traces have higher confidence
- ⚠ Very weak correlation (|r| < 0.2)

### Example Interpretations:

**Strong correlation (r = 0.8):**
```
✓ Confidence is a reliable indicator of correctness
✓ Can use confidence for filtering
✓ High-confidence branches likely to be correct
```

**Weak correlation (r = 0.2):**
```
⚠ Confidence not reliable for this question
⚠ May need more traces
⚠ Consider other filtering methods
```

**Negative correlation (r = -0.5):**
```
⚠ Model is overconfident on wrong answers!
⚠ Do NOT use confidence filtering
⚠ May indicate model has learned wrong patterns
```

## Example Workflows

### 1. Quick Test
```bash
# Simple math question
python run_branching_eval.py \
    --question "What is 20% of 150?" \
    --ground_truth "30" \
    --initial_branches 2 \
    --max_total_branches 4
```

### 2. Confidence Study
```bash
# Generate many traces to study correlation
python run_branching_eval.py \
    --question "Solve: 3x + 5 = 20" \
    --ground_truth "5" \
    --initial_branches 8 \
    --max_total_branches 32 \
    --confidence_threshold 1.2
```

### 3. Branch Effectiveness Test
```bash
# Compare different thresholds
for thresh in 1.0 1.5 2.0 2.5; do
    python run_branching_eval.py \
        --question "What is 15% of 240?" \
        --ground_truth "36" \
        --confidence_threshold $thresh \
        --output_dir outputs/thresh_$thresh
done
```

### 4. Model Comparison
```bash
# Test different models
for model in "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
             "Qwen/Qwen2.5-3B-Instruct"; do
    python run_branching_eval.py \
        --question "What is 15% of 240?" \
        --ground_truth "36" \
        --model "$model" \
        --output_dir "outputs/$(basename $model)"
done
```

## Analyzing Results

### Load Results for Further Analysis
```python
import pickle

# Load results
with open('outputs/branching_eval_TIMESTAMP.pkl', 'rb') as f:
    data = pickle.load(f)

# Access components
question = data['question']
ground_truth = data['ground_truth']
result = data['result']
config = data['config']

# Analyze traces
for trace in result.all_traces:
    print(f"Trace {trace['trace_id']}: {trace.get('extracted_answer')}")
    print(f"  Depth: {trace.get('depth', 0)}")
    print(f"  Mean conf: {np.mean(trace['confs']):.3f}")
    print(f"  Correct: {simple_answer_match(trace['extracted_answer'], ground_truth)}")
```

## Tips

### For Best Correlation Analysis:
1. **Use more traces** (8-32) for statistical significance
2. **Mix correct and incorrect** - need variance
3. **Multiple questions** - average correlations across problems
4. **Different difficulties** - see how correlation changes

### For Branch Studies:
1. **Try different thresholds** (1.0, 1.5, 2.0, 2.5)
2. **Vary branch budgets** - compare efficiency
3. **Compare to uniform sampling** - same total traces

### For Debugging:
1. **Check logs** - detailed execution info
2. **Examine individual traces** - look at wrong answers
3. **Analyze confidence patterns** - identify issues

## Common Issues

### "All traces have same correctness"
- **Cause:** Question too easy or too hard
- **Solution:** Try harder/easier question or more traces

### "Weak correlation"
- **Cause:** Not enough traces, or confidence not predictive
- **Solution:** Generate more traces (32+)

### "Negative correlation"
- **Cause:** Model overconfident on wrong answers
- **Solution:** Review training data, try different model

## Next Steps

After running:
1. **Check correlation** - Is confidence predictive?
2. **Compare depths** - Do branches improve accuracy?
3. **Review voting** - Which method works best?
4. **Analyze patterns** - What makes traces correct/incorrect?

## Summary

**Use this script when you need:**
- ✅ Detailed evaluation against ground truth
- ✅ Confidence-accuracy correlation analysis
- ✅ Branch effectiveness measurement
- ✅ Comprehensive statistics and logging

**The script automatically calculates:**
- Accuracy by depth
- Correlation coefficients (5 types)
- Confidence differences
- Branch improvements
- Voting method effectiveness
- Token savings from prefix caching
