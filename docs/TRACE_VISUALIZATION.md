# Trace Confidence Visualization Guide

This tool tracks and graphs the **tail confidence evolution** of each trace as it generates its answer.

## What It Does

For each reasoning trace:
1. **Tracks confidence at regular intervals** (every 100 tokens by default)
2. **Computes tail confidence** (mean of last N tokens, default N=2048)
3. **Graphs the evolution** showing how confidence changes during generation
4. **Separates correct vs incorrect traces** to identify patterns

## Quick Start

### Install matplotlib (optional, for graphs)

```bash
pip install matplotlib
```

### Run on a Single Question

```bash
# Generate 16 traces and visualize confidence evolution
python visualize_trace_confidence.py \
    --qid 0 \
    --dataset AIME2025-I \
    --num_traces 16
```

### Load and Visualize Previously Saved Results

```bash
# Visualize from saved pickle file
python visualize_trace_confidence.py \
    --load outputs_sc/trace_confidence_qid0_TIMESTAMP.pkl
```

## Output

### 1. Matplotlib Graphs (4 plots)

**Plot 1: All Traces**
- Green lines = correct traces
- Red lines = incorrect traces
- Shows overall patterns

**Plot 2: Correct Traces Only**
- Different shades of green for each trace
- Shows consensus among correct reasoning

**Plot 3: Incorrect Traces Only**
- Different shades of red for each trace
- Shows where model goes wrong

**Plot 4: Final Confidence Distribution**
- Histogram comparing final confidence of correct vs incorrect traces
- Tests hypothesis: "Do correct traces have higher final confidence?"

### 2. ASCII Visualization (always available)

Shows sparkline representation of confidence evolution:
```
Trace 5 (✓ CORRECT)
  Answer: 42
  Tokens: 15234
  Final Tail Confidence: 2.345
  Evolution: ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁
  Range: [2.123, 2.567]
```

### 3. Saved Files

- `trace_confidence_qidN_TIMESTAMP.pkl` - Full data (can reload for visualization)
- `trace_confidence_qidN_TIMESTAMP.json` - Summary (human-readable)
- `trace_confidence_plot_TIMESTAMP.png` - Standard resolution graph
- `trace_confidence_plot_TIMESTAMP_highres.png` - High resolution (300 DPI)

## Command-Line Options

### Basic Options

```bash
python visualize_trace_confidence.py \
    --dataset AIME2025-I \        # or AIME2025-II
    --qid 0 \                     # Question ID (0-14 for each dataset)
    --num_traces 16 \             # Number of reasoning traces
    --temperature 1.0             # Sampling temperature
```

### Confidence Tracking Parameters

```bash
python visualize_trace_confidence.py \
    --tail_size 2048 \            # Size of tail window (default: 2048)
    --step_size 100               # Sample every N tokens (default: 100)
```

**Explanation:**
- **tail_size**: How many recent tokens to average for confidence
  - Larger = smoother, more stable
  - Smaller = more sensitive to local changes
- **step_size**: How often to record confidence
  - Smaller = more detailed graph, more data
  - Larger = coarser graph, less data

### Output Options

```bash
python visualize_trace_confidence.py \
    --output_dir outputs_sc \     # Where to save results
    --no_plot                     # Skip matplotlib plots (ASCII only)
```

## Example Use Cases

### Case 1: Understand Why a Question Failed

```bash
# Generate traces for a difficult question
python visualize_trace_confidence.py \
    --qid 5 \
    --dataset AIME2025-I \
    --num_traces 32

# Look at the graphs:
# - Are incorrect traces consistently low confidence?
# - Do they start confident then drop off?
# - Is there a clear separation between correct/incorrect?
```

### Case 2: Compare Confidence Patterns

```bash
# Run on an easy question
python visualize_trace_confidence.py --qid 0 --num_traces 16

# Run on a hard question
python visualize_trace_confidence.py --qid 14 --num_traces 16

# Compare the graphs:
# - Easy questions should show clear separation
# - Hard questions might show overlapping confidence
```

### Case 3: Quick Check with Fewer Traces

```bash
# Fast test with just 8 traces
python visualize_trace_confidence.py \
    --qid 0 \
    --num_traces 8 \
    --dataset AIME2025-I
```

### Case 4: High Detail Analysis

```bash
# More traces, finer sampling
python visualize_trace_confidence.py \
    --qid 0 \
    --num_traces 64 \
    --step_size 50 \              # Sample every 50 tokens (more detailed)
    --tail_size 1024              # Smaller tail window (more sensitive)
```

## Understanding the Graphs

### Patterns to Look For

**1. Confidence Trajectory**
```
High confidence → Stays high → Correct answer
  ▆▇████████████████

Low confidence → Stays low → Incorrect answer
  ▁▂▃▂▁▂▃▂▁▂▃▂▁

High → Drops → Often incorrect
  ████▇▆▅▄▃▂▁
```

**2. Separation Between Correct/Incorrect**
- **Good separation**: Model knows when it's right
- **Poor separation**: Model can't distinguish correct reasoning
- **Overlap**: Question is genuinely difficult

**3. Evolution Patterns**
- **Stable**: Confidence stays consistent
- **Improving**: Confidence increases (model "figures it out")
- **Degrading**: Confidence decreases (model gets confused)

### Key Questions to Ask

1. **Do correct traces have higher final confidence?**
   - Look at Plot 4 (histogram)
   - If yes: confidence is a good signal

2. **Do confidence patterns differ during generation?**
   - Look at Plots 2 & 3
   - Some traces might start confident then degrade

3. **Is there a "tipping point" where traces diverge?**
   - Look at Plot 1
   - Early in generation: all similar
   - Later: correct and incorrect traces separate

## Integration with Main SC Script

You can add visualization to specific questions in your main experiment:

```bash
# Run full experiment
python run_traditional_sc_aime25.py --num_traces 64

# Then visualize interesting questions
python visualize_trace_confidence.py --qid 0 --num_traces 16
python visualize_trace_confidence.py --qid 5 --num_traces 16
python visualize_trace_confidence.py --qid 10 --num_traces 16
```

## Performance Notes

### Time and Memory

- **Generation**: Same as normal SC (most of the time)
- **Processing**: Extra ~1-2 seconds per trace
- **Storage**: ~10-50 MB per question (depends on traces and tokens)

### Recommendations

- Use **fewer traces** for quick exploration (8-16)
- Use **more traces** for publication-quality analysis (32-64)
- **Save the .pkl files** - you can reload and re-visualize without regenerating

## Advanced: Custom Analysis

The saved `.pkl` files contain full data. Load them for custom analysis:

```python
import pickle
import numpy as np

# Load data
with open('outputs_sc/trace_confidence_qid0_TIMESTAMP.pkl', 'rb') as f:
    data = pickle.load(f)

# Access trace data
for trace in data['traces']:
    print(f"Trace {trace['trace_id']}:")
    print(f"  Answer: {trace['answer']}")
    print(f"  Correct: {trace['is_correct']}")

    # Full confidence evolution
    for point in trace['confidence_evolution']:
        position = point['position']
        confidence = point['tail_confidence']
        print(f"    Position {position}: confidence {confidence:.3f}")

# Custom analysis
correct_traces = [t for t in data['traces'] if t['is_correct']]
incorrect_traces = [t for t in data['traces'] if not t['is_correct']]

# Compare confidence at different points
# ... your custom analysis here ...
```

## Troubleshooting

### "matplotlib not installed"

```bash
pip install matplotlib
```

Or run with `--no_plot` for ASCII-only visualization.

### "Out of memory"

Reduce number of traces:
```bash
python visualize_trace_confidence.py --num_traces 8
```

### Graphs look cluttered

Too many traces on one plot. Options:
1. Reduce `--num_traces`
2. Look at Plot 2 and Plot 3 (separate correct/incorrect)
3. Use the `.pkl` file to create custom plots

### "No confidence data available"

The model didn't generate logprobs. Make sure:
- `logprobs=20` is set in SamplingParams (already done in script)
- vLLM version supports logprobs

## Examples of Research Questions

Use this tool to answer:

1. **Does confidence predict correctness?**
   - Generate 64 traces
   - Check correlation between final confidence and correctness

2. **When do traces diverge?**
   - Look for the token position where correct and incorrect separate
   - Early divergence = model "knows" early
   - Late divergence = initially similar reasoning

3. **Can we use confidence for early stopping?**
   - If correct traces consistently have higher confidence
   - Could stop generation early for low-confidence traces

4. **How does question difficulty affect confidence?**
   - Compare easy vs hard questions
   - Hard questions might show less separation

## Citation

If you use this visualization in research, cite both:

1. Original SC paper (Wang et al., 2022)
2. Your analysis of confidence patterns

---

**Ready to visualize?**

```bash
pip install matplotlib
python visualize_trace_confidence.py --qid 0 --num_traces 16 --dataset AIME2025-I
```

Then check `outputs_sc/` for the generated graphs!
