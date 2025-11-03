# Trace Confidence Visualization

Track and graph tail confidence evolution for each reasoning trace.

## What It Does

- **Tracks confidence** at regular intervals (every 100 tokens)
- **Computes tail confidence** (mean of last 2048 tokens)
- **Graphs evolution** showing how confidence changes during generation
- **Separates correct vs incorrect** traces to identify patterns

## Quick Start

```bash
# Install matplotlib (optional, for graphs)
pip install matplotlib

# Visualize a question
python scripts/visualize_trace_confidence.py --qid 0 --num_traces 16
```

## Output

### 4-Panel Graph

1. **All traces** - Green=correct, Red=incorrect
2. **Correct traces only** - Different shades of green
3. **Incorrect traces only** - Different shades of red
4. **Final confidence histogram** - Distribution comparison

### ASCII Visualization (always available)

```
Trace 5 (✓ CORRECT)
  Answer: 42
  Tokens: 15234
  Final Tail Confidence: 2.345
  Evolution: ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁
  Range: [2.123, 2.567]
```

### Saved Files

- `trace_confidence_qidN_*.pkl` - Full data (reloadable)
- `trace_confidence_qidN_*.json` - Summary (human-readable)
- `trace_confidence_plot_*.png` - Graph (150 DPI)
- `trace_confidence_plot_*_highres.png` - Graph (300 DPI)

## Usage

### Basic

```bash
python scripts/visualize_trace_confidence.py \
    --qid 0 \
    --dataset AIME2025-I \
    --num_traces 16
```

### Load Previous Results

```bash
python scripts/visualize_trace_confidence.py \
    --load outputs_sc/trace_confidence_qid0_*.pkl
```

### Configuration

```bash
--tail_size 2048     # Size of tail window (default: 2048)
--step_size 100      # Sample every N tokens (default: 100)
--no_plot            # Skip matplotlib plots (ASCII only)
```

## Interpretation

### Patterns to Look For

**High confidence → Stays high**
```
▆▇████████████████  → Usually correct
```

**Low confidence → Stays low**
```
▁▂▃▂▁▂▃▂▁▂▃▂▁  → Usually incorrect
```

**High → Drops**
```
████▇▆▅▄▃▂▁  → Often incorrect (model loses confidence)
```

### Key Questions

1. **Do correct traces have higher confidence?** → Check histogram (Plot 4)
2. **When do traces diverge?** → Check where correct/incorrect separate (Plot 1)
3. **Does confidence predict correctness?** → Compare distributions

## Use Cases

### Understand a Difficult Question

```bash
python scripts/visualize_trace_confidence.py --qid 14 --num_traces 32
```

Look at patterns to see why the model struggled.

### Compare Easy vs Hard

```bash
python scripts/visualize_trace_confidence.py --qid 0 --num_traces 16  # Easy
python scripts/visualize_trace_confidence.py --qid 14 --num_traces 16 # Hard
```

Compare separation patterns.

### Research Questions

- Can we use confidence for early stopping?
- How does confidence correlate with correctness?
- When do reasoning paths diverge?

## Custom Analysis

Load `.pkl` files for custom analysis:

```python
import pickle

with open('outputs_sc/trace_confidence_qid0_*.pkl', 'rb') as f:
    data = pickle.load(f)

# Access trace data
for trace in data['traces']:
    print(f"Trace {trace['trace_id']}: {trace['answer']}")
    for point in trace['confidence_evolution']:
        print(f"  Position {point['position']}: {point['tail_confidence']:.3f}")
```

## Performance

- **Time**: Same as normal SC (generation dominates)
- **Memory**: ~10-50 MB per question
- **Processing**: Extra 1-2 seconds per trace

## Notes

- Works with or without matplotlib (ASCII fallback)
- Data is reusable - generate once, visualize many times
- High-res plots ready for publications (300 DPI)
