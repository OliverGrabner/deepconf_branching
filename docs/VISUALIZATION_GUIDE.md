# Branching SC Visualization Guide

## Overview

The branching SC system now includes comprehensive visualization infrastructure that automatically generates graphs showing:
1. **Branch genealogy trees** - Visual representation of when/where branches occurred
2. **Confidence evolution** - How confidence changes over time with branch points marked
3. **Token usage analysis** - Compare original vs branched traces
4. **Accuracy breakdown** - Success rates by trace type

## Automatic Visualization

When you run `run_branching_sc_aime25.py`, visualizations are **automatically generated** at the end!

```bash
python scripts/run_branching_sc_aime25.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --start_traces 8 \
    --max_traces 32 \
    --historical_stats historical_stats/aime25_token_stats_latest.json
```

**Output locations**:
- Results: `outputs_sc/branching_sc_aime25_detailed_YYYYMMDD_HHMMSS.json`
- Visualizations: `outputs_sc/visualizations/`

## Visualization Types

### 1. Branch Genealogy Tree

**File**: `genealogy_<dataset>_q<id>_<timestamp>.png`

Shows:
- **Nodes**: Each trace (squares = original, circles = branched)
- **Colors**: Green = correct answer, Red = incorrect answer
- **Arrows**: Parent → child relationships
- **Labels**: Branch iteration numbers on arrows

**Example**:
```
Original Trace 0 (square, green)
    ↓ iter 2
Branched Trace 8 (circle, green)
    ↓ iter 5
Branched Trace 16 (circle, red)
```

### 2. Confidence Evolution

**File**: `confidence_<dataset>_q<id>_<timestamp>.png`

Two plots:
- **Top**: All traces with branch points marked (vertical blue lines)
- **Bottom**: Correct traces only with detailed labels

Shows how confidence changes during generation and where branches occur.

### 3. Token Usage

**File**: `token_usage_<timestamp>.png`

Two plots:
- **Left**: Total tokens per question (line plot)
- **Right**: Histogram comparing original vs branched trace token counts

### 4. Accuracy Analysis

**File**: `accuracy_analysis_<timestamp>.png`

Two plots:
- **Left**: Bar chart comparing original vs branched trace accuracy
- **Right**: Per-question correctness (green/red bars)

## Manual Visualization

If auto-visualization fails or you want to re-generate:

```bash
python scripts/visualize_branching_results.py \
    --results outputs_sc/branching_sc_aime25_detailed_20250103_120000.json \
    --output_dir visualizations/
```

### Visualize Specific Question

```bash
python scripts/visualize_branching_results.py \
    --results outputs_sc/branching_sc_aime25_detailed_20250103_120000.json \
    --output_dir visualizations/ \
    --question_id 5
```

## Data Saved for Visualization

The experiment script saves:

### 1. Full Trace Data (`full_traces`)
```json
{
  "trace_idx": 8,
  "parent_idx": 0,
  "answer": "510",
  "num_tokens": 8234,
  "confs": [12.3, 13.1, 12.8, ...],  // For confidence plots
  "generation_started_at_iteration": 2,
  "generation_started_at_tokens": 1200
}
```

### 2. Branch Genealogy
```json
{
  "tree": {
    "0": {"parent": null, "children": [8, 10]},
    "8": {"parent": 0, "children": [16]}
  },
  "events": [
    {
      "iteration": 2,
      "parent_trace_idx": 0,
      "child_trace_idx": 8,
      "branch_point_tokens": 1200,
      "parent_tail_confidence": 12.45
    }
  ]
}
```

### 3. Statistics
- Total tokens per question
- Original vs branched trace counts
- Accuracy by trace type
- Timing information

## Requirements

Install visualization dependencies:

```bash
pip install matplotlib networkx
```

If not installed, the script will still save data but skip visualizations.

## Interpreting Results

### Genealogy Trees

**Good signs**:
- Correct traces (green) branching from other correct traces
- Branch points clustered early (before 75% deadline)
- Multiple children from high-confidence parents

**Warning signs**:
- All incorrect traces (many red nodes)
- Branching late (after 75% deadline)
- No branching occurred (hit max early)

### Confidence Evolution

**Look for**:
- Separation between correct (green) and incorrect (red) traces
- Branch points (blue lines) occurring in high-confidence regions
- Correct traces maintaining high confidence throughout

### Token Efficiency

**Compare**:
- Original traces often have full generation from start
- Branched traces share prefixes → potentially fewer total tokens
- Check if branched traces are more token-efficient

### Accuracy Analysis

**Key metrics**:
- Original trace accuracy: How good were starting traces?
- Branched trace accuracy: Did branching improve quality?
- Per-question: Which problems benefited from branching?

## Example Workflow

1. **Run experiment**:
```bash
python scripts/run_branching_sc_aime25.py \
    --start_traces 8 --max_traces 32 \
    --historical_stats historical_stats/aime25_token_stats_latest.json
```

2. **Check outputs**:
```bash
ls outputs_sc/visualizations/
```

3. **Analyze**:
   - Open genealogy trees for interesting questions
   - Check confidence evolution for separation patterns
   - Compare token usage vs traditional SC
   - Examine accuracy breakdown

4. **Re-visualize specific questions**:
```bash
python scripts/visualize_branching_results.py \
    --results outputs_sc/branching_sc_aime25_detailed_*.json \
    --question_id 5
```

## Troubleshooting

### "matplotlib not installed"
```bash
pip install matplotlib networkx
```

### "Visualization failed"
The script will print the error but still save all data. You can:
1. Fix the error
2. Run visualization script manually
3. Load JSON and create custom plots

### Missing confidence data
Ensure `full_traces` is saved in results JSON. The experiment script now automatically includes this.

### Graph layout looks messy
The genealogy tree uses automatic layout. For complex trees (many branches), nodes may overlap. Try visualizing specific questions instead of all at once.

## Advanced: Custom Analysis

Load saved data for custom analysis:

```python
import json
import matplotlib.pyplot as plt

# Load results
with open('outputs_sc/branching_sc_aime25_detailed_*.json', 'r') as f:
    data = json.load(f)

# Access data
for dataset_name, results in data['results'].items():
    for result in results:
        # Branch genealogy
        tree = result['branch_genealogy']['tree']
        events = result['branch_events']

        # Full traces with confidences
        traces = result['full_traces']

        # Your custom analysis here
        ...
```

## Summary

**Automatic**: Visualizations generated when running experiments
**Manual**: Can re-generate or customize using visualization script
**Complete**: All data saved including confidences, genealogy, statistics
**Flexible**: Can visualize all questions or specific ones

The visualization system makes it easy to understand and analyze branching SC behavior!
