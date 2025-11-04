# Complete Data Storage & Visualization Summary

## 📊 What's Saved Per Trace

Each trace now stores:

```json
{
  "trace_idx": 8,
  "parent_idx": 0,
  "answer": "70",
  "is_correct": true,                    // ✅ NEW: Per-trace correctness
  "num_tokens": 8500,                     // Total including inherited
  "tokens_generated": 2500,               // ✅ NEW: Only new tokens
  "final_tail_confidence": 12.45,         // ✅ NEW: Final confidence
  "generation_started_at_iteration": 2,
  "generation_started_at_tokens": 6000,
  "confs": [12.3, 13.1, ...]             // Full confidence array
}
```

## 📁 Output Structure

When you run `run_branching_sc_aime25.py`:

```
outputs_sc/
├── branching_sc_aime25_detailed_YYYYMMDD_HHMMSS.json
├── branching_sc_aime25_summary_YYYYMMDD_HHMMSS.csv
├── branching_sc_aime25_stats_YYYYMMDD_HHMMSS.json
└── visualizations/
    ├── PER-PROBLEM VISUALIZATIONS (3 files per question):
    │   ├── summary_AIME2025-I_q0_*.png          # 4-panel overview
    │   ├── genealogy_AIME2025-I_q0_*.png        # Branch tree
    │   ├── confidence_AIME2025-I_q0_*.png       # Confidence evolution
    │   ├── summary_AIME2025-I_q1_*.png
    │   ├── genealogy_AIME2025-I_q1_*.png
    │   ├── confidence_AIME2025-I_q1_*.png
    │   ... (3 per question × 15 questions = 45 files)
    │
    └── DATASET-WIDE VISUALIZATIONS:
        ├── token_usage_*.png                    # Token comparison
        └── accuracy_analysis_*.png              # Accuracy breakdown
```

## 🎨 Visualization Types

### Per-Problem (3 visualizations per question)

#### 1. **Summary Plot** (`summary_*_q*_*.png`)
**4-panel comprehensive overview:**

```
┌────────────────────┬────────────────────┐
│ Final Confidence   │ Token Usage        │
│ vs Correctness     │ (Original vs       │
│ (scatter plot)     │  Branched)         │
├────────────────────┼────────────────────┤
│ Branch Timeline    │ Answer Distribution│
│ (when branches     │ (vote counts)      │
│  occurred)         │                    │
└────────────────────┴────────────────────┘
```

**Shows:**
- Panel 1: Final tail confidence for correct vs incorrect traces
- Panel 2: Box plot of tokens generated (original vs branched)
- Panel 3: When branches occurred and parent confidence
- Panel 4: Answer distribution with ground truth highlighted

#### 2. **Genealogy Graph** (`genealogy_*_q*_*.png`)
**Branch tree visualization:**
- Nodes: Each trace (squares=original, circles=branched)
- Colors: Green=correct, Red=incorrect
- Arrows: Parent→child with iteration labels
- Shows complete branching hierarchy

#### 3. **Confidence Evolution** (`confidence_*_q*_*.png`)
**2-panel confidence over time:**
- Top: All traces with branch points marked (blue vertical lines)
- Bottom: Correct traces only with detailed labels
- Shows how confidence changes during generation

### Dataset-Wide (2 visualizations)

#### 1. **Token Usage** (`token_usage_*.png`)
**2-panel token analysis:**
- Left: Total tokens per question (line plot)
- Right: Histogram of original vs branched trace tokens

#### 2. **Accuracy Analysis** (`accuracy_analysis_*.png`)
**2-panel accuracy breakdown:**
- Left: Bar chart comparing original vs branched trace accuracy
- Right: Per-question correctness (green/red bars)

## 📈 Stored Metrics

### Per-Question Statistics

```json
{
  "question": "Find the sum...",
  "ground_truth": "70",
  "voted_answer": "70",
  "is_correct": true,                    // ✅ Question-level correctness

  "num_traces_generated": 32,
  "num_valid_traces": 32,
  "individual_trace_accuracy": 0.875,    // ✅ Fraction of traces correct

  "statistics": {
    "total_tokens": 263488,              // Includes inherited (legacy)
    "total_tokens_generated": 175624,    // ✅ Only new tokens (USE THIS!)
    "avg_tokens_per_trace": 8234.0,
    "avg_tokens_generated_per_trace": 5488.0,  // ✅ Accurate average
    "generation_time": 1265.5,
    "throughput_tokens_per_sec": 138.5
  },

  "branch_genealogy": {
    "statistics": {
      "total_traces": 32,
      "original_traces": 8,
      "branched_traces": 24,
      "total_branch_events": 24
    }
  }
}
```

### Per-Trace Metrics

```json
{
  "trace_idx": 8,
  "answer": "70",
  "is_correct": true,                    // ✅ Is this trace correct?
  "tokens_generated": 2500,              // ✅ Only new tokens
  "final_tail_confidence": 12.45,        // ✅ Confidence at end
  "parent_idx": 0                        // Parent trace (or null)
}
```

## 🔍 What You Can Analyze

### 1. **Confidence vs Correctness**
```python
# Load results
with open('branching_sc_results.json') as f:
    data = json.load(f)

for result in data['results']['AIME2025-I']:
    for trace in result['full_traces']:
        conf = trace['final_tail_confidence']
        correct = trace['is_correct']
        # Analyze correlation
```

### 2. **Token Efficiency**
```python
original = [t['tokens_generated'] for t in traces if t['parent_idx'] is None]
branched = [t['tokens_generated'] for t in traces if t['parent_idx'] is not None]

print(f"Original avg: {np.mean(original)}")
print(f"Branched avg: {np.mean(branched)}")
print(f"Total generated: {sum(original) + sum(branched)}")
```

### 3. **Branch Quality**
```python
# Which parent traces produced correct children?
for event in result['branch_events']:
    parent_idx = event['parent_trace_idx']
    child_idx = event['child_trace_idx']

    parent_correct = traces[parent_idx]['is_correct']
    child_correct = traces[child_idx]['is_correct']

    print(f"Parent {parent_idx} ({'✓' if parent_correct else '✗'}) "
          f"→ Child {child_idx} ({'✓' if child_correct else '✗'})")
```

### 4. **Per-Question Performance**
```python
for i, result in enumerate(data['results']['AIME2025-I']):
    correct = result['is_correct']
    trace_acc = result['individual_trace_accuracy']
    tokens = result['statistics']['total_tokens_generated']

    print(f"Q{i}: {'✓' if correct else '✗'} "
          f"(trace acc: {trace_acc:.1%}, tokens: {tokens})")
```

## 🎯 Key Metrics to Use

### For Token Comparison
✅ Use: `total_tokens_generated`, `tokens_generated` per trace
❌ Don't use: `total_tokens`, `num_tokens` (includes inherited)

### For Accuracy Analysis
✅ Use: `is_correct` (per trace and per question)
✅ Use: `individual_trace_accuracy` (fraction correct)

### For Confidence Analysis
✅ Use: `final_tail_confidence` (mean of last 2048 tokens)
✅ Use: `confs` array (full token-by-token confidence)

## 📋 Automatic Workflow

When you run the experiment:

1. ✅ **Generate traces** with branching
2. ✅ **Calculate metrics**:
   - Per-trace correctness
   - Per-trace final confidence
   - Token counts (generated only)
3. ✅ **Save JSON** with all data
4. ✅ **Save CSV** summary
5. ✅ **Create visualizations**:
   - 3 plots per question (summary, genealogy, confidence)
   - 2 dataset-wide plots (tokens, accuracy)

**Total output**: 47 files for 15-question dataset
- 1 detailed JSON
- 1 summary CSV
- 1 stats JSON
- 45 per-question images (3 × 15)
- 2 dataset-wide images

## 🚀 Example Analysis Script

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open('outputs_sc/branching_sc_aime25_detailed_*.json') as f:
    data = json.load(f)

# Analyze confidence vs correctness
all_correct_confs = []
all_incorrect_confs = []

for dataset_results in data['results'].values():
    for result in dataset_results:
        for trace in result['full_traces']:
            conf = trace['final_tail_confidence']
            if trace['is_correct']:
                all_correct_confs.append(conf)
            else:
                all_incorrect_confs.append(conf)

# Plot
plt.hist(all_correct_confs, alpha=0.5, label='Correct', bins=30)
plt.hist(all_incorrect_confs, alpha=0.5, label='Incorrect', bins=30)
plt.xlabel('Final Tail Confidence')
plt.ylabel('Count')
plt.legend()
plt.title('Confidence Distribution by Correctness')
plt.savefig('confidence_distribution.png')

print(f"Correct mean: {np.mean(all_correct_confs):.3f}")
print(f"Incorrect mean: {np.mean(all_incorrect_confs):.3f}")
```

## 📝 Summary

**Everything is stored and visualized automatically!**

✅ Accuracy per problem
✅ Accuracy per trace
✅ Final confidence per trace
✅ Token counts (accurate - generated only)
✅ Complete genealogy
✅ 3 visualizations per problem
✅ 2 dataset-wide visualizations

**Just run the experiment and everything is ready for analysis!**
