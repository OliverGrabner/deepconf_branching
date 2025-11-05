# Token Counting Fix - Accurate Measurement

## Problem

Previously, branching SC counted **total tokens including inherited prefixes**:

```
Example:
- Trace 0 (original): 8000 tokens
- Trace 8 (branched from 0 at 6000 tokens): 8500 tokens

OLD counting:
  total_tokens = 8000 + 8500 = 16,500 tokens

But 6000 tokens are SHARED! We counted them twice.
```

This inflated token counts unfairly compared to traditional SC.

## Solution

Now we track **two metrics**:

### 1. `total_tokens` (Legacy)
- Total tokens including inherited prefixes
- Kept for compatibility
- **Not accurate for cost comparison**

### 2. `total_tokens_generated` (NEW - Accurate)
- **Only counts newly generated tokens**
- Excludes inherited prefix from parent
- **Use this for fair comparisons!**

## Calculation

For each trace:
```python
tokens_generated = len(trace.token_ids) - trace.generation_started_at_tokens
```

Where:
- `len(trace.token_ids)` = Total tokens in this trace
- `generation_started_at_tokens` = Inherited prefix length

Example:
```
Trace 8 branched from Trace 0 at 6000 tokens:
- Total tokens: 8500
- Started at: 6000
- Tokens generated: 8500 - 6000 = 2500

Trace 0 (original):
- Total tokens: 8000
- Started at: 0
- Tokens generated: 8000 - 0 = 8000

Total generated = 8000 + 2500 = 10,500 tokens (accurate!)
Total (old way) = 8000 + 8500 = 16,500 tokens (double-counted)
```

## What Changed

### 1. Trace Data
Each trace now includes:
```json
{
  "trace_idx": 8,
  "num_tokens": 8500,           // Total including inherited
  "tokens_generated": 2500,     // NEW: Only new tokens
  "generation_started_at_tokens": 6000
}
```

### 2. Output Metrics
```python
output.total_tokens              # Old: includes inherited (16,500)
output.total_tokens_generated    # NEW: only generated (10,500)
output.avg_tokens_per_trace      # Old average
output.avg_tokens_generated_per_trace  # NEW: accurate average
```

### 3. Saved Results
JSON now includes both:
```json
{
  "statistics": {
    "total_tokens": 263488,           // Legacy
    "total_tokens_generated": 175624, // NEW: Use this!
    "avg_tokens_per_trace": 8234.0,
    "avg_tokens_generated_per_trace": 5488.0,
    "throughput_tokens_per_sec": 138.5  // Now based on generated tokens
  }
}
```

## Usage

### Comparing to Traditional SC

**Before (Wrong)**:
```python
traditional_tokens = 32 * 8000 = 256,000
branching_tokens = 263,488  # Double-counted!
# Looks like branching uses MORE tokens (wrong!)
```

**After (Correct)**:
```python
traditional_tokens = 32 * 8000 = 256,000
branching_tokens_generated = 175,624  # Accurate!
# Branching uses FEWER tokens (correct!)
```

### In Your Analysis

Always use `total_tokens_generated`:

```python
# Load results
with open('branching_sc_results.json', 'r') as f:
    data = json.load(f)

# Use generated tokens
for result in data['results']['AIME2025-I']:
    tokens = result['statistics']['total_tokens_generated']  # ✓ Correct
    # NOT: result['statistics']['total_tokens']  # ✗ Wrong
```

### Print Output

When running experiments, you'll see:
```
Total tokens: 263488
Tokens generated (excluding inherited): 175624
```

## Token Efficiency Calculation

**Branching SC efficiency**:
```
Original traces: 8 × 8000 = 64,000 tokens
Branched traces: 24 traces sharing prefixes
Total generated: ~175,000 tokens

Efficiency = 175,000 / (32 × 8000) = 68% of traditional SC
```

This shows the **true advantage** of branching - shared prefixes reduce total generation cost!

## CSV Export

The summary CSV includes both metrics:
```csv
dataset,question_id,total_tokens,total_tokens_generated,...
AIME2025-I,0,263488,175624,...
```

Use `total_tokens_generated` column for analysis.

## Visualization

Token usage plots will use `total_tokens_generated` for accurate comparison.

## Backward Compatibility

- `total_tokens` still exists for old code
- New code should use `total_tokens_generated`
- Both are saved in all outputs

## Summary

✅ **Use `total_tokens_generated`** for:
- Cost comparisons
- Efficiency analysis
- Token budgeting
- Throughput calculations

❌ **Don't use `total_tokens`** for:
- Comparing to traditional SC
- Measuring token efficiency
- Cost analysis

The fix ensures **fair comparisons** between branching SC and traditional SC!
