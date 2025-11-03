# Branching Self-Consistency

Dynamic trace branching during generation to efficiently explore reasoning paths.

## Overview

**Problem**: Traditional self-consistency generates N independent traces from start to finish. This can be inefficient since we're exploring diverse paths even in the verification phase (last 10-25% of reasoning).

**Solution**: Branching self-consistency starts with S traces and dynamically branches high-confidence traces during generation to reach M traces by ~75% of average generation length.

## Key Hypothesis

- In reasoning traces, the last 10-25% is mostly verification with few new ideas
- Better to explore diverse paths early, then let them converge naturally
- High-confidence traces are better candidates for branching

## Algorithm

### Parameters

```python
start_traces = 8           # Initial parallel traces
max_traces = 32            # Maximum capacity
selected_percent = 0.60    # Top 60% by confidence eligible for branching
n_iterations = 10          # Number of check points (e.g., every 10% of generation)
branch_goal = 0.75         # Want to reach max_traces by 75% of avg length
average_tokens = 8000      # Historical average from previous runs
```

### Schedule

```python
stride = (branch_goal * average_tokens) / n_iterations
branches_per_iteration = ceil((max_traces - start_traces) / n_iterations)
```

Example:
- Average tokens = 8000
- Branch goal = 75% → deadline at 6000 tokens
- 10 iterations → stride = 600 tokens
- Need 24 branches (32 - 8) over 10 iterations → 3 per iteration

### Process

1. **Initialization**: Start with S=8 traces
2. **At each iteration** (every 600 tokens):
   - All traces generate next chunk synchronously
   - Compute tail confidence (mean of last 2048 tokens) for all traces
   - Rank traces by tail confidence
   - Select top 60% as branch candidates
   - Uniformly sample to create ~3 new branches
   - Children inherit parent's prefix and continue from branch point
3. **After 75%**: Stop branching, continue generation to completion
4. **Voting**: Simple majority vote on all final answers

### Key Features

- **Synchronous generation**: All traces at same token position at each stride
- **Uniform sampling**: Among top 60%, sample uniformly (allows same trace to spawn multiple children)
- **No early stopping**: All traces generate to completion
- **Genealogy tracking**: Complete parent-child relationships recorded

## Usage

### Step 1: Compute Historical Statistics

Run once to bootstrap average token counts:

```bash
python scripts/compute_historical_stats.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --num_samples 2 \
    --dataset AIME2025-I
```

**Output**: `historical_stats/aime25_token_stats_latest.json`

This contains average tokens per question from 2-trace runs.

### Step 2: Run Branching Self-Consistency

```bash
python scripts/run_branching_sc_aime25.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --start_traces 8 \
    --max_traces 32 \
    --selected_percent 0.60 \
    --n_iterations 10 \
    --branch_goal 0.75 \
    --historical_stats historical_stats/aime25_token_stats_latest.json \
    --dataset AIME2025-I
```

### Step 3: Test on Single Question

Quick test on one question:

```bash
python scripts/test_branching_single_question.py \
    --qid 0 \
    --start_traces 4 \
    --max_traces 8 \
    --n_iterations 5 \
    --average_tokens 8000
```

## Output Files

### Detailed Results JSON

`outputs_sc/branching_sc_aime25_detailed_YYYYMMDD_HHMMSS.json`

Contains:
- All traces with text, answers, token counts
- Branch genealogy (parent-child tree)
- Branch events (chronological history)
- Statistics and timing

### Summary CSV

`outputs_sc/branching_sc_aime25_summary_YYYYMMDD_HHMMSS.csv`

Quick summary with:
- Correctness per question
- Number of traces (original vs branched)
- Branch events count
- Tokens and timing

### Aggregate Statistics

`outputs_sc/branching_sc_aime25_stats_YYYYMMDD_HHMMSS.json`

Per-dataset and overall accuracy, tokens, throughput, branching stats.

## Analysis

### Branch Genealogy

The output includes complete genealogy:

```python
result.branch_genealogy = {
    'tree': {
        0: {'parent': None, 'children': [8, 10]},      # Original trace 0 spawned 8, 10
        1: {'parent': None, 'children': []},            # Original trace 1
        8: {'parent': 0, 'children': [16]},             # Trace 8 from parent 0, spawned 16
        ...
    },
    'events': [
        {
            'iteration': 2,
            'parent_trace_idx': 0,
            'child_trace_idx': 8,
            'branch_point_tokens': 1200,
            'parent_tail_confidence': 12.45
        },
        ...
    ],
    'statistics': {
        'total_traces': 32,
        'original_traces': 8,
        'branched_traces': 24,
        'total_branch_events': 24
    }
}
```

### Research Questions

Use genealogy to analyze:
1. **Do branched traces help?** Compare accuracy of original vs branched traces
2. **When do branches diverge?** Look at branch points and final answers
3. **Which traces are most valuable?** Trace ancestors of correct answers
4. **Is confidence predictive?** Correlate parent confidence with child correctness

## Comparison to Traditional SC

| Metric | Traditional SC | Branching SC |
|--------|---------------|--------------|
| **Traces** | N independent | S → M dynamically |
| **Exploration** | Uniform throughout | Concentrated early |
| **Efficiency** | Same cost | Potentially better token efficiency |
| **Diversity** | High | High early, converges later |
| **Genealogy** | None | Full parent-child tracking |

## Implementation Details

### Core Components

1. **`deepconf/branching.py`**: BranchingManager class
   - Handles branching schedule
   - Selects high-confidence traces
   - Tracks genealogy

2. **`deepconf/wrapper.py`**: `_deepthink_branching()` method
   - Chunk-based synchronous generation
   - Updates trace states incrementally
   - Creates branches at each iteration

3. **`deepconf/outputs.py`**: Extended with branching fields
   - `branch_events`: Chronological history
   - `branch_genealogy`: Parent-child tree
   - `branching_config`: Parameters used

### Technical Notes

- **Prefix caching**: Enabled by default, helps with branched traces sharing prefixes
- **Chunk generation**: Uses `max_tokens` parameter to generate in strides
- **Confidence computation**: Uses existing tail confidence infrastructure
- **Token counting**: Accurate per-trace token counts including branched portions

## Parameters Tuning

### `start_traces`

- **Lower** (4-8): Faster initial phase, more aggressive branching
- **Higher** (16-32): More initial diversity, less branching needed

### `max_traces`

- Should match your token budget
- Consider: total_tokens ≈ max_traces × avg_tokens_per_trace

### `selected_percent`

- **0.50**: Top half eligible (more diverse branching)
- **0.60**: Top 60% (balanced)
- **0.80**: Top 80% (include lower confidence traces)

### `n_iterations`

- **5-10**: Fewer, larger strides (faster, less frequent decisions)
- **15-20**: More, smaller strides (finer-grained control)

### `branch_goal`

- **0.70**: Stop branching at 70% (more late-stage diversity)
- **0.75**: Stop at 75% (balanced)
- **0.80**: Stop at 80% (early convergence)

## Known Limitations

1. **Stride granularity**: Fixed stride size may not align perfectly with reasoning boundaries
2. **Historical dependency**: Requires historical data to estimate average tokens
3. **No adaptive stopping**: All traces generate to max_tokens (could add early stop)
4. **Synchronous assumption**: All traces move in lockstep (alternative: asynchronous branching)

## Future Work

- **Adaptive stride**: Adjust stride based on confidence changes
- **Semantic branching**: Branch at natural breakpoints (after claims, proofs, etc.)
- **Confidence-weighted voting**: Use confidence in final voting
- **Multi-level branching**: Allow branched traces to branch again more aggressively
- **Early stopping**: Stop traces that reach low confidence plateau

## Citation

If you use branching self-consistency in your research, please cite:

```bibtex
@misc{branching_sc_2025,
  title={Branching Self-Consistency: Dynamic Trace Exploration for Efficient Reasoning},
  author={Your Name},
  year={2025}
}
```

## References

- Wang et al. (2022): [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)
- Original self-consistency paper establishing the baseline technique
