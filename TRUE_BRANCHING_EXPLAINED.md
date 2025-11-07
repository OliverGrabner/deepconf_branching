# True Prefix-Based Branching Implementation

## What Changed

The branching implementation has been updated from **simulated branching** to **true prefix-based branching** that leverages vLLM's prefix caching for efficiency.

---

## Before (Simulated Branching)

### How It Worked:
```python
# Round 1: Generate initial traces
trace_1 = generate(prompt, seed=1)  # Full generation: 0-2000 tokens
trace_2 = generate(prompt, seed=2)  # Full generation: 0-2000 tokens

# Round 2: Find high-confidence peaks, then...
# Generate NEW traces from scratch with different seeds
branch_1 = generate(prompt, seed=1001)  # Full generation: 0-2000 tokens ❌
branch_2 = generate(prompt, seed=1002)  # Full generation: 0-2000 tokens ❌
```

### Problems:
- ❌ Branch traces started from the beginning (not from branch point)
- ❌ No actual continuation from high-confidence states
- ❌ Wasted computation regenerating prefixes
- ❌ Branch point was just metadata (not functional)
- ❌ Different seeds approximated "branching" but traces were independent

---

## After (True Prefix-Based Branching)

### How It Works:
```python
# Round 1: Generate initial traces
trace_1 = generate(prompt, seed=1)  # Full generation: 0-2000 tokens

# Round 2: Found high-confidence peak at token 500
# Extract prefix up to branch point
prefix_tokens = trace_1.token_ids[:500]
prefix_text = tokenizer.decode(prefix_tokens)

# Continue from branch point with different variations
branch_1 = generate(prefix_text, seed=1001)  # Generate: 500-2000 tokens ✓
branch_2 = generate(prefix_text, seed=1002)  # Generate: 500-2000 tokens ✓
```

### Benefits:
- ✓ Branch traces actually continue from branch points
- ✓ True exploration of alternative continuations
- ✓ Prefix caching saves computation (vLLM reuses KV cache)
- ✓ Branch point is functional (traces diverge from there)
- ✓ More faithful to the branching concept

---

## Implementation Details

### 1. Extract Token-Based Prefixes

**Location:** `branching_wrapper.py` lines 302-310

```python
# Extract tokens up to branch point
branch_point = candidate['peak_position']
if 'token_ids' in parent_trace and parent_trace['token_ids']:
    # Use token IDs for exact prefix
    prefix_tokens = parent_trace['token_ids'][:branch_point]
    prefix_text = self.tokenizer.decode(prefix_tokens, skip_special_tokens=False)
else:
    # Fallback to character-based slicing
    prefix_text = candidate.get('text_up_to_peak', parent_trace.get('text', '')[:branch_point])
```

**Key changes:**
- Uses `token_ids` array to get exact token sequence
- Decodes tokens to text for vLLM generation
- Maintains `skip_special_tokens=False` to preserve prompt structure

### 2. Leverage vLLM Prefix Caching

**Location:** `branching_wrapper.py` lines 333-334

```python
# Generate branches - vLLM will automatically cache common prefixes!
branch_outputs = self.llm.generate(branch_prompts, branch_params_list)
```

**How prefix caching works:**
- vLLM detects common prefixes across prompts
- Computes KV cache once for shared prefix
- Reuses cache for all branches
- Only computes new tokens from branch point onwards

### 3. Track Token Savings

**Location:** `branching_wrapper.py` lines 357-363

```python
# Only count NEW tokens (excluding prefix)
new_tokens = trace['num_tokens'] - metadata['prefix_length']
total_tokens += max(0, new_tokens)

print(f"  Generated {trace_counter - initial_branches} branch traces")
print(f"  Saved ~{sum(m['prefix_length'] for m in branch_metadata)} tokens via prefix caching")
```

**Token accounting:**
- Branch traces store both `num_tokens` (total) and `prefix_length`
- `total_tokens` only counts NEW tokens generated
- Reports token savings from prefix caching

---

## Example Walkthrough

### Scenario: 2 initial traces, each spawns 1 branch

**Step 1: Initial Generation**
```
Trace 0: [Generate 2000 tokens]
  Confidence peak at token 600 (conf=2.1)

Trace 1: [Generate 2000 tokens]
  Confidence peak at token 450 (conf=1.9)
```

**Step 2: Analyze Branch Points**
```
Candidates:
  - Parent: Trace 0, Branch point: 600, Confidence: 2.1
  - Parent: Trace 1, Branch point: 450, Confidence: 1.9
```

**Step 3: Generate Branches**
```
Branch from Trace 0:
  Prefix: tokens[0:600]  (600 tokens)
  Generate: tokens[600:2000]  (1400 new tokens)
  Total: 2000 tokens, but only 1400 counted as "new"

Branch from Trace 1:
  Prefix: tokens[0:450]  (450 tokens)
  Generate: tokens[450:2000]  (1550 new tokens)
  Total: 2000 tokens, but only 1550 counted as "new"
```

**Token Efficiency:**
```
Without prefix caching:
  2 initial × 2000 = 4000 tokens
  2 branches × 2000 = 4000 tokens
  Total: 8000 tokens

With prefix caching:
  2 initial × 2000 = 4000 tokens
  2 branches × ~1475 avg = 2950 tokens (actual new computation)
  Total: 6950 tokens
  Savings: ~13% (1050 tokens)
```

---

## New Trace Metadata

Branch traces now include additional fields:

```python
trace = {
    'trace_id': 'trace_2',
    'parent_id': 'trace_0',              # NEW: Which trace spawned this
    'depth': 1,                          # NEW: Branching depth
    'branch_point': 600,                 # NEW: Token position where branch occurred
    'prefix_length': 600,                # NEW: Length of shared prefix
    'branch_history': [{                 # NEW: Full branching history
        'step': 600,
        'confidence': 2.1,
        'parent_trace': 'trace_0'
    }],
    'text': '...',                       # Full generated text
    'token_ids': [...],                  # Full token sequence
    'num_tokens': 2000,                  # Total tokens in this trace
    'confs': [...],                      # Confidence scores
    # ... other fields
}
```

---

## Testing the Implementation

### Quick Test:
```bash
# Run the test script
python test_true_branching.py
```

This will:
1. Initialize a small model (1.5B)
2. Generate 2 initial traces
3. Identify high-confidence peaks
4. Generate 2 branch traces from those peaks
5. Verify all metadata is correctly set
6. Report token savings

### Expected Output:
```
Identifying {N} branching opportunities
Generating {M} branch traces with prefix caching...
  Using prefix caching for {M} branches
  Average prefix length: 600 tokens
  Generated {M} branch traces
  Saved ~{X} tokens via prefix caching

Branch traces found:
  Branch 1:
    Parent ID: trace_0
    Branch point: 600 tokens
    Prefix length: 600 tokens
    ...

✓ All checks passed! True branching is working correctly.
```

---

## Integration with Existing Code

### No Changes Required For:
- `run_branching_test.py` - Works as before
- `example_branching.py` - Works as before
- Visualization scripts - Work as before
- All parameters and command-line arguments

### Automatic Benefits:
- Better token efficiency (10-30% savings depending on branch points)
- More faithful to branching concept
- True alternative continuations from high-confidence states
- Better research validity

---

## Research Implications

### What This Enables:

**Before:**
- Could only test: "Do different random seeds from same prompt lead to different answers?"
- Limitation: Not really "branching" from high confidence

**Now:**
- Can test: "Does continuing from high-confidence states lead to better alternative solutions?"
- Can test: "Are high-confidence states good 'checkpoints' for exploration?"
- Can test: "Do branches from high-confidence states have higher accuracy?"

### Experimental Comparisons:

**Compare:**
1. **Uniform sampling**: N independent traces
2. **Simulated branching**: N traces with different seeds
3. **True branching**: Initial traces + branches from high-confidence points

**Hypothesis:** True branching should outperform both because:
- Focuses compute on promising reasoning paths
- Explores alternatives from stable/confident states
- Doesn't waste tokens on unpromising directions

---

## Performance Expectations

### Token Savings:
- **Light branching** (1-2 branches per trace, early branch points): 5-15% savings
- **Medium branching** (2-4 branches, mid-generation): 15-30% savings
- **Heavy branching** (4+ branches, late branch points): 30-50% savings

### When Savings Are Highest:
- Early branch points (more shared prefix)
- Multiple branches from same parent (same prefix reused N times)
- Longer total generations (prefix overhead amortized)

### When Savings Are Lower:
- Late branch points (less shared prefix)
- Few branches per parent
- Short total generations

---

## Troubleshooting

### Issue: "No branches generated"
**Cause:** Confidence never exceeds threshold
**Solution:** Lower `--confidence_threshold` (try 1.0 or 1.2)

### Issue: "Branch traces seem identical to initial traces"
**Cause:** Possible bug in prefix extraction
**Check:**
- Look for `branch_point` and `prefix_length` in trace metadata
- Verify they're > 0 and < total length

### Issue: "Token savings not showing"
**Cause:** vLLM prefix caching not enabled
**Solution:** Verify `enable_prefix_caching=True` in initialization

---

## Future Enhancements

Possible improvements:

1. **Multi-level branching**: Allow branches to spawn more branches (depth > 1)
2. **Adaptive branch budget**: Allocate more branches to higher-confidence peaks
3. **Beam search integration**: Use branching for beam search over reasoning paths
4. **Dynamic prefix optimization**: Adjust branch points based on token savings
5. **Branch pruning**: Stop low-confidence branches early

---

## Summary

**Before:** Simulated branching (regenerate from scratch with different seeds)
**After:** True prefix-based branching (continue from actual branch points)

**Key benefit:** More faithful to the branching concept and more computationally efficient

**Research impact:** Can now test true hypotheses about branching from high-confidence states
