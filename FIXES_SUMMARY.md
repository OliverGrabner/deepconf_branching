# Comprehensive Summary of Bug Fixes

## Overview
This document summarizes all critical bugs found and fixed during analysis of the Peak Branching implementation.

---

## 🐛 BUG 1: Incorrect Token Counting for Peak Branching
**File:** `scripts/run_experiment.py` (Line 341)

### The Problem:
```python
# WRONG - Both fields set to the same value
'total_tokens': result.total_tokens,
'total_tokens_generated': result.total_tokens,  # ❌ Should be NEW tokens only!
```

This caused the comparison charts to incorrectly show Peak Branching using ALL tokens (including inherited prefix tokens) instead of just NEW tokens generated.

### The Fix:
```python
# CORRECT - Get NEW tokens from peak_branching_stats
'total_tokens': result.total_tokens,  # Total INCLUDING prefix
'total_tokens_generated': peak_stats.get('total_tokens_generated', result.total_tokens),  # NEW tokens only
```

### Impact:
- **Before**: Charts showed Peak Branching at 105K tokens (incorrect - includes prefix)
- **After**: Charts will show ~84K tokens (correct - NEW tokens only)
- **Still an issue**: Even 84K is way more than Traditional SC's 20K due to other problems below

---

## 🔥 BUG 2: max() vs min() Bug - Token Explosion
**File:** `deepconf/wrapper.py` (Line 736 and 650)

### The Problem:
```python
# WRONG - Uses max() instead of min()
branch_params.max_tokens = max(1000, 64000 - len(branch_info['prompt_tokens']))
```

This allowed branches to generate up to 63,500 tokens when branching early!

### The Fix:
```python
# CORRECT - Use min() to cap at reasonable limit
branch_params.max_tokens = min(4000, 64000 - len(branch_info['prompt_tokens']))

# Also fixed initial traces:
initial_params.max_tokens = 8000  # Was 64000 before
```

### Impact:
- **366 traces** (12% of all traces) generated >5,000 tokens
- **14 traces** hit the 64K token limit
- With the fix:
  - Branches capped at 4,000 tokens maximum
  - Initial traces capped at 8,000 tokens
  - Expected reduction: ~84K → ~25-30K tokens per question

### Why It Happened:
1. Early branching (e.g., at token 500) → `max(1000, 63500) = 63,500` allowed!
2. Model starts reasoning almost from scratch with minimal context
3. Stop sequences don't always work reliably
4. Generation continues until hitting the massive limit

---

## 🎯 BUG 3: Answer Normalization - Vote Splitting
**Files:** `deepconf/utils.py`, `scripts/run_experiment.py`

### The Problem:
Answers like "18", "18.", "18.0", "18.00" were treated as **different answers**, causing vote splitting and potentially wrong final answers.

Example from your data:
```python
vote_distribution = {
    "18": 32.3,
    "18.": 1.0,
    "12": 1.1
}
# "18" should have had 33.3 votes total!
```

### The Fix:

**1. Added normalization function** (`deepconf/utils.py`):
```python
def normalize_answer(answer: str) -> str:
    """
    Normalize answers for consistent voting:
    - "1" vs "1." vs "1.0" -> all become "1"
    - "18" vs "18." vs "18.0" -> all become "18"
    - Removes trailing periods, commas
    - Detects invalid answers (e.g., very long repeated digits)
    """
```

**2. Updated voting functions** to use normalization:
- `simple_majority_vote()` - normalizes before counting
- `weighted_majority_vote()` - normalizes before weighting
- All voting in `run_experiment.py` - normalizes before Counter()

### Impact:
- Prevents vote splitting from format variations
- More accurate final answers
- Filters out invalid answers (extremely long repeated digits from runaway generation)

### Examples Handled:
- ✅ "18" vs "18." vs "18.0" → all count as "18"
- ✅ "1" vs "1." vs "1.0" vs "1.00" → all count as "1"
- ✅ "1,000" vs "1000" → both become "1000"
- ✅ "0.5" vs ".5" → both become "0.5"
- ✅ "3333...333" (1000+ digits) → marked as "Invalid"

---

## 📊 Analysis Results from Your Data

### Current State (with bugs):
```
Traditional SC:     ~20,000 tokens per question
Peak Branching:     84,101 tokens per question (4.2× worse!)
  - With bug showed: 105,287 tokens (5.3× worse!)
```

### Token Distribution Issues:
```
Median: 1,504 tokens (reasonable)
Mean:   3,693 tokens (pulled up by outliers)
Max:    63,482 tokens (hit 64K limit!)

Stage 0 (initial): Mean 2,160 tokens ✓ OK
Stage 1:           Mean 3,269 tokens ⚠️ Getting worse
Stage 2:           Mean 5,134 tokens 🔥 Explosion
```

### Why Peak Branching Used So Many Tokens:

1. **Early branching** (avg 783 tokens = 47% completion)
   - Should branch at ~75% (1200+ tokens)
   - Branching too early means generating ~53% new tokens per branch

2. **Token explosions** (max/min bug)
   - 366 traces >5,000 tokens
   - 14 traces hit 64K limit
   - Model couldn't stop generating

3. **Acceleration-based peaks** find "aha moments" too early
   - Confidence accelerates at 200-500 tokens
   - Not enough context to complete efficiently

---

## 🚀 Expected Improvements After Fixes

### Fix #1 (Correct token counting):
- Charts will show correct values
- Doesn't reduce actual token usage, just reports it correctly

### Fix #2 (Token limits):
- **Huge impact!** Caps branches at 4K tokens
- Expected reduction: 84K → 25-30K tokens per question
- Eliminates 14 traces that hit 64K
- Should reduce 366 outlier traces significantly

### Fix #3 (Answer normalization):
- Better accuracy from consolidated votes
- Fewer cases of vote splitting
- Invalid answers filtered out

---

## ⚙️ Recommended Next Steps

1. **Re-run experiments** with the fixes:
   ```bash
   python scripts/run_experiment.py --experiment peak_branching \
       --dataset gsm8k --initial_traces 8 --peak_max_traces 32 \
       --start_idx 0 --end_idx 100
   ```

2. **Tune peak detection** to branch later:
   - Increase `peak_selection_ratio` (currently 0.95)
   - Add minimum progress threshold (e.g., only after 40% completion)
   - Consider switching from acceleration to absolute confidence

3. **Improve stop sequences**:
   - Add more variants: `["}\n\n", "}\n", "}.", "} "]`
   - Consider detecting repeated text patterns

4. **Compare results**:
   ```bash
   python scripts/compare_experiments.py --output_dir comparisons/
   ```

---

## 📝 Summary

### Critical Bugs Fixed:
1. ✅ Token counting bug in `run_experiment.py` (line 341)
2. ✅ max() vs min() bug in `wrapper.py` (lines 650, 738)
3. ✅ Answer normalization throughout voting logic

### Expected Outcome:
- **Token usage**: Should drop from 84K to 25-30K per question
- **Accuracy**: Improved from better vote consolidation
- **Reliability**: No more 64K token explosions

### The Remaining Challenge:
Even with fixes, Peak Branching may still use more tokens than Traditional SC due to:
- Early branching from acceleration-based peaks
- Need to tune peak detection parameters
- Consider alternative peak detection strategies

But it should be MUCH better than before!