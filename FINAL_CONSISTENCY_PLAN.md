# Final Consistency Plan for All Three Experiments

## 🎯 Goal
Apply consistent token limits, stop sequences, and answer normalization across Traditional SC, Branching SC, and Peak Branching SC so they can be fairly compared.

---

## 📊 Current Status Analysis

### ✅ **Answer Normalization** - ALREADY CONSISTENT
All three experiments use the same normalization in `run_experiment.py`:
- Traditional SC: Lines 131-140 ✅
- Branching SC: Lines 486-495 ✅
- Peak Branching SC: Lines 131-140 and 306-310 ✅

**No action needed** - all experiments normalize answers before voting.

---

### ✅ **Traditional SC** - ALREADY FIXED
File: `wrapper.py` `_deepthink_offline()` (lines 316-317)
```python
offline_params.max_tokens = min(8000, offline_params.max_tokens)
offline_params.stop = ["}\n\n", "}\n"]
```
**Status: COMPLETE** - Just applied in previous edit.

---

### ⚠️ **Branching SC** - NEEDS FIXES

#### Current Implementation Issues:
1. **Line 520**: Final generation could use up to 130K tokens
   ```python
   final_params.max_tokens = sampling_params.max_tokens - manager.stride * n_iterations
   # If max_tokens=130K and stride*iterations=10K, this allows 120K tokens!
   ```

2. **No stop sequences** in final generation (line 518-520)

3. **Token counting** looks correct (line 579) - uses `total_tokens_generated`

#### Fixes Needed:

**Location 1: Line 520 in `_deepthink_branching()`**
```python
# BEFORE:
final_params.max_tokens = sampling_params.max_tokens - manager.stride * n_iterations

# AFTER:
# Cap final generation at reasonable limit
remaining_budget = sampling_params.max_tokens - manager.stride * n_iterations
final_params.max_tokens = min(8000, remaining_budget)
final_params.stop = ["}\n\n", "}\n"]  # Add stop sequences
```

---

### ✅ **Peak Branching SC** - ALREADY COMPLETE

All fixes already applied:
- Initial traces: 8K limit + stop sequences (line 650-651) ✅
- Branch traces: 4K limit + stop sequences (line 738-739) ✅
- Answer normalization (lines 131-140, 306-310) ✅
- Token counting fixed (line 341) ✅

---

## 🔧 Implementation Plan

### Step 1: Fix Branching SC Token Limits
**File:** `deepconf/wrapper.py`
**Line:** ~520
**Change:**
```python
# Before line 520, add:
remaining_budget = sampling_params.max_tokens - manager.stride * n_iterations
final_params.max_tokens = min(8000, remaining_budget)
final_params.stop = ["}\n\n", "}\n"]  # Add stop sequences

# Remove old line 520:
# final_params.max_tokens = sampling_params.max_tokens - manager.stride * n_iterations
```

### Step 2: Add Debug Output
Add after the change:
```python
print(f"  Final generation max_tokens: {final_params.max_tokens}")
print(f"  Stop sequences: {final_params.stop}")
```

---

## 📋 Complete Changes Summary

### Files Modified:

#### 1. `deepconf/utils.py`
- ✅ Added `normalize_answer()` function (lines 68-123)
- ✅ Updated `simple_majority_vote()` (lines 211-225)
- ✅ Updated `weighted_majority_vote()` (lines 228-245)

#### 2. `scripts/run_experiment.py`
- ✅ Import `normalize_answer` (line 42)
- ✅ Apply normalization in traditional voting (lines 131-140)
- ✅ Apply normalization in peak branching voting (lines 306-310)
- ✅ Apply normalization in branching voting (lines 486-495)
- ✅ Fix peak branching token counting (line 341)

#### 3. `deepconf/wrapper.py`
- ✅ Traditional SC: Add 8K limit + stop sequences (lines 316-317)
- ✅ Peak Branching initial: 8K limit + stop sequences (lines 650-651)
- ✅ Peak Branching branches: 4K limit with min() fix (line 738)
- ⚠️ **PENDING**: Branching SC: Add 8K limit + stop sequences (line ~520)

#### 4. `scripts/compare_experiments.py`
- ✅ Add `--max_questions` parameter (line 614)
- ✅ Update `extract_metrics()` to respect limit (line 93)

---

## 🚀 Commands to Run Experiments

### 1. Traditional SC (First 50 Questions)
```bash
python scripts/run_experiment.py \
    --experiment traditional \
    --dataset gsm8k \
    --num_traces 8 \
    --start_idx 0 \
    --end_idx 50 \
    --temperature 0.6
```

### 2. Branching SC (First 50 Questions)
```bash
python scripts/run_experiment.py \
    --experiment branching \
    --dataset gsm8k \
    --start_traces 8 \
    --max_traces 32 \
    --start_idx 0 \
    --end_idx 50 \
    --temperature 0.6
```

### 3. Peak Branching SC (First 50 Questions)
```bash
python scripts/run_experiment.py \
    --experiment peak_branching \
    --dataset gsm8k \
    --initial_traces 8 \
    --peak_max_traces 32 \
    --start_idx 0 \
    --end_idx 50 \
    --temperature 0.6
```

### 4. Compare Results
```bash
python scripts/compare_experiments.py \
    --max_questions 50 \
    --output_dir comparisons/
```

---

## 📈 Expected Outcomes

### Token Usage (per question, 8→32 traces):
| Experiment | Before Fixes | After Fixes | Change |
|------------|--------------|-------------|--------|
| Traditional SC | ~20K | ~12-16K | -20 to -40% |
| Branching SC | ~15-20K | ~12-18K | Minimal (already efficient) |
| Peak Branching SC | **84K** 🔥 | ~15-25K | **-70%** 🎉 |

### Key Improvements:
1. ✅ **No token explosions** - All capped at 8K per trace
2. ✅ **Consistent stopping** - All use `["}\n\n", "}\n"]`
3. ✅ **Unified normalization** - "18" vs "18." unified
4. ✅ **Fair comparison** - All experiments use same constraints

---

## ✅ Verification Checklist

After running experiments, verify:

### Token Usage:
- [ ] No traces >10K tokens in any experiment
- [ ] Peak Branching tokens comparable to Traditional/Branching
- [ ] Total tokens per question reasonable (<30K for 8→32 traces)

### Answer Quality:
- [ ] Vote distributions show consolidated answers (e.g., "18": 25 not "18": 18, "18.": 5, "18.0": 2)
- [ ] No "Invalid" answers in vote distributions
- [ ] Accuracy metrics make sense

### Comparison Charts:
- [ ] Token efficiency chart shows Peak Branching competitive with Traditional
- [ ] Accuracy charts show improvements from branching
- [ ] Chain length charts show reasonable trace lengths
- [ ] All three methods compared on same 50 questions

---

## 🐛 Troubleshooting

### If Peak Branching still uses too many tokens:
- Check that `peak_branching_stats.total_tokens_generated` is being used
- Verify branch traces aren't hitting 4K limit frequently
- May need to reduce `peak_max_traces` or branch earlier

### If vote splitting still occurs:
- Check that `normalize_answer()` is being called
- Verify normalization handles your specific answer formats
- Check vote_distribution output for split votes

### If comparisons look unfair:
- Ensure all three experiments run with same `--temperature`
- Verify all use same dataset split (0-50)
- Check that token counting is consistent (NEW tokens only)

---

## 📝 Notes

- All token limits are designed to be **generous enough** for GSM8K problems (typically 1.5-3K tokens)
- Stop sequences `["}\n\n", "}\n"]` work for `\boxed{answer}` format
- Answer normalization handles numbers, decimals, and invalid answers
- Comparison script auto-selects most recent files for each experiment type