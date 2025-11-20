# Experiment Consistency Checklist

## Summary
To ensure fair comparisons between Traditional SC, Branching SC, and Peak Branching SC, all three experiments need the same fixes applied.

---

## ✅ Changes Applied to Peak Branching

### 1. **Token Limits** (`wrapper.py` lines 650, 738)
```python
# Initial traces
initial_params.max_tokens = 8000  # Changed from 64000

# Branch traces
branch_params.max_tokens = min(4000, 64000 - len(branch_info['prompt_tokens']))  # Changed from max()
```

### 2. **Stop Sequences** (`wrapper.py` lines 651, 739)
```python
initial_params.stop = ["}\n\n", "}\n"]
branch_params.stop = ["}\n\n", "}\n"]
```

### 3. **Answer Normalization**
- Added `normalize_answer()` function in `utils.py` (lines 68-123)
- Applied to voting in `run_experiment.py` (lines 131-140, 306-310, 486-495)

### 4. **Correct Token Counting** (`run_experiment.py` line 341)
```python
# Uses peak_stats.get('total_tokens_generated') instead of result.total_tokens
'total_tokens_generated': peak_stats.get('total_tokens_generated', result.total_tokens)
```

---

## 🔧 Changes NEEDED for Traditional SC

### Traditional SC currently:
- ❌ Uses default `max_tokens = 130,000` from command line args
- ❌ No stop sequences set
- ✅ Answer normalization IS applied (we added it to `run_experiment.py`)
- ✅ Token counting is simple (all tokens are new)

### Changes to apply:

**File: `wrapper.py` in `_deepthink_offline()` method (line ~315)**
```python
# BEFORE:
sampling_params.n = budget
vllm_outputs = self.llm.generate([prompt], sampling_params)

# AFTER:
import copy
offline_params = copy.deepcopy(sampling_params)
offline_params.n = budget
offline_params.max_tokens = min(8000, offline_params.max_tokens)  # Cap at 8K
offline_params.stop = ["}\n\n", "}\n"]  # Add stop sequences
vllm_outputs = self.llm.generate([prompt], offline_params)
```

**Status: ✅ ALREADY DONE** (just applied in previous edit)

---

## 🔧 Changes NEEDED for Branching SC

### Branching SC currently:
- ❌ Uses default `max_tokens = 130,000` from command line args
- ❌ No explicit stop sequences
- ✅ Answer normalization IS applied (we added it to `run_experiment.py`)
- ✅ Token counting should be correct (uses `total_tokens_generated`)

### Changes to apply:

**File: `wrapper.py` in `_deepthink_branching()` method**

Need to find where traces are generated and add:
1. Token limit for initial traces
2. Token limit for branches (similar to peak branching)
3. Stop sequences

Let me check the branching implementation:

---

## 📋 CURRENT STATUS OF ALL THREE EXPERIMENTS

### Traditional SC (Offline Mode):
| Fix | Status | Location |
|-----|--------|----------|
| Token limit (8K) | ✅ DONE | `wrapper.py:316` |
| Stop sequences | ✅ DONE | `wrapper.py:317` |
| Answer normalization | ✅ DONE | `run_experiment.py:131-140` |
| Token counting | ✅ N/A | All tokens are new |

### Branching SC:
| Fix | Status | Location |
|-----|--------|----------|
| Token limit (initial) | ❓ NEED TO CHECK | `wrapper.py` branching mode |
| Token limit (branches) | ❓ NEED TO CHECK | `wrapper.py` branching mode |
| Stop sequences | ❓ NEED TO CHECK | `wrapper.py` branching mode |
| Answer normalization | ✅ DONE | `run_experiment.py:486-495` |
| Token counting | ✅ SHOULD BE OK | Uses `total_tokens_generated` |

### Peak Branching SC:
| Fix | Status | Location |
|-----|--------|----------|
| Token limit (initial) | ✅ DONE | `wrapper.py:650` (8K cap) |
| Token limit (branches) | ✅ DONE | `wrapper.py:738` (4K cap) |
| Stop sequences | ✅ DONE | `wrapper.py:651, 739` |
| Answer normalization | ✅ DONE | `run_experiment.py:131-140, 306-310` |
| Token counting | ✅ FIXED | `run_experiment.py:341` |

---

## 🎯 ACTION ITEMS

### 1. Check Branching SC Implementation
Need to verify:
- Where are initial traces generated in branching mode?
- Where are branch traces generated?
- Do they have token limits and stop sequences?

### 2. Apply Same Limits if Missing
If branching SC doesn't have limits, add:
```python
# For initial traces in branching
params.max_tokens = min(8000, params.max_tokens)
params.stop = ["}\n\n", "}\n"]

# For branch traces in branching
branch_params.max_tokens = min(4000, 64000 - existing_tokens)
branch_params.stop = ["}\n\n", "}\n"]
```

### 3. Run All Three Experiments with Same Settings
```bash
# Traditional SC - First 50 questions
python scripts/run_experiment.py --experiment traditional --dataset gsm8k \
    --num_traces 8 --start_idx 0 --end_idx 50

# Branching SC - First 50 questions
python scripts/run_experiment.py --experiment branching --dataset gsm8k \
    --start_traces 8 --max_traces 32 --start_idx 0 --end_idx 50

# Peak Branching SC - First 50 questions
python scripts/run_experiment.py --experiment peak_branching --dataset gsm8k \
    --initial_traces 8 --peak_max_traces 32 --start_idx 0 --end_idx 50
```

### 4. Compare Results
```bash
python scripts/compare_experiments.py --max_questions 50 --output_dir comparisons/
```

---

## 📊 Expected Results After Fixes

### Token Usage (per question):
- **Traditional SC**: ~12-16K tokens (8 traces × 1.5-2K each)
- **Branching SC**: ~15-20K tokens (should be similar or slightly less than Traditional)
- **Peak Branching SC**: ~15-25K tokens (was 84K, should drop dramatically)

### Key Improvements:
1. **No more 64K token explosions** - all capped at reasonable limits
2. **Consistent stop sequences** - all three stop after `\boxed{answer}`
3. **Unified answer normalization** - "18" vs "18." no longer split votes
4. **Fair comparison** - all using same constraints

---

## 🔍 What to Look For in Results

### Good Signs:
- ✅ Peak Branching uses comparable tokens to Traditional/Branching
- ✅ No traces >10K tokens
- ✅ Vote distributions show consolidated answers (no "18" vs "18." splitting)
- ✅ Accuracy improvements from branching methods

### Red Flags:
- ❌ Any traces hitting token limits frequently
- ❌ Vote splitting still occurring
- ❌ One method using significantly more tokens than others
- ❌ Token savings not showing up in comparison charts