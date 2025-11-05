# Incremental Saving and GSM8k Support

## Overview

This update adds two major improvements to the branching self-consistency system:

1. **Incremental Saving**: Both AIME25 and GSM8k scripts now save progress after every question
2. **GSM8k Support**: Complete implementation for running branching SC on the GSM8k benchmark

## Why These Changes Matter

### Problem with Original Implementation
- Scripts only saved results at the very end
- If you Ctrl+C or crash after 8 hours, you lose ALL work
- No way to resume from where you left off
- Visualizations only generated at the end (could fail and lose everything)

### Solution: Incremental Saving
- ✅ Save JSON results after EVERY question completes
- ✅ Generate visualizations immediately after each question (3 plots)
- ✅ Continue on errors (one bad question doesn't kill the run)
- ✅ Graceful Ctrl+C handling (preserves all completed work)
- ✅ Automatic cleanup of temp files when complete

## Files Modified

### 1. `deepconf/utils.py`
**Added GSM8k-specific utilities:**

```python
def extract_answer_gsm8k(text: str) -> Optional[str]:
    """Extract numerical answer from GSM8k format (#### 123)"""
    # Handles #### marker or extracts last number

def equal_func_gsm8k(answer: str, ground_truth: str) -> bool:
    """Numerical comparison for GSM8k answers"""
    # Integer/float comparison with tolerance
```

### 2. `scripts/run_branching_sc_aime25.py`
**Key additions:**

```python
# After each question completes:
1. Save incremental results to temp JSON file
2. Generate 3 visualizations (summary, genealogy, confidence)
3. Continue on error (don't crash entire run)

# At the end:
1. Save final results
2. Delete temp file
3. Generate dataset-wide visualizations (2 plots)
```

**New behavior:**
- Creates `branching_sc_aime25_detailed_TIMESTAMP_temp.json` during run
- Saves after each question (can Ctrl+C safely)
- Generates per-question plots immediately
- Try-except around each question (continue on errors)

### 3. `scripts/run_branching_sc_gsm8k.py` (NEW)
**Complete GSM8k implementation with:**
- GSM8k dataset loading from HuggingFace
- GSM8k-specific answer extraction (#### format)
- Numerical comparison for answers
- Incremental saving from the start
- Per-question visualization generation
- Resume capability (via temp files)

**Dataset info:**
- 1,319 test questions (vs 30 for AIME25)
- Shorter problems (~5k tokens vs ~8k for AIME)
- Format: `{'question': str, 'answer': 'reasoning ... #### 123'}`

### 4. `scripts/compute_historical_stats_gsm8k.py` (NEW)
**Safe historical statistics collection:**
- Timeout protection (30 min per question)
- Saves after each question
- Resume capability
- Fallback values on timeout
- Same safety features as AIME version

## Usage Examples

### Running AIME25 with Incremental Saving

```bash
# Basic run - saves progress after each question
python scripts/run_branching_sc_aime25.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --start_traces 8 \
    --max_traces 32 \
    --historical_stats historical_stats/aime25_token_stats_latest.json

# Run subset (questions 0-4)
python scripts/run_branching_sc_aime25.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --start_traces 8 \
    --max_traces 32 \
    --start_idx 0 \
    --end_idx 5 \
    --historical_stats historical_stats/aime25_token_stats_latest.json
```

**What you'll see:**
```
Q0: ✓
  Ground Truth: 42
  Voted Answer: 42
  Valid Traces: 32/32
  💾 Saved progress: outputs_sc/branching_sc_aime25_detailed_20250115_143022_temp.json
  📊 Generating visualizations for Q0...
  ✓ Visualizations created

Q1: ✗
  Ground Truth: 123
  Voted Answer: 124
  ...
  💾 Saved progress: outputs_sc/branching_sc_aime25_detailed_20250115_143022_temp.json
  📊 Generating visualizations for Q1...
  ✓ Visualizations created
```

### Running GSM8k

```bash
# First: Compute historical statistics (run on subset for speed)
python scripts/compute_historical_stats_gsm8k.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --num_samples 2 \
    --start_idx 0 \
    --end_idx 100 \
    --timeout 1800

# Then: Run branching SC
python scripts/run_branching_sc_gsm8k.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --start_traces 8 \
    --max_traces 32 \
    --historical_stats historical_stats/gsm8k_token_stats_latest.json

# Run in chunks (recommended for 1319 questions!)
python scripts/run_branching_sc_gsm8k.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --start_traces 8 \
    --max_traces 32 \
    --start_idx 0 \
    --end_idx 100 \
    --historical_stats historical_stats/gsm8k_token_stats_latest.json
```

## Output Structure

### During Execution
```
outputs_sc/
├── branching_sc_aime25_detailed_20250115_143022_temp.json  # Updated after each Q
└── visualizations/
    ├── summary_AIME2025-I_q0_20250115_143022.png          # Created immediately
    ├── genealogy_AIME2025-I_q0_20250115_143022.png
    ├── confidence_AIME2025-I_q0_20250115_143022.png
    ├── summary_AIME2025-I_q1_20250115_143022.png          # Created immediately
    ├── genealogy_AIME2025-I_q1_20250115_143022.png
    └── confidence_AIME2025-I_q1_20250115_143022.png
```

### After Completion
```
outputs_sc/
├── branching_sc_aime25_detailed_20250115_143022.json      # Final results
├── branching_sc_aime25_summary_20250115_143022.csv
├── branching_sc_aime25_stats_20250115_143022.json
└── visualizations/
    ├── [all per-question plots from above]
    ├── token_usage_20250115_143022.png                    # Dataset-wide
    └── accuracy_analysis_20250115_143022.png              # Dataset-wide
```

## Safety Features

### 1. Incremental Saving
```python
# After each question:
temp_file = save_incremental_results(all_results, args.output_dir, timestamp, args)
print(f"  💾 Saved progress: {temp_file}")
```

### 2. Error Handling
```python
try:
    result = process_question_branching(...)
    results.append(result)
except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user (Ctrl+C)")
    print(f"Progress saved: {len(results)}/{total} questions completed")
    raise  # Exit gracefully
except Exception as e:
    print(f"\n❌ Error processing Q{i}: {e}")
    print(f"Continuing with next question...")
    continue  # Skip this question, keep going
```

### 3. Graceful Shutdown
- Ctrl+C preserves all completed work
- Temp file contains all results so far
- Can manually inspect temp file if needed

### 4. Automatic Cleanup
```python
# After successful completion:
temp_filepath = os.path.join(output_dir, f"...{TEMP_RESULTS_SUFFIX}")
if os.path.exists(temp_filepath):
    os.remove(temp_filepath)
    print(f"Removed temporary file: {temp_filepath}")
```

## What Gets Saved After Each Question

### JSON Structure (Incremental Temp File)
```json
{
  "metadata": {
    "timestamp": "20250115_143022",
    "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "start_traces": 8,
    "max_traces": 32,
    "status": "in_progress"
  },
  "results": {
    "AIME2025-I": [
      {
        "question": "...",
        "ground_truth": "42",
        "voted_answer": "42",
        "is_correct": true,
        "full_traces": [...],
        "branch_genealogy": {...},
        "statistics": {...}
      }
      // More questions as they complete
    ]
  },
  "summary": null  // Computed at the end
}
```

### Visualizations Created Immediately
1. **Summary plot**: 4-panel overview (confidence, tokens, timeline, answers)
2. **Genealogy graph**: Branch tree showing parent-child relationships
3. **Confidence evolution**: Confidence over time with branch points marked

## Comparison: Old vs New Behavior

### Old Behavior (AIME25 only)
```
Process Q0... (waiting...)
Process Q1... (waiting...)
...
Process Q14... (waiting...)
❌ Ctrl+C → LOSE EVERYTHING
✅ Complete → Save all at once → Generate all visualizations
```

### New Behavior (Both AIME25 and GSM8k)
```
Process Q0... ✓ → Save → Generate 3 plots
Process Q1... ✓ → Save → Generate 3 plots
...
Process Q14... ✓ → Save → Generate 3 plots
🟢 Ctrl+C at any point → Keep all completed work
✅ Complete → Final save → Generate 2 dataset-wide plots → Cleanup temp file
```

## Recommendations

### For AIME25 (30 questions)
```bash
# Can run all at once safely
python scripts/run_branching_sc_aime25.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --start_traces 8 \
    --max_traces 32 \
    --historical_stats historical_stats/aime25_token_stats_latest.json
```

### For GSM8k (1,319 questions)
```bash
# Run in chunks of 100-200 questions
# Example: 100 questions per run

# Chunk 1 (Q0-99)
python scripts/run_branching_sc_gsm8k.py \
    --start_idx 0 --end_idx 100 \
    --historical_stats historical_stats/gsm8k_token_stats_latest.json

# Chunk 2 (Q100-199)
python scripts/run_branching_sc_gsm8k.py \
    --start_idx 100 --end_idx 200 \
    --historical_stats historical_stats/gsm8k_token_stats_latest.json

# etc...
```

**Note**: You'll need to manually merge results from multiple chunks if desired.

## Testing

### Test AIME25 Incremental Saving
```bash
# Run just 2 questions to test
python scripts/run_branching_sc_aime25.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --start_traces 4 \
    --max_traces 8 \
    --start_idx 0 \
    --end_idx 2 \
    --historical_stats historical_stats/aime25_token_stats_latest.json

# Check:
# 1. Temp file exists: outputs_sc/*_temp.json
# 2. Two sets of 3 visualizations created
# 3. After completion, temp file deleted
```

### Test GSM8k
```bash
# First generate historical stats for 10 questions
python scripts/compute_historical_stats_gsm8k.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --num_samples 2 \
    --start_idx 0 \
    --end_idx 10

# Then run branching SC on those 10
python scripts/run_branching_sc_gsm8k.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --start_traces 4 \
    --max_traces 8 \
    --start_idx 0 \
    --end_idx 10 \
    --historical_stats historical_stats/gsm8k_token_stats_latest.json

# Check output structure and temp file handling
```

## Troubleshooting

### Temp file not deleted?
- Script probably crashed or was force-killed
- Safe to manually delete `*_temp.json` files
- They contain valid results if you need to inspect

### Visualization failed?
- Check matplotlib/networkx installed: `pip install matplotlib networkx`
- Errors in visualization won't crash the run
- Can regenerate later with: `python scripts/visualize_branching_results.py --results <json_file>`

### GSM8k answer extraction issues?
- Check ground truth format with `print(dataset[0]['answer'])`
- Should see: "reasoning ... #### 123"
- Function handles both #### format and fallback to last number

## Benefits Summary

✅ **No data loss**: Ctrl+C at any point preserves completed work
✅ **Progress visibility**: See results as they complete, not just at the end
✅ **Faster debugging**: Visualizations available immediately to check correctness
✅ **GSM8k support**: Can now run on large-scale benchmark (1,319 questions)
✅ **Error resilience**: One bad question doesn't kill entire run
✅ **Chunked execution**: Can run GSM8k in manageable pieces
✅ **Resume capability**: Can continue from where you left off (via temp files)
