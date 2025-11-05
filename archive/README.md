# Archive Directory

This directory contains files that have been replaced by the unified experiment runner but are kept for reference and backward compatibility.

## Contents

### scripts/
**Replaced by:** `scripts/run_experiment.py`

- `test_sc_single_question.py` - Old single-question tester for traditional SC
- `test_branching_single_question.py` - Old single-question tester for branching SC
- `analyze_sc_results.py` - Old analysis script (use `scripts/visualize_results.py` instead)
- `compute_historical_stats.py` - Original unsafe version
- `compute_historical_stats_safe.py` - AIME-only safe version
- `compute_historical_stats_gsm8k.py` - GSM8k-only version

**Why replaced:**
- Duplicate code across multiple scripts
- Limited functionality (single question mode incomplete)
- No unified interface

**New approach:**
```bash
# Run any experiment (traditional/branching, AIME/GSM8k, single/batch)
python scripts/run_experiment.py --experiment <type> --dataset <name> [--question_id N]
```

### examples/
Files moved here are kept as examples but not actively maintained:

- `example_offline.py` - Example of offline mode usage
- `example_online.py` - Example of online mode usage

### docs/
Historical documentation kept for reference:

- `BRANCHING_QUICKSTART.md` - Original branching quickstart (pre-refactor)
- `TOKEN_COUNTING_FIX.md` - Documentation of token counting fix
- `INCREMENTAL_SAVING_AND_GSM8K.md` - Documentation of incremental saving feature

**Note:** These docs describe the old API. See `docs/` in the root directory for current documentation.

## Usage

These files are **not deleted** - they remain functional if you need them for backward compatibility or reference.

However, **we recommend using the new unified scripts** in the main `scripts/` directory:
- `run_experiment.py` - Run any experiment
- `visualize_results.py` - Visualize any results
- `compute_stats.py` - Compute historical token statistics
- `experiment_utils.py` - Shared utilities

## Restore Instructions

If you need to restore any archived file:

```bash
# Restore a script
mv archive/scripts/<filename> scripts/

# Restore documentation
mv archive/docs/<filename> .

# Restore an example
mv archive/examples/<filename> examples/
```
