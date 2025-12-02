# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

SCBranch is a research framework for investigating self-consistency methods in LLM reasoning. The goal is to reduce chain-of-thought tokens while maintaining/improving accuracy compared to traditional self-consistency.

**Three experiment types:**
1. **Traditional SC**: Generate N independent traces, majority vote on answers (baseline)
2. **Branching SC**: Start with 8 traces, branch from high-confidence traces at token checkpoints until 32 traces
3. **Peak Branching SC**: Generate initial traces fully, then branch from confidence peaks

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run traditional SC (baseline)
python scripts/run_experiment.py --experiment traditional --dataset AIME2025-I --question_id 0 --num_traces 64

# Run branching SC
python scripts/run_experiment.py --experiment branching --dataset AIME2025-I --question_id 0 --start_traces 8 --max_traces 32

# Run peak branching SC
python scripts/run_experiment.py --experiment peak_branching --dataset AIME2025-I --question_id 0 --initial_traces 8 --peak_max_traces 32

# Compute historical token statistics (required for branching)
python scripts/compute_stats.py --dataset AIME2025-I --num_samples 2

# Compare all experiments
python scripts/compare_experiments.py

# Visualize results
python scripts/visualize_results.py --results outputs/experiment_detailed_*.json
```

## Architecture

### Core Package (`scbranch/`)

- **wrapper.py**: `SCLLM` class - main interface wrapping vLLM. Three modes: `offline` (traditional), `branching`, `peak_branching`
- **branching.py**: `BranchingManager` for dynamic branching during generation
- **peak_branching.py**: `PeakBranchingManager` for confidence peak-based branching
- **utils.py**: Answer extraction, normalization, simple_majority_vote, confidence computation
- **outputs.py**: `SCOutput` dataclass for results

### Scripts (`scripts/`)

**Running experiments:**
- `run_experiment.py` - main entry point
- `experiment_utils.py` - shared utilities
- `compute_stats.py` - compute historical stats (required for branching)
- `extract_historical_from_results.py` - extract stats from traditional runs

**Visualization:**
- `compare_experiments.py` - compare traditional vs branching vs peak branching
- `visualize_results.py` - unified visualization dispatcher
- `visualize_trace_confidence.py` - confidence over time graphs
- `visualize_branching_results.py` - branching results
- `visualize_peak_branching.py` - peak branching results
- `visualize_sc_results.py` - traditional SC results

## Key Concepts

### Confidence
- **Tail confidence**: Mean of last N tokens' confidence scores (used for branch selection)
- **Peak detection**: Uses acceleration to find optimal branch points

### Token Accounting
- `total_tokens`: Total including inherited prefix
- `tokens_generated`: Only newly generated tokens (for accurate comparison)

### Datasets
- **AIME2025-I/II**: Math olympiad (15 questions), answers in `\boxed{}` format
- **GSM8k**: Grade school math (1,319 questions), answers after `####`

## GPU Configuration

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# or pass --tensor_parallel_size N
```
