# Quick Start Guide

## Installation

```bash
git clone <repository>
cd deepconf_branching
pip install -e .
```

## Basic Usage

### Single Question Testing
```bash
# Traditional SC on one question
python scripts/run_experiment.py \
    --experiment traditional \
    --dataset AIME2025-I \
    --question_id 0 \
    --num_traces 64

# Branching SC on one question  
python scripts/run_experiment.py \
    --experiment branching \
    --dataset AIME2025-I \
    --question_id 0 \
    --start_traces 8 \
    --max_traces 32
```

### Full Dataset Processing
```bash
# Traditional SC on full AIME25-I
python scripts/run_experiment.py \
    --experiment traditional \
    --dataset AIME2025-I \
    --num_traces 64

# Branching SC (requires historical stats first)
python scripts/compute_stats.py --dataset AIME2025-I --num_samples 2

python scripts/run_experiment.py \
    --experiment branching \
    --dataset AIME2025-I \
    --start_traces 8 \
    --max_traces 32 \
    --historical_stats historical_stats/aime2025_i_token_stats_latest.json
```

### Visualizations
```bash
python scripts/visualize_results.py \
    --results outputs/experiment_detailed_TIMESTAMP.json
```

## Key Features

- Single question OR full dataset processing
- Incremental saving (Ctrl+C safe)
- Automatic visualization generation
- Works with AIME2025-I, AIME2025-II, GSM8k, or both AIME datasets

See docs/EXPERIMENT_REFERENCE.md for all parameters.
