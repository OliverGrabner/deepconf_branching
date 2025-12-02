# SCBranch: Self-Consistency with Branching

Research framework for investigating self-consistency methods in LLM reasoning. The goal is to reduce chain-of-thought tokens while maintaining/improving accuracy compared to traditional self-consistency.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run traditional SC (baseline)
python scripts/run_experiment.py --experiment traditional --dataset AIME2025-I --question_id 0 --num_traces 64

# Run branching SC
python scripts/run_experiment.py --experiment branching --dataset AIME2025-I --question_id 0 --start_traces 8 --max_traces 32

# Run peak branching SC
python scripts/run_experiment.py --experiment peak_branching --dataset AIME2025-I --question_id 0 --initial_traces 8 --peak_max_traces 32
```

## Experiment Types

1. **Traditional SC** (baseline): Generate N independent traces, majority vote
2. **Branching SC**: Start with 8 traces, branch from high-confidence traces at checkpoints until 32 traces
3. **Peak Branching SC**: Generate initial traces fully, then branch from confidence peaks

## Project Structure

```
scbranch/                    # Core package
├── wrapper.py               # SCLLM class (main interface)
├── branching.py             # BranchingManager
├── peak_branching.py        # PeakBranchingManager
├── utils.py                 # Utilities
└── outputs.py               # SCOutput dataclass

scripts/                     # Scripts
├── run_experiment.py        # Main entry point
├── experiment_utils.py      # Shared utilities
├── compute_stats.py         # Historical stats (for branching)
├── extract_historical_from_results.py
├── compare_experiments.py   # Compare all experiment types
├── visualize_results.py     # Unified visualization
├── visualize_trace_confidence.py
├── visualize_branching_results.py
├── visualize_peak_branching.py
└── visualize_sc_results.py
```

## Usage

### Run Experiments

```bash
# Single question
python scripts/run_experiment.py --experiment traditional --dataset AIME2025-I --question_id 0 --num_traces 64

# Full dataset
python scripts/run_experiment.py --experiment traditional --dataset AIME2025-I --num_traces 64

# Branching (requires historical stats)
python scripts/compute_stats.py --dataset AIME2025-I --num_samples 2
python scripts/run_experiment.py --experiment branching --dataset AIME2025-I --start_traces 8 --max_traces 32
```

### Visualize Results

```bash
# Compare experiments
python scripts/compare_experiments.py

# View confidence evolution
python scripts/visualize_trace_confidence.py --results outputs/traditional_detailed_*.json

# Visualize specific experiment
python scripts/visualize_results.py --results outputs/experiment_detailed_*.json
```

## Datasets

- **AIME2025-I/II**: Math olympiad (15 questions each), answers in `\boxed{}` format
- **GSM8k**: Grade school math (1,319 questions), answers after `####`

## GPU Configuration

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# or pass --tensor_parallel_size N
```

## Citation

```bibtex
@article{wang2022self,
  title={Self-consistency improves chain of thought reasoning in language models},
  author={Wang, Xuezhi and Wei, Jason and others},
  journal={arXiv preprint arXiv:2203.11171},
  year={2022}
}
```
