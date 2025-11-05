# DeepConf: Deep Thinking LLM Framework

A powerful framework for enhanced LLM reasoning with **traditional self-consistency** and **branching self-consistency**. Features include confidence-based voting, trace analysis, and comprehensive visualization.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test traditional SC on single question
python scripts/run_experiment.py \
    --experiment traditional \
    --dataset AIME2025-I \
    --question_id 0 \
    --num_traces 64

# Test branching SC on single question
python scripts/run_experiment.py \
    --experiment branching \
    --dataset AIME2025-I \
    --question_id 0 \
    --start_traces 8 \
    --max_traces 32

# Run on full dataset
python scripts/run_experiment.py \
    --experiment traditional \
    --dataset AIME2025-I \
    --num_traces 64
```

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Traditional Self-Consistency](#traditional-self-consistency)
  - [Confidence Visualization](#confidence-visualization)
  - [Result Analysis](#result-analysis)
- [Documentation](#documentation)
- [Examples](#examples)
- [API Reference](#api-reference)

## ✨ Features

### Unified Experiment Runner
- **Single Entry Point**: One script for all experiment types (traditional/branching, AIME/GSM8k)
- **Single Question Testing**: Test on one question without running full dataset
- **Incremental Saving**: Progress saved after every question (Ctrl+C safe)
- **Auto-visualization**: Graphs generated immediately after each question

### Two SC Modes
- **Traditional SC**: Standard majority voting (N traces, simple vote)
- **Branching SC**: Dynamic branching from high-confidence traces (S→M traces)

### Datasets Supported
- **AIME 2025-I & II**: Math olympiad problems (30 questions total)
- **GSM8k**: Grade school math (1,319 questions)
- Both single-question and batch processing modes

### Visualization & Analysis
- **Branching-specific**: Genealogy trees, confidence evolution with branch points, 4-panel summaries
- **Traditional SC**: Consensus distribution, vote analysis
- **Dataset-wide**: Token usage, accuracy comparisons
- Auto-generated after each question or on-demand

## 📦 Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/OliverGrabner/deepconf_branching.git
cd deepconf_branching

# Install dependencies
pip install -r requirements.txt
```

### Optional: Visualization

```bash
# For trace confidence graphs
pip install matplotlib
```

### GPU Setup

Configure for your hardware (default: 4 GPUs):

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
```

## 📁 Project Structure

```
deepconf_branching/
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
├── setup.py                             # Package setup
│
├── deepconf/                            # Core package
│   ├── __init__.py                     # Public API
│   ├── wrapper.py                      # DeepThinkLLM class
│   ├── branching.py                    # Branching manager
│   ├── utils.py                        # Utilities (AIME + GSM8k)
│   └── outputs.py                      # Output dataclasses
│
├── scripts/                             # Main scripts (USE THESE)
│   ├── run_experiment.py               # 🌟 UNIFIED RUNNER (start here)
│   ├── visualize_results.py            # Unified visualization
│   ├── compute_stats.py                # Historical token statistics
│   ├── experiment_utils.py             # Shared utilities
│   ├── run_branching_sc_aime25.py      # [DEPRECATED] Use run_experiment.py
│   ├── run_branching_sc_gsm8k.py       # [DEPRECATED] Use run_experiment.py
│   ├── run_traditional_sc_aime25.py    # [DEPRECATED] Use run_experiment.py
│   ├── visualize_branching_results.py  # [DEPRECATED] Use visualize_results.py
│   └── visualize_sc_results.py         # [DEPRECATED] Use visualize_results.py
│
├── docs/                                # Documentation
│   ├── QUICKSTART.md                   # Getting started guide
│   └── (more docs coming soon)
│
├── archive/                             # Old files (kept for reference)
│   ├── scripts/                        # Replaced scripts
│   ├── examples/                       # Example files
│   ├── docs/                           # Historical documentation
│   └── README.md                       # What's in the archive
│
└── DATA_AND_VISUALIZATION_SUMMARY.md    # Detailed output reference
```

**Key Change**: Use `scripts/run_experiment.py` as the single entry point for all experiments!

## 🎯 Usage

### Unified Experiment Runner

**Single entry point for everything:**

```bash
python scripts/run_experiment.py \
    --experiment <traditional|branching> \
    --dataset <AIME2025-I|AIME2025-II|gsm8k|both> \
    [--question_id N]  # Optional: test single question
```

### Examples

#### Single Question Testing (Fast)
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

#### Full Dataset Processing
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

#### GSM8k (Batch Processing)
```bash
# GSM8k traditional SC (first 100 questions)
python scripts/run_experiment.py \
    --experiment traditional \
    --dataset gsm8k \
    --num_traces 64 \
    --start_idx 0 \
    --end_idx 100
```

**Output** (in `outputs/`):
- `<experiment>_sc_detailed_*.json` - Full trace data
- `<experiment>_sc_summary_*.csv` - Spreadsheet format
- `<experiment>_sc_stats_*.json` - Aggregate statistics
- `visualizations/` - Auto-generated plots (3 per question + 2 dataset-wide)

### Visualizing Results

```bash
# Visualize all questions
python scripts/visualize_results.py \
    --results outputs/experiment_detailed_20250115.json

# Visualize single question only
python scripts/visualize_results.py \
    --results outputs/experiment_detailed_20250115.json \
    --question_id 0
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [**docs/QUICKSTART.md**](docs/QUICKSTART.md) | Getting started guide |
| [**DATA_AND_VISUALIZATION_SUMMARY.md**](DATA_AND_VISUALIZATION_SUMMARY.md) | Complete output/data reference |
| [**archive/README.md**](archive/README.md) | What's in the archive |

**Old documentation** (pre-refactor) is in `archive/docs/` for reference.

## 💡 Examples

### Basic Self-Consistency

```python
from deepconf import DeepThinkLLM, prepare_prompt, equal_func

# Initialize model
deep_llm = DeepThinkLLM(model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")

# Prepare prompt
question = "What is the square root of 144?"
ground_truth = "12"
prompt = prepare_prompt(question, deep_llm.tokenizer, "deepseek")

# Generate 64 reasoning traces
result = deep_llm.deepthink(
    prompt=prompt,
    mode="offline",
    budget=64,
    compute_multiple_voting=True
)

# Check result
is_correct = equal_func(result.final_answer, ground_truth)
print(f"Answer: {result.final_answer} ({'✓' if is_correct else '✗'})")
print(f"Accuracy: {is_correct}")
```

### Analyzing Confidence

```python
# Get all voting results
for method, method_result in result.voting_results.items():
    print(f"{method}: {method_result['answer']}")
    if method_result.get('confidence'):
        print(f"  Confidence: {method_result['confidence']:.3f}")

# Analyze individual traces
correct_traces = [t for t in result.all_traces
                  if t['extracted_answer'] == ground_truth]
print(f"Individual trace accuracy: {len(correct_traces)}/{len(result.all_traces)}")
```

### Advanced: Custom Voting

```python
from deepconf.utils import weighted_majority_vote, calculate_tail_confidence

# Extract answers and confidences
answers = [t['extracted_answer'] for t in result.all_traces if t['extracted_answer']]
confidences = [calculate_tail_confidence(t) for t in result.all_traces if t['extracted_answer']]

# Custom weighted voting
final_answer = weighted_majority_vote(answers, confidences)
print(f"Custom weighted answer: {final_answer}")
```

## 🔧 Configuration

### Model Selection

```bash
# DeepSeek models (default)
python scripts/run_traditional_sc_aime25.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --model_type deepseek

# Other models
python scripts/run_traditional_sc_aime25.py \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --model_type gpt
```

### Sampling Parameters

```bash
python scripts/run_traditional_sc_aime25.py \
    --num_traces 64 \          # Number of reasoning paths
    --temperature 1.0 \        # Sampling temperature
    --top_p 1.0 \             # Nucleus sampling
    --top_k 40 \              # Top-k sampling
    --max_tokens 130000       # Max tokens per generation
```

### GPU Configuration

```bash
# For 4 GPUs (default)
python scripts/run_traditional_sc_aime25.py --tensor_parallel_size 4

# For 2 GPUs
python scripts/run_traditional_sc_aime25.py --tensor_parallel_size 2
```

## 📊 Voting Methods

The framework supports 7 voting strategies:

1. **Simple Majority Vote** - Traditional self-consistency
2. **Mean Confidence Weighted** - Weight by average token confidence
3. **Tail Confidence Weighted** - Weight by last 2048 tokens confidence
4. **Bottom Window Weighted** - Weight by bottom 10% of windows
5. **Min Window Weighted** - Weight by minimum window confidence
6. **Top 10% Tail Filtered** - Filter top 10% then vote
7. **Top 10% Bottom Window Filtered** - Filter by bottom window then vote

## 🎓 Citation

If you use this framework, please cite:

**Self-Consistency:**
```bibtex
@article{wang2022self,
  title={Self-consistency improves chain of thought reasoning in language models},
  author={Wang, Xuezhi and Wei, Jason and Schuurmans, Dale and Le, Quoc and Chi, Ed and Narang, Sharan and Chowdhery, Aakanksha and Zhou, Denny},
  journal={arXiv preprint arXiv:2203.11171},
  year={2022}
}
```

**AIME 2025 Dataset:**
```bibtex
@misc{aime2025,
  title={AIME 2025 Dataset},
  author={OpenCompass},
  year={2025},
  url={https://huggingface.co/datasets/opencompass/AIME2025}
}
```

## 🔍 Performance

### Expected Results on AIME 2025

AIME is extremely challenging (math olympiad level):
- **Individual trace accuracy**: 30-50%
- **Self-consistency accuracy**: 40-70%
- **Improvement from SC**: +10-20%

### Computational Costs

With 4x RTX 5000 Ada GPUs:
- **Time per question**: ~30-60s (64 traces)
- **Throughput**: 1500-2500 tokens/sec
- **Memory**: ~10-50 MB per question (trace data)

## 🐛 Troubleshooting

### "Out of memory"

```bash
# Reduce number of traces
python scripts/run_traditional_sc_aime25.py --num_traces 32

# Or reduce max tokens
python scripts/run_traditional_sc_aime25.py --max_tokens 65000
```

### "Module 'dynasor' not found"

```bash
# Install sympy as alternative
pip install sympy

# See docs/SELF_CONSISTENCY.md for details
```

### Slow generation

```bash
# Check GPU utilization
nvidia-smi

# Ensure correct tensor parallelism
python scripts/run_traditional_sc_aime25.py --tensor_parallel_size 4
```

## 🤝 Contributing

This is a research project. For issues or questions:
- Check the [documentation](docs/)
- Review [examples](examples/)
- Examine code comments in scripts

## 📜 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **DeepSeek** for the reasoning models
- **OpenCompass** for the AIME 2025 dataset
- **Meta/vLLM** for efficient inference
- **Wang et al.** for the self-consistency method

---

**Quick Links:**
- [📖 Quick Start Guide](docs/QUICKSTART.md)
- [📊 Data & Visualization Reference](DATA_AND_VISUALIZATION_SUMMARY.md)
- [🗂️ Archive (old files)](archive/README.md)

**Ready to start?**

```bash
pip install -r requirements.txt

# Test on single question (fastest way to verify setup)
python scripts/run_experiment.py \
    --experiment traditional \
    --dataset AIME2025-I \
    --question_id 0 \
    --num_traces 64
```

## 🔄 Migration from Old Scripts

If you were using the old scripts, here's how to migrate:

| Old Script | New Command |
|------------|-------------|
| `run_traditional_sc_aime25.py` | `run_experiment.py --experiment traditional --dataset AIME2025-I` |
| `run_branching_sc_aime25.py` | `run_experiment.py --experiment branching --dataset AIME2025-I` |
| `run_branching_sc_gsm8k.py` | `run_experiment.py --experiment branching --dataset gsm8k` |
| `test_sc_single_question.py` | `run_experiment.py --experiment traditional --question_id 0` |
| `test_branching_single_question.py` | `run_experiment.py --experiment branching --question_id 0` |
| `visualize_branching_results.py` | `visualize_results.py` (auto-detects type) |
| `compute_historical_stats*.py` | `compute_stats.py --dataset <name>` |

**Old scripts still work** but are deprecated. See [archive/README.md](archive/README.md) for details.
