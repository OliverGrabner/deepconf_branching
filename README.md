# DeepConf: Deep Thinking LLM Framework

A powerful framework for enhanced LLM reasoning with **self-consistency**, **confidence-based voting**, and **trace analysis**. Supports both online (confidence-based early stopping) and offline (batch generation) modes.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test on a single question
python scripts/test_sc_single_question.py

# Run full experiment on AIME 2025
python scripts/run_traditional_sc_aime25.py --num_traces 64
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

### Core Framework
- **7 Voting Strategies**: From simple majority to confidence-weighted voting
- **Dual Processing Modes**: Online (early stopping) and offline (batch) generation
- **Memory Optimized**: Stores only confidence values, not full logprobs
- **vLLM Backend**: Efficient parallel generation with prefix caching

### Self-Consistency on AIME 2025
- **Traditional SC**: Pure majority voting on challenging math problems
- **Comprehensive Metrics**: Accuracy, tokens, throughput, per-question analysis
- **Multi-format Output**: JSON (detailed), CSV (spreadsheet), JSON (stats)

### Trace Visualization
- **Confidence Evolution**: Track tail confidence as each trace generates
- **4-Panel Graphs**: All traces, correct only, incorrect only, distribution
- **ASCII Fallback**: Works without matplotlib
- **Research Ready**: High-res graphs (300 DPI) for publications

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
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
│
├── deepconf/                      # Core package
│   ├── __init__.py               # Public API
│   ├── wrapper.py                # DeepThinkLLM class
│   ├── utils.py                  # Utilities and voting methods
│   └── outputs.py                # Output dataclasses
│
├── scripts/                       # Executable scripts
│   ├── run_traditional_sc_aime25.py      # Main SC experiment
│   ├── test_sc_single_question.py        # Quick test
│   ├── analyze_sc_results.py             # Result analysis
│   ├── visualize_sc_results.py           # ASCII visualization
│   ├── visualize_trace_confidence.py     # Confidence graphs
│   └── run_sc_experiments.sh             # Automated experiments
│
├── examples/                      # Example usage
│   ├── example_offline.py        # Offline mode example
│   └── example_online.py         # Online mode example
│
└── docs/                          # Documentation
    ├── QUICKSTART.md             # Quick reference guide
    ├── SELF_CONSISTENCY.md       # Traditional SC guide
    ├── TRACE_VISUALIZATION.md    # Visualization guide
    ├── IMPLEMENTATION.md         # Technical details
    └── CHANGELOG.md              # Recent changes
```

## 🎯 Usage

### Traditional Self-Consistency

Run standard self-consistency on AIME 2025 datasets:

```bash
# Standard run (64 traces on both AIME25-I and AIME25-II)
python scripts/run_traditional_sc_aime25.py --num_traces 64

# Single dataset
python scripts/run_traditional_sc_aime25.py --dataset AIME2025-I --num_traces 64

# Quick test (first 5 questions)
python scripts/run_traditional_sc_aime25.py --end_idx 5 --num_traces 32
```

**Output** (in `outputs_sc/`):
- `traditional_sc_aime25_detailed_*.json` - Full trace data
- `traditional_sc_aime25_summary_*.csv` - Spreadsheet format
- `traditional_sc_aime25_stats_*.json` - Aggregate statistics

### Confidence Visualization

Track and graph confidence evolution for each trace:

```bash
# Visualize a single question
python scripts/visualize_trace_confidence.py \
    --qid 0 \
    --dataset AIME2025-I \
    --num_traces 16
```

**Output** (4-panel graph):
1. All traces (green=correct, red=incorrect)
2. Correct traces only
3. Incorrect traces only
4. Final confidence histogram

### Result Analysis

Analyze patterns in your SC results:

```bash
# Deep analysis
python scripts/analyze_sc_results.py outputs_sc/traditional_sc_aime25_detailed_*.json

# ASCII visualization
python scripts/visualize_sc_results.py outputs_sc/traditional_sc_aime25_detailed_*.json
```

**Insights:**
- Vote consensus vs correctness
- Individual trace accuracy vs voting accuracy
- Answer diversity patterns
- Interesting success/failure cases

## 📚 Documentation

Comprehensive guides in [`docs/`](docs/):

| Document | Description |
|----------|-------------|
| [**QUICKSTART.md**](docs/QUICKSTART.md) | TL;DR commands and examples |
| [**SELF_CONSISTENCY.md**](docs/SELF_CONSISTENCY.md) | Traditional SC on AIME 2025 |
| [**TRACE_VISUALIZATION.md**](docs/TRACE_VISUALIZATION.md) | Confidence tracking guide |
| [**IMPLEMENTATION.md**](docs/IMPLEMENTATION.md) | Technical implementation details |
| [**CHANGELOG.md**](docs/CHANGELOG.md) | Recent features and changes |

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
- [🧮 Self-Consistency Guide](docs/SELF_CONSISTENCY.md)
- [📊 Visualization Guide](docs/TRACE_VISUALIZATION.md)
- [🔧 Implementation Details](docs/IMPLEMENTATION.md)

**Ready to start?**

```bash
pip install -r requirements.txt
python scripts/test_sc_single_question.py
```
