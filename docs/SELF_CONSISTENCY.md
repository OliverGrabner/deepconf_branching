# Traditional Self-Consistency on AIME 2025

This implementation runs **traditional self-consistency** (Wang et al., 2022) on the AIME 2025 I and II datasets.

## What is Traditional Self-Consistency?

Traditional self-consistency is a simple but effective method to improve LLM reasoning:

1. **Generate N reasoning paths** for the same question (with temperature > 0 for diversity)
2. **Extract the answer** from each reasoning path
3. **Majority vote** to select the final answer (the answer that appears most frequently)

No confidence weighting, no filtering - just pure majority voting across diverse reasoning paths.

## Installation

```bash
# Install dependencies
pip install -r requirements_sc.txt

# Or install individually
pip install torch transformers vllm datasets numpy pandas tqdm dynasor
```

## GPU Setup

The script is configured for **4x RTX 5000 Ada GPUs**:
- Uses vLLM with `tensor_parallel_size=4`
- Set via `CUDA_VISIBLE_DEVICES="0,1,2,3"` (already in script)

## Usage

### Basic Usage (Run on Both Datasets)

```bash
python run_traditional_sc_aime25.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --num_traces 64
```

### Run on Specific Dataset Only

```bash
# AIME25-I only
python run_traditional_sc_aime25.py --dataset AIME2025-I --num_traces 64

# AIME25-II only
python run_traditional_sc_aime25.py --dataset AIME2025-II --num_traces 64
```

### Partial Runs (For Testing or Resuming)

```bash
# Run first 5 questions only
python run_traditional_sc_aime25.py --end_idx 5 --num_traces 32

# Run questions 10-20
python run_traditional_sc_aime25.py --start_idx 10 --end_idx 20 --num_traces 64
```

### Configuration Options

```bash
python run_traditional_sc_aime25.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --model_type deepseek \
    --num_traces 64 \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k 40 \
    --max_tokens 130000 \
    --tensor_parallel_size 4 \
    --output_dir outputs_sc
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | Model to use |
| `--num_traces` | 64 | Number of reasoning paths (N in self-consistency) |
| `--temperature` | 1.0 | Sampling temperature (1.0 for diversity) |
| `--tensor_parallel_size` | 4 | Number of GPUs for model parallelism |
| `--dataset` | None (both) | Run specific dataset: `AIME2025-I` or `AIME2025-II` |
| `--output_dir` | `outputs_sc` | Directory to save results |

## Output Files

The script generates three types of output files in `outputs_sc/`:

### 1. Detailed JSON (`traditional_sc_aime25_detailed_TIMESTAMP.json`)

Contains complete results for every question:
- Question text and ground truth
- All generated traces with extracted answers
- Vote distribution across answers
- Final voted answer and correctness
- Token statistics and timing for each question

### 2. Summary CSV (`traditional_sc_aime25_summary_TIMESTAMP.csv`)

Spreadsheet-friendly format with one row per question:
- Dataset, question_id, correctness
- Ground truth vs voted answer
- Number of valid traces
- Individual trace accuracy
- Token counts and timing

### 3. Statistics JSON (`traditional_sc_aime25_stats_TIMESTAMP.json`)

Aggregate statistics:
- Per-dataset accuracy and metrics
- Overall accuracy across both datasets
- Token usage and throughput
- Average time per question

## Understanding the Results

### Per-Question Output

For each question, you'll see:

```
Q0: ✓
  Ground Truth: 42
  Voted Answer: 42
  Valid Traces: 64/64
  Individual Accuracy: 78.1%
  Vote Distribution: {'42': 50, '43': 10, '41': 4}
  Tokens: 98432 (1538.0 avg)
  Time: 45.32s
```

**Interpretation:**
- **Individual Accuracy**: What % of reasoning traces got the correct answer
- **Vote Distribution**: How votes were distributed across different answers
- **Valid Traces**: How many traces successfully extracted an answer

### Final Summary

```
Overall Results (AIME25-I + AIME25-II):
  Total Questions: 30
  Total Correct: 18/30 (60.0%)
  Total Tokens: 2,456,789
  Avg Tokens/Question: 81,893
  Total Time: 1234.5s (20.6 minutes)
  Overall Throughput: 1989.2 tokens/sec
```

**Key Metrics:**
- **Accuracy**: The main metric - % of questions answered correctly using majority voting
- **Throughput**: Generation speed in tokens/second
- **Avg Tokens/Question**: How many tokens needed per question (across all N traces)

## Expected Performance

Based on typical self-consistency results on math reasoning:

- **AIME Difficulty**: Very challenging (designed for math olympiad students)
- **Expected Accuracy**: 20-60% depending on model and N
- **Individual Trace Accuracy**: Usually lower than final voted accuracy
- **Improvement from SC**: Typically 5-15% better than single-path greedy decoding

## Comparison with Other Methods

This script implements **only traditional SC (majority voting)**. The existing DeepConf framework supports 7 different voting methods including confidence-weighted variants. To compare:

```bash
# Traditional SC (this script)
python run_traditional_sc_aime25.py --num_traces 64

# All voting methods (DeepConf framework)
python examples/example_offline.py --dataset aime25.jsonl --qid 0 --budget 64
```

## Troubleshooting

### Out of Memory
- Reduce `--num_traces` (try 32 or 16)
- Reduce `--max_tokens` (try 65000)
- Ensure you're using `tensor_parallel_size=4` to split model across GPUs

### Dataset Loading Issues
```bash
# Make sure datasets library is installed
pip install datasets

# May need to login to HuggingFace
huggingface-cli login
```

### Slow Generation
- Check GPU utilization: `nvidia-smi`
- Verify `tensor_parallel_size` matches number of GPUs
- Enable prefix caching (already enabled by default)

## Citation

If you use this implementation, please cite:

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

## Notes

- **Temperature**: Set to 1.0 to ensure diverse reasoning paths (required for SC to work well)
- **Answer Extraction**: Uses `\boxed{...}` LaTeX format parsing
- **Math Equality**: Uses `dynasor` library for mathematical equivalence checking (handles different forms like "1/2" vs "0.5")
- **Reproducibility**: Results may vary slightly across runs due to non-deterministic GPU operations

## Questions?

For issues or questions about:
- This script: Check the code comments in `run_traditional_sc_aime25.py`
- DeepConf framework: See main `README.md`
- AIME dataset: Visit https://huggingface.co/datasets/opencompass/AIME2025
