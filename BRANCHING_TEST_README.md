# Branching Experiment Test Script

This script (`run_branching_test.py`) allows you to run the branching confidence experiment on limited hardware using smaller models for testing.

## Hardware Configuration

The script is configured to use **only GPUs 0, 1, and 2** (3x NVIDIA A5000s) via:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
```

## Default Model

By default, the script uses **DeepSeek-R1-Distill-Qwen-1.5B**, a small model suitable for testing on limited hardware. You can also try:
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (default, ~1.5B parameters)
- `Qwen/Qwen2.5-1.5B-Instruct` (alternative small model)

## Quick Start

### Basic Usage
```bash
python run_branching_test.py
```

This will run a simple test question with default parameters.

### Custom Question
```bash
python run_branching_test.py --question "What is the sum of all prime numbers between 10 and 20?"
```

### Adjust Branching Parameters
```bash
python run_branching_test.py \
    --initial_branches 3 \
    --max_total_branches 9 \
    --confidence_threshold 1.8 \
    --max_tokens 6000
```

### Use Different Model
```bash
python run_branching_test.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --initial_branches 2 \
    --max_total_branches 6
```

## Command Line Arguments

### Experiment Configuration
- `--question`: Question to process (default: "What is 15% of 240?")
- `--model`: Model to use (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

### Branching Parameters
- `--initial_branches`: Number of initial traces (default: 2)
- `--max_total_branches`: Maximum total traces including branches (default: 6)
- `--confidence_threshold`: Confidence threshold for branching (default: 1.5)
- `--window_size`: Sliding window size for confidence calculation (default: 128)
- `--max_tokens`: Maximum tokens per generation (default: 4000)

### Output Configuration
- `--output_dir`: Directory for output files (default: outputs)
- `--image_dir`: Directory for plots (default: images)
- `--log_dir`: Directory for log files (default: logs)

## Output Files

The script generates three types of output:

### 1. Logs (in `logs/` directory)
- `branching_test_TIMESTAMP.log`: Detailed log of the experiment run
- Contains initialization info, progress updates, and results

### 2. Data Files (in `outputs/` directory)
- `branching_test_TIMESTAMP.pkl`: Full results including all traces (pickle format)
- `branching_test_TIMESTAMP.json`: Summary results (JSON format for easy viewing)

### 3. Visualizations (in `images/` directory)
- `trace_confidence_TIMESTAMP.png`: Confidence plot for first trace with sliding window
- `branching_tree_TIMESTAMP.png`: Visualization of all traces organized by depth
- `branching_stats_TIMESTAMP.png`: Statistics about branching behavior

## Example Workflow

```bash
# Create necessary directories (done automatically, but good practice)
mkdir -p logs outputs images

# Run experiment with custom parameters
python run_branching_test.py \
    --question "Calculate the area of a circle with radius 7" \
    --initial_branches 2 \
    --max_total_branches 8 \
    --confidence_threshold 1.5 \
    --max_tokens 4000

# Check the logs
tail -f logs/branching_test_*.log

# View the generated images
ls -lh images/
```

## Understanding the Output

### Branching Statistics
The script outputs statistics about the branching behavior:
- `total_branches`: Number of new branches spawned
- `avg_confidence_at_branch`: Average confidence value at branching points
- Branch position distribution: Where in the generation branches occur

### Confidence Plots
- **Raw Confidence**: Token-by-token confidence scores
- **Sliding Window**: Smoothed confidence showing trends
- **High Confidence Regions**: Areas where branching is most likely (highlighted in red)

### Branching Tree
Shows all traces organized by depth:
- **Depth 0**: Initial traces
- **Depth 1+**: Traces spawned from high-confidence regions

## Performance Tips

1. **For faster testing**: Use fewer branches and shorter max_tokens
   ```bash
   python run_branching_test.py --initial_branches 2 --max_total_branches 4 --max_tokens 2000
   ```

2. **For more thorough exploration**: Increase branches and lower threshold
   ```bash
   python run_branching_test.py --initial_branches 4 --max_total_branches 12 --confidence_threshold 1.2
   ```

3. **If running out of memory**: Reduce max_tokens or number of branches
   ```bash
   python run_branching_test.py --max_tokens 3000 --max_total_branches 4
   ```

## Troubleshooting

### GPU Memory Issues
If you encounter CUDA OOM errors:
- Reduce `--max_tokens` (try 2000 or 3000)
- Reduce `--max_total_branches` (try 4)
- Use an even smaller model

### Model Download Issues
First-time run will download the model. Ensure you have:
- Internet connection
- Sufficient disk space (~3GB for default model)
- HuggingFace access (some models may require authentication)

### Import Errors
Ensure you have installed deepconf:
```bash
pip install -e .
```

And required dependencies:
```bash
pip install vllm transformers matplotlib numpy
```

## Advanced Usage

### Batch Processing Multiple Questions
Create a script to process multiple questions:

```python
questions = [
    "What is 15% of 240?",
    "Calculate 7 * 8 + 3",
    "What is the square root of 144?"
]

for i, q in enumerate(questions):
    os.system(f'python run_branching_test.py --question "{q}"')
```

### Custom Analysis
Load the pickle files for custom analysis:

```python
import pickle

with open('outputs/branching_test_TIMESTAMP.pkl', 'rb') as f:
    results = pickle.load(f)

# Access trace data
for trace in results['all_traces']:
    print(f"Trace {trace['trace_id']}: {len(trace['confs'])} tokens")
```

## Notes

- The script uses a non-interactive matplotlib backend for server compatibility
- Logs are written to both file and console simultaneously
- All directories are created automatically if they don't exist
- Each run generates unique timestamped files
