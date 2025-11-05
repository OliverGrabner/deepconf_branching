"""
Unified Historical Token Statistics Collection (SAFE VERSION)

Collects token statistics for any supported dataset with safety features:
- Saves progress after EACH question
- Can resume from partial results
- Timeout protection (30 min per question default)
- Skips questions that take too long

Usage:
    # AIME2025
    python compute_stats.py --dataset AIME2025-I --num_samples 2

    # GSM8k (chunk 0-99)
    python compute_stats.py --dataset gsm8k --num_samples 2 --start_idx 0 --end_idx 100

    # Resume previous run
    python compute_stats.py --dataset gsm8k --num_samples 2 --resume
"""

import os
import sys
import json
import argparse
import signal
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from vllm import SamplingParams

from deepconf import DeepThinkLLM, prepare_prompt
from experiment_utils import load_dataset_by_name, get_question_and_ground_truth


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


def load_existing_stats(filepath: str) -> Dict[str, Any]:
    """Load existing stats if available"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {'metadata': {}, 'statistics': {}}


def save_stats(filepath: str, data: Dict[str, Any]):
    """Save stats to file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  → Saved to {filepath}")


def collect_token_stats_with_timeout(
    deep_llm: DeepThinkLLM,
    question: str,
    num_samples: int,
    sampling_params: SamplingParams,
    model_type: str,
    timeout_seconds: int = 1800  # 30 minutes default
) -> Dict[str, Any]:
    """
    Generate traces with timeout protection

    Returns None if timeout occurs
    """
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        # Prepare prompt
        prompt = prepare_prompt(question, deep_llm.tokenizer, model_type)

        # Generate traces
        result = deep_llm.deepthink(
            prompt=prompt,
            mode="offline",
            budget=num_samples,
            sampling_params=sampling_params,
            compute_multiple_voting=False
        )

        # Cancel timeout
        signal.alarm(0)

        # Extract token counts
        token_counts = [trace['num_tokens'] for trace in result.all_traces]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

        return {
            'avg_tokens': round(avg_tokens, 1),
            'samples': token_counts,
            'num_samples': len(token_counts),
            'timeout': False
        }

    except TimeoutException:
        signal.alarm(0)
        print(f"  ⚠️  TIMEOUT after {timeout_seconds}s!")
        return None

    except Exception as e:
        signal.alarm(0)
        print(f"  ⚠️  ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Unified Historical Token Statistics Collection (SAFE VERSION)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset configuration
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name: AIME2025-I, AIME2025-II, gsm8k, or both (for AIME)')

    # Model configuration
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument('--model_type', type=str, default="deepseek")
    parser.add_argument('--tensor_parallel_size', type=int, default=4)

    # Sampling parameters
    parser.add_argument('--num_samples', type=int, default=2,
                       help='Number of traces to generate per question')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--max_tokens', type=int, default=130000)

    # Safety parameters
    parser.add_argument('--timeout', type=int, default=1800,
                       help='Timeout per question in seconds (default: 30 min)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing progress file')

    # Dataset selection
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Start from this question index')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='End at this question index')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default="historical_stats",
                       help='Output directory for statistics')

    args = parser.parse_args()

    # Setup GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare output filename (dataset-specific)
    dataset_clean = args.dataset.lower().replace("-", "_")
    output_file = os.path.join(args.output_dir, f"{dataset_clean}_token_stats_latest.json")

    # Load existing stats if resuming
    if args.resume:
        print(f"\nAttempting to resume from: {output_file}")
        existing_data = load_existing_stats(output_file)
        statistics = existing_data.get('statistics', {})
        print(f"Loaded {len(statistics)} existing question statistics")
    else:
        statistics = {}

    # Print header
    print("\n" + "="*80)
    print("COMPUTING HISTORICAL TOKEN STATISTICS")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Samples per question: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Timeout: {args.timeout}s ({args.timeout/60:.1f} minutes)")
    print(f"Resume: {args.resume}")
    print("="*80 + "\n")

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    datasets = load_dataset_by_name(args.dataset, split="test")

    # Initialize model
    print(f"\nInitializing DeepThinkLLM with {args.model}...")
    deep_llm = DeepThinkLLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
        trust_remote_code=True
    )

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=20,
    )

    # Process all datasets
    for dataset_name, dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name}")
        print('='*80)

        # Determine range
        start = args.start_idx
        end = args.end_idx if args.end_idx is not None else len(dataset)

        print(f"Processing questions {start} to {end-1} ({end-start} total)")

        # Track statistics
        num_processed = 0
        num_skipped = 0
        num_timeout = 0

        for i in tqdm(range(start, end), desc=f"Processing {dataset_name}"):
            question_data = dataset[i]
            question, ground_truth = get_question_and_ground_truth(dataset_name, question_data)

            # For nested statistics (AIME), use dataset name as key
            # For flat statistics (GSM8k), use question index as key
            if "gsm8k" in dataset_name.lower():
                stats_key = str(i)
                if stats_key in statistics:
                    print(f"\nQ{i}: Already processed, skipping...")
                    num_skipped += 1
                    continue
            else:
                # AIME format: nested by dataset
                if dataset_name not in statistics:
                    statistics[dataset_name] = {}
                stats_key = str(i)
                if stats_key in statistics[dataset_name]:
                    print(f"\nQ{i}: Already processed, skipping...")
                    num_skipped += 1
                    continue

            print(f"\nQ{i}: {question[:100]}...")

            # Collect token statistics with timeout
            stats = collect_token_stats_with_timeout(
                deep_llm,
                question,
                args.num_samples,
                sampling_params,
                args.model_type,
                args.timeout
            )

            if stats is None:
                # Timeout or error - use fallback value
                fallback = 5000 if "gsm8k" in dataset_name.lower() else 8000
                print(f"  Using fallback value: {fallback} tokens")

                stats = {
                    'avg_tokens': fallback,
                    'samples': [],
                    'num_samples': 0,
                    'timeout': True
                }
                num_timeout += 1
            else:
                print(f"  Avg tokens: {stats['avg_tokens']:.1f}")
                num_processed += 1

            # Store statistics
            if "gsm8k" in dataset_name.lower():
                statistics[stats_key] = stats
            else:
                statistics[dataset_name][stats_key] = stats

            # Save after each question
            output_data = {
                'metadata': {
                    'model': args.model,
                    'num_samples': args.num_samples,
                    'temperature': args.temperature,
                    'top_p': args.top_p,
                    'top_k': args.top_k,
                    'timeout_seconds': args.timeout,
                    'dataset': args.dataset,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                'statistics': statistics
            }

            # Calculate questions processed and timeouts
            if "gsm8k" in args.dataset.lower():
                total_processed = len(statistics)
                total_timeout = sum(1 for s in statistics.values() if s.get('timeout', False))
            else:
                total_processed = sum(len(ds_stats) for ds_stats in statistics.values())
                total_timeout = sum(
                    1 for ds_stats in statistics.values()
                    for s in ds_stats.values() if s.get('timeout', False)
                )

            output_data['metadata']['questions_processed'] = total_processed
            output_data['metadata']['questions_timeout'] = total_timeout

            save_stats(output_file, output_data)

    # Final summary
    print("\n" + "="*80)
    print("STATISTICS COLLECTION COMPLETE")
    print("="*80)
    print(f"Questions processed: {num_processed}")
    print(f"Questions skipped (already done): {num_skipped}")
    print(f"Questions timeout: {num_timeout}")
    print(f"\nSaved to: {output_file}")

    # Compute overall average
    if "gsm8k" in args.dataset.lower():
        valid_stats = [s for s in statistics.values() if not s.get('timeout', False)]
    else:
        valid_stats = [
            s for ds_stats in statistics.values()
            for s in ds_stats.values() if not s.get('timeout', False)
        ]

    if valid_stats:
        overall_avg = sum(s['avg_tokens'] for s in valid_stats) / len(valid_stats)
        print(f"Overall average tokens (excluding timeouts): {overall_avg:.1f}")

    print("="*80)


if __name__ == "__main__":
    main()
