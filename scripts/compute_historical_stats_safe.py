"""
Compute Historical Token Statistics for AIME 2025 (SAFE VERSION)

This version:
- Saves progress after EACH question
- Can resume from partial results
- Has timeout protection
- Skips questions that take too long

Usage:
    python scripts/compute_historical_stats_safe.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
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
from datasets import load_dataset
from vllm import SamplingParams

from deepconf import DeepThinkLLM, prepare_prompt


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


def load_aime25(subset=None):
    """Load AIME 2025 dataset"""
    if subset:
        ds = load_dataset("opencompass/AIME2025", name=subset, split="test")
        datasets = [(subset, ds)]
    else:
        ds1 = load_dataset("opencompass/AIME2025", name="AIME2025-I", split="test")
        ds2 = load_dataset("opencompass/AIME2025", name="AIME2025-II", split="test")
        datasets = [("AIME2025-I", ds1), ("AIME2025-II", ds2)]
    return datasets


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
        description='Compute historical token statistics (SAFE VERSION)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument('--model_type', type=str, default="deepseek")
    parser.add_argument('--tensor_parallel_size', type=int, default=4)

    # Sampling parameters
    parser.add_argument('--num_samples', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--max_tokens', type=int, default=130000)

    # Safety parameters
    parser.add_argument('--timeout', type=int, default=1800,
                       help='Timeout per question in seconds (default: 30 min)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing progress file')

    # Dataset selection
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['AIME2025-I', 'AIME2025-II'])
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)

    # Output
    parser.add_argument('--output_dir', type=str, default="historical_stats")

    args = parser.parse_args()

    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"aime25_token_stats_{timestamp}.json")

    # Load existing progress if resuming
    if args.resume:
        latest_file = os.path.join(args.output_dir, "aime25_token_stats_latest.json")
        if os.path.exists(latest_file):
            print(f"Resuming from: {latest_file}")
            all_stats = load_existing_stats(latest_file)
        else:
            print("No existing progress found, starting fresh")
            all_stats = {'metadata': {}, 'statistics': {}}
    else:
        all_stats = {'metadata': {}, 'statistics': {}}

    # Update metadata
    all_stats['metadata'] = {
        'timestamp': timestamp,
        'model': args.model,
        'num_samples': args.num_samples,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'timeout_seconds': args.timeout
    }

    print("\n" + "="*80)
    print("COMPUTING HISTORICAL TOKEN STATISTICS (SAFE VERSION)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Samples per question: {args.num_samples}")
    print(f"Timeout per question: {args.timeout}s ({args.timeout/60:.1f} min)")
    print(f"Output: {output_file}")
    print("="*80 + "\n")

    # Load datasets
    datasets = load_aime25(args.dataset)

    # Initialize model
    print("Initializing model...")
    deep_llm = DeepThinkLLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=20,
    )

    # Process datasets
    for dataset_name, dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name}")
        print('='*80)

        # Initialize dataset stats if not exists
        if dataset_name not in all_stats['statistics']:
            all_stats['statistics'][dataset_name] = {}

        start = args.start_idx
        end = args.end_idx if args.end_idx is not None else len(dataset)

        for i in range(start, end):
            q_key = str(i)

            # Skip if already done
            if args.resume and q_key in all_stats['statistics'][dataset_name]:
                print(f"[{i+1}/{len(dataset)}] Question {i} - ALREADY DONE, skipping")
                continue

            question_data = dataset[i]
            question = question_data['question']

            print(f"\n[{i+1}/{len(dataset)}] Question {i}")
            print(f"Q: {question[:100]}...")

            # Collect with timeout
            stats = collect_token_stats_with_timeout(
                deep_llm=deep_llm,
                question=question,
                num_samples=args.num_samples,
                sampling_params=sampling_params,
                model_type=args.model_type,
                timeout_seconds=args.timeout
            )

            if stats is None:
                # Timeout or error - use fallback value
                print(f"  Using fallback avg_tokens=50000")
                stats = {
                    'avg_tokens': 50000.0,
                    'samples': [],
                    'num_samples': 0,
                    'timeout': True,
                    'note': 'Timeout or error - using fallback'
                }
            else:
                print(f"  ✓ Avg tokens: {stats['avg_tokens']:.1f}")
                print(f"  Samples: {stats['samples']}")

            # Save immediately
            all_stats['statistics'][dataset_name][q_key] = stats
            save_stats(output_file, all_stats)

            # Also update latest
            latest_file = os.path.join(args.output_dir, "aime25_token_stats_latest.json")
            save_stats(latest_file, all_stats)

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for dataset_name, dataset_stats in all_stats['statistics'].items():
        completed = [k for k, v in dataset_stats.items() if not v.get('timeout', False)]
        timeouts = [k for k, v in dataset_stats.items() if v.get('timeout', False)]

        print(f"\n{dataset_name}:")
        print(f"  Completed: {len(completed)}/{len(dataset_stats)}")
        if timeouts:
            print(f"  Timeouts: {len(timeouts)} (questions: {', '.join(timeouts)})")

        if completed:
            valid_avgs = [dataset_stats[k]['avg_tokens'] for k in completed]
            print(f"  Mean avg tokens: {sum(valid_avgs)/len(valid_avgs):.1f}")

    print(f"\n✓ Final results saved to: {output_file}")
    print(f"✓ Latest version: {latest_file}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
