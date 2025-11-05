"""
Compute Historical Token Statistics for GSM8k (SAFE VERSION)

This version:
- Saves progress after EACH question
- Can resume from partial results
- Has timeout protection
- Skips questions that take too long

Usage:
    python scripts/compute_historical_stats_gsm8k.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
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


def load_gsm8k(split="test"):
    """Load GSM8k dataset"""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return ds


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
        description='Compute historical token statistics for GSM8k (SAFE VERSION)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

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
                       help='End at this question index (default: all 1319)')

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

    # Prepare output filename
    output_file = os.path.join(args.output_dir, "gsm8k_token_stats_latest.json")

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
    print("COMPUTING HISTORICAL TOKEN STATISTICS - GSM8K")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Samples per question: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Timeout: {args.timeout}s ({args.timeout/60:.1f} minutes)")
    print(f"Resume: {args.resume}")
    print("="*80 + "\n")

    # Load dataset
    print("Loading GSM8k test set...")
    dataset = load_gsm8k(split="test")
    print(f"Loaded {len(dataset)} questions")

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

    # Determine range
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(dataset)

    print(f"\nProcessing questions {start} to {end-1} ({end-start} total)")
    print('='*80)

    # Track statistics
    num_processed = 0
    num_skipped = 0
    num_timeout = 0

    for i in tqdm(range(start, end), desc="Processing GSM8k"):
        question_data = dataset[i]
        question = question_data['question']

        # Skip if already processed (when resuming)
        q_key = str(i)
        if q_key in statistics:
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
            print(f"  Using fallback value: 5000 tokens (typical for GSM8k)")
            statistics[q_key] = {
                'avg_tokens': 5000,
                'samples': [],
                'num_samples': 0,
                'timeout': True
            }
            num_timeout += 1
        else:
            statistics[q_key] = stats
            print(f"  Avg tokens: {stats['avg_tokens']:.1f}")
            num_processed += 1

        # Save after each question
        output_data = {
            'metadata': {
                'model': args.model,
                'num_samples': args.num_samples,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'top_k': args.top_k,
                'timeout_seconds': args.timeout,
                'dataset': 'GSM8k',
                'split': 'test',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'questions_processed': len(statistics),
                'questions_timeout': sum(1 for s in statistics.values() if s.get('timeout', False))
            },
            'statistics': statistics
        }

        save_stats(output_file, output_data)

    # Final summary
    print("\n" + "="*80)
    print("STATISTICS COLLECTION COMPLETE")
    print("="*80)
    print(f"Questions processed: {num_processed}")
    print(f"Questions skipped (already done): {num_skipped}")
    print(f"Questions timeout: {num_timeout}")
    print(f"Total in file: {len(statistics)}")
    print(f"\nSaved to: {output_file}")

    # Compute overall average
    valid_stats = [s for s in statistics.values() if not s.get('timeout', False)]
    if valid_stats:
        overall_avg = sum(s['avg_tokens'] for s in valid_stats) / len(valid_stats)
        print(f"\nOverall average tokens (excluding timeouts): {overall_avg:.1f}")

    print("="*80)


if __name__ == "__main__":
    main()
