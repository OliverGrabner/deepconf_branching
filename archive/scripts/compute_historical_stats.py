"""
Compute Historical Token Statistics for AIME 2025

This script generates a small number of traces (2) for each question to establish
baseline statistics about average token usage. These statistics are used by the
branching self-consistency algorithm to determine when to stop branching.

Usage:
    python scripts/compute_historical_stats.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

Output:
    historical_stats/aime25_token_stats.json
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm

# Add parent directory to path to import local deepconf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from datasets import load_dataset
from vllm import SamplingParams

from deepconf import DeepThinkLLM, prepare_prompt


def load_aime25(subset=None):
    """Load AIME 2025 dataset from Hugging Face"""
    if subset:
        ds = load_dataset("opencompass/AIME2025", name=subset, split="test")
        datasets = [(subset, ds)]
    else:
        ds1 = load_dataset("opencompass/AIME2025", name="AIME2025-I", split="test")
        ds2 = load_dataset("opencompass/AIME2025", name="AIME2025-II", split="test")
        datasets = [("AIME2025-I", ds1), ("AIME2025-II", ds2)]

    return datasets


def collect_token_stats(
    deep_llm: DeepThinkLLM,
    question: str,
    num_samples: int,
    sampling_params: SamplingParams,
    model_type: str = "deepseek"
) -> Dict[str, Any]:
    """
    Generate a small number of traces and collect token statistics

    Returns:
        Dictionary with avg_tokens and individual samples
    """
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

    # Extract token counts
    token_counts = [trace['num_tokens'] for trace in result.all_traces]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

    return {
        'avg_tokens': round(avg_tokens, 1),
        'samples': token_counts,
        'num_samples': len(token_counts)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute historical token statistics for AIME 2025',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                       help='Model path or name')
    parser.add_argument('--model_type', type=str, default="deepseek",
                       choices=["deepseek", "gpt"],
                       help='Model type for prompt formatting')
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                       help='Tensor parallel size (number of GPUs)')

    # Sampling parameters
    parser.add_argument('--num_samples', type=int, default=2,
                       help='Number of samples per question for statistics')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1.0,
                       help='Top-p sampling parameter')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Top-k sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=130000,
                       help='Maximum tokens per generation')

    # Dataset selection
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['AIME2025-I', 'AIME2025-II'],
                       help='Run on specific dataset only (default: both)')
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

    # Print header
    print("\n" + "="*80)
    print("COMPUTING HISTORICAL TOKEN STATISTICS FOR AIME 2025")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Samples per question: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"GPUs: {args.tensor_parallel_size}")
    print("="*80 + "\n")

    # Load datasets
    print("Loading AIME 2025 datasets...")
    datasets = load_aime25(args.dataset)

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
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=20,
    )

    # Collect statistics for all datasets
    all_stats = {}

    for dataset_name, dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name} ({len(dataset)} questions)")
        print('='*80)

        dataset_stats = {}

        # Determine range
        start = args.start_idx
        end = args.end_idx if args.end_idx is not None else len(dataset)

        for i in tqdm(range(start, end), desc=f"{dataset_name}"):
            question_data = dataset[i]
            question = question_data['question']

            print(f"\n[{i+1}/{len(dataset)}] Collecting stats for question {i}")
            print(f"Q: {question[:100]}...")

            # Collect statistics
            stats = collect_token_stats(
                deep_llm=deep_llm,
                question=question,
                num_samples=args.num_samples,
                sampling_params=sampling_params,
                model_type=args.model_type
            )

            dataset_stats[str(i)] = stats

            print(f"  Avg tokens: {stats['avg_tokens']:.1f}")
            print(f"  Samples: {stats['samples']}")

        all_stats[dataset_name] = dataset_stats

    # Calculate overall statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for dataset_name, dataset_stats in all_stats.items():
        all_avgs = [stats['avg_tokens'] for stats in dataset_stats.values()]
        dataset_mean = sum(all_avgs) / len(all_avgs) if all_avgs else 0
        dataset_min = min(all_avgs) if all_avgs else 0
        dataset_max = max(all_avgs) if all_avgs else 0

        print(f"\n{dataset_name}:")
        print(f"  Questions processed: {len(dataset_stats)}")
        print(f"  Mean avg tokens: {dataset_mean:.1f}")
        print(f"  Range: [{dataset_min:.1f}, {dataset_max:.1f}]")

    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"aime25_token_stats_{timestamp}.json")

    output_data = {
        'metadata': {
            'timestamp': timestamp,
            'model': args.model,
            'num_samples': args.num_samples,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k,
            'max_tokens': args.max_tokens,
            'model_type': args.model_type,
        },
        'statistics': all_stats
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Statistics saved to: {output_file}")
    print("="*80)

    # Also save a "latest" symlink for easy reference
    latest_file = os.path.join(args.output_dir, "aime25_token_stats_latest.json")
    with open(latest_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Latest version also saved to: {latest_file}")
    print("\nYou can now use this file with run_branching_sc_aime25.py")


if __name__ == "__main__":
    main()
