#!/usr/bin/env python3
"""
Create historical stats from Traditional SC results for use in Branching SC

Usage:
    python scripts/create_historical_stats_from_traditional.py \
        --traditional_results outputs/traditional_sc_detailed_*.json \
        --output historical_stats/gsm8k_from_traditional.json
"""

import json
import argparse
import numpy as np
from pathlib import Path


def extract_token_stats(traditional_results_path: str, output_path: str):
    """
    Extract per-question token statistics from Traditional SC results
    to create historical stats for Branching SC
    """
    print(f"Loading Traditional SC results from: {traditional_results_path}")

    with open(traditional_results_path, 'r') as f:
        data = json.load(f)

    # Extract dataset name
    if 'results' not in data:
        raise ValueError("Invalid results file format")

    dataset_names = list(data['results'].keys())
    if not dataset_names:
        raise ValueError("No dataset results found")

    dataset_name = dataset_names[0]
    results = data['results'][dataset_name]

    print(f"Dataset: {dataset_name}")
    print(f"Questions: {len(results)}")

    # Create historical stats structure (format expected by experiment_utils.py)
    # Must have 'statistics' key with question_idx as keys
    statistics = {}

    # Extract token stats for each question
    for i, question_result in enumerate(results):
        valid_traces = question_result.get('valid_traces', [])

        # Get token counts from valid traces
        token_counts = [t.get('num_tokens', 0) for t in valid_traces if t.get('num_tokens', 0) > 0]

        if token_counts:
            avg_tokens = int(np.mean(token_counts))
            median_tokens = int(np.median(token_counts))
            max_tokens = int(np.max(token_counts))
            min_tokens = int(np.min(token_counts))
        else:
            # Fallback if no token data
            avg_tokens = 2500  # GSM8K default
            median_tokens = 2500
            max_tokens = 5000
            min_tokens = 1000

        statistics[str(i)] = {
            'avg_tokens': avg_tokens,
            'median_tokens': median_tokens,
            'max_tokens': max_tokens,
            'min_tokens': min_tokens,
            'num_traces': len(token_counts)
        }

    # Create final structure with 'statistics' key that experiment_utils.py expects
    historical_stats = {
        'dataset': dataset_name,
        'source': 'traditional_sc',
        'num_questions': len(results),
        'statistics': statistics  # This is the key format expected by load_historical_stats()
    }

    # Calculate overall statistics for display
    all_avgs = [stats['avg_tokens'] for stats in statistics.values()]
    overall = {
        'avg_tokens': int(np.mean(all_avgs)),
        'median_tokens': int(np.median(all_avgs)),
        'max_tokens': int(np.max(all_avgs)),
        'min_tokens': int(np.min(all_avgs))
    }

    # Save historical stats
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(historical_stats, f, indent=2)

    print(f"\n✓ Historical stats saved to: {output_path}")
    print(f"\nOverall Statistics:")
    print(f"  Average tokens per question: {overall['avg_tokens']}")
    print(f"  Median tokens per question: {overall['median_tokens']}")
    print(f"  Range: {overall['min_tokens']} - {overall['max_tokens']}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Create historical stats from Traditional SC results for Branching SC'
    )
    parser.add_argument('--traditional_results', type=str, required=True,
                       help='Path to traditional SC results JSON file')
    parser.add_argument('--output', type=str,
                       default='historical_stats/gsm8k_from_traditional.json',
                       help='Output path for historical stats JSON')

    args = parser.parse_args()

    extract_token_stats(args.traditional_results, args.output)

    print("\n" + "="*60)
    print("Now you can run Branching SC with:")
    print(f"python scripts/run_experiment.py --experiment branching --dataset gsm8k \\")
    print(f"    --start_traces 8 --max_traces 32 --start_idx 0 --end_idx 50 \\")
    print(f"    --historical_stats {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
