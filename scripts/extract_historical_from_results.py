"""
Extract Historical Token Statistics from Traditional SC Results

Takes a traditional SC results JSON file and extracts token statistics
to create a historical stats file that can be used for branching SC.

Usage:
    python scripts/extract_historical_from_results.py \
        --results outputs/traditional_sc_detailed_20250115_143022.json \
        --output historical_stats/gsm8k_token_stats_from_sc.json

This is useful if you've already run traditional SC and want to use
those token counts for branching SC without re-running compute_stats.py
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_results(filepath: str) -> Dict[str, Any]:
    """Load experiment results from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_token_statistics(results_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract token statistics from traditional SC results

    Args:
        results_data: Loaded results JSON

    Returns:
        Dictionary in historical stats format
    """
    metadata = results_data.get('metadata', {})
    results_by_dataset = results_data.get('results', {})

    # Determine if this is AIME (nested) or GSM8k (flat)
    first_dataset = next(iter(results_by_dataset.keys()), None)
    is_aime = first_dataset and 'AIME' in first_dataset

    statistics = {}

    for dataset_name, questions in results_by_dataset.items():
        print(f"\nProcessing {dataset_name}...")
        print(f"  Found {len(questions)} questions")

        if is_aime:
            # AIME format: nested by dataset name
            if dataset_name not in statistics:
                statistics[dataset_name] = {}

            for q_idx, question_result in enumerate(questions):
                token_counts = extract_token_counts_from_question(question_result)

                if token_counts:
                    avg_tokens = sum(token_counts) / len(token_counts)
                    statistics[dataset_name][str(q_idx)] = {
                        'avg_tokens': round(avg_tokens, 1),
                        'samples': token_counts,
                        'num_samples': len(token_counts),
                        'timeout': False,
                        'source': 'extracted_from_traditional_sc'
                    }
                    print(f"  Q{q_idx}: {avg_tokens:.1f} avg tokens ({len(token_counts)} samples)")
                else:
                    print(f"  Q{q_idx}: No token data found")

        else:
            # GSM8k format: flat by question index
            for q_idx, question_result in enumerate(questions):
                token_counts = extract_token_counts_from_question(question_result)

                if token_counts:
                    avg_tokens = sum(token_counts) / len(token_counts)
                    statistics[str(q_idx)] = {
                        'avg_tokens': round(avg_tokens, 1),
                        'samples': token_counts,
                        'num_samples': len(token_counts),
                        'timeout': False,
                        'source': 'extracted_from_traditional_sc'
                    }
                    if q_idx % 50 == 0:  # Print every 50th for GSM8k (too many otherwise)
                        print(f"  Q{q_idx}: {avg_tokens:.1f} avg tokens ({len(token_counts)} samples)")
                else:
                    print(f"  Q{q_idx}: No token data found")

    return statistics


def extract_token_counts_from_question(question_result: Dict[str, Any]) -> List[int]:
    """
    Extract token counts from a single question's results

    Args:
        question_result: Question result dictionary

    Returns:
        List of token counts from all traces
    """
    token_counts = []

    # Check valid_traces first (traditional SC format)
    valid_traces = question_result.get('valid_traces', [])
    if valid_traces:
        for trace in valid_traces:
            num_tokens = trace.get('num_tokens', 0)
            if num_tokens > 0:
                token_counts.append(num_tokens)

    # Fallback: check full_traces (branching SC format, shouldn't happen for traditional)
    if not token_counts:
        full_traces = question_result.get('full_traces', [])
        for trace in full_traces:
            num_tokens = trace.get('num_tokens', 0)
            if num_tokens > 0:
                token_counts.append(num_tokens)

    # Fallback: check all_traces (old format)
    if not token_counts:
        all_traces = question_result.get('all_traces', [])
        for trace in all_traces:
            num_tokens = trace.get('num_tokens', 0)
            if num_tokens > 0:
                token_counts.append(num_tokens)

    return token_counts


def save_historical_stats(
    statistics: Dict[str, Any],
    output_path: str,
    source_metadata: Dict[str, Any]
):
    """
    Save statistics in historical stats format

    Args:
        statistics: Statistics dictionary
        output_path: Output file path
        source_metadata: Metadata from source results
    """
    # Determine dataset name
    if any('AIME' in key for key in statistics.keys()):
        dataset_name = 'AIME2025'
        total_questions = sum(len(ds_stats) for ds_stats in statistics.values())
    else:
        dataset_name = 'GSM8k'
        total_questions = len(statistics)

    output_data = {
        'metadata': {
            'source': 'extracted_from_traditional_sc',
            'source_file': source_metadata.get('timestamp', 'unknown'),
            'model': source_metadata.get('model', 'unknown'),
            'num_samples': source_metadata.get('num_traces', 'unknown'),
            'temperature': source_metadata.get('temperature', 'unknown'),
            'dataset': dataset_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'questions_processed': total_questions,
            'questions_timeout': 0,
            'extraction_note': 'Token statistics extracted from traditional SC results'
        },
        'statistics': statistics
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Historical statistics saved to: {output_path}")
    print(f"  Total questions: {total_questions}")

    # Calculate overall average
    if any('AIME' in key for key in statistics.keys()):
        all_avgs = [
            s['avg_tokens'] for ds_stats in statistics.values()
            for s in ds_stats.values()
        ]
    else:
        all_avgs = [s['avg_tokens'] for s in statistics.values()]

    if all_avgs:
        overall_avg = sum(all_avgs) / len(all_avgs)
        print(f"  Overall average tokens: {overall_avg:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract Historical Token Statistics from Traditional SC Results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--results', type=str, required=True,
                       help='Path to traditional SC results JSON file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for historical statistics JSON')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite output file if it exists')

    args = parser.parse_args()

    # Check if output exists
    if os.path.exists(args.output) and not args.force:
        print(f"Error: Output file already exists: {args.output}")
        print("Use --force to overwrite")
        return 1

    # Load results
    print(f"Loading results from: {args.results}")
    results_data = load_results(args.results)

    # Check experiment type
    metadata = results_data.get('metadata', {})
    experiment_type = metadata.get('experiment_type', 'unknown')

    if experiment_type == 'branching':
        print("\n⚠️  Warning: This appears to be a branching SC results file.")
        print("Branching SC doesn't need extracted stats (it already used historical stats).")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0

    print(f"\nExtracting token statistics from {experiment_type} SC results...")

    # Extract statistics
    statistics = extract_token_statistics(results_data)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save historical stats
    save_historical_stats(statistics, args.output, metadata)

    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nYou can now use this file for branching SC:")
    print(f"  python scripts/run_experiment.py \\")
    print(f"      --experiment branching \\")
    print(f"      --dataset <name> \\")
    print(f"      --start_traces 8 \\")
    print(f"      --max_traces 32 \\")
    print(f"      --historical_stats {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
