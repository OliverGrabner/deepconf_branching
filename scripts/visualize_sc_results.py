"""
Visualization script for Traditional Self-Consistency results

Creates plots and visualizations to understand SC performance

Usage:
    python visualize_sc_results.py outputs_sc/traditional_sc_aime25_detailed_TIMESTAMP.json
"""

import json
import argparse
import pandas as pd
import numpy as np
from collections import Counter


def print_ascii_bar_chart(data, title, max_width=60):
    """Create ASCII bar chart"""
    print(f"\n{title}")
    print("=" * (max_width + 20))

    if not data:
        print("No data available")
        return

    max_value = max(data.values())

    for label, value in sorted(data.items(), key=lambda x: x[1], reverse=True):
        bar_length = int((value / max_value) * max_width) if max_value > 0 else 0
        bar = "█" * bar_length
        print(f"{str(label)[:15]:>15s} | {bar} {value}")


def print_ascii_histogram(values, title, num_bins=10, max_width=50):
    """Create ASCII histogram"""
    print(f"\n{title}")
    print("=" * (max_width + 30))

    if not values:
        print("No data available")
        return

    # Create bins
    min_val, max_val = min(values), max(values)
    bin_width = (max_val - min_val) / num_bins if max_val > min_val else 1
    bins = [min_val + i * bin_width for i in range(num_bins + 1)]

    # Count values in each bin
    counts = [0] * num_bins
    for val in values:
        for i in range(num_bins):
            if bins[i] <= val < bins[i + 1] or (i == num_bins - 1 and val == bins[i + 1]):
                counts[i] += 1
                break

    max_count = max(counts) if counts else 1

    for i in range(num_bins):
        bar_length = int((counts[i] / max_count) * max_width) if max_count > 0 else 0
        bar = "█" * bar_length
        range_str = f"{bins[i]:.2f}-{bins[i+1]:.2f}"
        print(f"{range_str:>15s} | {bar} {counts[i]}")


def visualize_accuracy_comparison(results):
    """Visualize individual vs voting accuracy"""
    print("\n" + "="*80)
    print("ACCURACY COMPARISON: Individual Traces vs Voting")
    print("="*80)

    for dataset_name, questions in results.items():
        individual_accs = [q['individual_trace_accuracy'] for q in questions]
        voting_accs = [1.0 if q['is_correct'] else 0.0 for q in questions]

        avg_individual = np.mean(individual_accs)
        avg_voting = np.mean(voting_accs)

        print(f"\n{dataset_name}:")
        print(f"  Individual: {'█' * int(avg_individual * 50)} {avg_individual:.1%}")
        print(f"  Voting:     {'█' * int(avg_voting * 50)} {avg_voting:.1%}")
        print(f"  Improvement: {avg_voting - avg_individual:+.1%}")


def visualize_consensus_distribution(results):
    """Visualize distribution of vote consensus"""
    print("\n" + "="*80)
    print("VOTE CONSENSUS DISTRIBUTION")
    print("="*80)

    all_consensus = []
    for dataset_name, questions in results.items():
        for q in questions:
            if q['vote_distribution']:
                total = sum(q['vote_distribution'].values())
                max_votes = max(q['vote_distribution'].values())
                consensus = max_votes / total if total > 0 else 0
                all_consensus.append(consensus)

    if all_consensus:
        print_ascii_histogram(all_consensus, "Distribution of Consensus Scores", num_bins=10)

        print(f"\nStatistics:")
        print(f"  Mean: {np.mean(all_consensus):.1%}")
        print(f"  Median: {np.median(all_consensus):.1%}")
        print(f"  Std Dev: {np.std(all_consensus):.1%}")


def visualize_correctness_by_consensus(results):
    """Show correctness rate by consensus level"""
    print("\n" + "="*80)
    print("CORRECTNESS BY CONSENSUS LEVEL")
    print("="*80)

    # Group by consensus buckets
    buckets = {
        "Very High (>90%)": [],
        "High (75-90%)": [],
        "Medium (60-75%)": [],
        "Low (40-60%)": [],
        "Very Low (<40%)": []
    }

    for dataset_name, questions in results.items():
        for q in questions:
            if q['vote_distribution']:
                total = sum(q['vote_distribution'].values())
                max_votes = max(q['vote_distribution'].values())
                consensus = max_votes / total if total > 0 else 0

                if consensus > 0.9:
                    buckets["Very High (>90%)"].append(q['is_correct'])
                elif consensus > 0.75:
                    buckets["High (75-90%)"].append(q['is_correct'])
                elif consensus > 0.6:
                    buckets["Medium (60-75%)"].append(q['is_correct'])
                elif consensus > 0.4:
                    buckets["Low (40-60%)"].append(q['is_correct'])
                else:
                    buckets["Very Low (<40%)"].append(q['is_correct'])

    # Calculate accuracy for each bucket
    bucket_stats = {}
    for bucket_name, correct_list in buckets.items():
        if correct_list:
            accuracy = sum(correct_list) / len(correct_list)
            bucket_stats[f"{bucket_name} (n={len(correct_list)})"] = accuracy

    print_ascii_bar_chart(
        {k: v * 100 for k, v in bucket_stats.items()},
        "Accuracy (%) by Consensus Level",
        max_width=60
    )

    print("\nInsight: Higher consensus → Higher accuracy (usually)")


def visualize_answer_diversity(results):
    """Visualize answer diversity patterns"""
    print("\n" + "="*80)
    print("ANSWER DIVERSITY")
    print("="*80)

    diversity_counts = Counter()

    for dataset_name, questions in results.items():
        for q in questions:
            if q['vote_distribution']:
                num_unique = len(q['vote_distribution'])
                diversity_counts[num_unique] += 1

    print_ascii_bar_chart(
        dict(diversity_counts),
        "Number of Questions by Unique Answers",
        max_width=40
    )

    print("\nInsight:")
    print("  Few unique answers (1-3) → Model converged, high confidence")
    print("  Many unique answers (>10) → Model uncertain, difficult question")


def visualize_tokens_and_time(results):
    """Visualize computational costs"""
    print("\n" + "="*80)
    print("COMPUTATIONAL COSTS")
    print("="*80)

    for dataset_name, questions in results.items():
        tokens = [q['statistics']['total_tokens'] for q in questions]
        times = [q['statistics']['total_time'] for q in questions]

        print(f"\n{dataset_name}:")
        print(f"  Tokens per question:")
        print(f"    Mean: {np.mean(tokens):,.0f}")
        print(f"    Median: {np.median(tokens):,.0f}")
        print(f"    Min/Max: {np.min(tokens):,.0f} / {np.max(tokens):,.0f}")

        print(f"\n  Time per question:")
        print(f"    Mean: {np.mean(times):.1f}s")
        print(f"    Median: {np.median(times):.1f}s")
        print(f"    Min/Max: {np.min(times):.1f}s / {np.max(times):.1f}s")


def visualize_failure_patterns(results):
    """Analyze patterns in failures"""
    print("\n" + "="*80)
    print("FAILURE PATTERN ANALYSIS")
    print("="*80)

    failure_reasons = {
        "Low individual accuracy (<30%)": 0,
        "Medium individual accuracy (30-50%)": 0,
        "High individual accuracy (>50%, voting failed!)": 0
    }

    all_failures = []

    for dataset_name, questions in results.items():
        for i, q in enumerate(questions):
            if not q['is_correct']:
                all_failures.append(q)

                ind_acc = q['individual_trace_accuracy']
                if ind_acc < 0.3:
                    failure_reasons["Low individual accuracy (<30%)"] += 1
                elif ind_acc < 0.5:
                    failure_reasons["Medium individual accuracy (30-50%)"] += 1
                else:
                    failure_reasons["High individual accuracy (>50%, voting failed!)"] += 1

    print_ascii_bar_chart(
        failure_reasons,
        "Failure Breakdown by Individual Trace Accuracy",
        max_width=50
    )

    print(f"\nTotal failures: {len(all_failures)}")


def create_summary_table(data):
    """Create a nice summary table"""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY TABLE")
    print("="*80)

    summary = data['summary']
    metadata = data['metadata']

    print(f"\nModel: {metadata['model']}")
    print(f"Number of traces: {metadata['num_traces']}")
    print(f"Temperature: {metadata['temperature']}")
    print(f"Date: {metadata['timestamp']}")

    print("\n" + "-"*80)
    print(f"{'Dataset':<20} {'Questions':<10} {'Correct':<10} {'Accuracy':<12} {'Tokens':<12}")
    print("-"*80)

    for dataset_name, stats in summary['by_dataset'].items():
        print(f"{dataset_name:<20} "
              f"{stats['num_questions']:<10} "
              f"{stats['num_correct']:<10} "
              f"{stats['accuracy']:.1%}      "
              f"{stats['total_tokens']:>10,}")

    overall = summary['overall']
    print("-"*80)
    print(f"{'TOTAL':<20} "
          f"{overall['num_questions']:<10} "
          f"{overall['num_correct']:<10} "
          f"{overall['accuracy']:.1%}      "
          f"{overall['total_tokens']:>10,}")
    print("-"*80)

    print(f"\nTotal Time: {overall['total_time']:.1f}s ({overall['total_time']/60:.1f} minutes)")
    print(f"Throughput: {overall['throughput_tokens_per_sec']:.1f} tokens/second")


def main():
    parser = argparse.ArgumentParser(description='Visualize Traditional SC results')
    parser.add_argument('results_file', type=str,
                       help='Path to detailed results JSON file')

    args = parser.parse_args()

    # Load results
    print("="*80)
    print("TRADITIONAL SELF-CONSISTENCY VISUALIZATION")
    print("="*80)
    print(f"\nLoading results from: {args.results_file}")

    with open(args.results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data['results']

    # Create visualizations
    create_summary_table(data)
    visualize_accuracy_comparison(results)
    visualize_consensus_distribution(results)
    visualize_correctness_by_consensus(results)
    visualize_answer_diversity(results)
    visualize_tokens_and_time(results)
    visualize_failure_patterns(results)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print("\nFor more detailed analysis, use:")
    print(f"  python analyze_sc_results.py {args.results_file}")


if __name__ == "__main__":
    main()
