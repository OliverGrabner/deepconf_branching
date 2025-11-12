"""
Compare Traditional SC vs Branching SC vs Peak Branching SC

Creates three visualizations:
1. Total New Tokens Generated comparison
2. Overall Accuracy comparison
3. Individual Trace Accuracy comparison

Usage:
    python scripts/compare_experiments.py \
        --traditional results/traditional_*.json \
        --branching results/branching_*.json \
        --peak_branching results/peak_branching_*.json \
        --output_dir comparisons/
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_metrics(results: Dict[str, Any], experiment_type: str) -> Dict[str, List[float]]:
    """
    Extract metrics from results for each experiment type

    Returns:
        Dictionary with metrics:
        - total_tokens_new: Total NEW tokens generated per question
        - overall_accuracy: Overall accuracy (voted answer correct)
        - individual_accuracy: Individual trace accuracy
    """
    metrics = {
        'total_tokens_new': [],
        'overall_accuracy': [],
        'individual_accuracy': [],
        'question_indices': []
    }

    # Navigate through results structure
    # JSON has structure: {"metadata": {}, "results": {"gsm8k": [...]}}
    if 'results' in results:
        results_dict = results['results']
        for dataset_name, dataset_results in results_dict.items():
            if not isinstance(dataset_results, list):
                continue

            for question_result in dataset_results:
                # Question accuracy (voted answer)
                is_correct = question_result.get('is_correct', False)
                metrics['overall_accuracy'].append(1.0 if is_correct else 0.0)

                # Individual trace accuracy
                individual_acc = question_result.get('individual_trace_accuracy', 0.0)
                metrics['individual_accuracy'].append(individual_acc)

                # Total NEW tokens generated (varies by experiment type)
                stats = question_result.get('statistics', {})

                if experiment_type == 'traditional':
                    # Traditional: all tokens are new tokens
                    total_tokens = stats.get('total_tokens', 0)
                    metrics['total_tokens_new'].append(total_tokens)

                elif experiment_type == 'branching':
                    # Branching: use total_tokens_generated (only NEW tokens)
                    total_tokens_generated = stats.get('total_tokens_generated', 0)
                    metrics['total_tokens_new'].append(total_tokens_generated)

                elif experiment_type == 'peak_branching':
                    # Peak branching: use total_tokens_generated (only NEW tokens)
                    # Note: peak_branching_stats has the breakdown
                    peak_stats = question_result.get('peak_branching_stats', {})
                    total_tokens_generated = peak_stats.get('total_tokens_generated', 0)
                    # Fallback to statistics if not in peak_stats
                    if total_tokens_generated == 0:
                        total_tokens_generated = stats.get('total_tokens', 0)
                    metrics['total_tokens_new'].append(total_tokens_generated)

    return metrics


def create_comparison_plots(
    traditional_metrics: Dict[str, List[float]],
    branching_metrics: Dict[str, List[float]],
    peak_branching_metrics: Dict[str, List[float]],
    output_dir: str
):
    """Create three comparison visualizations"""

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Total New Tokens Generated
    fig, ax = plt.subplots(figsize=(12, 7))

    # Calculate averages
    traditional_avg = np.mean(traditional_metrics['total_tokens_new']) if traditional_metrics['total_tokens_new'] else 0
    branching_avg = np.mean(branching_metrics['total_tokens_new']) if branching_metrics['total_tokens_new'] else 0
    peak_branching_avg = np.mean(peak_branching_metrics['total_tokens_new']) if peak_branching_metrics['total_tokens_new'] else 0

    methods = ['Traditional SC', 'Branching SC', 'Peak Branching SC']
    averages = [traditional_avg, branching_avg, peak_branching_avg]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = ax.bar(methods, averages, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, avg in zip(bars, averages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(avg):,}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add savings percentage
    if traditional_avg > 0:
        branching_savings = ((traditional_avg - branching_avg) / traditional_avg) * 100
        peak_savings = ((traditional_avg - peak_branching_avg) / traditional_avg) * 100

        ax.text(1, branching_avg * 0.5, f'↓ {branching_savings:.1f}%\nsavings',
                ha='center', va='center', fontsize=11, fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        ax.text(2, peak_branching_avg * 0.5, f'↓ {peak_savings:.1f}%\nsavings',
                ha='center', va='center', fontsize=11, fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ax.set_ylabel('Average Total NEW Tokens Generated', fontsize=13, fontweight='bold')
    ax.set_title('Token Efficiency Comparison\n(Lower is Better)', fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(averages) * 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_tokens.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/comparison_tokens.png")

    # Plot 2: Overall Accuracy (Voted Answer)
    fig, ax = plt.subplots(figsize=(12, 7))

    traditional_acc = np.mean(traditional_metrics['overall_accuracy']) if traditional_metrics['overall_accuracy'] else 0
    branching_acc = np.mean(branching_metrics['overall_accuracy']) if branching_metrics['overall_accuracy'] else 0
    peak_branching_acc = np.mean(peak_branching_metrics['overall_accuracy']) if peak_branching_metrics['overall_accuracy'] else 0

    accuracies = [traditional_acc * 100, branching_acc * 100, peak_branching_acc * 100]

    bars = ax.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add improvement markers
    if traditional_acc > 0:
        branching_improvement = ((branching_acc - traditional_acc) / traditional_acc) * 100
        peak_improvement = ((peak_branching_acc - traditional_acc) / traditional_acc) * 100

        if branching_improvement > 0:
            ax.text(1, branching_acc * 100 * 0.5, f'↑ +{branching_improvement:.1f}%',
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))

        if peak_improvement > 0:
            ax.text(2, peak_branching_acc * 100 * 0.5, f'↑ +{peak_improvement:.1f}%',
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))

    ax.set_ylabel('Overall Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Overall Accuracy Comparison\n(Voted Answer Correctness)', fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_overall_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/comparison_overall_accuracy.png")

    # Plot 3: Individual Trace Accuracy
    fig, ax = plt.subplots(figsize=(12, 7))

    traditional_ind = np.mean(traditional_metrics['individual_accuracy']) if traditional_metrics['individual_accuracy'] else 0
    branching_ind = np.mean(branching_metrics['individual_accuracy']) if branching_metrics['individual_accuracy'] else 0
    peak_branching_ind = np.mean(peak_branching_metrics['individual_accuracy']) if peak_branching_metrics['individual_accuracy'] else 0

    individual_accs = [traditional_ind * 100, branching_ind * 100, peak_branching_ind * 100]

    bars = ax.bar(methods, individual_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, acc in zip(bars, individual_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add improvement markers
    if traditional_ind > 0:
        branching_ind_improvement = ((branching_ind - traditional_ind) / traditional_ind) * 100
        peak_ind_improvement = ((peak_branching_ind - traditional_ind) / traditional_ind) * 100

        if branching_ind_improvement > 0:
            ax.text(1, branching_ind * 100 * 0.5, f'↑ +{branching_ind_improvement:.1f}%',
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))

        if peak_ind_improvement > 0:
            ax.text(2, peak_branching_ind * 100 * 0.5, f'↑ +{peak_ind_improvement:.1f}%',
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))

    ax.set_ylabel('Individual Trace Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Individual Trace Accuracy Comparison\n(Percentage of Correct Individual Traces)', fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_individual_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/comparison_individual_accuracy.png")

    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Metric':<40} {'Traditional':<15} {'Branching':<15} {'Peak Branch':<15}")
    print("-"*70)
    print(f"{'Avg NEW Tokens Generated':<40} {traditional_avg:<15,.0f} {branching_avg:<15,.0f} {peak_branching_avg:<15,.0f}")
    print(f"{'Overall Accuracy':<40} {traditional_acc*100:<15.1f}% {branching_acc*100:<15.1f}% {peak_branching_acc*100:<15.1f}%")
    print(f"{'Individual Trace Accuracy':<40} {traditional_ind*100:<15.1f}% {branching_ind*100:<15.1f}% {peak_branching_ind*100:<15.1f}%")
    print("-"*70)

    if traditional_avg > 0:
        print(f"\nToken Savings vs Traditional:")
        print(f"  Branching SC: {((traditional_avg - branching_avg) / traditional_avg) * 100:.1f}%")
        print(f"  Peak Branching SC: {((traditional_avg - peak_branching_avg) / traditional_avg) * 100:.1f}%")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Compare Traditional, Branching, and Peak Branching SC')
    parser.add_argument('--traditional', type=str, required=True,
                       help='Path to traditional SC results JSON')
    parser.add_argument('--branching', type=str, required=True,
                       help='Path to branching SC results JSON')
    parser.add_argument('--peak_branching', type=str, required=True,
                       help='Path to peak branching SC results JSON')
    parser.add_argument('--output_dir', type=str, default='comparisons',
                       help='Output directory for comparison plots')

    args = parser.parse_args()

    print("Loading results...")
    print(f"  Traditional: {args.traditional}")
    print(f"  Branching: {args.branching}")
    print(f"  Peak Branching: {args.peak_branching}")

    # Load results
    traditional_results = load_results(args.traditional)
    branching_results = load_results(args.branching)
    peak_branching_results = load_results(args.peak_branching)

    # Extract metrics
    print("\nExtracting metrics...")
    traditional_metrics = extract_metrics(traditional_results, 'traditional')
    branching_metrics = extract_metrics(branching_results, 'branching')
    peak_branching_metrics = extract_metrics(peak_branching_results, 'peak_branching')

    print(f"  Traditional: {len(traditional_metrics['total_tokens_new'])} questions")
    print(f"  Branching: {len(branching_metrics['total_tokens_new'])} questions")
    print(f"  Peak Branching: {len(peak_branching_metrics['total_tokens_new'])} questions")

    # Create comparison plots
    print("\nGenerating comparison visualizations...")
    create_comparison_plots(
        traditional_metrics,
        branching_metrics,
        peak_branching_metrics,
        args.output_dir
    )

    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
