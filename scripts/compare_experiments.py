"""
Compare Branching SC vs Traditional SC Performance

Creates 3-panel bar graph comparing:
1. Majority vote accuracy
2. Individual trace accuracy (aggregated correct answer choice)
3. Tokens generated

Usage:
    python scripts/compare_experiments.py
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_results(filepath):
    """Load experiment results from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_metrics(data, experiment_type):
    """
    Compute metrics from experiment data

    Returns:
        dict with keys:
            - majority_vote_accuracy: % of questions where voted answer is correct
            - individual_trace_accuracy: % of all traces that are individually correct
            - total_tokens_generated: total tokens generated across all questions
    """
    # Check if this is a stats summary file (has 'overall' key) or detailed results file (has 'results' key)
    if 'overall' in data:
        # Stats summary format
        overall = data.get('overall', {})
        by_dataset = data.get('by_dataset', {})

        # Extract metrics directly from summary
        majority_vote_acc = overall.get('accuracy', 0) * 100  # Convert to percentage

        # For individual trace accuracy, aggregate from by_dataset
        total_individual_acc = 0
        dataset_count = 0
        for dataset_name, dataset_stats in by_dataset.items():
            total_individual_acc += dataset_stats.get('avg_individual_trace_accuracy', 0)
            dataset_count += 1

        individual_trace_acc = (total_individual_acc / dataset_count * 100) if dataset_count > 0 else 0

        # Get total tokens - use total_tokens for stats summary
        total_tokens = overall.get('total_tokens', 0)
        total_questions = overall.get('num_questions', 0)

        # Estimate total traces (not directly in summary, compute from avg)
        total_traces = 0
        for dataset_name, dataset_stats in by_dataset.items():
            total_traces += dataset_stats.get('num_questions', 0) * 32  # Approximate

        return {
            'majority_vote_accuracy': majority_vote_acc,
            'individual_trace_accuracy': individual_trace_acc,
            'total_tokens_generated': total_tokens,
            'total_questions': total_questions,
            'total_traces': total_traces
        }

    else:
        # Detailed results format
        results = data.get('results', {})

        majority_correct = 0
        total_questions = 0
        individual_correct = 0
        total_traces = 0
        total_tokens = 0

        for dataset_name, questions in results.items():
            for question_result in questions:
                total_questions += 1

                # Majority vote accuracy
                if question_result.get('is_correct', False):
                    majority_correct += 1

                # Individual trace accuracy
                individual_acc = question_result.get('individual_trace_accuracy', 0)
                num_traces = question_result.get('num_valid_traces', 0)
                individual_correct += individual_acc * num_traces
                total_traces += num_traces

                # Tokens generated
                stats = question_result.get('statistics', {})
                if experiment_type == 'branching':
                    # Use total_tokens_generated (excludes inherited tokens)
                    total_tokens += stats.get('total_tokens_generated', stats.get('total_tokens', 0))
                else:
                    # Traditional SC: total_tokens is correct
                    total_tokens += stats.get('total_tokens', 0)

        majority_vote_acc = (majority_correct / total_questions * 100) if total_questions > 0 else 0
        individual_trace_acc = (individual_correct / total_traces * 100) if total_traces > 0 else 0

        return {
            'majority_vote_accuracy': majority_vote_acc,
            'individual_trace_accuracy': individual_trace_acc,
            'total_tokens_generated': total_tokens,
            'total_questions': total_questions,
            'total_traces': total_traces
        }


def create_comparison_plot(branching_metrics, traditional_metrics, output_path):
    """Create 3-panel bar graph comparing the two approaches"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Branching SC vs Traditional SC Comparison', fontsize=16, fontweight='bold')

    # Colors
    colors = ['#2E86AB', '#A23B72']  # Blue for branching, Purple for traditional

    # Panel 1: Majority Vote Accuracy
    ax1 = axes[0]
    majority_accs = [
        branching_metrics['majority_vote_accuracy'],
        traditional_metrics['majority_vote_accuracy']
    ]
    bars1 = ax1.bar(['Branching SC', 'Traditional SC'], majority_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Majority Vote Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, val in zip(bars1, majority_accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Panel 2: Individual Trace Accuracy
    ax2 = axes[1]
    individual_accs = [
        branching_metrics['individual_trace_accuracy'],
        traditional_metrics['individual_trace_accuracy']
    ]
    bars2 = ax2.bar(['Branching SC', 'Traditional SC'], individual_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Individual Trace Accuracy\n(Aggregated Correct Answer Choice)', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, val in zip(bars2, individual_accs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Panel 3: Tokens Generated
    ax3 = axes[2]
    tokens = [
        branching_metrics['total_tokens_generated'],
        traditional_metrics['total_tokens_generated']
    ]
    bars3 = ax3.bar(['Branching SC', 'Traditional SC'], tokens, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax3.set_ylabel('Total Tokens Generated', fontsize=12, fontweight='bold')
    ax3.set_title('Tokens Generated', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Format y-axis with commas
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Add value labels on bars
    for bar, val in zip(bars3, tokens):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add summary statistics as text box
    avg_tok_branch = branching_metrics['total_tokens_generated']/branching_metrics['total_questions'] if branching_metrics['total_questions'] > 0 else 0
    avg_tok_trad = traditional_metrics['total_tokens_generated']/traditional_metrics['total_questions'] if traditional_metrics['total_questions'] > 0 else 0

    stats_text = (
        f"Branching SC:\n"
        f"  Questions: {branching_metrics['total_questions']}\n"
        f"  Total Traces: {branching_metrics['total_traces']}\n"
        f"  Tokens/Question: {avg_tok_branch:.0f}\n\n"
        f"Traditional SC:\n"
        f"  Questions: {traditional_metrics['total_questions']}\n"
        f"  Total Traces: {traditional_metrics['total_traces']}\n"
        f"  Tokens/Question: {avg_tok_trad:.0f}"
    )

    fig.text(0.99, 0.02, stats_text, fontsize=9, family='monospace',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_path}")
    plt.close()


def print_summary(branching_metrics, traditional_metrics):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("="*80)

    print("\nBRANCHING SC:")
    print(f"  Majority Vote Accuracy:      {branching_metrics['majority_vote_accuracy']:.2f}%")
    print(f"  Individual Trace Accuracy:   {branching_metrics['individual_trace_accuracy']:.2f}%")
    print(f"  Total Tokens Generated:      {branching_metrics['total_tokens_generated']:,}")
    print(f"  Total Questions:             {branching_metrics['total_questions']}")
    print(f"  Total Traces:                {branching_metrics['total_traces']}")
    avg_tok_branch = branching_metrics['total_tokens_generated']/branching_metrics['total_questions'] if branching_metrics['total_questions'] > 0 else 0
    print(f"  Avg Tokens/Question:         {avg_tok_branch:.0f}")

    print("\nTRADITIONAL SC:")
    print(f"  Majority Vote Accuracy:      {traditional_metrics['majority_vote_accuracy']:.2f}%")
    print(f"  Individual Trace Accuracy:   {traditional_metrics['individual_trace_accuracy']:.2f}%")
    print(f"  Total Tokens Generated:      {traditional_metrics['total_tokens_generated']:,}")
    print(f"  Total Questions:             {traditional_metrics['total_questions']}")
    print(f"  Total Traces:                {traditional_metrics['total_traces']}")
    avg_tok_trad = traditional_metrics['total_tokens_generated']/traditional_metrics['total_questions'] if traditional_metrics['total_questions'] > 0 else 0
    print(f"  Avg Tokens/Question:         {avg_tok_trad:.0f}")

    print("\nCOMPARISON:")
    maj_diff = branching_metrics['majority_vote_accuracy'] - traditional_metrics['majority_vote_accuracy']
    ind_diff = branching_metrics['individual_trace_accuracy'] - traditional_metrics['individual_trace_accuracy']
    tok_diff = branching_metrics['total_tokens_generated'] - traditional_metrics['total_tokens_generated']
    tok_pct = (tok_diff / traditional_metrics['total_tokens_generated'] * 100) if traditional_metrics['total_tokens_generated'] > 0 else 0

    print(f"  Majority Vote Accuracy Δ:    {maj_diff:+.2f}%")
    print(f"  Individual Trace Accuracy Δ: {ind_diff:+.2f}%")
    print(f"  Tokens Generated Δ:          {tok_diff:+,} ({tok_pct:+.1f}%)")

    print("\n" + "="*80)


def main():
    # Hardcoded file paths
    branching_file = "outputs/branching_sc_stats_20251105_163947.json"
    traditional_file = "outputs/traditional_sc_stats_20251105_143014.json"

    # Alternative: try detailed versions if stats not found
    if not os.path.exists(branching_file):
        branching_file = "outputs/branching_sc_detailed_20251105_163947.json"
    if not os.path.exists(traditional_file):
        traditional_file = "outputs/traditional_sc_detailed_20251105_143014.json"

    print("="*80)
    print("LOADING EXPERIMENT RESULTS")
    print("="*80)

    # Load data
    print(f"\nLoading branching SC results from: {branching_file}")
    if not os.path.exists(branching_file):
        print(f"ERROR: File not found: {branching_file}")
        print("Please ensure the file exists in the outputs/ directory")
        sys.exit(1)

    branching_data = load_results(branching_file)
    print(f"✓ Loaded branching SC results")

    print(f"\nLoading traditional SC results from: {traditional_file}")
    if not os.path.exists(traditional_file):
        print(f"ERROR: File not found: {traditional_file}")
        print("Please ensure the file exists in the outputs/ directory")
        sys.exit(1)

    traditional_data = load_results(traditional_file)
    print(f"✓ Loaded traditional SC results")

    # Compute metrics
    print("\nComputing metrics...")
    branching_metrics = compute_metrics(branching_data, 'branching')
    traditional_metrics = compute_metrics(traditional_data, 'traditional')
    print("✓ Metrics computed")

    # Print summary
    print_summary(branching_metrics, traditional_metrics)

    # Create comparison plot
    print("\nGenerating comparison plot...")
    output_path = "outputs/comparison_branching_vs_traditional.png"
    os.makedirs("outputs", exist_ok=True)
    create_comparison_plot(branching_metrics, traditional_metrics, output_path)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
