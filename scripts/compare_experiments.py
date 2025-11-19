"""
Compare Traditional SC vs Branching SC vs Peak Branching SC

Creates visualizations comparing:
1. Total New Tokens Generated comparison
2. Overall Accuracy comparison
3. Individual Trace Accuracy comparison
4. Individual vs Branched Accuracy comparison (new)
5. Chain of Thought Length comparison (new)

Usage:
    # Auto-select most recent files (all questions):
    python scripts/compare_experiments.py --output_dir comparisons/

    # Auto-select and compare only first 50 questions:
    python scripts/compare_experiments.py --max_questions 50 --output_dir comparisons/

    # Or specify files manually:
    python scripts/compare_experiments.py \
        --traditional results/traditional_*.json \
        --branching results/branching_*.json \
        --peak_branching results/peak_branching_*.json \
        --max_questions 50 \
        --output_dir comparisons/
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def find_most_recent_file(pattern: str, base_dir: str = "outputs") -> Optional[str]:
    """
    Find the most recent file matching the pattern

    Args:
        pattern: File pattern like "traditional_sc_detailed_*.json"
        base_dir: Base directory to search in

    Returns:
        Path to most recent file or None if not found
    """
    # Search for files matching the pattern
    search_pattern = os.path.join(base_dir, pattern)
    matching_files = glob.glob(search_pattern)

    if not matching_files:
        return None

    # Sort by modification time and return the most recent
    most_recent = max(matching_files, key=os.path.getmtime)
    return most_recent


def extract_timestamp_from_filename(filepath: str) -> str:
    """
    Extract timestamp from filename
    Format expected: {type}_sc_detailed_{timestamp}.json
    """
    basename = os.path.basename(filepath)
    # Remove extension
    name_no_ext = basename.rsplit('.', 1)[0]
    # Extract timestamp (last part after underscore)
    parts = name_no_ext.split('_')
    if len(parts) >= 4:
        return parts[-1]  # Should be the timestamp
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def normalize_accuracy(value: float) -> float:
    """
    Normalize accuracy value to 0-1 range.
    Handles both decimal (0-1) and percentage (0-100) formats.
    """
    if value > 1.0:
        # Assume it's already a percentage, convert to decimal
        return value / 100.0
    return value


def extract_metrics(results: Dict[str, Any], experiment_type: str, max_questions: Optional[int] = None) -> Dict[str, List[float]]:
    """
    Extract metrics from results for each experiment type

    Args:
        results: Results dictionary from JSON file
        experiment_type: Type of experiment (traditional, branching, peak_branching)
        max_questions: Maximum number of questions to include (None = all questions)

    Returns:
        Dictionary with metrics including new metrics for branching analysis
    """
    metrics = {
        'total_tokens_new': [],
        'overall_accuracy': [],
        'individual_accuracy': [],
        'question_indices': [],
        # New metrics for branching analysis
        'initial_trace_accuracy': [],
        'branch_trace_accuracy': [],
        'initial_trace_lengths': [],
        'branch_trace_lengths': [],
        'all_trace_lengths': []
    }

    # Navigate through results structure
    # JSON has structure: {"metadata": {}, "results": {"gsm8k": [...]}}
    if 'results' in results:
        results_dict = results['results']
        for dataset_name, dataset_results in results_dict.items():
            if not isinstance(dataset_results, list):
                continue

            # Limit to first max_questions if specified
            questions_to_process = dataset_results[:max_questions] if max_questions else dataset_results

            for question_result in questions_to_process:
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

                    # Get trace lengths for traditional
                    valid_traces = question_result.get('valid_traces', [])
                    trace_lengths = [t.get('num_tokens', 0) for t in valid_traces]
                    if trace_lengths:
                        metrics['all_trace_lengths'].extend(trace_lengths)
                        avg_length = np.mean(trace_lengths)
                    else:
                        avg_length = 0
                    metrics['initial_trace_lengths'].append(avg_length)
                    metrics['branch_trace_lengths'].append(0)  # No branches in traditional

                    # For traditional, initial accuracy = individual accuracy
                    metrics['initial_trace_accuracy'].append(individual_acc)
                    metrics['branch_trace_accuracy'].append(0)  # No branches

                elif experiment_type == 'branching':
                    # Branching: use total_tokens_generated (only NEW tokens)
                    total_tokens_generated = stats.get('total_tokens_generated', 0)
                    metrics['total_tokens_new'].append(total_tokens_generated)

                    # Extract trace lengths for branching
                    full_traces = question_result.get('full_traces', [])
                    valid_traces = question_result.get('valid_traces', [])

                    # Separate original vs branched traces
                    original_traces = []
                    branched_traces = []

                    for trace in full_traces:
                        if trace.get('parent_idx') is None or trace.get('parent_idx') == -1:
                            # Original trace
                            original_traces.append(trace)
                        else:
                            # Branched trace
                            branched_traces.append(trace)

                    # Calculate lengths
                    original_lengths = [t.get('num_tokens', 0) for t in original_traces]
                    branch_lengths = [t.get('tokens_generated', 0) for t in branched_traces]

                    avg_original = np.mean(original_lengths) if original_lengths else 0
                    avg_branch = np.mean(branch_lengths) if branch_lengths else 0

                    metrics['initial_trace_lengths'].append(avg_original)
                    metrics['branch_trace_lengths'].append(avg_branch)

                    # Calculate accuracies
                    original_correct = sum(1 for t in original_traces if t.get('is_correct', False))
                    branch_correct = sum(1 for t in branched_traces if t.get('is_correct', False))

                    original_acc = original_correct / len(original_traces) if original_traces else 0
                    branch_acc = branch_correct / len(branched_traces) if branched_traces else 0

                    metrics['initial_trace_accuracy'].append(original_acc)
                    metrics['branch_trace_accuracy'].append(branch_acc)

                elif experiment_type == 'peak_branching':
                    # Peak branching: use total_tokens_generated (only NEW tokens)
                    peak_stats = question_result.get('peak_branching_stats', {})
                    total_tokens_generated = peak_stats.get('total_tokens_generated', 0)

                    # Debug information
                    from_peak_stats = total_tokens_generated > 0

                    # IMPORTANT: Do NOT fallback to total_tokens as it includes prefix tokens!
                    # If peak_branching_stats is missing, try to calculate from valid_traces
                    if total_tokens_generated == 0 and 'valid_traces' in question_result:
                        valid_traces = question_result.get('valid_traces', [])
                        # Sum up tokens_generated from each trace (NEW tokens only)
                        total_tokens_generated = sum(t.get('tokens_generated', 0) for t in valid_traces)

                        # Debug: Check if tokens_generated field exists
                        if valid_traces and 'tokens_generated' not in valid_traces[0]:
                            print(f"WARNING: 'tokens_generated' field missing in valid_traces!")
                            # Try using num_tokens as fallback (this would be WRONG!)
                            total_from_num_tokens = sum(t.get('num_tokens', 0) for t in valid_traces)
                            print(f"  Would get {total_from_num_tokens} tokens using num_tokens (WRONG!)")

                    # If still 0, use statistics but this might be incorrect for older data
                    if total_tokens_generated == 0:
                        # Warning: This includes prefix tokens and will overestimate!
                        print(f"WARNING: Using fallback from statistics for question!")
                        total_tokens_generated = stats.get('total_tokens_generated', stats.get('total_tokens', 0))

                    metrics['total_tokens_new'].append(total_tokens_generated)

                    # Get initial and branch accuracies directly
                    initial_acc = question_result.get('initial_trace_accuracy', 0.0)
                    branch_acc = question_result.get('branch_trace_accuracy', 0.0)
                    metrics['initial_trace_accuracy'].append(initial_acc)
                    metrics['branch_trace_accuracy'].append(branch_acc)

                    # Extract trace lengths for peak branching
                    valid_traces = question_result.get('valid_traces', [])
                    initial_traces = [t for t in valid_traces if t.get('stage', 0) == 0]
                    branch_traces = [t for t in valid_traces if t.get('stage', 0) > 0]

                    # Calculate average lengths
                    initial_lengths = [t.get('num_tokens', 0) for t in initial_traces]
                    branch_lengths = [t.get('tokens_generated', 0) for t in branch_traces]

                    avg_initial = np.mean(initial_lengths) if initial_lengths else 0
                    avg_branch = np.mean(branch_lengths) if branch_lengths else 0

                    metrics['initial_trace_lengths'].append(avg_initial)
                    metrics['branch_trace_lengths'].append(avg_branch)

    return metrics


def create_comparison_plots(
    traditional_metrics: Dict[str, List[float]],
    branching_metrics: Dict[str, List[float]],
    peak_branching_metrics: Dict[str, List[float]],
    output_dir: str,
    timestamp: str
):
    """Create comparison visualizations with timestamp in filenames"""

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
    ax.set_title('Token Efficiency Comparison (NEW Tokens Only)\n(Lower is Better)', fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(averages) * 1.2)

    plt.tight_layout()
    filename = f'comparison_tokens_{timestamp}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/{filename}")

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
    filename = f'comparison_overall_accuracy_{timestamp}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/{filename}")

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
    filename = f'comparison_individual_accuracy_{timestamp}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/{filename}")

    # Plot 4: Individual vs Branched Accuracy (NEW)
    fig, ax = plt.subplots(figsize=(14, 7))

    # Prepare data for grouped bar chart
    x = np.arange(len(methods))
    width = 0.35

    # Initial trace accuracies
    initial_accs = [
        np.mean(traditional_metrics['initial_trace_accuracy']) * 100 if traditional_metrics['initial_trace_accuracy'] else 0,
        np.mean(branching_metrics['initial_trace_accuracy']) * 100 if branching_metrics['initial_trace_accuracy'] else 0,
        np.mean(peak_branching_metrics['initial_trace_accuracy']) * 100 if peak_branching_metrics['initial_trace_accuracy'] else 0
    ]

    # Branch trace accuracies (0 for traditional)
    branch_accs = [
        0,  # Traditional has no branches
        np.mean(branching_metrics['branch_trace_accuracy']) * 100 if branching_metrics['branch_trace_accuracy'] else 0,
        np.mean(peak_branching_metrics['branch_trace_accuracy']) * 100 if peak_branching_metrics['branch_trace_accuracy'] else 0
    ]

    # Create grouped bars
    bars1 = ax.bar(x - width/2, initial_accs, width, label='Initial Traces',
                   color='#3498db', alpha=0.7, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, branch_accs, width, label='Branched Traces',
                   color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, acc in zip(bars1, initial_accs):
        if acc > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    for bar, acc in zip(bars2, branch_accs):
        if acc > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add improvement annotations for branching methods
    for i, method in enumerate([1, 2]):  # Skip traditional (index 0)
        if initial_accs[method] > 0 and branch_accs[method] > 0:
            improvement = branch_accs[method] - initial_accs[method]
            if improvement > 0:
                ax.annotate(f'↑ +{improvement:.1f}%',
                           xy=(method, branch_accs[method]),
                           xytext=(method, branch_accs[method] + 5),
                           ha='center', fontweight='bold', color='green',
                           arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Initial vs Branched Trace Accuracy Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)

    plt.tight_layout()
    filename = f'comparison_initial_vs_branched_accuracy_{timestamp}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/{filename}")

    # Plot 5: Chain of Thought Length Comparison (NEW)
    fig, ax = plt.subplots(figsize=(14, 7))

    # Prepare data for grouped bar chart
    x = np.arange(len(methods))
    width = 0.35

    # Initial trace lengths
    initial_lengths = [
        np.mean(traditional_metrics['initial_trace_lengths']) if traditional_metrics['initial_trace_lengths'] else 0,
        np.mean(branching_metrics['initial_trace_lengths']) if branching_metrics['initial_trace_lengths'] else 0,
        np.mean(peak_branching_metrics['initial_trace_lengths']) if peak_branching_metrics['initial_trace_lengths'] else 0
    ]

    # Branch trace lengths (0 for traditional)
    branch_lengths = [
        0,  # Traditional has no branches
        np.mean(branching_metrics['branch_trace_lengths']) if branching_metrics['branch_trace_lengths'] else 0,
        np.mean(peak_branching_metrics['branch_trace_lengths']) if peak_branching_metrics['branch_trace_lengths'] else 0
    ]

    # Create grouped bars
    bars1 = ax.bar(x - width/2, initial_lengths, width, label='Initial/Start Traces',
                   color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, branch_lengths, width, label='Branched Traces (NEW tokens only)',
                   color='#f39c12', alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, length in zip(bars1, initial_lengths):
        if length > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(length):,}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    for bar, length in zip(bars2, branch_lengths):
        if length > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(length):,}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add annotations for efficiency
    for i, method in enumerate([1, 2]):  # Skip traditional (index 0)
        if initial_lengths[method] > 0 and branch_lengths[method] > 0:
            ratio = branch_lengths[method] / initial_lengths[method]
            ax.text(i, max(initial_lengths[method], branch_lengths[method]) + 2000,
                   f'Branch/Initial: {ratio:.1%}',
                   ha='center', fontsize=10, fontweight='bold', color='navy')

    ax.set_ylabel('Average Token Length', fontsize=13, fontweight='bold')
    ax.set_title('Chain of Thought Length Comparison\n(Initial/Start Traces vs Branched Traces)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Set y-axis limit with some headroom
    max_length = max(max(initial_lengths), max(branch_lengths))
    ax.set_ylim(0, max_length * 1.3 if max_length > 0 else 100)

    plt.tight_layout()
    filename = f'comparison_chain_length_{timestamp}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/{filename}")

    # Print detailed summary
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Metric':<45} {'Traditional':<15} {'Branching':<15} {'Peak Branch':<15}")
    print("-"*80)

    # Token metrics
    print(f"{'Avg NEW Tokens Generated':<45} {traditional_avg:<15,.0f} {branching_avg:<15,.0f} {peak_branching_avg:<15,.0f}")

    # Debug: Show per-question token counts and trace details
    if len(peak_branching_metrics['total_tokens_new']) <= 10:  # Only show if small dataset
        print(f"\nDEBUG - Peak Branching token counts per question:")
        for i, tokens in enumerate(peak_branching_metrics['total_tokens_new']):
            print(f"  Question {i}: {tokens:,.0f} tokens")

        print(f"\nDEBUG - Comparison of average tokens per method:")
        print(f"  Traditional (per question): {traditional_avg:.0f}")
        print(f"  Branching (per question): {branching_avg:.0f}")
        print(f"  Peak Branching (per question): {peak_branching_avg:.0f}")

        if peak_branching_avg > traditional_avg:
            print(f"\n⚠️ ALERT: Peak Branching uses MORE tokens than Traditional!")
            print(f"  This indicates a serious issue with the token counting.")
            print(f"  Possible causes:")
            print(f"  1. tokens_generated field is missing/wrong in the data")
            print(f"  2. Fallback to total_tokens (includes prefix) is being triggered")
            print(f"  3. Peak branching is actually generating way more traces than expected")

    # Accuracy metrics
    print(f"{'Overall Accuracy':<45} {traditional_acc*100:<15.1f}% {branching_acc*100:<15.1f}% {peak_branching_acc*100:<15.1f}%")
    print(f"{'Individual Trace Accuracy (All)':<45} {traditional_ind*100:<15.1f}% {branching_ind*100:<15.1f}% {peak_branching_ind*100:<15.1f}%")

    # Branching-specific accuracy
    print(f"{'Initial Trace Accuracy':<45} {initial_accs[0]:<15.1f}% {initial_accs[1]:<15.1f}% {initial_accs[2]:<15.1f}%")
    print(f"{'Branched Trace Accuracy':<45} {'N/A':<15} {branch_accs[1]:<15.1f}% {branch_accs[2]:<15.1f}%")

    # Chain length metrics
    print(f"{'Avg Initial/Start Trace Length':<45} {initial_lengths[0]:<15,.0f} {initial_lengths[1]:<15,.0f} {initial_lengths[2]:<15,.0f}")
    print(f"{'Avg Branched Trace Length (NEW only)':<45} {'N/A':<15} {branch_lengths[1]:<15,.0f} {branch_lengths[2]:<15,.0f}")

    print("-"*80)

    # Additional analysis for peak branching
    if branch_lengths[2] > branch_lengths[1] and branch_lengths[1] > 0:
        print(f"\nWARNING: Peak Branching generates longer branches than regular Branching!")
        print(f"  This suggests peak branching is occurring too early in the traces.")
        print(f"  Regular branching: {branch_lengths[1]:.0f} tokens per branch")
        print(f"  Peak branching: {branch_lengths[2]:.0f} tokens per branch")
        print(f"  Difference: +{branch_lengths[2] - branch_lengths[1]:.0f} tokens")
        print(f"\n  Hypothesis: Peak detection finds confidence peaks too early,")
        print(f"  requiring more tokens to complete each branch.")

    if traditional_avg > 0:
        print(f"\nToken Savings vs Traditional:")
        print(f"  Branching SC: {((traditional_avg - branching_avg) / traditional_avg) * 100:.1f}%")
        print(f"  Peak Branching SC: {((traditional_avg - peak_branching_avg) / traditional_avg) * 100:.1f}%")

    if initial_accs[1] > 0 and branch_accs[1] > 0:
        print(f"\nAccuracy Improvements from Branching:")
        print(f"  Branching SC: {branch_accs[1] - initial_accs[1]:.1f}% improvement")
    if initial_accs[2] > 0 and branch_accs[2] > 0:
        print(f"  Peak Branching SC: {branch_accs[2] - initial_accs[2]:.1f}% improvement")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Compare Traditional, Branching, and Peak Branching SC')
    parser.add_argument('--traditional', type=str, default=None,
                       help='Path to traditional SC results JSON (auto-selects if not provided)')
    parser.add_argument('--branching', type=str, default=None,
                       help='Path to branching SC results JSON (auto-selects if not provided)')
    parser.add_argument('--peak_branching', type=str, default=None,
                       help='Path to peak branching SC results JSON (auto-selects if not provided)')
    parser.add_argument('--output_dir', type=str, default='comparisons',
                       help='Output directory for comparison plots')
    parser.add_argument('--base_dir', type=str, default='outputs',
                       help='Base directory to search for result files when auto-selecting')
    parser.add_argument('--max_questions', type=int, default=None,
                       help='Maximum number of questions to compare (default: all questions)')

    args = parser.parse_args()

    # Auto-select files if not provided
    if args.traditional is None:
        args.traditional = find_most_recent_file("traditional_sc_detailed_*.json", args.base_dir)
        if args.traditional is None:
            parser.error("Could not find traditional SC results. Please specify --traditional")
        print(f"Auto-selected traditional: {args.traditional}")

    if args.branching is None:
        args.branching = find_most_recent_file("branching_sc_detailed_*.json", args.base_dir)
        if args.branching is None:
            parser.error("Could not find branching SC results. Please specify --branching")
        print(f"Auto-selected branching: {args.branching}")

    if args.peak_branching is None:
        args.peak_branching = find_most_recent_file("peak_branching_sc_detailed_*.json", args.base_dir)
        if args.peak_branching is None:
            parser.error("Could not find peak branching SC results. Please specify --peak_branching")
        print(f"Auto-selected peak branching: {args.peak_branching}")

    print("\nLoading results...")
    print(f"  Traditional: {args.traditional}")
    print(f"  Branching: {args.branching}")
    print(f"  Peak Branching: {args.peak_branching}")

    # Load results
    traditional_results = load_results(args.traditional)
    branching_results = load_results(args.branching)
    peak_branching_results = load_results(args.peak_branching)

    # Extract timestamp from most recent file for output naming
    timestamps = [
        extract_timestamp_from_filename(args.traditional),
        extract_timestamp_from_filename(args.branching),
        extract_timestamp_from_filename(args.peak_branching)
    ]
    # Use the most recent timestamp for output files
    output_timestamp = max(timestamps)

    # Extract metrics
    print("\nExtracting metrics...")
    if args.max_questions:
        print(f"  Limiting comparison to first {args.max_questions} questions")
    traditional_metrics = extract_metrics(traditional_results, 'traditional', args.max_questions)
    branching_metrics = extract_metrics(branching_results, 'branching', args.max_questions)
    peak_branching_metrics = extract_metrics(peak_branching_results, 'peak_branching', args.max_questions)

    print(f"  Traditional: {len(traditional_metrics['total_tokens_new'])} questions")
    print(f"  Branching: {len(branching_metrics['total_tokens_new'])} questions")
    print(f"  Peak Branching: {len(peak_branching_metrics['total_tokens_new'])} questions")

    # Create comparison plots
    print(f"\nGenerating comparison visualizations (timestamp: {output_timestamp})...")
    create_comparison_plots(
        traditional_metrics,
        branching_metrics,
        peak_branching_metrics,
        args.output_dir,
        output_timestamp
    )

    print(f"\n✓ Comparison complete! All charts saved with timestamp: {output_timestamp}")


if __name__ == "__main__":
    main()