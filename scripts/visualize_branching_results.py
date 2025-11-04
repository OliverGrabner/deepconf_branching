"""
Visualize Branching Self-Consistency Results

Creates comprehensive visualizations:
1. Branch genealogy tree (who branched from who, when)
2. Confidence evolution over time (with branch points marked)
3. Token usage comparison
4. Accuracy by trace type (original vs branched)

Usage:
    python scripts/visualize_branching_results.py \
        --results outputs_sc/branching_sc_aime25_detailed_*.json \
        --output_dir visualizations/
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import networkx as nx
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib networkx")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load branching SC results from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_genealogy_graph(
    genealogy: Dict[str, Any],
    traces: List[Dict[str, Any]],
    ground_truth: str,
    output_path: str
):
    """
    Create a visual genealogy tree showing branch relationships

    - Green nodes: correct answer
    - Red nodes: incorrect answer
    - Blue nodes: original traces
    - Orange nodes: branched traces
    - Arrows show parent -> child with branch iteration labeled
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping genealogy graph (matplotlib not available)")
        return

    tree = genealogy.get('tree', {})
    events = genealogy.get('events', [])

    # Build networkx graph
    G = nx.DiGraph()

    # Add nodes
    trace_map = {t['trace_idx']: t for t in traces}

    for trace_idx_str, info in tree.items():
        trace_idx = int(trace_idx_str)
        trace = trace_map.get(trace_idx, {})

        is_original = info['parent'] is None
        is_correct = trace.get('extracted_answer') == ground_truth

        G.add_node(trace_idx,
                   parent=info['parent'],
                   is_original=is_original,
                   is_correct=is_correct,
                   answer=trace.get('extracted_answer', 'N/A'))

    # Add edges with branch iteration labels
    event_map = {e['child_trace_idx']: e for e in events}

    for trace_idx_str, info in tree.items():
        trace_idx = int(trace_idx_str)
        if info['parent'] is not None:
            parent_idx = info['parent']
            event = event_map.get(trace_idx, {})
            iteration = event.get('iteration', '?')
            G.add_edge(parent_idx, trace_idx, iteration=iteration)

    # Layout
    fig, ax = plt.subplots(figsize=(16, 10))

    # Use hierarchical layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw nodes with colors
    for node in G.nodes():
        x, y = pos[node]
        is_original = G.nodes[node]['is_original']
        is_correct = G.nodes[node]['is_correct']
        answer = G.nodes[node]['answer']

        # Color scheme
        if is_correct:
            color = 'lightgreen'
            edge_color = 'darkgreen'
        else:
            color = 'lightcoral'
            edge_color = 'darkred'

        if is_original:
            shape = 's'  # square for original
            size = 800
        else:
            shape = 'o'  # circle for branched
            size = 600

        # Draw node
        ax.scatter([x], [y], c=color, s=size, marker=shape,
                  edgecolors=edge_color, linewidths=2, zorder=3)

        # Label with trace index
        ax.text(x, y, str(node), fontsize=10, ha='center', va='center',
               fontweight='bold', zorder=4)

        # Answer below
        ax.text(x, y-0.08, f"ans: {str(answer)[:10]}", fontsize=7,
               ha='center', va='top', style='italic', zorder=4)

    # Draw edges with iteration labels
    for edge in G.edges():
        parent, child = edge
        x1, y1 = pos[parent]
        x2, y2 = pos[child]
        iteration = G.edges[edge]['iteration']

        # Arrow
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', mutation_scale=20,
            color='gray', linewidth=1.5, alpha=0.7,
            zorder=1
        )
        ax.add_patch(arrow)

        # Iteration label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, f"iter {iteration}", fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
               ha='center', va='center', zorder=2)

    ax.set_xlim(min(x for x, y in pos.values()) - 0.2,
                max(x for x, y in pos.values()) + 0.2)
    ax.set_ylim(min(y for x, y in pos.values()) - 0.2,
                max(y for x, y in pos.values()) + 0.2)

    ax.axis('off')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='lightgreen', edgecolor='darkgreen', label='Correct Answer'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='darkred', label='Incorrect Answer'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='gray', label='Original Trace (square)'),
        mpatches.Circle((0.5, 0.5), 0.5, facecolor='gray', label='Branched Trace (circle)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.title(f'Branch Genealogy Tree\nGround Truth: {ground_truth}',
             fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Genealogy graph saved to: {output_path}")
    plt.close()


def create_confidence_evolution_plot(
    traces: List[Dict[str, Any]],
    genealogy: Dict[str, Any],
    branching_config: Dict[str, Any],
    ground_truth: str,
    output_path: str
):
    """
    Plot confidence evolution over time with branch points marked

    Shows:
    - Confidence curves for each trace
    - Vertical lines at branch points
    - Color-coded by correctness
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping confidence evolution plot (matplotlib not available)")
        return

    events = genealogy.get('events', [])
    stride = branching_config.get('stride', 600)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Separate correct and incorrect
    correct_traces = [t for t in traces if t.get('answer') == ground_truth or t.get('extracted_answer') == ground_truth]
    incorrect_traces = [t for t in traces if t.get('answer') != ground_truth and t.get('extracted_answer') != ground_truth]

    # Plot 1: All traces
    for trace in traces:
        confs = trace.get('confs', [])
        if not confs:
            continue

        # Compute tail confidence at regular intervals
        tail_window = 2048
        positions = []
        tail_confs = []

        for i in range(tail_window, len(confs), 100):
            positions.append(i)
            tail = confs[max(0, i-tail_window):i]
            tail_confs.append(np.mean(tail))

        if not positions:
            continue

        is_correct = trace.get('extracted_answer') == ground_truth
        color = 'green' if is_correct else 'red'
        alpha = 0.6 if is_correct else 0.3
        linewidth = 1.5 if is_correct else 1.0

        ax1.plot(positions, tail_confs, color=color, alpha=alpha, linewidth=linewidth)

    # Mark branch points
    for event in events:
        branch_tokens = event.get('branch_point_tokens', 0)
        ax1.axvline(branch_tokens, color='blue', linestyle='--', alpha=0.3, linewidth=1)
        ax1.text(branch_tokens, ax1.get_ylim()[1]*0.95,
                f"Branch\niter {event.get('iteration', '?')}",
                fontsize=7, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Tail Confidence (mean of last 2048 tokens)')
    ax1.set_title('Confidence Evolution - All Traces (Green=Correct, Red=Incorrect, Blue=Branch Point)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Correct', 'Incorrect', 'Branch Point'], loc='upper right')

    # Plot 2: Correct traces only (detailed)
    if correct_traces:
        for trace in correct_traces:
            confs = trace.get('confs', [])
            if not confs:
                continue

            tail_window = 2048
            positions = []
            tail_confs = []

            for i in range(tail_window, len(confs), 100):
                positions.append(i)
                tail = confs[max(0, i-tail_window):i]
                tail_confs.append(np.mean(tail))

            if not positions:
                continue

            trace_idx = trace.get('trace_idx', '?')
            parent_idx = trace.get('parent_idx')
            label = f"Trace {trace_idx}" + (f" (from {parent_idx})" if parent_idx is not None else " (orig)")

            ax2.plot(positions, tail_confs, alpha=0.7, linewidth=1.5, label=label)

        # Mark branch points
        for event in events:
            branch_tokens = event.get('branch_point_tokens', 0)
            ax2.axvline(branch_tokens, color='blue', linestyle='--', alpha=0.3, linewidth=1)

        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Tail Confidence')
        ax2.set_title(f'Correct Traces Only (n={len(correct_traces)})')
        ax2.grid(True, alpha=0.3)
        if len(correct_traces) <= 10:
            ax2.legend(fontsize=8, loc='best')
    else:
        ax2.text(0.5, 0.5, 'No correct traces', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Confidence evolution saved to: {output_path}")
    plt.close()


def create_token_usage_plot(
    results_by_dataset: Dict[str, List[Dict[str, Any]]],
    output_path: str
):
    """
    Plot token usage statistics
    - Tokens per question
    - Original vs branched trace tokens
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping token usage plot (matplotlib not available)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for dataset_name, results in results_by_dataset.items():
        question_ids = list(range(len(results)))
        total_tokens = [r['statistics']['total_tokens'] for r in results]

        ax1.plot(question_ids, total_tokens, marker='o', label=dataset_name, linewidth=2)

    ax1.set_xlabel('Question ID')
    ax1.set_ylabel('Total Tokens')
    ax1.set_title('Token Usage per Question')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Original vs branched token distribution
    all_original_tokens = []
    all_branched_tokens = []

    for dataset_name, results in results_by_dataset.items():
        for result in results:
            for trace in result.get('valid_traces', []):
                tokens = trace.get('num_tokens', 0)
                if trace.get('parent_idx') is None:
                    all_original_tokens.append(tokens)
                else:
                    all_branched_tokens.append(tokens)

    if all_original_tokens and all_branched_tokens:
        ax2.hist(all_original_tokens, bins=30, alpha=0.6, label='Original Traces', color='blue')
        ax2.hist(all_branched_tokens, bins=30, alpha=0.6, label='Branched Traces', color='orange')
        ax2.set_xlabel('Tokens per Trace')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Token Distribution: Original vs Branched Traces')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Token usage plot saved to: {output_path}")
    plt.close()


def create_accuracy_analysis_plot(
    results_by_dataset: Dict[str, List[Dict[str, Any]]],
    output_path: str
):
    """
    Analyze accuracy: original vs branched traces
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping accuracy analysis plot (matplotlib not available)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy by trace type
    original_correct = 0
    original_total = 0
    branched_correct = 0
    branched_total = 0

    for dataset_name, results in results_by_dataset.items():
        for result in results:
            ground_truth = result.get('ground_truth')

            for trace in result.get('valid_traces', []):
                answer = trace.get('answer')
                is_correct = (answer == ground_truth)

                if trace.get('parent_idx') is None:
                    original_total += 1
                    if is_correct:
                        original_correct += 1
                else:
                    branched_total += 1
                    if is_correct:
                        branched_correct += 1

    original_acc = original_correct / original_total if original_total > 0 else 0
    branched_acc = branched_correct / branched_total if branched_total > 0 else 0

    ax1.bar(['Original Traces', 'Branched Traces'],
           [original_acc * 100, branched_acc * 100],
           color=['blue', 'orange'], alpha=0.7)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Trace Accuracy by Type')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')

    for i, (acc, total) in enumerate([(original_acc, original_total), (branched_acc, branched_total)]):
        ax1.text(i, acc*100 + 2, f'{acc*100:.1f}%\n(n={total})',
                ha='center', va='bottom', fontweight='bold')

    # Question-level accuracy
    question_accs = []
    question_labels = []

    for dataset_name, results in results_by_dataset.items():
        for i, result in enumerate(results):
            is_correct = result.get('is_correct', False)
            question_accs.append(100 if is_correct else 0)
            question_labels.append(f"{dataset_name}-Q{i}")

    colors = ['green' if acc == 100 else 'red' for acc in question_accs]
    ax2.bar(range(len(question_accs)), question_accs, color=colors, alpha=0.7)
    ax2.set_xlabel('Question')
    ax2.set_ylabel('Correct (%)')
    ax2.set_title('Per-Question Accuracy')
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis='y')

    # Overall accuracy text
    overall_acc = sum(1 for acc in question_accs if acc == 100) / len(question_accs) * 100
    ax2.text(0.5, 0.95, f'Overall Accuracy: {overall_acc:.1f}%',
            transform=ax2.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Accuracy analysis saved to: {output_path}")
    plt.close()


def create_all_visualizations(
    results_filepath: str,
    output_dir: str,
    question_id: Optional[int] = None
):
    """Create all visualizations for branching SC results"""

    if not MATPLOTLIB_AVAILABLE:
        print("\nERROR: matplotlib not installed!")
        print("Install with: pip install matplotlib networkx")
        return

    print(f"\nLoading results from: {results_filepath}")
    data = load_results(results_filepath)

    results_by_dataset = data.get('results', {})
    metadata = data.get('metadata', {})

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print('='*80)

    # If specific question requested, visualize just that
    if question_id is not None:
        for dataset_name, results in results_by_dataset.items():
            if question_id < len(results):
                result = results[question_id]
                print(f"\nVisualizing {dataset_name} Question {question_id}")

                # Genealogy graph
                if result.get('branch_genealogy'):
                    genealogy_path = os.path.join(
                        output_dir,
                        f"genealogy_{dataset_name}_q{question_id}_{timestamp}.png"
                    )
                    create_genealogy_graph(
                        result['branch_genealogy'],
                        result.get('full_traces', result.get('valid_traces', [])),
                        result.get('ground_truth', ''),
                        genealogy_path
                    )

                # Confidence evolution
                if result.get('full_traces') and result.get('branching_config'):
                    conf_path = os.path.join(
                        output_dir,
                        f"confidence_{dataset_name}_q{question_id}_{timestamp}.png"
                    )
                    create_confidence_evolution_plot(
                        result['full_traces'],
                        result['branch_genealogy'],
                        result['branching_config'],
                        result.get('ground_truth', ''),
                        conf_path
                    )

                break

    # Overall visualizations
    print("\nCreating dataset-wide visualizations...")

    # Token usage
    token_path = os.path.join(output_dir, f"token_usage_{timestamp}.png")
    create_token_usage_plot(results_by_dataset, token_path)

    # Accuracy analysis
    accuracy_path = os.path.join(output_dir, f"accuracy_analysis_{timestamp}.png")
    create_accuracy_analysis_plot(results_by_dataset, accuracy_path)

    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"Output directory: {output_dir}")
    print('='*80)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Branching SC Results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--results', type=str, required=True,
                       help='Path to branching SC results JSON file')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--question_id', type=int, default=None,
                       help='Visualize specific question (optional)')

    args = parser.parse_args()

    create_all_visualizations(args.results, args.output_dir, args.question_id)


if __name__ == "__main__":
    main()
