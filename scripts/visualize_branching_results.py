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
from collections import defaultdict, Counter

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


def create_per_problem_summary(
    result: Dict[str, Any],
    dataset_name: str,
    q_idx: int,
    output_path: str
):
    """
    Create a comprehensive per-problem summary visualization

    4-panel plot:
    1. Final confidence vs correctness (scatter)
    2. Token usage by trace type
    3. Branch timeline
    4. Answer distribution
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping per-problem summary (matplotlib not available)")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    traces = result.get('full_traces', [])
    genealogy = result.get('branch_genealogy', {})
    events = result.get('branch_events', [])
    ground_truth = result.get('ground_truth', '')

    # Panel 1: Final Confidence vs Correctness
    correct_confs = [t['final_tail_confidence'] for t in traces if t.get('is_correct')]
    incorrect_confs = [t['final_tail_confidence'] for t in traces if not t.get('is_correct')]

    if correct_confs:
        ax1.scatter(range(len(correct_confs)), correct_confs,
                   c='green', s=100, alpha=0.6, label='Correct', marker='o')
    if incorrect_confs:
        ax1.scatter(range(len(incorrect_confs)), incorrect_confs,
                   c='red', s=100, alpha=0.6, label='Incorrect', marker='x')

    ax1.set_xlabel('Trace Index (sorted by correctness)')
    ax1.set_ylabel('Final Tail Confidence')
    ax1.set_title(f'Final Confidence by Correctness\n{len(correct_confs)}/{len(traces)} correct')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Token Usage by Trace Type
    original_tokens = [t.get('tokens_generated', 0) for t in traces if t.get('parent_idx') is None]
    branched_tokens = [t.get('tokens_generated', 0) for t in traces if t.get('parent_idx') is not None]

    if original_tokens and branched_tokens:
        ax2.boxplot([original_tokens, branched_tokens],
                   labels=['Original', 'Branched'],
                   patch_artist=True)
        ax2.set_ylabel('Tokens Generated')
        ax2.set_title('Token Distribution by Trace Type')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add stats
        ax2.text(0.5, 0.95, f'Original: {np.mean(original_tokens):.0f} ± {np.std(original_tokens):.0f}\n'
                            f'Branched: {np.mean(branched_tokens):.0f} ± {np.std(branched_tokens):.0f}',
                transform=ax2.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 3: Branch Timeline
    if events:
        iterations = [e.get('iteration', 0) for e in events]
        parent_confs = [e.get('parent_tail_confidence', 0) for e in events]

        ax3.scatter(iterations, parent_confs, c='blue', s=50, alpha=0.7)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Parent Tail Confidence at Branch')
        ax3.set_title(f'Branch Timeline ({len(events)} events)')
        ax3.grid(True, alpha=0.3)

        # Add trend line
        if len(iterations) > 1:
            z = np.polyfit(iterations, parent_confs, 1)
            p = np.poly1d(z)
            ax3.plot(sorted(set(iterations)), [p(x) for x in sorted(set(iterations))],
                    "r--", alpha=0.5, linewidth=2)

    # Panel 4: Answer Distribution
    answers = [t.get('answer', 'None') for t in traces if t.get('answer')]
    answer_counts = Counter(answers)
    answer_labels = [str(ans)[:15] for ans in answer_counts.keys()]
    answer_values = list(answer_counts.values())

    colors = ['green' if str(ans) == str(ground_truth) else 'red'
             for ans in answer_counts.keys()]

    ax4.bar(range(len(answer_labels)), answer_values, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(answer_labels)))
    ax4.set_xticklabels(answer_labels, rotation=45, ha='right')
    ax4.set_ylabel('Number of Traces')
    ax4.set_title(f'Answer Distribution\nGround Truth: {ground_truth}')
    ax4.grid(True, alpha=0.3, axis='y')

    # Overall title
    fig.suptitle(f'{dataset_name} Question {q_idx}\nCorrectness: {"✓ CORRECT" if result.get("is_correct") else "✗ INCORRECT"}',
                fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f".", end="", flush=True)
    plt.close()


def create_confidence_evolution_plot(
    traces: List[Dict[str, Any]],
    genealogy: Dict[str, Any],
    branching_config: Dict[str, Any],
    ground_truth: str,
    output_path: str
):
    """
    Plot confidence evolution over time - 4-panel visualization

    Shows:
    - Panel 1: All traces (green=correct, red=incorrect)
    - Panel 2: Correct traces only
    - Panel 3: Incorrect traces only
    - Panel 4: Distribution of final tail confidence
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping confidence evolution plot (matplotlib not available)")
        return

    # Separate correct and incorrect traces
    correct_traces = [t for t in traces if t.get('answer') == ground_truth or t.get('extracted_answer') == ground_truth]
    incorrect_traces = [t for t in traces if t.get('answer') != ground_truth and t.get('extracted_answer') != ground_truth]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Trace Confidence Evolution', fontsize=14, fontweight='bold')

    # Helper function to compute tail confidence evolution
    def compute_tail_evolution(confs, tail_window=2048, step_size=100):
        positions = []
        tail_confs = []
        # Start from 0 and sample every step_size tokens
        for i in range(0, len(confs), step_size):
            if i == 0:
                # For position 0, just use the first token if available
                if len(confs) > 0:
                    positions.append(0)
                    tail_confs.append(confs[0])
                continue
            positions.append(i)
            # Use tail window (all tokens from max(0, i-tail_window) to i)
            tail = confs[max(0, i-tail_window):i]
            tail_confs.append(np.mean(tail))
        return positions, tail_confs

    # Panel 1: All traces
    ax1 = axes[0, 0]

    # Plot correct and incorrect traces separately for proper legend
    for trace in correct_traces:
        confs = trace.get('confs', [])
        if not confs:
            continue

        positions, tail_confs = compute_tail_evolution(confs)
        if not positions:
            continue

        ax1.plot(positions, tail_confs, color='green', alpha=0.6, linewidth=1)

    for trace in incorrect_traces:
        confs = trace.get('confs', [])
        if not confs:
            continue

        positions, tail_confs = compute_tail_evolution(confs)
        if not positions:
            continue

        ax1.plot(positions, tail_confs, color='red', alpha=0.3, linewidth=1)

    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Tail Confidence (mean of last N tokens)')
    ax1.set_title('All Traces (Green=Correct, Red=Incorrect)')
    ax1.grid(True, alpha=0.3)

    # Create proper legend with actual plot handles
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Correct'),
        Patch(facecolor='red', alpha=0.3, label='Incorrect')
    ]
    ax1.legend(handles=legend_elements)

    # Panel 2: Correct traces only
    ax2 = axes[0, 1]
    if correct_traces:
        cmap = plt.colormaps.get_cmap('Greens')
        for i, trace in enumerate(correct_traces):
            confs = trace.get('confs', [])
            if not confs:
                continue

            positions, tail_confs = compute_tail_evolution(confs)
            if not positions:
                continue

            trace_idx = trace.get('trace_idx', i)
            answer = trace.get('answer', trace.get('extracted_answer', 'N/A'))
            color = cmap(0.3 + 0.7 * (i / max(1, len(correct_traces))))

            ax2.plot(positions, tail_confs, color=color, alpha=0.7, linewidth=1.5,
                    label=f"Trace {trace_idx} (ans={answer})")

        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Tail Confidence')
        ax2.set_title(f'Correct Traces Only (n={len(correct_traces)})')
        ax2.grid(True, alpha=0.3)
        if len(correct_traces) <= 10:
            ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No correct traces', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Correct Traces Only (n=0)')

    # Panel 3: Incorrect traces only
    ax3 = axes[1, 0]
    if incorrect_traces:
        cmap = plt.colormaps.get_cmap('Reds')
        for i, trace in enumerate(incorrect_traces):
            confs = trace.get('confs', [])
            if not confs:
                continue

            positions, tail_confs = compute_tail_evolution(confs)
            if not positions:
                continue

            trace_idx = trace.get('trace_idx', i)
            answer = trace.get('answer', trace.get('extracted_answer', 'N/A'))
            color = cmap(0.3 + 0.7 * (i / max(1, len(incorrect_traces))))

            ax3.plot(positions, tail_confs, color=color, alpha=0.7, linewidth=1.5,
                    label=f"Trace {trace_idx} (ans={answer})")

        ax3.set_xlabel('Token Position')
        ax3.set_ylabel('Tail Confidence')
        ax3.set_title(f'Incorrect Traces Only (n={len(incorrect_traces)})')
        ax3.grid(True, alpha=0.3)
        if len(incorrect_traces) <= 10:
            ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No incorrect traces', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Incorrect Traces Only (n=0)')

    # Panel 4: Final confidence distribution
    ax4 = axes[1, 1]

    # Compute final tail confidence for each trace
    tail_window = 2048
    correct_final = []
    incorrect_final = []

    for trace in correct_traces:
        confs = trace.get('confs', [])
        if confs:
            final_conf = np.mean(confs[-tail_window:]) if len(confs) >= tail_window else np.mean(confs)
            correct_final.append(final_conf)

    for trace in incorrect_traces:
        confs = trace.get('confs', [])
        if confs:
            final_conf = np.mean(confs[-tail_window:]) if len(confs) >= tail_window else np.mean(confs)
            incorrect_final.append(final_conf)

    if correct_final or incorrect_final:
        all_final = correct_final + incorrect_final
        bins = np.linspace(min(all_final), max(all_final), 20)

        if correct_final:
            ax4.hist(correct_final, bins=bins, alpha=0.6, color='green', label='Correct', edgecolor='black')
        if incorrect_final:
            ax4.hist(incorrect_final, bins=bins, alpha=0.6, color='red', label='Incorrect', edgecolor='black')

        ax4.set_xlabel('Final Tail Confidence')
        ax4.set_ylabel('Number of Traces')
        ax4.set_title('Distribution of Final Tail Confidence')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
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

    # Per-question visualizations
    questions_to_visualize = []
    if question_id is not None:
        # Specific question requested
        questions_to_visualize = [(dataset_name, question_id)
                                  for dataset_name in results_by_dataset.keys()]
    else:
        # Visualize ALL questions
        for dataset_name, results in results_by_dataset.items():
            for q_idx in range(len(results)):
                questions_to_visualize.append((dataset_name, q_idx))

    print(f"\nCreating per-question visualizations ({len(questions_to_visualize)} questions)...")

    for dataset_name, q_idx in questions_to_visualize:
        results = results_by_dataset.get(dataset_name, [])
        if q_idx >= len(results):
            continue

        result = results[q_idx]
        print(f"  [{dataset_name} Q{q_idx}]", end=" ")

        # Per-problem summary (4-panel overview)
        summary_path = os.path.join(
            output_dir,
            f"summary_{dataset_name}_q{q_idx}_{timestamp}.png"
        )
        create_per_problem_summary(result, dataset_name, q_idx, summary_path)

        # Genealogy graph
        if result.get('branch_genealogy'):
            genealogy_path = os.path.join(
                output_dir,
                f"genealogy_{dataset_name}_q{q_idx}_{timestamp}.png"
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
                f"confidence_{dataset_name}_q{q_idx}_{timestamp}.png"
            )
            create_confidence_evolution_plot(
                result['full_traces'],
                result['branch_genealogy'],
                result['branching_config'],
                result.get('ground_truth', ''),
                conf_path
            )

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
