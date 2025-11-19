"""
Visualization utilities for Peak Branching Self-Consistency

Creates specialized visualizations to show:
- Confidence curves with peak markers
- Branch points and genealogy
- Token savings from prefix caching
- Initial vs branch trace accuracy
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import os
from datetime import datetime


def create_confidence_peaks_plot(
    traces: List[Dict[str, Any]],
    ground_truth: str = None,
    peak_branching_config: Dict[str, Any] = None,
    save_path: Optional[str] = None
):
    """
    Create comprehensive confidence evolution plot for peak branching

    Shows:
    - Y-axis: Confidence score
    - X-axis: Token position (cumulative tokens used)
    - Green lines: Correct answer traces
    - Red lines: Incorrect answer traces
    - Different line styles for different stages (initial, stage 1, stage 2, etc.)
    - Peak markers showing where branches originated
    - Branch connection indicators
    """
    # Organize traces by stage
    traces_by_stage = {}
    for trace in traces:
        stage = trace.get('stage', 0)
        if stage not in traces_by_stage:
            traces_by_stage[stage] = []
        traces_by_stage[stage].append(trace)

    num_stages = len(traces_by_stage)

    # Create subplots: one for each stage + one overview
    fig, axes = plt.subplots(num_stages + 1, 1, figsize=(16, 4 * (num_stages + 1)))

    if num_stages == 0:
        axes = [axes]  # Handle single subplot case

    # Line styles for different stages
    line_styles = ['-', '--', '-.', ':']

    # Overview plot (all stages together)
    ax_overview = axes[0] if num_stages > 0 else axes
    ax_overview.set_title('Peak Branching - All Stages Overview', fontsize=14, fontweight='bold')

    # Track legend entries to avoid duplicates
    legend_added = {'correct': False, 'incorrect': False}

    # Plot all traces on overview
    for stage in sorted(traces_by_stage.keys()):
        stage_traces = traces_by_stage[stage]
        line_style = line_styles[stage % len(line_styles)]

        for trace in stage_traces:
            confs = trace.get('confs', [])
            if not confs:
                continue

            # Determine correctness
            answer = trace.get('extracted_answer') or trace.get('answer')
            is_correct = (answer == ground_truth) if ground_truth else False

            # Color based on correctness
            color = 'green' if is_correct else 'red'
            alpha = 0.7 if stage == 0 else 0.4  # Initial traces more opaque

            # X-axis: token positions
            x = range(len(confs))

            # Label for legend (only once per category)
            label = None
            if is_correct and not legend_added['correct']:
                label = f'Correct (ans={answer})'
                legend_added['correct'] = True
            elif not is_correct and not legend_added['incorrect']:
                label = f'Incorrect'
                legend_added['incorrect'] = True

            ax_overview.plot(x, confs, color=color, alpha=alpha,
                           linewidth=1.5 if stage == 0 else 1.0,
                           linestyle=line_style, label=label)

            # Mark peaks for initial traces
            if stage == 0:
                peaks = trace.get('confidence_peaks', [])
                for peak in peaks:
                    pos = peak['position']
                    if pos < len(confs):
                        ax_overview.scatter([pos], [confs[pos]],
                                          color='orange', s=100, marker='*',
                                          edgecolors='black', linewidths=1, zorder=5)

    # Add confidence threshold line
    if peak_branching_config:
        threshold = peak_branching_config.get('confidence_threshold', 1.5)
        ax_overview.axhline(y=threshold, color='purple', linestyle='--',
                          alpha=0.6, linewidth=2, label=f'Threshold ({threshold})')

    ax_overview.set_xlabel('Token Position', fontsize=11)
    ax_overview.set_ylabel('Confidence Score', fontsize=11)
    ax_overview.grid(True, alpha=0.3)
    ax_overview.legend(loc='upper right', fontsize=9)

    # Individual stage plots
    for stage_idx, stage in enumerate(sorted(traces_by_stage.keys())):
        if num_stages == 0:
            break

        ax = axes[stage_idx + 1]
        stage_traces = traces_by_stage[stage]

        # Title based on stage
        if stage == 0:
            ax.set_title(f'Stage {stage}: Initial Traces (n={len(stage_traces)})',
                        fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'Stage {stage}: Branches (n={len(stage_traces)})',
                        fontsize=12, fontweight='bold')

        for trace in stage_traces:
            confs = trace.get('confs', [])
            if not confs:
                continue

            # Determine correctness
            answer = trace.get('extracted_answer') or trace.get('answer')
            is_correct = (answer == ground_truth) if ground_truth else False

            # Color and styling
            color = 'green' if is_correct else 'red'
            alpha = 0.7

            trace_idx = trace.get('trace_idx', '?')
            parent_idx = trace.get('parent_idx')
            branch_point = trace.get('branch_point_tokens', 0)

            # Build label
            if stage == 0:
                label = f"T{trace_idx} → {answer}"
            else:
                label = f"T{trace_idx} (from T{parent_idx}@{branch_point}) → {answer}"

            # X-axis: token positions
            x = range(len(confs))

            ax.plot(x, confs, color=color, alpha=alpha, linewidth=1.5, label=label)

            # Mark peaks for initial traces
            if stage == 0:
                peaks = trace.get('confidence_peaks', [])
                for peak in peaks:
                    pos = peak['position']
                    conf = peak['confidence']
                    if pos < len(confs):
                        ax.scatter([pos], [confs[pos]],
                                 color='orange', s=150, marker='*',
                                 edgecolors='black', linewidths=1.5, zorder=5)
                        # Add peak confidence value
                        ax.text(pos, confs[pos] + 0.1, f'{conf:.2f}',
                               fontsize=7, ha='center', color='orange')

            # Mark branch point for branch traces
            if stage > 0 and branch_point > 0:
                ax.axvline(x=branch_point, color='blue', alpha=0.5,
                          linestyle=':', linewidth=2, label=f'Branch point')

        # Add confidence threshold line
        if peak_branching_config:
            threshold = peak_branching_config.get('confidence_threshold', 1.5)
            ax.axhline(y=threshold, color='purple', linestyle='--',
                      alpha=0.6, linewidth=1.5)

        ax.set_xlabel('Token Position', fontsize=10)
        ax.set_ylabel('Confidence Score', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7, ncol=2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_peak_branching_summary(
    result: Dict[str, Any],
    save_path: Optional[str] = None
):
    """
    Create a 4-panel summary plot for peak branching

    Panels:
    1. Token usage comparison (initial vs branches)
    2. Answer distribution with correctness
    3. Confidence peak distribution
    4. Prefix cache savings visualization
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    traces = result.get('valid_traces', [])
    initial_traces = [t for t in traces if t.get('depth', 0) == 0]
    branch_traces = [t for t in traces if t.get('depth', 0) == 1]

    # Panel 1: Token usage comparison
    ax1.set_title('Token Usage: Initial vs Branch Traces', fontsize=11, fontweight='bold')

    initial_tokens = [t.get('tokens_generated', 0) for t in initial_traces]
    branch_tokens = [t.get('tokens_generated', 0) for t in branch_traces]

    if initial_tokens or branch_tokens:
        box_data = []
        labels = []
        if initial_tokens:
            box_data.append(initial_tokens)
            labels.append(f'Initial\n(n={len(initial_tokens)})')
        if branch_tokens:
            box_data.append(branch_tokens)
            labels.append(f'Branches\n(n={len(branch_tokens)})')

        bp = ax1.boxplot(box_data, labels=labels, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(['lightblue', 'lightgreen'][i % 2])

        # Add mean markers
        for i, data in enumerate(box_data):
            ax1.scatter(i+1, np.mean(data), color='red', s=50, marker='D', zorder=5)

    ax1.set_ylabel('Tokens Generated')
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Answer distribution
    ax2.set_title('Answer Distribution & Correctness', fontsize=11, fontweight='bold')

    from collections import Counter
    answer_counts = Counter()
    correct_counts = Counter()

    for trace in traces:
        ans = trace.get('answer')
        if ans:
            answer_counts[ans] += 1
            if trace.get('is_correct', False):
                correct_counts[ans] += 1

    if answer_counts:
        answers = list(answer_counts.keys())
        counts = list(answer_counts.values())
        correct = [correct_counts.get(ans, 0) for ans in answers]

        x = range(len(answers))
        width = 0.35

        bars1 = ax2.bar([i - width/2 for i in x], counts, width, label='Total', color='lightblue')
        bars2 = ax2.bar([i + width/2 for i in x], correct, width, label='Correct', color='lightgreen')

        ax2.set_xticks(x)
        ax2.set_xticklabels([str(a)[:10] for a in answers], rotation=45, ha='right')
        ax2.set_ylabel('Count')
        ax2.legend()

        # Mark ground truth
        gt = result.get('ground_truth')
        if gt in answers:
            gt_idx = answers.index(gt)
            ax2.scatter(gt_idx, counts[gt_idx] + 0.5, marker='*', s=200, color='gold', zorder=5)

    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Peak confidence distribution
    ax3.set_title('Confidence Peak Distribution', fontsize=11, fontweight='bold')

    all_peak_confs = []
    for trace in initial_traces:
        peaks = trace.get('confidence_peaks', [])
        all_peak_confs.extend([p['confidence'] for p in peaks])

    if all_peak_confs:
        ax3.hist(all_peak_confs, bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax3.axvline(x=1.5, color='red', linestyle='--', label='Threshold (1.5)')
        ax3.set_xlabel('Peak Confidence')
        ax3.set_ylabel('Frequency')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No peaks detected', ha='center', va='center', transform=ax3.transAxes)

    ax3.grid(True, alpha=0.3)

    # Panel 4: Prefix cache savings
    ax4.set_title('Token Efficiency & Prefix Caching', fontsize=11, fontweight='bold')

    stats = result.get('peak_branching_stats', {})

    if stats:
        categories = ['Generated', 'With Prefix', 'Saved']
        values = [
            stats.get('total_tokens_generated', 0),
            stats.get('total_tokens_with_prefix', 0),
            stats.get('prefix_cache_savings', 0)
        ]
        colors = ['lightblue', 'lightcoral', 'lightgreen']

        bars = ax4.bar(categories, values, color=colors)

        # Add percentage on saved bar
        if values[2] > 0:
            pct = stats.get('prefix_cache_savings_pct', 0)
            ax4.text(2, values[2]/2, f'{pct:.1f}%', ha='center', va='center', fontweight='bold')

        ax4.set_ylabel('Total Tokens')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height + max(values)*0.01,
                    f'{val:,}', ha='center', va='bottom')

    ax4.grid(True, alpha=0.3, axis='y')

    # Overall title
    voted = result.get('voted_answer', 'N/A')
    gt = result.get('ground_truth', 'N/A')
    correct = '✓' if result.get('is_correct', False) else '✗'

    fig.suptitle(
        f'Peak Branching Analysis - Voted: {voted} | GT: {gt} | {correct}',
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_genealogy_graph(
    traces: List[Dict[str, Any]],
    save_path: Optional[str] = None
):
    """
    Create a genealogy graph showing parent-child relationships
    """
    try:
        import networkx as nx
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("NetworkX not installed. Skipping genealogy graph.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for trace in traces:
        idx = trace.get('trace_idx', trace.get('trace_idx'))
        depth = trace.get('depth', 0)
        answer = trace.get('answer', trace.get('extracted_answer', 'N/A'))
        is_correct = trace.get('is_correct', False)

        G.add_node(idx, depth=depth, answer=answer, correct=is_correct)

    # Add edges
    for trace in traces:
        idx = trace.get('trace_idx', trace.get('trace_idx'))
        parent = trace.get('parent_idx')
        if parent is not None:
            branch_point = trace.get('branch_point_tokens', 0)
            G.add_edge(parent, idx, branch_point=branch_point)

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw nodes
    for node in G.nodes():
        x, y = pos[node]
        depth = G.nodes[node]['depth']
        correct = G.nodes[node]['correct']
        answer = str(G.nodes[node]['answer'])[:10]

        # Node color based on correctness
        color = 'lightgreen' if correct else 'lightcoral'
        # Shape based on depth
        if depth == 0:
            # Square for initial traces
            box = FancyBboxPatch((x-0.05, y-0.05), 0.1, 0.1,
                                 boxstyle="round,pad=0.01",
                                 facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(box)
        else:
            # Circle for branch traces
            circle = plt.Circle((x, y), 0.05, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)

        # Add label
        ax.text(x, y-0.08, f"{node}\n{answer}", ha='center', va='top', fontsize=8)

    # Draw edges
    for edge in G.edges():
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        branch_point = G.edges[edge].get('branch_point', 0)

        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

        # Add branch point label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, f'@{branch_point}', fontsize=7, color='blue', ha='center')

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Peak Branching Genealogy Tree', fontsize=14, fontweight='bold', pad=20)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='black', label='Correct'),
        Patch(facecolor='lightcoral', edgecolor='black', label='Incorrect'),
        Patch(facecolor='white', edgecolor='black', label='□ Initial'),
        Patch(facecolor='white', edgecolor='black', label='○ Branch')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_peak_branching_results(
    results_file: str,
    output_dir: str = "visualizations",
    question_idx: Optional[int] = None
):
    """
    Main function to generate all peak branching visualizations
    """
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Process each dataset
    for dataset_name, results in data.items():
        if dataset_name == 'metadata':
            continue

        print(f"\nProcessing {dataset_name}...")

        # Filter to specific question if requested
        if question_idx is not None:
            if question_idx < len(results):
                results = [results[question_idx]]
                indices = [question_idx]
            else:
                print(f"Question {question_idx} not found in {dataset_name}")
                continue
        else:
            indices = range(len(results))

        # Generate visualizations for each question
        for i, result in zip(indices, results):
            print(f"  Question {i}...")

            # Skip if no valid traces
            if not result.get('valid_traces'):
                print(f"    Skipping - no valid traces")
                continue

            # 1. Confidence peaks plot
            save_path = os.path.join(output_dir, f"peaks_{dataset_name}_q{i}_{timestamp}.png")
            create_confidence_peaks_plot(
                result.get('valid_traces', []),
                ground_truth=result.get('ground_truth'),
                peak_branching_config=result.get('peak_branching_config'),
                save_path=save_path
            )
            print(f"    Created: {save_path}")

            # 2. Summary plot
            save_path = os.path.join(output_dir, f"summary_{dataset_name}_q{i}_{timestamp}.png")
            create_peak_branching_summary(result, save_path)
            print(f"    Created: {save_path}")

            # 3. Genealogy graph
            save_path = os.path.join(output_dir, f"genealogy_{dataset_name}_q{i}_{timestamp}.png")
            create_genealogy_graph(result.get('valid_traces', []), save_path)
            print(f"    Created: {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize Peak Branching Results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to peak branching results JSON file')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--question_id', type=int, default=None,
                       help='Visualize single question only')

    args = parser.parse_args()

    visualize_peak_branching_results(
        args.results,
        args.output_dir,
        args.question_id
    )