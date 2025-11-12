"""
Visualize tail confidence evolution for a single question

Shows how confidence evolves over token positions for all traces
in a traditional Self-Consistency experiment.

Usage:
    python scripts/visualize_single_question_confidence.py \
        --results outputs/traditional_sc_detailed_20251105_143014.json \
        --question_id 4 \
        --output confidence_q4.png
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_tail_evolution(confs: List[float], tail_window: int = 2048, step_size: int = 100):
    """
    Compute tail confidence at each position
    
    Args:
        confs: List of confidence scores
        tail_window: Size of tail window (default 2048)
        step_size: Step between samples (default 100)
    
    Returns:
        positions: List of token positions
        tail_confs: List of tail confidence values
    """
    positions = []
    tail_confs = []
    
    # Start from 0 and sample every step_size tokens
    for i in range(0, len(confs), step_size):
        if i == 0:
            # For position 0, just use the first token
            if len(confs) > 0:
                positions.append(0)
                tail_confs.append(confs[0])
            continue
        
        positions.append(i)
        # Use tail window (all tokens from max(0, i-tail_window) to i)
        tail = confs[max(0, i-tail_window):i]
        tail_confs.append(np.mean(tail))
    
    return positions, tail_confs


def visualize_question_confidence(
    results: Dict[str, Any],
    question_id: int,
    output_path: str
):
    """
    Create tail confidence visualization for a single question
    
    Args:
        results: Loaded results dictionary
        question_id: Question index to visualize
        output_path: Output path for PNG file
    """
    # Extract question data
    if 'results' not in results:
        raise ValueError("Results JSON must have 'results' key")
    
    results_dict = results['results']
    
    # Get first dataset (gsm8k, aime, etc.)
    dataset_name = list(results_dict.keys())[0]
    dataset_results = results_dict[dataset_name]
    
    if question_id >= len(dataset_results):
        raise ValueError(f"Question ID {question_id} out of range (max: {len(dataset_results)-1})")
    
    question_result = dataset_results[question_id]
    
    # Get ground truth and traces
    ground_truth = question_result.get('ground_truth', '')
    traces = question_result.get('valid_traces', [])
    
    if not traces:
        raise ValueError(f"No traces found for question {question_id}")
    
    print(f"Question {question_id}:")
    print(f"  Ground truth: {ground_truth}")
    print(f"  Number of traces: {len(traces)}")

    # Debug: Check first trace structure
    if traces:
        print(f"\nDebug - First trace keys: {traces[0].keys()}")
        confs_sample = traces[0].get('confs', [])
        print(f"Debug - confs type: {type(confs_sample)}")
        print(f"Debug - confs length: {len(confs_sample) if isinstance(confs_sample, list) else 'N/A'}")
        if isinstance(confs_sample, list) and len(confs_sample) > 0:
            print(f"Debug - First 5 confs: {confs_sample[:5]}")
            print(f"Debug - Confs range: [{min(confs_sample):.4f}, {max(confs_sample):.4f}]")

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot each trace
    traces_plotted = 0
    for trace in traces:
        confs = trace.get('confs', [])
        if not confs:
            continue

        if not isinstance(confs, list) or len(confs) == 0:
            continue

        traces_plotted += 1
        
        # Determine correctness
        answer = trace.get('extracted_answer') or trace.get('answer')
        is_correct = (answer == ground_truth)
        
        # Compute tail confidence evolution
        positions, tail_confs = compute_tail_evolution(confs)
        
        # Plot
        color = 'green' if is_correct else 'red'
        alpha = 0.6 if is_correct else 0.3
        linewidth = 1.5 if is_correct else 1.0
        
        ax.plot(positions, tail_confs, color=color, alpha=alpha, linewidth=linewidth)
    
    # Formatting
    ax.set_xlabel('Token Position', fontsize=14, fontweight='bold')
    ax.set_ylabel('Tail Confidence (mean of last 2048 tokens)', fontsize=14, fontweight='bold')
    ax.set_title(f'Tail Confidence Evolution - Question {question_id}\n'
                 f'Ground Truth: {ground_truth} | {len(traces)} traces',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label=f'Correct (answer={ground_truth})'),
        Patch(facecolor='red', alpha=0.3, label='Incorrect')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Set x-axis to start at 0
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize tail confidence for a single question')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results JSON file')
    parser.add_argument('--question_id', type=int, required=True,
                       help='Question ID to visualize')
    parser.add_argument('--output', type=str, default='confidence_visualization.png',
                       help='Output path for visualization (default: confidence_visualization.png)')
    
    args = parser.parse_args()
    
    print("Loading results...")
    results = load_results(args.results)
    
    print("Creating visualization...")
    visualize_question_confidence(results, args.question_id, args.output)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
