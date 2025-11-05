"""
Unified Visualization Script for Self-Consistency Experiments

Auto-detects experiment type (traditional vs branching) and generates appropriate visualizations.

Usage:
    # Visualize all questions
    python visualize_results.py --results outputs/experiment_detailed_20250115.json

    # Visualize specific question only
    python visualize_results.py --results outputs/experiment_detailed_20250115.json --question_id 0

    # Specify output directory
    python visualize_results.py --results outputs/experiment_detailed_20250115.json --output_dir viz/
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Check matplotlib availability
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib networkx")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load experiment results from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


def detect_experiment_type(data: Dict[str, Any]) -> str:
    """
    Auto-detect experiment type from results data

    Returns:
        "traditional" or "branching"
    """
    metadata = data.get('metadata', {})

    # Check metadata first
    if 'experiment_type' in metadata:
        return metadata['experiment_type']

    # Check for branching-specific fields
    results = data.get('results', {})
    if results:
        first_dataset = next(iter(results.values()))
        if first_dataset:
            first_question = first_dataset[0]
            if 'branch_genealogy' in first_question or 'branching_config' in first_question:
                return "branching"

    # Default to traditional
    return "traditional"


def create_visualizations_branching(
    results_filepath: str,
    output_dir: str,
    question_id: Optional[int] = None
):
    """Create visualizations for branching self-consistency experiments"""
    # Import branching visualization module
    from visualize_branching_results import (
        create_per_problem_summary,
        create_genealogy_graph,
        create_confidence_evolution_plot,
        create_token_usage_plot,
        create_accuracy_analysis_plot
    )

    print(f"\nLoading results from: {results_filepath}")
    data = load_results(results_filepath)

    results_by_dataset = data.get('results', {})
    metadata = data.get('metadata', {})

    os.makedirs(output_dir, exist_ok=True)

    timestamp = metadata.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))

    print(f"\n{'='*80}")
    print("CREATING BRANCHING SC VISUALIZATIONS")
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

        try:
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

            print("✓")

        except Exception as e:
            print(f"✗ ({e})")
            continue

    # Overall visualizations (only if not single question)
    if question_id is None:
        print("\nCreating dataset-wide visualizations...")

        # Token usage
        token_path = os.path.join(output_dir, f"token_usage_{timestamp}.png")
        try:
            create_token_usage_plot(results_by_dataset, token_path)
            print(f"  ✓ Token usage plot")
        except Exception as e:
            print(f"  ✗ Token usage plot ({e})")

        # Accuracy analysis
        accuracy_path = os.path.join(output_dir, f"accuracy_analysis_{timestamp}.png")
        try:
            create_accuracy_analysis_plot(results_by_dataset, accuracy_path)
            print(f"  ✓ Accuracy analysis plot")
        except Exception as e:
            print(f"  ✗ Accuracy analysis plot ({e})")

    print(f"\n{'='*80}")
    print("BRANCHING VISUALIZATION COMPLETE")
    print(f"Output directory: {output_dir}")
    print('='*80)


def create_visualizations_traditional(
    results_filepath: str,
    output_dir: str,
    question_id: Optional[int] = None
):
    """
    Create visualizations for traditional self-consistency experiments

    Note: Currently uses ASCII-based visualizations from visualize_sc_results.py
    TODO: Could be extended with matplotlib plots similar to branching SC
    """
    print(f"\nLoading results from: {results_filepath}")
    data = load_results(results_filepath)

    results_by_dataset = data.get('results', {})

    print(f"\n{'='*80}")
    print("TRADITIONAL SC VISUALIZATIONS")
    print('='*80)

    # For now, use the existing visualize_sc_results.py functionality
    print("\nTraditional SC currently uses ASCII-based visualizations.")
    print("For detailed analysis, run:")
    print(f"  python scripts/visualize_sc_results.py {results_filepath}")

    # Could add matplotlib-based visualizations here in the future
    print(f"\n{'='*80}")
    print("TRADITIONAL VISUALIZATION COMPLETE")
    print('='*80)


def main():
    parser = argparse.ArgumentParser(
        description='Unified Visualization for Self-Consistency Experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--results', type=str, required=True,
                       help='Path to experiment results JSON file')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--question_id', type=int, default=None,
                       help='Visualize specific question only (optional)')
    parser.add_argument('--experiment_type', type=str, default=None,
                       choices=['traditional', 'branching'],
                       help='Force experiment type (auto-detected if not specified)')

    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("\nERROR: matplotlib not installed!")
        print("Install with: pip install matplotlib networkx")
        return 1

    # Load data to detect experiment type
    print(f"Loading results from: {args.results}")
    data = load_results(args.results)

    # Detect or use specified experiment type
    if args.experiment_type:
        experiment_type = args.experiment_type
        print(f"Using specified experiment type: {experiment_type}")
    else:
        experiment_type = detect_experiment_type(data)
        print(f"Auto-detected experiment type: {experiment_type}")

    # Create appropriate visualizations
    if experiment_type == "branching":
        create_visualizations_branching(args.results, args.output_dir, args.question_id)
    else:
        create_visualizations_traditional(args.results, args.output_dir, args.question_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
