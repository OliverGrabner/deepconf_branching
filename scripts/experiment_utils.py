"""
Shared Utility Functions for Self-Consistency Experiments

This module contains common functions used across different experiment scripts:
- Dataset loading (AIME25, GSM8k)
- Historical statistics management
- Result saving and loading
- Visualization generation
- Summary printing

Usage:
    from experiment_utils import load_dataset, save_results, generate_visualizations
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_dataset

# Constants
TEMP_RESULTS_SUFFIX = "_temp.json"


# ============= DATASET LOADERS =============

def load_dataset_by_name(dataset_name: str, split: str = "test"):
    """
    Load dataset by name with auto-detection

    Args:
        dataset_name: "AIME2025-I", "AIME2025-II", "gsm8k", or "both" (for AIME)
        split: Dataset split (default: "test")

    Returns:
        List of (dataset_name, dataset) tuples
    """
    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        return [("gsm8k", ds)]

    elif dataset_name_lower in ["aime2025-i", "aime2025-ii"]:
        ds = load_dataset("opencompass/AIME2025", name=dataset_name, split=split)
        return [(dataset_name, ds)]

    elif dataset_name_lower == "both":
        # Load both AIME datasets
        ds1 = load_dataset("opencompass/AIME2025", name="AIME2025-I", split=split)
        ds2 = load_dataset("opencompass/AIME2025", name="AIME2025-II", split=split)
        return [("AIME2025-I", ds1), ("AIME2025-II", ds2)]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'AIME2025-I', 'AIME2025-II', 'gsm8k', or 'both'")


def extract_gsm8k_ground_truth(answer_text: str) -> str:
    """
    Extract ground truth number from GSM8k answer field

    GSM8k format: "reasoning text ... #### 123"
    """
    import re

    if "####" in answer_text:
        parts = answer_text.split("####")
        if len(parts) > 1:
            gt = parts[-1].strip()
            numbers = re.findall(r'-?\d+\.?\d*', gt)
            if numbers:
                return numbers[0]

    return answer_text.strip()


def get_question_and_ground_truth(dataset_name: str, question_data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract question and ground truth from dataset item

    Returns:
        (question_text, ground_truth)
    """
    question = question_data['question']

    if "gsm8k" in dataset_name.lower():
        answer_text = question_data['answer']
        ground_truth = extract_gsm8k_ground_truth(answer_text)
    else:
        # AIME format
        ground_truth = str(question_data.get('answer', '')).strip()

    return question, ground_truth


# ============= HISTORICAL STATISTICS =============

def load_historical_stats(stats_file: str) -> Dict[str, Dict[str, Any]]:
    """Load historical token statistics from JSON file"""
    with open(stats_file, 'r') as f:
        data = json.load(f)
    return data['statistics']


def get_average_tokens(
    historical_stats: Dict,
    dataset_name: str,
    question_idx: int,
    fallback_aime: int = 8000,
    fallback_gsm8k: int = 5000
) -> int:
    """
    Get historical average tokens for a specific question

    Args:
        historical_stats: Statistics dictionary
        dataset_name: Name of dataset
        question_idx: Question index
        fallback_aime: Fallback for AIME if not found
        fallback_gsm8k: Fallback for GSM8k if not found

    Returns:
        Average token count
    """
    # For GSM8k, stats are indexed directly by question index
    if "gsm8k" in dataset_name.lower():
        q_key = str(question_idx)
        if q_key in historical_stats:
            return int(historical_stats[q_key]['avg_tokens'])

        # Fallback: compute mean
        if historical_stats:
            all_avgs = [stats['avg_tokens'] for stats in historical_stats.values()]
            return int(sum(all_avgs) / len(all_avgs)) if all_avgs else fallback_gsm8k

        return fallback_gsm8k

    # For AIME, stats are nested by dataset name
    if dataset_name in historical_stats:
        q_key = str(question_idx)
        if q_key in historical_stats[dataset_name]:
            return int(historical_stats[dataset_name][q_key]['avg_tokens'])

        # Fallback: compute mean for this dataset
        all_avgs = [stats['avg_tokens'] for stats in historical_stats[dataset_name].values()]
        return int(sum(all_avgs) / len(all_avgs)) if all_avgs else fallback_aime

    return fallback_aime


# ============= RESULT SAVING =============

def save_results(
    all_results: Dict[str, List[Dict[str, Any]]],
    summary: Dict[str, Any],
    output_dir: str,
    metadata: Dict[str, Any],
    experiment_type: str
) -> str:
    """
    Save experiment results in multiple formats

    Args:
        all_results: Dictionary of {dataset_name: [question_results]}
        summary: Summary statistics
        output_dir: Output directory
        metadata: Metadata dictionary (model, parameters, etc.)
        experiment_type: "traditional" or "branching"

    Returns:
        Path to detailed JSON file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add experiment type to metadata
    metadata['experiment_type'] = experiment_type
    metadata['timestamp'] = timestamp

    # Save detailed results as JSON
    detailed_output = {
        'metadata': metadata,
        'results': all_results,
        'summary': summary
    }

    json_filename = os.path.join(output_dir, f"{experiment_type}_sc_detailed_{timestamp}.json")
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(detailed_output, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {json_filename}")

    # Save summary as CSV
    summary_rows = []
    for dataset_name, results in all_results.items():
        for i, result in enumerate(results):
            row = {
                'dataset': dataset_name,
                'question_id': i,
                'is_correct': result['is_correct'],
                'ground_truth': result['ground_truth'],
                'voted_answer': result['voted_answer'],
                'num_valid_traces': result['num_valid_traces'],
                'num_traces_generated': result['num_traces_generated'],
                'individual_trace_accuracy': result['individual_trace_accuracy'],
                'total_tokens': result['statistics']['total_tokens'],
                'total_time': result['statistics']['total_time'],
            }

            # Add branching-specific fields if present
            if experiment_type == "branching" and result.get('branch_genealogy'):
                genealogy_stats = result['branch_genealogy'].get('statistics', {})
                row.update({
                    'original_traces': genealogy_stats.get('original_traces', 0),
                    'branched_traces': genealogy_stats.get('branched_traces', 0),
                    'branch_events': genealogy_stats.get('total_branch_events', 0),
                })

            summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    csv_filename = os.path.join(output_dir, f"{experiment_type}_sc_summary_{timestamp}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Summary CSV saved to: {csv_filename}")

    # Save aggregate statistics
    stats_filename = os.path.join(output_dir, f"{experiment_type}_sc_stats_{timestamp}.json")
    with open(stats_filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Aggregate statistics saved to: {stats_filename}")

    return json_filename


def save_incremental_results(
    all_results: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
    timestamp: str,
    metadata: Dict[str, Any],
    experiment_type: str
) -> str:
    """
    Save incremental results during processing (temp file)

    Args:
        all_results: Current results
        output_dir: Output directory
        timestamp: Timestamp for this run
        metadata: Metadata dictionary
        experiment_type: "traditional" or "branching"

    Returns:
        Path to temp file
    """
    metadata_copy = metadata.copy()
    metadata_copy['experiment_type'] = experiment_type
    metadata_copy['timestamp'] = timestamp
    metadata_copy['status'] = 'in_progress'

    temp_output = {
        'metadata': metadata_copy,
        'results': all_results,
        'summary': None  # Computed at the end
    }

    temp_filename = os.path.join(output_dir, f"{experiment_type}_sc_detailed_{timestamp}{TEMP_RESULTS_SUFFIX}")
    with open(temp_filename, 'w', encoding='utf-8') as f:
        json.dump(temp_output, f, indent=2, ensure_ascii=False)

    return temp_filename


# ============= SUMMARY GENERATION =============

def generate_summary_report(all_results: Dict[str, List[Dict[str, Any]]], experiment_type: str) -> Dict[str, Any]:
    """
    Generate comprehensive summary statistics

    Args:
        all_results: Dictionary of {dataset_name: [question_results]}
        experiment_type: "traditional" or "branching"

    Returns:
        Summary dictionary with per-dataset and overall statistics
    """
    summary = {
        'overall': {},
        'by_dataset': {}
    }

    # Calculate per-dataset statistics
    for dataset_name, results in all_results.items():
        num_questions = len(results)
        num_correct = sum(1 for r in results if r['is_correct'])
        accuracy = num_correct / num_questions if num_questions > 0 else 0.0

        total_tokens = sum(r['statistics']['total_tokens'] for r in results)
        total_time = sum(r['statistics']['total_time'] for r in results)
        avg_tokens = total_tokens / num_questions if num_questions > 0 else 0
        avg_time = total_time / num_questions if num_questions > 0 else 0

        avg_individual_accuracy = sum(r['individual_trace_accuracy'] for r in results) / num_questions if num_questions > 0 else 0.0

        dataset_stats = {
            'num_questions': num_questions,
            'num_correct': num_correct,
            'accuracy': accuracy,
            'avg_individual_trace_accuracy': avg_individual_accuracy,
            'total_tokens': total_tokens,
            'avg_tokens_per_question': avg_tokens,
            'total_time': total_time,
            'avg_time_per_question': avg_time,
            'throughput_tokens_per_sec': total_tokens / total_time if total_time > 0 else 0,
        }

        # Add branching-specific statistics
        if experiment_type == "branching":
            total_branch_events = sum(
                r.get('branch_genealogy', {}).get('statistics', {}).get('total_branch_events', 0)
                for r in results
            )
            dataset_stats['total_branch_events'] = total_branch_events
            dataset_stats['avg_branch_events_per_question'] = total_branch_events / num_questions if num_questions > 0 else 0

        summary['by_dataset'][dataset_name] = dataset_stats

    # Calculate overall statistics
    all_question_results = [r for results in all_results.values() for r in results]
    total_questions = len(all_question_results)
    total_correct = sum(1 for r in all_question_results if r['is_correct'])
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0

    overall_tokens = sum(r['statistics']['total_tokens'] for r in all_question_results)
    overall_time = sum(r['statistics']['total_time'] for r in all_question_results)

    summary['overall'] = {
        'num_questions': total_questions,
        'num_correct': total_correct,
        'accuracy': overall_accuracy,
        'total_tokens': overall_tokens,
        'total_time': overall_time,
        'avg_tokens_per_question': overall_tokens / total_questions if total_questions > 0 else 0,
        'avg_time_per_question': overall_time / total_questions if total_questions > 0 else 0,
        'throughput_tokens_per_sec': overall_tokens / overall_time if overall_time > 0 else 0,
    }

    # Add branching-specific overall statistics
    if experiment_type == "branching":
        overall_branch_events = sum(
            r.get('branch_genealogy', {}).get('statistics', {}).get('total_branch_events', 0)
            for r in all_question_results
        )
        summary['overall']['total_branch_events'] = overall_branch_events
        summary['overall']['avg_branch_events_per_question'] = overall_branch_events / total_questions if total_questions > 0 else 0

    return summary


# ============= PRINTING UTILITIES =============

def print_question_summary(qid: int, result: Dict[str, Any], experiment_type: str):
    """Print a concise summary for a single question"""
    correctness = "✓" if result['is_correct'] else "✗"
    print(f"\nQ{qid}: {correctness}")
    print(f"  Ground Truth: {result['ground_truth']}")
    print(f"  Voted Answer: {result['voted_answer']}")
    print(f"  Valid Traces: {result['num_valid_traces']}/{result['num_traces_generated']}")
    print(f"  Individual Accuracy: {result['individual_trace_accuracy']:.1%}")

    if experiment_type == "branching" and result.get('branch_genealogy'):
        stats = result['branch_genealogy'].get('statistics', {})
        print(f"  Original Traces: {stats.get('original_traces', 0)}")
        print(f"  Branched Traces: {stats.get('branched_traces', 0)}")
        print(f"  Branch Events: {stats.get('total_branch_events', 0)}")

    print(f"  Tokens: {result['statistics']['total_tokens']} ({result['statistics']['avg_tokens_per_trace']:.1f} avg)")
    print(f"  Time: {result['statistics']['total_time']:.2f}s")


def print_final_summary(summary: Dict[str, Any], experiment_type: str):
    """Print formatted final summary"""
    title = f"{experiment_type.upper()} SELF-CONSISTENCY - FINAL SUMMARY"
    print("\n" + "="*80)
    print(title)
    print("="*80)

    # Per-dataset results
    if len(summary['by_dataset']) > 1:
        print("\nPer-Dataset Results:")
        print("-" * 80)
        for dataset_name, stats in summary['by_dataset'].items():
            print(f"\n{dataset_name}:")
            print(f"  Questions: {stats['num_questions']}")
            print(f"  Correct: {stats['num_correct']}/{stats['num_questions']} ({stats['accuracy']:.1%})")
            print(f"  Avg Individual Trace Accuracy: {stats['avg_individual_trace_accuracy']:.1%}")

            if experiment_type == "branching" and 'avg_branch_events_per_question' in stats:
                print(f"  Avg Branch Events: {stats['avg_branch_events_per_question']:.1f}")

            print(f"  Total Tokens: {stats['total_tokens']:,}")
            print(f"  Avg Tokens/Question: {stats['avg_tokens_per_question']:.1f}")
            print(f"  Total Time: {stats['total_time']:.2f}s")
            print(f"  Avg Time/Question: {stats['avg_time_per_question']:.2f}s")
            print(f"  Throughput: {stats['throughput_tokens_per_sec']:.1f} tokens/sec")

    # Overall results
    print("\n" + "-" * 80)
    if len(summary['by_dataset']) > 1:
        print("Overall Results (All Datasets):")
    else:
        print("Results:")
    print("-" * 80)
    overall = summary['overall']
    print(f"  Total Questions: {overall['num_questions']}")
    print(f"  Total Correct: {overall['num_correct']}/{overall['num_questions']} ({overall['accuracy']:.1%})")

    if experiment_type == "branching" and 'avg_branch_events_per_question' in overall:
        print(f"  Avg Branch Events: {overall['avg_branch_events_per_question']:.1f}")

    print(f"  Total Tokens: {overall['total_tokens']:,}")
    print(f"  Avg Tokens/Question: {overall['avg_tokens_per_question']:.1f}")
    print(f"  Total Time: {overall['total_time']:.2f}s ({overall['total_time']/60:.1f} minutes)")
    print(f"  Avg Time/Question: {overall['avg_time_per_question']:.2f}s")
    print(f"  Overall Throughput: {overall['throughput_tokens_per_sec']:.1f} tokens/sec")
    print("="*80)


# ============= VISUALIZATION =============

def generate_question_visualizations(
    result: Dict[str, Any],
    dataset_name: str,
    question_idx: int,
    viz_dir: str,
    timestamp: str,
    experiment_type: str
) -> bool:
    """
    Generate visualizations for a single question

    Args:
        result: Question result dictionary
        dataset_name: Name of dataset
        question_idx: Question index
        viz_dir: Visualization output directory
        timestamp: Timestamp for filenames
        experiment_type: "traditional" or "branching"

    Returns:
        True if successful, False otherwise
    """
    try:
        if experiment_type == "branching":
            from visualize_branching_results import (
                create_per_problem_summary,
                create_genealogy_graph,
                create_confidence_evolution_plot
            )

            # Prepare paths
            summary_path = os.path.join(viz_dir, f"summary_{dataset_name}_q{question_idx}_{timestamp}.png")
            genealogy_path = os.path.join(viz_dir, f"genealogy_{dataset_name}_q{question_idx}_{timestamp}.png")
            confidence_path = os.path.join(viz_dir, f"confidence_{dataset_name}_q{question_idx}_{timestamp}.png")

            # Generate 3 visualizations
            create_per_problem_summary(result, dataset_name, question_idx, summary_path)
            create_genealogy_graph(
                result.get('branch_genealogy', {}),
                result.get('full_traces', []),
                result.get('ground_truth', ''),
                genealogy_path
            )
            create_confidence_evolution_plot(
                result.get('full_traces', []),
                result.get('branch_genealogy', {}),
                result.get('branching_config', {}),
                result.get('ground_truth', ''),
                confidence_path
            )

        elif experiment_type == "traditional":
            # Import matplotlib for traditional SC plots
            import matplotlib.pyplot as plt
            import numpy as np
            from collections import Counter

            # Create a 2-panel summary for traditional SC
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Panel 1: Vote distribution
            vote_dist = result.get('vote_distribution', {})
            if vote_dist:
                answers = list(vote_dist.keys())
                votes = list(vote_dist.values())
                colors = ['green' if str(ans) == str(result.get('ground_truth', '')) else 'red' for ans in answers]

                ax1.bar(range(len(answers)), votes, color=colors, alpha=0.7)
                ax1.set_xlabel('Answer', fontsize=12)
                ax1.set_ylabel('Number of Votes', fontsize=12)
                ax1.set_title(f'Vote Distribution (Q{question_idx})', fontsize=14, fontweight='bold')
                ax1.set_xticks(range(len(answers)))
                ax1.set_xticklabels([str(a)[:10] for a in answers], rotation=45, ha='right')
                ax1.grid(axis='y', alpha=0.3)

                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='green', alpha=0.7, label='Correct'),
                    Patch(facecolor='red', alpha=0.7, label='Incorrect')
                ]
                ax1.legend(handles=legend_elements, loc='upper right')

            # Panel 2: Token distribution across traces
            valid_traces = result.get('valid_traces', [])
            if valid_traces:
                token_counts = [t.get('num_tokens', 0) for t in valid_traces if t.get('num_tokens', 0) > 0]
                if token_counts:
                    ax2.hist(token_counts, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
                    ax2.axvline(np.mean(token_counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(token_counts):.0f}')
                    ax2.set_xlabel('Tokens per Trace', fontsize=12)
                    ax2.set_ylabel('Frequency', fontsize=12)
                    ax2.set_title(f'Token Distribution (Q{question_idx})', fontsize=14, fontweight='bold')
                    ax2.legend()
                    ax2.grid(axis='y', alpha=0.3)

            # Add overall stats
            fig.suptitle(
                f"{dataset_name} - Q{question_idx} - Traditional SC\n"
                f"Voted: {result.get('voted_answer', 'N/A')} | GT: {result.get('ground_truth', 'N/A')} | "
                f"{'✓ CORRECT' if result.get('is_correct') else '✗ INCORRECT'}",
                fontsize=16,
                fontweight='bold',
                color='green' if result.get('is_correct') else 'red'
            )

            plt.tight_layout(rect=[0, 0, 1, 0.93])

            summary_path = os.path.join(viz_dir, f"summary_{dataset_name}_q{question_idx}_{timestamp}.png")
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.close()

        return True

    except ImportError:
        print(f"  ⚠️  Visualization module not found for Q{question_idx}")
        return False
    except Exception as e:
        print(f"  ⚠️  Visualization failed for Q{question_idx}: {e}")
        return False


def generate_dataset_visualizations(
    all_results: Dict[str, List[Dict[str, Any]]],
    viz_dir: str,
    timestamp: str,
    experiment_type: str
) -> bool:
    """
    Generate dataset-wide visualizations

    Args:
        all_results: Dictionary of {dataset_name: [question_results]}
        viz_dir: Visualization output directory
        timestamp: Timestamp for filenames
        experiment_type: "traditional" or "branching"

    Returns:
        True if successful, False otherwise
    """
    try:
        if experiment_type == "branching":
            from visualize_branching_results import create_token_usage_plot, create_accuracy_analysis_plot

            # Token usage plot
            token_path = os.path.join(viz_dir, f"token_usage_{timestamp}.png")
            create_token_usage_plot(all_results, token_path)
            print(f"  ✓ Token usage plot: {token_path}")

            # Accuracy analysis plot
            accuracy_path = os.path.join(viz_dir, f"accuracy_analysis_{timestamp}.png")
            create_accuracy_analysis_plot(all_results, accuracy_path)
            print(f"  ✓ Accuracy analysis plot: {accuracy_path}")

        # Traditional SC visualizations would go here (not implemented yet)
        # else:
        #     pass

        return True

    except ImportError:
        print("\n⚠️  Visualization module not found")
        return False
    except Exception as e:
        print(f"\n⚠️  Dataset-wide visualization failed: {e}")
        return False


# ============= HELPER FUNCTIONS =============

def create_metadata_dict(args: argparse.Namespace, experiment_type: str) -> Dict[str, Any]:
    """
    Create metadata dictionary from command-line arguments

    Args:
        args: Parsed command-line arguments
        experiment_type: "traditional" or "branching"

    Returns:
        Metadata dictionary
    """
    metadata = {
        'model': args.model,
        'model_type': args.model_type,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'max_tokens': args.max_tokens,
    }

    if experiment_type == "traditional":
        metadata['num_traces'] = args.num_traces
    elif experiment_type == "branching":
        metadata['start_traces'] = args.start_traces
        metadata['max_traces'] = args.max_traces
        metadata['selected_percent'] = args.selected_percent
        metadata['n_iterations'] = args.n_iterations
        metadata['branch_goal'] = args.branch_goal

        if hasattr(args, 'historical_stats'):
            metadata['historical_stats_file'] = args.historical_stats

    return metadata
