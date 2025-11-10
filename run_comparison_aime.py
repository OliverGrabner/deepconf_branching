#!/usr/bin/env python3
"""
Run comparison between branching and standard self-consistency on AIME2025-I
Generates comparison charts for accuracy and token usage

Usage:
    python run_comparison_aime.py
    python run_comparison_aime.py --dry_run  # Test on first 3 problems
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def run_experiment(mode, output_dir, args):
    """Run a single experiment (branching or standard)"""

    cmd = [
        sys.executable, "run_unified.py",
        "--mode", mode,
        "--dataset", "AIME2025-I",
        "--model", args.model,
        "--tensor_parallel_size", str(args.tensor_parallel_size),
        "--gpu_memory_utilization", str(args.gpu_memory_utilization),
        "--max_num_seqs", str(args.max_num_seqs),
        "--temperature", str(args.temperature),
        "--max_tokens", str(args.max_tokens),
        "--output_dir", output_dir
    ]

    if mode == "branching":
        cmd.extend([
            "--initial_branches", str(args.initial_branches),
            "--max_total_branches", str(args.max_total_branches),
            "--confidence_threshold", str(args.confidence_threshold)
        ])
    else:  # standard
        cmd.extend([
            "--num_traces", str(args.num_traces)
        ])

    if args.dry_run:
        cmd.extend(["--end_idx", "3"])

    print(f"\n{'='*80}")
    print(f"Running {mode.upper()} experiment")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output directory: {output_dir}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"ERROR: {mode} experiment failed with return code {result.returncode}")
        return None

    # Find the latest JSON results file
    json_files = list(Path(output_dir).glob(f"{mode}_*.json"))
    if not json_files:
        print(f"ERROR: No results JSON file found in {output_dir}")
        return None

    latest_json = max(json_files, key=lambda p: p.stat().st_mtime)

    with open(latest_json, 'r') as f:
        return json.load(f)


def extract_statistics(results):
    """Extract summary statistics from results"""

    if not results or 'results' not in results:
        return None

    all_results = []
    # Handle nested structure - results might be per dataset
    if isinstance(results['results'], dict):
        for dataset_results in results['results'].values():
            if isinstance(dataset_results, list):
                all_results.extend(dataset_results)
    elif isinstance(results['results'], list):
        all_results = results['results']

    if not all_results:
        return None

    total_problems = len(all_results)
    voted_correct = sum(1 for r in all_results if r.get('voted_correct', False))
    accuracy = (voted_correct / total_problems * 100) if total_problems > 0 else 0
    total_tokens = sum(r.get('total_tokens', 0) for r in all_results)
    avg_tokens = total_tokens / total_problems if total_problems > 0 else 0

    # Also calculate average individual trace accuracy
    individual_accuracies = [r.get('accuracy', 0) for r in all_results]
    avg_individual_accuracy = np.mean(individual_accuracies) if individual_accuracies else 0

    return {
        'total_problems': total_problems,
        'correct': voted_correct,
        'accuracy': accuracy,
        'total_tokens': total_tokens,
        'avg_tokens_per_problem': avg_tokens,
        'avg_individual_trace_accuracy': avg_individual_accuracy
    }


def create_comparison_charts(branching_stats, standard_stats, output_dir):
    """Create comparison bar charts"""

    # Set style for better-looking plots
    plt.style.use('seaborn-v0_8-darkgrid')

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Chart 1: Accuracy Comparison
    methods = ['Branching\n(8→32)', 'Standard SC\n(32 traces)']
    accuracies = [branching_stats['accuracy'], standard_stats['accuracy']]
    colors = ['#2E86AB', '#A23B72']

    bars1 = ax1.bar(methods, accuracies, color=colors, width=0.6)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy Comparison on AIME2025-I', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        # Also show correct/total
        if bar.get_x() == bars1[0].get_x():
            correct = branching_stats['correct']
            total = branching_stats['total_problems']
        else:
            correct = standard_stats['correct']
            total = standard_stats['total_problems']
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                f'({correct}/{total})', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')

    # Chart 2: Token Usage Comparison
    tokens = [branching_stats['total_tokens'], standard_stats['total_tokens']]
    tokens_k = [t/1000 for t in tokens]  # Convert to thousands

    bars2 = ax2.bar(methods, tokens_k, color=colors, width=0.6)
    ax2.set_ylabel('Total Tokens (thousands)', fontsize=12)
    ax2.set_title('Token Usage Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, tok, tok_k in zip(bars2, tokens, tokens_k):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(tokens_k)*0.01,
                f'{tok:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        # Show average per problem
        if bar.get_x() == bars2[0].get_x():
            avg = branching_stats['avg_tokens_per_problem']
        else:
            avg = standard_stats['avg_tokens_per_problem']
        ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                f'Avg: {avg:,.0f}', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')

    # Add subtitle with model info
    fig.suptitle(f'DeepSeek-R1-Distill-Qwen-7B | Temperature: 0.6 | Max Tokens: 32768',
                 fontsize=10, y=0.02)

    plt.tight_layout()

    # Save the figure
    chart_path = os.path.join(output_dir, 'comparison_charts.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison charts saved to: {chart_path}")

    # Also save individual charts
    # Accuracy chart alone
    fig_acc, ax_acc = plt.subplots(figsize=(8, 6))
    bars_acc = ax_acc.bar(methods, accuracies, color=colors, width=0.6)
    ax_acc.set_ylabel('Accuracy (%)', fontsize=12)
    ax_acc.set_title('Accuracy Comparison on AIME2025-I', fontsize=14, fontweight='bold')
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(axis='y', alpha=0.3)

    for bar, acc in zip(bars_acc, accuracies):
        height = bar.get_height()
        ax_acc.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    accuracy_path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(accuracy_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Token chart alone
    fig_tok, ax_tok = plt.subplots(figsize=(8, 6))
    bars_tok = ax_tok.bar(methods, tokens_k, color=colors, width=0.6)
    ax_tok.set_ylabel('Total Tokens (thousands)', fontsize=12)
    ax_tok.set_title('Token Usage Comparison', fontsize=14, fontweight='bold')
    ax_tok.grid(axis='y', alpha=0.3)

    for bar, tok in zip(bars_tok, tokens):
        height = bar.get_height()
        ax_tok.text(bar.get_x() + bar.get_width()/2., height + max(tokens_k)*0.01,
                   f'{tok:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    tokens_path = os.path.join(output_dir, 'tokens_comparison.png')
    plt.savefig(tokens_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Individual charts saved to:")
    print(f"   - {accuracy_path}")
    print(f"   - {tokens_path}")

    return chart_path


def main():
    parser = argparse.ArgumentParser(
        description='Compare branching vs standard self-consistency on AIME2025-I'
    )

    # Model configuration
    parser.add_argument('--model', type=str,
                       default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
                       help='Model to use')
    parser.add_argument('--tensor_parallel_size', type=int, default=2,
                       help='Number of GPUs for tensor parallelism')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                       help='GPU memory utilization')
    parser.add_argument('--max_num_seqs', type=int, default=128,
                       help='Maximum number of sequences')

    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=32768,
                       help='Maximum tokens per generation')

    # Branching parameters
    parser.add_argument('--initial_branches', type=int, default=8,
                       help='Initial branches for branching mode')
    parser.add_argument('--max_total_branches', type=int, default=32,
                       help='Max total branches for branching mode')
    parser.add_argument('--confidence_threshold', type=float, default=1.5,
                       help='Confidence threshold for branching')

    # Standard SC parameters
    parser.add_argument('--num_traces', type=int, default=32,
                       help='Number of traces for standard mode')

    # Other options
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: results/comparison_[timestamp])')
    parser.add_argument('--dry_run', action='store_true',
                       help='Test on first 3 problems only')
    parser.add_argument('--skip_branching', action='store_true',
                       help='Skip branching experiment')
    parser.add_argument('--skip_standard', action='store_true',
                       help='Skip standard experiment')

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir is None:
        args.output_dir = f'results/comparison_{timestamp}'

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("AIME2025-I COMPARISON EXPERIMENT")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Branching: {args.initial_branches} initial → {args.max_total_branches} total")
    print(f"Standard: {args.num_traces} traces")
    print(f"Output directory: {args.output_dir}")
    if args.dry_run:
        print("🔬 DRY RUN MODE: Testing on first 3 problems only")
    print("="*80)

    # Run experiments
    branching_results = None
    standard_results = None

    if not args.skip_branching:
        branching_dir = os.path.join(args.output_dir, 'branching')
        os.makedirs(branching_dir, exist_ok=True)
        branching_results = run_experiment('branching', branching_dir, args)

    if not args.skip_standard:
        standard_dir = os.path.join(args.output_dir, 'standard')
        os.makedirs(standard_dir, exist_ok=True)
        standard_results = run_experiment('standard', standard_dir, args)

    # Extract statistics
    branching_stats = None
    standard_stats = None

    if branching_results:
        branching_stats = extract_statistics(branching_results)
        if branching_stats:
            print("\n" + "="*80)
            print("BRANCHING RESULTS")
            print("="*80)
            print(f"Problems: {branching_stats['total_problems']}")
            print(f"Correct: {branching_stats['correct']}")
            print(f"Accuracy: {branching_stats['accuracy']:.1f}%")
            print(f"Total tokens: {branching_stats['total_tokens']:,}")
            print(f"Avg tokens/problem: {branching_stats['avg_tokens_per_problem']:,.0f}")
            print(f"Avg individual trace accuracy: {branching_stats['avg_individual_trace_accuracy']:.1f}%")

    if standard_results:
        standard_stats = extract_statistics(standard_results)
        if standard_stats:
            print("\n" + "="*80)
            print("STANDARD SELF-CONSISTENCY RESULTS")
            print("="*80)
            print(f"Problems: {standard_stats['total_problems']}")
            print(f"Correct: {standard_stats['correct']}")
            print(f"Accuracy: {standard_stats['accuracy']:.1f}%")
            print(f"Total tokens: {standard_stats['total_tokens']:,}")
            print(f"Avg tokens/problem: {standard_stats['avg_tokens_per_problem']:,.0f}")
            print(f"Avg individual trace accuracy: {standard_stats['avg_individual_trace_accuracy']:.1f}%")

    # Create comparison charts if we have both results
    if branching_stats and standard_stats:
        print("\n" + "="*80)
        print("CREATING COMPARISON CHARTS")
        print("="*80)
        create_comparison_charts(branching_stats, standard_stats, args.output_dir)

        # Save combined summary
        combined_summary = {
            'metadata': {
                'model': args.model,
                'temperature': args.temperature,
                'max_tokens': args.max_tokens,
                'timestamp': timestamp,
                'dry_run': args.dry_run
            },
            'branching': branching_stats,
            'standard': standard_stats,
            'comparison': {
                'accuracy_improvement': branching_stats['accuracy'] - standard_stats['accuracy'],
                'token_increase': branching_stats['total_tokens'] - standard_stats['total_tokens'],
                'token_increase_pct': ((branching_stats['total_tokens'] / standard_stats['total_tokens']) - 1) * 100
            }
        }

        summary_path = os.path.join(args.output_dir, 'comparison_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(combined_summary, f, indent=2)

        print(f"\n✅ Summary saved to: {summary_path}")

        # Print final comparison
        print("\n" + "="*80)
        print("FINAL COMPARISON")
        print("="*80)
        print(f"Accuracy: Branching {branching_stats['accuracy']:.1f}% vs Standard {standard_stats['accuracy']:.1f}%")
        if branching_stats['accuracy'] > standard_stats['accuracy']:
            print(f"🎯 Branching is {branching_stats['accuracy'] - standard_stats['accuracy']:.1f}% better!")
        elif standard_stats['accuracy'] > branching_stats['accuracy']:
            print(f"🎯 Standard is {standard_stats['accuracy'] - branching_stats['accuracy']:.1f}% better!")
        else:
            print("🎯 Both methods achieved the same accuracy!")

        print(f"\nToken usage: Branching {branching_stats['total_tokens']:,} vs Standard {standard_stats['total_tokens']:,}")
        if branching_stats['total_tokens'] > standard_stats['total_tokens']:
            print(f"💰 Branching used {combined_summary['comparison']['token_increase_pct']:.1f}% more tokens")
        else:
            print(f"💰 Branching used {abs(combined_summary['comparison']['token_increase_pct']):.1f}% fewer tokens")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()