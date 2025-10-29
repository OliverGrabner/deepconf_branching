"""
Branching Experiment Test Script
Configured for limited hardware (3x NVIDIA A5000 GPUs)
Using smaller models suitable for testing

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import sys
import json
import pickle
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

from vllm import SamplingParams
from deepconf.branching_wrapper import BranchingDeepThinkLLM

# Configure GPU visibility - Use only GPUs 0, 1, 2
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging to both file and console"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"branching_test_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Using GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")

    return logger


def plot_trace_confidence(trace: dict, ax=None, label=None, alpha=1.0):
    """Plot confidence values over token positions for a single trace"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    if 'confs' not in trace or not trace['confs']:
        return ax

    confs = trace['confs']
    positions = np.arange(len(confs))

    # Plot confidence line
    line = ax.plot(positions, confs, alpha=alpha, label=label, linewidth=1.5)[0]
    color = line.get_color()

    # Mark branch points if they exist
    if 'branch_history' in trace:
        for branch in trace['branch_history']:
            step = branch.get('step', 0)
            conf = branch.get('confidence', 0)
            ax.scatter(step, conf, color=color, s=100, marker='*',
                      edgecolor='black', linewidth=1, zorder=5)

    ax.set_xlabel('Token Position')
    ax.set_ylabel('Confidence Score')
    ax.set_title('Token-Level Confidence Over Generation')
    ax.grid(True, alpha=0.3)

    return ax


def plot_confidence_with_sliding_window(trace: dict, window_size: int = 128, save_path: str = None):
    """Plot both raw confidence and sliding window average"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    if 'confs' not in trace or not trace['confs']:
        plt.close(fig)
        return None

    confs = np.array(trace['confs'])
    positions = np.arange(len(confs))

    # Plot raw confidence
    ax1.plot(positions, confs, alpha=0.6, label='Raw Confidence', color='blue')
    ax1.set_ylabel('Raw Confidence')
    ax1.set_title(f'Token-Level Confidence (Trace: {trace.get("trace_id", "unknown")})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Calculate and plot sliding window average
    if len(confs) >= window_size:
        sliding_avg = np.convolve(confs, np.ones(window_size)/window_size, mode='valid')
        sliding_positions = np.arange(window_size//2, len(confs) - window_size//2)

        ax2.plot(sliding_positions, sliding_avg, label=f'Sliding Avg (window={window_size})',
                color='green', linewidth=2)

        # Mark high confidence regions
        threshold = np.percentile(sliding_avg, 75)
        ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.5,
                   label=f'75th percentile: {threshold:.3f}')

        # Highlight potential branching regions
        high_conf_mask = sliding_avg > threshold
        if np.any(high_conf_mask):
            ax2.fill_between(sliding_positions, 0, sliding_avg,
                           where=high_conf_mask, alpha=0.3, color='red',
                           label='High Confidence Regions')

    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Sliding Window Confidence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_branching_tree_confidence(result_data: dict, save_path: str = None):
    """Plot confidence traces organized by their branching relationships"""
    if 'all_traces' not in result_data:
        return None

    traces = result_data['all_traces']

    # Group traces by depth
    depth_groups = {}
    for trace in traces:
        depth = trace.get('depth', 0)
        if depth not in depth_groups:
            depth_groups[depth] = []
        depth_groups[depth].append(trace)

    # Create subplots for each depth level
    num_depths = len(depth_groups)
    fig, axes = plt.subplots(num_depths, 1, figsize=(14, 5*num_depths),
                            sharex=True, squeeze=False)

    # Plot traces by depth
    for depth_idx, (depth, traces_at_depth) in enumerate(sorted(depth_groups.items())):
        ax = axes[depth_idx, 0]

        for trace_idx, trace in enumerate(traces_at_depth):
            label = f"{trace.get('trace_id', f'trace_{trace_idx}')} (parent: {trace.get('parent_id', 'None')})"
            plot_trace_confidence(trace, ax=ax, label=label, alpha=0.7)

        ax.set_title(f'Depth {depth} Traces ({len(traces_at_depth)} traces)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    axes[-1, 0].set_xlabel('Token Position')
    plt.suptitle('Branching Tree: Confidence by Depth', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_branching_statistics(result_data: dict, save_path: str = None):
    """Plot statistics about the branching experiment"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    branching_stats = result_data.get('branching_stats', {})
    all_traces = result_data.get('all_traces', [])

    # 1. Confidence distribution at branch points
    ax = axes[0, 0]
    branch_confidences = []
    for trace in all_traces:
        if 'branch_history' in trace:
            for branch in trace['branch_history']:
                branch_confidences.append(branch.get('confidence', 0))

    if branch_confidences:
        ax.hist(branch_confidences, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(branch_confidences), color='red', linestyle='--',
                  label=f'Mean: {np.mean(branch_confidences):.3f}')
        ax.set_xlabel('Confidence at Branch Point')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Confidence at Branching Points')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Branch position distribution
    ax = axes[0, 1]
    branch_positions = []
    for trace in all_traces:
        if 'branch_history' in trace:
            for branch in trace['branch_history']:
                branch_positions.append(branch.get('step', 0))

    if branch_positions:
        ax.hist(branch_positions, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Branch Count')
        ax.set_title('Distribution of Branching Positions')
        ax.grid(True, alpha=0.3)

    # 3. Trace length distribution
    ax = axes[1, 0]
    trace_lengths = [len(t.get('confs', [])) for t in all_traces if 'confs' in t]
    if trace_lengths:
        ax.hist(trace_lengths, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(trace_lengths), color='red', linestyle='--',
                  label=f'Mean: {np.mean(trace_lengths):.0f}')
        ax.set_xlabel('Trace Length (tokens)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Trace Lengths')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. Confidence statistics by depth
    ax = axes[1, 1]
    depth_confs = {}
    for trace in all_traces:
        depth = trace.get('depth', 0)
        if 'confs' in trace and trace['confs']:
            if depth not in depth_confs:
                depth_confs[depth] = []
            depth_confs[depth].extend(trace['confs'])

    if depth_confs:
        depths = sorted(depth_confs.keys())
        means = [np.mean(depth_confs[d]) for d in depths]
        stds = [np.std(depth_confs[d]) for d in depths]

        ax.errorbar(depths, means, yerr=stds, marker='o', capsize=5, capthick=2, linewidth=2)
        ax.set_xlabel('Trace Depth')
        ax.set_ylabel('Average Confidence')
        ax.set_title('Confidence by Trace Depth')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def run_branching_experiment(
    question: str,
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    initial_branches: int = 2,
    max_total_branches: int = 6,
    confidence_threshold: float = 1.5,
    window_size: int = 128,
    max_tokens: int = 4000,
    logger: logging.Logger = None
) -> dict:
    """
    Run a branching experiment with the specified configuration

    Args:
        question: Question to process
        model: Model name/path (default: small model for testing)
        initial_branches: Number of initial traces
        max_total_branches: Maximum total traces
        confidence_threshold: Confidence threshold for branching
        window_size: Sliding window size
        max_tokens: Maximum tokens per generation
        logger: Logger instance

    Returns:
        Dictionary with experiment results
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Starting Branching Experiment")
    logger.info("=" * 80)
    logger.info(f"Model: {model}")
    logger.info(f"Question: {question}")
    logger.info(f"Initial branches: {initial_branches}")
    logger.info(f"Max total branches: {max_total_branches}")
    logger.info(f"Confidence threshold: {confidence_threshold}")
    logger.info(f"Window size: {window_size}")
    logger.info(f"Max tokens: {max_tokens}")

    # Initialize the branching LLM
    logger.info("\nInitializing BranchingDeepThinkLLM...")
    branching_llm = BranchingDeepThinkLLM(
        model=model,
        tensor_parallel_size=3,  # 3 GPUs available
        enable_prefix_caching=True,
        trust_remote_code=True
    )

    # Prepare prompt
    logger.info("Preparing prompt...")
    messages = [{"role": "user", "content": question}]
    prompt = branching_llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=max_tokens,
        logprobs=20,
    )

    # Run the experiment
    logger.info("\nRunning branching experiment...")
    result = branching_llm.branching_deepthink(
        prompt=prompt,
        initial_branches=initial_branches,
        max_total_branches=max_total_branches,
        confidence_threshold=confidence_threshold,
        window_size=window_size,
        sampling_params=sampling_params
    )

    # Convert to dict for saving and visualization
    result_data = result.to_dict()
    result_data['question'] = question
    result_data['timestamp'] = datetime.now().isoformat()

    logger.info("\nExperiment completed successfully!")
    logger.info(f"Total traces generated: {result.total_traces_count}")
    logger.info(f"Total tokens: {result.total_tokens}")
    logger.info(f"Average tokens per trace: {result.avg_tokens_per_trace:.1f}")
    logger.info(f"Total time: {result.total_time:.2f}s")

    if result.voting_results:
        logger.info("\nVoting Results:")
        for method, vote_result in result.voting_results.items():
            if vote_result:
                logger.info(f"  {method}: {vote_result.get('answer', 'N/A')}")

    logger.info("\nBranching Statistics:")
    for key, value in result.branching_stats.items():
        logger.info(f"  {key}: {value}")

    return result_data


def save_results_and_plots(result_data: dict, output_dir: str = "outputs",
                          image_dir: str = "images", logger: logging.Logger = None):
    """Save experiment results and generate plots"""
    if logger is None:
        logger = logging.getLogger(__name__)

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save pickle file
    pkl_path = os.path.join(output_dir, f"branching_test_{timestamp}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(result_data, f)
    logger.info(f"\nResults saved to: {pkl_path}")

    # Save JSON file (for human readability, excluding some fields)
    json_data = {k: v for k, v in result_data.items()
                 if k not in ['all_traces']}  # Exclude large trace data
    json_data['num_traces'] = len(result_data.get('all_traces', []))

    json_path = os.path.join(output_dir, f"branching_test_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    logger.info(f"Summary saved to: {json_path}")

    # Generate and save plots
    logger.info("\nGenerating plots...")

    # Plot 1: First trace with sliding window
    if result_data.get('all_traces'):
        first_trace = result_data['all_traces'][0]
        plot_path = os.path.join(image_dir, f"trace_confidence_{timestamp}.png")
        plot_confidence_with_sliding_window(first_trace, window_size=128, save_path=plot_path)
        logger.info(f"  Saved trace confidence plot: {plot_path}")

        # Plot 2: Branching tree
        tree_path = os.path.join(image_dir, f"branching_tree_{timestamp}.png")
        plot_branching_tree_confidence(result_data, save_path=tree_path)
        logger.info(f"  Saved branching tree plot: {tree_path}")

        # Plot 3: Branching statistics
        stats_path = os.path.join(image_dir, f"branching_stats_{timestamp}.png")
        plot_branching_statistics(result_data, save_path=stats_path)
        logger.info(f"  Saved branching statistics plot: {stats_path}")

    logger.info("\nAll plots saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Run branching experiment on limited hardware (3x NVIDIA A5000)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment configuration
    parser.add_argument('--question', type=str,
                       default="What is 15% of 240?",
                       help='Question to process')
    parser.add_argument('--model', type=str,
                       default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                       help='Model to use (small model for testing)')

    # Branching parameters
    parser.add_argument('--initial_branches', type=int, default=2,
                       help='Number of initial traces')
    parser.add_argument('--max_total_branches', type=int, default=6,
                       help='Maximum total traces including branches')
    parser.add_argument('--confidence_threshold', type=float, default=1.5,
                       help='Confidence threshold for branching')
    parser.add_argument('--window_size', type=int, default=128,
                       help='Sliding window size for confidence')
    parser.add_argument('--max_tokens', type=int, default=4000,
                       help='Maximum tokens per generation')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory for output files')
    parser.add_argument('--image_dir', type=str, default='images',
                       help='Directory for plots')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for log files')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_dir)

    try:
        # Run experiment
        result_data = run_branching_experiment(
            question=args.question,
            model=args.model,
            initial_branches=args.initial_branches,
            max_total_branches=args.max_total_branches,
            confidence_threshold=args.confidence_threshold,
            window_size=args.window_size,
            max_tokens=args.max_tokens,
            logger=logger
        )

        # Save results and plots
        save_results_and_plots(
            result_data,
            output_dir=args.output_dir,
            image_dir=args.image_dir,
            logger=logger
        )

        logger.info("\n" + "=" * 80)
        logger.info("Experiment completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\nError during experiment: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
