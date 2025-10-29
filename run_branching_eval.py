"""
Branching Experiment with Evaluation Script
Performs true prefix-based branching with detailed evaluation and prints

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
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

from vllm import SamplingParams
from deepconf.branching_wrapper import BranchingDeepThinkLLM


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging to both file and console"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"branching_eval_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Using GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")

    return logger


def simple_answer_match(predicted: str, ground_truth: str) -> bool:
    """Simple answer matching - exact match ignoring case and whitespace"""
    if predicted is None or ground_truth is None:
        return False

    # Clean and normalize
    pred_clean = str(predicted).strip().lower()
    gt_clean = str(ground_truth).strip().lower()

    # Try exact match
    if pred_clean == gt_clean:
        return True

    # Try numeric match
    try:
        pred_num = float(pred_clean)
        gt_num = float(gt_clean)
        return abs(pred_num - gt_num) < 1e-6
    except:
        pass

    return False


def print_section_header(title: str, char: str = "=", width: int = 80):
    """Print a formatted section header"""
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_subsection(title: str, char: str = "-", width: int = 80):
    """Print a formatted subsection"""
    print()
    print(f"{title}")
    print(char * width)


def analyze_trace_confidences(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze confidence patterns in traces"""
    stats = {
        'mean_confidences': [],
        'min_confidences': [],
        'max_confidences': [],
        'trace_lengths': []
    }

    for trace in traces:
        if 'confs' in trace and trace['confs']:
            confs = trace['confs']
            stats['mean_confidences'].append(np.mean(confs))
            stats['min_confidences'].append(np.min(confs))
            stats['max_confidences'].append(np.max(confs))
            stats['trace_lengths'].append(len(confs))

    if stats['mean_confidences']:
        return {
            'avg_mean_conf': np.mean(stats['mean_confidences']),
            'avg_min_conf': np.mean(stats['min_confidences']),
            'avg_max_conf': np.mean(stats['max_confidences']),
            'avg_trace_length': np.mean(stats['trace_lengths']),
            'std_trace_length': np.std(stats['trace_lengths'])
        }

    return {}


def calculate_confidence_accuracy_correlation(traces: List[Dict[str, Any]], ground_truth: str) -> Dict[str, Any]:
    """
    Calculate correlation between various confidence metrics and accuracy

    Returns correlation coefficients for:
    - Mean confidence vs correctness
    - Min confidence vs correctness
    - Max confidence vs correctness
    - Tail confidence (last 512 tokens) vs correctness
    - Bottom 10% confidence vs correctness
    """
    if not traces:
        return {}

    # Collect data for correlation
    mean_confs = []
    min_confs = []
    max_confs = []
    tail_confs = []
    bottom_10_confs = []
    correctness = []

    for trace in traces:
        # Check if trace is correct
        answer = trace.get('extracted_answer')
        if answer is None:
            continue

        is_correct = 1 if simple_answer_match(answer, ground_truth) else 0
        correctness.append(is_correct)

        # Get confidence metrics
        if 'confs' in trace and trace['confs']:
            confs = trace['confs']

            # Mean confidence
            mean_confs.append(np.mean(confs))

            # Min/Max confidence
            min_confs.append(np.min(confs))
            max_confs.append(np.max(confs))

            # Tail confidence (last 512 tokens or all if shorter)
            tail_length = min(512, len(confs))
            tail_confs.append(np.mean(confs[-tail_length:]))

            # Bottom 10% confidence
            if len(confs) >= 10:
                bottom_k = max(1, len(confs) // 10)
                bottom_confs = np.partition(confs, bottom_k-1)[:bottom_k]
                bottom_10_confs.append(np.mean(bottom_confs))
            else:
                bottom_10_confs.append(np.min(confs))

    # Calculate correlations
    results = {}

    if len(correctness) >= 2:  # Need at least 2 points for correlation
        try:
            # Pearson correlation
            if len(set(correctness)) > 1:  # Need variance in correctness
                results['mean_conf_correlation'] = np.corrcoef(mean_confs, correctness)[0, 1]
                results['min_conf_correlation'] = np.corrcoef(min_confs, correctness)[0, 1]
                results['max_conf_correlation'] = np.corrcoef(max_confs, correctness)[0, 1]
                results['tail_conf_correlation'] = np.corrcoef(tail_confs, correctness)[0, 1]
                results['bottom_10_conf_correlation'] = np.corrcoef(bottom_10_confs, correctness)[0, 1]

                # Add mean confidence for correct vs incorrect
                correct_mean_confs = [mean_confs[i] for i, c in enumerate(correctness) if c == 1]
                incorrect_mean_confs = [mean_confs[i] for i, c in enumerate(correctness) if c == 0]

                if correct_mean_confs:
                    results['correct_avg_conf'] = np.mean(correct_mean_confs)
                if incorrect_mean_confs:
                    results['incorrect_avg_conf'] = np.mean(incorrect_mean_confs)

                # Confidence difference
                if correct_mean_confs and incorrect_mean_confs:
                    results['conf_difference'] = np.mean(correct_mean_confs) - np.mean(incorrect_mean_confs)
            else:
                results['note'] = 'All traces have same correctness - cannot compute correlation'
        except Exception as e:
            results['error'] = f'Error computing correlation: {str(e)}'
    else:
        results['note'] = 'Insufficient data for correlation (need at least 2 traces)'

    results['total_traces'] = len(correctness)
    results['correct_traces'] = sum(correctness)

    return results


def evaluate_traces_by_depth(traces: List[Dict[str, Any]], ground_truth: str) -> Dict[int, Dict]:
    """Evaluate traces grouped by depth"""
    depth_stats = {}

    for trace in traces:
        depth = trace.get('depth', 0)
        if depth not in depth_stats:
            depth_stats[depth] = {
                'total': 0,
                'correct': 0,
                'answers': [],
                'confidences': []
            }

        depth_stats[depth]['total'] += 1

        # Check correctness
        answer = trace.get('extracted_answer')
        if answer:
            depth_stats[depth]['answers'].append(answer)
            if simple_answer_match(answer, ground_truth):
                depth_stats[depth]['correct'] += 1

        # Track confidence
        if 'confs' in trace and trace['confs']:
            depth_stats[depth]['confidences'].append(np.mean(trace['confs']))

    # Calculate accuracy per depth
    for depth in depth_stats:
        total = depth_stats[depth]['total']
        correct = depth_stats[depth]['correct']
        depth_stats[depth]['accuracy'] = (correct / total * 100) if total > 0 else 0

    return depth_stats


def print_detailed_results(result, ground_truth: str, logger):
    """Print comprehensive results with evaluation"""

    print_section_header("DETAILED RESULTS")

    # Basic stats
    print_subsection("Generation Statistics")
    print(f"  Total traces generated: {result.total_traces_count}")
    print(f"  Total tokens: {result.total_tokens:,}")
    print(f"  Average tokens per trace: {result.avg_tokens_per_trace:.1f}")
    print(f"  Generation time: {result.generation_time:.2f}s")
    print(f"  Total time: {result.total_time:.2f}s")

    # Traces by depth
    print_subsection("Traces by Depth")
    depth_counts = {}
    for trace in result.all_traces:
        depth = trace.get('depth', 0)
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    for depth in sorted(depth_counts.keys()):
        count = depth_counts[depth]
        label = "Initial" if depth == 0 else f"Branched (depth {depth})"
        print(f"  {label}: {count} traces")

    # Branch details
    branch_traces = [t for t in result.all_traces if t.get('depth', 0) > 0]
    if branch_traces:
        print_subsection("Branch Details")
        print(f"  Number of branches: {len(branch_traces)}")

        # Prefix statistics
        prefix_lengths = [t.get('prefix_length', 0) for t in branch_traces if 'prefix_length' in t]
        if prefix_lengths:
            print(f"  Average prefix length: {np.mean(prefix_lengths):.0f} tokens")
            print(f"  Min prefix length: {min(prefix_lengths)} tokens")
            print(f"  Max prefix length: {max(prefix_lengths)} tokens")

            total_prefix = sum(prefix_lengths)
            print(f"  Total tokens saved via prefix caching: ~{total_prefix:,}")

        # Branch points
        branch_points = [t.get('branch_point', 0) for t in branch_traces if 'branch_point' in t]
        if branch_points:
            print(f"  Average branch point: {np.mean(branch_points):.0f} tokens")
            print(f"  Earliest branch: {min(branch_points)} tokens")
            print(f"  Latest branch: {max(branch_points)} tokens")

        # Parent distribution
        parents = {}
        for trace in branch_traces:
            parent_id = trace.get('parent_id', 'unknown')
            parents[parent_id] = parents.get(parent_id, 0) + 1

        print(f"  Branches per parent:")
        for parent_id, count in sorted(parents.items()):
            print(f"    {parent_id}: {count} branch(es)")

    # Confidence analysis
    print_subsection("Confidence Analysis")
    conf_stats = analyze_trace_confidences(result.all_traces)
    if conf_stats:
        print(f"  Average mean confidence: {conf_stats['avg_mean_conf']:.3f}")
        print(f"  Average min confidence: {conf_stats['avg_min_conf']:.3f}")
        print(f"  Average max confidence: {conf_stats['avg_max_conf']:.3f}")
        print(f"  Average trace length: {conf_stats['avg_trace_length']:.0f} tokens")

    # Branching stats
    if result.branching_stats:
        print_subsection("Branching Statistics")
        for key, value in result.branching_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

    # EVALUATION
    print_section_header("EVALUATION RESULTS")
    print(f"Ground truth answer: {ground_truth}")
    print()

    # Evaluate by depth
    depth_eval = evaluate_traces_by_depth(result.all_traces, ground_truth)

    print_subsection("Accuracy by Depth")
    for depth in sorted(depth_eval.keys()):
        stats = depth_eval[depth]
        label = "Initial traces" if depth == 0 else f"Depth {depth} branches"
        print(f"  {label}:")
        print(f"    Total: {stats['total']}")
        print(f"    Correct: {stats['correct']}")
        print(f"    Accuracy: {stats['accuracy']:.1f}%")
        if stats['confidences']:
            print(f"    Avg confidence: {np.mean(stats['confidences']):.3f}")

    # Overall accuracy
    total_traces = len(result.all_traces)
    total_correct = sum(
        1 for trace in result.all_traces
        if trace.get('extracted_answer') and simple_answer_match(trace['extracted_answer'], ground_truth)
    )
    overall_accuracy = (total_correct / total_traces * 100) if total_traces > 0 else 0

    print_subsection("Overall Accuracy")
    print(f"  Correct traces: {total_correct}/{total_traces}")
    print(f"  Overall accuracy: {overall_accuracy:.1f}%")

    # Confidence-Accuracy Correlation
    print_subsection("Confidence-Accuracy Correlation")
    corr_results = calculate_confidence_accuracy_correlation(result.all_traces, ground_truth)

    if 'note' in corr_results:
        print(f"  {corr_results['note']}")
    elif 'error' in corr_results:
        print(f"  Error: {corr_results['error']}")
    else:
        print(f"  Traces analyzed: {corr_results['total_traces']} ({corr_results['correct_traces']} correct)")
        print()
        print(f"  Correlation coefficients (Pearson r):")
        print(f"    Mean confidence:       {corr_results.get('mean_conf_correlation', 0):.4f}")
        print(f"    Tail confidence:       {corr_results.get('tail_conf_correlation', 0):.4f}")
        print(f"    Bottom 10% confidence: {corr_results.get('bottom_10_conf_correlation', 0):.4f}")
        print(f"    Min confidence:        {corr_results.get('min_conf_correlation', 0):.4f}")
        print(f"    Max confidence:        {corr_results.get('max_conf_correlation', 0):.4f}")

        if 'correct_avg_conf' in corr_results and 'incorrect_avg_conf' in corr_results:
            print()
            print(f"  Average confidence by correctness:")
            print(f"    Correct traces:   {corr_results['correct_avg_conf']:.4f}")
            print(f"    Incorrect traces: {corr_results['incorrect_avg_conf']:.4f}")
            print(f"    Difference:       {corr_results['conf_difference']:+.4f}")

            if corr_results['conf_difference'] > 0:
                print(f"\n  ✓ Correct traces have higher confidence on average")
            elif corr_results['conf_difference'] < 0:
                print(f"\n  ⚠ Incorrect traces have higher confidence on average (overconfident errors)")
            else:
                print(f"\n  ≈ Similar confidence for correct and incorrect traces")

        # Interpretation
        print()
        print(f"  Interpretation:")
        mean_corr = corr_results.get('mean_conf_correlation', 0)
        if abs(mean_corr) < 0.3:
            print(f"    Weak correlation between confidence and accuracy")
        elif abs(mean_corr) < 0.7:
            print(f"    Moderate correlation between confidence and accuracy")
        else:
            print(f"    Strong correlation between confidence and accuracy")

        if mean_corr > 0:
            print(f"    Higher confidence → higher likelihood of correctness")
        elif mean_corr < 0:
            print(f"    ⚠ Higher confidence → lower likelihood of correctness (problematic!)")

    # Voting results
    if result.voting_results:
        print_subsection("Voting Results")
        print(f"  {'Method':<30} {'Answer':<20} {'Correct':<10}")
        print(f"  {'-'*60}")

        for method, vote_result in result.voting_results.items():
            if vote_result and vote_result.get('answer'):
                answer = str(vote_result['answer'])[:18]
                correct = simple_answer_match(vote_result['answer'], ground_truth)
                correct_str = "✓ YES" if correct else "✗ NO"
                print(f"  {method:<30} {answer:<20} {correct_str:<10}")

        # Find best voting method
        correct_methods = [
            method for method, result in result.voting_results.items()
            if result and simple_answer_match(result.get('answer'), ground_truth)
        ]

        if correct_methods:
            print(f"\n  ✓ {len(correct_methods)}/{len(result.voting_results)} voting methods got correct answer")
        else:
            print(f"\n  ✗ No voting methods got the correct answer")

    # Answer distribution
    print_subsection("Answer Distribution")
    answer_counts = {}
    for trace in result.all_traces:
        answer = trace.get('extracted_answer')
        if answer:
            answer_str = str(answer)
            answer_counts[answer_str] = answer_counts.get(answer_str, 0) + 1

    for answer, count in sorted(answer_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_traces * 100)
        correct_mark = "✓" if simple_answer_match(answer, ground_truth) else " "
        print(f"  {correct_mark} {answer}: {count} traces ({percentage:.1f}%)")

    # Summary verdict
    print_section_header("SUMMARY")

    # Key metrics
    print(f"Question: {result.config.get('question', 'N/A')[:100]}...")
    print(f"Ground truth: {ground_truth}")
    print()
    print(f"  Generated: {result.total_traces_count} traces ({result.total_tokens:,} tokens)")
    print(f"  Time: {result.total_time:.1f}s")
    print(f"  Overall accuracy: {overall_accuracy:.1f}%")

    # Branch improvement analysis
    if len(depth_eval) > 1:
        initial_acc = depth_eval.get(0, {}).get('accuracy', 0)
        branch_accs = [stats['accuracy'] for d, stats in depth_eval.items() if d > 0]
        if branch_accs:
            avg_branch_acc = np.mean(branch_accs)
            improvement = avg_branch_acc - initial_acc
            print(f"  Initial trace accuracy: {initial_acc:.1f}%")
            print(f"  Average branch accuracy: {avg_branch_acc:.1f}%")
            print(f"  Improvement: {improvement:+.1f}%")

            if improvement > 5:
                print(f"\n  ✓ Branching provided significant improvement!")
            elif improvement > 0:
                print(f"\n  ✓ Branching provided modest improvement")
            else:
                print(f"\n  ⚠ Branching did not improve over initial traces")

    # Voting success
    if correct_methods:
        print(f"\n  ✓ Voting successfully found correct answer ({len(correct_methods)} methods)")
    else:
        print(f"\n  ✗ Voting did not find correct answer")

    print()


def run_branching_evaluation(
    question: str,
    ground_truth: str,
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    initial_branches: int = 2,
    max_total_branches: int = 6,
    confidence_threshold: float = 1.5,
    window_size: int = 128,
    max_tokens: int = 4000,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Run branching experiment with evaluation

    Args:
        question: Question to process
        ground_truth: Correct answer for evaluation
        model: Model name/path
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

    print_section_header(f"BRANCHING EXPERIMENT WITH EVALUATION", "=", 80)

    print_subsection("Configuration")
    print(f"  Model: {model}")
    print(f"  Question: {question}")
    print(f"  Ground truth: {ground_truth}")
    print(f"  Initial branches: {initial_branches}")
    print(f"  Max total branches: {max_total_branches}")
    print(f"  Confidence threshold: {confidence_threshold}")
    print(f"  Window size: {window_size}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

    # Initialize the branching LLM
    print_subsection("Initialization")
    print("  Initializing BranchingDeepThinkLLM...")

    gpu_count = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))

    branching_llm = BranchingDeepThinkLLM(
        model=model,
        tensor_parallel_size=gpu_count,
        enable_prefix_caching=True,
        trust_remote_code=True
    )

    print("  ✓ Model initialized")

    # Prepare prompt
    print("  Preparing prompt...")
    messages = [{"role": "user", "content": question}]
    prompt = branching_llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("  ✓ Prompt ready")

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=max_tokens,
        logprobs=20,
    )

    # Run the experiment
    print_section_header("GENERATION")
    print("Starting branching generation...")
    print()

    result = branching_llm.branching_deepthink(
        prompt=prompt,
        initial_branches=initial_branches,
        max_total_branches=max_total_branches,
        confidence_threshold=confidence_threshold,
        window_size=window_size,
        sampling_params=sampling_params
    )

    # Add metadata
    result_data = {
        'question': question,
        'ground_truth': ground_truth,
        'result': result,
        'config': {
            'model': model,
            'initial_branches': initial_branches,
            'max_total_branches': max_total_branches,
            'confidence_threshold': confidence_threshold,
            'window_size': window_size,
            'max_tokens': max_tokens
        },
        'timestamp': datetime.now().isoformat()
    }

    # Print detailed results
    print_detailed_results(result, ground_truth, logger)

    return result_data


def save_results(result_data: Dict, output_dir: str = "outputs", logger=None):
    """Save experiment results"""
    if logger is None:
        logger = logging.getLogger(__name__)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save pickle
    pkl_path = os.path.join(output_dir, f"branching_eval_{timestamp}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(result_data, f)
    logger.info(f"Results saved to: {pkl_path}")

    # Save JSON summary (excluding traces for readability)
    json_data = {
        'question': result_data['question'],
        'ground_truth': result_data['ground_truth'],
        'config': result_data['config'],
        'timestamp': result_data['timestamp'],
        'total_traces': result_data['result'].total_traces_count,
        'total_tokens': result_data['result'].total_tokens,
        'voting_results': result_data['result'].voting_results
    }

    json_path = os.path.join(output_dir, f"branching_eval_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    logger.info(f"Summary saved to: {json_path}")

    return pkl_path, json_path


def main():
    parser = argparse.ArgumentParser(
        description='Run branching experiment with evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment configuration
    parser.add_argument('--question', type=str,
                       default="What is 15% of 240?",
                       help='Question to process')
    parser.add_argument('--ground_truth', type=str,
                       default="36",
                       help='Ground truth answer for evaluation')
    parser.add_argument('--model', type=str,
                       default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                       help='Model to use')

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
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for log files')

    # GPU configuration
    parser.add_argument('--gpus', type=str, default=None,
                       help='Comma-separated GPU IDs (e.g., "0,1,2"). If not set, uses CUDA_VISIBLE_DEVICES env var or all GPUs')

    args = parser.parse_args()

    # Set GPU visibility
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # Setup logging
    logger = setup_logging(args.log_dir)

    try:
        # Run experiment
        result_data = run_branching_evaluation(
            question=args.question,
            ground_truth=args.ground_truth,
            model=args.model,
            initial_branches=args.initial_branches,
            max_total_branches=args.max_total_branches,
            confidence_threshold=args.confidence_threshold,
            window_size=args.window_size,
            max_tokens=args.max_tokens,
            logger=logger
        )

        # Save results
        pkl_path, json_path = save_results(result_data, args.output_dir, logger)

        print_section_header("EXPERIMENT COMPLETE")
        print(f"  Results saved to: {pkl_path}")
        print(f"  Summary saved to: {json_path}")
        print(f"  Logs saved to: {args.log_dir}/")
        print()

    except Exception as e:
        logger.error(f"Error during experiment: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
