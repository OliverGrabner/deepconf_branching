"""
Full AIME 2025 Evaluation Script
Evaluates on complete AIME 2025-I and AIME 2025-II datasets (30 problems total)
Uses DeepSeek-R1-Distill-Qwen-32B model with optimized settings

Hardware Configuration:
    - 4x NVIDIA RTX 5000 Ada (32GB VRAM each)
    - Total VRAM: 128GB
    - Model: DeepSeek-R1-Distill-Qwen-32B (~32B parameters)
    - Tensor Parallelism: 4 GPUs (32B model uses all 4)

Dataset:
    - Source: opencompass/AIME2025 from HuggingFace
    - AIME 2025-I: 15 problems
    - AIME 2025-II: 15 problems
    - Total: 30 problems

Usage:
    # Standard mode on full AIME 2025
    python run_aime25_full.py --mode standard

    # Branching mode on full AIME 2025
    python run_aime25_full.py --mode branching

    # Run only AIME-I subset
    python run_aime25_full.py --mode standard --subset AIME2025-I

    # Resume from checkpoint
    python run_aime25_full.py --mode standard --resume results/aime25_full/checkpoint.pkl

    # Dry run (first 3 problems only)
    python run_aime25_full.py --mode standard --dry_run
"""
import re
import os
import sys
import json
import pickle
import argparse
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import SamplingParams
from deepconf import DeepThinkLLM
from deepconf.branching_wrapper import BranchingDeepThinkLLM

# Import robust answer extraction
from utils_robust import extract_answer_robust, check_answer_equality


########################################
# 1. Dataset Loading
########################################

def load_aime25_dataset(subset: Optional[str] = None) -> List[Dict]:
    """
    Load AIME 2025 dataset from HuggingFace

    Args:
        subset: Optional subset name ('AIME2025-I' or 'AIME2025-II')

    Returns:
        List of problem dictionaries with 'problem_id', 'question', 'expected_answer'
    """
    print("Loading AIME 2025 dataset from HuggingFace...")

    # Load from opencompass/AIME2025 (uses 'test' split, not 'train')
    if subset:
        dataset = load_dataset("opencompass/AIME2025", name=subset, split="test")
        print(f"  Loaded {len(dataset)} problems from {subset}")
    else:
        # Load both subsets
        dataset_i = load_dataset("opencompass/AIME2025", name="AIME2025-I", split="test")
        dataset_ii = load_dataset("opencompass/AIME2025", name="AIME2025-II", split="test")
        dataset = list(dataset_i) + list(dataset_ii)
        print(f"  Loaded {len(dataset_i)} problems from AIME2025-I")
        print(f"  Loaded {len(dataset_ii)} problems from AIME2025-II")
        print(f"  Total: {len(dataset)} problems")

    # Convert to standard format
    problems = []
    for idx, item in enumerate(dataset):
        # Determine problem ID
        if idx < 15:
            problem_id = f"I-{idx + 1}"
            source = "AIME2025-I"
        else:
            problem_id = f"II-{idx - 14}"
            source = "AIME2025-II"

        # Format question with boxed instruction
        question = item['question'].strip()
        if not question.endswith("\\boxed"):
            question += "\nMark your solution with \\boxed\nAnswer:"

        problems.append({
            'problem_id': problem_id,
            'source': source,
            'question': question,
            'expected_answer': str(item['answer']).strip()
        })

    return problems


########################################
# 2. Answer extraction
########################################

def extract_boxed_answer(text: str) -> str:
    """Extract answer from various formats including \\boxed{...}"""
    import re

    if not text:
        return ""

    # Try standard boxed format first (with or without backslash)
    patterns = [
        r'\\boxed\{([^}]+)\}',  # \boxed{answer}
        r'boxed\{([^}]+)\}',     # boxed{answer} without backslash
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Return the last boxed answer found
            answer = matches[-1].strip()
            # Clean up any LaTeX artifacts
            answer = answer.replace('$', '').strip()
            return answer

    # Try to find answer at the end of the text
    lines = text.strip().split('\n')

    # Check last few lines for answer patterns
    for line in reversed(lines[-10:]):  # Check last 10 lines
        line = line.strip()

        # Pattern: "The answer is X" or "Therefore, X" or "Thus, X"
        answer_patterns = [
            r'(?:The answer is|Therefore,?|Thus,?|So,?|Hence,?)\s*[:=]?\s*(\d+)',
            r'Final answer\s*[:=]\s*(\d+)',
            r'^Answer\s*[:=]\s*(\d+)',  # "Answer: 123"
            r'^=\s*(\d+)$',  # Just "= 123" on its own line
            r'^\$?(\d+)\$?$',  # Just a number on its own line (possibly in math mode)
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).strip()

    # Look for the last number in the text (risky but might help)
    numbers = re.findall(r'\b(\d{1,3})\b', text[-200:])  # AIME answers are 0-999
    if numbers and len(lines) > 0:
        # Only use this if the last line seems to be presenting an answer
        last_line_lower = lines[-1].lower()
        if any(word in last_line_lower for word in ['answer', 'therefore', 'thus', 'so', '=']):
            return numbers[-1]

    return ""


def simple_answer_match(predicted: str, ground_truth: str, dataset_type: str = "aime") -> bool:
    """Answer matching using robust comparison"""
    return check_answer_equality(predicted, ground_truth, dataset_type)


########################################
# 3. Correlation calculation
########################################

def calculate_confidence_correlation(traces: List[Dict], ground_truth: str) -> Dict:
    """Calculate confidence-accuracy correlation"""
    if not traces:
        return {}

    mean_confs = []
    tail_confs = []
    correctness = []

    for trace in traces:
        answer = trace.get('extracted_answer')
        if answer is None:
            continue

        is_correct = 1 if simple_answer_match(answer, ground_truth, 'aime') else 0
        correctness.append(is_correct)

        if 'confs' in trace and trace['confs']:
            confs = trace['confs']
            mean_confs.append(np.mean(confs))

            tail_length = min(512, len(confs))
            tail_confs.append(np.mean(confs[-tail_length:]))

    results = {
        'total_traces': len(correctness),
        'correct_traces': sum(correctness)
    }

    if len(correctness) >= 2 and len(set(correctness)) > 1:
        try:
            results['mean_conf_corr'] = np.corrcoef(mean_confs, correctness)[0, 1]
            results['tail_conf_corr'] = np.corrcoef(tail_confs, correctness)[0, 1]

            correct_confs = [mean_confs[i] for i, c in enumerate(correctness) if c == 1]
            incorrect_confs = [mean_confs[i] for i, c in enumerate(correctness) if c == 0]

            if correct_confs:
                results['correct_avg_conf'] = np.mean(correct_confs)
            if incorrect_confs:
                results['incorrect_avg_conf'] = np.mean(incorrect_confs)
            if correct_confs and incorrect_confs:
                results['conf_difference'] = np.mean(correct_confs) - np.mean(incorrect_confs)
        except:
            pass

    return results


########################################
# 4. Main evaluation functions
########################################

def run_standard_eval(problem: Dict, model: str, budget: int, tensor_parallel_size: int = 1) -> Dict:
    """Run standard DeepConf evaluation (no branching)"""
    print(f"\n{'='*80}")
    print(f"STANDARD MODE: {problem['problem_id']} ({problem['source']})")
    print(f"{'='*80}")

    # Initialize with optimized settings for 32B model
    deep_llm = DeepThinkLLM(
        model=model,
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.85,  # 32B needs more headroom (~4.8GB per GPU)
        max_num_seqs=32,               # Reduce for larger model
        max_model_len=4096             # Match budget of 4000 tokens
    )

    # Prepare prompt
    messages = [{"role": "user", "content": problem['question']}]
    prompt = deep_llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Sampling params
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=4000,
        logprobs=20,
    )

    # Run
    result = deep_llm.deepthink(
        prompt=prompt,
        mode="offline",
        budget=budget,
        window_size=2048,
        sampling_params=sampling_params,
        compute_multiple_voting=True
    )

    # Extract answers using robust method
    for trace in result.all_traces:
        if 'text' in trace:
            trace['extracted_answer'] = extract_answer_robust(trace['text'], dataset_type='aime')

    # Evaluate
    correct_count = sum(
        1 for t in result.all_traces
        if simple_answer_match(t.get('extracted_answer'), problem['expected_answer'], 'aime')
    )

    accuracy = (correct_count / len(result.all_traces) * 100) if result.all_traces else 0

    # Correlation
    corr_stats = calculate_confidence_correlation(result.all_traces, problem['expected_answer'])

    # Print results
    print(f"\nResults:")
    print(f"  Traces: {result.total_traces_count}")
    print(f"  Correct: {correct_count}/{result.total_traces_count}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Tokens: {result.total_tokens:,}")

    if 'mean_conf_corr' in corr_stats:
        print(f"\nCorrelation:")
        print(f"  Mean conf correlation: {corr_stats['mean_conf_corr']:.4f}")
        print(f"  Tail conf correlation: {corr_stats.get('tail_conf_corr', 0):.4f}")
        if 'conf_difference' in corr_stats:
            print(f"  Conf difference: {corr_stats['conf_difference']:+.4f}")

    return {
        'problem_id': problem['problem_id'],
        'source': problem['source'],
        'mode': 'standard',
        'accuracy': accuracy,
        'correct': correct_count,
        'total': result.total_traces_count,
        'tokens': result.total_tokens,
        'correlation': corr_stats,
        'expected_answer': problem['expected_answer'],
        'result': result.to_dict()
    }


def run_branching_eval(problem: Dict, model: str, initial: int, max_total: int, tensor_parallel_size: int = 1) -> Dict:
    """Run branching DeepConf evaluation"""
    print(f"\n{'='*80}")
    print(f"BRANCHING MODE: {problem['problem_id']} ({problem['source']})")
    print(f"{'='*80}")

    # Initialize with optimized settings for 32B model
    branching_llm = BranchingDeepThinkLLM(
        model=model,
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.85,  # 32B needs more headroom (~4.8GB per GPU)
        max_num_seqs=32,               # Reduce for larger model
        max_model_len=4096             # Match budget of 4000 tokens
    )

    # Prepare prompt
    messages = [{"role": "user", "content": problem['question']}]
    prompt = branching_llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Sampling params
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=4000,
        logprobs=20,
    )

    # Run
    result = branching_llm.branching_deepthink(
        prompt=prompt,
        initial_branches=initial,
        max_total_branches=max_total,
        confidence_threshold=1.5,
        window_size=128,
        sampling_params=sampling_params
    )

    # Extract answers using robust method
    for trace in result.all_traces:
        if 'text' in trace:
            trace['extracted_answer'] = extract_answer_robust(trace['text'], dataset_type='aime')

    # Evaluate by depth
    depth_stats = {}
    for trace in result.all_traces:
        depth = trace.get('depth', 0)
        if depth not in depth_stats:
            depth_stats[depth] = {'total': 0, 'correct': 0}

        depth_stats[depth]['total'] += 1
        if simple_answer_match(trace.get('extracted_answer'), problem['expected_answer'], 'aime'):
            depth_stats[depth]['correct'] += 1

    # Overall
    correct_count = sum(stats['correct'] for stats in depth_stats.values())
    total_count = sum(stats['total'] for stats in depth_stats.values())
    accuracy = (correct_count / total_count * 100) if total_count else 0

    # Correlation
    corr_stats = calculate_confidence_correlation(result.all_traces, problem['expected_answer'])

    # Print results
    print(f"\nResults:")
    print(f"  Total traces: {result.total_traces_count}")
    print(f"  Correct: {correct_count}/{total_count}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Tokens: {result.total_tokens:,}")

    print(f"\nBy depth:")
    for depth in sorted(depth_stats.keys()):
        stats = depth_stats[depth]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] else 0
        print(f"  Depth {depth}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

    if 'mean_conf_corr' in corr_stats:
        print(f"\nCorrelation:")
        print(f"  Mean conf correlation: {corr_stats['mean_conf_corr']:.4f}")
        print(f"  Tail conf correlation: {corr_stats.get('tail_conf_corr', 0):.4f}")
        if 'conf_difference' in corr_stats:
            print(f"  Conf difference: {corr_stats['conf_difference']:+.4f}")

    # Branch stats
    branch_traces = [t for t in result.all_traces if t.get('depth', 0) > 0]
    if branch_traces:
        print(f"\nBranching:")
        print(f"  Branches: {len(branch_traces)}")
        prefix_lengths = [t.get('prefix_length', 0) for t in branch_traces if 'prefix_length' in t]
        if prefix_lengths:
            print(f"  Avg prefix: {np.mean(prefix_lengths):.0f} tokens")
            print(f"  Saved: ~{sum(prefix_lengths):,} tokens")

    return {
        'problem_id': problem['problem_id'],
        'source': problem['source'],
        'mode': 'branching',
        'accuracy': accuracy,
        'correct': correct_count,
        'total': total_count,
        'tokens': result.total_tokens,
        'depth_stats': depth_stats,
        'correlation': corr_stats,
        'expected_answer': problem['expected_answer'],
        'branching_stats': result.branching_stats if hasattr(result, 'branching_stats') else {},
        'result': result.to_dict()
    }


########################################
# 5. Checkpoint management
########################################

def save_checkpoint(output_dir: str, results: List[Dict], config: Dict):
    """Save intermediate checkpoint"""
    checkpoint_file = os.path.join(output_dir, 'checkpoint.pkl')
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({
            'config': config,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f)
    print(f"  Checkpoint saved: {checkpoint_file}")


def load_checkpoint(checkpoint_path: str) -> Optional[Dict]:
    """Load checkpoint if exists"""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  WARNING: Could not load checkpoint: {e}")
            return None
    return None


########################################
# 6. Visualization functions
########################################

def plot_initial_traces_confidence(result_dict: Dict, problem_id: str, ground_truth: str,
                                  save_dir: str = None) -> Optional[str]:
    """
    Plot confidence over time for ONLY initial traces (depth=0)

    Args:
        result_dict: Result dictionary from evaluation
        problem_id: Problem identifier
        ground_truth: Expected answer for coloring correct/incorrect
        save_dir: Directory to save plot

    Returns:
        Path to saved plot or None
    """
    if 'result' not in result_dict:
        return None

    result = result_dict['result']
    if not hasattr(result, 'all_traces') or not result.all_traces:
        return None

    # Filter for initial traces only (depth=0)
    initial_traces = [t for t in result.all_traces if t.get('depth', 0) == 0]

    if not initial_traces:
        print(f"  No initial traces found for {problem_id}")
        return None

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each initial trace
    for trace in initial_traces:
        if 'confs' not in trace or not trace['confs']:
            continue

        confs = trace['confs']
        positions = np.arange(len(confs))

        # Determine if trace is correct
        answer = trace.get('extracted_answer', '')
        is_correct = str(answer).strip() == str(ground_truth).strip()

        # Set style based on correctness
        color = 'green' if is_correct else 'red'
        alpha = 0.7
        linewidth = 2 if is_correct else 1.5
        linestyle = '-'  # Solid for all initial traces

        label = f"{trace['trace_id']} ({'✓' if is_correct else '✗'})"
        ax.plot(positions, confs, color=color, alpha=alpha, linewidth=linewidth,
               linestyle=linestyle, label=label)

    # Add average confidence line
    all_confs = []
    for trace in initial_traces:
        if 'confs' in trace:
            all_confs.extend(trace['confs'])

    if all_confs:
        avg_conf = np.mean(all_confs)
        ax.axhline(y=avg_conf, color='blue', linestyle='--', alpha=0.5,
                  label=f'Avg confidence: {avg_conf:.3f}')

    # Styling
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_title(f'Initial Traces Confidence - {problem_id}\n(Ground Truth: {ground_truth})',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    # Save plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{problem_id}_initial_only.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path

    plt.close(fig)
    return None


def plot_all_traces_with_branches(result_dict: Dict, problem_id: str, ground_truth: str,
                                 save_dir: str = None) -> Optional[str]:
    """
    Plot ALL traces (initial + branches) on same graph with different styles

    Args:
        result_dict: Result dictionary from evaluation
        problem_id: Problem identifier
        ground_truth: Expected answer for coloring correct/incorrect
        save_dir: Directory to save plot

    Returns:
        Path to saved plot or None
    """
    if 'result' not in result_dict:
        return None

    result = result_dict['result']
    if not hasattr(result, 'all_traces') or not result.all_traces:
        return None

    fig, ax = plt.subplots(figsize=(16, 10))

    # Define line styles based on depth
    # NOTE: Current implementation only goes to depth 1 (no multi-level branching yet)
    depth_styles = {
        0: '-',   # Solid for initial traces
        1: '--',  # Dashed for branch traces (depth 1 is max currently)
    }

    # Color maps for correct/incorrect (only need 2 levels)
    correct_colors = ['darkgreen', 'green']  # Dark for initial, lighter for branches
    incorrect_colors = ['darkred', 'red']    # Dark for initial, lighter for branches

    # Track branch points for vertical lines
    branch_points = []

    # Group traces by depth for organized plotting
    traces_by_depth = {}
    for trace in result.all_traces:
        depth = trace.get('depth', 0)
        if depth not in traces_by_depth:
            traces_by_depth[depth] = []
        traces_by_depth[depth].append(trace)

    # Plot traces by depth (initial first, then branches)
    for depth in sorted(traces_by_depth.keys()):
        for i, trace in enumerate(traces_by_depth[depth]):
            if 'confs' not in trace or not trace['confs']:
                continue

            confs = trace['confs']

            # Handle branch traces - align with parent
            if depth > 0 and 'prefix_length' in trace:
                # For branches, start plotting from branch point
                prefix_len = trace['prefix_length']
                positions = np.arange(prefix_len, prefix_len + len(confs))

                # Record branch point
                if 'branch_point' in trace:
                    branch_points.append(trace['branch_point'])
            else:
                positions = np.arange(len(confs))

            # Determine correctness
            answer = trace.get('extracted_answer', '')
            is_correct = str(answer).strip() == str(ground_truth).strip()

            # Set style based on depth and correctness
            linestyle = depth_styles.get(depth, ':')
            if is_correct:
                color = correct_colors[min(depth, len(correct_colors)-1)]
                alpha = 0.8 - depth * 0.1
            else:
                color = incorrect_colors[min(depth, len(incorrect_colors)-1)]
                alpha = 0.6 - depth * 0.1

            linewidth = 2 - depth * 0.3

            # Create label
            depth_str = f"D{depth}"
            correct_str = '✓' if is_correct else '✗'
            parent_str = f" (from {trace.get('parent_id', 'N/A')})" if depth > 0 else ""
            label = f"{trace['trace_id']} [{depth_str}] {correct_str}{parent_str}"

            ax.plot(positions, confs, color=color, alpha=alpha, linewidth=linewidth,
                   linestyle=linestyle, label=label)

    # Add vertical lines at branch points
    for bp in set(branch_points):
        ax.axvline(x=bp, color='gray', linestyle=':', alpha=0.3)

    # Add confidence threshold if present
    if hasattr(result, 'config') and 'confidence_threshold' in result.config.get('branching_params', {}):
        threshold = result.config['branching_params']['confidence_threshold']
        ax.axhline(y=threshold, color='purple', linestyle='-.', alpha=0.5,
                  label=f'Branch threshold: {threshold}')

    # Styling
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_title(f'All Traces with Branching - {problem_id}\n(Ground Truth: {ground_truth})',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Create custom legend
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Correct', linestyle='-'),
        Line2D([0], [0], color='red', lw=2, label='Incorrect', linestyle='-'),
        Line2D([0], [0], color='black', lw=2, label='Initial (Depth 0)', linestyle='-'),
        Line2D([0], [0], color='black', lw=2, label='Branches (Depth 1)', linestyle='--'),
    ]

    # Add first legend for line types
    first_legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    ax.add_artist(first_legend)

    # Add second legend for traces (make it scrollable if too many)
    if len(result.all_traces) <= 15:
        ax.legend(loc='upper right', fontsize=8, ncol=2)
    else:
        # For many traces, just show summary
        summary_text = f"Total: {len(result.all_traces)} traces\n"
        summary_text += f"Initial: {len(traces_by_depth.get(0, []))}\n"
        summary_text += f"Branches: {sum(len(traces_by_depth.get(d, [])) for d in traces_by_depth if d > 0)}"
        ax.text(0.98, 0.98, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{problem_id}_all_traces.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path

    plt.close(fig)
    return None


########################################
# 7. Main runner
########################################

def main():
    # Configure GPU environment for 4x RTX 5000 Ada (32GB each, 128GB total)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    parser = argparse.ArgumentParser(description='AIME 2025 Full Evaluation')
    parser.add_argument('--mode', type=str, choices=['standard', 'branching'], required=True,
                       help='Evaluation mode')
    parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
                       help='Model to use')
    parser.add_argument('--budget', type=int, default=16,
                       help='Number of traces for standard mode')
    parser.add_argument('--initial_branches', type=int, default=4,
                       help='Initial branches for branching mode')
    parser.add_argument('--max_total_branches', type=int, default=16,
                       help='Max total branches for branching mode')
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                       help='Number of GPUs for tensor parallelism (default: 4 for 32B model)')
    parser.add_argument('--subset', type=str, choices=['AIME2025-I', 'AIME2025-II'], default=None,
                       help='Run only on specific subset')
    parser.add_argument('--output_dir', type=str, default='results/aime25_full',
                       help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint file')
    parser.add_argument('--dry_run', action='store_true',
                       help='Run on first 3 problems only (for testing)')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save confidence visualization plots for each problem')

    args = parser.parse_args()

    # Load dataset
    problems = load_aime25_dataset(subset=args.subset)

    if args.dry_run:
        print("\n*** DRY RUN MODE: Processing first 3 problems only ***\n")
        problems = problems[:3]

    # Check for resume
    completed_ids = set()
    results = []
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if checkpoint:
            results = checkpoint.get('results', [])
            completed_ids = {r['problem_id'] for r in results}
            print(f"\nResuming from checkpoint:")
            print(f"  Already completed: {len(completed_ids)} problems")
            print(f"  Remaining: {len(problems) - len(completed_ids)} problems\n")

    # Filter out already completed problems
    problems_to_run = [p for p in problems if p['problem_id'] not in completed_ids]

    print(f"\n{'='*80}")
    print(f"AIME 2025 FULL EVALUATION")
    print(f"{'='*80}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"Total problems: {len(problems)}")
    print(f"To process: {len(problems_to_run)}")
    if args.dry_run:
        print(f"DRY RUN: Limited to first 3 problems")
    print(f"{'='*80}\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run evaluations
    for i, problem in enumerate(problems_to_run, 1):
        print(f"\n{'#'*80}")
        print(f"# Problem {len(results) + 1}/{len(problems)}: {problem['problem_id']}")
        print(f"# Expected answer: {problem['expected_answer']}")
        print(f"{'#'*80}")

        try:
            if args.mode == 'standard':
                result = run_standard_eval(problem, args.model, args.budget, args.tensor_parallel_size)
            else:  # branching
                result = run_branching_eval(
                    problem, args.model,
                    args.initial_branches, args.max_total_branches,
                    args.tensor_parallel_size
                )

            results.append(result)

            # Save plots if requested (only for branching mode)
            if args.save_plots and args.mode == 'branching':
                plots_dir = os.path.join(args.output_dir, 'plots')
                print(f"  Generating plots...")

                # Plot initial traces only
                plot_path = plot_initial_traces_confidence(
                    result, problem['problem_id'], problem['expected_answer'], plots_dir
                )
                if plot_path:
                    print(f"    Saved: {plot_path}")

                # Plot all traces with branches
                plot_path = plot_all_traces_with_branches(
                    result, problem['problem_id'], problem['expected_answer'], plots_dir
                )
                if plot_path:
                    print(f"    Saved: {plot_path}")

            # Save checkpoint after each problem
            save_checkpoint(args.output_dir, results, vars(args))

        except Exception as e:
            print(f"\nERROR on {problem['problem_id']}: {e}")
            import traceback
            traceback.print_exc()

            # Save checkpoint even on error
            save_checkpoint(args.output_dir, results, vars(args))
            continue

    # Aggregate results
    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS")
    print(f"{'='*80}")

    total_correct = sum(r['correct'] for r in results)
    total_traces = sum(r['total'] for r in results)
    total_tokens = sum(r['tokens'] for r in results)

    overall_accuracy = (total_correct / total_traces * 100) if total_traces else 0

    print(f"\nOverall:")
    print(f"  Problems: {len(results)}")
    print(f"  Correct traces: {total_correct}/{total_traces}")
    print(f"  Accuracy: {overall_accuracy:.1f}%")
    print(f"  Total tokens: {total_tokens:,}")
    if len(results) > 0:
        print(f"  Avg tokens/problem: {total_tokens/len(results):,.0f}")

    # Per-subset breakdown
    subset_results = {}
    for r in results:
        source = r.get('source', 'unknown')
        if source not in subset_results:
            subset_results[source] = {'correct': 0, 'total': 0}
        subset_results[source]['correct'] += r['correct']
        subset_results[source]['total'] += r['total']

    print(f"\nPer-subset:")
    for source, stats in sorted(subset_results.items()):
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] else 0
        print(f"  {source}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

    print(f"\nPer-problem:")
    for r in results:
        print(f"  {r['problem_id']}: {r['correct']}/{r['total']} ({r['accuracy']:.1f}%)")

    # Correlation summary
    correlations = [r['correlation'].get('mean_conf_corr', 0) for r in results
                   if 'mean_conf_corr' in r['correlation']]
    if correlations:
        print(f"\nAverage correlation: {np.mean(correlations):.4f}")

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save pickle
    output_file = os.path.join(args.output_dir, f"{args.mode}_{timestamp}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump({
            'mode': args.mode,
            'config': vars(args),
            'results': results,
            'summary': {
                'overall_accuracy': overall_accuracy,
                'total_correct': total_correct,
                'total_traces': total_traces,
                'total_tokens': total_tokens,
                'avg_correlation': np.mean(correlations) if correlations else None,
                'subset_breakdown': subset_results
            }
        }, f)

    # Save JSON summary
    json_file = os.path.join(args.output_dir, f"{args.mode}_{timestamp}_summary.json")
    with open(json_file, 'w') as f:
        json.dump({
            'mode': args.mode,
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_traces': total_traces,
            'total_tokens': total_tokens,
            'avg_correlation': np.mean(correlations) if correlations else None,
            'subset_breakdown': {k: {'correct': v['correct'], 'total': v['total'],
                                     'accuracy': (v['correct']/v['total']*100) if v['total'] else 0}
                                for k, v in subset_results.items()},
            'per_problem': [
                {
                    'problem_id': r['problem_id'],
                    'source': r.get('source', 'unknown'),
                    'accuracy': r['accuracy'],
                    'correct': r['correct'],
                    'total': r['total'],
                    'tokens': r['tokens']
                }
                for r in results
            ]
        }, f, indent=2)

    print(f"\nFinal results saved to:")
    print(f"  {output_file}")
    print(f"  {json_file}")
    print()


if __name__ == "__main__":
    main()
