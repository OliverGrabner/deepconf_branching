"""
Manual AIME 2025 Evaluation Script
Run on 3 sample problems with both branching and standard modes
Shows detailed correlation analysis for each question

Hardware Configuration:
    - 4x NVIDIA RTX 5000 Ada (32GB VRAM each)
    - Total VRAM: 128GB
    - Model: DeepSeek-R1-Distill-Qwen-32B (~32B parameters)
    - Tensor Parallelism: 4 GPUs (default)

Usage:
    # Standard mode with default 32B model on 4 GPUs
    python run_aime25_manual.py --mode standard

    # Branching mode with default settings
    python run_aime25_manual.py --mode branching

    # Custom tensor parallelism (if using fewer GPUs)
    python run_aime25_manual.py --mode standard --tensor_parallel_size 2

    # Specific problems only
    python run_aime25_manual.py --mode branching --problem_ids I-1 I-2
"""
import re
import os
import sys
import json
import pickle
import argparse
from typing import List, Dict
from datetime import datetime

import numpy as np
from transformers import AutoTokenizer
from vllm import SamplingParams
from deepconf import DeepThinkLLM
from deepconf.branching_wrapper import BranchingDeepThinkLLM


########################################
# 1. AIME25 subset (3 problems)
########################################

AIME25_SUBSET: List[Dict] = [
    {
        "problem_id": "I-1",
        "expected_answer": "70",
        "question": (
            "Problem: On triangle ABC, points A, D, E, and B lie in that order on side AB "
            "with AD = 4, DE = 16, and EB = 8. Points A, F, G, and C lie in that order on "
            "side AC with AF = 13, FG = 52, and GC = 26. Let M be the reflection of D through F, "
            "and let N be the reflection of G through E. Quadrilateral DEGF has area 288. "
            "Find the area of heptagon AFNBCEM.\n"
            "Mark your solution with \\boxed\n"
            "Answer:"
        ),
    },
    {
        "problem_id": "I-2",
        "expected_answer": "588",
        "question": (
            "Problem: The 9 members of a baseball team went to an ice-cream parlor after "
            "their game. Each player had a single-scoop cone of chocolate, vanilla, or "
            "strawberry ice cream. At least one player chose each flavor, and the number of "
            "players who chose chocolate was greater than the number of players who chose "
            "vanilla, which was greater than the number of players who chose strawberry. "
            "Let N be the number of different assignments of flavors to players that meet "
            "these conditions. Find the remainder when N is divided by 1000.\n"
            "Mark your solution with \\boxed\n"
            "Answer:"
        ),
    },
    {
        "problem_id": "I-3",
        "expected_answer": "16",
        "question": (
            "Problem: Find the number of ordered pairs (x,y), where both x and y are "
            "integers between -100 and 100, inclusive, such that 12x^2 - xy - 6y^2 = 0.\n"
            "Mark your solution with \\boxed\n"
            "Answer:"
        ),
    },
]


########################################
# 2. Answer extraction
########################################

def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{...} notation"""
    if "\\boxed" in text:
        ans = text.split("\\boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
            return a.strip()
        else:
            a = ans.split("$")[0].strip()
            return a.strip()
    return ""


def simple_answer_match(predicted: str, ground_truth: str) -> bool:
    """Simple answer matching"""
    if predicted is None or ground_truth is None:
        return False

    pred_clean = str(predicted).strip().lower()
    gt_clean = str(ground_truth).strip().lower()

    if pred_clean == gt_clean:
        return True

    try:
        pred_num = float(pred_clean)
        gt_num = float(gt_clean)
        return abs(pred_num - gt_num) < 1e-6
    except:
        pass

    return False


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

        is_correct = 1 if simple_answer_match(answer, ground_truth) else 0
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
    print(f"STANDARD MODE: {problem['problem_id']}")
    print(f"{'='*80}")

    # Initialize with memory-optimized settings for 32B model
    deep_llm = DeepThinkLLM(
        model=model,
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.85,  # Leave 15% headroom (~4.8GB per GPU)
        max_num_seqs=32,               # Reduce warm-up batch size from 256
        max_model_len=4096             # Limit context window (budget=4000 tokens)
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

    # Extract answers
    for trace in result.all_traces:
        if 'text' in trace:
            trace['extracted_answer'] = extract_boxed_answer(trace['text'])

    # Evaluate
    correct_count = sum(
        1 for t in result.all_traces
        if simple_answer_match(t.get('extracted_answer'), problem['expected_answer'])
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
        'mode': 'standard',
        'accuracy': accuracy,
        'correct': correct_count,
        'total': result.total_traces_count,
        'tokens': result.total_tokens,
        'correlation': corr_stats,
        'result': result.to_dict()
    }


def run_branching_eval(problem: Dict, model: str, initial: int, max_total: int, tensor_parallel_size: int = 1) -> Dict:
    """Run branching DeepConf evaluation"""
    print(f"\n{'='*80}")
    print(f"BRANCHING MODE: {problem['problem_id']}")
    print(f"{'='*80}")

    # Initialize with memory-optimized settings for 32B model
    branching_llm = BranchingDeepThinkLLM(
        model=model,
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.85,  # Leave 15% headroom (~4.8GB per GPU)
        max_num_seqs=32,               # Reduce warm-up batch size from 256
        max_model_len=4096             # Limit context window (budget=4000 tokens)
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

    # Extract answers
    for trace in result.all_traces:
        if 'text' in trace:
            trace['extracted_answer'] = extract_boxed_answer(trace['text'])

    # Evaluate by depth
    depth_stats = {}
    for trace in result.all_traces:
        depth = trace.get('depth', 0)
        if depth not in depth_stats:
            depth_stats[depth] = {'total': 0, 'correct': 0}

        depth_stats[depth]['total'] += 1
        if simple_answer_match(trace.get('extracted_answer'), problem['expected_answer']):
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
        'mode': 'branching',
        'accuracy': accuracy,
        'correct': correct_count,
        'total': total_count,
        'tokens': result.total_tokens,
        'depth_stats': depth_stats,
        'correlation': corr_stats,
        'branching_stats': result.branching_stats if hasattr(result, 'branching_stats') else {},
        'result': result.to_dict()
    }


########################################
# 5. Main runner
########################################

def main():
    # Configure GPU environment for 4x RTX 5000 Ada (32GB each, 128GB total)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    parser = argparse.ArgumentParser(description='AIME 2025 Manual Evaluation')
    parser.add_argument('--mode', type=str, choices=['standard', 'branching'], required=True,
                       help='Evaluation mode')
    parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
                       help='Model to use')
    parser.add_argument('--budget', type=int, default=8,
                       help='Number of traces for standard mode')
    parser.add_argument('--initial_branches', type=int, default=2,
                       help='Initial branches for branching mode')
    parser.add_argument('--max_total_branches', type=int, default=6,
                       help='Max total branches for branching mode')
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                       help='Number of GPUs for tensor parallelism (default: 4 for 4x RTX 5000 Ada)')
    parser.add_argument('--output_dir', type=str, default='results/aime25_manual',
                       help='Output directory')
    parser.add_argument('--problem_ids', type=str, nargs='+', default=None,
                       help='Specific problem IDs to run (e.g., I-1 I-2)')

    args = parser.parse_args()

    # Filter problems if specified
    problems_to_run = AIME25_SUBSET
    if args.problem_ids:
        problems_to_run = [p for p in AIME25_SUBSET if p['problem_id'] in args.problem_ids]

    print(f"\n{'='*80}")
    print(f"AIME 2025 MANUAL EVALUATION")
    print(f"{'='*80}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Problems: {len(problems_to_run)}")
    print(f"{'='*80}\n")

    # Run evaluations
    results = []
    for i, problem in enumerate(problems_to_run, 1):
        print(f"\n{'#'*80}")
        print(f"# Problem {i}/{len(problems_to_run)}: {problem['problem_id']}")
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

        except Exception as e:
            print(f"\nERROR on {problem['problem_id']}: {e}")
            import traceback
            traceback.print_exc()
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
    print(f"  Avg tokens/problem: {total_tokens/len(results):,.0f}")

    print(f"\nPer-problem:")
    for r in results:
        print(f"  {r['problem_id']}: {r['correct']}/{r['total']} ({r['accuracy']:.1f}%)")

    # Correlation summary
    correlations = [r['correlation'].get('mean_conf_corr', 0) for r in results
                   if 'mean_conf_corr' in r['correlation']]
    if correlations:
        print(f"\nAverage correlation: {np.mean(correlations):.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
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
                'avg_correlation': np.mean(correlations) if correlations else None
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
            'per_problem': [
                {
                    'problem_id': r['problem_id'],
                    'accuracy': r['accuracy'],
                    'correct': r['correct'],
                    'total': r['total']
                }
                for r in results
            ]
        }, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  {output_file}")
    print(f"  {json_file}")
    print()


if __name__ == "__main__":
    main()
