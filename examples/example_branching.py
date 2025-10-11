"""
Example usage of BranchingDeepThinkLLM with confidence-based trace spawning

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import json
import pickle
import argparse
from datetime import datetime
from vllm import SamplingParams
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepconf.branching_wrapper import BranchingDeepThinkLLM
from dynasor.core.evaluator import math_equal


def prepare_prompt(question: str, tokenizer, model_type: str = "deepseek") -> str:
    """Prepare prompt for a single question"""
    if model_type == "deepseek":
        messages = [
            {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
            {"role": "user", "content": question}
        ]
    else:
        messages = [
            {"role": "user", "content": question}
        ]
    
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return full_prompt


def prepare_prompt_gpt(question: str, tokenizer, reasoning_effort: str = "high") -> str:
    """Prepare prompt for GPT models with reasoning effort"""
    messages = [
        {"role": "user", "content": question}
    ]
    
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        reasoning_effort=reasoning_effort,
        add_generation_prompt=True
    )
    
    return full_prompt


def quick_parse(text: str) -> str:
    """Parse LaTeX text content"""
    if '\\text{' in text and '}' in text:
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            content = text[start + 6:end]
            text = text[:start] + content + text[end + 1:]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    """Check if answer equals ground truth"""
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)


def evaluate_branching_results(output, ground_truth):
    """Evaluate branching results with detailed analysis"""
    evaluation = {
        'voting_results': {},
        'trace_accuracy': {},
        'branching_analysis': {}
    }
    
    # Evaluate voting results
    for method, result in output.voting_results.items():
        if result and result.get('answer'):
            try:
                is_correct = equal_func(result['answer'], ground_truth)
            except:
                is_correct = str(result['answer']) == str(ground_truth)
            
            evaluation['voting_results'][method] = {
                'answer': result['answer'],
                'is_correct': is_correct,
                'confidence': result.get('confidence'),
                'num_votes': result.get('num_votes', 0)
            }
    
    # Analyze individual trace accuracy by depth
    depth_accuracy = {}
    for trace in output.all_traces:
        depth = trace.get('depth', 0)
        if depth not in depth_accuracy:
            depth_accuracy[depth] = {'correct': 0, 'total': 0}
        
        if trace.get('extracted_answer'):
            depth_accuracy[depth]['total'] += 1
            try:
                if equal_func(trace['extracted_answer'], ground_truth):
                    depth_accuracy[depth]['correct'] += 1
            except:
                pass
    
    evaluation['trace_accuracy'] = depth_accuracy
    
    # Analyze branching effectiveness
    if output.branching_stats['total_branches'] > 0:
        # Check if branches improved accuracy
        base_traces = [t for t in output.all_traces if t.get('depth', 0) == 0]
        branch_traces = [t for t in output.all_traces if t.get('depth', 0) > 0]
        
        base_correct = sum(1 for t in base_traces 
                          if t.get('extracted_answer') and 
                          equal_func(t['extracted_answer'], ground_truth))
        branch_correct = sum(1 for t in branch_traces 
                            if t.get('extracted_answer') and 
                            equal_func(t['extracted_answer'], ground_truth))
        
        base_accuracy = base_correct / len(base_traces) if base_traces else 0
        branch_accuracy = branch_correct / len(branch_traces) if branch_traces else 0
        
        evaluation['branching_analysis'] = {
            'base_accuracy': base_accuracy,
            'branch_accuracy': branch_accuracy,
            'improvement': branch_accuracy - base_accuracy,
            'avg_confidence_at_branch': output.branching_stats.get('avg_confidence_at_branch', 0)
        }
    
    return evaluation


def print_branching_report(question, ground_truth, evaluation, output):
    """Print detailed branching experiment report"""
    print(f"\n=== Branching Experiment Report ===")
    print(f"Question: {question[:100]}...")
    print(f"Ground truth: {ground_truth}")
    print(f"Total traces: {output.total_traces_count}")
    print(f"  - Initial traces: {output.config['initial_branches']}")
    print(f"  - Branched traces: {output.branching_stats['total_branches']}")
    
    print(f"\n=== Trace Accuracy by Depth ===")
    for depth, stats in evaluation['trace_accuracy'].items():
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            print(f"  Depth {depth}: {stats['correct']}/{stats['total']} ({acc:.1%})")
    
    if evaluation.get('branching_analysis'):
        print(f"\n=== Branching Effectiveness ===")
        ba = evaluation['branching_analysis']
        print(f"  Base trace accuracy: {ba['base_accuracy']:.1%}")
        print(f"  Branch trace accuracy: {ba['branch_accuracy']:.1%}")
        print(f"  Improvement: {ba['improvement']:+.1%}")
        print(f"  Avg confidence at branch points: {ba['avg_confidence_at_branch']:.3f}")
    
    print(f"\n=== Voting Method Results ===")
    print("-" * 80)
    print(f"{'Method':<25} {'Answer':<15} {'Correct':<8} {'Confidence':<12}")
    print("-" * 80)
    
    for method, result in evaluation['voting_results'].items():
        answer = str(result['answer'])[:13] + '..' if len(str(result['answer'])) > 15 else str(result['answer'])
        correct = '✓' if result['is_correct'] else '✗'
        conf = f"{result['confidence']:.3f}" if result['confidence'] else '-'
        print(f"{method:<25} {answer:<15} {correct:<8} {conf:<12}")


def main():
    parser = argparse.ArgumentParser(description='Branching DeepThink Experiment')
    parser.add_argument('--model', type=str, default="openai/gpt-oss-120b",
                       help='Model path or name')
    parser.add_argument('--tensor_parallel_size', type=int, default=8,
                       help='Tensor parallel size for model')
    parser.add_argument('--dataset', type=str, default="aime25.jsonl",
                       help='Dataset file path')
    parser.add_argument('--qid', type=int, required=True,
                       help='Question ID to process (0-based index)')
    parser.add_argument('--rid', type=str, default="branching_run",
                       help='Run ID for identification')
    
    # Branching parameters
    parser.add_argument('--initial_branches', type=int, default=4,
                       help='Number of initial traces')
    parser.add_argument('--max_total_branches', type=int, default=32,
                       help='Maximum total traces including branches')
    parser.add_argument('--base_branch_prob', type=float, default=0.05,
                       help='Minimum branching probability')
    parser.add_argument('--max_branch_prob', type=float, default=0.5,
                       help='Maximum branching probability at high confidence')
    parser.add_argument('--confidence_threshold', type=float, default=1.5,
                       help='Confidence value for max branching probability')
    parser.add_argument('--min_steps_before_branch', type=int, default=100,
                       help='Minimum tokens before allowing branching')
    parser.add_argument('--branch_cooldown', type=int, default=200,
                       help='Minimum tokens between branches')
    
    # Generation parameters
    parser.add_argument('--window_size', type=int, default=128,
                       help='Sliding window size for confidence')
    parser.add_argument('--max_tokens', type=int, default=32000,
                       help='Maximum tokens per generation')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top-p sampling parameter')
    parser.add_argument('--model_type', type=str, default="gpt", 
                       choices=["deepseek", "gpt"],
                       help='Model type for prompt formatting')
    parser.add_argument('--reasoning_effort', type=str, default="high",
                       help='Reasoning effort for GPT models')
    parser.add_argument('--output_dir', type=str, default="outputs",
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]
    
    if args.qid >= len(data) or args.qid < 0:
        raise ValueError(f"Question ID {args.qid} is out of range (0-{len(data)-1})")
    
    question_data = data[args.qid]
    question = question_data['question']
    ground_truth = str(question_data.get('answer', '')).strip()
    
    print(f"\nProcessing question {args.qid}: {question[:100]}...")
    print(f"Ground truth: {ground_truth}")
    
    # Initialize BranchingDeepThinkLLM
    branching_llm = BranchingDeepThinkLLM(
        model=args.model, 
        tensor_parallel_size=args.tensor_parallel_size, 
        enable_prefix_caching=True
    )
    
    # Prepare prompt
    print("\nPreparing prompt...")
    if args.model_type == "gpt":
        prompt = prepare_prompt_gpt(question, branching_llm.tokenizer, args.reasoning_effort)
    else:
        prompt = prepare_prompt(question, branching_llm.tokenizer, args.model_type)
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        logprobs=20,
    )
    
    # Run branching deep thinking
    print(f"\nStarting branching experiment...")
    print(f"  Initial branches: {args.initial_branches}")
    print(f"  Max total branches: {args.max_total_branches}")
    print(f"  Branch probability range: {args.base_branch_prob:.1%} - {args.max_branch_prob:.1%}")
    print(f"  Confidence threshold: {args.confidence_threshold}")
    
    result = branching_llm.branching_deepthink(
        prompt=prompt,
        initial_branches=args.initial_branches,
        max_total_branches=args.max_total_branches,
        base_branch_prob=args.base_branch_prob,
        max_branch_prob=args.max_branch_prob,
        confidence_threshold=args.confidence_threshold,
        min_steps_before_branch=args.min_steps_before_branch,
        branch_cooldown=args.branch_cooldown,
        window_size=args.window_size,
        sampling_params=sampling_params
    )
    
    # Evaluate results
    if ground_truth and result.voting_results:
        evaluation = evaluate_branching_results(result, ground_truth)
        print_branching_report(question, ground_truth, evaluation, result)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_data = result.to_dict()
    result_data.update({
        'question': question,
        'ground_truth': ground_truth,
        'qid': args.qid,
        'run_id': args.rid,
        'evaluation': evaluation if ground_truth and result.voting_results else None,
        'experiment_type': 'branching'
    })
    
    result_filename = f"{args.output_dir}/branching_qid{args.qid}_rid{args.rid}_{timestamp}.pkl"
    
    with open(result_filename, 'wb') as f:
        pickle.dump(result_data, f)
    
    print(f"\nResults saved to {result_filename}")


if __name__ == "__main__":
    main()