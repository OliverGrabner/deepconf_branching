"""
Traditional Self-Consistency on AIME 2025 I and II

This script implements traditional self-consistency (Wang et al., 2022):
1. Generate N reasoning paths for each question
2. Extract answers from each path
3. Use simple majority voting to select final answer
4. Evaluate against ground truth

Usage:
    python run_traditional_sc_aime25.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --num_traces 64
"""

import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import Counter
from tqdm import tqdm

# Add parent directory to path to import local deepconf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from datasets import load_dataset
from vllm import SamplingParams

from deepconf import DeepThinkLLM, prepare_prompt, equal_func


def load_aime25(subset=None):
    """Load AIME 2025 dataset from Hugging Face (opencompass/AIME2025)."""
    if subset:
        ds = load_dataset("opencompass/AIME2025", name=subset, split="test")
        datasets = [(subset, ds)]
    else:
        ds1 = load_dataset("opencompass/AIME2025", name="AIME2025-I", split="test")
        ds2 = load_dataset("opencompass/AIME2025", name="AIME2025-II", split="test")
        datasets = [("AIME2025-I", ds1), ("AIME2025-II", ds2)]

    return datasets


def traditional_majority_vote(answers: List[str]) -> Tuple[str, Dict[str, int]]:
    """
    Traditional self-consistency: simple majority voting

    Returns:
        voted_answer: The answer with the most votes
        vote_distribution: Dictionary showing vote counts for each answer
    """
    if not answers:
        return None, {}

    # Count votes
    vote_counts = Counter(answers)

    # Get most common answer
    voted_answer = vote_counts.most_common(1)[0][0]

    # Create distribution dictionary
    vote_distribution = dict(vote_counts)

    return voted_answer, vote_distribution


def process_question(
    deep_llm: DeepThinkLLM,
    question: str,
    ground_truth: str,
    num_traces: int,
    sampling_params: SamplingParams,
    model_type: str = "deepseek"
) -> Dict[str, Any]:
    """
    Process a single question using traditional self-consistency

    Returns detailed results including:
    - All individual traces and their answers
    - Vote distribution
    - Final answer and correctness
    - Token statistics and timing
    """

    # Prepare prompt
    prompt = prepare_prompt(question, deep_llm.tokenizer, model_type)

    # Generate traces using offline mode (but we'll only use majority voting)
    result = deep_llm.deepthink(
        prompt=prompt,
        mode="offline",
        budget=num_traces,
        sampling_params=sampling_params,
        compute_multiple_voting=False  # We'll do our own majority voting
    )

    # Extract answers from all traces
    all_answers = []
    valid_traces = []

    for trace in result.all_traces:
        extracted_answer = trace.get('extracted_answer')
        if extracted_answer is not None:
            all_answers.append(extracted_answer)
            valid_traces.append({
                'answer': extracted_answer,
                'text': trace.get('text', '')[:200] + '...',  # First 200 chars
                'num_tokens': trace.get('num_tokens', 0),
                'stop_reason': trace.get('stop_reason', 'unknown')
            })

    # Perform traditional majority voting
    voted_answer, vote_distribution = traditional_majority_vote(all_answers)

    # Evaluate correctness
    is_correct = False
    if voted_answer and ground_truth:
        try:
            is_correct = equal_func(voted_answer, ground_truth)
        except Exception as e:
            print(f"Warning: Error in equal_func: {e}")
            is_correct = str(voted_answer) == str(ground_truth)

    # Calculate individual trace accuracy
    trace_accuracies = []
    for answer in all_answers:
        try:
            trace_correct = equal_func(answer, ground_truth)
        except:
            trace_correct = str(answer) == str(ground_truth)
        trace_accuracies.append(trace_correct)

    individual_accuracy = sum(trace_accuracies) / len(trace_accuracies) if trace_accuracies else 0.0

    # Compile results
    question_result = {
        'question': question,
        'ground_truth': ground_truth,
        'voted_answer': voted_answer,
        'is_correct': is_correct,
        'num_traces_generated': len(result.all_traces),
        'num_valid_traces': len(all_answers),
        'individual_trace_accuracy': individual_accuracy,
        'vote_distribution': vote_distribution,
        'valid_traces': valid_traces,
        'statistics': {
            'total_tokens': result.total_tokens,
            'avg_tokens_per_trace': result.avg_tokens_per_trace,
            'generation_time': result.generation_time,
            'processing_time': result.processing_time,
            'total_time': result.total_time,
            'throughput_tokens_per_sec': result.total_tokens / result.generation_time if result.generation_time > 0 else 0
        }
    }

    return question_result


def print_question_summary(qid: int, result: Dict[str, Any]):
    """Print a concise summary for a single question"""
    correctness = "✓" if result['is_correct'] else "✗"
    print(f"\nQ{qid}: {correctness}")
    print(f"  Ground Truth: {result['ground_truth']}")
    print(f"  Voted Answer: {result['voted_answer']}")
    print(f"  Valid Traces: {result['num_valid_traces']}/{result['num_traces_generated']}")
    print(f"  Individual Accuracy: {result['individual_trace_accuracy']:.1%}")
    print(f"  Vote Distribution: {result['vote_distribution']}")
    print(f"  Tokens: {result['statistics']['total_tokens']} ({result['statistics']['avg_tokens_per_trace']:.1f} avg)")
    print(f"  Time: {result['statistics']['total_time']:.2f}s")


def generate_summary_report(all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Generate comprehensive summary statistics across all datasets"""

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

        summary['by_dataset'][dataset_name] = {
            'num_questions': num_questions,
            'num_correct': num_correct,
            'accuracy': accuracy,
            'avg_individual_trace_accuracy': avg_individual_accuracy,
            'total_tokens': total_tokens,
            'avg_tokens_per_question': avg_tokens,
            'total_time': total_time,
            'avg_time_per_question': avg_time,
            'throughput_tokens_per_sec': total_tokens / total_time if total_time > 0 else 0
        }

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
        'throughput_tokens_per_sec': overall_tokens / overall_time if overall_time > 0 else 0
    }

    return summary


def print_final_summary(summary: Dict[str, Any]):
    """Print formatted final summary"""
    print("\n" + "="*80)
    print("TRADITIONAL SELF-CONSISTENCY - FINAL SUMMARY")
    print("="*80)

    # Per-dataset results
    print("\nPer-Dataset Results:")
    print("-" * 80)
    for dataset_name, stats in summary['by_dataset'].items():
        print(f"\n{dataset_name}:")
        print(f"  Questions: {stats['num_questions']}")
        print(f"  Correct: {stats['num_correct']}/{stats['num_questions']} ({stats['accuracy']:.1%})")
        print(f"  Avg Individual Trace Accuracy: {stats['avg_individual_trace_accuracy']:.1%}")
        print(f"  Total Tokens: {stats['total_tokens']:,}")
        print(f"  Avg Tokens/Question: {stats['avg_tokens_per_question']:.1f}")
        print(f"  Total Time: {stats['total_time']:.2f}s")
        print(f"  Avg Time/Question: {stats['avg_time_per_question']:.2f}s")
        print(f"  Throughput: {stats['throughput_tokens_per_sec']:.1f} tokens/sec")

    # Overall results
    print("\n" + "-" * 80)
    print("Overall Results (AIME25-I + AIME25-II):")
    print("-" * 80)
    overall = summary['overall']
    print(f"  Total Questions: {overall['num_questions']}")
    print(f"  Total Correct: {overall['num_correct']}/{overall['num_questions']} ({overall['accuracy']:.1%})")
    print(f"  Total Tokens: {overall['total_tokens']:,}")
    print(f"  Avg Tokens/Question: {overall['avg_tokens_per_question']:.1f}")
    print(f"  Total Time: {overall['total_time']:.2f}s ({overall['total_time']/60:.1f} minutes)")
    print(f"  Avg Time/Question: {overall['avg_time_per_question']:.2f}s")
    print(f"  Overall Throughput: {overall['throughput_tokens_per_sec']:.1f} tokens/sec")
    print("="*80)


def save_results(
    all_results: Dict[str, List[Dict[str, Any]]],
    summary: Dict[str, Any],
    output_dir: str,
    args: argparse.Namespace
):
    """Save results in multiple formats for easy analysis"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results as JSON
    detailed_output = {
        'metadata': {
            'timestamp': timestamp,
            'model': args.model,
            'num_traces': args.num_traces,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k,
            'max_tokens': args.max_tokens,
            'model_type': args.model_type,
        },
        'results': all_results,
        'summary': summary
    }

    json_filename = os.path.join(output_dir, f"traditional_sc_aime25_detailed_{timestamp}.json")
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(detailed_output, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {json_filename}")

    # Save summary as CSV for easy viewing
    summary_rows = []
    for dataset_name, results in all_results.items():
        for i, result in enumerate(results):
            summary_rows.append({
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
            })

    df = pd.DataFrame(summary_rows)
    csv_filename = os.path.join(output_dir, f"traditional_sc_aime25_summary_{timestamp}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Summary CSV saved to: {csv_filename}")

    # Save aggregate statistics
    stats_filename = os.path.join(output_dir, f"traditional_sc_aime25_stats_{timestamp}.json")
    with open(stats_filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Aggregate statistics saved to: {stats_filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Traditional Self-Consistency on AIME 2025 I and II',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                       help='Model path or name')
    parser.add_argument('--model_type', type=str, default="deepseek",
                       choices=["deepseek", "gpt"],
                       help='Model type for prompt formatting')
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                       help='Tensor parallel size (number of GPUs)')

    # Self-consistency parameters
    parser.add_argument('--num_traces', type=int, default=64,
                       help='Number of reasoning traces to generate (N in SC)')

    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (use 1.0 for diversity)')
    parser.add_argument('--top_p', type=float, default=1.0,
                       help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Top-k sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=130000,
                       help='Maximum tokens per generation')

    # Dataset selection
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['AIME2025-I', 'AIME2025-II'],
                       help='Run on specific dataset only (default: both)')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Start from this question index (for resuming)')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='End at this question index (for partial runs)')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default="outputs_sc",
                       help='Output directory for results')

    args = parser.parse_args()

    # Setup GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    print("\n" + "="*80)
    print("TRADITIONAL SELF-CONSISTENCY ON AIME 2025")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Number of traces (N): {args.num_traces}")
    print(f"Temperature: {args.temperature}")
    print(f"GPUs: {args.tensor_parallel_size}")
    print("="*80 + "\n")

    print("Loading AIME 2025 datasets...")
    datasets = load_aime25(args.dataset)

    # Initialize model
    print(f"\nInitializing DeepThinkLLM with {args.model}...")
    deep_llm = DeepThinkLLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
        trust_remote_code=True
    )

    # Create sampling parameters
    sampling_params = SamplingParams(
        n=args.num_traces,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=20,  # Needed for confidence computation (even though we won't use it for voting)
    )

    # Process all datasets
    all_results = {}

    for dataset_name, dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name} ({len(dataset)} questions)")
        print('='*80)

        dataset_results = []

        # Determine range
        start = args.start_idx
        end = args.end_idx if args.end_idx is not None else len(dataset)

        for i in tqdm(range(start, end), desc=f"{dataset_name}"):
            question_data = dataset[i]
            question = question_data['question']
            ground_truth = str(question_data.get('answer', '')).strip()

            print(f"\n{'='*60}")
            print(f"Question {i+1}/{len(dataset)}")
            print('='*60)
            print(f"Q: {question[:150]}...")

            # Process question
            result = process_question(
                deep_llm=deep_llm,
                question=question,
                ground_truth=ground_truth,
                num_traces=args.num_traces,
                sampling_params=sampling_params,
                model_type=args.model_type
            )

            dataset_results.append(result)

            # Print summary
            print_question_summary(i, result)

        all_results[dataset_name] = dataset_results

    # Generate summary
    print("\n\nGenerating summary report...")
    summary = generate_summary_report(all_results)

    # Print final summary
    print_final_summary(summary)

    # Save results
    save_results(all_results, summary, args.output_dir, args)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
