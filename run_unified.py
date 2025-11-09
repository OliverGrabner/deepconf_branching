#!/usr/bin/env python3
"""
Unified experiment runner for deepconf branching experiments.

This runner:
- Supports multiple datasets (AIME2025-I, AIME2025-II, GSM8k)
- Uses YOUR confidence-threshold based branching from BranchingDeepThinkLLM
- Provides standardized output format
- Supports both branching and standard self-consistency

Usage:
    # Run YOUR branching on AIME
    python run_unified.py --mode branching --dataset AIME2025-I \
        --initial_branches 8 --max_total_branches 32 --confidence_threshold 1.5

    # Run YOUR branching on GSM8k
    python run_unified.py --mode branching --dataset gsm8k \
        --initial_branches 8 --max_total_branches 32

    # Standard self-consistency for comparison
    python run_unified.py --mode standard --dataset AIME2025-I --num_traces 32
"""

import os
import sys
import json
import pickle
import argparse
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import numpy as np

# Import dataset utilities and robust answer extraction
from utils_robust import (
    load_dataset_by_name,
    get_question_and_ground_truth,
    extract_answer_robust,
    check_answer_equality
)

# Import YOUR branching implementation
from deepconf import DeepThinkLLM
from deepconf.branching_wrapper import BranchingDeepThinkLLM
from vllm import SamplingParams
from transformers import AutoTokenizer


def process_question_standard(
    llm: DeepThinkLLM,
    prompt: str,
    num_traces: int,
    sampling_params: SamplingParams,
    dataset_type: str
) -> Dict[str, Any]:
    """Process a question using standard self-consistency."""

    # Run standard mode
    result = llm.deepthink(
        prompt=prompt,
        mode="offline",
        budget=num_traces,
        sampling_params=sampling_params,
        compute_multiple_voting=True
    )

    # Extract answers using robust method
    for trace in result.all_traces:
        if 'text' in trace:
            trace['extracted_answer'] = extract_answer_robust(trace['text'], dataset_type)

    return result


def process_question_branching(
    llm: BranchingDeepThinkLLM,
    prompt: str,
    initial_branches: int,
    max_total_branches: int,
    confidence_threshold: float,
    sampling_params: SamplingParams,
    dataset_type: str
) -> Dict[str, Any]:
    """Process a question using YOUR confidence-threshold branching."""

    # Run YOUR branching implementation
    result = llm.branching_deepthink(
        prompt=prompt,
        initial_branches=initial_branches,
        max_total_branches=max_total_branches,
        confidence_threshold=confidence_threshold,
        sampling_params=sampling_params,
        compute_multiple_voting=True
    )

    # Extract answers using robust method
    for trace in result.all_traces:
        if 'text' in trace:
            trace['extracted_answer'] = extract_answer_robust(trace['text'], dataset_type)

    return result


def evaluate_result(result: Any, ground_truth: str, dataset_type: str) -> Dict[str, Any]:
    """Evaluate results and compute metrics."""

    # Count correct answers
    correct_count = sum(
        1 for trace in result.all_traces
        if check_answer_equality(trace.get('extracted_answer'), ground_truth, dataset_type)
    )

    total_traces = len(result.all_traces)
    accuracy = (correct_count / total_traces * 100) if total_traces > 0 else 0.0

    # Get voted answer if available
    voted_answer = result.voted_answer if hasattr(result, 'voted_answer') else None
    voted_correct = check_answer_equality(voted_answer, ground_truth, dataset_type) if voted_answer else False

    return {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_traces': total_traces,
        'voted_answer': voted_answer,
        'voted_correct': voted_correct,
        'ground_truth': ground_truth,
        'total_tokens': result.total_tokens if hasattr(result, 'total_tokens') else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Unified experiment runner for deepconf branching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment mode
    parser.add_argument('--mode', type=str, required=True,
                       choices=['branching', 'standard'],
                       help='Experiment mode: branching (YOUR method) or standard SC')

    # Dataset selection
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset: AIME2025-I, AIME2025-II, gsm8k, or both (for all AIME)')

    # Processing range
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Start question index')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='End question index (None = all)')
    parser.add_argument('--single_question', type=int, default=None,
                       help='Process only this question index')

    # Model configuration
    parser.add_argument('--model', type=str,
                       default='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
                       help='Model name or path')
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                       help='Number of GPUs for tensor parallelism')

    # Standard SC parameters
    parser.add_argument('--num_traces', type=int, default=32,
                       help='[Standard] Number of traces')

    # YOUR Branching parameters
    parser.add_argument('--initial_branches', type=int, default=8,
                       help='[Branching] Initial traces')
    parser.add_argument('--max_total_branches', type=int, default=32,
                       help='[Branching] Maximum total traces')
    parser.add_argument('--confidence_threshold', type=float, default=1.5,
                       help='[Branching] Confidence threshold for branch points')
    parser.add_argument('--max_depth', type=int, default=1,
                       help='[Branching] Maximum branching depth')

    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1.0,
                       help='Top-p sampling')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Top-k sampling')
    parser.add_argument('--max_tokens', type=int, default=8192,
                       help='Maximum tokens per generation')

    # Output
    parser.add_argument('--output_dir', type=str, default='results/unified',
                       help='Output directory for results')
    parser.add_argument('--save_plots', action='store_true',
                       help='Generate visualization plots')

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Print configuration
    print("\n" + "="*80)
    print("UNIFIED DEEPCONF BRANCHING EXPERIMENT")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")

    if args.mode == "branching":
        print(f"Initial branches: {args.initial_branches}")
        print(f"Max total branches: {args.max_total_branches}")
        print(f"Confidence threshold: {args.confidence_threshold}")
    else:
        print(f"Num traces: {args.num_traces}")

    print(f"Temperature: {args.temperature}")
    print(f"Output dir: {args.output_dir}")
    print("="*80)

    # Load datasets
    print(f"\nLoading dataset {args.dataset}...")
    datasets = load_dataset_by_name(args.dataset)

    # Initialize model
    print(f"\nInitializing model...")
    if args.mode == "branching":
        llm = BranchingDeepThinkLLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
    else:
        llm = DeepThinkLLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )


    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=20
    )

    # Process each dataset
    all_results = {}

    for dataset_name, dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name} ({len(dataset)} questions)")
        print('='*80)

        # Determine range
        if args.single_question is not None:
            start_idx = args.single_question
            end_idx = args.single_question + 1
        else:
            start_idx = args.start_idx
            end_idx = args.end_idx if args.end_idx is not None else len(dataset)

        dataset_results = []

        # Process questions
        for i in tqdm(range(start_idx, min(end_idx, len(dataset))), desc=dataset_name):
            question_data = dataset[i]
            question, ground_truth = get_question_and_ground_truth(dataset_name, question_data)

            # Determine dataset type for answer extraction
            dataset_type = 'gsm8k' if 'gsm8k' in dataset_name.lower() else 'aime'

            print(f"\n{'='*60}")
            print(f"Question {i}/{len(dataset)}")
            print('='*60)
            print(f"Q: {question[:200]}...")
            print(f"GT: {ground_truth}")

            try:
                # Format prompt
                if 'gsm8k' in dataset_name.lower():
                    # GSM8k style prompt
                    prompt = question + "\n\nLet's solve this step by step."
                else:
                    # AIME style prompt
                    prompt = question + "\n\nSolve this problem step by step. Present your final answer clearly."

                # Process based on mode
                if args.mode == "branching":
                    result = process_question_branching(
                        llm=llm,
                        prompt=prompt,
                        initial_branches=args.initial_branches,
                        max_total_branches=args.max_total_branches,
                        confidence_threshold=args.confidence_threshold,
                        sampling_params=sampling_params,
                        dataset_type=dataset_type
                    )
                else:
                    result = process_question_standard(
                        llm=llm,
                        prompt=prompt,
                        num_traces=args.num_traces,
                        sampling_params=sampling_params,
                        dataset_type=dataset_type
                    )

                # Evaluate
                eval_metrics = evaluate_result(result, ground_truth, dataset_type)

                # Store results
                question_result = {
                    'question_idx': i,
                    'question': question,
                    'dataset_name': dataset_name,
                    'dataset_type': dataset_type,
                    **eval_metrics,
                    'mode': args.mode,
                }

                # Add branching-specific info
                if args.mode == "branching" and hasattr(result, 'branching_stats'):
                    question_result['branching_stats'] = result.branching_stats

                dataset_results.append(question_result)

                # Print summary
                print(f"\nResults:")
                print(f"  Voted answer: {eval_metrics['voted_answer']}")
                print(f"  Voted correct: {'✓' if eval_metrics['voted_correct'] else '✗'}")
                print(f"  Individual accuracy: {eval_metrics['accuracy']:.1f}%")
                print(f"  Correct traces: {eval_metrics['correct_count']}/{eval_metrics['total_traces']}")
                print(f"  Total tokens: {eval_metrics['total_tokens']:,}")

                # Save incremental results
                if (i - start_idx + 1) % 5 == 0:  # Save every 5 questions
                    temp_file = os.path.join(
                        args.output_dir,
                        f"{args.mode}_{dataset_name}_{timestamp}_temp.json"
                    )
                    with open(temp_file, 'w') as f:
                        json.dump(dataset_results, f, indent=2)
                    print(f"  💾 Saved progress to {temp_file}")

            except Exception as e:
                print(f"\n❌ Error processing question {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

        all_results[dataset_name] = dataset_results

    # Generate summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for dataset_name, results in all_results.items():
        if not results:
            continue

        total_questions = len(results)
        voted_correct = sum(1 for r in results if r['voted_correct'])
        avg_individual_accuracy = np.mean([r['accuracy'] for r in results])
        total_tokens = sum(r['total_tokens'] for r in results)

        print(f"\n{dataset_name}:")
        print(f"  Questions processed: {total_questions}")
        print(f"  Voted accuracy: {voted_correct}/{total_questions} ({voted_correct/total_questions*100:.1f}%)")
        print(f"  Avg individual trace accuracy: {avg_individual_accuracy:.1f}%")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Avg tokens/question: {total_tokens/total_questions:,.0f}")

    # Save final results
    output_file = os.path.join(
        args.output_dir,
        f"{args.mode}_{args.dataset}_{timestamp}.json"
    )

    final_output = {
        'metadata': {
            'mode': args.mode,
            'dataset': args.dataset,
            'model': args.model,
            'timestamp': timestamp,
            'args': vars(args)
        },
        'results': all_results,
        'summary': {
            dataset_name: {
                'total_questions': len(results),
                'voted_correct': sum(1 for r in results if r['voted_correct']),
                'voted_accuracy': sum(1 for r in results if r['voted_correct']) / len(results) * 100 if results else 0,
                'avg_individual_accuracy': np.mean([r['accuracy'] for r in results]) if results else 0,
                'total_tokens': sum(r['total_tokens'] for r in results)
            }
            for dataset_name, results in all_results.items()
        }
    }

    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    # Generate plots if requested
    if args.save_plots and args.mode == "branching":
        print("\n📊 Generating visualizations...")
        # This would call your existing visualization functions
        # You can integrate the plot generation from run_aime25_full.py here
        print("  (Visualization integration pending)")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()