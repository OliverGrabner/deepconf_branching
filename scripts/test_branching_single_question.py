"""
Test Branching Self-Consistency on a Single Question

Quick test to verify branching implementation works correctly

Usage:
    python scripts/test_branching_single_question.py --qid 0
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from datasets import load_dataset
from vllm import SamplingParams

from deepconf import DeepThinkLLM, prepare_prompt, equal_func


def main():
    parser = argparse.ArgumentParser(description='Test branching SC on single question')

    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                       help='Model path')
    parser.add_argument('--model_type', type=str, default="deepseek", help='Model type')
    parser.add_argument('--tensor_parallel_size', type=int, default=4, help='Number of GPUs')
    parser.add_argument('--dataset', type=str, default='AIME2025-I', help='Dataset')
    parser.add_argument('--qid', type=int, default=0, help='Question ID')

    # Branching parameters
    parser.add_argument('--start_traces', type=int, default=4, help='Initial traces')
    parser.add_argument('--max_traces', type=int, default=8, help='Max traces')
    parser.add_argument('--selected_percent', type=float, default=0.60, help='Top %')
    parser.add_argument('--n_iterations', type=int, default=5, help='Iterations')
    parser.add_argument('--branch_goal', type=float, default=0.75, help='Branch goal')
    parser.add_argument('--average_tokens', type=int, default=8000, help='Avg tokens estimate')

    # Sampling
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=130000)

    args = parser.parse_args()

    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading {args.dataset}...")
    dataset = load_dataset("opencompass/AIME2025", name=args.dataset, split="test")

    if args.qid >= len(dataset):
        print(f"Error: Question ID {args.qid} out of range (0-{len(dataset)-1})")
        return

    question_data = dataset[args.qid]
    question = question_data['question']
    ground_truth = str(question_data.get('answer', '')).strip()

    print(f"\nQuestion {args.qid}:")
    print(f"Q: {question}")
    print(f"Ground Truth: {ground_truth}")

    # Initialize model
    print(f"\nInitializing model...")
    deep_llm = DeepThinkLLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
        trust_remote_code=True
    )

    # Prepare prompt
    prompt = prepare_prompt(question, deep_llm.tokenizer, args.model_type)

    # Sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        top_k=40,
        max_tokens=args.max_tokens,
        logprobs=20
    )

    # Run branching SC
    print("\n" + "="*80)
    print("RUNNING BRANCHING SELF-CONSISTENCY")
    print("="*80)

    result = deep_llm.deepthink(
        prompt=prompt,
        mode="branching",
        start_traces=args.start_traces,
        max_traces=args.max_traces,
        selected_percent=args.selected_percent,
        n_iterations=args.n_iterations,
        branch_goal=args.branch_goal,
        average_tokens=args.average_tokens,
        window_size=2048,
        sampling_params=sampling_params,
        compute_multiple_voting=False
    )

    # Evaluate
    is_correct = False
    if result.final_answer and ground_truth:
        try:
            is_correct = equal_func(result.final_answer, ground_truth)
        except:
            is_correct = str(result.final_answer) == str(ground_truth)

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Ground Truth: {ground_truth}")
    print(f"Final Answer: {result.final_answer}")
    print(f"Correct: {'✓' if is_correct else '✗'}")
    print(f"\nTotal traces: {len(result.all_traces)}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Generation time: {result.generation_time:.2f}s")

    if result.branch_genealogy:
        stats = result.branch_genealogy['statistics']
        print(f"\nBranching Statistics:")
        print(f"  Original traces: {stats['original_traces']}")
        print(f"  Branched traces: {stats['branched_traces']}")
        print(f"  Total branch events: {stats['total_branch_events']}")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
