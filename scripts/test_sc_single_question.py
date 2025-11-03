"""
Quick test script for Traditional Self-Consistency on a single AIME question

Usage:
    python test_sc_single_question.py
"""

import os
import torch
from datasets import load_dataset
from vllm import SamplingParams
from deepconf import DeepThinkLLM, prepare_prompt, equal_func
from collections import Counter


def main():
    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    # Load a single question from AIME25-I
    print("\nLoading AIME 2025-I dataset...")
    dataset = load_dataset("opencompass/AIME2025", name="AIME2025-I", split="test")

    # Use first question
    question_data = dataset[0]
    question = question_data['question']
    ground_truth = str(question_data.get('answer', '')).strip()

    print(f"\nQuestion: {question}")
    print(f"Ground Truth: {ground_truth}")

    # Initialize model
    print("\nInitializing model (this may take a minute)...")
    model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    deep_llm = DeepThinkLLM(
        model=model_name,
        tensor_parallel_size=4,
        enable_prefix_caching=True,
        trust_remote_code=True
    )

    # Prepare prompt
    prompt = prepare_prompt(question, deep_llm.tokenizer, "deepseek")

    # Create sampling parameters for SC
    # Small number of traces for quick testing
    num_traces = 8
    sampling_params = SamplingParams(
        n=num_traces,
        temperature=1.0,  # Important: temperature > 0 for diversity
        top_p=1.0,
        top_k=40,
        max_tokens=130000,
        logprobs=20,
    )

    print(f"\nGenerating {num_traces} reasoning traces...")
    print("(This will take a few minutes with 8 traces)")

    # Generate traces
    result = deep_llm.deepthink(
        prompt=prompt,
        mode="offline",
        budget=num_traces,
        sampling_params=sampling_params,
        compute_multiple_voting=False  # We'll do our own majority voting
    )

    # Extract answers
    print("\nExtracting answers from traces...")
    all_answers = []
    for i, trace in enumerate(result.all_traces):
        extracted_answer = trace.get('extracted_answer')
        if extracted_answer:
            all_answers.append(extracted_answer)
            print(f"  Trace {i+1}: {extracted_answer}")
        else:
            print(f"  Trace {i+1}: [No answer extracted]")

    # Perform majority voting
    if all_answers:
        vote_counts = Counter(all_answers)
        voted_answer = vote_counts.most_common(1)[0][0]

        print(f"\nVote Distribution:")
        for answer, count in vote_counts.most_common():
            print(f"  {answer}: {count} votes")

        print(f"\nFinal Answer (Majority Vote): {voted_answer}")
        print(f"Ground Truth: {ground_truth}")

        # Check correctness
        try:
            is_correct = equal_func(voted_answer, ground_truth)
        except:
            is_correct = str(voted_answer) == str(ground_truth)

        result_str = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        print(f"\nResult: {result_str}")

        # Individual trace accuracy
        correct_traces = sum(1 for ans in all_answers if equal_func(ans, ground_truth))
        individual_accuracy = correct_traces / len(all_answers)
        print(f"Individual Trace Accuracy: {correct_traces}/{len(all_answers)} ({individual_accuracy:.1%})")

        print(f"\nStatistics:")
        print(f"  Total tokens: {result.total_tokens:,}")
        print(f"  Avg tokens/trace: {result.avg_tokens_per_trace:.1f}")
        print(f"  Generation time: {result.generation_time:.2f}s")
        print(f"  Throughput: {result.total_tokens / result.generation_time:.1f} tokens/sec")
    else:
        print("\nNo valid answers extracted from any trace!")

    print("\n" + "="*60)
    print("Test complete! Now try running the full script:")
    print("python run_traditional_sc_aime25.py --num_traces 64")
    print("="*60)


if __name__ == "__main__":
    main()
