"""
Unified Self-Consistency Experiment Runner

Run traditional or branching self-consistency experiments on any supported dataset.
Supports both single-question testing and full dataset processing.

Usage Examples:
    # Branching SC on AIME25, single question
    python run_experiment.py --experiment branching --dataset AIME2025-I \
        --question_id 0 --start_traces 8 --max_traces 32

    # Branching SC on AIME25, full dataset
    python run_experiment.py --experiment branching --dataset AIME2025-I \
        --start_traces 8 --max_traces 32 \
        --historical_stats historical_stats/aime25_token_stats_latest.json

    # Traditional SC on GSM8k, single question
    python run_experiment.py --experiment traditional --dataset gsm8k \
        --question_id 42 --num_traces 64

    # Traditional SC on GSM8k, batch (questions 0-99)
    python run_experiment.py --experiment traditional --dataset gsm8k \
        --num_traces 64 --start_idx 0 --end_idx 100
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from collections import Counter
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from vllm import SamplingParams

from deepconf import DeepThinkLLM, prepare_prompt, equal_func
from deepconf.utils import extract_answer_gsm8k, equal_func_gsm8k
from experiment_utils import (
    load_dataset_by_name,
    get_question_and_ground_truth,
    load_historical_stats,
    get_average_tokens,
    save_results,
    save_incremental_results,
    generate_summary_report,
    print_question_summary,
    print_final_summary,
    generate_question_visualizations,
    generate_dataset_visualizations,
    create_metadata_dict,
    TEMP_RESULTS_SUFFIX
)


def process_question_traditional(
    deep_llm: DeepThinkLLM,
    question: str,
    ground_truth: str,
    num_traces: int,
    sampling_params: SamplingParams,
    model_type: str,
    dataset_name: str
) -> Dict[str, Any]:
    """
    Process a single question using traditional self-consistency

    Args:
        deep_llm: Initialized model
        question: Question text
        ground_truth: Ground truth answer
        num_traces: Number of traces to generate
        sampling_params: Sampling parameters
        model_type: Model type for prompt formatting
        dataset_name: Dataset name (for answer extraction)

    Returns:
        Question result dictionary
    """
    # Prepare prompt
    prompt = prepare_prompt(question, deep_llm.tokenizer, model_type)

    # Generate traces using offline mode
    result = deep_llm.deepthink(
        prompt=prompt,
        mode="offline",
        budget=num_traces,
        sampling_params=sampling_params,
        compute_multiple_voting=False
    )

    # Extract answers from all traces
    all_answers = []
    valid_traces = []

    is_gsm8k = "gsm8k" in dataset_name.lower()
    equal_func_to_use = equal_func_gsm8k if is_gsm8k else equal_func

    for trace in result.all_traces:
        # Use appropriate extraction method
        if is_gsm8k:
            extracted_answer = extract_answer_gsm8k(trace.get('text', ''))
        else:
            extracted_answer = trace.get('extracted_answer')

        if extracted_answer is not None:
            all_answers.append(extracted_answer)

            # Check correctness
            is_correct = False
            try:
                is_correct = equal_func_to_use(extracted_answer, ground_truth)
            except:
                is_correct = str(extracted_answer) == str(ground_truth)

            valid_traces.append({
                'answer': extracted_answer,
                'is_correct': is_correct,
                'num_tokens': trace.get('num_tokens', 0)
            })

    # Perform majority voting
    if all_answers:
        vote_counts = Counter(all_answers)
        voted_answer = vote_counts.most_common(1)[0][0]
        vote_distribution = dict(vote_counts)
    else:
        voted_answer = None
        vote_distribution = {}

    # Evaluate correctness
    is_correct = False
    if voted_answer and ground_truth:
        try:
            is_correct = equal_func_to_use(voted_answer, ground_truth)
        except:
            is_correct = str(voted_answer) == str(ground_truth)

    # Calculate individual trace accuracy
    trace_accuracies = []
    for answer in all_answers:
        try:
            trace_correct = equal_func_to_use(answer, ground_truth)
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


def process_question_peak_branching(
    deep_llm: DeepThinkLLM,
    question: str,
    ground_truth: str,
    initial_traces: int,
    max_traces: int,
    confidence_threshold: float,
    peak_window_size: int,
    min_peak_distance: int,
    peak_selection_ratio: float,
    exclusion_zone_size: int,
    sampling_params: SamplingParams,
    model_type: str,
    dataset_name: str
) -> Dict[str, Any]:
    """
    Process a single question using peak branching self-consistency

    Args:
        deep_llm: Initialized model
        question: Question text
        ground_truth: Ground truth answer
        initial_traces: Number of initial traces
        max_traces: Maximum total traces (stops when next doubling would exceed)
        confidence_threshold: Minimum confidence for peaks
        peak_window_size: Window size for peak detection
        min_peak_distance: Minimum distance between peaks in same trace
        peak_selection_ratio: Valid range for peaks
        exclusion_zone_size: Size of exclusion zone around used peaks
        sampling_params: Sampling parameters
        model_type: Model type for prompt formatting
        dataset_name: Dataset name (for answer extraction)

    Returns:
        Question result dictionary
    """
    # Prepare prompt
    prompt = prepare_prompt(question, deep_llm.tokenizer, model_type)

    # Generate traces using peak branching mode
    result = deep_llm.deepthink(
        prompt=prompt,
        mode="peak_branching",
        initial_traces=initial_traces,
        max_traces=max_traces,
        confidence_threshold=confidence_threshold,
        peak_window_size=peak_window_size,
        min_peak_distance=min_peak_distance,
        peak_selection_ratio=peak_selection_ratio,
        exclusion_zone_size=exclusion_zone_size,
        sampling_params=sampling_params,
        compute_multiple_voting=False
    )

    # Extract answers and check correctness
    all_answers = []
    valid_traces = []
    initial_traces_list = []
    branch_traces_list = []

    is_gsm8k = "gsm8k" in dataset_name.lower()
    equal_func_to_use = equal_func_gsm8k if is_gsm8k else equal_func

    for trace in result.all_traces:
        # Use appropriate extraction method
        if is_gsm8k:
            extracted_answer = extract_answer_gsm8k(trace.get('text', ''))
        else:
            extracted_answer = trace.get('extracted_answer')

        if extracted_answer is not None:
            all_answers.append(extracted_answer)

            # Check correctness
            is_correct = False
            try:
                is_correct = equal_func_to_use(extracted_answer, ground_truth)
            except:
                is_correct = str(extracted_answer) == str(ground_truth)

            trace_info = {
                'trace_idx': trace.get('trace_idx'),
                'stage': trace.get('stage', 0),  # Multi-stage support
                'depth': trace.get('depth', 0),  # Kept for compatibility
                'parent_idx': trace.get('parent_idx'),
                'branch_point_tokens': trace.get('branch_point_tokens'),
                'answer': extracted_answer,
                'is_correct': is_correct,
                'tokens_generated': trace.get('tokens_generated', 0),
                'confidence_peaks': trace.get('confidence_peaks', [])
            }
            valid_traces.append(trace_info)

            # Separate by stage (0 = initial, 1+ = branches)
            if trace.get('stage', 0) == 0:
                initial_traces_list.append(trace_info)
            else:
                branch_traces_list.append(trace_info)

    # Perform weighted voting
    if all_answers:
        # Apply weights based on depth
        weighted_answers = []
        weights = []
        for trace_info in valid_traces:
            weighted_answers.append(trace_info['answer'])
            # Weight: 1.0 for initial (stage 0), 1.1 for ALL branches (stage 1+)
            weight = 1.0 if trace_info['stage'] == 0 else 1.1
            weights.append(weight)

        # Calculate weighted vote
        from collections import Counter
        weighted_counts = Counter()
        for ans, w in zip(weighted_answers, weights):
            weighted_counts[ans] += w

        voted_answer = weighted_counts.most_common(1)[0][0] if weighted_counts else None
        vote_distribution = dict(weighted_counts)
    else:
        voted_answer = None
        vote_distribution = {}

    # Evaluate correctness
    is_correct = False
    if voted_answer and ground_truth:
        try:
            is_correct = equal_func_to_use(voted_answer, ground_truth)
        except:
            is_correct = str(voted_answer) == str(ground_truth)

    # Calculate accuracies
    initial_accuracy = sum(1 for t in initial_traces_list if t['is_correct']) / len(initial_traces_list) if initial_traces_list else 0.0
    branch_accuracy = sum(1 for t in branch_traces_list if t['is_correct']) / len(branch_traces_list) if branch_traces_list else 0.0
    overall_accuracy = sum(1 for t in valid_traces if t['is_correct']) / len(valid_traces) if valid_traces else 0.0

    # Get peak branching statistics
    peak_stats = result.peak_branching_stats if hasattr(result, 'peak_branching_stats') else {}

    # Compile results
    question_result = {
        'question': question,
        'ground_truth': ground_truth,
        'voted_answer': voted_answer,
        'is_correct': is_correct,
        'num_traces_generated': len(result.all_traces),
        'num_initial_traces': len(initial_traces_list),
        'num_branch_traces': len(branch_traces_list),
        'initial_trace_accuracy': initial_accuracy,
        'branch_trace_accuracy': branch_accuracy,
        'overall_trace_accuracy': overall_accuracy,
        'vote_distribution': vote_distribution,
        'valid_traces': valid_traces,
        'peak_branching_stats': peak_stats,
        'statistics': {
            'total_tokens_generated': result.total_tokens,
            'avg_tokens_per_trace': result.avg_tokens_per_trace,
            'generation_time': result.generation_time,
            'processing_time': result.processing_time,
            'total_time': result.total_time,
            'throughput_tokens_per_sec': result.total_tokens / result.generation_time if result.generation_time > 0 else 0,
            'prefix_cache_savings': peak_stats.get('prefix_cache_savings', 0),
            'prefix_cache_savings_pct': peak_stats.get('prefix_cache_savings_pct', 0)
        }
    }

    return question_result


def process_question_branching(
    deep_llm: DeepThinkLLM,
    question: str,
    ground_truth: str,
    start_traces: int,
    max_traces: int,
    selected_percent: float,
    n_iterations: int,
    branch_goal: float,
    average_tokens: int,
    sampling_params: SamplingParams,
    model_type: str,
    dataset_name: str
) -> Dict[str, Any]:
    """
    Process a single question using branching self-consistency

    Args:
        deep_llm: Initialized model
        question: Question text
        ground_truth: Ground truth answer
        start_traces: Number of initial traces
        max_traces: Maximum traces after branching
        selected_percent: Percent eligible for branching
        n_iterations: Number of branching checkpoints
        branch_goal: Goal completion percentage for branching
        average_tokens: Estimated average tokens
        sampling_params: Sampling parameters
        model_type: Model type for prompt formatting
        dataset_name: Dataset name (for answer extraction)

    Returns:
        Question result dictionary
    """
    # Prepare prompt
    prompt = prepare_prompt(question, deep_llm.tokenizer, model_type)

    # Run branching mode
    result = deep_llm.deepthink(
        prompt=prompt,
        mode="branching",
        start_traces=start_traces,
        max_traces=max_traces,
        selected_percent=selected_percent,
        n_iterations=n_iterations,
        branch_goal=branch_goal,
        average_tokens=average_tokens,
        window_size=2048,
        sampling_params=sampling_params,
        compute_multiple_voting=False
    )

    # Extract answers from all traces
    all_answers = []
    valid_traces = []
    full_traces = []

    is_gsm8k = "gsm8k" in dataset_name.lower()
    equal_func_to_use = equal_func_gsm8k if is_gsm8k else equal_func

    for trace in result.all_traces:
        # Use appropriate extraction method
        if is_gsm8k:
            extracted_answer = extract_answer_gsm8k(trace.get('text', ''))
        else:
            extracted_answer = trace.get('extracted_answer')

        confs = trace.get('confs', [])

        # Calculate final tail confidence
        tail_window = 2048
        if len(confs) >= tail_window:
            final_tail_confidence = float(np.mean(confs[-tail_window:]))
        elif confs:
            final_tail_confidence = float(np.mean(confs))
        else:
            final_tail_confidence = 0.0

        # Check correctness
        is_correct = False
        if extracted_answer and ground_truth:
            try:
                is_correct = equal_func_to_use(extracted_answer, ground_truth)
            except:
                is_correct = str(extracted_answer) == str(ground_truth)

        # Store full trace data
        full_traces.append({
            'trace_idx': trace.get('trace_idx'),
            'parent_idx': trace.get('parent_idx'),
            'answer': extracted_answer,
            'is_correct': is_correct,
            'num_tokens': trace.get('num_tokens', 0),
            'tokens_generated': trace.get('tokens_generated', 0),
            'generation_started_at_iteration': trace.get('generation_started_at_iteration', 0),
            'generation_started_at_tokens': trace.get('generation_started_at_tokens', 0),
            'confs': confs,
            'final_tail_confidence': final_tail_confidence,
            'extracted_answer': extracted_answer
        })

        if extracted_answer is not None:
            all_answers.append(extracted_answer)
            valid_traces.append({
                'trace_idx': trace.get('trace_idx'),
                'parent_idx': trace.get('parent_idx'),
                'answer': extracted_answer,
                'is_correct': is_correct,
                'num_tokens': trace.get('num_tokens', 0),
                'tokens_generated': trace.get('tokens_generated', 0),
                'final_tail_confidence': final_tail_confidence,
                'generation_started_at_iteration': trace.get('generation_started_at_iteration', 0),
                'generation_started_at_tokens': trace.get('generation_started_at_tokens', 0)
            })

    # Get voted answer
    if all_answers:
        vote_counts = Counter(all_answers)
        voted_answer = vote_counts.most_common(1)[0][0]
        vote_distribution = dict(vote_counts)
    else:
        voted_answer = None
        vote_distribution = {}

    # Evaluate correctness
    is_correct = False
    if voted_answer and ground_truth:
        try:
            is_correct = equal_func_to_use(voted_answer, ground_truth)
        except:
            is_correct = str(voted_answer) == str(ground_truth)

    # Calculate individual trace accuracy
    trace_accuracies = []
    for answer in all_answers:
        try:
            trace_correct = equal_func_to_use(answer, ground_truth)
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
        'full_traces': full_traces,
        'branch_genealogy': result.branch_genealogy,
        'branch_events': result.branch_events,
        'branching_config': result.branching_config,
        'statistics': {
            'total_tokens': result.total_tokens,
            'total_tokens_generated': result.total_tokens_generated,
            'avg_tokens_per_trace': result.avg_tokens_per_trace,
            'avg_tokens_generated_per_trace': result.avg_tokens_generated_per_trace,
            'generation_time': result.generation_time,
            'processing_time': result.processing_time,
            'total_time': result.total_time,
            'throughput_tokens_per_sec': result.total_tokens_generated / result.generation_time if result.generation_time > 0 else 0
        }
    }

    return question_result


def main():
    parser = argparse.ArgumentParser(
        description='Unified Self-Consistency Experiment Runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment configuration
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['traditional', 'branching', 'peak_branching'],
                       help='Experiment type')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name: AIME2025-I, AIME2025-II, gsm8k, or both (for AIME)')

    # Single question or batch
    parser.add_argument('--question_id', type=int, default=None,
                       help='Run on single question (takes precedence over batch)')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Start question index (batch mode)')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='End question index (batch mode)')

    # Model configuration
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                       help='Model path or name')
    parser.add_argument('--model_type', type=str, default="deepseek",
                       choices=["deepseek", "gpt"],
                       help='Model type for prompt formatting')
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                       help='Tensor parallel size (number of GPUs)')

    # Traditional SC parameters
    parser.add_argument('--num_traces', type=int, default=64,
                       help='[Traditional] Number of traces to generate')

    # Branching SC parameters
    parser.add_argument('--start_traces', type=int, default=8,
                       help='[Branching] Number of initial traces')
    parser.add_argument('--max_traces', type=int, default=32,
                       help='[Branching] Maximum number of traces after branching')
    parser.add_argument('--selected_percent', type=float, default=0.60,
                       help='[Branching] Top % of traces eligible for branching')
    parser.add_argument('--n_iterations', type=int, default=10,
                       help='[Branching] Number of branching checkpoints')
    parser.add_argument('--branch_goal', type=float, default=0.75,
                       help='[Branching] Target completion percentage for branching')
    parser.add_argument('--historical_stats', type=str, default=None,
                       help='[Branching] Path to historical token statistics JSON file')

    # Peak Branching parameters
    parser.add_argument('--initial_traces', type=int, default=8,
                       help='[Peak Branching] Number of initial traces')
    parser.add_argument('--max_traces', type=int, default=64,
                       help='[Peak Branching] Maximum total traces (stops when next doubling would exceed)')
    parser.add_argument('--confidence_threshold', type=float, default=1.5,
                       help='[Peak Branching] Minimum confidence for peaks')
    parser.add_argument('--peak_window_size', type=int, default=512,
                       help='[Peak Branching] Window size for peak detection')
    parser.add_argument('--min_peak_distance', type=int, default=256,
                       help='[Peak Branching] Minimum distance between peaks in same trace')
    parser.add_argument('--peak_selection_ratio', type=float, default=0.8,
                       help='[Peak Branching] Valid range for peaks (0.8 = middle 80%)')
    parser.add_argument('--exclusion_zone_size', type=int, default=200,
                       help='[Peak Branching] Size of exclusion zone around used peaks')

    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1.0,
                       help='Top-p sampling parameter')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Top-k sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=130000,
                       help='Maximum tokens per generation')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default="outputs",
                       help='Output directory for results')
    parser.add_argument('--no_visualizations', action='store_true',
                       help='Skip visualization generation')

    args = parser.parse_args()

    # Validate experiment-specific requirements
    if args.experiment == "branching" and args.question_id is None and args.historical_stats is None:
        parser.error("Branching experiments require --historical_stats for batch processing")

    # Setup GPU configuration
    # Only set CUDA_VISIBLE_DEVICES if not already set by user
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    # Create output directory and visualization directory
    os.makedirs(args.output_dir, exist_ok=True)
    viz_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load historical statistics (if needed)
    historical_stats = None
    if args.experiment == "branching" and args.historical_stats:
        print(f"\nLoading historical statistics from: {args.historical_stats}")
        historical_stats = load_historical_stats(args.historical_stats)

    # Print header
    print("\n" + "="*80)
    print(f"{args.experiment.upper()} SELF-CONSISTENCY EXPERIMENT")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")

    if args.question_id is not None:
        print(f"Mode: Single question (ID: {args.question_id})")
    else:
        print(f"Mode: Batch processing")

    if args.experiment == "traditional":
        print(f"Num traces: {args.num_traces}")
    elif args.experiment == "branching":
        print(f"Start traces: {args.start_traces}, Max traces: {args.max_traces}")
        print(f"Selected percent: {args.selected_percent*100:.0f}%")
        print(f"Iterations: {args.n_iterations}, Branch goal: {args.branch_goal*100:.0f}%")
    else:  # peak_branching
        print(f"Initial traces: {args.initial_traces}, Max traces: {args.max_traces}")
        print(f"Confidence threshold: {args.confidence_threshold}")
        print(f"Peak window: {args.peak_window_size}, Min distance: {args.min_peak_distance}")
        print(f"Peak selection ratio: {args.peak_selection_ratio*100:.0f}%")
        print(f"Exclusion zone size: {args.exclusion_zone_size} tokens")

    print(f"Temperature: {args.temperature}")
    print(f"GPUs: {args.tensor_parallel_size}")
    print("="*80 + "\n")

    # Load datasets
    print(f"Loading {args.dataset} dataset...")
    datasets = load_dataset_by_name(args.dataset, split="test")

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
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=20,
    )

    # Process datasets
    all_results = {}

    for dataset_name, dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name} ({len(dataset)} questions)")
        print('='*80)

        dataset_results = []

        # Single question mode
        if args.question_id is not None:
            if args.question_id >= len(dataset):
                print(f"Error: Question ID {args.question_id} out of range (0-{len(dataset)-1})")
                continue

            # Process single question
            start = args.question_id
            end = args.question_id + 1
        else:
            # Batch mode
            start = args.start_idx
            end = args.end_idx if args.end_idx is not None else len(dataset)

        for i in tqdm(range(start, end), desc=f"{dataset_name}"):
            question_data = dataset[i]
            question, ground_truth = get_question_and_ground_truth(dataset_name, question_data)

            print(f"\n{'='*60}")
            print(f"Question {i+1}/{len(dataset)}")
            print('='*60)
            print(f"Q: {question[:150]}...")
            if ground_truth:
                print(f"GT: {ground_truth}")

            try:
                # Process based on experiment type
                if args.experiment == "traditional":
                    result = process_question_traditional(
                        deep_llm=deep_llm,
                        question=question,
                        ground_truth=ground_truth,
                        num_traces=args.num_traces,
                        sampling_params=sampling_params,
                        model_type=args.model_type,
                        dataset_name=dataset_name
                    )

                elif args.experiment == "branching":
                    # Get average tokens
                    if historical_stats and args.question_id is None:
                        avg_tokens = get_average_tokens(historical_stats, dataset_name, i)
                    else:
                        # Use fallback for single question testing
                        avg_tokens = 5000 if "gsm8k" in dataset_name.lower() else 8000

                    print(f"Avg tokens estimate: {avg_tokens}")

                    result = process_question_branching(
                        deep_llm=deep_llm,
                        question=question,
                        ground_truth=ground_truth,
                        start_traces=args.start_traces,
                        max_traces=args.max_traces,
                        selected_percent=args.selected_percent,
                        n_iterations=args.n_iterations,
                        branch_goal=args.branch_goal,
                        average_tokens=avg_tokens,
                        sampling_params=sampling_params,
                        model_type=args.model_type,
                        dataset_name=dataset_name
                    )

                else:  # peak_branching
                    result = process_question_peak_branching(
                        deep_llm=deep_llm,
                        question=question,
                        ground_truth=ground_truth,
                        initial_traces=args.initial_traces,
                        max_traces=args.max_traces,
                        confidence_threshold=args.confidence_threshold,
                        peak_window_size=args.peak_window_size,
                        min_peak_distance=args.min_peak_distance,
                        peak_selection_ratio=args.peak_selection_ratio,
                        exclusion_zone_size=args.exclusion_zone_size,
                        sampling_params=sampling_params,
                        model_type=args.model_type,
                        dataset_name=dataset_name
                    )

                dataset_results.append(result)

                # Print summary
                print_question_summary(i, result, args.experiment)

                # Save incremental results (batch mode only)
                if args.question_id is None:
                    all_results[dataset_name] = dataset_results
                    metadata = create_metadata_dict(args, args.experiment)
                    temp_file = save_incremental_results(
                        all_results, args.output_dir, timestamp, metadata, args.experiment
                    )
                    print(f"  💾 Saved progress: {temp_file}")

                # Generate visualizations
                if not args.no_visualizations:
                    print(f"  📊 Generating visualizations for Q{i}...")
                    viz_success = generate_question_visualizations(
                        result, dataset_name, i, viz_dir, timestamp, args.experiment
                    )
                    if viz_success:
                        print(f"  ✓ Visualizations created")

            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user (Ctrl+C)")
                print(f"Progress saved: {len(dataset_results)}/{end-start} questions completed")
                raise
            except Exception as e:
                print(f"\n❌ Error processing Q{i}: {e}")
                print(f"Continuing with next question...")
                import traceback
                traceback.print_exc()
                continue

        all_results[dataset_name] = dataset_results

    # Generate summary
    print("\n\nGenerating summary report...")
    summary = generate_summary_report(all_results, args.experiment)

    # Print final summary
    print_final_summary(summary, args.experiment)

    # Save final results
    metadata = create_metadata_dict(args, args.experiment)
    json_filepath = save_results(all_results, summary, args.output_dir, metadata, args.experiment)

    # Remove temporary file (batch mode only)
    if args.question_id is None:
        temp_filepath = os.path.join(args.output_dir, f"{args.experiment}_sc_detailed_{timestamp}{TEMP_RESULTS_SUFFIX}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            print(f"Removed temporary file: {temp_filepath}")

    # Generate dataset-wide visualizations
    if not args.no_visualizations:
        print("\n" + "="*80)
        print("GENERATING DATASET-WIDE VISUALIZATIONS")
        print("="*80)

        generate_dataset_visualizations(all_results, viz_dir, timestamp, args.experiment)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults: {json_filepath}")
    print(f"Visualizations: {viz_dir}")


if __name__ == "__main__":
    main()
