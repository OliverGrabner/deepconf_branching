"""
⚠️  DEPRECATED: Use scripts/run_experiment.py instead.

Branching Self-Consistency on GSM8k

This script implements branching self-consistency on the GSM8k benchmark:
1. Start with S traces
2. During generation, dynamically branch high-confidence traces
3. Reach M traces by ~75% of average generation length
4. Use simple majority voting on final answers

Usage:
    python run_branching_sc_gsm8k.py \
        --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
        --start_traces 8 \
        --max_traces 32 \
        --historical_stats historical_stats/gsm8k_token_stats_latest.json
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from collections import Counter
from tqdm import tqdm

# Add parent directory to path to import local deepconf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from datasets import load_dataset
from vllm import SamplingParams

from deepconf import DeepThinkLLM, prepare_prompt
from deepconf.utils import extract_answer_gsm8k, equal_func_gsm8k

# For incremental saving
TEMP_RESULTS_SUFFIX = "_temp.json"


def load_gsm8k(split="test"):
    """Load GSM8k dataset from Hugging Face"""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return ds


def extract_gsm8k_ground_truth(answer_text: str) -> str:
    """
    Extract ground truth from GSM8k answer field

    GSM8k format: "reasoning text ... #### 123"
    """
    if "####" in answer_text:
        parts = answer_text.split("####")
        if len(parts) > 1:
            # Extract number after ####
            gt = parts[-1].strip()
            # Remove any non-numeric characters except minus and decimal
            import re
            numbers = re.findall(r'-?\d+\.?\d*', gt)
            if numbers:
                return numbers[0]

    return answer_text.strip()


def load_historical_stats(stats_file: str) -> Dict[str, Dict[str, Any]]:
    """Load historical token statistics"""
    with open(stats_file, 'r') as f:
        data = json.load(f)
    return data['statistics']


def get_average_tokens(historical_stats: Dict, question_idx: int) -> int:
    """Get historical average tokens for a specific question"""
    q_key = str(question_idx)
    if q_key in historical_stats:
        return int(historical_stats[q_key]['avg_tokens'])

    # Fallback: compute mean across all questions
    if historical_stats:
        all_avgs = [stats['avg_tokens'] for stats in historical_stats.values()]
        return int(sum(all_avgs) / len(all_avgs)) if all_avgs else 8000

    # Ultimate fallback (GSM8k problems are typically shorter than AIME)
    return 5000


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
    model_type: str = "deepseek"
) -> Dict[str, Any]:
    """
    Process a single question using branching self-consistency

    Returns detailed results including branching genealogy
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
        compute_multiple_voting=False  # Use simple majority voting
    )

    # Extract answers from all traces
    all_answers = []
    valid_traces = []
    full_traces = []  # Store full trace data for visualization

    for trace in result.all_traces:
        # Use GSM8k-specific extraction
        extracted_answer = extract_answer_gsm8k(trace.get('text', ''))
        confs = trace.get('confs', [])

        # Calculate final tail confidence (mean of last 2048 tokens)
        tail_window = 2048
        if len(confs) >= tail_window:
            final_tail_confidence = float(np.mean(confs[-tail_window:]))
        elif confs:
            final_tail_confidence = float(np.mean(confs))
        else:
            final_tail_confidence = 0.0

        # Check if this trace is correct
        is_correct = False
        if extracted_answer and ground_truth:
            try:
                is_correct = equal_func_gsm8k(extracted_answer, ground_truth)
            except:
                is_correct = str(extracted_answer) == str(ground_truth)

        # Store full trace data (including confidences for visualization)
        full_traces.append({
            'trace_idx': trace.get('trace_idx'),
            'parent_idx': trace.get('parent_idx'),
            'answer': extracted_answer,
            'is_correct': is_correct,
            'num_tokens': trace.get('num_tokens', 0),
            'tokens_generated': trace.get('tokens_generated', 0),
            'generation_started_at_iteration': trace.get('generation_started_at_iteration', 0),
            'generation_started_at_tokens': trace.get('generation_started_at_tokens', 0),
            'confs': confs,  # Include for visualization
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

    # Get voted answer using GSM8k-specific voting
    if all_answers:
        vote_counts = Counter(all_answers)
        voted_answer = vote_counts.most_common(1)[0][0]
    else:
        voted_answer = None

    # Evaluate correctness
    is_correct = False
    if voted_answer and ground_truth:
        try:
            is_correct = equal_func_gsm8k(voted_answer, ground_truth)
        except Exception as e:
            print(f"Warning: Error in equal_func_gsm8k: {e}")
            is_correct = str(voted_answer) == str(ground_truth)

    # Calculate individual trace accuracy
    trace_accuracies = []
    for answer in all_answers:
        try:
            trace_correct = equal_func_gsm8k(answer, ground_truth)
        except:
            trace_correct = str(answer) == str(ground_truth)
        trace_accuracies.append(trace_correct)

    individual_accuracy = sum(trace_accuracies) / len(trace_accuracies) if trace_accuracies else 0.0

    # Count votes
    vote_counts = Counter(all_answers)
    vote_distribution = dict(vote_counts)

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
        'full_traces': full_traces,  # Include full trace data with confidences
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


def print_question_summary(qid: int, result: Dict[str, Any]):
    """Print a concise summary for a single question"""
    correctness = "✓" if result['is_correct'] else "✗"
    print(f"\nQ{qid}: {correctness}")
    print(f"  Ground Truth: {result['ground_truth']}")
    print(f"  Voted Answer: {result['voted_answer']}")
    print(f"  Valid Traces: {result['num_valid_traces']}/{result['num_traces_generated']}")
    print(f"  Individual Accuracy: {result['individual_trace_accuracy']:.1%}")

    if result.get('branch_genealogy'):
        stats = result['branch_genealogy'].get('statistics', {})
        print(f"  Original Traces: {stats.get('original_traces', 0)}")
        print(f"  Branched Traces: {stats.get('branched_traces', 0)}")
        print(f"  Branch Events: {stats.get('total_branch_events', 0)}")

    print(f"  Tokens: {result['statistics']['total_tokens']} ({result['statistics']['avg_tokens_per_trace']:.1f} avg)")
    print(f"  Time: {result['statistics']['total_time']:.2f}s")


def generate_summary_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive summary statistics"""

    num_questions = len(results)
    num_correct = sum(1 for r in results if r['is_correct'])
    accuracy = num_correct / num_questions if num_questions > 0 else 0.0

    total_tokens = sum(r['statistics']['total_tokens'] for r in results)
    total_time = sum(r['statistics']['total_time'] for r in results)
    avg_tokens = total_tokens / num_questions if num_questions > 0 else 0
    avg_time = total_time / num_questions if num_questions > 0 else 0

    avg_individual_accuracy = sum(r['individual_trace_accuracy'] for r in results) / num_questions if num_questions > 0 else 0.0

    # Branching statistics
    total_branch_events = sum(
        r.get('branch_genealogy', {}).get('statistics', {}).get('total_branch_events', 0)
        for r in results
    )
    avg_branch_events = total_branch_events / num_questions if num_questions > 0 else 0

    summary = {
        'num_questions': num_questions,
        'num_correct': num_correct,
        'accuracy': accuracy,
        'avg_individual_trace_accuracy': avg_individual_accuracy,
        'total_tokens': total_tokens,
        'avg_tokens_per_question': avg_tokens,
        'total_time': total_time,
        'avg_time_per_question': avg_time,
        'throughput_tokens_per_sec': total_tokens / total_time if total_time > 0 else 0,
        'total_branch_events': total_branch_events,
        'avg_branch_events_per_question': avg_branch_events
    }

    return summary


def print_final_summary(summary: Dict[str, Any]):
    """Print formatted final summary"""
    print("\n" + "="*80)
    print("BRANCHING SELF-CONSISTENCY - GSM8K FINAL SUMMARY")
    print("="*80)

    print(f"\nTotal Questions: {summary['num_questions']}")
    print(f"Correct: {summary['num_correct']}/{summary['num_questions']} ({summary['accuracy']:.1%})")
    print(f"Avg Individual Trace Accuracy: {summary['avg_individual_trace_accuracy']:.1%}")
    print(f"Avg Branch Events: {summary['avg_branch_events_per_question']:.1f}")
    print(f"Total Tokens: {summary['total_tokens']:,}")
    print(f"Avg Tokens/Question: {summary['avg_tokens_per_question']:.1f}")
    print(f"Total Time: {summary['total_time']:.2f}s ({summary['total_time']/60:.1f} minutes)")
    print(f"Avg Time/Question: {summary['avg_time_per_question']:.2f}s")
    print(f"Overall Throughput: {summary['throughput_tokens_per_sec']:.1f} tokens/sec")
    print("="*80)


def save_results(
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_dir: str,
    args: argparse.Namespace
):
    """Save results in multiple formats"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results as JSON
    detailed_output = {
        'metadata': {
            'timestamp': timestamp,
            'model': args.model,
            'start_traces': args.start_traces,
            'max_traces': args.max_traces,
            'selected_percent': args.selected_percent,
            'n_iterations': args.n_iterations,
            'branch_goal': args.branch_goal,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k,
            'max_tokens': args.max_tokens,
            'model_type': args.model_type,
            'historical_stats_file': args.historical_stats,
            'dataset': 'GSM8k',
            'split': 'test'
        },
        'results': {'GSM8k': results},  # Wrap in dict for compatibility with visualization
        'summary': summary
    }

    json_filename = os.path.join(output_dir, f"branching_sc_gsm8k_detailed_{timestamp}.json")
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(detailed_output, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {json_filename}")

    # Save summary as CSV
    summary_rows = []
    for i, result in enumerate(results):
        genealogy_stats = result.get('branch_genealogy', {}).get('statistics', {})

        summary_rows.append({
            'question_id': i,
            'is_correct': result['is_correct'],
            'ground_truth': result['ground_truth'],
            'voted_answer': result['voted_answer'],
            'num_valid_traces': result['num_valid_traces'],
            'num_traces_generated': result['num_traces_generated'],
            'individual_trace_accuracy': result['individual_trace_accuracy'],
            'original_traces': genealogy_stats.get('original_traces', 0),
            'branched_traces': genealogy_stats.get('branched_traces', 0),
            'branch_events': genealogy_stats.get('total_branch_events', 0),
            'total_tokens': result['statistics']['total_tokens'],
            'total_time': result['statistics']['total_time'],
        })

    df = pd.DataFrame(summary_rows)
    csv_filename = os.path.join(output_dir, f"branching_sc_gsm8k_summary_{timestamp}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Summary CSV saved to: {csv_filename}")

    # Save aggregate statistics
    stats_filename = os.path.join(output_dir, f"branching_sc_gsm8k_stats_{timestamp}.json")
    with open(stats_filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Aggregate statistics saved to: {stats_filename}")

    # Return filenames for visualization
    return json_filename


def save_incremental_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    timestamp: str,
    args: argparse.Namespace
):
    """Save incremental results during processing"""
    temp_output = {
        'metadata': {
            'timestamp': timestamp,
            'model': args.model,
            'start_traces': args.start_traces,
            'max_traces': args.max_traces,
            'selected_percent': args.selected_percent,
            'n_iterations': args.n_iterations,
            'branch_goal': args.branch_goal,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k,
            'max_tokens': args.max_tokens,
            'model_type': args.model_type,
            'historical_stats_file': args.historical_stats,
            'dataset': 'GSM8k',
            'split': 'test',
            'status': 'in_progress'
        },
        'results': {'GSM8k': results},
        'summary': None  # Will be computed at the end
    }

    temp_filename = os.path.join(output_dir, f"branching_sc_gsm8k_detailed_{timestamp}{TEMP_RESULTS_SUFFIX}")
    with open(temp_filename, 'w', encoding='utf-8') as f:
        json.dump(temp_output, f, indent=2, ensure_ascii=False)

    return temp_filename


def generate_question_visualizations(
    result: Dict[str, Any],
    question_idx: int,
    viz_dir: str,
    timestamp: str
):
    """Generate visualizations for a single question"""
    try:
        from visualize_branching_results import (
            create_per_problem_summary,
            create_genealogy_graph,
            create_confidence_evolution_plot
        )

        # Prepare paths
        summary_path = os.path.join(viz_dir, f"summary_GSM8k_q{question_idx}_{timestamp}.png")
        genealogy_path = os.path.join(viz_dir, f"genealogy_GSM8k_q{question_idx}_{timestamp}.png")
        confidence_path = os.path.join(viz_dir, f"confidence_GSM8k_q{question_idx}_{timestamp}.png")

        # Generate 3 visualizations
        create_per_problem_summary(result, "GSM8k", question_idx, summary_path)
        create_genealogy_graph(
            result.get('branch_genealogy', {}),
            result.get('full_traces', []),
            result.get('ground_truth', ''),
            genealogy_path
        )
        create_confidence_evolution_plot(
            result.get('full_traces', []),
            result.get('branch_genealogy', {}),
            result.get('branching_config', {}),
            result.get('ground_truth', ''),
            confidence_path
        )

        return True

    except ImportError:
        print(f"  ⚠️  Visualization module not found for Q{question_idx}")
        return False
    except Exception as e:
        print(f"  ⚠️  Visualization failed for Q{question_idx}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Branching Self-Consistency on GSM8k',
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

    # Branching parameters
    parser.add_argument('--start_traces', type=int, default=8,
                       help='Number of initial traces')
    parser.add_argument('--max_traces', type=int, default=32,
                       help='Maximum number of traces after branching')
    parser.add_argument('--selected_percent', type=float, default=0.60,
                       help='Top % of traces eligible for branching')
    parser.add_argument('--n_iterations', type=int, default=10,
                       help='Number of branching check points')
    parser.add_argument('--branch_goal', type=float, default=0.75,
                       help='Target completion percentage for branching')
    parser.add_argument('--historical_stats', type=str, required=True,
                       help='Path to historical token statistics JSON file')

    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1.0,
                       help='Top-p sampling parameter')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Top-k sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=130000,
                       help='Maximum tokens per generation')

    # Dataset selection
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Start from this question index')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='End at this question index (default: all 1319 questions)')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default="outputs_sc",
                       help='Output directory for results')

    args = parser.parse_args()

    # Setup GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    # Create output directory and visualization directory
    os.makedirs(args.output_dir, exist_ok=True)
    viz_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Create timestamp for this run (used for all incremental saves)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load historical statistics
    print(f"\nLoading historical statistics from: {args.historical_stats}")
    historical_stats = load_historical_stats(args.historical_stats)

    # Print header
    print("\n" + "="*80)
    print("BRANCHING SELF-CONSISTENCY ON GSM8K")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Start traces: {args.start_traces}, Max traces: {args.max_traces}")
    print(f"Selected percent: {args.selected_percent*100:.0f}%")
    print(f"Iterations: {args.n_iterations}, Branch goal: {args.branch_goal*100:.0f}%")
    print(f"Temperature: {args.temperature}")
    print(f"GPUs: {args.tensor_parallel_size}")
    print("="*80 + "\n")

    # Load dataset
    print("Loading GSM8k test set...")
    dataset = load_gsm8k(split="test")
    print(f"Loaded {len(dataset)} questions")

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

    # Process questions
    results = []

    # Determine range
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(dataset)

    print(f"\nProcessing questions {start} to {end-1} ({end-start} total)")
    print('='*80)

    for i in tqdm(range(start, end), desc="GSM8k"):
        question_data = dataset[i]
        question = question_data['question']
        answer_text = question_data['answer']

        # Extract ground truth number from answer
        ground_truth = extract_gsm8k_ground_truth(answer_text)

        # Get historical average tokens
        avg_tokens = get_average_tokens(historical_stats, i)

        print(f"\n{'='*60}")
        print(f"Question {i+1}/{len(dataset)}")
        print('='*60)
        print(f"Q: {question[:150]}...")
        print(f"GT: {ground_truth}")
        print(f"Historical avg tokens: {avg_tokens}")

        try:
            # Process question
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
                model_type=args.model_type
            )

            results.append(result)

            # Print summary
            print_question_summary(i, result)

            # Save incremental results
            temp_file = save_incremental_results(results, args.output_dir, timestamp, args)
            print(f"  💾 Saved progress: {temp_file}")

            # Generate visualizations for this question
            print(f"  📊 Generating visualizations for Q{i}...")
            viz_success = generate_question_visualizations(
                result, i, viz_dir, timestamp
            )
            if viz_success:
                print(f"  ✓ Visualizations created")

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user (Ctrl+C)")
            print(f"Progress saved: {len(results)}/{end-start} questions completed")
            raise
        except Exception as e:
            print(f"\n❌ Error processing Q{i}: {e}")
            print(f"Continuing with next question...")
            import traceback
            traceback.print_exc()
            continue

    # Generate summary
    print("\n\nGenerating summary report...")
    summary = generate_summary_report(results)

    # Print final summary
    print_final_summary(summary)

    # Save final results
    json_filepath = save_results(results, summary, args.output_dir, args)

    # Remove temporary file
    temp_filepath = os.path.join(args.output_dir, f"branching_sc_gsm8k_detailed_{timestamp}{TEMP_RESULTS_SUFFIX}")
    if os.path.exists(temp_filepath):
        os.remove(temp_filepath)
        print(f"Removed temporary file: {temp_filepath}")

    # Generate dataset-wide visualizations
    print("\n" + "="*80)
    print("GENERATING DATASET-WIDE VISUALIZATIONS")
    print("="*80)

    try:
        # Import visualization module
        from visualize_branching_results import create_token_usage_plot, create_accuracy_analysis_plot

        # Token usage plot
        token_path = os.path.join(viz_dir, f"token_usage_{timestamp}.png")
        create_token_usage_plot({'GSM8k': results}, token_path)
        print(f"  ✓ Token usage plot: {token_path}")

        # Accuracy analysis plot
        accuracy_path = os.path.join(viz_dir, f"accuracy_analysis_{timestamp}.png")
        create_accuracy_analysis_plot({'GSM8k': results}, accuracy_path)
        print(f"  ✓ Accuracy analysis plot: {accuracy_path}")

    except ImportError:
        print("\n⚠️  Visualization module not found")
        print("Run manually: python scripts/visualize_branching_results.py --results " + json_filepath)
    except Exception as e:
        print(f"\n⚠️  Dataset-wide visualization failed: {e}")
        print("Run manually: python scripts/visualize_branching_results.py --results " + json_filepath)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults: {json_filepath}")
    print(f"Visualizations: {viz_dir}")


if __name__ == "__main__":
    main()
