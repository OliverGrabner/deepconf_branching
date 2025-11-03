"""
Visualize Trace Confidence Evolution

Tracks and graphs the tail confidence of each trace as it generates its answer.
This helps understand how confidence evolves during generation and which traces
are more/less confident throughout the reasoning process.

Usage:
    # Run SC with visualization on a specific question
    python visualize_trace_confidence.py --qid 0 --num_traces 16 --dataset AIME2025-I

    # Or visualize from saved results
    python visualize_trace_confidence.py --load outputs_sc/trace_data_qid0.pkl
"""

import os
import sys
import json
import pickle
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# Add parent directory to path to import local deepconf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from datasets import load_dataset
from vllm import SamplingParams

from deepconf import DeepThinkLLM, prepare_prompt, equal_func

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Will generate ASCII plots only.")
    print("Install with: pip install matplotlib")


def compute_tail_confidence_evolution(confs: List[float], tail_size: int = 2048, step_size: int = 100) -> List[float]:
    """
    Compute tail confidence at regular intervals as the trace is generated

    Args:
        confs: List of token confidences
        tail_size: Size of tail window to compute confidence over
        step_size: How often to sample (every N tokens)

    Returns:
        List of tail confidence values at each step
    """
    evolution = []

    # Sample at regular intervals
    for i in range(0, len(confs), step_size):
        if i == 0:
            continue

        # Get tail window (last tail_size tokens up to current position)
        start_idx = max(0, i - tail_size)
        tail_window = confs[start_idx:i]

        if tail_window:
            tail_conf = np.mean(tail_window)
            evolution.append({
                'position': i,
                'tail_confidence': tail_conf,
                'window_size': len(tail_window)
            })

    return evolution


def generate_and_track_traces(
    deep_llm: DeepThinkLLM,
    question: str,
    ground_truth: str,
    num_traces: int,
    sampling_params: SamplingParams,
    model_type: str = "deepseek",
    tail_size: int = 2048,
    step_size: int = 100
) -> Dict[str, Any]:
    """
    Generate traces and track confidence evolution for each

    Returns:
        Dictionary with traces, their confidence evolution, and metadata
    """
    print(f"\nGenerating {num_traces} traces with confidence tracking...")
    print(f"Tail window size: {tail_size} tokens")
    print(f"Sampling every: {step_size} tokens")

    # Prepare prompt
    prompt = prepare_prompt(question, deep_llm.tokenizer, model_type)

    # Generate traces using offline mode
    result = deep_llm.deepthink(
        prompt=prompt,
        mode="offline",
        budget=num_traces,
        sampling_params=sampling_params,
        compute_multiple_voting=True
    )

    print(f"\nProcessing {len(result.all_traces)} traces...")

    # Process each trace to extract confidence evolution
    traces_data = []
    for i, trace in enumerate(result.all_traces):
        confs = trace.get('confs', [])
        extracted_answer = trace.get('extracted_answer')

        if not confs:
            print(f"  Trace {i+1}: No confidence data available")
            continue

        # Compute confidence evolution
        evolution = compute_tail_confidence_evolution(confs, tail_size, step_size)

        # Check if answer is correct
        is_correct = False
        if extracted_answer and ground_truth:
            try:
                is_correct = equal_func(extracted_answer, ground_truth)
            except:
                is_correct = str(extracted_answer) == str(ground_truth)

        trace_info = {
            'trace_id': i,
            'answer': extracted_answer,
            'is_correct': is_correct,
            'num_tokens': len(confs),
            'final_tail_confidence': np.mean(confs[-tail_size:]) if len(confs) >= tail_size else np.mean(confs),
            'confidence_evolution': evolution,
            'text_preview': trace.get('text', '')[:500]  # First 500 chars
        }

        traces_data.append(trace_info)

        print(f"  Trace {i+1}: {len(confs)} tokens, answer={extracted_answer}, correct={is_correct}")

    return {
        'question': question,
        'ground_truth': ground_truth,
        'traces': traces_data,
        'voting_results': result.voting_results,
        'final_answer': result.final_answer,
        'num_traces': num_traces,
        'tail_size': tail_size,
        'step_size': step_size,
        'statistics': {
            'total_tokens': result.total_tokens,
            'generation_time': result.generation_time,
            'avg_tokens_per_trace': result.avg_tokens_per_trace
        }
    }


def plot_confidence_evolution_matplotlib(data: Dict[str, Any], output_path: str):
    """Create matplotlib visualization of confidence evolution"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping graphical plot.")
        return

    traces = data['traces']
    if not traces:
        print("No traces to plot")
        return

    # Separate correct and incorrect traces
    correct_traces = [t for t in traces if t['is_correct']]
    incorrect_traces = [t for t in traces if not t['is_correct']]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Trace Confidence Evolution\nQuestion: {data["question"][:100]}...',
                 fontsize=14, fontweight='bold')

    # Plot 1: All traces
    ax1 = axes[0, 0]
    for trace in traces:
        evolution = trace['confidence_evolution']
        if evolution:
            positions = [e['position'] for e in evolution]
            confidences = [e['tail_confidence'] for e in evolution]
            color = 'green' if trace['is_correct'] else 'red'
            alpha = 0.6 if trace['is_correct'] else 0.3
            ax1.plot(positions, confidences, color=color, alpha=alpha, linewidth=1)

    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Tail Confidence (mean of last N tokens)')
    ax1.set_title('All Traces (Green=Correct, Red=Incorrect)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Correct', 'Incorrect'])

    # Plot 2: Correct traces only
    ax2 = axes[0, 1]
    if correct_traces:
        cmap = cm.get_cmap('Greens')
        for i, trace in enumerate(correct_traces):
            evolution = trace['confidence_evolution']
            if evolution:
                positions = [e['position'] for e in evolution]
                confidences = [e['tail_confidence'] for e in evolution]
                color = cmap(0.3 + 0.7 * (i / max(1, len(correct_traces))))
                ax2.plot(positions, confidences, color=color, alpha=0.7, linewidth=1.5,
                        label=f"Trace {trace['trace_id']} (ans={trace['answer']})")

        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Tail Confidence')
        ax2.set_title(f'Correct Traces Only (n={len(correct_traces)})')
        ax2.grid(True, alpha=0.3)
        if len(correct_traces) <= 10:
            ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No correct traces', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Correct Traces Only (n=0)')

    # Plot 3: Incorrect traces only
    ax3 = axes[1, 0]
    if incorrect_traces:
        cmap = cm.get_cmap('Reds')
        for i, trace in enumerate(incorrect_traces):
            evolution = trace['confidence_evolution']
            if evolution:
                positions = [e['position'] for e in evolution]
                confidences = [e['tail_confidence'] for e in evolution]
                color = cmap(0.3 + 0.7 * (i / max(1, len(incorrect_traces))))
                ax3.plot(positions, confidences, color=color, alpha=0.7, linewidth=1.5,
                        label=f"Trace {trace['trace_id']} (ans={trace['answer']})")

        ax3.set_xlabel('Token Position')
        ax3.set_ylabel('Tail Confidence')
        ax3.set_title(f'Incorrect Traces Only (n={len(incorrect_traces)})')
        ax3.grid(True, alpha=0.3)
        if len(incorrect_traces) <= 10:
            ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No incorrect traces', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Incorrect Traces Only (n=0)')

    # Plot 4: Final confidence distribution
    ax4 = axes[1, 1]
    correct_final = [t['final_tail_confidence'] for t in correct_traces]
    incorrect_final = [t['final_tail_confidence'] for t in incorrect_traces]

    bins = np.linspace(
        min([t['final_tail_confidence'] for t in traces]),
        max([t['final_tail_confidence'] for t in traces]),
        20
    )

    if correct_final:
        ax4.hist(correct_final, bins=bins, alpha=0.6, color='green', label='Correct', edgecolor='black')
    if incorrect_final:
        ax4.hist(incorrect_final, bins=bins, alpha=0.6, color='red', label='Incorrect', edgecolor='black')

    ax4.set_xlabel('Final Tail Confidence')
    ax4.set_ylabel('Number of Traces')
    ax4.set_title('Distribution of Final Tail Confidence')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    stats_text = f"""
    Ground Truth: {data['ground_truth']}
    Final Answer: {data['final_answer']}
    Correct Traces: {len(correct_traces)}/{len(traces)}

    Correct Mean Final Conf: {np.mean(correct_final):.3f if correct_final else 'N/A'}
    Incorrect Mean Final Conf: {np.mean(incorrect_final):.3f if incorrect_final else 'N/A'}

    Total Tokens: {data['statistics']['total_tokens']:,}
    Avg Tokens/Trace: {data['statistics']['avg_tokens_per_trace']:.1f}
    Generation Time: {data['statistics']['generation_time']:.1f}s
    """

    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nMatplotlib plot saved to: {output_path}")

    # Also save as high-res version
    highres_path = output_path.replace('.png', '_highres.png')
    plt.savefig(highres_path, dpi=300, bbox_inches='tight')
    print(f"High-res version saved to: {highres_path}")

    plt.close()


def plot_confidence_evolution_ascii(data: Dict[str, Any]):
    """Create ASCII visualization of confidence evolution"""
    print("\n" + "="*80)
    print("ASCII CONFIDENCE EVOLUTION VISUALIZATION")
    print("="*80)

    traces = data['traces']
    if not traces:
        print("No traces to plot")
        return

    correct_traces = [t for t in traces if t['is_correct']]
    incorrect_traces = [t for t in traces if not t['is_correct']]

    print(f"\nQuestion: {data['question'][:100]}...")
    print(f"Ground Truth: {data['ground_truth']}")
    print(f"Final Answer: {data['final_answer']}")
    print(f"\nCorrect Traces: {len(correct_traces)}/{len(traces)}")

    # Show confidence evolution for a few representative traces
    print("\n" + "-"*80)
    print("REPRESENTATIVE TRACE CONFIDENCE EVOLUTION")
    print("-"*80)

    # Show up to 5 correct and 5 incorrect traces
    selected_traces = correct_traces[:5] + incorrect_traces[:5]

    for trace in selected_traces:
        evolution = trace['confidence_evolution']
        if not evolution:
            continue

        print(f"\nTrace {trace['trace_id']} ({'✓ CORRECT' if trace['is_correct'] else '✗ INCORRECT'})")
        print(f"  Answer: {trace['answer']}")
        print(f"  Tokens: {trace['num_tokens']}")
        print(f"  Final Tail Confidence: {trace['final_tail_confidence']:.3f}")

        # Create ASCII sparkline
        confidences = [e['tail_confidence'] for e in evolution]
        positions = [e['position'] for e in evolution]

        if len(confidences) > 1:
            # Normalize to 0-10 range for ASCII
            min_conf = min(confidences)
            max_conf = max(confidences)
            conf_range = max_conf - min_conf if max_conf > min_conf else 1

            normalized = [int((c - min_conf) / conf_range * 10) for c in confidences]

            # Create sparkline
            chars = ' ▁▂▃▄▅▆▇█'
            sparkline = ''.join(chars[min(n, 8)] for n in normalized)

            print(f"  Evolution: {sparkline}")
            print(f"  Range: [{min_conf:.3f}, {max_conf:.3f}]")

    # Summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS")
    print("-"*80)

    if correct_traces:
        correct_final = [t['final_tail_confidence'] for t in correct_traces]
        print(f"\nCorrect Traces (n={len(correct_traces)}):")
        print(f"  Mean final confidence: {np.mean(correct_final):.3f}")
        print(f"  Median final confidence: {np.median(correct_final):.3f}")
        print(f"  Std dev: {np.std(correct_final):.3f}")

    if incorrect_traces:
        incorrect_final = [t['final_tail_confidence'] for t in incorrect_traces]
        print(f"\nIncorrect Traces (n={len(incorrect_traces)}):")
        print(f"  Mean final confidence: {np.mean(incorrect_final):.3f}")
        print(f"  Median final confidence: {np.median(incorrect_final):.3f}")
        print(f"  Std dev: {np.std(incorrect_final):.3f}")

    if correct_traces and incorrect_traces:
        correct_mean = np.mean([t['final_tail_confidence'] for t in correct_traces])
        incorrect_mean = np.mean([t['final_tail_confidence'] for t in incorrect_traces])
        print(f"\nDifference (Correct - Incorrect): {correct_mean - incorrect_mean:+.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize trace confidence evolution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode selection
    parser.add_argument('--load', type=str, default=None,
                       help='Load and visualize from saved pickle file')

    # Generation parameters (if not loading)
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                       help='Model to use')
    parser.add_argument('--model_type', type=str, default="deepseek",
                       help='Model type for prompt formatting')
    parser.add_argument('--dataset', type=str, default='AIME2025-I',
                       choices=['AIME2025-I', 'AIME2025-II'],
                       help='Dataset to use')
    parser.add_argument('--qid', type=int, default=0,
                       help='Question ID to visualize')
    parser.add_argument('--num_traces', type=int, default=16,
                       help='Number of traces to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                       help='Number of GPUs')

    # Confidence tracking parameters
    parser.add_argument('--tail_size', type=int, default=2048,
                       help='Size of tail window for confidence computation')
    parser.add_argument('--step_size', type=int, default=100,
                       help='Sample confidence every N tokens')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='outputs_sc',
                       help='Output directory')
    parser.add_argument('--no_plot', action='store_true',
                       help='Skip matplotlib plots (ASCII only)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load or generate data
    if args.load:
        print(f"Loading data from: {args.load}")
        with open(args.load, 'rb') as f:
            data = pickle.load(f)
    else:
        # Setup GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Load dataset
        print(f"\nLoading {args.dataset}...")
        dataset = load_dataset("opencompass/AIME2025", name=args.dataset, split="test")

        if args.qid >= len(dataset):
            raise ValueError(f"Question ID {args.qid} out of range (0-{len(dataset)-1})")

        question_data = dataset[args.qid]
        question = question_data['question']
        ground_truth = str(question_data.get('answer', '')).strip()

        print(f"\nQuestion {args.qid}: {question[:150]}...")
        print(f"Ground Truth: {ground_truth}")

        # Initialize model
        print(f"\nInitializing model...")
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
            top_p=1.0,
            top_k=40,
            max_tokens=130000,
            logprobs=20,
        )

        # Generate and track
        data = generate_and_track_traces(
            deep_llm=deep_llm,
            question=question,
            ground_truth=ground_truth,
            num_traces=args.num_traces,
            sampling_params=sampling_params,
            model_type=args.model_type,
            tail_size=args.tail_size,
            step_size=args.step_size
        )

        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pickle_path = os.path.join(args.output_dir, f'trace_confidence_qid{args.qid}_{timestamp}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"\nData saved to: {pickle_path}")

        # Also save JSON (without full evolution for readability)
        json_data = {
            'question': data['question'],
            'ground_truth': data['ground_truth'],
            'final_answer': data['final_answer'],
            'voting_results': data['voting_results'],
            'num_traces': data['num_traces'],
            'traces_summary': [
                {
                    'trace_id': t['trace_id'],
                    'answer': t['answer'],
                    'is_correct': t['is_correct'],
                    'num_tokens': t['num_tokens'],
                    'final_tail_confidence': t['final_tail_confidence']
                }
                for t in data['traces']
            ],
            'statistics': data['statistics']
        }
        json_path = os.path.join(args.output_dir, f'trace_confidence_qid{args.qid}_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Summary saved to: {json_path}")

    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # ASCII plot (always)
    plot_confidence_evolution_ascii(data)

    # Matplotlib plot (if available and not disabled)
    if not args.no_plot and MATPLOTLIB_AVAILABLE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(args.output_dir, f'trace_confidence_plot_{timestamp}.png')
        plot_confidence_evolution_matplotlib(data, plot_path)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
