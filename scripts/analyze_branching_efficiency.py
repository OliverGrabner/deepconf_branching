"""
Analyze branching efficiency from detailed results

This will:
1. Look at actual branching behavior (how many traces branched)
2. Calculate actual vs inherited tokens
3. Compare to what traditional SC would use
"""
import json
import sys

def analyze_branching_results(filepath):
    print("="*80)
    print("BRANCHING SC EFFICIENCY ANALYSIS")
    print("="*80)

    with open(filepath, 'r') as f:
        data = json.load(f)

    if 'results' not in data:
        print("ERROR: This appears to be a stats summary file, not a detailed results file")
        print("Need the detailed results file to analyze branching efficiency")
        sys.exit(1)

    results = data['results']

    total_questions = 0
    total_traces = 0
    total_original_traces = 0
    total_branched_traces = 0
    total_tokens = 0
    total_tokens_generated = 0
    total_branch_events = 0

    for dataset_name, questions in results.items():
        print(f"\n{dataset_name}:")

        for q_idx, question in enumerate(questions):
            total_questions += 1

            # Get statistics
            stats = question.get('statistics', {})
            q_total_tokens = stats.get('total_tokens', 0)
            q_tokens_generated = stats.get('total_tokens_generated', q_total_tokens)

            total_tokens += q_total_tokens
            total_tokens_generated += q_tokens_generated

            # Count traces
            valid_traces = question.get('valid_traces', [])
            full_traces = question.get('full_traces', [])
            traces_to_analyze = full_traces if full_traces else valid_traces

            num_traces = len(traces_to_analyze)
            total_traces += num_traces

            # Count original vs branched
            num_original = sum(1 for t in traces_to_analyze if t.get('parent_idx') is None)
            num_branched = num_traces - num_original

            total_original_traces += num_original
            total_branched_traces += num_branched

            # Count branch events
            branch_events = question.get('branch_events', [])
            total_branch_events += len(branch_events)

            if q_idx < 3:  # Print first 3 questions as examples
                print(f"  Q{q_idx}:")
                print(f"    Traces: {num_traces} ({num_original} original + {num_branched} branched)")
                print(f"    Tokens (total): {q_total_tokens:,}")
                print(f"    Tokens (generated): {q_tokens_generated:,}")
                print(f"    Inherited tokens: {q_total_tokens - q_tokens_generated:,}")
                print(f"    Branch events: {len(branch_events)}")

    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print('='*80)
    print(f"Total Questions: {total_questions}")
    print(f"Total Traces: {total_traces} ({total_traces/total_questions:.1f} per question)")
    print(f"  Original traces: {total_original_traces} ({total_original_traces/total_questions:.1f} per question)")
    print(f"  Branched traces: {total_branched_traces} ({total_branched_traces/total_questions:.1f} per question)")
    print(f"  Branch events: {total_branch_events} ({total_branch_events/total_questions:.1f} per question)")

    print(f"\nTOKEN USAGE:")
    print(f"  Total tokens (includes inherited): {total_tokens:,} ({total_tokens/total_questions:,.0f} per question)")
    print(f"  Tokens generated (actual new):     {total_tokens_generated:,} ({total_tokens_generated/total_questions:,.0f} per question)")
    print(f"  Inherited tokens (double-counted): {total_tokens - total_tokens_generated:,}")
    print(f"  Inflation rate: {(total_tokens/total_tokens_generated - 1)*100:.1f}%")

    print(f"\nCOMPARISON TO TRADITIONAL SC:")
    metadata = data.get('metadata', {})
    max_traces = metadata.get('max_traces', total_traces / total_questions)

    # Estimate what traditional SC would use
    avg_tokens_per_trace = total_tokens_generated / total_traces
    traditional_tokens_estimate = max_traces * avg_tokens_per_trace * total_questions

    print(f"  Avg tokens per trace: {avg_tokens_per_trace:.0f}")
    print(f"  Max traces (target): {max_traces}")
    print(f"  Estimated Traditional SC tokens: {traditional_tokens_estimate:,}")
    print(f"  Branching SC actual tokens: {total_tokens_generated:,}")
    print(f"  Savings: {traditional_tokens_estimate - total_tokens_generated:,} ({(1 - total_tokens_generated/traditional_tokens_estimate)*100:.1f}%)")

    print(f"\n{'='*80}")

def analyze_traditional_results(filepath):
    print("\n" + "="*80)
    print("TRADITIONAL SC ANALYSIS")
    print("="*80)

    with open(filepath, 'r') as f:
        data = json.load(f)

    if 'results' not in data:
        print("ERROR: This appears to be a stats summary file, not a detailed results file")
        sys.exit(1)

    results = data['results']

    total_questions = 0
    total_traces = 0
    total_tokens = 0

    for dataset_name, questions in results.items():
        for question in questions:
            total_questions += 1

            stats = question.get('statistics', {})
            total_tokens += stats.get('total_tokens', 0)

            valid_traces = question.get('valid_traces', [])
            total_traces += len(valid_traces)

    print(f"Total Questions: {total_questions}")
    print(f"Total Traces: {total_traces} ({total_traces/total_questions:.1f} per question)")
    print(f"Total Tokens: {total_tokens:,} ({total_tokens/total_questions:,.0f} per question)")
    print(f"Avg tokens per trace: {total_tokens/total_traces:.0f}")
    print("="*80)

# Analyze both
branching_file = "outputs/branching_sc_detailed_20251105_163947.json"
traditional_file = "outputs/traditional_sc_detailed_20251105_143014.json"

import os

if os.path.exists(branching_file):
    analyze_branching_results(branching_file)
else:
    print(f"Branching file not found: {branching_file}")
    print("Please provide the path to the branching detailed results file")

if os.path.exists(traditional_file):
    analyze_traditional_results(traditional_file)
else:
    print(f"\nTraditional file not found: {traditional_file}")
    print("Please provide the path to the traditional detailed results file")
