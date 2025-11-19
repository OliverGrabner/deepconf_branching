#!/usr/bin/env python3
"""
Analyze the peak branching tokens to understand the real NEW tokens vs total tokens
"""

import json
import numpy as np

# Load the data file
with open('peak_branching_sc_detailed_20251118_202853.customization', 'r') as f:
    data = json.load(f)

# Analyze each question
total_new_tokens_all = 0
total_tokens_all = 0
questions_analyzed = 0

print("="*80)
print("PEAK BRANCHING TOKEN ANALYSIS")
print("="*80)

for question_idx, question_result in enumerate(data['results']['gsm8k']):
    questions_analyzed += 1

    # Get the saved statistics (which has the bug)
    stats = question_result.get('statistics', {})
    saved_total_tokens = stats.get('total_tokens', 0)
    saved_total_tokens_generated = stats.get('total_tokens_generated', 0)  # This is WRONG (same as total_tokens)

    # Get peak_branching_stats (which should be correct)
    peak_stats = question_result.get('peak_branching_stats', {})
    peak_total_tokens_generated = peak_stats.get('total_tokens_generated', 0)
    peak_total_tokens_with_prefix = peak_stats.get('total_tokens_with_prefix', 0)

    # Calculate from valid_traces (ground truth)
    valid_traces = question_result.get('valid_traces', [])

    # Sum up NEW tokens from each trace
    calculated_new_tokens = sum(t.get('tokens_generated', 0) for t in valid_traces)

    # Sum up TOTAL tokens from each trace
    calculated_total_tokens = sum(t.get('num_tokens', 0) for t in valid_traces)

    # Track totals
    total_new_tokens_all += calculated_new_tokens
    total_tokens_all += calculated_total_tokens

    # Show details for first few questions
    if question_idx < 3:
        print(f"\nQuestion {question_idx}:")
        print(f"  Traces: {len(valid_traces)} total ({question_result['num_initial_traces']} initial, {question_result['num_branch_traces']} branches)")
        print(f"  From statistics (BUGGY):")
        print(f"    total_tokens: {saved_total_tokens:,}")
        print(f"    total_tokens_generated: {saved_total_tokens_generated:,} (WRONG - same as total_tokens!)")
        print(f"  From peak_branching_stats (CORRECT):")
        print(f"    total_tokens_generated: {peak_total_tokens_generated:,}")
        print(f"    total_tokens_with_prefix: {peak_total_tokens_with_prefix:,}")
        print(f"  Calculated from valid_traces:")
        print(f"    NEW tokens (sum of tokens_generated): {calculated_new_tokens:,}")
        print(f"    TOTAL tokens (sum of num_tokens): {calculated_total_tokens:,}")

        # Show breakdown by stage
        initial_traces = [t for t in valid_traces if t.get('stage', 0) == 0]
        branch_traces = [t for t in valid_traces if t.get('stage', 0) > 0]

        initial_new = sum(t.get('tokens_generated', 0) for t in initial_traces)
        branch_new = sum(t.get('tokens_generated', 0) for t in branch_traces)

        print(f"  Token breakdown:")
        print(f"    Initial traces ({len(initial_traces)}): {initial_new:,} new tokens")
        print(f"    Branch traces ({len(branch_traces)}): {branch_new:,} new tokens")

        # Check for anomalies
        max_tokens = max(t.get('tokens_generated', 0) for t in valid_traces)
        min_tokens = min(t.get('tokens_generated', 0) for t in valid_traces if t.get('tokens_generated', 0) > 0)
        zero_traces = sum(1 for t in valid_traces if t.get('tokens_generated', 0) == 0)

        print(f"  Anomalies:")
        print(f"    Max tokens in single trace: {max_tokens:,}")
        print(f"    Min tokens (non-zero): {min_tokens:,}")
        print(f"    Traces with 0 tokens: {zero_traces}")

print("\n" + "="*80)
print("SUMMARY ACROSS ALL QUESTIONS")
print("="*80)

avg_new_tokens = total_new_tokens_all / questions_analyzed if questions_analyzed > 0 else 0
avg_total_tokens = total_tokens_all / questions_analyzed if questions_analyzed > 0 else 0

print(f"Questions analyzed: {questions_analyzed}")
print(f"Average NEW tokens per question: {avg_new_tokens:,.0f}")
print(f"Average TOTAL tokens per question (with prefix): {avg_total_tokens:,.0f}")
print(f"Prefix reuse savings: {avg_total_tokens - avg_new_tokens:,.0f} tokens ({(1 - avg_new_tokens/avg_total_tokens)*100:.1f}%)")

print("\n" + "="*80)
print("COMPARISON WITH TRADITIONAL SC")
print("="*80)

# Assuming traditional SC with 8 traces of ~2500 tokens each
traditional_tokens = 8 * 2500  # Rough estimate
print(f"Traditional SC (8 traces × 2500 tokens): ~{traditional_tokens:,} tokens")
print(f"Peak Branching (actual NEW tokens): {avg_new_tokens:,.0f} tokens")
print(f"Real savings: {traditional_tokens - avg_new_tokens:,.0f} tokens ({(1 - avg_new_tokens/traditional_tokens)*100:.1f}%)")

print("\nBUT the chart was showing:")
print(f"Peak Branching (using buggy total_tokens): {avg_total_tokens:,.0f} tokens")
print(f"That's why it looked worse than Traditional!")