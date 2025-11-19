#!/usr/bin/env python3
"""
Deep dive into why Peak Branching is generating so many tokens
"""

import json
import numpy as np

# Load the data file
with open('peak_branching_sc_detailed_20251118_202853.customization', 'r') as f:
    data = json.load(f)

print("="*80)
print("DEEP DIVE: Why is Peak Branching using so many tokens?")
print("="*80)

# Look at first question in detail
question = data['results']['gsm8k'][0]
valid_traces = question['valid_traces']

# Separate by stage
initial_traces = [t for t in valid_traces if t.get('stage', 0) == 0]
stage1_traces = [t for t in valid_traces if t.get('stage', 0) == 1]
stage2_traces = [t for t in valid_traces if t.get('stage', 0) == 2]

print(f"\nQuestion 0 Analysis:")
print(f"Total traces: {len(valid_traces)}")
print(f"  Stage 0 (initial): {len(initial_traces)}")
print(f"  Stage 1: {len(stage1_traces)}")
print(f"  Stage 2: {len(stage2_traces)}")

# Analyze initial traces
print(f"\nInitial Traces (Stage 0):")
for i, trace in enumerate(initial_traces):
    tokens_gen = trace.get('tokens_generated', 0)
    num_tokens = trace.get('num_tokens', 0)
    print(f"  Trace {i}: {tokens_gen:,} tokens generated, {num_tokens:,} total tokens")

avg_initial = np.mean([t.get('tokens_generated', 0) for t in initial_traces])
print(f"  Average: {avg_initial:,.0f} tokens per initial trace")
print(f"  Total for initial: {sum(t.get('tokens_generated', 0) for t in initial_traces):,}")

# Analyze Stage 1 branches
print(f"\nStage 1 Branches:")
for i, trace in enumerate(stage1_traces[:5]):  # Show first 5
    tokens_gen = trace.get('tokens_generated', 0)
    num_tokens = trace.get('num_tokens', 0)
    branch_point = trace.get('branch_point_tokens', 0)
    print(f"  Branch {i}: {tokens_gen:,} NEW tokens, branched at token {branch_point}, total {num_tokens:,}")

avg_stage1 = np.mean([t.get('tokens_generated', 0) for t in stage1_traces])
print(f"  Average: {avg_stage1:,.0f} NEW tokens per Stage 1 branch")
print(f"  Total for Stage 1: {sum(t.get('tokens_generated', 0) for t in stage1_traces):,}")

# Analyze Stage 2 branches
print(f"\nStage 2 Branches:")
for i, trace in enumerate(stage2_traces[:5]):  # Show first 5
    tokens_gen = trace.get('tokens_generated', 0)
    num_tokens = trace.get('num_tokens', 0)
    branch_point = trace.get('branch_point_tokens', 0)
    print(f"  Branch {i}: {tokens_gen:,} NEW tokens, branched at token {branch_point}, total {num_tokens:,}")

avg_stage2 = np.mean([t.get('tokens_generated', 0) for t in stage2_traces])
print(f"  Average: {avg_stage2:,.0f} NEW tokens per Stage 2 branch")
print(f"  Total for Stage 2: {sum(t.get('tokens_generated', 0) for t in stage2_traces):,}")

# Look for anomalies
print(f"\n⚠️ ANOMALY CHECK:")
zero_token_traces = [t for t in valid_traces if t.get('tokens_generated', 0) == 0]
print(f"Traces with 0 tokens: {len(zero_token_traces)}")
for trace in zero_token_traces[:3]:
    print(f"  Stage {trace.get('stage')}, parent {trace.get('parent_idx')}, branch point {trace.get('branch_point_tokens')}")

huge_token_traces = [t for t in valid_traces if t.get('tokens_generated', 0) > 5000]
print(f"\nTraces with >5000 tokens: {len(huge_token_traces)}")
for trace in huge_token_traces:
    print(f"  Stage {trace.get('stage')}, generated {trace.get('tokens_generated'):,} tokens, branch point {trace.get('branch_point_tokens')}")

# Check peak branching config
peak_config = question.get('peak_branching_config', {})
peak_stats = question.get('peak_branching_stats', {})
print(f"\n📊 Peak Branching Configuration:")
for key, value in peak_config.items():
    print(f"  {key}: {value}")

print(f"\n📈 Peak Branching Statistics:")
print(f"  Total traces: {peak_stats.get('total_traces')}")
print(f"  Branching stages: {peak_stats.get('branching_stages')}")
print(f"  Total peaks found: {peak_stats.get('total_peaks_found')}")
print(f"  Branches created: {peak_stats.get('branches_created')}")
print(f"  Average branch point: {peak_stats.get('avg_branch_point', 0):.0f} tokens")

# The problem analysis
print("\n" + "="*80)
print("🔍 PROBLEM IDENTIFIED:")
print("="*80)
avg_branch_point = peak_stats.get('avg_branch_point', 0)
print(f"1. Average branch point is at token {avg_branch_point:.0f}")
print(f"2. Initial traces average {avg_initial:.0f} tokens")
print(f"3. Branches generate {avg_stage1:.0f} NEW tokens on average")
print(f"4. This means branches are occurring at ~{(avg_branch_point/avg_initial)*100:.0f}% completion")
print(f"5. So branches need to generate ~{100 - (avg_branch_point/avg_initial)*100:.0f}% of a full trace!")

print(f"\n💡 EXPECTED vs ACTUAL:")
print(f"Expected (if branching at 75%): ~{avg_initial * 0.25:.0f} tokens per branch")
print(f"Actual Stage 1 branches: {avg_stage1:.0f} tokens per branch")
print(f"That's {avg_stage1 / (avg_initial * 0.25):.1f}x more than expected!")

print("\n🎯 ROOT CAUSE:")
print("Peak detection (acceleration-based) is finding peaks TOO EARLY in traces!")
print("When you branch from 25% completion, you generate 75% new tokens.")
print("With 24 branches, this adds up to way more tokens than traditional SC.")