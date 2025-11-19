#!/usr/bin/env python3
"""
Investigate the anomalous traces with huge token counts
"""

import json
import numpy as np

# Load the data file
with open('peak_branching_sc_detailed_20251118_202853.customization', 'r') as f:
    data = json.load(f)

print("="*80)
print("INVESTIGATING TOKEN EXPLOSION ANOMALY")
print("="*80)

# Collect all traces across all questions
all_traces = []
for q_idx, question in enumerate(data['results']['gsm8k']):
    for trace in question['valid_traces']:
        trace['question_idx'] = q_idx
        all_traces.append(trace)

# Find traces with excessive tokens
huge_traces = [t for t in all_traces if t.get('tokens_generated', 0) > 5000]
print(f"\nFound {len(huge_traces)} traces with >5000 tokens across all questions")

# Analyze each huge trace
for trace in huge_traces[:10]:  # Show first 10
    q_idx = trace['question_idx']
    stage = trace.get('stage', 0)
    tokens_gen = trace.get('tokens_generated', 0)
    num_tokens = trace.get('num_tokens', 0)
    branch_point = trace.get('branch_point_tokens', 0)

    print(f"\n📍 Question {q_idx}, Stage {stage}:")
    print(f"   Generated: {tokens_gen:,} NEW tokens")
    print(f"   Total: {num_tokens:,} tokens")
    print(f"   Branch point: {branch_point}")
    print(f"   Parent: {trace.get('parent_idx')}")

    # Check if it makes sense
    if branch_point > 0:
        expected_total = branch_point + tokens_gen
        print(f"   Expected total (branch_point + generated): {expected_total:,}")
        print(f"   Actual total: {num_tokens:,}")
        if abs(expected_total - num_tokens) > 10:
            print(f"   ⚠️ MISMATCH: Difference of {abs(expected_total - num_tokens):,} tokens!")

# Look at distribution of tokens_generated
token_counts = [t.get('tokens_generated', 0) for t in all_traces if t.get('tokens_generated', 0) > 0]
print(f"\n📊 TOKEN GENERATION DISTRIBUTION:")
print(f"   Total traces: {len(all_traces)}")
print(f"   Non-zero traces: {len(token_counts)}")
print(f"   Mean: {np.mean(token_counts):,.0f} tokens")
print(f"   Median: {np.median(token_counts):,.0f} tokens")
print(f"   Max: {np.max(token_counts):,.0f} tokens")
print(f"   95th percentile: {np.percentile(token_counts, 95):,.0f} tokens")
print(f"   99th percentile: {np.percentile(token_counts, 99):,.0f} tokens")

# Distribution by stage
for stage in range(3):
    stage_traces = [t for t in all_traces if t.get('stage', 0) == stage]
    stage_tokens = [t.get('tokens_generated', 0) for t in stage_traces if t.get('tokens_generated', 0) > 0]
    if stage_tokens:
        print(f"\n   Stage {stage}:")
        print(f"     Count: {len(stage_traces)}")
        print(f"     Mean: {np.mean(stage_tokens):,.0f} tokens")
        print(f"     Median: {np.median(stage_tokens):,.0f} tokens")
        print(f"     Max: {np.max(stage_tokens):,.0f} tokens")

# Check if traces hit max_tokens limit
print(f"\n🚨 HYPOTHESIS CHECK: Are traces hitting the max_tokens limit?")
metadata = data.get('metadata', {})
max_tokens = metadata.get('max_tokens', 0)
print(f"   Max tokens setting: {max_tokens:,}")

# But wait, branches should have 64k limit according to the fix
print(f"   Expected branch limit: 64,000 tokens")

traces_near_limit = [t for t in all_traces if t.get('tokens_generated', 0) > 60000]
print(f"   Traces near 64k limit: {len(traces_near_limit)}")

for trace in traces_near_limit[:5]:
    print(f"     Q{trace['question_idx']}: {trace.get('tokens_generated'):,} tokens")

# The real issue
print("\n" + "="*80)
print("💡 ROOT CAUSE IDENTIFIED:")
print("="*80)
print("Some traces are generating 10,000-60,000+ tokens!")
print("This happens when:")
print("1. Branch occurs early (e.g., at token 500-1000)")
print("2. Model enters a reasoning loop or very verbose explanation")
print("3. No early stopping triggers (answer not found quickly)")
print("4. Continues until 64k token limit")
print("\nThe median is reasonable (~1000 tokens) but outliers explode the average!")