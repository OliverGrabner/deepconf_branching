#!/usr/bin/env python3
"""
Investigate why some traces hit the 64k token limit
"""

import json
import numpy as np

# Load the data file
with open('peak_branching_sc_detailed_20251118_202853.customization', 'r') as f:
    data = json.load(f)

print("="*80)
print("INVESTIGATING 64K TOKEN LIMIT TRACES")
print("="*80)

# Find traces near 64k limit (>60k tokens)
extreme_traces = []
for q_idx, question in enumerate(data['results']['gsm8k']):
    for trace in question['valid_traces']:
        tokens_gen = trace.get('tokens_generated', 0)
        if tokens_gen > 60000:
            trace_info = {
                'q_idx': q_idx,
                'question': question['question'][:100] + "...",
                'ground_truth': question['ground_truth'],
                'trace_idx': trace.get('trace_idx'),
                'stage': trace.get('stage', 0),
                'parent_idx': trace.get('parent_idx'),
                'branch_point': trace.get('branch_point_tokens', 0),
                'tokens_generated': tokens_gen,
                'num_tokens': trace.get('num_tokens', 0),
                'answer': trace.get('answer'),
                'is_correct': trace.get('is_correct', False)
            }
            extreme_traces.append(trace_info)

print(f"\nFound {len(extreme_traces)} traces with >60k tokens\n")

# Group by question to see patterns
questions_with_extremes = {}
for trace in extreme_traces:
    q_idx = trace['q_idx']
    if q_idx not in questions_with_extremes:
        questions_with_extremes[q_idx] = []
    questions_with_extremes[q_idx].append(trace)

# Analyze each question with extreme traces
for q_idx in sorted(questions_with_extremes.keys())[:5]:  # Show first 5
    traces = questions_with_extremes[q_idx]
    print(f"\n{'='*60}")
    print(f"Question {q_idx}:")
    print(f"Text: {traces[0]['question']}")
    print(f"Ground truth: {traces[0]['ground_truth']}")
    print(f"\nExtreme traces in this question: {len(traces)}")

    for trace in traces:
        print(f"\n  Trace {trace['trace_idx']} (Stage {trace['stage']}):")
        print(f"    Parent: {trace['parent_idx']}")
        print(f"    Branch point: {trace['branch_point']} tokens")
        print(f"    Generated: {trace['tokens_generated']:,} NEW tokens")
        print(f"    Total: {trace['num_tokens']:,} tokens")
        print(f"    Answer: {trace['answer']}")
        print(f"    Correct: {trace['is_correct']}")

# Look for patterns
print("\n" + "="*80)
print("PATTERN ANALYSIS:")
print("="*80)

# Branch points
branch_points = [t['branch_point'] for t in extreme_traces if t['branch_point'] > 0]
if branch_points:
    print(f"\nBranch points for extreme traces:")
    print(f"  Mean: {np.mean(branch_points):.0f} tokens")
    print(f"  Median: {np.median(branch_points):.0f} tokens")
    print(f"  Min: {np.min(branch_points):.0f} tokens")
    print(f"  Max: {np.max(branch_points):.0f} tokens")

# Stage distribution
stage_counts = {}
for trace in extreme_traces:
    stage = trace['stage']
    stage_counts[stage] = stage_counts.get(stage, 0) + 1

print(f"\nDistribution by stage:")
for stage, count in sorted(stage_counts.items()):
    pct = count / len(extreme_traces) * 100
    print(f"  Stage {stage}: {count} traces ({pct:.1f}%)")

# Correctness
correct = sum(1 for t in extreme_traces if t['is_correct'])
print(f"\nCorrectness of extreme traces:")
print(f"  Correct: {correct}/{len(extreme_traces)} ({correct/len(extreme_traces)*100:.1f}%)")

# Check if answer was None
no_answer = sum(1 for t in extreme_traces if t['answer'] is None or t['answer'] == '')
print(f"  No answer extracted: {no_answer}/{len(extreme_traces)}")

print("\n" + "="*80)
print("💡 HYPOTHESIS: Why traces explode to 64k tokens")
print("="*80)
print("\n1. EARLY BRANCHING + NO CONTEXT:")
print("   - When branching at tokens 200-500, the model has minimal problem context")
print("   - It essentially starts reasoning from scratch")

print("\n2. LOST/CONFUSED REASONING:")
print("   - Without strong context, the model may not understand what to solve")
print("   - Enters exploration mode, trying different approaches")

print("\n3. NO ANSWER CONVERGENCE:")
print("   - Model can't find a clear answer")
print("   - Keeps generating until hitting the 64k limit")

print("\n4. MISSING EARLY STOPPING:")
print("   - Early stopping sequences ('}\\n\\n', '}\\n') may not trigger")
print("   - Model doesn't use \\boxed{} format or uses it very late")

print("\n5. BRANCH FROM BRANCH AMPLIFICATION:")
print("   - Stage 2 branches (branch from branch) have even less context")
print("   - More likely to explode")

# Look at the actual early stopping configuration
metadata = data.get('metadata', {})
print(f"\n📝 Configuration check:")
print(f"  Max tokens setting: {metadata.get('max_tokens', 'Not set')}")
print(f"  Temperature: {metadata.get('temperature', 'Not set')}")

# Check wrapper.py settings
print("\n⚠️  The fix in wrapper.py sets:")
print("  - branch_params.max_tokens = max(1000, 64000 - len(branch_info['prompt_tokens']))")
print("  - branch_params.stop = ['}\\n\\n', '}\\n']")
print("\nBut clearly the 64k limit is being hit, suggesting:")
print("  1. The stop sequences aren't working")
print("  2. The model never generates \\boxed{} in these traces")
print("  3. OR the model generates it but continues anyway")