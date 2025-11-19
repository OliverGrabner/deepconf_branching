#!/usr/bin/env python3
"""
Analyze why stop sequences aren't working
"""

import json

# Load the data file
with open('peak_branching_sc_detailed_20251118_202853.customization', 'r') as f:
    data = json.load(f)

print("="*80)
print("ANALYZING STOP SEQUENCE FAILURE")
print("="*80)

# Check traces that hit the limit
extreme_traces_with_text = []
for q_idx, question in enumerate(data['results']['gsm8k'][:50]):  # Check first 50 questions
    for trace in question.get('full_traces', question.get('valid_traces', [])):
        tokens_gen = trace.get('tokens_generated', 0)
        if tokens_gen > 60000:
            # Try to get the answer and check if it exists
            answer = trace.get('answer')
            is_correct = trace.get('is_correct', False)

            print(f"\nQuestion {q_idx}, Trace {trace.get('trace_idx')}:")
            print(f"  Stage: {trace.get('stage', 0)}")
            print(f"  Branch point: {trace.get('branch_point_tokens', 0)} tokens")
            print(f"  Generated: {tokens_gen:,} NEW tokens")
            print(f"  Answer extracted: {answer}")
            print(f"  Is correct: {is_correct}")

            # The key insight: if answer was extracted, \boxed{} WAS generated
            # But generation continued anyway!
            if answer is not None and answer != '':
                print(f"  ⚠️ ANSWER WAS FOUND but generation continued!")
                print(f"  This means \\boxed{{}} was generated but stop sequence failed!")

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)

print("""
1. STOP SEQUENCES ARE NOT WORKING:
   - The stop sequences ["}\\n\\n", "}\\n"] are supposed to stop after \\boxed{answer}
   - But traces continue to 64k tokens even AFTER finding the answer

2. WHY STOP SEQUENCES MIGHT FAIL:
   a) The model uses a different format like \\boxed{answer}. (with period)
   b) The model adds explanation after \\boxed{answer}
   c) The model generates multiple \\boxed{} attempts
   d) There's extra whitespace or formatting

3. THE REAL PROBLEM:
   Line 736 in wrapper.py calculates:
   max_tokens = max(1000, 64000 - len(branch_info['prompt_tokens']))

   If branch_point is 500 tokens:
   max_tokens = max(1000, 64000 - 500) = 63,500 tokens!

   This allows branches to generate up to 63,500 NEW tokens!

4. THE FIX NEEDED:
   Should be something like:
   max_tokens = min(4000, 64000 - len(branch_info['prompt_tokens']))

   This would cap branches at 4000 tokens maximum.
""")

print("\n🔥 ROOT CAUSE IDENTIFIED:")
print("The max() function should be min()!")
print("Currently: max(1000, 64000 - prefix) → allows up to 63k tokens")
print("Should be: min(4000, 64000 - prefix) → caps at 4k tokens")
print("\nThis is why traces explode to 64k - they're ALLOWED to!")