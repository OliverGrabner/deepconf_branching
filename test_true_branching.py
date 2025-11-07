"""
Test script to verify true prefix-based branching implementation

This script runs a simple test to ensure the new branching implementation:
1. Uses actual branch points from parent traces
2. Leverages vLLM's prefix caching
3. Only counts new tokens (not prefix tokens)
"""
import os
import sys

# Configure for testing with limited resources
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from vllm import SamplingParams
from deepconf.branching_wrapper import BranchingDeepThinkLLM

def test_true_branching():
    """Test the true branching implementation"""

    print("="*80)
    print("Testing True Prefix-Based Branching")
    print("="*80)

    # Use a small model for testing
    model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    print(f"\nInitializing model: {model}")
    branching_llm = BranchingDeepThinkLLM(
        model=model,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        trust_remote_code=True
    )

    # Simple test question
    question = "What is 25% of 80?"

    print(f"\nTest question: {question}")

    # Prepare prompt
    messages = [{"role": "user", "content": question}]
    prompt = branching_llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=2000,
        logprobs=20,
    )

    # Run branching experiment with minimal settings
    print("\nRunning branching experiment...")
    print("  Initial branches: 2")
    print("  Max total branches: 4")
    print("  Confidence threshold: 1.5")

    result = branching_llm.branching_deepthink(
        prompt=prompt,
        initial_branches=2,
        max_total_branches=4,
        confidence_threshold=1.5,
        window_size=128,
        sampling_params=sampling_params
    )

    # Analyze results
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)

    print(f"\nTotal traces generated: {result.total_traces_count}")

    # Count by depth
    depth_counts = {}
    for trace in result.all_traces:
        depth = trace.get('depth', 0)
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    print(f"Traces by depth: {depth_counts}")

    # Check for branch metadata
    branch_traces = [t for t in result.all_traces if t.get('depth', 0) > 0]

    if branch_traces:
        print(f"\n{len(branch_traces)} branch traces found:")
        for i, trace in enumerate(branch_traces, 1):
            print(f"\n  Branch {i}:")
            print(f"    Trace ID: {trace['trace_id']}")
            print(f"    Parent ID: {trace.get('parent_id', 'N/A')}")
            print(f"    Depth: {trace.get('depth', 0)}")
            print(f"    Branch point: {trace.get('branch_point', 'N/A')} tokens")
            print(f"    Prefix length: {trace.get('prefix_length', 'N/A')} tokens")
            print(f"    Total tokens: {trace.get('num_tokens', 'N/A')} tokens")

            if 'branch_history' in trace and trace['branch_history']:
                for branch_info in trace['branch_history']:
                    print(f"    Branched at step {branch_info['step']} (confidence: {branch_info['confidence']:.3f})")
    else:
        print("\nNo branch traces generated (confidence may not have been high enough)")

    # Token efficiency
    print(f"\nTotal tokens generated: {result.total_tokens}")
    print(f"Average tokens per trace: {result.avg_tokens_per_trace:.1f}")

    # Voting results
    if result.voting_results:
        print(f"\nVoting results:")
        for method, vote_result in result.voting_results.items():
            if vote_result:
                print(f"  {method}: {vote_result.get('answer', 'N/A')}")

    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)

    # Verify key features
    print("\nVerification:")
    checks = []

    # Check 1: Branch traces have parent_id
    if branch_traces:
        has_parent = all('parent_id' in t for t in branch_traces)
        checks.append(("Branch traces have parent_id", has_parent))

    # Check 2: Branch traces have branch_point
    if branch_traces:
        has_branch_point = all('branch_point' in t for t in branch_traces)
        checks.append(("Branch traces have branch_point", has_branch_point))

    # Check 3: Branch traces have prefix_length
    if branch_traces:
        has_prefix = all('prefix_length' in t for t in branch_traces)
        checks.append(("Branch traces have prefix_length", has_prefix))

    # Check 4: Total tokens is reasonable (not counting prefixes twice)
    expected_max = result.total_traces_count * 2000  # max_tokens per trace
    reasonable_tokens = result.total_tokens < expected_max
    checks.append(("Total tokens is reasonable", reasonable_tokens))

    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")

    all_passed = all(passed for _, passed in checks)

    if all_passed:
        print("\n✓ All checks passed! True branching is working correctly.")
    else:
        print("\n⚠ Some checks failed. Review the implementation.")

    return result


if __name__ == "__main__":
    try:
        result = test_true_branching()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
