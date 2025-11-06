"""
Quick diagnostic script to check token counting in experiment results
"""
import json
import sys

def check_file(filepath, name):
    print(f"\n{'='*80}")
    print(f"{name}: {filepath}")
    print('='*80)

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Check if it's a stats summary or detailed results
    if 'overall' in data:
        print("Format: Stats Summary")
        overall = data['overall']
        by_dataset = data['by_dataset']

        print(f"\nOVERALL:")
        print(f"  Questions: {overall.get('num_questions', 'N/A')}")
        print(f"  Accuracy: {overall.get('accuracy', 0)*100:.2f}%")
        print(f"  Total Tokens: {overall.get('total_tokens', 'N/A'):,}")
        print(f"  Avg Tokens/Question: {overall.get('avg_tokens_per_question', 'N/A'):.0f}")
        print(f"  Total Branch Events: {overall.get('total_branch_events', 'N/A')}")

        print(f"\nBY DATASET:")
        for ds_name, ds_stats in by_dataset.items():
            print(f"  {ds_name}:")
            print(f"    Questions: {ds_stats.get('num_questions', 'N/A')}")
            print(f"    Total Tokens: {ds_stats.get('total_tokens', 'N/A'):,}")
            print(f"    Avg Tokens/Question: {ds_stats.get('avg_tokens_per_question', 'N/A'):.0f}")
            print(f"    Throughput: {ds_stats.get('throughput_tokens_per_sec', 'N/A'):.1f} tok/s")

    elif 'results' in data:
        print("Format: Detailed Results")
        results = data['results']

        total_questions = 0
        total_tokens = 0
        total_tokens_generated = 0

        for ds_name, questions in results.items():
            print(f"\n{ds_name}:")
            print(f"  Questions: {len(questions)}")

            ds_tokens = 0
            ds_tokens_gen = 0

            for q in questions:
                stats = q.get('statistics', {})
                ds_tokens += stats.get('total_tokens', 0)
                ds_tokens_gen += stats.get('total_tokens_generated', stats.get('total_tokens', 0))
                total_questions += 1

            total_tokens += ds_tokens
            total_tokens_generated += ds_tokens_gen

            print(f"  Total Tokens: {ds_tokens:,}")
            print(f"  Total Tokens Generated: {ds_tokens_gen:,}")
            print(f"  Avg Tokens/Question: {ds_tokens/len(questions):.0f}")

        print(f"\nOVERALL:")
        print(f"  Total Questions: {total_questions}")
        print(f"  Total Tokens: {total_tokens:,}")
        print(f"  Total Tokens Generated: {total_tokens_generated:,}")
        print(f"  Difference: {total_tokens - total_tokens_generated:,} ({(total_tokens-total_tokens_generated)/total_tokens*100:.1f}%)")

    # Check metadata if available
    if 'metadata' in data:
        print(f"\nMETADATA:")
        meta = data['metadata']
        print(f"  Experiment Type: {meta.get('experiment_type', 'N/A')}")
        if meta.get('experiment_type') == 'traditional':
            print(f"  Num Traces: {meta.get('num_traces', 'N/A')}")
        else:
            print(f"  Start Traces: {meta.get('start_traces', 'N/A')}")
            print(f"  Max Traces: {meta.get('max_traces', 'N/A')}")
            print(f"  Selected Percent: {meta.get('selected_percent', 'N/A')}")

# Check both files
branching_file = "outputs/branching_sc_stats_20251105_163947.json"
traditional_file = "outputs/traditional_sc_stats_20251105_143014.json"

try:
    check_file(branching_file, "BRANCHING SC")
except Exception as e:
    print(f"Error reading branching file: {e}")

try:
    check_file(traditional_file, "TRADITIONAL SC")
except Exception as e:
    print(f"Error reading traditional file: {e}")

print("\n" + "="*80)
