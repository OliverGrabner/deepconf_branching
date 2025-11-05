"""
Analysis script for Traditional Self-Consistency results

Usage:
    python analyze_sc_results.py outputs_sc/traditional_sc_aime25_detailed_TIMESTAMP.json
"""

import json
import argparse
import pandas as pd
import numpy as np
from collections import Counter


def analyze_vote_consensus(results):
    """Analyze the relationship between vote consensus and correctness"""
    print("\n" + "="*80)
    print("VOTE CONSENSUS ANALYSIS")
    print("="*80)

    consensus_correct = []
    consensus_incorrect = []

    for dataset_name, questions in results.items():
        for q in questions:
            if not q['vote_distribution']:
                continue

            # Calculate consensus: max votes / total votes
            total_votes = sum(q['vote_distribution'].values())
            max_votes = max(q['vote_distribution'].values())
            consensus = max_votes / total_votes if total_votes > 0 else 0

            if q['is_correct']:
                consensus_correct.append(consensus)
            else:
                consensus_incorrect.append(consensus)

    if consensus_correct:
        print(f"\nCorrect answers:")
        print(f"  Mean consensus: {np.mean(consensus_correct):.1%}")
        print(f"  Median consensus: {np.median(consensus_correct):.1%}")
        print(f"  Min/Max consensus: {np.min(consensus_correct):.1%} / {np.max(consensus_correct):.1%}")

    if consensus_incorrect:
        print(f"\nIncorrect answers:")
        print(f"  Mean consensus: {np.mean(consensus_incorrect):.1%}")
        print(f"  Median consensus: {np.median(consensus_incorrect):.1%}")
        print(f"  Min/Max consensus: {np.min(consensus_incorrect):.1%} / {np.max(consensus_incorrect):.1%}")

    if consensus_correct and consensus_incorrect:
        print(f"\nInsight: Correct answers had {np.mean(consensus_correct) - np.mean(consensus_incorrect):.1%} higher consensus on average")


def analyze_individual_vs_voting(results):
    """Compare individual trace accuracy with voting accuracy"""
    print("\n" + "="*80)
    print("INDIVIDUAL TRACE vs VOTING ACCURACY")
    print("="*80)

    for dataset_name, questions in results.items():
        print(f"\n{dataset_name}:")

        individual_accs = [q['individual_trace_accuracy'] for q in questions]
        voting_correct = [q['is_correct'] for q in questions]

        avg_individual = np.mean(individual_accs)
        voting_accuracy = np.mean(voting_correct)
        improvement = voting_accuracy - avg_individual

        print(f"  Avg individual trace accuracy: {avg_individual:.1%}")
        print(f"  Voting accuracy: {voting_accuracy:.1%}")
        print(f"  Improvement from voting: {improvement:+.1%}")

        # Find questions where voting helped
        helped = sum(1 for q in questions
                    if q['is_correct'] and q['individual_trace_accuracy'] < 0.5)
        print(f"  Questions where voting saved us (correct despite <50% individual): {helped}")

        # Find questions where voting failed
        failed = sum(1 for q in questions
                    if not q['is_correct'] and q['individual_trace_accuracy'] > 0.5)
        print(f"  Questions where voting failed us (wrong despite >50% individual): {failed}")


def analyze_answer_diversity(results):
    """Analyze diversity of answers across traces"""
    print("\n" + "="*80)
    print("ANSWER DIVERSITY ANALYSIS")
    print("="*80)

    for dataset_name, questions in results.items():
        print(f"\n{dataset_name}:")

        diversity_scores = []
        for q in questions:
            if q['vote_distribution']:
                num_unique_answers = len(q['vote_distribution'])
                diversity_scores.append(num_unique_answers)

        if diversity_scores:
            print(f"  Avg unique answers per question: {np.mean(diversity_scores):.1f}")
            print(f"  Min/Max unique answers: {np.min(diversity_scores)} / {np.max(diversity_scores)}")

            # Distribution of diversity
            diversity_counts = Counter(diversity_scores)
            print(f"  Distribution:")
            for num_answers, count in sorted(diversity_counts.items()):
                print(f"    {num_answers} unique answers: {count} questions")


def find_interesting_cases(results):
    """Find interesting success and failure cases"""
    print("\n" + "="*80)
    print("INTERESTING CASES")
    print("="*80)

    all_questions = []
    for dataset_name, questions in results.items():
        for i, q in enumerate(questions):
            q['dataset'] = dataset_name
            q['qid'] = i
            all_questions.append(q)

    # High consensus successes
    print("\nHigh Consensus Successes (>90% agreement, correct):")
    high_consensus_correct = []
    for q in all_questions:
        if q['is_correct'] and q['vote_distribution']:
            total = sum(q['vote_distribution'].values())
            max_votes = max(q['vote_distribution'].values())
            consensus = max_votes / total if total > 0 else 0
            if consensus > 0.9:
                high_consensus_correct.append((q, consensus))

    for q, consensus in sorted(high_consensus_correct, key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {q['dataset']} Q{q['qid']}: {consensus:.1%} consensus")
        print(f"    Answer: {q['voted_answer']}")
        print(f"    Distribution: {q['vote_distribution']}")

    # Low consensus successes (where voting really helped)
    print("\nLow Consensus Successes (<60% agreement, correct - voting really helped!):")
    low_consensus_correct = []
    for q in all_questions:
        if q['is_correct'] and q['vote_distribution']:
            total = sum(q['vote_distribution'].values())
            max_votes = max(q['vote_distribution'].values())
            consensus = max_votes / total if total > 0 else 0
            if consensus < 0.6:
                low_consensus_correct.append((q, consensus))

    for q, consensus in sorted(low_consensus_correct, key=lambda x: x[1])[:3]:
        print(f"  {q['dataset']} Q{q['qid']}: {consensus:.1%} consensus")
        print(f"    Answer: {q['voted_answer']} (ground truth: {q['ground_truth']})")
        print(f"    Distribution: {q['vote_distribution']}")

    # High consensus failures (model was confidently wrong)
    print("\nHigh Consensus Failures (>80% agreement, wrong - model was confidently wrong!):")
    high_consensus_wrong = []
    for q in all_questions:
        if not q['is_correct'] and q['vote_distribution']:
            total = sum(q['vote_distribution'].values())
            max_votes = max(q['vote_distribution'].values())
            consensus = max_votes / total if total > 0 else 0
            if consensus > 0.8:
                high_consensus_wrong.append((q, consensus))

    for q, consensus in sorted(high_consensus_wrong, key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {q['dataset']} Q{q['qid']}: {consensus:.1%} consensus")
        print(f"    Voted: {q['voted_answer']}, Ground truth: {q['ground_truth']}")
        print(f"    Distribution: {q['vote_distribution']}")


def generate_dataframe_summary(results):
    """Create pandas DataFrame for easy analysis"""
    print("\n" + "="*80)
    print("DATAFRAME SUMMARY")
    print("="*80)

    rows = []
    for dataset_name, questions in results.items():
        for i, q in enumerate(questions):
            if q['vote_distribution']:
                total_votes = sum(q['vote_distribution'].values())
                max_votes = max(q['vote_distribution'].values())
                consensus = max_votes / total_votes if total_votes > 0 else 0
                num_unique = len(q['vote_distribution'])
            else:
                consensus = 0
                num_unique = 0

            rows.append({
                'dataset': dataset_name,
                'qid': i,
                'is_correct': q['is_correct'],
                'individual_accuracy': q['individual_trace_accuracy'],
                'consensus': consensus,
                'num_unique_answers': num_unique,
                'num_valid_traces': q['num_valid_traces'],
                'total_tokens': q['statistics']['total_tokens'],
                'total_time': q['statistics']['total_time']
            })

    df = pd.DataFrame(rows)

    print("\nOverall Statistics:")
    print(df[['is_correct', 'individual_accuracy', 'consensus', 'num_unique_answers']].describe())

    print("\nCorrelations:")
    print(df[['is_correct', 'individual_accuracy', 'consensus', 'num_unique_answers']].corr())

    return df


def main():
    parser = argparse.ArgumentParser(description='Analyze Traditional SC results')
    parser.add_argument('results_file', type=str,
                       help='Path to detailed results JSON file')

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_file}...")
    with open(args.results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data['metadata']
    results = data['results']
    summary = data['summary']

    # Print metadata
    print("\n" + "="*80)
    print("EXPERIMENT METADATA")
    print("="*80)
    print(f"Model: {metadata['model']}")
    print(f"Number of traces: {metadata['num_traces']}")
    print(f"Temperature: {metadata['temperature']}")
    print(f"Timestamp: {metadata['timestamp']}")

    # Print summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    overall = summary['overall']
    print(f"Total questions: {overall['num_questions']}")
    print(f"Correct: {overall['num_correct']}/{overall['num_questions']} ({overall['accuracy']:.1%})")
    print(f"Total tokens: {overall['total_tokens']:,}")
    print(f"Total time: {overall['total_time']:.2f}s ({overall['total_time']/60:.1f} minutes)")

    # Run analyses
    analyze_individual_vs_voting(results)
    analyze_vote_consensus(results)
    analyze_answer_diversity(results)
    find_interesting_cases(results)
    df = generate_dataframe_summary(results)

    # Save DataFrame
    output_csv = args.results_file.replace('.json', '_analysis.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nAnalysis DataFrame saved to: {output_csv}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
