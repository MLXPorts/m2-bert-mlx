#!/usr/bin/env python
"""
Jira data loading for M2-BERT training

Loads phrases.pkl and converts to sentence_transformers format
"""

import pickle
import random
from typing import List, Tuple
from sentence_transformers import InputExample


def gather_jira_training_examples(
    phrases_pkl_path='../datafiles/phrases.pkl',
    eval_split=0.2,
    random_seed=42
) -> Tuple[List[InputExample], List[InputExample]]:
    """
    Load Jira data and convert to sentence_transformers format

    Args:
        phrases_pkl_path: Path to phrases.pkl file
        eval_split: Fraction of data to use for evaluation (0.0 to 1.0)
        random_seed: Random seed for reproducible splits

    Returns:
        (train_examples, eval_examples): Tuple of training and evaluation examples

    Format:
        InputExample(texts=[problem, solution])
        - problem: The Jira issue problem description (query)
        - solution: The Jira issue solution (positive passage)
    """
    print(f"Loading Jira data from: {phrases_pkl_path}")

    # Load phrases
    with open(phrases_pkl_path, 'rb') as f:
        phrases = pickle.load(f)

    print(f"Loaded {len(phrases)} Jira items")

    # Create (problem, solution) pairs
    all_examples = []
    skipped = 0

    for item in phrases:
        problem = item.get('problem', '').strip()
        solution = item.get('solution', '').strip()

        if problem and solution:
            # Create InputExample with (query, positive_passage)
            all_examples.append(
                InputExample(texts=[problem, solution])
            )
        else:
            skipped += 1

    print(f"Created {len(all_examples)} training pairs")
    if skipped > 0:
        print(f"Skipped {skipped} items (missing problem or solution)")

    # Shuffle with fixed seed for reproducibility
    random.Random(random_seed).shuffle(all_examples)

    # Split train/eval
    split_idx = int(len(all_examples) * (1 - eval_split))
    train_examples = all_examples[:split_idx]
    eval_examples = all_examples[split_idx:]

    print(f"Train: {len(train_examples)} examples")
    print(f"Eval: {len(eval_examples)} examples")

    # Print first example for verification
    if len(train_examples) > 0:
        print("\n" + "="*70)
        print("First training example:")
        print("="*70)
        first = train_examples[0]
        print(f"Problem (query): {first.texts[0][:200]}")
        print(f"Solution (positive): {first.texts[1][:200]}")
        print("="*70 + "\n")

    return train_examples, eval_examples


def gather_jira_examples_with_hard_negatives(
    phrases_pkl_path='../datafiles/phrases.pkl',
    eval_split=0.2,
    negatives_per_query=4,
    random_seed=42
) -> Tuple[List[InputExample], List[InputExample]]:
    """
    Load Jira data with explicit hard negatives

    Args:
        phrases_pkl_path: Path to phrases.pkl file
        eval_split: Fraction of data to use for evaluation
        negatives_per_query: Number of hard negatives per query
        random_seed: Random seed for reproducibility

    Returns:
        (train_examples, eval_examples): Training and evaluation examples
        with hard negatives included

    Format:
        InputExample(texts=[problem, solution, neg1, neg2, ...])
        - problem: Query
        - solution: Correct answer (positive)
        - neg1, neg2, ...: Wrong solutions (hard negatives)
    """
    print(f"Loading Jira data with hard negatives from: {phrases_pkl_path}")

    # Load phrases
    with open(phrases_pkl_path, 'rb') as f:
        phrases = pickle.load(f)

    print(f"Loaded {len(phrases)} Jira items")

    # Extract all problems and solutions
    all_pairs = []
    for item in phrases:
        problem = item.get('problem', '').strip()
        solution = item.get('solution', '').strip()
        if problem and solution:
            all_pairs.append((problem, solution))

    print(f"Created {len(all_pairs)} valid pairs")

    # Create examples with hard negatives
    all_examples = []

    for i, (problem, solution) in enumerate(all_pairs):
        # Get hard negatives: solutions from other problems
        negative_solutions = [
            sol for j, (_, sol) in enumerate(all_pairs) if j != i
        ]

        # Sample random negatives (could be improved with BM25)
        if len(negative_solutions) >= negatives_per_query:
            sampled_negatives = random.Random(random_seed + i).sample(
                negative_solutions,
                negatives_per_query
            )

            # Create InputExample with query, positive, and negatives
            all_examples.append(
                InputExample(texts=[problem, solution] + sampled_negatives)
            )
        else:
            # Not enough negatives, use simple format
            all_examples.append(
                InputExample(texts=[problem, solution])
            )

    print(f"Created {len(all_examples)} examples with hard negatives")

    # Shuffle and split
    random.Random(random_seed).shuffle(all_examples)
    split_idx = int(len(all_examples) * (1 - eval_split))

    train_examples = all_examples[:split_idx]
    eval_examples = all_examples[split_idx:]

    print(f"Train: {len(train_examples)} examples")
    print(f"Eval: {len(eval_examples)} examples")

    return train_examples, eval_examples


if __name__ == '__main__':
    """Test data loading"""
    print("Testing Jira data loading...")
    print()

    # Test simple version
    train, eval_data = gather_jira_training_examples(
        phrases_pkl_path='/m2-bert/datafiles/phrases.pkl',
        eval_split=0.2
    )

    print("\n" + "="*70)
    print("SIMPLE FORMAT TEST")
    print("="*70)
    print(f"Total examples: {len(train) + len(eval_data)}")
    print(f"Training: {len(train)}")
    print(f"Evaluation: {len(eval_data)}")

    # Test with hard negatives
    print("\n" + "="*70)
    print("HARD NEGATIVES FORMAT TEST")
    print("="*70)

    train_hn, eval_hn = gather_jira_examples_with_hard_negatives(
        phrases_pkl_path='/m2-bert/datafiles/phrases.pkl',
        eval_split=0.2,
        negatives_per_query=3
    )

    if len(train_hn) > 0:
        print("\nFirst example with hard negatives:")
        first = train_hn[0]
        print(f"Number of texts: {len(first.texts)}")
        print(f"Query: {first.texts[0][:100]}...")
        print(f"Positive: {first.texts[1][:100]}...")
        if len(first.texts) > 2:
            print(f"Negative 1: {first.texts[2][:100]}...")

    print("\n✅ Data loading test complete!")
