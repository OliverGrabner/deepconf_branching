"""
Integration utilities to bridge the standardized infrastructure with our branching implementation.

This module imports only the necessary functions from the added infrastructure
while preserving our existing branching logic.
"""

import re
import sys
import os
from typing import Optional

# Add the added folder to path to import utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'added'))

# Import only the answer extraction and comparison functions we need
from utils import (
    extract_answer as extract_answer_base,
    extract_answer_gsm8k,
    equal_func as equal_func_aime,
    equal_func_gsm8k
)

# Import dataset utilities
from experiment_utils import (
    load_dataset_by_name,
    get_question_and_ground_truth,
)


def extract_answer_robust(text: str, dataset_type: str = "aime") -> Optional[str]:
    """
    Robust answer extraction that handles multiple formats.

    Args:
        text: The model output text
        dataset_type: Either "aime" or "gsm8k"

    Returns:
        Extracted answer or None
    """
    if not text:
        return None

    if "gsm8k" in dataset_type.lower():
        return extract_answer_gsm8k(text)

    # For AIME, try the base extractor first
    answer = extract_answer_base(text)
    if answer:
        return answer

    # If base extractor fails, try additional patterns for AIME
    # These are common patterns in model outputs
    patterns = [
        r'(?:The answer is|Therefore,?|Thus,?|So,?|Hence,?)\s*[:=]?\s*(\d+)',
        r'Final answer\s*[:=]\s*(\d+)',
        r'^Answer\s*[:=]\s*(\d+)',
        r'^\$?(\d+)\$?$',  # Just a number (possibly in math mode)
    ]

    # Check last few lines for answer patterns
    lines = text.strip().split('\n')
    for line in reversed(lines[-10:]):
        line = line.strip()
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).strip()

    return None


def check_answer_equality(predicted: str, ground_truth: str, dataset_type: str = "aime") -> bool:
    """
    Check if predicted answer equals ground truth.

    Args:
        predicted: The predicted answer
        ground_truth: The ground truth answer
        dataset_type: Either "aime" or "gsm8k"

    Returns:
        True if answers match, False otherwise
    """
    if predicted is None or ground_truth is None:
        return False

    if "gsm8k" in dataset_type.lower():
        return equal_func_gsm8k(predicted, ground_truth)
    else:
        return equal_func_aime(predicted, ground_truth)


# Re-export the functions we want to use
__all__ = [
    'extract_answer_robust',
    'check_answer_equality',
    'load_dataset_by_name',
    'get_question_and_ground_truth',
    'extract_answer_gsm8k',
    'extract_answer_base',
]