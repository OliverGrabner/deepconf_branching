"""
Robust answer extraction and dataset utilities without external dependencies.
This provides the core functionality needed for multi-dataset support.
"""

import re
from typing import Optional, Tuple, List, Dict, Any
from datasets import load_dataset


def extract_answer_aime(text: str) -> Optional[str]:
    """Extract answer from AIME/math problem format."""
    if not text:
        return None

    # Try standard boxed format (with or without backslash)
    patterns = [
        r'\\boxed\{([^}]+)\}',  # \boxed{answer}
        r'boxed\{([^}]+)\}',     # boxed{answer}
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            answer = matches[-1].strip()
            # Clean up any LaTeX artifacts
            answer = answer.replace('$', '').strip()
            return answer

    # Try to find answer at the end of text
    lines = text.strip().split('\n')

    # Check last few lines for answer patterns
    for line in reversed(lines[-10:]):
        line = line.strip()

        answer_patterns = [
            r'(?:The answer is|Therefore,?|Thus,?|So,?|Hence,?)\s*[:=]?\s*(\d+)',
            r'Final answer\s*[:=]\s*(\d+)',
            r'^Answer\s*[:=]\s*(\d+)',
            r'^=\s*(\d+)$',
            r'^\$?(\d+)\$?$',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).strip()

    return None


def extract_answer_gsm8k(text: str) -> Optional[str]:
    """Extract numerical answer from GSM8k format text."""
    if not text:
        return None

    # Look for #### marker (standard GSM8k format)
    if "####" in text:
        parts = text.split("####")
        if len(parts) > 1:
            answer_text = parts[-1].strip()
            # Extract first number after ####
            numbers = re.findall(r'-?\d+\.?\d*', answer_text)
            if numbers:
                return numbers[0]

    # Fallback: extract last number in text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]

    return None


def extract_answer_robust(text: str, dataset_type: str = "aime") -> Optional[str]:
    """
    Robust answer extraction that handles multiple formats.

    Args:
        text: The model output text
        dataset_type: Either "aime" or "gsm8k"

    Returns:
        Extracted answer or None
    """
    if "gsm8k" in dataset_type.lower():
        return extract_answer_gsm8k(text)
    else:
        return extract_answer_aime(text)


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

    # Clean and normalize
    pred_clean = str(predicted).strip()
    gt_clean = str(ground_truth).strip()

    # Try exact match first
    if pred_clean == gt_clean:
        return True

    # Try numeric comparison
    try:
        pred_clean = pred_clean.replace(',', '')
        gt_clean = gt_clean.replace(',', '')

        pred_num = float(pred_clean)
        gt_num = float(gt_clean)

        # For integers, compare exactly
        if pred_num == int(pred_num) and gt_num == int(gt_num):
            return int(pred_num) == int(gt_num)

        # For floats, use small tolerance
        return abs(pred_num - gt_num) < 1e-6

    except (ValueError, TypeError):
        # Fall back to string comparison
        return pred_clean.lower() == gt_clean.lower()


def load_dataset_by_name(dataset_name: str, split: str = "test") -> List[Tuple[str, Any]]:
    """
    Load dataset by name with auto-detection.

    Args:
        dataset_name: "AIME2025-I", "AIME2025-II", "gsm8k", or "both" (for AIME)
        split: Dataset split (default: "test")

    Returns:
        List of (dataset_name, dataset) tuples
    """
    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        return [("gsm8k", ds)]

    elif "aime2025-i" in dataset_name_lower:
        ds = load_dataset("opencompass/AIME2025", name="AIME2025-I", split=split)
        return [("AIME2025-I", ds)]

    elif "aime2025-ii" in dataset_name_lower:
        ds = load_dataset("opencompass/AIME2025", name="AIME2025-II", split=split)
        return [("AIME2025-II", ds)]

    elif dataset_name_lower == "both":
        # Load both AIME datasets
        ds1 = load_dataset("opencompass/AIME2025", name="AIME2025-I", split=split)
        ds2 = load_dataset("opencompass/AIME2025", name="AIME2025-II", split=split)
        return [("AIME2025-I", ds1), ("AIME2025-II", ds2)]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'AIME2025-I', 'AIME2025-II', 'gsm8k', or 'both'")


def get_question_and_ground_truth(dataset_name: str, question_data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract question and ground truth from dataset item.

    Returns:
        (question_text, ground_truth)
    """
    question = question_data['question']

    if "gsm8k" in dataset_name.lower():
        # For GSM8k, extract answer from the answer field
        answer_text = question_data.get('answer', '')
        # GSM8k format: "reasoning text ... #### 123"
        if "####" in answer_text:
            parts = answer_text.split("####")
            if len(parts) > 1:
                gt = parts[-1].strip()
                # Extract number
                numbers = re.findall(r'-?\d+\.?\d*', gt)
                if numbers:
                    ground_truth = numbers[0]
                else:
                    ground_truth = gt
            else:
                ground_truth = answer_text.strip()
        else:
            ground_truth = answer_text.strip()
    else:
        # AIME format - answer is directly provided
        ground_truth = str(question_data.get('answer', '')).strip()

    return question, ground_truth


# Export all functions
__all__ = [
    'extract_answer_robust',
    'extract_answer_aime',
    'extract_answer_gsm8k',
    'check_answer_equality',
    'load_dataset_by_name',
    'get_question_and_ground_truth',
]