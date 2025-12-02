"""
Utility functions for SCLLM

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from collections import Counter, defaultdict
from dynasor.core.evaluator import math_equal
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


def extract_answer(text: str) -> Optional[str]:
    """Extract boxed answer from text (for AIME/math problems)"""
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        return a.strip()
    return None


def extract_answer_gsm8k(text: str) -> Optional[str]:
    """
    Extract numerical answer from GSM8k format text

    GSM8k answers are formatted as: "reasoning ... #### <number>"
    If #### is present, extract the number after it.
    Otherwise, extract the last number in the text.
    """
    import re

    # Method 1: Look for #### marker (standard GSM8k format)
    if "####" in text:
        parts = text.split("####")
        if len(parts) > 1:
            answer_text = parts[-1].strip()
            # Extract first number after ####
            numbers = re.findall(r'-?\d+\.?\d*', answer_text)
            if numbers:
                return numbers[0]

    # Method 2: Extract last number in text (fallback)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]

    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for consistent comparison and voting.

    Handles common variations like:
    - "1" vs "1." vs "1.0" vs "1.00" -> all become "1"
    - "18" vs "18." vs "18.0" vs "18.00" -> all become "18"
    - "-5" vs "-5." vs "-5.0" -> all become "-5"
    - "0.5" vs ".5" -> both become "0.5"
    - Removes trailing periods and spaces
    - Handles very long repeated digits (from runaway generation)

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer string
    """
    if answer is None or answer == '':
        return ''

    # Convert to string if not already
    answer = str(answer).strip()

    # Handle extremely long repeated digit strings (from runaway generation)
    # e.g., "3333333333333..." -> "Invalid"
    if len(answer) > 50 and all(c == answer[0] for c in answer if c.isdigit()):
        return 'Invalid'

    # Remove trailing periods that aren't part of decimals
    if answer.endswith('.'):
        answer = answer[:-1]

    # Remove commas from numbers
    answer = answer.replace(',', '')

    # Try to parse as a number
    try:
        # Parse as float
        num = float(answer)

        # Check for invalid values
        if not np.isfinite(num):
            return 'Invalid'

        # If it's a whole number, return as integer string
        if num == int(num):
            return str(int(num))
        else:
            # Otherwise return as simplified decimal
            # This removes trailing zeros: 1.50 -> 1.5
            formatted = f"{num:.10f}".rstrip('0').rstrip('.')
            return formatted
    except (ValueError, TypeError, OverflowError):
        # Not a number, just return cleaned string
        return answer.strip()


def quick_parse(text: str) -> str:
    """Parse LaTeX text content"""
    if '\\text{' in text and '}' in text:
        # Find all occurrences of \text{...} and remove them
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            # Replace \text{content} with just content
            content = text[start + 6:end]  # 6 is length of '\text{'
            text = text[:start] + content + text[end + 1:]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    """Check if answer equals ground truth (for AIME/math problems)"""
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)


def equal_func_gsm8k(answer: str, ground_truth: str) -> bool:
    """
    Check if answer equals ground truth for GSM8k (numerical comparison)

    Handles:
    - Integer comparison
    - Float comparison with tolerance
    - String-to-number conversion
    """
    if answer is None or ground_truth is None:
        return False

    try:
        # Remove commas from numbers
        answer_clean = str(answer).replace(',', '').strip()
        gt_clean = str(ground_truth).replace(',', '').strip()

        # Convert to float for comparison
        answer_num = float(answer_clean)
        gt_num = float(gt_clean)

        # Check if both are integers
        if answer_num == int(answer_num) and gt_num == int(gt_num):
            return int(answer_num) == int(gt_num)

        # Float comparison with small tolerance
        return abs(answer_num - gt_num) < 1e-6

    except (ValueError, TypeError):
        # Fallback to string comparison
        return str(answer).strip() == str(ground_truth).strip()


def compute_confidence(logprobs: List[Dict]) -> List[float]:
    """Compute confidence score from logprobs and return only confidence values"""
    confs = []
    for token_logprobs in logprobs:
        if token_logprobs:
            # vLLM returns a dict of {token_id: Logprob object}
            # Get the selected token's logprob (the one with highest probability)
            mean_logprob = np.mean([lp.logprob for lp in token_logprobs.values()])
            confs.append(round(-mean_logprob, 3))
    return confs


# ============= VOTING FUNCTIONS =============

def simple_majority_vote(answers: List[str]) -> Optional[str]:
    """Simple majority voting with answer normalization"""
    if not answers:
        return None

    # Normalize answers before counting
    normalized_answers = [normalize_answer(ans) for ans in answers]

    # Filter out invalid answers
    valid_answers = [ans for ans in normalized_answers if ans != 'Invalid']
    if not valid_answers:
        return None

    vote_counts = Counter(valid_answers)
    return vote_counts.most_common(1)[0][0]


# ============= OUTPUT PROCESSING =============

def process_output_offline(output, window_size: int) -> Dict[str, Any]:
    """Process a single vLLM output for offline mode - stores full confidence array"""
    text = output.text
    token_ids = output.token_ids
    logprobs = output.logprobs
    
    # Calculate confidence but don't store full logprobs
    confs = compute_confidence(logprobs) if logprobs else []
    
    extracted_answer = extract_answer(text)
    
    return {
        "stop_reason": output.finish_reason,
        "text": text,
        "token_ids": token_ids,
        "num_tokens": len(token_ids) if token_ids else 0,
        "confs": confs,  # Store full confidence array for offline analysis
        "extracted_answer": extracted_answer,
    }


def process_batch_results_offline(batch_outputs, window_size: int) -> Dict[str, Any]:
    """Process batch results from vLLM for offline mode"""
    question_outputs = batch_outputs[0].outputs
    
    # Process all traces for this question
    traces = []
    total_tokens = 0
    
    for output in question_outputs:
        trace_data = process_output_offline(output, window_size)
        traces.append(trace_data)
        total_tokens += trace_data["num_tokens"]
    
    return {
        'traces': traces,
        'total_tokens': total_tokens,
        'num_traces': len(traces)
    }


# ============= PROMPT PREPARATION FUNCTIONS =============

def prepare_prompt(question: str, tokenizer, model_type: str = "deepseek") -> str:
    """Prepare prompt for a single question"""
    if model_type == "deepseek":
        # Format prompt using chat template for DeepSeek
        messages = [
            {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
            {"role": "user", "content": question}
        ]
    else:
        # Format for GPT-like models
        messages = [
            {"role": "user", "content": question}
        ]
    
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return full_prompt


# ============= PEAK DETECTION UTILITIES =============

def sliding_window_confidence(confs: List[float], window_size: int) -> List[float]:
    """
    Compute sliding window average confidence

    Args:
        confs: List of confidence scores
        window_size: Size of sliding window

    Returns:
        List of windowed averages (length = len(confs) - window_size + 1)
    """
    if not confs or len(confs) < window_size:
        return []

    window_avgs = []
    for i in range(len(confs) - window_size + 1):
        window = confs[i:i + window_size]
        window_avgs.append(np.mean(window))

    return window_avgs


def find_local_maxima(values: List[float], min_distance: int = 1) -> List[int]:
    """
    Find indices of local maxima in a list of values

    Args:
        values: List of values
        min_distance: Minimum distance between peaks

    Returns:
        List of indices where local maxima occur
    """
    if len(values) < 3:
        return []

    maxima = []
    for i in range(1, len(values) - 1):
        # Check if it's a local maximum
        if values[i] > values[i-1] and values[i] > values[i+1]:
            # Check minimum distance from previous maxima
            if not maxima or (i - maxima[-1]) >= min_distance:
                maxima.append(i)

    return maxima


def detect_confidence_peaks(
    confs: List[float],
    window_size: int = 512,
    threshold: float = 1.5,
    min_peak_distance: int = 256,
    peak_selection_ratio: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Detect confidence peaks in a trace

    Args:
        confs: List of token confidence scores
        window_size: Size of sliding window
        threshold: Minimum confidence for a peak
        min_peak_distance: Minimum tokens between peaks
        peak_selection_ratio: Valid range for peaks (e.g., 0.8 = middle 80% of trace)

    Returns:
        List of peak dictionaries with position and confidence
    """
    if not confs or len(confs) < window_size:
        return []

    # Compute sliding window averages
    window_avgs = sliding_window_confidence(confs, window_size)

    # Find local maxima
    maxima_indices = find_local_maxima(window_avgs, min_distance=min_peak_distance)

    # Filter by threshold and position
    peaks = []
    min_fraction = (1 - peak_selection_ratio) / 2
    max_fraction = 1 - min_fraction

    for idx in maxima_indices:
        if window_avgs[idx] > threshold:
            # Calculate position in original trace
            position = idx + window_size // 2
            trace_fraction = position / len(confs)

            # Check if in valid range
            if min_fraction <= trace_fraction <= max_fraction:
                peaks.append({
                    'position': position,
                    'confidence': window_avgs[idx],
                    'trace_fraction': trace_fraction
                })

    return peaks


def prepare_prompt_from_tokens(token_ids: List[int], tokenizer) -> str:
    """
    Reconstruct prompt from token IDs for branching

    Args:
        token_ids: List of token IDs to use as prefix
        tokenizer: Tokenizer to decode tokens

    Returns:
        Reconstructed text prompt
    """
    # Decode tokens back to text
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    return text