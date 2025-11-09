"""
Fixed answer extraction for AIME problems
This script patches the answer extraction to be more robust
"""

import re
from typing import Optional

def extract_answer_robust(text: str) -> Optional[str]:
    """
    Extract answer from various formats:
    - \\boxed{answer}
    - boxed{answer}
    - The answer is X
    - Therefore, X
    - Final answer: X
    - = X (at the end)
    """
    if not text:
        return None

    # Try standard boxed format first (with or without backslash)
    patterns = [
        r'\\boxed\{([^}]+)\}',  # \boxed{answer}
        r'boxed\{([^}]+)\}',     # boxed{answer}
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Return the last boxed answer found
            return matches[-1].strip()

    # Try to find answer at the end of the text
    lines = text.strip().split('\n')

    # Check last few lines for answer patterns
    for line in reversed(lines[-5:]):
        line = line.strip()

        # Pattern: "The answer is X" or "Therefore, X" or "Thus, X"
        answer_patterns = [
            r'(?:The answer is|Therefore,?|Thus,?|So,?|Hence,?)\s*[:=]?\s*(\d+)',
            r'Final answer\s*[:=]\s*(\d+)',
            r'^=\s*(\d+)$',  # Just "= 123" on its own line
            r'^\$?(\d+)\$?$',  # Just a number on its own line (possibly in math mode)
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).strip()

    # Last resort: look for a standalone number at the very end
    # This is risky but might catch some cases
    last_line = lines[-1].strip() if lines else ""
    if last_line and last_line.isdigit() and len(last_line) <= 4:  # AIME answers are typically 0-999
        return last_line

    return None


def test_extraction():
    """Test the extraction function with various formats"""
    test_cases = [
        ("The answer is \\boxed{123}", "123"),
        ("Solution: We get \\boxed{456}", "456"),
        ("Therefore, the answer is 789", "789"),
        ("Final answer: 321", "321"),
        ("= 654", "654"),
        ("The calculation gives us 987", None),  # No clear answer format
        ("So we have\n\\boxed{111}", "111"),
        ("Multiple boxes \\boxed{222} and \\boxed{333}", "333"),  # Should get the last one
    ]

    for text, expected in test_cases:
        result = extract_answer_robust(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input: {text[:50]}... → Got: {result}, Expected: {expected}")


if __name__ == "__main__":
    test_extraction()
    print("\nTo fix your AIME results, replace extract_boxed_answer with extract_answer_robust")