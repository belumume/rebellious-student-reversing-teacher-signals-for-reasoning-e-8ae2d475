"""Verifiable reward: extract the \\boxed{} integer and compare to ground truth."""

import re

_BOXED = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")
_LAST_INT = re.compile(r"(-?\d+)")


def extract_answer(text):
    m = list(_BOXED.finditer(text))
    if m:
        try:
            return int(m[-1].group(1))
        except ValueError:
            return None
    # fallback: last integer mentioned
    m2 = list(_LAST_INT.finditer(text))
    if m2:
        try:
            return int(m2[-1].group(1))
        except ValueError:
            return None
    return None


def reward(generated_text, gold_answer):
    pred = extract_answer(generated_text)
    if pred is None:
        return 0.0
    return 1.0 if pred == int(gold_answer) else 0.0
