"""Minimal verifiable-reasoning task: multi-step integer arithmetic word problems.

Each sample is a short word problem with a unique integer answer. The reward is
verifiable (1 if the model's \\boxed{} answer matches, else 0), exactly the RLVR
setting the paper studies. We keep the task small so a tiny model can partially
solve it and GRPO/RLRT have room to improve, on a single GPU in minutes.
"""

import random

PROMPT_TEMPLATE = (
    "Solve the following problem. Reason step by step, and put your final "
    "answer (an integer) within \\boxed{{}}.\n\n{question}"
)


def _gen_problem(rng):
    """A 3-operation arithmetic word problem with an integer answer."""
    a = rng.randint(2, 20)
    b = rng.randint(2, 20)
    c = rng.randint(2, 12)
    d = rng.randint(1, 10)
    style = rng.randint(0, 3)
    if style == 0:
        q = (f"A shop has {a} boxes with {b} apples each. "
             f"They sell {c} apples and then buy {d} more boxes of {b}. "
             f"How many apples are there now?")
        ans = a * b - c + d * b
    elif style == 1:
        q = (f"There are {a} rows of chairs with {b} chairs per row. "
             f"{c} rows are removed and {d} single chairs are added. "
             f"How many chairs remain?")
        ans = a * b - c * b + d
    elif style == 2:
        q = (f"A tank holds {a} liters. Each of {b} pumps adds {c} liters, "
             f"then {d} liters leak out. How many liters are in the tank?")
        ans = a + b * c - d
    else:
        q = (f"A library has {a} shelves of {b} books. {c} books are borrowed "
             f"from each of {d} shelves. How many books remain?")
        ans = a * b - c * d
    return q, ans


def make_dataset(n_train=256, n_eval=128, seed=0):
    rng = random.Random(seed)
    seen = set()
    out = []
    while len(out) < n_train + n_eval:
        q, ans = _gen_problem(rng)
        key = (q, ans)
        if key in seen:
            continue
        seen.add(key)
        out.append({"question": q, "answer": int(ans)})
    train = out[:n_train]
    eval_ = out[n_train:n_train + n_eval]
    return train, eval_


def format_prompt(question):
    return PROMPT_TEMPLATE.format(question=question)
