# Minimal reproduction — RLRT (arXiv:2605.10781)

Minimal, self-contained proof-of-concept for **"Rebellious Student: Reversing
Teacher Signals for Reasoning Exploration with Self-Distilled RLVR"** (RLRT).

The paper's full pipeline trains Qwen3-4B/8B with a verl-based RLVR stack on
2xB200 GPUs for ~1 day per run. This repro instead isolates and demonstrates the
paper's **central mechanism** — reversing the self-distillation teacher signal on
correct rollouts — in a single script that runs on one GPU in minutes.

## What it does

Trains the **same** tiny model (`Qwen2.5-0.5B-Instruct` by default) twice on a
verifiable multi-step arithmetic task, changing only the advantage rule:

- **GRPO**: group-relative advantage `A^(k)`, standard token-level credit.
- **RLRT** (Algorithm 1 of the paper): on *correct* rollouts only, reweight the
  per-token advantage by the token-level information asymmetry
  `D_hat_t = log P^S_t(y_t) - log P^T_t(y_t)`:
  - `w_t = exp(sign(A) * D_hat_t)`  (Eq. 6)
  - `A_t = A * [ (1-lambda) + lambda * clip(w_t, 1-eps_w, 1+eps_w) ]`  (Eq. 7)

  The **teacher** `P^T` is the same model conditioned on privileged context
  `c` (the correct answer); the **student** `P^S` is conditioned only on the
  prompt. `lambda=0` recovers GRPO.

The core claim being checked: **RLRT outperforms GRPO** (faster/higher eval
score) by amplifying verified self-driven tokens.

## Run

```bash
bash run.sh
```

Produces `EVAL.md` (GRPO vs RLRT final metrics, deltas, per-step trajectory) and
`.openresearch/artifacts/` (`results.json`, `train_log.txt`, `EVAL.md`).

## Files
- `rlrt_min/task.py`   — verifiable arithmetic word-problem task
- `rlrt_min/reward.py` — `\boxed{}` answer extraction + 0/1 reward
- `rlrt_min/train.py`  — GRPO and RLRT trainer (the mechanism)
- `run.sh`             — installs deps, runs both, prints EVAL.md
