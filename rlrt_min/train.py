"""Minimal GRPO vs RLRT trainer — a faithful, self-contained reproduction of the
core mechanism of arXiv:2605.10781 ("Rebellious Student: Reversing Teacher
Signals for Reasoning Exploration with Self-Distilled RLVR").

What this reproduces (the paper's central claim/mechanism):
  - RLVR with GRPO: sample K rollouts per prompt, verifiable {0,1} reward,
    group-standardized advantage A^(k).
  - Self-distillation teacher = the SAME model conditioned on privileged context
    c (here: the correct answer), i.e. P^T_t = pi_theta(.| h_t, c), while the
    student P^S_t = pi_theta(.| h_t).
  - RLRT (Algorithm 1): on CORRECT rollouts only, reweight the per-token advantage
    by the token-level information asymmetry  D_hat_t = log P^S_t(y_t) - log P^T_t(y_t):
        w_t   = exp( sign(A) * D_hat_t )                          (Eq. 6)
        A_t   = A * [ (1-lambda) + lambda * clip(w_t, 1-eps_w, 1+eps_w) ]   (Eq. 7)
    On incorrect rollouts, A_t = A (vanilla GRPO). lambda=0 recovers GRPO.

The only difference between the GRPO and RLRT runs is this advantage reweighting
(`algo` flag) — everything else (model, data, sampling, optimizer) is identical,
so the training curves are directly comparable, mirroring the paper's Fig. 5.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from task import make_dataset, format_prompt
from reward import reward as verifiable_reward

ARTIFACT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".openresearch", "artifacts",
)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_chat(tokenizer, user_content, assistant_prefix=None):
    msgs = [{"role": "user", "content": user_content}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    if assistant_prefix:
        text = text + assistant_prefix
    return text


@torch.no_grad()
def generate_group(model, tokenizer, prompt_text, K, max_new_tokens, device, temperature):
    enc = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = enc.input_ids.repeat(K, 1)
    attn = enc.attention_mask.repeat(K, 1)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = input_ids.shape[1]
    gen_ids = out[:, prompt_len:]  # (K, gen_len)
    texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return gen_ids, prompt_len, texts


def token_logprobs(model, full_ids, attn, response_start, response_ids, device):
    """Per-token logprob of `response_ids` under the model, given the full sequence.

    full_ids: (1, L) prompt+response. Returns logprobs for the response tokens
    (positions response_start .. L-1), shape (resp_len,).
    """
    out = model(input_ids=full_ids, attention_mask=attn)
    logits = out.logits[0]  # (L, V)
    # logits at position i predict token i+1
    logp = F.log_softmax(logits.float(), dim=-1)  # (L, V)
    idx = torch.arange(response_start - 1, full_ids.shape[1] - 1, device=device)
    tok = response_ids  # (resp_len,)
    sel = logp[idx].gather(-1, tok.unsqueeze(-1)).squeeze(-1)  # (resp_len,)
    return sel


def evaluate(model, tokenizer, eval_set, device, K_eval, max_new_tokens, temperature, n_problems):
    """avg@K and pass@K on a held-out set."""
    model.eval()
    subset = eval_set[:n_problems]
    avg_scores, pass_scores = [], []
    for ex in subset:
        prompt = build_chat(tokenizer, format_prompt(ex["question"]))
        _, _, texts = generate_group(
            model, tokenizer, prompt, K_eval, max_new_tokens, device, temperature
        )
        rs = [verifiable_reward(t, ex["answer"]) for t in texts]
        avg_scores.append(float(np.mean(rs)))
        pass_scores.append(1.0 if max(rs) > 0 else 0.0)
    return float(np.mean(avg_scores)), float(np.mean(pass_scores))


def train_one(algo, args, device, log):
    """Train a fresh copy of the model with `algo` in {grpo, rlrt}. Returns history."""
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    model.config.use_cache = True

    train_set, eval_set = make_dataset(args.n_train, args.n_eval, seed=args.data_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    log(f"[{algo}] eval @ step 0 ...")
    model.config.use_cache = True
    avg0, pass0 = evaluate(
        model, tokenizer, eval_set, device, args.k_eval,
        args.max_new_tokens, args.eval_temperature, args.n_eval_problems,
    )
    log(f"[{algo}] step 0: eval_avg@{args.k_eval}={avg0:.4f} pass@{args.k_eval}={pass0:.4f}")
    history.append({"step": 0, "train_reward": None, "eval_avg": avg0, "eval_pass": pass0})

    step = 0
    explore_amp_log = []  # track fraction of tokens up-weighted (valuable exploration)
    for it in range(args.steps):
        step += 1
        batch = random.sample(train_set, args.prompts_per_step)
        # --- collect rollouts + rewards + advantages for the whole step ---
        per_prompt = []
        step_rewards = []
        for ex in batch:
            student_prompt = build_chat(tokenizer, format_prompt(ex["question"]))
            model.config.use_cache = True
            model.eval()
            gen_ids, _, texts = generate_group(
                model, tokenizer, student_prompt, args.group_size,
                args.max_new_tokens, device, args.temperature,
            )
            rewards = np.array([verifiable_reward(t, ex["answer"]) for t in texts], dtype=np.float32)
            step_rewards.append(rewards.mean())
            mean, std = rewards.mean(), rewards.std()
            adv = (rewards - mean) / (std + 1e-6)  # group-standardized advantage A^(k)
            per_prompt.append({
                "ex": ex, "student_prompt": student_prompt,
                "gen_ids": gen_ids, "rewards": rewards, "adv": adv,
            })

        # --- policy gradient update over the rollouts ---
        model.train()
        model.config.use_cache = False
        optimizer.zero_grad()
        total_loss = 0.0
        n_tok_total = 0
        amp_num, amp_den = 0, 0
        for pp in per_prompt:
            ex = pp["ex"]
            # teacher prompt: privileged context c = the correct answer
            teacher_user = (
                format_prompt(ex["question"])
                + f"\n\n(Hint: the correct final answer is {ex['answer']}.)"
            )
            teacher_prompt = build_chat(tokenizer, teacher_user)
            t_enc = tokenizer(teacher_prompt, return_tensors="pt").to(device)
            s_enc = tokenizer(pp["student_prompt"], return_tensors="pt").to(device)

            for k in range(args.group_size):
                A = float(pp["adv"][k])
                if abs(A) < 1e-8:
                    continue  # zero advantage -> no gradient anyway
                resp = pp["gen_ids"][k]
                # trim trailing pad
                resp = resp[resp != tokenizer.pad_token_id]
                if resp.numel() == 0:
                    continue
                resp = resp.unsqueeze(0)  # (1, R)

                # student logprobs (gradient path)
                s_full = torch.cat([s_enc.input_ids, resp], dim=1)
                s_attn = torch.ones_like(s_full)
                s_logp = token_logprobs(
                    model, s_full, s_attn, s_enc.input_ids.shape[1], resp[0], device
                )  # (R,) requires grad

                A_t = torch.full_like(s_logp, A)  # default GRPO: advantage broadcast

                if algo == "rlrt" and pp["rewards"][k] > 0.5:
                    # RLRT reweighting on CORRECT rollouts only (Algorithm 1)
                    with torch.no_grad():
                        s_logp_det = s_logp.detach()
                        t_full = torch.cat([t_enc.input_ids, resp], dim=1)
                        t_attn = torch.ones_like(t_full)
                        t_logp = token_logprobs(
                            model, t_full, t_attn, t_enc.input_ids.shape[1], resp[0], device
                        )  # P^T_t(y_t)
                        D_hat = s_logp_det - t_logp           # token info asymmetry
                        sign_A = 1.0 if A > 0 else -1.0
                        w = torch.exp(sign_A * D_hat)         # Eq. (6)
                        w = torch.clamp(w, 1.0 - args.eps_w, 1.0 + args.eps_w)
                        factor = (1.0 - args.lam) + args.lam * w
                        A_t = A * factor                      # Eq. (7)
                        amp_num += int((factor > 1.0).sum().item())
                        amp_den += factor.numel()

                # REINFORCE/GRPO surrogate: -E[ A_t * logp ]
                loss = -(A_t.detach() * s_logp).sum()
                loss.backward()
                total_loss += float(loss.detach())
                n_tok_total += s_logp.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        train_reward = float(np.mean(step_rewards))
        amp_frac = (amp_num / amp_den) if amp_den > 0 else 0.0
        explore_amp_log.append(amp_frac)
        msg = (f"[{algo}] step {step}: train_reward={train_reward:.4f} "
               f"loss/tok={total_loss / max(n_tok_total,1):.4f}")
        if algo == "rlrt":
            msg += f" upweighted_tok_frac={amp_frac:.3f}"
        log(msg)

        if step % args.eval_every == 0 or step == args.steps:
            model.config.use_cache = True
            avg, passk = evaluate(
                model, tokenizer, eval_set, device, args.k_eval,
                args.max_new_tokens, args.eval_temperature, args.n_eval_problems,
            )
            log(f"[{algo}] step {step}: eval_avg@{args.k_eval}={avg:.4f} "
                f"pass@{args.k_eval}={passk:.4f}")
            history.append({"step": step, "train_reward": train_reward,
                            "eval_avg": avg, "eval_pass": passk})

    del model
    torch.cuda.empty_cache()
    return history, float(np.mean(explore_amp_log)) if explore_amp_log else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--prompts_per_step", type=int, default=8)
    ap.add_argument("--group_size", type=int, default=8)   # K rollouts per prompt
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--eval_temperature", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)       # RLRT mixing lambda
    ap.add_argument("--eps_w", type=float, default=1.0)     # RLRT clip radius
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--n_eval_problems", type=int, default=48)
    ap.add_argument("--k_eval", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--algos", default="grpo,rlrt")
    args = ap.parse_args()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    logfile = open(os.path.join(ARTIFACT_DIR, "train_log.txt"), "w")

    def log(m):
        print(m, flush=True)
        logfile.write(m + "\n")
        logfile.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} model={args.model}")
    log(f"config: {vars(args)}")

    results = {}
    explore = {}
    for algo in args.algos.split(","):
        algo = algo.strip()
        t0 = time.time()
        hist, amp = train_one(algo, args, device, log)
        results[algo] = hist
        explore[algo] = amp
        log(f"[{algo}] done in {time.time()-t0:.1f}s")

    with open(os.path.join(ARTIFACT_DIR, "results.json"), "w") as f:
        json.dump({"config": vars(args), "results": results,
                   "upweighted_token_frac": explore}, f, indent=2)

    # ---- write EVAL.md ----
    write_eval(args, results, explore)
    logfile.close()


def write_eval(args, results, explore):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lines = []
    lines.append("# EVAL: Minimal reproduction of RLRT (arXiv:2605.10781)\n")
    lines.append("Minimal, self-contained GRPO vs RLRT comparison on a verifiable "
                 "arithmetic-reasoning task. RLRT reverses the self-distillation "
                 "teacher signal on correct rollouts (Algorithm 1, Eq. 6-7). The "
                 "only difference between the two runs is the advantage reweighting.\n")
    lines.append(f"- Model: `{args.model}`")
    lines.append(f"- Steps: {args.steps}, prompts/step: {args.prompts_per_step}, "
                 f"group size K: {args.group_size}")
    lines.append(f"- RLRT: lambda={args.lam}, eps_w={args.eps_w}\n")

    def final(algo, key):
        return results[algo][-1][key]

    lines.append("## Final metrics (held-out eval)\n")
    lines.append("| Algo | eval avg@K | eval pass@K | final train reward |")
    lines.append("|------|-----------|------------|--------------------|")
    for algo in results:
        tr = results[algo][-1]["train_reward"]
        lines.append(f"| {algo.upper()} | {final(algo,'eval_avg'):.4f} | "
                     f"{final(algo,'eval_pass'):.4f} | "
                     f"{tr:.4f} |")
    lines.append("")

    if "grpo" in results and "rlrt" in results:
        d_avg = final("rlrt", "eval_avg") - final("grpo", "eval_avg")
        d_pass = final("rlrt", "eval_pass") - final("grpo", "eval_pass")
        s0 = results["grpo"][0]["eval_avg"]
        lines.append("## Core-claim check: does RLRT beat GRPO?\n")
        lines.append(f"- Start (step 0) eval avg@K: {s0:.4f}")
        lines.append(f"- Delta eval avg@K (RLRT - GRPO): **{d_avg:+.4f}**")
        lines.append(f"- Delta eval pass@K (RLRT - GRPO): **{d_pass:+.4f}**")
        amp = explore.get("rlrt", 0.0)
        lines.append(f"- RLRT mean up-weighted-token fraction (self-driven / "
                     f"\"valuable exploration\" signal): {amp:.3f}")
        verdict = "RLRT > GRPO (claim reproduced)" if d_avg > 0 else (
            "RLRT == GRPO" if abs(d_avg) < 1e-9 else "RLRT < GRPO (not reproduced)")
        lines.append(f"\n**Verdict: {verdict}**\n")

    lines.append("## Per-step eval trajectory\n")
    for algo in results:
        lines.append(f"### {algo.upper()}")
        lines.append("| step | train_reward | eval avg@K | eval pass@K |")
        lines.append("|------|--------------|-----------|------------|")
        for h in results[algo]:
            tr = "-" if h["train_reward"] is None else f"{h['train_reward']:.4f}"
            lines.append(f"| {h['step']} | {tr} | {h['eval_avg']:.4f} | "
                         f"{h['eval_pass']:.4f} |")
        lines.append("")

    with open(os.path.join(repo_root, "EVAL.md"), "w") as f:
        f.write("\n".join(lines))
    # also drop a copy into artifacts
    with open(os.path.join(ARTIFACT_DIR, "EVAL.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
