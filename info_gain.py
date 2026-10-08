#!/usr/bin/env python3
"""
Verified Information Gain: does a teacher trace actually reduce the STUDENT's
uncertainty about the answer?

Adapts EntropyLong (Jia et al. 2025) from long-context data construction to
reasoning distillation:

    EntropyLong                     ->   here
    document D                      ->   math problem x
    base model                      ->   STUDENT model
    candidate context C             ->   TEACHER reasoning trace r
    Delta_I = (H_before - H_after)/H_before

Concretely we measure the student's predictive entropy at the answer position:

    H_before = H_student(y | x,  "The final answer is:")
    H_after  = H_student(y | x, r, "The final answer is:")
    Delta_I  = (H_before - H_after) / H_before

Delta_I > 0  the trace resolves uncertainty  -> learnable, worth training on
Delta_I ~ 0  the trace changes nothing       -> student cannot use it yet
Delta_I < 0  the trace increases confusion   -> possibly harmful supervision

Only forward passes are needed (no generation), so this is fast.

Usage:
    python info_gain.py --student Qwen/Qwen2.5-1.5B-Instruct \
                        --traces traces_Qwen2-5-7B-Instruct_math500.jsonl
"""

import argparse, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed "
    "responses. Think step by step and provide the final answer in the form of "
    "'The final answer is: \\boxed{answer}'."
)
ANSWER_CUE = "\nThe final answer is: \\boxed{"


@torch.no_grad()
def entropy_at_answer(model, tok, question, reasoning=None):
    """Predictive entropy of the student at the answer position.

    reasoning=None  -> H_before (problem only)
    reasoning=text  -> H_after  (problem + teacher's reasoning)
    """
    user = question if reasoning is None else f"{question}\n\n{reasoning}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    if not hasattr(enc, "keys"):
        enc = {"input_ids": enc}
    ids = enc["input_ids"]

    # append the answer cue so the next token is the answer itself
    cue = tok(ANSWER_CUE, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = torch.cat([ids, cue], dim=-1).to(model.device)

    logits = model(ids).logits[0, -1].float()
    lp = torch.log_softmax(logits, dim=-1)
    return -(lp.exp() * lp).sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--traces", required=True,
                    help="trace file from the TEACHER")
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--max-reasoning-chars", type=int, default=3000)
    ap.add_argument("--out", default="info_gain.jsonl")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--truncate-frac", type=float, default=1.0,
                    help="CONTROL: keep only this fraction of the reasoning. "
                         "0.5 removes the final steps where the answer is "
                         "computed, testing whether Delta_I measures reasoning "
                         "value or merely answer read-off.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.traces)][:args.max_items]
    print(f"[setup] {len(rows)} teacher traces from {args.traces}")
    print(f"[setup] loading student {args.student}")

    quant = None
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.student)
    model = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype="auto", device_map="auto",
        quantization_config=quant).eval()

    # question text is not stored in the trace file, so reload the dataset
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    qmap = {f"math500-{i}": r["problem"] for i, r in enumerate(ds)}

    out = []
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            q = qmap.get(r["id"])
            if q is None:
                continue
            reasoning = (r.get("reasoning_text") or "")[:args.max_reasoning_chars]
            if args.truncate_frac < 1.0:
                reasoning = reasoning[:int(len(reasoning) * args.truncate_frac)]
            if len(reasoning) < 20:
                continue

            h_before = entropy_at_answer(model, tok, q, None)
            h_after = entropy_at_answer(model, tok, q, reasoning)
            gain = (h_before - h_after) / h_before if h_before > 0 else 0.0

            rec = {"id": r["id"], "level": r.get("level"),
                   "teacher_correct": r.get("correct"),
                   "teacher_conf": r.get("logp_r_mean"),
                   "n_tokens": r.get("n_tokens_total"),
                   "H_before": h_before, "H_after": h_after,
                   "delta_I": gain}
            f.write(json.dumps(rec) + "\n")
            out.append(rec)

            if (i + 1) % 25 == 0:
                print(f"[gain] {i+1}/{len(rows)}")

    # ---- summary ----------------------------------------------------------
    import statistics as st
    gains = [r["delta_I"] for r in out]
    print("\n" + "=" * 66)
    print("VERIFIED INFORMATION GAIN  (student uncertainty reduction)")
    print("=" * 66)
    print(f"  traces scored        : {len(out)}")
    if args.truncate_frac < 1.0:
        print(f"  CONTROL: reasoning truncated to {args.truncate_frac:.0%} "
              f"(final steps removed)")
    print(f"  mean H_before        : {st.mean(r['H_before'] for r in out):.3f}")
    print(f"  mean H_after         : {st.mean(r['H_after'] for r in out):.3f}")
    print(f"  mean delta_I         : {st.mean(gains):+.3f}")
    print(f"  median delta_I       : {st.median(gains):+.3f}")
    print(f"  traces with gain > 0 : {sum(1 for g in gains if g > 0)}/{len(gains)}"
          f"  ({sum(1 for g in gains if g > 0)/len(gains):.0%})")
    print(f"  traces with gain >0.4: {sum(1 for g in gains if g > 0.4)}/{len(gains)}"
          f"  ({sum(1 for g in gains if g > 0.4)/len(gains):.0%})"
          f"   <- EntropyLong's threshold")

    # gain by whether the teacher was right
    # NOTE: delta_I is a ratio, so when H_before is small a tiny absolute
    # increase produces a huge negative value. The MEAN of ratios is therefore
    # unstable and outlier-dominated; the MEDIAN and the absolute entropy
    # reduction are the reliable summaries. Report all three.
    print(f"\n  {'group':<18}{'n':>5}{'median dI':>11}{'mean dI':>10}"
          f"{'H_before':>10}{'H_after':>9}{'abs drop':>10}")
    for label, sel in [("teacher CORRECT", True), ("teacher WRONG", False)]:
        g = [r for r in out if r["teacher_correct"] is sel]
        if len(g) > 1:
            d = [r["delta_I"] for r in g]
            hb = st.mean(r["H_before"] for r in g)
            ha = st.mean(r["H_after"] for r in g)
            print(f"  {label:<18}{len(g):>5}{st.median(d):>+11.3f}{st.mean(d):>+10.3f}"
                  f"{hb:>10.3f}{ha:>9.3f}{hb-ha:>+10.3f}")

    # gain by difficulty level
    print("\n  delta_I by difficulty level:")
    print(f"  {'level':<10}{'n':>5}{'H_before':>11}{'H_after':>10}{'delta_I':>10}")
    lv = {}
    for r in out:
        lv.setdefault(str(r["level"]), []).append(r)
    for k in sorted(lv):
        g = lv[k]
        print(f"  {k:<10}{len(g):>5}{st.mean(x['H_before'] for x in g):>11.3f}"
              f"{st.mean(x['H_after'] for x in g):>10.3f}"
              f"{st.mean(x['delta_I'] for x in g):>+10.3f}")

    # correlation with teacher confidence -- is this a NEW signal?
    try:
        from scipy import stats as sps
        pairs = [(r["delta_I"], r["teacher_conf"]) for r in out
                 if r["teacher_conf"] is not None]
        if len(pairs) > 10:
            rho, p = sps.spearmanr([a for a, _ in pairs], [b for _, b in pairs])
            print(f"\n  corr(delta_I, teacher confidence): rho={rho:+.3f}  p={p:.1e}")
            print("  Low correlation => information gain is a DIFFERENT signal")
            print("  from confidence, not a proxy for it.")
    except ImportError:
        pass

    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
