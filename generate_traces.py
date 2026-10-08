#!/usr/bin/env python3
"""
Pilot 0/1: generate reasoning traces + extract PiCSAR-style split confidences.

One generation run produces everything needed for four analyses:
  (1) calibration / confirmation bias   (2) trace length comparison
  (3) confidence vs ground-truth difficulty   (4) accuracy

Usage:
  python generate_traces.py --model Qwen/Qwen2.5-7B-Instruct --dataset math500 --n 100 --k 4
  python generate_traces.py --model openai/gpt-oss-20b      --dataset math500 --n 100 --k 4

Output: JSONL, one row per (problem, sample).
"""

import argparse, json, os, random, re, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ----------------------------------------------------------------------------
# Prompt (PiCSAR Appendix B wording, lightly trimmed)
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "Think step by step and provide the final answer in the form of "
    "'The final answer is: \\boxed{answer}'."
)

# ----------------------------------------------------------------------------
# Dataset loading. MATH500 is the primary set: it carries difficulty levels 1-5,
# which give us a ground-truth difficulty label to validate signals against.
# ----------------------------------------------------------------------------
def load_problems(name, n, sample="random", seed=0):
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rows = [{"id": f"math500-{i}", "question": r["problem"],
                 "answer": r["answer"], "level": r.get("level"),
                 "subject": r.get("subject"), "dataset": "math500"}
                for i, r in enumerate(ds)]
    elif name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rows = [{"id": f"gsm8k-{i}", "question": r["question"],
                 # GSM8K answers look like "...\n#### 42"
                 "answer": r["answer"].split("####")[-1].strip(),
                 "level": None, "subject": None, "dataset": "gsm8k"}
                for i, r in enumerate(ds)]
    elif name == "aime24":
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        rows = [{"id": f"aime24-{i}", "question": r["Problem"],
                 "answer": str(r["Answer"]), "level": None,
                 "subject": None, "dataset": "aime24"}
                for i, r in enumerate(ds)]
    else:
        raise ValueError(f"unknown dataset {name}")

    if not n or n >= len(rows):
        print(f"[data] using all {len(rows)} problems (no sampling)")
        return rows

    rng = random.Random(seed)   # fixed seed => reproducible

    if sample == "random":
        chosen = rng.sample(rows, n)
    elif sample == "stratified":
        # Equal-ish draw from each difficulty level. Better than plain random
        # when analysing confidence-vs-difficulty, since it guarantees enough
        # samples in every level rather than leaving it to chance.
        buckets = {}
        for r in rows:
            buckets.setdefault(str(r.get("level")), []).append(r)
        keys = sorted(buckets)
        per = max(1, n // len(keys))
        chosen = []
        for k in keys:
            chosen += rng.sample(buckets[k], min(per, len(buckets[k])))
        # top up / trim to exactly n
        if len(chosen) < n:
            rest = [r for r in rows if r not in chosen]
            chosen += rng.sample(rest, min(n - len(chosen), len(rest)))
        chosen = chosen[:n]
        rng.shuffle(chosen)
    elif sample == "stride":
        stride = len(rows) / n
        chosen = [rows[int(i * stride)] for i in range(n)]
    else:
        raise ValueError(f"unknown sample mode {sample}")

    print(f"[data] {sample} sample of {len(chosen)}/{len(rows)} problems (seed={seed})")
    if any(r.get("level") for r in chosen):
        from collections import Counter
        print(f"[data] level distribution: "
              f"{dict(sorted(Counter(str(r['level']) for r in chosen).items()))}")
    return chosen


# ----------------------------------------------------------------------------
# Splitting a generation into reasoning (r) and answer (y).
# gpt-oss uses the harmony format with explicit channels, which gives a much
# cleaner split than parsing Qwen's free text -- worth noting in the write-up.
# ----------------------------------------------------------------------------
def split_reasoning_answer(text, token_ids, tokenizer, model_name):
    """Return (split_index_in_tokens, reasoning_text, answer_text).

    split_index = first token index belonging to the answer segment.
    """
    lower = text.lower()
    split_char = None

    if "gpt-oss" in model_name.lower():
        # harmony: reasoning lives in the analysis channel, answer in the final channel
        m = re.search(r"<\|channel\|>final<\|message\|>", text)
        if m:
            split_char = m.end()
    if split_char is None:
        # generic: look for the answer cue, else the boxed expression
        m = re.search(r"the final answer is", lower)
        if m:
            split_char = m.start()
        else:
            m = re.search(r"\\boxed\{", text)
            split_char = m.start() if m else None

    if split_char is None:
        # No identifiable answer segment: treat the last 10% as the answer.
        split_idx = max(1, int(len(token_ids) * 0.9))
        return split_idx, text, ""

    # Map character offset -> token index by re-decoding prefixes (robust, if slow).
    prefix = text[:split_char]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    split_idx = min(max(1, len(prefix_ids)), len(token_ids) - 1)
    return split_idx, text[:split_char], text[split_char:]


# ----------------------------------------------------------------------------
# Answer extraction + checking.
# NOTE: this is a pragmatic checker. For final results use `math_verify` or
# sympy-based equivalence -- string matching will under-count correct answers
# written in a different but equivalent form.
# ----------------------------------------------------------------------------
def extract_answer(text):
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed:
        return boxed[-1].strip()
    m = re.search(r"final answer is:?\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".").strip("$ ")
    return ""


def normalise(a):
    if a is None:
        return ""
    a = str(a).strip().strip("$").replace(" ", "").replace("\\!", "")
    a = a.replace("\\left", "").replace("\\right", "")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = a.replace(",", "")           # 1,000 -> 1000
    a = re.sub(r"\.0+$", "", a)      # 5.0 -> 5
    return a.lower()


def is_correct(pred, gold):
    p, g = normalise(pred), normalise(gold)
    if not p:
        return False
    if p == g:
        return True
    try:                              # numeric fallback
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return False


# ----------------------------------------------------------------------------
# Generation + log-probability extraction
# ----------------------------------------------------------------------------
@torch.no_grad()
def generate_one(model, tokenizer, question, args):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]

    # transformers >=5 returns a BatchEncoding dict here (return_dict defaults
    # to True); transformers 4.x returned a bare tensor. Handle both.
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    if not hasattr(enc, "keys"):
        enc = {"input_ids": enc}
    enc = {k: v.to(model.device) for k, v in enc.items() if hasattr(v, "to")}
    prompt_len = enc["input_ids"].shape[-1]

    out = model.generate(
        **enc,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        return_dict_in_generate=True,
        output_logits=True,   # RAW model logits.
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    gen_ids = out.sequences[0][prompt_len:]

    # IMPORTANT: use out.logits, not out.scores. `scores` are post-processed by
    # temperature and top-p (most entries become -inf), which would distort every
    # confidence number. `logits` are the model's unmodified distribution.
    steps = out.logits if getattr(out, "logits", None) is not None else out.scores
    logprobs = []
    for step, tensor in enumerate(steps):
        if step >= len(gen_ids):
            break
        lp = torch.log_softmax(tensor[0].float(), dim=-1)
        logprobs.append(lp[gen_ids[step]].item())

    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    return gen_ids.tolist(), logprobs, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="math500",
                    choices=["math500", "gsm8k", "aime24"])
    ap.add_argument("--n", type=int, default=100, help="number of problems")
    ap.add_argument("--k", type=int, default=4, help="samples per problem")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--load-4bit", action="store_true",
                    help="NF4 quantisation; needed to fit a 7B model in 8GB VRAM. "
                         "NOTE: alters log-probabilities, so results are directional "
                         "only -- regenerate at full precision for final numbers.")
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--sample", default="random",
                    choices=["random", "stratified", "stride"],
                    help="how to pick the subset when --n < dataset size")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed; keep fixed so the subset is reproducible")
    args = ap.parse_args()

    tag = args.model.split("/")[-1].replace(".", "-")
    out_path = args.out or f"traces_{tag}_{args.dataset}.jsonl"

    print(f"[setup] loading {args.model}")
    if not torch.cuda.is_available():
        print("[warn] no CUDA device found. Generation on CPU will be very slow; "
              "consider --n 10 --k 2 --max-new-tokens 512 for a smoke test only.")

    quant_cfg, quant_tag = None, "none"
    if args.load_4bit or args.load_8bit:
        from transformers import BitsAndBytesConfig
        if args.load_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
            quant_tag = "nf4"
        else:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
            quant_tag = "int8"
        print(f"[setup] quantisation: {quant_tag} -- log-probabilities are affected; "
              f"treat results as directional")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto" if args.dtype == "auto" else getattr(torch, args.dtype),
        device_map="auto",
        quantization_config=quant_cfg,
    )
    model.eval()

    problems = load_problems(args.dataset, args.n, args.sample, args.seed)
    print(f"[setup] {len(problems)} problems x {args.k} samples -> {out_path}")

    t0 = time.time()
    with open(out_path, "w") as f:
        for pi, prob in enumerate(problems):
            for si in range(args.k):
                ids, logprobs, text = generate_one(model, tokenizer,
                                                   prob["question"], args)
                split_idx, r_text, y_text = split_reasoning_answer(
                    text, ids, tokenizer, args.model)

                r_lp = logprobs[:split_idx]
                y_lp = logprobs[split_idx:]
                pred = extract_answer(text)

                rec = {
                    **{k: prob[k] for k in
                       ("id", "dataset", "level", "subject", "answer")},
                    "sample_idx": si,
                    "model": args.model,
                    "quantisation": quant_tag,
                    "sample_mode": args.sample,
                    "sample_seed": args.seed,
                    "pred": pred,
                    "correct": is_correct(pred, prob["answer"]),
                    # --- PiCSAR split confidence -------------------------------
                    # Raw sums are length-dependent: a model emitting 3000 tokens
                    # scores far lower than one emitting 500 purely from length.
                    # Use the *_mean fields for any cross-model comparison.
                    "logp_r_sum": sum(r_lp),
                    "logp_r_mean": (sum(r_lp) / len(r_lp)) if r_lp else None,
                    "logp_y_sum": sum(y_lp),
                    "logp_y_mean": (sum(y_lp) / len(y_lp)) if y_lp else None,
                    "picsar_sum": sum(r_lp) + sum(y_lp),
                    # --- lengths (feeds the Qwen-vs-gpt-oss token question) ----
                    "n_tokens_total": len(logprobs),
                    "n_tokens_reasoning": len(r_lp),
                    "n_tokens_answer": len(y_lp),
                    "reasoning_text": r_text[:4000],
                    "answer_text": y_text[:1000],
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()

            if (pi + 1) % 5 == 0:
                el = time.time() - t0
                print(f"[gen] {pi+1}/{len(problems)} problems  "
                      f"({el:.0f}s elapsed, {el/(pi+1):.1f}s/problem)")

    print(f"[done] wrote {out_path} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
